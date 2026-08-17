# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Custom httpx.AsyncClient for LLM calls.

Always builds a controlled client (not just for ``use_proxy: true``) so we own:

* **Transport-level retries** (``retries=1``) — recover from a single failed
  ``connect`` / ``recv`` (typical when an upstream gateway closed an idle
  keep-alive connection between teleport node transitions).
* **No keep-alive** (``max_keepalive_connections=0``) — LLM traffic is < 1 QPS;
  reusing idle connections across multi-second / multi-minute gaps repeatedly
  hits ``APIConnectionError`` cascades when gateways like ``az.gptplus5.com``
  silently drop idle TCP. New TCP + TLS per request is ~50 ms, negligible
  compared to multi-second streaming responses.
* **Explicit per-phase timeouts** — short ``connect`` so DNS / SYN black-holes
  fast-fail and ``PooledLLM`` can rotate keys promptly; ``read`` / ``pool``
  retain ``stage.http_timeout`` for long streaming bodies.

Optional corporate proxy is wired through the same client when
``stage.use_proxy: true`` (reads ``http_proxy`` / ``HTTP_PROXY`` and
``USER_NAME`` / ``USER_PWD`` from the environment).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from scienceflow.config.settings import StageConfig

logger = logging.getLogger("scienceflow")

_EMPTY_SENTINEL = "EMPTY"

# httpx transport retries=1: one transparent retry on connect / read failures
# at the socket layer. Targets the keep-alive death-connection case where the
# first reuse fails but a fresh connection works. Higher values increase tail
# latency on real outages.
_HTTPX_TRANSPORT_RETRIES = 1

# Disable keep-alive (max_keepalive_connections=0) and cap concurrent
# connections per stage. ``max_connections`` is a safety cap; PooledLLM
# typically issues 1 in-flight per stage.
_HTTPX_MAX_CONNECTIONS = 20

# Short connect / write so socket-level black holes surface quickly and
# PooledLLM can rotate keys; read/pool retain stage.http_timeout for streams.
_HTTPX_CONNECT_TIMEOUT_SEC = 10.0
_HTTPX_WRITE_TIMEOUT_SEC = 30.0


def is_empty_api_key_sentinel(key: str | None) -> bool:
    """True when *key* is the literal placeholder ``EMPTY`` (case-insensitive)."""
    if not key:
        return False
    return key.strip().upper() == _EMPTY_SENTINEL


def proxy_basic_auth_header(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    return f"Basic {token}"


def _resolve_proxy_settings(stage: StageConfig) -> tuple[str | None, dict[str, str]]:
    """Return ``(proxy_url, proxy_headers)`` from env when ``stage.use_proxy``.

    Returns ``(None, {})`` when not using a proxy. When ``use_proxy=true`` but
    no proxy URL is set in the environment, logs a warning and returns
    ``(None, {})`` (the client is still built, just without proxy).
    """
    if not stage.use_proxy:
        return None, {}

    proxy_url = (os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY") or "").strip()
    if not proxy_url:
        logger.warning(
            "[LLM] use_proxy=true but http_proxy/HTTP_PROXY is unset; "
            "building AsyncClient without proxy",
        )
        return None, {}

    user = (os.environ.get("USER_NAME") or "").strip()
    pwd = (os.environ.get("USER_PWD") or "").strip()
    headers: dict[str, str] = {}
    if user and pwd:
        headers["Proxy-Authorization"] = proxy_basic_auth_header(user, pwd)
    return proxy_url, headers


def build_llm_async_client(stage: StageConfig, *, pooled: bool) -> httpx.AsyncClient | None:
    """Build the httpx.AsyncClient passed to ``AsyncOpenAI(http_client=...)``.

    Returns ``None`` only when the effective API key is the ``EMPTY`` sentinel
    (caller wants no client at all — e.g. local mock setups). Otherwise returns
    a fully configured client even when ``use_proxy=False`` so we own transport
    retries, no-keep-alive, and explicit timeouts.
    """
    if pooled:
        if any(is_empty_api_key_sentinel(k) for k in stage.api_keys):
            return None
    elif is_empty_api_key_sentinel(stage.api_key):
        return None

    proxy_url, proxy_headers = _resolve_proxy_settings(stage)

    timeout = httpx.Timeout(
        connect=_HTTPX_CONNECT_TIMEOUT_SEC,
        read=float(stage.http_timeout),
        write=_HTTPX_WRITE_TIMEOUT_SEC,
        pool=float(stage.http_timeout),
    )

    limits = httpx.Limits(
        max_keepalive_connections=0,
        max_connections=_HTTPX_MAX_CONNECTIONS,
    )

    try:
        transport = httpx.AsyncHTTPTransport(
            retries=_HTTPX_TRANSPORT_RETRIES,
            verify=not stage.use_proxy,
            limits=limits,
            proxy=proxy_url,
        )
        client_kwargs: dict[str, Any] = {
            "transport": transport,
            "timeout": timeout,
        }
        if proxy_headers:
            client_kwargs["headers"] = proxy_headers
        return httpx.AsyncClient(**client_kwargs)
    except Exception as e:
        logger.warning("[LLM] failed to build AsyncClient: %s", e)
        return None


def build_optional_proxy_async_client(
    stage: StageConfig, *, pooled: bool
) -> httpx.AsyncClient | None:
    """Backwards-compatible alias for :func:`build_llm_async_client`.

    Earlier the function only returned a client under ``use_proxy=true``; the
    new implementation returns a controlled client unconditionally so we can
    enforce no-keep-alive + transport retries on every LLM call.
    """
    return build_llm_async_client(stage, pooled=pooled)


def llm_extra_client_kwargs(stage: StageConfig) -> dict[str, object]:
    """``headers`` and ``http_asyncclient`` for OpenAI-compatible SDK.

    Always injects ``http_asyncclient`` (unless API key is the EMPTY sentinel)
    so the SDK never falls back to its default httpx client — that default
    keeps connections alive across calls and is exactly what causes the
    ``APIConnectionError`` cascade on teleport-reuse first calls.
    """
    extra: dict[str, object] = {}
    if stage.headers:
        extra["headers"] = dict(stage.headers)
    client = build_llm_async_client(stage, pooled=bool(stage.api_keys))
    if client is not None:
        extra["http_asyncclient"] = client
    return extra


async def aclose_llm_clients(llm: object) -> None:
    """Close OpenAI SDK client and shared custom httpx client(s) before loop exit."""
    from scienceflow.core.key_pool import PooledLLM

    async def _close_one(client: object) -> None:
        if client is None:
            return
        is_closed = getattr(client, "is_closed", None)
        try:
            if callable(is_closed) and is_closed():
                return
        except Exception:
            pass
        try:
            await client.close()
        except Exception as e:
            logger.debug("LLM client close: %s", e)

    instances = llm._instances if isinstance(llm, PooledLLM) else [llm]
    seen_http: set[int] = set()
    for inst in instances:
        await _close_one(getattr(inst, "client", None))
        http_c = getattr(inst, "http_asyncclient", None)
        if http_c is not None and id(http_c) not in seen_http:
            seen_http.add(id(http_c))
            await _close_one(http_c)
