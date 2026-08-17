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

"""API Key pool with round-robin rotation and rate-limit failover.

Provides ``PooledLLM`` — a drop-in replacement for ``OnlineLLM`` that
distributes requests across multiple (base_url, api_key) pairs.

When every key returns a rate limit in one pass, the pool waits until the
earliest cooldown expires (bounded by ``SCIENCEFLOW_POOL_RL_MAX_WAIT_SEC``,
default 300s) instead of failing immediately — useful when several keys share
one upstream RPM limit.

Typical usage::

    pool = PooledLLM.from_endpoints(
        endpoints=[("https://api.openai.com/v1", "sk-AAA"),
                   ("https://api.deepseek.com/v1", "sk-BBB")],
        model="gpt-4o", max_tokens=32768, stream=True, tracker=True,
    )
    response = await pool.ask(messages)
"""

from __future__ import annotations

import asyncio
import contextvars
import email.utils
import hashlib
import logging
import os
import time
from typing import Any

import httpx
from openai import APIError, RateLimitError

from deepcraft_core import OnlineLLM

from scienceflow.core.llm_reasoning_compat import is_missing_reasoning_replay_error

logger = logging.getLogger("scienceflow")

COOLDOWN_RATE_LIMIT = 60.0
COOLDOWN_CONNECTION = 15.0
# After every key hits rate limit in one pass, wait until the earliest slot unlocks and retry
# (bounded by this wall-clock budget per _call_with_failover invocation).
_DEFAULT_POOL_RL_MAX_WAIT_SEC = 300.0
# Same-key soft-retry delay (seconds). After a transient connection-class error
# we wait briefly and retry the same key once before cooling it down. Targets
# the keep-alive death case where the first attempt reuses a stale socket but
# a second attempt opens a fresh connection. Combined with httpx
# ``transport=AsyncHTTPTransport(retries=1)`` and ``max_keepalive_connections=0``
# in :mod:`scienceflow.core.llm_http`, this is a defensive third layer.
_SOFT_RETRY_DELAY_SEC = 0.5

_pooled_last_idx: contextvars.ContextVar[int] = contextvars.ContextVar("_pooled_last_idx")


class KeyPool:
    """Round-robin selector for (base_url, api_key) pairs with cooldown."""

    def __init__(self, endpoints: list[tuple[str, str]]):
        if not endpoints:
            raise ValueError("KeyPool requires at least one endpoint")
        self._endpoints = list(endpoints)
        self._size = len(self._endpoints)
        self._index = 0
        self._cooldowns: dict[int, float] = {}

    @property
    def size(self) -> int:
        return self._size

    def is_available(self, idx: int) -> bool:
        """True when *idx* is not cooling down."""
        return time.monotonic() >= self._cooldowns.get(idx, 0)

    def pick(self, *, exclude: set[int] | None = None) -> tuple[int, str, str]:
        """Return ``(index, base_url, api_key)`` for the next usable slot."""
        exclude = exclude or set()
        if len(exclude) >= self._size:
            raise RuntimeError("KeyPool: all slots excluded")
        now = time.monotonic()
        for _ in range(self._size):
            idx = self._index % self._size
            self._index += 1
            if idx in exclude:
                continue
            if now >= self._cooldowns.get(idx, 0):
                return idx, self._endpoints[idx][0], self._endpoints[idx][1]
        candidates = [idx for idx in range(self._size) if idx not in exclude]
        idx = min(candidates, key=lambda i: self._cooldowns.get(i, 0))
        return idx, self._endpoints[idx][0], self._endpoints[idx][1]

    def cooldown(self, idx: int, seconds: float) -> None:
        self._cooldowns[idx] = time.monotonic() + seconds
        url_preview = self._endpoints[idx][0][:50]
        logger.warning(f"[KeyPool] Slot #{idx} ({url_preview}) cooled down for {seconds:.0f}s")

    def reset_cooldown(self, idx: int) -> None:
        self._cooldowns.pop(idx, None)

    def seconds_until_earliest_unlock(self) -> float:
        """Seconds until some slot leaves cooldown (0 if none or already usable)."""
        if not self._cooldowns:
            return 0.0
        now = time.monotonic()
        return max(0.0, min(self._cooldowns.values()) - now)


def _stable_index(seed: str, size: int) -> int:
    if size <= 0:
        return 0
    if not seed:
        return 0
    digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:16], 16) % size


def _retry_after_seconds(exc: BaseException, default: float) -> float:
    """Return Retry-After seconds when present; otherwise *default*."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        headers = getattr(exc, "headers", None)
    value = None
    if headers is not None:
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            value = None
    if value is None:
        return default
    raw = str(value).strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(raw)
            return max(0.0, dt.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return default


class PooledLLM:
    """Drop-in replacement for ``OnlineLLM`` that rotates across multiple keys.

    * **Round-robin**: each ``ask()`` call picks the next key in sequence,
      distributing load evenly and reducing per-key rate-limit pressure.
    * **Failover**: if one key exhausts its inner retries (10 attempts with
      exponential backoff), the pool cools it down and tries the next key.
    * **Transparent proxy**: attribute access (``_last_call_input_tokens``,
      ``chunk_queue``, etc.) is forwarded to the last-used ``OnlineLLM``.
    """

    def __init__(
        self,
        instances: list[OnlineLLM],
        *,
        routing_mode: str = "round_robin",
        sticky_id: str = "",
        sticky_primary_index: int | None = None,
        rate_limit_cooldown_sec: float = COOLDOWN_RATE_LIMIT,
        connection_cooldown_sec: float = COOLDOWN_CONNECTION,
    ):
        if not instances:
            raise ValueError("PooledLLM requires at least one OnlineLLM instance")
        self._instances = instances
        self._pool = KeyPool(
            [(str(inst.base_url or ""), str(inst.api_key or "")) for inst in instances]
        )
        self._last_idx = 0
        self._routing_mode = str(routing_mode or "round_robin").strip().lower()
        self._sticky_id = str(sticky_id or "")
        self._sticky_enabled = self._routing_mode in {
            "sticky",
            "task_sticky",
            "sticky_failover",
            "task_sticky_failover",
        }
        if sticky_primary_index is None:
            self._primary_idx = _stable_index(self._sticky_id, self._pool.size)
        else:
            self._primary_idx = max(0, min(int(sticky_primary_index), self._pool.size - 1))
        self._rate_limit_cooldown_sec = max(0.0, float(rate_limit_cooldown_sec))
        self._connection_cooldown_sec = max(0.0, float(connection_cooldown_sec))
        self._last_call_pool_index: int | None = None
        self._last_call_routing_mode = self._routing_mode
        self._last_call_failover_count = 0

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_endpoints(
        cls,
        endpoints: list[tuple[str, str]],
        *,
        endpoint_models: list[str] | None = None,
        **llm_kwargs: Any,
    ) -> PooledLLM:
        """Create a pool from endpoints plus shared or per-endpoint LLM params.

        ``endpoint_models`` keeps model/key/url triples aligned for heterogeneous
        providers while preserving key-pool failover. When omitted, all endpoints
        use the shared ``model`` from ``llm_kwargs``.
        """
        pool_kwargs = {
            name: llm_kwargs.pop(name)
            for name in (
                "routing_mode",
                "sticky_id",
                "sticky_primary_index",
                "rate_limit_cooldown_sec",
                "connection_cooldown_sec",
            )
            if name in llm_kwargs
        }
        models = [str(x).strip() for x in (endpoint_models or []) if str(x).strip()]
        if models and len(models) != len(endpoints):
            raise ValueError(
                f"endpoint_models length {len(models)} must match endpoints length {len(endpoints)}"
            )
        instances: list[OnlineLLM] = []
        for idx, (url, key) in enumerate(endpoints):
            endpoint_kwargs = dict(llm_kwargs)
            if models:
                endpoint_kwargs["model"] = models[idx]
            instances.append(OnlineLLM(base_url=url, api_key=key, **endpoint_kwargs))
        return cls(instances, **pool_kwargs)

    @classmethod
    def from_single(cls, llm: OnlineLLM) -> PooledLLM:
        """Wrap a single ``OnlineLLM`` for uniform interface (no rotation)."""
        return cls([llm])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next(self, *, tried: set[int] | None = None) -> tuple[int, OnlineLLM]:
        tried = tried or set()
        if (
            self._sticky_enabled
            and self._primary_idx not in tried
            and self._pool.is_available(self._primary_idx)
        ):
            idx = self._primary_idx
        else:
            idx, _, _ = self._pool.pick(exclude=tried)
        self._last_idx = idx
        return idx, self._instances[idx]

    async def _call_with_failover(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Try each key in turn; on rate-limit / connection failure, rotate.

        If *all* keys return rate-limit in one sweep, wait until the earliest cooldown
        expires (up to ``SCIENCEFLOW_POOL_RL_MAX_WAIT_SEC``, default 300) and retry, so a
        short RPM burst does not fail the whole agent step when keys share one upstream.
        """
        max_wait = float(os.getenv("SCIENCEFLOW_POOL_RL_MAX_WAIT_SEC", str(_DEFAULT_POOL_RL_MAX_WAIT_SEC)))
        deadline = time.monotonic() + max(0.0, max_wait)
        last_err: Exception | None = None
        failover_count = 0

        while time.monotonic() < deadline:
            last_err = None
            tried: set[int] = set()

            for _ in range(self._pool.size):
                idx, llm = self._next(tried=tried)
                if idx in tried:
                    continue
                tried.add(idx)
                is_failover = self._sticky_enabled and idx != self._primary_idx
                if is_failover:
                    failover_count += 1

                try:
                    result = await getattr(llm, method)(*args, **kwargs)
                    self._pool.reset_cooldown(idx)
                    _pooled_last_idx.set(idx)
                    self._last_call_pool_index = idx
                    self._last_call_routing_mode = self._routing_mode
                    self._last_call_failover_count = failover_count
                    return result
                except RateLimitError as e:
                    cooldown = _retry_after_seconds(e, self._rate_limit_cooldown_sec)
                    self._pool.cooldown(idx, cooldown)
                    last_err = e
                    if "function call turn" in str(e).lower():
                        logger.warning(
                            "[PooledLLM] RateLimitError mentions chat-turn wording — "
                            "treating as rate limit (upstream may wrap quota errors)."
                        )
                    logger.warning(f"[PooledLLM] Key #{idx} rate-limited, trying next...")
                except (
                    APIError,
                    ConnectionError,
                    OSError,
                    TimeoutError,
                    httpx.ReadTimeout,
                ) as e:
                    if is_missing_reasoning_replay_error(e):
                        last_err = e
                        logger.warning(
                            "[PooledLLM] Key #%d returned deterministic "
                            "reasoning_content replay error; deferring to caller "
                            "for request adaptation",
                            idx,
                        )
                        raise
                    # Same-key soft retry once before cooldown. Targets keep-alive
                    # death: the first attempt may reuse a stale connection, but a
                    # second attempt (after the brief pause) opens a fresh one.
                    # Without this, a single dead idle connection per key cools
                    # all 6 keys within seconds (observed on az.gptplus5.com after
                    # teleport node transitions).
                    logger.warning(
                        "[PooledLLM] Key #%d transient error (%s); "
                        "same-key soft-retry in %.1fs",
                        idx,
                        type(e).__name__,
                        _SOFT_RETRY_DELAY_SEC,
                    )
                    try:
                        await asyncio.sleep(_SOFT_RETRY_DELAY_SEC)
                        result = await getattr(llm, method)(*args, **kwargs)
                    except RateLimitError as e_rl:
                        cooldown = _retry_after_seconds(e_rl, self._rate_limit_cooldown_sec)
                        self._pool.cooldown(idx, cooldown)
                        last_err = e_rl
                        logger.warning(
                            "[PooledLLM] Key #%d soft-retry hit rate limit, trying next...",
                            idx,
                        )
                    except (
                        APIError,
                        ConnectionError,
                        OSError,
                        TimeoutError,
                        httpx.ReadTimeout,
                    ) as e2:
                        self._pool.cooldown(idx, self._connection_cooldown_sec)
                        last_err = e2
                        logger.warning(
                            "[PooledLLM] Key #%d error after soft-retry (%s), trying next...",
                            idx,
                            type(e2).__name__,
                        )
                    else:
                        self._pool.reset_cooldown(idx)
                        _pooled_last_idx.set(idx)
                        self._last_call_pool_index = idx
                        self._last_call_routing_mode = self._routing_mode
                        self._last_call_failover_count = failover_count
                        logger.info(
                            "[PooledLLM] Key #%d soft-retry succeeded",
                            idx,
                        )
                        return result

            if last_err is None:
                raise RuntimeError("PooledLLM: no keys available")
            if isinstance(last_err, RateLimitError):
                wait = self._pool.seconds_until_earliest_unlock()
                now = time.monotonic()
                slip = min(wait + 0.25, max(0.0, deadline - now))
                if slip > 0.05 and now + slip <= deadline:
                    logger.info(
                        "[PooledLLM] All keys rate-limited this pass; waiting %.1fs "
                        "for cooldown (budget %.0fs)",
                        slip,
                        max_wait,
                    )
                    await asyncio.sleep(slip)
                    continue
            raise last_err

        if last_err is not None:
            raise last_err
        raise RuntimeError("PooledLLM: rate-limit wait budget exhausted")

    # ------------------------------------------------------------------
    # Public API — mirrors OnlineLLM
    # ------------------------------------------------------------------

    async def ask(self, *args: Any, **kwargs: Any) -> str:
        return await self._call_with_failover("ask", *args, **kwargs)

    async def ask_tool(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_with_failover("ask_tool", *args, **kwargs)

    async def ask_tool_stream(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_with_failover("ask_tool_stream", *args, **kwargs)

    async def ask_logits(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_with_failover("ask_logits", *args, **kwargs)

    async def ask_vllm_logits(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_with_failover("ask_vllm_logits", *args, **kwargs)

    async def ask_embedding(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_with_failover("ask_embedding", *args, **kwargs)

    async def interrupt(self) -> None:
        for inst in self._instances:
            await inst.interrupt()

    # ------------------------------------------------------------------
    # Attribute proxy
    # ------------------------------------------------------------------

    _STATS_FIELDS = frozenset({
        "_last_call_input_tokens", "_last_call_output_tokens",
        "_last_call_input_cached_tokens",
        "_last_call_ttft", "_last_call_tpot", "_last_finish_reason",
    })

    def __getattr__(self, name: str) -> Any:
        if name in self._STATS_FIELDS:
            try:
                idx = _pooled_last_idx.get()
            except LookupError:
                idx = self._last_idx
            return getattr(self._instances[idx], name)
        return getattr(self._instances[self._last_idx], name)

    def __repr__(self) -> str:
        models = {str(i.model) for i in self._instances}
        return f"PooledLLM(keys={self._pool.size}, models={models})"


# ------------------------------------------------------------------
# Helpers for building endpoints from config
# ------------------------------------------------------------------

def parse_key_env(
    api_keys_csv: str | None = None,
    base_urls_csv: str | None = None,
    fallback_key: str = "",
    fallback_url: str = "",
) -> list[tuple[str, str]]:
    """Parse comma/space-separated key/url strings into endpoint pairs.

    Supports these patterns::

        API_KEYS=sk-A,sk-B,sk-C  BASE_URLS=https://a/v1,https://b/v1
        API_KEYS=sk-A,sk-B        BASE_URL=https://api.openai.com/v1   (broadcast)
        API_KEY=sk-A               BASE_URL=https://api.openai.com/v1   (single)

    Returns list of ``(base_url, api_key)`` tuples.
    """
    import re

    def _split(s: str) -> list[str]:
        return [x.strip() for x in re.split(r"[,\s]+", s.strip()) if x.strip()]

    keys = _split(api_keys_csv) if api_keys_csv else ([fallback_key] if fallback_key else [])
    urls = _split(base_urls_csv) if base_urls_csv else ([fallback_url] if fallback_url else [])

    if not keys:
        raise ValueError("No API keys provided")

    if len(urls) == 0:
        urls = [""] * len(keys)
    elif len(urls) == 1 and len(keys) > 1:
        urls = urls * len(keys)
    elif len(urls) != len(keys):
        raise ValueError(
            f"Mismatch: {len(keys)} API keys but {len(urls)} base URLs. "
            "Provide 1 URL (broadcast) or the same count as keys."
        )

    return list(zip(urls, keys))
