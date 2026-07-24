"""Helpers for building litellm routers from MindMemOS model config."""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm
from litellm import Router
from omegaconf import DictConfig, ListConfig, OmegaConf

from ..config import ModelEndpointConfig, ModelRouterConfig
from ..typing import Usage

litellm.drop_params = True  # Let litellm drop optional params unsupported by a provider.
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.turn_off_message_logging = True
for _logger_name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

# Fields aligned with litellm ``litellm_params``. None values are omitted so
# litellm/provider defaults can take effect. ``extra_body`` is handled separately
# because OmegaConf wrappers need to be converted back to plain containers.
_LITELLM_PARAM_FIELDS: tuple[str, ...] = (
    "model",
    "api_key",
    "api_base",
    "rpm",
    "tpm",
    "timeout",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "encoding_format",
    "dimensions",
    "num_retries",
)


def _model_supports_dimensions(model: str, whitelist: list[str]) -> bool:
    """Whether ``model`` is whitelisted to pass ``dimensions`` to the provider.

    Matches on the model name with the provider prefix (e.g. ``openai/``)
    stripped, so an entry like ``doubao-embedding`` covers all suffixed variants
    (``doubao-embedding-large``, ``...-240`` ...).
    """

    stripped = model.split("/", 1)[1] if "/" in model else model
    return any(stripped.startswith(prefix) for prefix in whitelist)


def build_litellm_params(
    ep: ModelEndpointConfig,
    *,
    dimensions_supported_models: list[str] | None = None,
) -> dict[str, Any]:
    """Flatten one endpoint config into litellm Router params."""
    params: dict[str, Any] = {}
    for key in _LITELLM_PARAM_FIELDS:
        value = getattr(ep, key, None)
        if value is not None:
            params[key] = value

    extra_body = getattr(ep, "extra_body", None)
    if extra_body:
        if isinstance(extra_body, (DictConfig, ListConfig)):
            extra_body = OmegaConf.to_container(extra_body, resolve=True)
        if extra_body:
            params["extra_body"] = extra_body

    # litellm drops `dimensions` for any openai-compatible model whose name lacks
    # the literal "text-embedding-3" (litellm/utils.py). Whitelisted models get
    # `allowed_openai_params` so litellm keeps `dimensions` as a top-level param.
    if ep.dimensions is not None and _model_supports_dimensions(ep.model, dimensions_supported_models or []):
        params["allowed_openai_params"] = ["dimensions"]

    return params


def build_router(router_cfg: ModelRouterConfig, alias: str, *, num_retries: int | None = None) -> tuple[Router, int]:
    """Build a litellm Router whose endpoints share one public alias."""
    model_list = []
    deployment_counts: dict[str, int] = {}
    for ep in router_cfg.endpoints:
        deployment_key = f"{ep.model}@{ep.api_base}"
        deployment_counts[deployment_key] = deployment_counts.get(deployment_key, 0) + 1
        model_list.append(
            {
                "model_name": alias,
                "litellm_params": build_litellm_params(
                    ep, dimensions_supported_models=router_cfg.dimensions_supported_models
                ),
                "model_info": {"id": f"{deployment_key}#{deployment_counts[deployment_key]}"},
            }
        )

    max_retries = (
        num_retries if num_retries is not None else max((ep.num_retries for ep in router_cfg.endpoints), default=0)
    )
    router = Router(
        model_list=model_list,
        routing_strategy=router_cfg.routing_strategy,
        num_retries=max_retries,
        allowed_fails=router_cfg.allowed_fails,
        cooldown_time=router_cfg.cool_down,
    )
    return router, max_retries


# Routers are expensive to rebuild and carry stateful routing data (endpoint
# cooldowns, load counters) plus the deployment->httpx client binding, so they
# are cached per resolved endpoint config and shared across requests. Tenant or
# project overrides that change any endpoint param produce a different key and a
# separate Router; identical configs (even across tenants) share one Router so
# their cooldown/load state accumulates correctly. The thin LLM/Embed/Rerank
# client wrappers stay per-request since they hold no cross-request state.
_ROUTER_CACHE: dict[str, tuple[Router, int]] = {}


def _router_cache_key(router_cfg: ModelRouterConfig, alias: str, num_retries: int | None) -> str:
    """Build a stable cache key from everything that affects Router identity."""
    payload = {
        "alias": alias,
        "routing_strategy": router_cfg.routing_strategy,
        "allowed_fails": router_cfg.allowed_fails,
        "cool_down": router_cfg.cool_down,
        "num_retries": num_retries,
        "dimensions_supported_models": list(router_cfg.dimensions_supported_models),
        "endpoints": [
            build_litellm_params(ep, dimensions_supported_models=router_cfg.dimensions_supported_models)
            for ep in router_cfg.endpoints
        ],
    }
    return json.dumps(payload, sort_keys=True, default=str)


def get_router(router_cfg: ModelRouterConfig, alias: str, *, num_retries: int | None = None) -> tuple[Router, int]:
    """Return a cached Router for this resolved endpoint config, building it once.

    The Router is the expensive, stateful piece (cooldowns, load balancing, httpx
    client binding) and is reused across requests with the same effective config;
    request-time config overrides that change endpoints yield a separate Router.
    """
    key = _router_cache_key(router_cfg, alias, num_retries)
    cached = _ROUTER_CACHE.get(key)
    if cached is not None:
        return cached
    result = build_router(router_cfg, alias, num_retries=num_retries)
    _ROUTER_CACHE[key] = result
    return result


def clear_router_cache() -> None:
    """Drop all cached Routers so the next ``get_router`` rebuilds from current config."""
    _ROUTER_CACHE.clear()


def dump_response(obj: Any) -> dict:
    """Best-effort conversion of provider response objects to plain dicts."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            return {}
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except Exception:
        return {}


def get_response_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read one field from dict-like or attribute-style provider objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def litellm_response_headers(obj: Any) -> dict[str, Any]:
    """Read LiteLLM synthetic response headers such as retry counters."""
    hidden = getattr(obj, "_hidden_params", {}) or {}
    if hasattr(hidden, "model_dump"):
        hidden = hidden.model_dump()
    if not isinstance(hidden, dict):
        return {}
    headers = hidden.get("additional_headers", {}) or {}
    return headers if isinstance(headers, dict) else {}


def usage_tokens(usage: Any) -> Usage:
    """Extract token counters exposed as stable response fields."""
    return Usage(
        completion_tokens=get_response_value(usage, "completion_tokens"),
        prompt_tokens=get_response_value(usage, "prompt_tokens"),
        total_tokens=get_response_value(usage, "total_tokens"),
    )
