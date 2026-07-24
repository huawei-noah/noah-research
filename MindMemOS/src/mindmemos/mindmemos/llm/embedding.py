"""Embedding client backed by litellm.Router."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from opentelemetry import trace

from ..config import get_config
from ..errors import ConfigNotInitializedError, EmbeddingDimensionError
from ..logging import add_span_event, get_logger, traced, traced_awaitable
from ..typing import EmbeddingResponse
from .router import dump_response, get_response_value, litellm_response_headers, usage_tokens

if TYPE_CHECKING:
    from litellm import Router

logger = get_logger(__name__)


def _resolved_expected_dim() -> int | None:
    """Resolve the collection dimension every embedding must match.

    Returns ``database.qdrant.vector_size`` when config is bound, or ``None``
    when config is uninitialized (e.g. unit tests without a config context),
    in which case dimension validation is skipped.
    """

    try:
        return get_config().database.qdrant.vector_size
    except ConfigNotInitializedError:
        return None


def _input_stats(text: str | list[str]) -> dict[str, int]:
    texts = [text] if isinstance(text, str) else text
    lengths = [len(item) for item in texts]
    return {
        "llm.embed.input.count": len(texts),
        "llm.embed.input.chars": sum(lengths),
        "llm.embed.input.max_chars": max(lengths, default=0),
    }


def _router_attrs(router: Router, target: str) -> dict[str, Any]:
    deployments = [item for item in (getattr(router, "model_list", None) or []) if item.get("model_name") == target]
    params = [item.get("litellm_params", {}) for item in deployments]

    def values(key: str) -> list[Any]:
        return [value for value in (item.get(key) for item in params) if value is not None]

    attrs: dict[str, Any] = {
        "llm.model.alias": target,
        "llm.router.strategy": getattr(router, "routing_strategy", "unknown"),
        "llm.router.num_retries": getattr(router, "num_retries", "unknown"),
        "llm.router.allowed_fails": getattr(router, "allowed_fails", "unknown"),
        "llm.router.cooldown_time": getattr(router, "cooldown_time", "unknown"),
        "llm.router.deployment_count": len(deployments),
    }
    for key in ("timeout", "rpm", "tpm", "num_retries"):
        nums = [value for value in values(key) if isinstance(value, int | float)]
        if nums:
            attrs[f"llm.endpoint.{key}.min"] = min(nums)
            attrs[f"llm.endpoint.{key}.max"] = max(nums)
    models = sorted({str(value) for value in values("model")})
    hosts = sorted({urlparse(str(value)).netloc or str(value) for value in values("api_base")})
    if models:
        attrs["llm.endpoint.models"] = ",".join(models[:5])
    if hosts:
        attrs["llm.endpoint.api_hosts"] = ",".join(hosts[:5])
    return attrs


def _set_current_span_attrs(attrs: dict[str, Any]) -> None:
    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return
    for key, value in attrs.items():
        span.set_attribute(key, value)


class EmbedClient:
    """Embedding client that routes requests through litellm.Router."""

    ALIAS = "embedding"

    def __init__(self, router: Router, *, default_model: str | None = ALIAS) -> None:
        """Wrap a pre-built litellm Router for embedding calls.

        Args:
            router: Shared litellm Router; built and cached by the registry layer.
            default_model: Router alias to target, or ``None`` when no endpoint is
                configured (embed then raises a clear error).
        """
        self._router = router
        self._default_model = default_model

    @traced("llm.embed", record_args=False, record_result=False)
    async def embed(
        self,
        task: str,
        text: str | list[str],
        *,
        model: str | None = None,
        expected_dim: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        target = model or self._default_model
        if target is None:
            msg = "No embed model endpoint configured"
            raise RuntimeError(msg)

        attrs = {"llm.task": task, **_input_stats(text), **_router_attrs(self._router, target)}
        add_span_event(
            "llm.embed.input",
            {
                "task": task,
                "model": target,
                "text": text,
                **_input_stats(text),
            },
        )
        _set_current_span_attrs(attrs)
        start = perf_counter()
        try:
            resp = await traced_awaitable(
                "llm.embed.provider",
                self._router.aembedding(model=target, input=text, **kwargs),
                attributes=attrs,
                tracer_name=__name__,
            )
        except Exception as exc:
            logger.info(
                "litellm_call",
                kind="embedding",
                task=task,
                model=target,
                status="error",
                latency_ms=round((perf_counter() - start) * 1000, 2),
                error=str(exc),
            )
            raise
        embeddings: list[list[float]] = []
        for item in getattr(resp, "data", []) or []:
            if isinstance(item, dict):
                embeddings.append(item.get("embedding") or [])
            else:
                embeddings.append(getattr(item, "embedding", []) or [])

        resolved_dim = expected_dim if expected_dim is not None else _resolved_expected_dim()
        if resolved_dim is not None:
            for vec in embeddings:
                if len(vec) != resolved_dim:
                    raise EmbeddingDimensionError(expected=resolved_dim, actual=len(vec), model=target, task=task)

        _set_current_span_attrs({"llm.embed.output.count": len(embeddings)})
        usage = usage_tokens(getattr(resp, "usage", None))
        headers = litellm_response_headers(resp)
        logger.info(
            "litellm_call",
            kind="embedding",
            task=task,
            model=target,
            status="ok",
            latency_ms=round((perf_counter() - start) * 1000, 2),
            litellm_attempted_retries=headers.get("x-litellm-attempted-retries"),
            litellm_max_retries=headers.get("x-litellm-max-retries"),
        )
        return EmbeddingResponse(
            embeddings=embeddings,
            model=get_response_value(resp, "model", target) or target,
            usage=usage,
            raw_response=dump_response(resp),
        )
