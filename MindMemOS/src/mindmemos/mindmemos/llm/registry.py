"""LLM client factory helpers."""

from __future__ import annotations

import litellm

from ..config import get_config
from ..errors import InvalidConfigError
from .chat import LLMClient
from .embedding import EmbedClient
from .rerank import RerankClient
from .router import clear_router_cache, get_router


def init_llm_client() -> LLMClient:
    """Build a chat client once so router config errors fail fast at startup.

    The returned client is not cached; request-time calls use the currently
    bound config context.
    """
    return get_llm_client()


def init_embed_client() -> EmbedClient:
    """Build an embedding client once so router config errors fail fast at startup.

    The returned client is not cached; request-time calls use the currently
    bound config context.
    """
    return get_embed_client()


def init_rerank_client() -> RerankClient:
    """Build a rerank client once so router config errors fail fast at startup."""
    return get_rerank_client()


def get_llm_client() -> LLMClient:
    """Create a chat client from the currently bound config context.

    The thin client wrapper is per-request, but its underlying Router is cached
    by resolved endpoint config so cooldown/load-balancing state is preserved.
    """
    cfg = get_config()
    router_cfg = cfg.chat_model_router
    router, _ = get_router(router_cfg, LLMClient.ALIAS)
    default_model = LLMClient.ALIAS if router_cfg.endpoints else None
    return LLMClient(router, default_model=default_model, max_attempts=router_cfg.format_parser_max_attempts)


def get_embed_client() -> EmbedClient:
    """Create an embedding client from the currently bound config context."""
    cfg = get_config()
    router_cfg = cfg.embed_model_router
    router, _ = get_router(router_cfg, EmbedClient.ALIAS)
    default_model = EmbedClient.ALIAS if router_cfg.endpoints else None
    return EmbedClient(router, default_model=default_model)


PROBE_TEXT = "ping"


async def validate_embedding_dimension() -> None:
    """Probe the embedding endpoint once and fail fast on dimension mismatch.

    Skips silently when no embedding endpoint is configured. Raises
    ``InvalidConfigError`` if any endpoint declares ``dimensions`` that differs
    from ``database.qdrant.vector_size``, and ``EmbeddingDimensionError`` (via
    ``EmbedClient.embed``) if the provider actually returns the wrong dimension
    — the symptom of ``dimensions`` being silently dropped by litellm
    ``drop_params=True`` or of switching to a model with a different native
    dimension. Intended to run at startup, right after ``init_embed_client``.
    """

    cfg = get_config()
    if not cfg.embed_model_router.endpoints:
        return

    vector_size = cfg.database.qdrant.vector_size
    for ep in cfg.embed_model_router.endpoints:
        if ep.dimensions is not None and ep.dimensions != vector_size:
            raise InvalidConfigError(
                field=f"embed_model_router.endpoints[model={ep.model}].dimensions",
                support=f"equal to database.qdrant.vector_size={vector_size}",
            )

    # embed() resolves expected_dim from config (vector_size) and raises
    # EmbeddingDimensionError on mismatch, so the probe needs no extra checks.
    # await get_embed_client().embed(task="startup.probe", text=PROBE_TEXT)


def get_rerank_client() -> RerankClient:
    """Create a rerank client from the currently bound config context.

    When no rerank endpoint is configured, the client falls back to keyword
    overlap scoring, or identity ordering as last resort.
    """
    cfg = get_config()
    rerank_cfg = cfg.algo_config.search.rerank
    router_cfg = cfg.rerank_model_router
    router = None
    if rerank_cfg.enabled and router_cfg.endpoints:
        router, _ = get_router(router_cfg, RerankClient.ALIAS)
    return RerankClient(
        router,
        max_query_length=rerank_cfg.max_query_length,
        max_doc_length=rerank_cfg.max_doc_length,
        max_batch_size=rerank_cfg.max_batch_size,
        max_concurrent_batches=rerank_cfg.max_concurrent_batches,
        request_timeout=rerank_cfg.request_timeout,
    )


def reset_clients() -> None:
    """Drop cached Routers so the next client picks up refreshed config."""
    clear_router_cache()


async def close_llm_clients() -> None:
    """Close LiteLLM-managed async HTTP clients and drop cached Routers."""
    clear_router_cache()
    await litellm.close_litellm_async_clients()
    litellm.aclient_session = None
