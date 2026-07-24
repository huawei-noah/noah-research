"""Runtime helpers for standalone worker processes."""

from __future__ import annotations

from ..config import get_config, init_config_from_env
from ..infra import shutdown_tracer_provider
from ..infra.db import close_database_clients, ensure_database_schema
from ..llm import close_llm_clients, init_embed_client, init_llm_client, validate_embedding_dimension
from ..logging import configure_logging, configure_tracing, get_logger

logger = get_logger(__name__)


async def start_worker_runtime() -> None:
    """Initialize shared process resources for standalone workers."""

    config_source = init_config_from_env()
    cfg = get_config()
    configure_logging(level=cfg.telemetry.log_level)
    configure_tracing(cfg.telemetry)
    await ensure_database_schema()
    init_llm_client()
    init_embed_client()
    await validate_embedding_dimension()
    logger.info("mindmemos worker runtime started", config_source=config_source)


async def stop_worker_runtime() -> None:
    """Close shared process resources for standalone workers."""

    await close_llm_clients()
    await close_database_clients()
    logger.info("mindmemos worker runtime stopped")
    shutdown_tracer_provider()
