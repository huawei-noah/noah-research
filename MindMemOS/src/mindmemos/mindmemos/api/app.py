"""FastAPI application factory and entry point.

``make dev`` launches uvicorn against the module-level ``app`` below. All
process-level initialization (config, logging, tracing, DB schema) happens in
the lifespan handler so importing this module stays cheap and side-effect free.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..config import get_config, init_config_from_env
from ..errors import ApiError
from ..infra import shutdown_tracer_provider
from ..infra.db import close_database_clients, ensure_database_schema
from ..infra.kafka import start_kafka, stop_kafka
from ..llm import close_llm_clients, init_embed_client, init_llm_client, validate_embedding_dimension
from ..logging import configure_logging, configure_tracing, get_logger
from ..workers import register_workers
from .internal_routes import router as internal_router
from .routes import router as memory_router
from .skill_routes import router as skill_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize config, logging, tracing and DB on startup; tear down on exit."""
    config_source = init_config_from_env()
    cfg = get_config()

    configure_logging(level=cfg.telemetry.log_level)
    configure_tracing(cfg.telemetry)
    logger.info("starting mindmemos api", config_source=config_source)

    # Initialize databases (Qdrant + Neo4j): connect and ensure schema exists.
    await ensure_database_schema()
    logger.info("memory databases ready")

    # startup instead of on the first request.
    init_llm_client()
    init_embed_client()
    await validate_embedding_dimension()
    logger.info("llm clients ready")

    # Register all Kafka consumer handlers before starting; start_kafka is a
    # no-op when kafka.enabled is false, so this stays safe without a broker.
    register_workers()
    await start_kafka()

    try:
        yield
    finally:
        await stop_kafka()
        await _close_llm_clients()
        await _close_databases()
        logger.info("mindmemos api stopped")
        shutdown_tracer_provider()


async def _close_llm_clients() -> None:
    """Close LiteLLM-managed async HTTP clients."""
    try:
        await close_llm_clients()
    except Exception:  # noqa: BLE001 - shutdown best-effort
        logger.warning("failed to close llm clients", exc_info=True)


async def _close_databases() -> None:
    """Close Qdrant/Neo4j clients for the current event loop."""
    try:
        await close_database_clients()
    except Exception:  # noqa: BLE001 - shutdown best-effort
        logger.warning("failed to close database clients", exc_info=True)


def create_app() -> FastAPI:
    app = FastAPI(title="MindMemOS API", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)

    @app.get("/healthz", tags=["meta"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(memory_router)
    app.include_router(skill_router)
    app.include_router(internal_router)

    return app


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "invalid_request",
                "message": _format_validation_errors(exc.errors()),
                "data": None,
            },
        )


def _format_validation_errors(errors: list[dict]) -> str:
    return "; ".join(_format_validation_error(error) for error in errors) or "request validation failed"


def _format_validation_error(error: dict) -> str:
    field = ".".join(str(part) for part in error.get("loc", ()))
    message = str(error.get("msg", "invalid value"))
    return f"{field}: {message}" if field else message


app = create_app()
