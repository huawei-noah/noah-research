import logging

import structlog

from .tracing import (
    add_span_event,
    carrier_to_headers,
    configure_tracing,
    current_trace_id,
    extract_trace_context,
    get_tracer,
    headers_to_carrier,
    inject_trace_context,
    traced,
    traced_awaitable,
    traced_gather,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "configure_tracing",
    "get_tracer",
    "inject_trace_context",
    "extract_trace_context",
    "headers_to_carrier",
    "carrier_to_headers",
    "current_trace_id",
    "add_span_event",
    "traced",
    "traced_awaitable",
    "traced_gather",
]


def configure_logging(level: str = "INFO", *, json: bool = False, colors: bool | None = None) -> None:
    """Call once at application startup to set up structlog.

    Args:
        level: Root log level.
        json: Use JSON renderer (for production).
        colors: Force ANSI colors on/off. ``None`` = auto-detect (off when
            stderr is redirected to a file / pipe).
    """
    import os
    import sys

    if colors is None:
        colors = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json:
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=colors)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Strip ANSI escapes from LiteLLM and other third-party loggers that
    # force colors regardless of terminal detection.
    if not colors:
        os.environ.setdefault("NO_COLOR", "1")
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
