"""OpenTelemetry tracing and logging infrastructure."""

from __future__ import annotations

import logging
import socket
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from ..config import TelemetryConfig
from ..logging import get_logger

logger = get_logger(__name__)

_provider: TracerProvider | None = None
_log_provider: LoggerProvider | None = None
_log_handler: LoggingHandler | None = None


def _signal_endpoint(base: str, signal: str) -> str:
    """Build the OTLP HTTP endpoint for one telemetry signal."""
    base = base.rstrip("/")
    if "/v1/" in base:
        base = base.rsplit("/v1/", 1)[0]
    return f"{base}/v1/{signal}"


def setup_tracer_provider(config: TelemetryConfig) -> TracerProvider | None:
    """Build and install the global tracer provider."""
    global _provider
    if _provider is not None:
        return _provider

    if config.span_type != "simple" and not _otlp_endpoint_available(config.telemetry_endpoint):
        logger.warning(
            "telemetry endpoint unavailable; telemetry export disabled",
            endpoint=config.telemetry_endpoint,
        )
        return None

    resource = Resource.create({SERVICE_NAME: config.service_name})
    provider = TracerProvider(resource=resource)

    if config.span_type == "simple":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        exporter = OTLPSpanExporter(
            endpoint=_signal_endpoint(config.telemetry_endpoint, "traces"),
            timeout=config.telemetry_timeout,
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                max_queue_size=config.max_queue_size,
                max_export_batch_size=config.max_export_batch_size,
            )
        )

    trace.set_tracer_provider(provider)
    _provider = provider

    if config.logs_enabled:
        _setup_logger_provider(config, resource)

    logger.info(
        "tracer provider installed",
        service_name=config.service_name,
        span_type=config.span_type,
        endpoint=config.telemetry_endpoint,
        logs_enabled=config.logs_enabled,
    )
    return provider


def _otlp_endpoint_available(endpoint: str | None) -> bool:
    """Return whether the OTLP HTTP endpoint accepts TCP connections."""
    parsed = urlparse(endpoint or "")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not parsed.hostname:
        return False

    try:
        # ponytail: one-second startup probe; make it configurable if remote OTel endpoints need slower cold starts.
        with socket.create_connection((parsed.hostname, port), timeout=1):
            return True
    except OSError:
        return False


def _setup_logger_provider(config: TelemetryConfig, resource: Resource) -> None:
    """Install the OpenTelemetry logger provider."""
    global _log_provider, _log_handler

    log_provider = LoggerProvider(resource=resource)
    if config.span_type == "simple":
        log_provider.add_log_record_processor(SimpleLogRecordProcessor(ConsoleLogExporter()))
    else:
        log_exporter = OTLPLogExporter(
            endpoint=_signal_endpoint(config.telemetry_endpoint, "logs"),
            timeout=config.telemetry_timeout,
        )
        log_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                log_exporter,
                max_queue_size=config.max_queue_size,
                max_export_batch_size=config.max_export_batch_size,
            )
        )

    set_logger_provider(log_provider)
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=log_provider)
    # Do not forward the OTel SDK's own logs (e.g. exporter connection errors).
    # Forwarding them would re-enter the log pipeline and create a feedback loop
    # whenever the collector is unreachable.
    handler.addFilter(lambda record: not record.name.startswith("opentelemetry"))
    logging.getLogger().addHandler(handler)

    _log_provider = log_provider
    _log_handler = handler


def shutdown_tracer_provider() -> None:
    """Flush and shut down the tracer provider."""
    global _provider, _log_provider, _log_handler

    if _log_handler is not None:
        logging.getLogger().removeHandler(_log_handler)
        _log_handler = None
    if _log_provider is not None:
        _log_provider.shutdown()
        _log_provider = None
    if _provider is not None:
        _provider.shutdown()
        _provider = None
