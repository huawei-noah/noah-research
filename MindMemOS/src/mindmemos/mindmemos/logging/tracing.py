"""Public tracing helpers."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import json
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Mapping, TypeVar, overload

import structlog
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.trace import Status, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from ..config import TelemetryConfig

_TRACEPARENT = "traceparent"

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")
_MAX_ATTRIBUTE_CHARS = 8192


def configure_tracing(config: "TelemetryConfig") -> None:
    """Install the global tracer provider during application startup."""
    if not config.enabled:
        return
    # Import lazily to avoid a module-load cycle between logging and infra.
    from ..infra import setup_tracer_provider

    setup_tracer_provider(config)


def get_tracer(name: str) -> Tracer:
    """Return a tracer, falling back to the no-op provider when tracing is disabled."""
    return trace.get_tracer(name)


def inject_trace_context(carrier: dict[str, str] | None = None) -> dict[str, str]:
    """Inject the active span context into a carrier dictionary."""
    if carrier is None:
        carrier = {}
    inject(carrier)
    return carrier


def extract_trace_context(carrier: Mapping[str, str]) -> otel_context.Context:
    """Extract an OpenTelemetry context from a carrier dictionary."""
    return extract(carrier)


def headers_to_carrier(
    headers: Iterable[tuple[str, bytes]] | None,
) -> dict[str, str]:
    """Kafka headers(list[(str, bytes)]) → carrier(dict[str,str])。"""
    carrier: dict[str, str] = {}
    if not headers:
        return carrier
    for key, value in headers:
        if value is not None:
            carrier[key] = value.decode("utf-8")
    return carrier


def carrier_to_headers(carrier: Mapping[str, str]) -> list[tuple[str, bytes]]:
    """carrier(dict[str,str]) → Kafka headers(list[(str, bytes)])。"""
    return [(key, value.encode("utf-8")) for key, value in carrier.items()]


def current_trace_id() -> str | None:
    """Return the active trace id as a hex string when available."""
    span_ctx = trace.get_current_span().get_span_context()
    if not span_ctx.is_valid:
        return None
    return trace.format_trace_id(span_ctx.trace_id)


def add_span_event(name: str, attributes: Mapping[str, Any] | None = None) -> None:
    """Record an event on the active span when one is recording."""
    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return
    attrs = {key: _attr_value(val) for key, val in attributes.items()} if attributes else None
    span.add_event(name, attributes=attrs)


def _attr_value(value: Any) -> AttributeValue:
    """Convert arbitrary values to OpenTelemetry-compatible attribute values."""
    if value is None:
        return "null"
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_attribute(value)
    if isinstance(value, bytes):
        try:
            return _truncate_attribute(value.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return _truncate_attribute(repr(value))

    try:
        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            return _truncate_attribute(value.model_dump_json())
    except Exception:  # noqa: BLE001
        pass

    try:
        target = value
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            target = dataclasses.asdict(value)
        return _truncate_attribute(json.dumps(target, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001 - non-JSON-friendly values fall back to repr below.
        pass

    try:
        return _truncate_attribute(repr(value))
    except Exception:  # noqa: BLE001
        return f"<unreprable {type(value).__name__}>"


def _truncate_attribute(value: str) -> str:
    if len(value) <= _MAX_ATTRIBUTE_CHARS:
        return value
    return f"{value[:_MAX_ATTRIBUTE_CHARS]}...<truncated chars={len(value) - _MAX_ATTRIBUTE_CHARS}>"


def _bind_args(func: Callable[..., Any], args: tuple, kwargs: dict) -> dict[str, AttributeValue]:
    """Handle bind args."""
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
    except TypeError:
        # If signature binding fails, fall back to raw positional and keyword arguments.
        result = {f"args.{i}": _attr_value(a) for i, a in enumerate(args)}
        result.update({f"kwargs.{k}": _attr_value(v) for k, v in kwargs.items()})
        return result
    # Skip self and cls because they add noise for method calls.
    return {f"arg.{name}": _attr_value(val) for name, val in bound.arguments.items() if name not in ("self", "cls")}


def _record_error(span: trace.Span, exc: BaseException) -> None:
    """Record exception details and timestamp on a span."""
    from datetime import datetime, timezone

    span.set_status(Status(StatusCode.ERROR, str(exc)))
    # record_exception includes an event timestamp; add a readable ISO attribute as well.
    span.set_attribute("error.time", datetime.now(timezone.utc).isoformat())
    span.set_attribute("error.type", type(exc).__name__)
    span.record_exception(exc)


async def traced_awaitable(
    name: str,
    awaitable: Awaitable[T],
    *,
    attributes: Mapping[str, Any] | None = None,
    record_result: bool = False,
    tracer_name: str | None = None,
) -> T:
    """Run one awaitable inside a child span.

    Args:
        name: Span name.
        awaitable: Awaitable to run while the span is active.
        attributes: Optional span attributes.
        record_result: Whether to store the awaited result as ``result``.
        tracer_name: Optional tracer name; defaults to this module.

    Returns:
        The awaited result.
    """
    span_tracer = get_tracer(tracer_name or __name__)
    with span_tracer.start_as_current_span(name, record_exception=False) as span:
        for key, value in (attributes or {}).items():
            span.set_attribute(key, _attr_value(value))
        try:
            result = await awaitable
        except BaseException as exc:
            _record_error(span, exc)
            raise
        if record_result:
            span.set_attribute("result", _attr_value(result))
        span.set_status(Status(StatusCode.OK))
        return result


@overload
async def traced_gather(
    name: str,
    *awaitables: Awaitable[T],
    attributes: Mapping[str, Any] | None = None,
    record_result: bool = False,
    return_exceptions: Literal[False] = False,
    tracer_name: str | None = None,
) -> list[T]: ...


@overload
async def traced_gather(
    name: str,
    *awaitables: Awaitable[T],
    attributes: Mapping[str, Any] | None = None,
    record_result: bool = False,
    return_exceptions: Literal[True] = True,
    tracer_name: str | None = None,
) -> list[T | BaseException]: ...


async def traced_gather(
    name: str,
    *awaitables: Awaitable[T],
    attributes: Mapping[str, Any] | None = None,
    record_result: bool = False,
    return_exceptions: bool = False,
    tracer_name: str | None = None,
) -> list[T] | list[T | BaseException]:
    """Run ``asyncio.gather`` inside a child span.

    Gather is created after the span starts, so child tasks inherit this span as
    their active parent when they create nested spans.
    """
    span_tracer = get_tracer(tracer_name or __name__)
    with span_tracer.start_as_current_span(name, record_exception=False) as span:
        for key, value in (attributes or {}).items():
            span.set_attribute(key, _attr_value(value))
        try:
            result = await asyncio.gather(*awaitables, return_exceptions=return_exceptions)
        except BaseException as exc:
            _record_error(span, exc)
            raise
        if record_result:
            span.set_attribute("result", _attr_value(result))
        span.set_status(Status(StatusCode.OK))
        return result


def traced(
    name: str | None = None,
    *,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[F], F]:
    """Create a decorator that records function calls as OpenTelemetry spans."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        tracer = get_tracer(func.__module__)
        # Use structlog directly to avoid a cycle with this package's get_logger export.
        logger = structlog.stdlib.get_logger(func.__module__)

        def _start(args: tuple, kwargs: dict) -> tuple[trace.Span, dict[str, AttributeValue]]:
            span = tracer.start_span(span_name)
            arg_fields = _bind_args(func, args, kwargs) if record_args else {}
            for key, val in arg_fields.items():
                span.set_attribute(key, val)
            return span, arg_fields

        def _on_success(span: trace.Span, arg_fields: dict[str, AttributeValue], result: Any) -> None:
            result_repr = _attr_value(result) if record_result else None
            if result_repr is not None:
                span.set_attribute("result", result_repr)
            span.set_status(Status(StatusCode.OK))
            # Fall back to logs when the span is not recording.
            if not span.is_recording():
                fields: dict[str, Any] = {"function": span_name, **arg_fields}
                if result_repr is not None:
                    fields["result"] = result_repr
                logger.debug("traced call", **fields)

        def _on_error(span: trace.Span, arg_fields: dict[str, AttributeValue], exc: BaseException) -> None:
            _record_error(span, exc)
            if not span.is_recording():
                logger.error(
                    "traced call failed",
                    function=span_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    exc_info=exc,
                    **arg_fields,
                )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                span, arg_fields = _start(args, kwargs)
                # _record_error records exceptions once, so avoid duplicate exception events.
                with trace.use_span(span, end_on_exit=True, record_exception=False):
                    try:
                        result = await func(*args, **kwargs)
                    except BaseException as exc:
                        _on_error(span, arg_fields, exc)
                        raise
                    _on_success(span, arg_fields, result)
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            span, arg_fields = _start(args, kwargs)
            with trace.use_span(span, end_on_exit=True, record_exception=False):
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    _on_error(span, arg_fields, exc)
                    raise
                _on_success(span, arg_fields, result)
                return result

        return sync_wrapper  # type: ignore[return-value]

    return decorator
