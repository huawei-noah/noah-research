"""Small async retry helpers for infrastructure clients."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from opentelemetry.trace import SpanKind, Status, StatusCode

from ..logging import get_logger, get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class AsyncRetryProxy:
    """Proxy coroutine methods through bounded retry."""

    def __init__(
        self,
        target: Any,
        *,
        operation_name: str,
        max_attempts: int,
        base_delay: float,
        retryable: Callable[[Exception], bool],
    ) -> None:
        self._target = target
        self._operation_name = operation_name
        self._max_attempts = max(1, max_attempts)
        self._base_delay = max(0.0, base_delay)
        self._retryable = retryable
        self._wrapped_methods: dict[str, Callable[..., Any]] = {}

    @property
    def target(self) -> Any:
        """Return the wrapped target."""

        return self._target

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if not inspect.iscoroutinefunction(attr):
            return attr
        if name not in self._wrapped_methods:
            self._wrapped_methods[name] = self._wrap(name, attr)
        return self._wrapped_methods[name]

    def _wrap(self, method_name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        span_name = f"{self._operation_name}.{method_name}"

        @wraps(method)
        async def retried(*args: Any, **kwargs: Any) -> Any:
            # One CLIENT span per logical call wraps the whole retry loop, so the
            # recorded Duration is the latency the caller actually experiences
            # (including retries + backoff). Lightweight: one span per call, the
            # underlying op already does network I/O.
            with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attribute("db.system", self._operation_name)
                span.set_attribute("db.operation", method_name)
                collection = kwargs.get("collection_name")
                if isinstance(collection, str):
                    span.set_attribute("db.collection", collection)

                last_err: Exception | None = None
                for attempt in range(1, self._max_attempts + 1):
                    try:
                        result = method(*args, **kwargs)
                        if inspect.isawaitable(result):
                            result = await result
                        span.set_attribute("db.retry.attempts", attempt)
                        return result
                    except Exception as exc:
                        if not self._retryable(exc):
                            span.set_attribute("db.retry.attempts", attempt)
                            span.record_exception(exc)
                            span.set_status(Status(StatusCode.ERROR, str(exc)))
                            raise
                        last_err = exc
                        if attempt == self._max_attempts:
                            logger.error(
                                f"{self._operation_name}_retry_exhausted",
                                method=method_name,
                                attempt=attempt,
                                max_attempts=self._max_attempts,
                                error=str(exc),
                            )
                            span.set_attribute("db.retry.attempts", attempt)
                            span.record_exception(exc)
                            span.set_status(Status(StatusCode.ERROR, str(exc)))
                            raise
                        delay = retry_delay(self._base_delay, attempt)
                        logger.warning(
                            f"{self._operation_name}_retrying",
                            method=method_name,
                            attempt=attempt,
                            max_attempts=self._max_attempts,
                            retry_after=delay,
                            error=str(exc),
                        )
                        await asyncio.sleep(delay)

                assert last_err is not None
                raise last_err

        return retried


def run_sync_with_retry(
    operation: Callable[[], Any],
    *,
    operation_name: str,
    max_attempts: int,
    base_delay: float,
    retryable: Callable[[Exception], bool],
) -> Any:
    """Run a sync operation with bounded retry."""

    max_attempts = max(1, max_attempts)
    base_delay = max(0.0, base_delay)
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if not retryable(exc):
                raise
            if attempt == max_attempts:
                logger.error(
                    f"{operation_name}_retry_exhausted",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=str(exc),
                )
                raise
            delay = retry_delay(base_delay, attempt)
            logger.warning(
                f"{operation_name}_retrying",
                attempt=attempt,
                max_attempts=max_attempts,
                retry_after=delay,
                error=str(exc),
            )
            if delay > 0:
                time.sleep(delay)

    raise RuntimeError(f"unreachable retry state for {operation_name}")


def retry_delay(base_delay: float, attempt: int, *, cap: float = 10.0) -> float:
    if base_delay <= 0:
        return 0.0
    return min(base_delay * (2 ** max(0, attempt - 1)), cap)
