"""Retry helpers for optional NLP libraries."""

from __future__ import annotations

import errno
from collections.abc import Callable
from typing import TypeVar

from ...config import TextProcessingConfig
from ...infra import run_sync_with_retry

T = TypeVar("T")


def run_nlp_with_retry(operation: Callable[[], T], *, config: TextProcessingConfig, operation_name: str) -> T:
    """Run one NLP-library operation with bounded transient retry."""

    return run_sync_with_retry(
        operation,
        operation_name=operation_name,
        max_attempts=config.nlp_max_retries,
        base_delay=config.nlp_retry_base_delay,
        retryable=is_retryable_nlp_error,
    )


def is_retryable_nlp_error(exc: Exception) -> bool:
    module = type(exc).__module__
    if module.startswith(("httpx", "httpcore", "requests", "urllib3")):
        return True
    if isinstance(exc, (TimeoutError, InterruptedError, BlockingIOError)):
        return True
    return isinstance(exc, OSError) and getattr(exc, "errno", None) in {
        errno.EAGAIN,
        errno.EINTR,
        errno.ETIMEDOUT,
    }
