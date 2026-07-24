"""Async client-side concurrency gates for database drivers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal

DatabaseClientKind = Literal["qdrant", "neo4j"]


@dataclass(frozen=True, slots=True)
class DatabaseClientConcurrencyLimits:
    """Client-side concurrency limits used by database drivers."""

    qdrant: int
    neo4j: int


def capped_db_client_concurrency(value: int | None, *, cap: int) -> int:
    """Return a positive database client concurrency limit capped by configured policy."""

    configured = value if value is not None and value > 0 else cap
    if configured <= 0:
        raise ValueError("database client concurrency must be positive when cap is disabled")
    if cap <= 0:
        return configured
    return min(configured, cap)


class AsyncClientConcurrencyLimiter:
    """Proxy async client calls through a per-client semaphore.

    Instances are owned by one event-loop-scoped database client registry entry.
    They should not be shared across event loops.
    """

    def __init__(self, target: Any, *, max_concurrency: int | None = None) -> None:
        self._target = target
        if max_concurrency is None or max_concurrency <= 0:
            raise ValueError("database client concurrency limiter requires a positive max_concurrency")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._wrapped_methods: dict[str, Callable[..., Any]] = {}

    @property
    def target(self) -> Any:
        """Return the wrapped client or driver."""

        return self._target

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if not inspect.iscoroutinefunction(attr):
            return attr
        if name not in self._wrapped_methods:
            self._wrapped_methods[name] = self._wrap(attr)
        return self._wrapped_methods[name]

    def _wrap(self, method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        async def limited(*args: Any, **kwargs: Any) -> Any:
            async with self._semaphore:
                result = method(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

        return limited
