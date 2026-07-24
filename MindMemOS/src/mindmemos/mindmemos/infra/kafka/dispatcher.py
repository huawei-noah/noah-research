"""Concurrent dispatcher that preserves ordering within each dispatch key.

The dispatcher is infrastructure-only: it groups work by an opaque key, applies
per-key and global concurrency limits, and back-pressures the consumer when the
in-memory buffer is full. Business code decides what the key means.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from ...logging import get_logger

logger = get_logger(__name__)

# Infrastructure only defines this header as the grouping key carrier.
# API or pipeline code chooses stable values such as account_id or project_id.
DISPATCH_KEY_HEADER = "x-mm-dispatch-key"

ProcessFn = Callable[[Any], Awaitable[None]]
CompleteFn = Callable[[Any], Awaitable[None]]


class OrderedKeyedDispatcher:
    """Ordered keyed dispatcher owned by one Kafka consumer."""

    def __init__(
        self,
        *,
        global_max_concurrency: int,
        per_key_max_concurrency: int,
        max_buffered: int,
        process: ProcessFn,
        on_complete: CompleteFn,
        shared_global_semaphore: asyncio.Semaphore | None = None,
    ):
        """Initialize the dispatcher.

        Args:
            global_max_concurrency: Global concurrency limit across all keys.
            per_key_max_concurrency: Concurrency limit within one key; 1 preserves strict order.
            max_buffered: Maximum number of submitted unfinished items before ``submit`` waits.
            process: Callback that processes one item, including retry and DLQ behavior.
            on_complete: Callback that runs after processing completes, typically to commit offsets.
            shared_global_semaphore: Optional process-wide concurrency gate shared across dispatchers.
        """
        _validate_positive("global_max_concurrency", global_max_concurrency)
        _validate_positive("per_key_max_concurrency", per_key_max_concurrency)
        _validate_positive("max_buffered", max_buffered)
        self._sem = asyncio.Semaphore(global_max_concurrency)
        self._per_key = per_key_max_concurrency
        self._max_buffered = max_buffered
        self._process = process
        self._on_complete = on_complete
        self._shared_sem = shared_global_semaphore

        self._cond = asyncio.Condition()
        self._queues: dict[str, asyncio.Queue] = {}
        self._workers: dict[str, set[asyncio.Task]] = {}
        self._buffered = 0
        self._closed = False

    async def submit(self, key: str, item: Any) -> None:
        """Submit one item to its key queue and wait when the buffer is full."""
        async with self._cond:
            while self._buffered >= self._max_buffered and not self._closed:
                await self._cond.wait()
            if self._closed:
                return
            self._buffered += 1
            queue = self._queues.get(key)
            if queue is None:
                queue = asyncio.Queue()
                self._queues[key] = queue
            queue.put_nowait(item)

            workers = self._workers.setdefault(key, set())
            if len(workers) < self._per_key:
                task = asyncio.create_task(self._run_key(key), name=f"kafka-dispatch-{key}")
                workers.add(task)

    async def _run_key(self, key: str) -> None:
        """Consume one key queue, preserving the configured per-key concurrency."""
        current = asyncio.current_task()
        while True:
            async with self._cond:
                queue = self._queues.get(key)
                # Exit while holding the lock to avoid racing submit's worker-spawn check.
                if queue is None or queue.empty():
                    workers = self._workers.get(key)
                    if workers is not None:
                        workers.discard(current)
                        if not workers:
                            self._workers.pop(key, None)
                            self._queues.pop(key, None)
                    return
                item = queue.get_nowait()

            try:
                async with self._sem:
                    if self._shared_sem is None:
                        await self._process(item)
                    else:
                        async with self._shared_sem:
                            await self._process(item)
            except Exception:
                logger.exception("dispatcher process callback crashed", dispatch_key=key)
            finally:
                try:
                    await self._on_complete(item)
                except Exception:
                    logger.exception("dispatcher on_complete callback crashed", dispatch_key=key)
                async with self._cond:
                    self._buffered -= 1
                    self._cond.notify_all()

    async def drain(self) -> None:
        """Wait for all submitted items to finish before graceful shutdown."""
        async with self._cond:
            while self._buffered > 0:
                await self._cond.wait()
        tasks = [task for workers in self._workers.values() for task in workers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        """Close immediately by waking blocked submitters and cancelling workers."""
        async with self._cond:
            self._closed = True
            self._cond.notify_all()
        tasks = [task for workers in self._workers.values() for task in workers]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._queues.clear()


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be a positive integer >= 1")
