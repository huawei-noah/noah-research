"""Client-side Qdrant upsert batching."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from qdrant_client import models as qmodels


@dataclass(slots=True)
class _QueuedUpsert:
    collection: str
    points: list[qmodels.PointStruct]
    future: asyncio.Future[None]


class QdrantBatchWriter:
    """Collect small upserts into per-collection batches while preserving await semantics."""

    def __init__(
        self,
        raw_upsert: Callable[[str, list[qmodels.PointStruct]], Awaitable[None]],
        *,
        batch_size: int,
        flush_interval_ms: int,
        max_queue_size: int,
        max_inflight_batches: int,
    ) -> None:
        self._raw_upsert = raw_upsert
        self._batch_size = max(1, batch_size)
        self._flush_interval = max(1, flush_interval_ms) / 1000
        self._queue: asyncio.Queue[_QueuedUpsert | None] = asyncio.Queue(maxsize=max(1, max_queue_size))
        self._inflight = asyncio.Semaphore(max(1, max_inflight_batches))
        self._batches: dict[str, list[_QueuedUpsert]] = defaultdict(list)
        self._runner: asyncio.Task[None] | None = None
        self._flush_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    async def upsert(self, collection: str, points: list[qmodels.PointStruct]) -> None:
        """Queue points and wait until their batch is flushed."""

        if not points:
            return
        if self._closed:
            raise RuntimeError("qdrant batch writer is closed")
        self._ensure_runner()
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._queue.put(_QueuedUpsert(collection=collection, points=points, future=future))
        await future

    async def close(self) -> None:
        """Flush queued writes and stop the background runner."""

        self._closed = True
        if self._runner is None:
            return
        await self._queue.put(None)
        await self._runner
        if self._flush_tasks:
            await asyncio.gather(*self._flush_tasks)

    def _ensure_runner(self) -> None:
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
            except TimeoutError:
                await self._flush_all()
                continue

            if item is None:
                await self._flush_all()
                break

            self._batches[item.collection].append(item)
            if self._point_count(self._batches[item.collection]) >= self._batch_size:
                await self._flush_collection(item.collection)

    async def _flush_all(self) -> None:
        for collection in list(self._batches):
            await self._flush_collection(collection)

    async def _flush_collection(self, collection: str) -> None:
        requests = self._batches.pop(collection, [])
        if not requests:
            return
        await self._inflight.acquire()
        task = asyncio.create_task(self._flush(collection, requests))
        self._flush_tasks.add(task)
        task.add_done_callback(self._flush_tasks.discard)

    async def _flush(self, collection: str, requests: list[_QueuedUpsert]) -> None:
        points = [point for request in requests for point in request.points]
        try:
            await self._raw_upsert(collection, points)
        except Exception as exc:
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(exc)
            return
        finally:
            self._inflight.release()

        for request in requests:
            if not request.future.done():
                request.future.set_result(None)

    @staticmethod
    def _point_count(requests: list[_QueuedUpsert]) -> int:
        return sum(len(request.points) for request in requests)
