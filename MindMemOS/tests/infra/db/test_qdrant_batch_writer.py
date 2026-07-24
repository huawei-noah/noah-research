import asyncio

import pytest
from mindmemos.config import QdrantConfig
from mindmemos.infra.db.engine import QdrantEngine
from mindmemos.infra.db.qdrant_batch_writer import QdrantBatchWriter
from qdrant_client import models as qmodels


def _point(point_id: str) -> qmodels.PointStruct:
    return qmodels.PointStruct(id=point_id, vector={}, payload={"project_id": "proj"})


@pytest.mark.asyncio
async def test_qdrant_batch_writer_batches_same_collection_by_size():
    calls: list[tuple[str, list[qmodels.PointStruct]]] = []

    async def raw_upsert(collection: str, points: list[qmodels.PointStruct]) -> None:
        calls.append((collection, points))

    writer = QdrantBatchWriter(
        raw_upsert,
        batch_size=3,
        flush_interval_ms=1000,
        max_queue_size=10,
        max_inflight_batches=1,
    )

    await asyncio.gather(
        writer.upsert("a", [_point("1")]),
        writer.upsert("a", [_point("2"), _point("3")]),
    )
    await writer.close()

    assert [(collection, [point.id for point in points]) for collection, points in calls] == [("a", ["1", "2", "3"])]


@pytest.mark.asyncio
async def test_qdrant_batch_writer_keeps_collections_separate():
    calls: list[tuple[str, list[qmodels.PointStruct]]] = []

    async def raw_upsert(collection: str, points: list[qmodels.PointStruct]) -> None:
        calls.append((collection, points))

    writer = QdrantBatchWriter(
        raw_upsert,
        batch_size=10,
        flush_interval_ms=10,
        max_queue_size=10,
        max_inflight_batches=1,
    )

    await asyncio.gather(
        writer.upsert("a", [_point("1")]),
        writer.upsert("b", [_point("2")]),
    )
    await writer.close()

    assert {collection: [point.id for point in points] for collection, points in calls} == {
        "a": ["1"],
        "b": ["2"],
    }


@pytest.mark.asyncio
async def test_qdrant_batch_writer_propagates_batch_failure():
    async def raw_upsert(collection: str, points: list[qmodels.PointStruct]) -> None:
        raise RuntimeError("qdrant unavailable")

    writer = QdrantBatchWriter(
        raw_upsert,
        batch_size=1,
        flush_interval_ms=1000,
        max_queue_size=10,
        max_inflight_batches=1,
    )

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        await writer.upsert("a", [_point("1")])

    await writer.close()


@pytest.mark.asyncio
async def test_qdrant_engine_uses_batch_writer_when_enabled():
    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[qmodels.PointStruct]]] = []

        async def upsert(self, *, collection_name: str, points: list[qmodels.PointStruct]) -> None:
            self.calls.append((collection_name, points))

        async def close(self) -> None:
            return None

    client = Client()
    engine = QdrantEngine(
        QdrantConfig(
            url="http://unused",
            batch_upsert_enabled=True,
            batch_upsert_size=2,
            batch_upsert_flush_interval_ms=1000,
        ),
        client=client,
    )

    await asyncio.gather(
        engine.upsert("a", [_point("1")]),
        engine.upsert("a", [_point("2")]),
    )
    await engine.close()

    assert [(collection, [point.id for point in points]) for collection, points in client.calls] == [("a", ["1", "2"])]
