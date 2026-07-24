import asyncio

import pytest
from mindmemos.config import Neo4jConfig, QdrantConfig
from mindmemos.infra.db import engine as qdrant_engine
from mindmemos.infra.db.engine import QdrantEngine
from mindmemos.infra.db.models import QdrantCollectionSpec
from mindmemos.infra.db.neo4j import Neo4jStore
from mindmemos.infra.db.qdrant import QdrantStore


class _TrackingQdrantClient:
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.active = 0
        self.max_active = 0

    async def retrieve(self, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self.release.wait()
            return []
        finally:
            self.active -= 1


class _TrackingNeo4jDriver:
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.active = 0
        self.max_active = 0

    async def execute_query(self, query, params=None, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self.release.wait()
            return _Neo4jResult()
        finally:
            self.active -= 1


class _Neo4jResult:
    records = []


@pytest.mark.asyncio
async def test_qdrant_and_neo4j_client_concurrency_are_counted_independently():
    qdrant_release = asyncio.Event()
    neo4j_release = asyncio.Event()
    qdrant_client = _TrackingQdrantClient(qdrant_release)
    neo4j_driver = _TrackingNeo4jDriver(neo4j_release)
    qdrant = QdrantStore(_qdrant_config(max_client_concurrency=3), client=qdrant_client)
    neo4j = Neo4jStore(Neo4jConfig(uri="bolt://unused", max_client_concurrency=3), driver=neo4j_driver)

    qdrant_tasks = [asyncio.create_task(qdrant.get_memories("proj", [str(index)])) for index in range(10)]
    neo4j_tasks = [asyncio.create_task(neo4j.run_read("MATCH (n) RETURN n")) for _ in range(10)]
    await _wait_for_active(qdrant_client, 3)
    await _wait_for_active(neo4j_driver, 3)

    assert qdrant_client.max_active == 3
    assert neo4j_driver.max_active == 3
    assert qdrant_client.active + neo4j_driver.active == 6

    qdrant_release.set()
    neo4j_release.set()
    await asyncio.gather(*qdrant_tasks, *neo4j_tasks)


def test_qdrant_engine_passes_grpc_config(monkeypatch):
    captured = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(qdrant_engine, "AsyncQdrantClient", FakeQdrantClient)

    QdrantEngine(
        QdrantConfig(
            url="http://localhost:6333",
            grpc_port=6334,
            prefer_grpc=True,
            max_client_concurrency=1,
            max_client_concurrency_cap=1,
        )
    )

    assert captured["url"] == "http://localhost:6333"
    assert captured["grpc_port"] == 6334
    assert captured["prefer_grpc"] is True


@pytest.mark.asyncio
async def test_qdrant_engine_updates_existing_collection_payload_storage():
    class FakeQdrantClient:
        def __init__(self):
            self.updated = []

        async def collection_exists(self, collection_name):
            return True

        async def update_collection(self, **kwargs):
            self.updated.append(kwargs)

        async def create_payload_index(self, **kwargs):
            return None

    client = FakeQdrantClient()
    engine = QdrantEngine(_qdrant_config(max_client_concurrency=1), client=client)

    await engine.ensure_collection(
        QdrantCollectionSpec(name="test_memos", vector_size=2, on_disk_payload=False),
    )

    assert client.updated[0]["collection_name"] == "test_memos"
    assert client.updated[0]["collection_params"].on_disk_payload is False


def _qdrant_config(*, max_client_concurrency: int) -> QdrantConfig:
    return QdrantConfig(
        url="http://unused",
        memory_collection="test_memos",
        entity_collection="test_entities",
        source_collection="test_sources",
        add_record_collection="test_add_records",
        search_record_collection="test_search_records",
        vector_size=2,
        max_client_concurrency=max_client_concurrency,
    )


async def _wait_for_active(client, expected: int) -> None:
    for _ in range(100):
        if client.active == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {expected} active calls, got {client.active}")
