import asyncio

import pytest
import pytest_asyncio
from mindmemos.infra.db.filters import build_filter, match_any, match_value
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException

from mindmemos.config import QdrantConfig
from mindmemos.infra.db import (
    AddRecordPoint,
    EntityPoint,
    MemoryPoint,
    QdrantStore,
    SearchRecordPoint,
    SparseVectorData,
)


class ConcurrencyTrackingQdrantClient:
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

    async def close(self):
        return None


class RetryThenSucceedQdrantClient:
    def __init__(self, fail_count: int) -> None:
        self.fail_count = fail_count
        self.calls = 0

    async def retrieve(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise ResponseHandlingException(RuntimeError("temporary qdrant transport failure"))
        return []

    async def close(self):
        return None


@pytest_asyncio.fixture
async def qdrant_store():
    client = AsyncQdrantClient(":memory:")
    cfg = QdrantConfig(
        url="http://unused",
        memory_collection="test_memos",
        entity_collection="test_entities",
        source_collection="test_sources",
        add_record_collection="test_add_records",
        search_record_collection="test_search_records",
        vector_size=2,
    )
    store = QdrantStore(cfg, client=client)
    await store.ensure_schema()
    try:
        yield store
    finally:
        await client.close()


def _point(memory_id: str, content: str, dense: list[float], sparse_index: int, *, user_id: str | None = None):
    return MemoryPoint(
        memory_id=memory_id,
        semantic_vector=dense,
        bm25_vector=SparseVectorData(indices=[sparse_index], values=[1.0]),
        payload={
            "memory_id": memory_id,
            "project_id": "proj",
            "user_id": user_id,
            "content": content,
            "status": "active",
            "mem_type": "fact",
        },
    )


def _entity_point(entity_id: str, name: str, dense: list[float], sparse_index: int):
    return EntityPoint(
        entity_id=entity_id,
        vector=dense,
        bm25_vector=SparseVectorData(indices=[sparse_index], values=[1.0]),
        payload={
            "entity_id": entity_id,
            "project_id": "proj",
            "entity_name": name,
            "entity_type": "person",
            "status": "active",
            "metadata": {"search_fields": [name]},
        },
    )


def test_qdrant_project_filter_injection_is_idempotent() -> None:
    qfilter = build_filter(must=[match_value("project_id", "proj"), match_any("status", ["active"])])

    scoped = QdrantStore._with_project_filter("proj", qfilter)

    project_clauses = [
        clause
        for clause in scoped.must or []
        if getattr(clause, "key", None) == "project_id"
        and getattr(getattr(clause, "match", None), "value", None) == "proj"
    ]
    assert len(project_clauses) == 1
    assert scoped.must[0].key == "project_id"
    assert scoped.must[1].key == "status"


def test_qdrant_project_filter_is_added_when_missing_or_different() -> None:
    without_project = build_filter(must=[match_any("status", ["active"])])
    scoped = QdrantStore._with_project_filter("proj", without_project)
    assert scoped.must[0].key == "project_id"
    assert scoped.must[0].match.value == "proj"
    assert scoped.must[1].key == "status"

    wrong_project = build_filter(must=[match_value("project_id", "other")])
    scoped_wrong = QdrantStore._with_project_filter("proj", wrong_project)
    assert [clause.match.value for clause in scoped_wrong.must if clause.key == "project_id"] == ["proj", "other"]


@pytest.mark.asyncio
async def test_qdrant_upsert_retrieve_and_search_modes(qdrant_store):
    await qdrant_store.upsert_memories(
        [
            _point("00000000-0000-0000-0000-000000000001", "FastAPI memory", [1.0, 0.0], 1),
            _point("00000000-0000-0000-0000-000000000002", "Neo4j memory", [0.0, 1.0], 2),
        ],
    )

    records = await qdrant_store.get_memories("proj", ["00000000-0000-0000-0000-000000000001"])
    assert records[0].payload["content"] == "FastAPI memory"

    qfilter = build_filter(must=[match_value("project_id", "proj"), match_any("status", ["active"])])
    semantic_hits = await qdrant_store.search_memory_dense("proj", [1.0, 0.0], filter_=qfilter, limit=2)
    assert semantic_hits[0].point_id == "00000000-0000-0000-0000-000000000001"

    bm25_hits = await qdrant_store.search_memory_sparse(
        "proj",
        SparseVectorData(indices=[2], values=[1.0]),
        filter_=qfilter,
        limit=2,
    )
    assert [hit.point_id for hit in bm25_hits] == ["00000000-0000-0000-0000-000000000002"]

    rrf_hits = await qdrant_store.search_memory_hybrid(
        "proj",
        [1.0, 0.0],
        SparseVectorData(indices=[1], values=[1.0]),
        filter_=qfilter,
        limit=2,
    )
    assert rrf_hits[0].point_id == "00000000-0000-0000-0000-000000000001"


@pytest.mark.asyncio
async def test_qdrant_entity_search_modes_include_sparse_vectors(qdrant_store):
    await qdrant_store.upsert_entities(
        [
            _entity_point("00000000-0000-0000-0000-000000000021", "Alice", [1.0, 0.0], 21),
            _entity_point("00000000-0000-0000-0000-000000000022", "Bob", [0.0, 1.0], 22),
        ]
    )

    qfilter = build_filter(must=[match_value("project_id", "proj"), match_value("entity_type", "person")])
    semantic_hits = await qdrant_store.search_entity_dense("proj", [1.0, 0.0], filter_=qfilter, limit=2)
    assert semantic_hits[0].point_id == "00000000-0000-0000-0000-000000000021"

    bm25_hits = await qdrant_store.search_entity_sparse(
        "proj",
        SparseVectorData(indices=[22], values=[1.0]),
        filter_=qfilter,
        limit=2,
    )
    assert [hit.point_id for hit in bm25_hits] == ["00000000-0000-0000-0000-000000000022"]

    hybrid_hits = await qdrant_store.search_entity_hybrid(
        "proj",
        [1.0, 0.0],
        SparseVectorData(indices=[21], values=[1.0]),
        filter_=qfilter,
        limit=2,
    )
    assert hybrid_hits[0].point_id == "00000000-0000-0000-0000-000000000021"


@pytest.mark.asyncio
async def test_qdrant_patch_memory_updates_payload_and_bm25_in_one_call(qdrant_store):
    memory_id = "00000000-0000-0000-0000-000000000010"
    await qdrant_store.upsert_memories([_point(memory_id, "old content", [1.0, 0.0], 1)])

    # Old BM25 term (index 1) matches; the new term (index 5) does not yet.
    qfilter = build_filter(must=[match_value("project_id", "proj")])
    assert await qdrant_store.search_memory_sparse(
        "proj", SparseVectorData(indices=[1], values=[1.0]), filter_=qfilter, limit=2
    )

    await qdrant_store.patch_memory(
        "proj",
        memory_id,
        {"content": "new content"},
        dense_vector=[0.0, 1.0],
        sparse_vector=SparseVectorData(indices=[5], values=[1.0]),
    )

    record = (await qdrant_store.get_memories("proj", [memory_id], with_vectors=True))[0]
    assert record.payload["content"] == "new content"
    assert record.vectors[qdrant_store.semantic_vector_name] == [0.0, 1.0]
    # The BM25 index now matches the new term and no longer the old one.
    new_hits = await qdrant_store.search_memory_sparse(
        "proj", SparseVectorData(indices=[5], values=[1.0]), filter_=qfilter, limit=2
    )
    assert [hit.point_id for hit in new_hits] == [memory_id]
    old_hits = await qdrant_store.search_memory_sparse(
        "proj", SparseVectorData(indices=[1], values=[1.0]), filter_=qfilter, limit=2
    )
    assert old_hits == []


@pytest.mark.asyncio
async def test_qdrant_filtering_is_caller_owned(qdrant_store):
    await qdrant_store.upsert_memories(
        [
            _point("00000000-0000-0000-0000-000000000003", "Scoped memory", [1.0, 0.0], 3, user_id="u1"),
        ],
    )

    qfilter = build_filter(
        must=[
            match_value("project_id", "proj"),
            match_value("user_id", "u2"),
            match_any("status", ["active"]),
        ]
    )
    hits = await qdrant_store.search_memory_dense("proj", [1.0, 0.0], filter_=qfilter, limit=2)

    assert hits == []


@pytest.mark.asyncio
async def test_qdrant_upserts_add_and_search_records(qdrant_store):
    await qdrant_store.upsert_add_record(
        AddRecordPoint(
            add_record_id="00000000-0000-0000-0000-000000000101",
            payload={
                "project_id": "proj",
                "request_id": "00000000-0000-0000-0000-000000000101",
                "status": "ok",
                "mode": "sync",
                "request_submitted_at": "2026-05-28T00:00:00+00:00",
                "task_completed_at": "2026-05-28T00:00:01+00:00",
            },
        )
    )
    await qdrant_store.upsert_search_record(
        SearchRecordPoint(
            search_record_id="00000000-0000-0000-0000-000000000102",
            payload={
                "project_id": "proj",
                "request_id": "00000000-0000-0000-0000-000000000102",
                "status": "ok",
                "query": "Qdrant",
                "request_submitted_at": "2026-05-28T00:00:00+00:00",
                "task_completed_at": "2026-05-28T00:00:01+00:00",
            },
        )
    )

    add_records = await qdrant_store.engine.retrieve(
        qdrant_store.add_record_collection,
        ["00000000-0000-0000-0000-000000000101"],
        with_vectors=False,
    )
    search_records = await qdrant_store.engine.retrieve(
        qdrant_store.search_record_collection,
        ["00000000-0000-0000-0000-000000000102"],
        with_vectors=False,
    )

    assert add_records[0].payload["request_id"] == "00000000-0000-0000-0000-000000000101"
    assert search_records[0].payload["query"] == "Qdrant"


@pytest.mark.asyncio
async def test_qdrant_scroll_add_records_uses_caller_order(qdrant_store):
    await qdrant_store.upsert_add_records(
        [
            AddRecordPoint(
                add_record_id="00000000-0000-0000-0000-000000000201",
                payload={
                    "project_id": "proj",
                    "request_id": "00000000-0000-0000-0000-000000000201",
                    "status": "queued",
                    "buffer_sequence": 20,
                },
            ),
            AddRecordPoint(
                add_record_id="00000000-0000-0000-0000-000000000202",
                payload={
                    "project_id": "proj",
                    "request_id": "00000000-0000-0000-0000-000000000202",
                    "status": "queued",
                    "buffer_sequence": 10,
                },
            ),
        ]
    )

    records, _ = await qdrant_store.scroll_add_records(
        "proj",
        order_by=qmodels.OrderBy(key="buffer_sequence", direction=qmodels.Direction.ASC),
    )
    global_records, _ = await qdrant_store.scroll_add_records_global(
        filter_=build_filter(must=[match_value("project_id", "proj")]),
        order_by=qmodels.OrderBy(key="buffer_sequence", direction=qmodels.Direction.ASC),
    )

    assert [record.payload["buffer_sequence"] for record in records] == [10, 20]
    assert [record.payload["buffer_sequence"] for record in global_records] == [10, 20]


@pytest.mark.asyncio
async def test_qdrant_scroll_search_records_combines_project_and_caller_filter(qdrant_store):
    await qdrant_store.upsert_search_records(
        [
            SearchRecordPoint(
                search_record_id="00000000-0000-0000-0000-000000000211",
                payload={
                    "project_id": "proj",
                    "request_id": "00000000-0000-0000-0000-000000000211",
                    "query": "Qdrant",
                    "status": "ok",
                },
            ),
            SearchRecordPoint(
                search_record_id="00000000-0000-0000-0000-000000000212",
                payload={
                    "project_id": "other-proj",
                    "request_id": "00000000-0000-0000-0000-000000000212",
                    "query": "Qdrant",
                    "status": "ok",
                },
            ),
            SearchRecordPoint(
                search_record_id="00000000-0000-0000-0000-000000000213",
                payload={
                    "project_id": "proj",
                    "request_id": "00000000-0000-0000-0000-000000000213",
                    "query": "Qdrant",
                    "status": "error",
                },
            ),
        ]
    )

    records, cursor = await qdrant_store.scroll_search_records(
        "proj",
        filter_=build_filter(must=[match_value("status", "ok")]),
        limit=10,
    )

    assert cursor is None
    assert [record.payload["request_id"] for record in records] == ["00000000-0000-0000-0000-000000000211"]


@pytest.mark.asyncio
async def test_qdrant_client_requests_are_limited_by_config():
    release = asyncio.Event()
    client = ConcurrencyTrackingQdrantClient(release)
    cfg = QdrantConfig(
        url="http://unused",
        memory_collection="test_memos",
        entity_collection="test_entities",
        source_collection="test_sources",
        add_record_collection="test_add_records",
        search_record_collection="test_search_records",
        vector_size=2,
        max_client_concurrency=3,
    )
    store = QdrantStore(cfg, client=client)

    tasks = [asyncio.create_task(store.get_memories("proj", [str(index)])) for index in range(10)]
    await _wait_for_active(client, 3)

    assert client.max_active == 3

    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_qdrant_client_concurrency_is_capped_by_config():
    release = asyncio.Event()
    client = ConcurrencyTrackingQdrantClient(release)
    cfg = QdrantConfig(
        url="http://unused",
        memory_collection="test_memos",
        entity_collection="test_entities",
        source_collection="test_sources",
        add_record_collection="test_add_records",
        search_record_collection="test_search_records",
        vector_size=2,
        max_client_concurrency=200,
        max_client_concurrency_cap=12,
    )
    store = QdrantStore(cfg, client=client)

    tasks = [asyncio.create_task(store.get_memories("proj", [str(index)])) for index in range(20)]
    await _wait_for_active(client, 12)

    assert client.max_active == 12

    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_qdrant_client_retries_retryable_errors():
    client = RetryThenSucceedQdrantClient(fail_count=2)
    cfg = QdrantConfig(
        url="http://unused",
        memory_collection="test_memos",
        entity_collection="test_entities",
        source_collection="test_sources",
        add_record_collection="test_add_records",
        search_record_collection="test_search_records",
        vector_size=2,
        max_retries=3,
        retry_base_delay=0.0,
    )
    store = QdrantStore(cfg, client=client)

    assert await store.get_memories("proj", ["mem"]) == []
    assert client.calls == 3


async def _wait_for_active(client, expected: int) -> None:
    for _ in range(100):
        if client.active == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {expected} active calls, got {client.active}")
