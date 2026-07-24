import asyncio
import json
from types import SimpleNamespace

import pytest
from mindmemos.config import Neo4jConfig
from mindmemos.infra.db import EntityNode, GraphRelationship, MemoryNode, NodeRef, SourceNode
from mindmemos.infra.db.neo4j import Neo4jStore
from neo4j.exceptions import TransientError


class FakeAsyncDriver:
    def __init__(self):
        self.calls = []
        self.records = []

    async def execute_query(self, query, params=None, **kwargs):
        self.calls.append((query, params or {}, kwargs))
        return SimpleNamespace(records=[_FakeRecord(record) for record in self.records])

    async def close(self):
        return None


class _FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class ConcurrencyTrackingNeo4jDriver:
    def __init__(self, release: asyncio.Event):
        self.release = release
        self.active = 0
        self.max_active = 0

    async def execute_query(self, query, params=None, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self.release.wait()
            return _FakeExecuteQueryResult()
        finally:
            self.active -= 1

    async def close(self):
        return None


class _FakeExecuteQueryResult:
    records = []


@pytest.mark.asyncio
async def test_upsert_memory_node_uses_documented_key():
    driver = FakeAsyncDriver()
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    await store.upsert_memory_node(MemoryNode(project_id="proj", memory_id="mem", content="User likes tea."))

    query, params, _ = driver.calls[0]
    assert "UNWIND $rows AS row" in query
    assert "MERGE (n:`Memory`" in query
    assert params["rows"][0]["project_id"] == "proj"
    assert params["rows"][0]["memory_id"] == "mem"
    assert params["rows"][0]["properties"]["content"] == "User likes tea."


@pytest.mark.asyncio
async def test_upsert_memory_node_writes_active_status():
    """Memory upsert must persist status='active'.

    The graph-expansion Cypher (shared_entity / direct_related) filters on
    ``Memory.status``. If the ``status`` property key is never written to any
    node, Neo4j emits ``UnknownPropertyKeyWarning`` on every such query.
    Writing ``status='active'`` at creation (mirroring ``archive_memory_node``
    which writes ``'archived'``) registers the key and makes the filter real.
    """

    driver = FakeAsyncDriver()
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    await store.upsert_memory_node(MemoryNode(project_id="proj", memory_id="mem", content="User likes tea."))

    _, params, _ = driver.calls[0]
    assert params["rows"][0]["properties"]["status"] == "active"


@pytest.mark.asyncio
async def test_upsert_nodes_batches_by_label():
    driver = FakeAsyncDriver()
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    await store.upsert_nodes(
        memories=[
            MemoryNode(project_id="proj", memory_id="mem-1", content="User likes tea."),
            MemoryNode(project_id="proj", memory_id="mem-2", content="User likes coffee."),
        ],
        entities=[
            EntityNode(
                project_id="proj",
                entity_id="ent-1",
                entity_name="Tea",
                entity_type="preference",
                description=None,
            )
        ],
        sources=[SourceNode(project_id="proj", source_id="src-1", parsed_content_path=None)],
    )

    assert len(driver.calls) == 3
    memory_query, memory_params, _ = driver.calls[0]
    entity_query, entity_params, _ = driver.calls[1]
    source_query, source_params, _ = driver.calls[2]

    assert "UNWIND $rows AS row" in memory_query
    assert "MERGE (n:`Memory` {`project_id`: row.project_id, `memory_id`: row.memory_id})" in memory_query
    assert len(memory_params["rows"]) == 2
    assert memory_params["rows"][1]["properties"]["content"] == "User likes coffee."
    assert "MERGE (n:`Entity` {`project_id`: row.project_id, `entity_id`: row.entity_id})" in entity_query
    assert entity_params["rows"][0]["properties"] == {"entity_name": "Tea", "entity_type": "preference"}
    assert "MERGE (n:`Source` {`project_id`: row.project_id, `source_id`: row.source_id})" in source_query
    assert source_params["rows"][0]["properties"] == {}


@pytest.mark.asyncio
async def test_archive_memory_node_marks_status_without_deleting_node():
    driver = FakeAsyncDriver()
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    await store.archive_memory_node("proj", "mem", reason="user_request")

    query, params, _ = driver.calls[0]
    assert "MERGE (n:`Memory`" in query
    assert "DELETE" not in query
    assert params["key_project_id"] == "proj"
    assert params["key_memory_id"] == "mem"
    assert params["properties"]["status"] == "archived"
    assert params["properties"]["delete_reason"] == "user_request"
    assert "status_changed_at" in params["properties"]


@pytest.mark.asyncio
async def test_upsert_relationship_uses_source_target_refs_and_serializes_metadata():
    driver = FakeAsyncDriver()
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    await store.upsert_relationship(
        GraphRelationship(
            source=NodeRef(label="Memory", key={"project_id": "proj", "memory_id": "mem"}),
            target=NodeRef(label="Entity", key={"project_id": "proj", "entity_id": "ent"}),
            rel_type="MENTIONS",
            key={"project_id": "proj"},
            properties={"metadata": {"confidence": 0.9}},
        )
    )

    query, params, _ = driver.calls[0]
    assert "UNWIND $rows AS row" in query
    assert "MATCH (source:`Memory`" in query
    assert "MERGE (source)-[r:`MENTIONS`" in query
    assert params["rows"][0]["rel"]["project_id"] == "proj"
    assert json.loads(params["rows"][0]["properties"]["metadata_json"]) == {"confidence": 0.9}


@pytest.mark.asyncio
async def test_upsert_relationships_batches_same_shape_relationships():
    driver = FakeAsyncDriver()
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    await store.upsert_relationships(
        [
            GraphRelationship(
                source=NodeRef(label="Memory", key={"project_id": "proj", "memory_id": "mem-1"}),
                target=NodeRef(label="Entity", key={"project_id": "proj", "entity_id": "ent-1"}),
                rel_type="MENTIONS",
                key={"project_id": "proj"},
                properties={"mention_count": 1},
            ),
            GraphRelationship(
                source=NodeRef(label="Memory", key={"project_id": "proj", "memory_id": "mem-2"}),
                target=NodeRef(label="Entity", key={"project_id": "proj", "entity_id": "ent-2"}),
                rel_type="MENTIONS",
                key={"project_id": "proj"},
                properties={"mention_count": 2},
            ),
        ]
    )

    assert len(driver.calls) == 1
    query, params, _ = driver.calls[0]
    assert "UNWIND $rows AS row" in query
    assert "MATCH (source:`Memory` {`project_id`: row.source.project_id, `memory_id`: row.source.memory_id})" in query
    assert "MATCH (target:`Entity` {`project_id`: row.target.project_id, `entity_id`: row.target.entity_id})" in query
    assert "MERGE (source)-[r:`MENTIONS` {`project_id`: row.rel.project_id}]->(target)" in query
    assert len(params["rows"]) == 2
    assert params["rows"][1]["source"]["memory_id"] == "mem-2"
    assert params["rows"][1]["properties"]["mention_count"] == 2


@pytest.mark.asyncio
async def test_upsert_relationships_splits_different_relationship_shapes():
    driver = FakeAsyncDriver()
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    await store.upsert_relationships(
        [
            GraphRelationship(
                source=NodeRef(label="Memory", key={"project_id": "proj", "memory_id": "mem"}),
                target=NodeRef(label="Entity", key={"project_id": "proj", "entity_id": "ent"}),
                rel_type="MENTIONS",
                key={"project_id": "proj"},
            ),
            GraphRelationship(
                source=NodeRef(label="Memory", key={"project_id": "proj", "memory_id": "mem"}),
                target=NodeRef(label="Source", key={"project_id": "proj", "source_id": "src"}),
                rel_type="EXTRACTED_FROM",
                key={"project_id": "proj"},
            ),
        ]
    )

    assert len(driver.calls) == 2


@pytest.mark.asyncio
async def test_get_related_memory_ids_queries_one_hop_relates_to_neighbors():
    driver = FakeAsyncDriver()
    driver.records = [{"memory_id": "mem-3", "seed_memory_id": "mem-1", "relation": "RELATES_TO"}]
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    rows = await store.get_related_memory_ids(
        "proj",
        ["mem-1", "mem-2"],
        limit_per_memory=3,
        max_candidates=5,
    )

    query, params, kwargs = driver.calls[0]
    assert (
        "MATCH (m:`Memory` {project_id: $project_id})-[r:`RELATES_TO`]-(n:`Memory` {project_id: $project_id})" in query
    )
    assert "m.memory_id IN $memory_ids" in query
    assert "n.memory_id <> m.memory_id" in query
    assert "LIMIT $max_candidates" in query
    assert params == {
        "project_id": "proj",
        "memory_ids": ["mem-1", "mem-2"],
        "limit_per_memory": 3,
        "max_candidates": 5,
    }
    assert kwargs["routing_"].name == "READ"
    assert rows == [{"memory_id": "mem-3", "seed_memory_id": "mem-1", "relation": "RELATES_TO"}]


@pytest.mark.asyncio
async def test_get_memory_lineage_does_not_reference_unmirrored_time_properties():
    driver = FakeAsyncDriver()
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    await store.get_memory_lineage("proj", ["mem-1"])

    query, params, kwargs = driver.calls[0]
    assert "created_at" not in query
    assert "update_at" not in query
    assert params == {"project_id": "proj", "memory_ids": ["mem-1"]}
    assert kwargs["routing_"].name == "READ"


@pytest.mark.asyncio
async def test_get_entity_neighbors_uses_limit_when_provided():
    driver = FakeAsyncDriver()
    driver.records = [{"entity_id": "ent-2", "entity_name": "Kai", "entity_type": "person", "relation": "KNOWS"}]
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    rows = await store.get_entity_neighbors("proj", "ent-1", rel_type="KNOWS", limit=7)

    query, params, kwargs = driver.calls[0]
    assert "MATCH (e)-[r]-(n)" in query
    assert "AND n.`project_id` = $project_id" in query
    assert "AND type(r) = $rel_type" in query
    assert "LIMIT $limit" in query
    assert params == {"project_id": "proj", "entity_id": "ent-1", "rel_type": "KNOWS", "limit": 7}
    assert kwargs["routing_"].name == "READ"
    assert rows == [{"entity_id": "ent-2", "entity_name": "Kai", "entity_type": "person", "relation": "KNOWS"}]


@pytest.mark.asyncio
async def test_get_entity_neighbors_returns_empty_without_query_for_non_positive_limit():
    driver = FakeAsyncDriver()
    store = Neo4jStore(Neo4jConfig(uri="bolt://unused"), driver=driver)

    rows = await store.get_entity_neighbors("proj", "ent-1", limit=0)

    assert rows == []
    assert driver.calls == []


class TransientFakeDriver:
    """Driver that raises TransientError for the first N calls, then succeeds."""

    def __init__(self, fail_count: int):
        self._fail_count = fail_count
        self.attempt_count = 0

    async def execute_query(self, query, params=None, **kwargs):
        self.attempt_count += 1
        if self.attempt_count <= self._fail_count:
            raise TransientError("deadlock detected")
        return _FakeExecuteQueryResult()

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_internal_run_write_retries_on_transient_error():
    driver = TransientFakeDriver(fail_count=2)
    cfg = Neo4jConfig(uri="bolt://unused", write_max_retries=3, write_retry_base_delay=0.0)
    store = Neo4jStore(cfg, driver=driver)

    await store._run_write("MERGE (n:Test {id: $id})", id="t1")  # noqa: SLF001

    assert driver.attempt_count == 3


@pytest.mark.asyncio
async def test_run_read_retries_on_transient_error():
    driver = TransientFakeDriver(fail_count=2)
    cfg = Neo4jConfig(uri="bolt://unused", read_max_retries=3, read_retry_base_delay=0.0)
    store = Neo4jStore(cfg, driver=driver)

    rows = await store.run_read("MATCH (n) RETURN n")

    assert rows == []
    assert driver.attempt_count == 3


@pytest.mark.asyncio
async def test_internal_run_write_raises_after_max_retries_exhausted():
    driver = TransientFakeDriver(fail_count=5)
    cfg = Neo4jConfig(uri="bolt://unused", write_max_retries=3, write_retry_base_delay=0.0)
    store = Neo4jStore(cfg, driver=driver)

    with pytest.raises(TransientError):
        await store._run_write("MERGE (n:Test {id: $id})", id="t1")  # noqa: SLF001

    assert driver.attempt_count == 3


@pytest.mark.asyncio
async def test_internal_run_write_treats_zero_max_retries_as_one_attempt():
    driver = TransientFakeDriver(fail_count=0)
    cfg = Neo4jConfig(uri="bolt://unused", write_max_retries=0, write_retry_base_delay=0.0)
    store = Neo4jStore(cfg, driver=driver)

    await store._run_write("MERGE (n:Test {id: $id})", id="t1")  # noqa: SLF001

    assert driver.attempt_count == 1


@pytest.mark.asyncio
async def test_neo4j_client_requests_are_limited_by_config():
    release = asyncio.Event()
    driver = ConcurrencyTrackingNeo4jDriver(release)
    cfg = Neo4jConfig(uri="bolt://unused", max_client_concurrency=3)
    store = Neo4jStore(cfg, driver=driver)

    tasks = [asyncio.create_task(store.run_read("MATCH (n) RETURN n")) for _ in range(10)]
    await _wait_for_active(driver, 3)

    assert driver.max_active == 3

    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_neo4j_client_concurrency_is_capped_by_config():
    release = asyncio.Event()
    driver = ConcurrencyTrackingNeo4jDriver(release)
    cfg = Neo4jConfig(uri="bolt://unused", max_client_concurrency=200, max_client_concurrency_cap=12)
    store = Neo4jStore(cfg, driver=driver)

    tasks = [asyncio.create_task(store.run_read("MATCH (n) RETURN n")) for _ in range(20)]
    await _wait_for_active(driver, 12)

    assert driver.max_active == 12

    release.set()
    await asyncio.gather(*tasks)


async def _wait_for_active(client, expected: int) -> None:
    for _ in range(100):
        if client.active == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {expected} active calls, got {client.active}")
