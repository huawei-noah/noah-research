from types import SimpleNamespace

import pytest
from mindmemos.config import init_config, reset_config
from mindmemos.infra.db import reset_database_clients
from mindmemos.infra.db.models import QdrantSearchRecord
from mindmemos.pipelines.memory_db.reader import MemoryDbReader
from mindmemos.typing.algo import SparseVector
from mindmemos.typing.memory import DatabaseRequestBudget, MemoryEdgeFilter, MemoryRequestContext
from mindmemos.typing.memory_db import MemoryDbSearchQuery


@pytest.fixture(autouse=True)
def memory_db_reader_config() -> None:
    init_config(config_path="config/mindmemos/dev.example.yaml")
    reset_database_clients()
    try:
        yield
    finally:
        reset_database_clients()
        reset_config()


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


def record_with_payload(payload: dict):
    return SimpleNamespace(payload=payload)


class FakeQdrant:
    def __init__(self, record=None):
        self.record = record
        self.updated_payloads = []
        self.upserted_memories = []
        self.deleted = []
        self.entity_searches = []
        self.entity_dense_hits = []
        self.memory_dense_searches = []
        self.memory_sparse_searches = []
        self.memory_hybrid_searches = []
        self.get_memory_calls = []
        self.get_memories_calls = []
        self.memory_records = {}
        self.entity_records = {}
        self.get_entity_calls = []

    @property
    def semantic_vector_name(self) -> str:
        return "semantic"

    @property
    def bm25_vector_name(self) -> str:
        return "bm25"

    async def get_memory(self, project_id: str, memory_id: str, *, with_vectors: bool = False):
        self.get_memory_calls.append((project_id, memory_id, with_vectors))
        return self.record

    async def get_memories(self, project_id: str, memory_ids: list[str], *, with_vectors: bool = False):
        self.get_memories_calls.append((project_id, list(memory_ids), with_vectors))
        return [self.memory_records[memory_id] for memory_id in memory_ids if memory_id in self.memory_records]

    async def get_entity(self, project_id: str, entity_id: str, *, with_vectors: bool = False):
        self.get_entity_calls.append((project_id, entity_id, with_vectors))
        return self.entity_records.get(entity_id)

    async def search_entity_dense(self, project_id, vector, *, filter_=None, limit=10, score_threshold=None):
        self.entity_searches.append((project_id, vector, filter_, limit, score_threshold))
        return self.entity_dense_hits

    async def search_memory_dense(self, project_id, vector, *, filter_=None, limit=10, score_threshold=None):
        self.memory_dense_searches.append((project_id, vector, filter_, limit, score_threshold))
        return []

    async def search_memory_sparse(self, project_id, vector, *, filter_=None, limit=10):
        self.memory_sparse_searches.append((project_id, vector, filter_, limit))
        return []

    async def search_memory_hybrid(self, project_id, dense_vector, sparse_vector, *, filter_=None, limit=10, dense_limit=None, sparse_limit=None):
        self.memory_hybrid_searches.append((project_id, dense_vector, sparse_vector, filter_, limit))
        return []


class FakeNeo4j:
    def __init__(self) -> None:
        self.related_calls = []
        self.read_calls = []
        self.read_rows = []
        self.neighbor_calls = []
        self.neighbor_rows = []
        self.lineage_calls = []
        self.lineage_rows = []
        self.related_rows = [
            {"memory_id": "mem-2", "seed_memory_id": "seed-a"},
            {"memory_id": "mem-1", "seed_memory_id": "seed-b"},
            {"memory_id": "mem-2", "seed_memory_id": "seed-c"},
            {"memory_id": ""},
            {},
        ]

    async def get_related_memory_ids(self, project_id, memory_ids, *, limit_per_memory=3, max_candidates=10):
        self.related_calls.append((project_id, memory_ids, limit_per_memory, max_candidates))
        return self.related_rows

    async def get_memory_lineage(self, project_id, memory_ids):
        self.lineage_calls.append((project_id, memory_ids))
        return self.lineage_rows

    async def run_read(self, query: str, **params):
        self.read_calls.append((query, params))
        return self.read_rows

    async def get_entity_neighbors(self, project_id, entity_id, *, direction="both", rel_type=None, limit=None):
        self.neighbor_calls.append((project_id, entity_id, direction, rel_type, limit))
        return self.neighbor_rows[:limit] if limit is not None else self.neighbor_rows


@pytest.mark.asyncio
async def test_search_entities_dense_uses_entity_filter_mapper():
    qdrant = FakeQdrant(record=None)
    reader = MemoryDbReader(clients=SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace()))

    result = await reader.search_entities_dense(
        make_context(),
        query="User",
        query_vector=[1.0, 0.0],
        filters=None,
        limit=3,
    )

    assert result.total == 0
    assert qdrant.entity_searches[0][0] == "proj-1"
    assert qdrant.entity_searches[0][2].must[0].key == "project_id"
    assert qdrant.entity_searches[0][3] == 23


@pytest.mark.asyncio
async def test_search_entities_dense_dedupes_search_field_points_to_canonical_entity():
    qdrant = FakeQdrant(record=None)
    qdrant.entity_dense_hits = [
        QdrantSearchRecord(
            point_id="entity-1-sf",
            score=0.95,
            source="entity_semantic",
            payload={
                "entity_id": "entity-1",
                "project_id": "proj-1",
                "entity_name": "Kai",
                "metadata": {
                    "is_search_field": True,
                    "search_field_content": "Kai likes Qdrant",
                    "search_field_index": 1,
                },
            },
        ),
        QdrantSearchRecord(
            point_id="entity-1",
            score=0.7,
            source="entity_semantic",
            payload={
                "entity_id": "entity-1",
                "project_id": "proj-1",
                "entity_name": "Kai",
                "metadata": {"search_fields": ["Kai likes Qdrant"]},
            },
        ),
        QdrantSearchRecord(
            point_id="entity-2",
            score=0.6,
            source="entity_semantic",
            payload={"entity_id": "entity-2", "project_id": "proj-1", "entity_name": "Lin", "metadata": {}},
        ),
    ]
    reader = MemoryDbReader(clients=SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace()))

    result = await reader.search_entities_dense(
        make_context(),
        query="Qdrant",
        query_vector=[1.0, 0.0],
        limit=2,
    )

    assert [hit.entity_id for hit in result.hits] == ["entity-1", "entity-2"]
    assert result.hits[0].entity is not None
    assert result.hits[0].entity.entity_id == "entity-1"
    assert result.hits[0].best_search_field == "Kai likes Qdrant"
    assert result.hits[0].best_search_field_index == 1
    assert result.hits[0].best_search_field_score == 0.95
    assert result.hits[0].matched_point_role == "search_field"


@pytest.mark.asyncio
async def test_search_entities_dense_uses_core_search_field_for_core_point():
    qdrant = FakeQdrant(record=None)
    qdrant.entity_dense_hits = [
        QdrantSearchRecord(
            point_id="entity-1",
            score=0.7,
            source="entity_semantic",
            payload={
                "entity_id": "entity-1",
                "project_id": "proj-1",
                "entity_name": "Kai",
                "metadata": {"core_search_field": "Kai person Qdrant preference"},
            },
        ),
    ]
    reader = MemoryDbReader(clients=SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace()))

    result = await reader.search_entities_dense(
        make_context(),
        query="Qdrant",
        query_vector=[1.0, 0.0],
        limit=1,
    )

    assert result.hits[0].best_search_field == "Kai person Qdrant preference"
    assert result.hits[0].best_search_field_score == 0.7
    assert result.hits[0].matched_point_role == "core"


@pytest.mark.asyncio
async def test_memory_vector_searches_inject_active_status_filter():
    qdrant = FakeQdrant(record=None)
    reader = MemoryDbReader(clients=SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace()))
    ctx = make_context()
    query = MemoryDbSearchQuery(query="q", top_k=3)

    await reader.search_dense(ctx, query, query_vector=[1.0, 0.0])
    await reader.search_sparse(ctx, query, indices=[1], values=[1.0])
    await reader.search_hybrid(
        ctx,
        query,
        dense_vector=[1.0, 0.0],
        sparse_vector=SparseVector(indices=[1], values=[1.0], model="test", hash_dim=128),
    )

    assert _filter_has_status_active(qdrant.memory_dense_searches[0][2])
    assert _filter_has_status_active(qdrant.memory_sparse_searches[0][2])
    assert _filter_has_status_active(qdrant.memory_hybrid_searches[0][3])


@pytest.mark.asyncio
async def test_get_related_memory_ids_delegates_to_neo4j_and_dedupes_order() -> None:
    neo4j = FakeNeo4j()
    reader = MemoryDbReader(clients=SimpleNamespace(qdrant=FakeQdrant(), neo4j=neo4j))

    result = await reader.get_related_memory_ids(
        make_context(),
        ["mem-1"],
        limit_per_memory=2,
        max_candidates=3,
    )

    assert result == [
        {"memory_id": "mem-2", "seed_memory_id": "seed-a"},
        {"memory_id": "mem-1", "seed_memory_id": "seed-b"},
    ]
    assert neo4j.related_calls == [("proj-1", ["mem-1"], 2, 3)]


@pytest.mark.asyncio
async def test_get_memory_lineage_returns_directed_derived_from_ids() -> None:
    neo4j = FakeNeo4j()
    neo4j.lineage_rows = [
        {"memory_id": "new", "derived_from_memory_ids": ["old", "old", ""]},
        {"memory_id": "leaf", "derived_from_memory_ids": []},
    ]
    reader = MemoryDbReader(clients=SimpleNamespace(qdrant=FakeQdrant(), neo4j=neo4j))

    result = await reader.get_memory_lineage(make_context(), ["new", "new", "leaf"])

    assert result == {"new": ["old"], "leaf": []}
    assert neo4j.lineage_calls == [("proj-1", ["new", "leaf"])]


@pytest.mark.asyncio
async def test_list_memories_by_shared_entities_uses_only_mentions_traversal() -> None:
    neo4j = FakeNeo4j()
    neo4j.read_rows = [
        {
            "seed_memory_id": "mem-1",
            "entity_id": "entity-1",
            "entity_name": "Kai",
            "entity_type": "person",
            "memory_ids": ["mem-1", "mem-2"],
        }
    ]
    reader = MemoryDbReader(clients=SimpleNamespace(qdrant=FakeQdrant(), neo4j=neo4j))

    scopes = await reader.list_memories_by_shared_entities(make_context(), ["mem-1"], limit_per_entity=5)

    query, params = neo4j.read_calls[0]
    assert "[:MENTIONS]" in query
    assert "RELATES_TO" not in query
    assert "created_at" not in query
    assert params["limit_per_entity"] == 5
    assert scopes[0].source == "shared_entity"
    assert scopes[0].memory_ids == ("mem-1", "mem-2")


@pytest.mark.asyncio
async def test_list_direct_related_memories_uses_typed_relation_whitelist_and_property_filters() -> None:
    neo4j = FakeNeo4j()
    neo4j.read_rows = [
        {
            "seed_memory_id": "mem-1",
            "memory_id": "mem-2",
            "rel_type": "RELATES_TO",
            "edge_type": "supports",
            "relation_type": "dreaming_evidence",
        }
    ]
    reader = MemoryDbReader(clients=SimpleNamespace(qdrant=FakeQdrant(), neo4j=neo4j))

    related = await reader.list_direct_related_memories(
        make_context(),
        ["mem-1"],
        edge_filter=MemoryEdgeFilter(edge_types=("supports",), relation_types=("dreaming_evidence",)),
        limit_per_memory=7,
        max_candidates=11,
    )

    query, params = neo4j.read_calls[0]
    assert "[r:RELATES_TO]" in query
    assert "r.edge_type IN $edge_types" in query
    assert "r.relation_type IN $relation_types" in query
    assert "created_at" not in query
    assert params["edge_types"] == ["supports"]
    assert params["relation_types"] == ["dreaming_evidence"]
    assert params["limit_per_memory"] == 7
    assert params["max_candidates"] == 11
    assert related[0].memory_id == "mem-2"
    assert related[0].rel_type == "RELATES_TO"


@pytest.mark.asyncio
async def test_get_entity_neighbors_applies_request_budget_to_neo4j_and_qdrant() -> None:
    qdrant = FakeQdrant()
    qdrant.entity_records = {
        "ent-1": _entity_record("ent-1", user_id="user-1"),
        "ent-2": _entity_record("ent-2", user_id="user-1"),
        "ent-3": _entity_record("ent-3", user_id="user-1"),
    }
    neo4j = FakeNeo4j()
    neo4j.neighbor_rows = [
        {"entity_id": "ent-1", "relation": "RELATES_TO", "direction": "out"},
        {"entity_id": "ent-2", "relation": "RELATES_TO", "direction": "in"},
        {"entity_id": "ent-3", "relation": "RELATES_TO", "direction": "out"},
    ]
    reader = MemoryDbReader(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))
    ctx = make_context().model_copy(update={"database_budget": DatabaseRequestBudget(qdrant_reads=2, neo4j_rows=5)})

    neighbors = await reader.get_entity_neighbors(ctx, "seed-ent", limit=4)

    assert [entity.entity_id for entity in neighbors] == ["ent-1", "ent-2"]
    assert neo4j.neighbor_calls == [("proj-1", "seed-ent", "both", None, 2)]
    assert [call[1] for call in qdrant.get_entity_calls] == ["ent-1", "ent-2"]
    assert ctx.database_budget is not None
    assert ctx.database_budget.qdrant_reads == 0
    assert ctx.database_budget.neo4j_rows == 3
    assert neighbors[0].metadata["_relation"] == "RELATES_TO"


@pytest.mark.asyncio
async def test_get_entity_neighbors_skips_db_when_request_budget_is_exhausted() -> None:
    qdrant = FakeQdrant()
    neo4j = FakeNeo4j()
    reader = MemoryDbReader(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))
    ctx = make_context().model_copy(update={"database_budget": DatabaseRequestBudget(qdrant_reads=0, neo4j_rows=10)})

    neighbors = await reader.get_entity_neighbors(ctx, "seed-ent")

    assert neighbors == []
    assert neo4j.neighbor_calls == []
    assert qdrant.get_entity_calls == []


def _filter_has_status_active(filter_) -> bool:
    clauses = list(getattr(filter_, "must", []) or [])
    for clause in clauses:
        if (
            getattr(clause, "key", None) == "status"
            and getattr(getattr(clause, "match", None), "value", None) == "active"
        ):
            return True
        if _filter_has_status_active(clause):
            return True
    return False


def _entity_record(entity_id: str, *, user_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        payload={
            "entity_id": entity_id,
            "project_id": "proj-1",
            "entity_name": entity_id,
            "entity_type": "person",
            "user_id": user_id,
            "metadata": {},
        }
    )
