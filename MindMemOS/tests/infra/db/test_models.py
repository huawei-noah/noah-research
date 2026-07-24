import pytest

from mindmemos.infra.db import (
    NEO4J_SCHEMA_STATEMENTS,
    GraphRelationship,
    MemoryNode,
    MemoryPoint,
    NodeRef,
    SparseVectorData,
)


def test_sparse_vector_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        SparseVectorData(indices=[1, 2], values=[1.0])


def test_memory_point_is_payload_and_vector_primitive():
    point = MemoryPoint(
        memory_id="00000000-0000-0000-0000-000000000001",
        semantic_vector=[0.1, 0.2],
        bm25_vector=SparseVectorData(indices=[1], values=[1.0]),
        payload={"project_id": "proj", "content": "Project uses Qdrant."},
    )

    assert point.payload["project_id"] == "proj"
    assert point.semantic_vector == [0.1, 0.2]


def test_neo4j_primitives_use_documented_node_refs():
    memory = MemoryNode(project_id="proj", memory_id="mem-1", content="User likes tea.")
    entity = NodeRef(label="Entity", key={"project_id": "proj", "entity_id": "ent-1"})

    relationship = GraphRelationship(
        source=memory.ref,
        target=entity,
        rel_type="MENTIONS",
        key={"project_id": "proj"},
        properties={"confidence": 0.9},
    )

    assert memory.ref.key["memory_id"] == "mem-1"
    assert relationship.rel_type == "MENTIONS"


def test_neo4j_schema_uses_community_supported_uniqueness_constraints():
    schema = "\n".join(NEO4J_SCHEMA_STATEMENTS)

    assert "IS NODE KEY" not in schema
    assert "REQUIRE (m.project_id, m.memory_id) IS UNIQUE" in schema
    assert "REQUIRE (e.project_id, e.entity_id) IS UNIQUE" in schema
    assert "REQUIRE (s.project_id, s.source_id) IS UNIQUE" in schema
