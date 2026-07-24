from datetime import UTC, datetime

import pytest
from mindmemos.errors import InvalidFilterError
from mindmemos.infra import db
from mindmemos.mappers import (
    MappingError,
    ProjectIsolationError,
    parse_search_dsl,
    search_filter_to_qdrant,
    to_add_record_point,
    to_db_write_primitives,
    to_memory_point,
    to_search_hit,
    to_search_record_point,
)
from mindmemos.typing.memory import (
    DialogueMessage,
    EntityVectorWrite,
    EntityWrite,
    FieldCondition,
    GraphNodeRef,
    GraphRelationship,
    MemoryRequestContext,
    MemoryWrite,
    SearchFilter,
    SourceWrite,
    VectorWrite,
    build_tenant_conditions,
)
from mindmemos.typing.memory_db import MemoryDbWritePlan
from mindmemos.typing.service import (
    AddPipelineInput,
    AddPipelineSyncResult,
    DeletePipelineInput,
    GetPipelineInput,
    MemoryAddEventItem,
    MemorySearchItem,
    SearchPipelineInput,
    SearchPipelineResult,
    UpdatePipelineInput,
)


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        agent_id="agent-1",
        session_id="session-1",
    )


def make_memory_write(*, project_id: str = "proj-1") -> MemoryWrite:
    return MemoryWrite(
        memory_id="mem-1",
        account_id="acc-1",
        project_id=project_id,
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
        agent_id="agent-1",
        content="Project uses Qdrant.",
        mem_type="fact",
        mem_extract_type="vanilla",
        mem_extract_version="test_v1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        root_id=["mem-1"],
        property_name="tooling",
        entity_id="ent-1",
        entity_type="database",
        metadata={"quality_score": 0.8},
    )


def test_request_context_and_service_add_input_use_shared_contracts() -> None:
    ctx = make_context()
    add_input = AddPipelineInput(
        messages=[DialogueMessage(role="user", content="Remember Qdrant.")],
        timestamp=1770000000000,
        metadata={"trace": "test"},
    )

    assert ctx.project_id == "proj-1"
    assert ctx.api_key_uuid == "key-1"
    assert add_input.messages[0].content == "Remember Qdrant."
    assert add_input.metadata == {"trace": "test"}


def test_service_add_input_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        AddPipelineInput(
            messages=[DialogueMessage(role="user", content="Remember Qdrant.")],
            dryrun=True,
        )


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("infer", True),
        ("consistency", "fast"),
        ("allowed_memory_types", ["fact"]),
        ("sources", [{"source_type": "message", "message_id": "msg-1"}]),
    ],
)
def test_service_add_input_rejects_removed_add_options(field_name: str, value: object) -> None:
    with pytest.raises(ValueError):
        AddPipelineInput(
            messages=[DialogueMessage(role="user", content="Remember Qdrant.")],
            **{field_name: value},
        )


def test_memory_id_aliases_keep_delete_update_api_compatible() -> None:
    assert DeletePipelineInput(memory_id="mem-3").id == "mem-3"
    assert DeletePipelineInput(id="mem-4").id == "mem-4"
    assert UpdatePipelineInput(memory_id="mem-5", content="updated").id == "mem-5"
    assert UpdatePipelineInput(id="mem-6", content="updated").id == "mem-6"


def test_get_pipeline_input_takes_filters_not_memory_id() -> None:
    inp = GetPipelineInput(filters={"mem_type": "fact"}, top_k=5)
    assert inp.filters == {"mem_type": "fact"}
    assert inp.top_k == 5
    assert GetPipelineInput().top_k is None
    with pytest.raises(ValueError):
        GetPipelineInput(memory_id="mem-1")


def test_search_filter_maps_to_qdrant_with_project_isolation_and_nested_clauses() -> None:
    ctx = make_context()

    qfilter = search_filter_to_qdrant(
        ctx,
        SearchFilter(
            must=[FieldCondition(field="user_id", op="match", value="user-1")],
            should=[
                SearchFilter(
                    must=[FieldCondition(field="content", op="text", value="Qdrant")],
                    must_not=[FieldCondition(field="status", op="except", values=["archived"])],
                )
            ],
        ),
    )

    assert qfilter.must is not None
    assert qfilter.must[0].key == "project_id"
    assert qfilter.must[1].key == "user_id"
    assert qfilter.should is not None
    assert qfilter.should[0].must[0].key == "content"
    assert qfilter.should[0].must_not[0].key == "status"


def test_search_dsl_rejects_unfilterable_fields() -> None:
    with pytest.raises(InvalidFilterError):
        parse_search_dsl({"project_id": "other"})


def test_search_dsl_maps_scalar_and_operators_to_internal_filter() -> None:
    internal_filter = parse_search_dsl(
        {
            "mem_type": "fact",
            "entity_type": {"in": ["database", "service"]},
            "created_at": {"gte": "2026-01-01T00:00:00+00:00"},
            "content": {"icontains": "Qdrant"},
        }
    )

    mem_type = internal_filter.must[0]
    assert isinstance(mem_type, FieldCondition)
    assert mem_type.field == "mem_type"
    assert mem_type.op == "match"
    assert mem_type.value == "fact"

    entity = internal_filter.must[1]
    assert isinstance(entity, FieldCondition)
    assert entity.op == "any"
    assert entity.values == ["database", "service"]

    created = internal_filter.must[2]
    assert isinstance(created, FieldCondition)
    assert created.op == "datetime"
    assert created.gte == datetime(2026, 1, 1, tzinfo=UTC)

    content = internal_filter.must[3]
    assert isinstance(content, FieldCondition)
    assert content.op == "text"
    assert content.value == "Qdrant"


def test_entity_search_filter_uses_entity_field_allowlist() -> None:
    ctx = make_context()

    qfilter = search_filter_to_qdrant(
        ctx,
        SearchFilter(must=[FieldCondition(field="entity_type", op="match", value="user")]),
        target="entity",
    )

    assert qfilter.must[0].key == "project_id"
    assert qfilter.must[1].key == "entity_type"
    with pytest.raises(MappingError):
        search_filter_to_qdrant(
            ctx,
            SearchFilter(must=[FieldCondition(field="content", op="text", value="Qdrant")]),
            target="entity",
        )


def test_search_dsl_wildcard_and_ne_and_logical() -> None:
    internal_filter = parse_search_dsl(
        {
            "entity_id": "*",
            "AND": [{"mem_type": {"ne": "fact"}}],
            "OR": [{"property_name": "tooling"}],
        }
    )

    wildcard = internal_filter.must[0]
    assert isinstance(wildcard, SearchFilter)
    assert wildcard.must_not[0].op == "is_empty"
    assert wildcard.must_not[0].field == "entity_id"

    ne_clause = internal_filter.must[1]
    assert isinstance(ne_clause, SearchFilter)
    assert ne_clause.must[0].must_not[0].op == "match"
    assert ne_clause.must[0].must_not[0].value == "fact"

    assert internal_filter.should[0].must[0].field == "property_name"


def test_search_dsl_rejects_unknown_operator_and_text_on_keyword() -> None:
    with pytest.raises(InvalidFilterError):
        parse_search_dsl({"mem_type": {"bogus": 1}})
    with pytest.raises(InvalidFilterError):
        parse_search_dsl({"mem_type": {"contains": "fa"}})
    with pytest.raises(InvalidFilterError):
        parse_search_dsl({"AND": {"mem_type": "fact"}})


def test_memory_write_maps_to_qdrant_point_and_graph_relationships() -> None:
    ctx = make_context()
    write = make_memory_write()
    vector = VectorWrite(
        memory_id="mem-1", semantic_vector=[0.1, 0.2, 0.3], bm25_indices=[1, 8], bm25_values=[0.7, 1.2]
    )
    relationship = GraphRelationship(
        source=GraphNodeRef(kind="Memory", project_id="proj-1", node_id="mem-1"),
        target=GraphNodeRef(kind="Entity", project_id="proj-1", node_id="ent-1"),
        rel_type="MENTIONS",
        project_id="proj-1",
        metadata={"confidence": 0.8},
    )

    point = to_memory_point(write, vector=vector, ctx=ctx)
    memory_points, _, _, relationships = to_db_write_primitives(
        MemoryDbWritePlan(memories=[write], vectors=[vector], relationships=[relationship]),
        ctx=ctx,
    )

    assert point.payload["mem_type"] == "fact"
    assert point.payload["project_id"] == "proj-1"
    assert point.bm25_vector is not None
    assert point.bm25_vector.indices == [1, 8]
    assert memory_points[0].semantic_vector == [0.1, 0.2, 0.3]
    assert relationships[0].rel_type == "MENTIONS"
    assert relationships[0].key == {"project_id": "proj-1"}


def test_entity_write_maps_to_qdrant_point_with_vector() -> None:
    ctx = make_context()
    entity = EntityWrite(
        entity_id="ent-1",
        account_id="acct-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="sess-1",
        entity_name="Qdrant",
        entity_type="organization",
        description="Vector database.",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    vector = EntityVectorWrite(entity_id="ent-1", semantic_vector=[0.3, 0.2, 0.1])

    _, entity_points, _, _ = to_db_write_primitives(
        MemoryDbWritePlan(entities=[entity], entity_vectors=[vector]),
        ctx=ctx,
    )

    assert entity_points[0].entity_id == "ent-1"
    assert entity_points[0].payload["project_id"] == "proj-1"
    assert entity_points[0].vector == [0.3, 0.2, 0.1]


def test_search_field_entity_point_allows_sparse_only_vector() -> None:
    ctx = make_context()
    entity = EntityWrite(
        entity_id="ent-1",
        account_id="acct-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="sess-1",
        entity_name="Qdrant",
        entity_type="organization",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"search_fields": ["Qdrant vector search"]},
    )
    vectors = [
        EntityVectorWrite(entity_id="ent-1", bm25_indices=[1], bm25_values=[1.0]),
        EntityVectorWrite(entity_id="ent-1#sf0", bm25_indices=[2], bm25_values=[2.0]),
    ]

    _, entity_points, _, _ = to_db_write_primitives(
        MemoryDbWritePlan(entities=[entity], entity_vectors=vectors),
        ctx=ctx,
    )

    search_field_points = [point for point in entity_points if point.payload["metadata"].get("is_search_field")]
    assert len(search_field_points) == 1
    assert search_field_points[0].vector is None
    assert search_field_points[0].bm25_vector is not None
    assert search_field_points[0].payload["entity_id"] == "ent-1"
    assert search_field_points[0].payload["metadata"]["search_field_content"] == "Qdrant vector search"


def _make_source_write(source_id: str, source_type: str, *, persist_payload: bool) -> SourceWrite:
    return SourceWrite(
        source_id=source_id,
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        source_type=source_type,
        file_path=f"/tmp/{source_id}",
        file_name=source_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        persist_payload=persist_payload,
    )


def test_to_db_write_primitives_excludes_non_persisted_sources() -> None:
    """Only persist_payload=True sources become source_item_v1 Qdrant points.

    message sources carry persist_payload=False so they stay in plan.sources as
    Neo4j Source graph nodes (provenance edge endpoints) without being written to
    source_item_v1. The routing flag itself must not leak into the Qdrant payload.
    """
    ctx = make_context()
    message_source = _make_source_write("src-msg", "message", persist_payload=False)
    file_source = _make_source_write("src-file", "file", persist_payload=True)

    _, _, source_points, _ = to_db_write_primitives(
        MemoryDbWritePlan(sources=[message_source, file_source]),
        ctx=ctx,
    )

    assert [point.payload["source_id"] for point in source_points] == ["src-file"]
    assert "persist_payload" not in source_points[0].payload


def test_cross_project_memory_write_is_rejected() -> None:
    ctx = make_context()

    with pytest.raises(ProjectIsolationError):
        to_memory_point(make_memory_write(project_id="other"), ctx=ctx)


def test_service_search_input_uses_dsl_filter_contract() -> None:
    search_input = SearchPipelineInput(
        query="Qdrant",
        filters={"entity_type": "database"},
        top_k=5,
    )

    internal_filter = parse_search_dsl(search_input.filters)

    assert internal_filter.must[0].field == "entity_type"
    assert search_input.top_k == 5


@pytest.mark.parametrize("top_k", [0, -1])
def test_service_search_input_rejects_out_of_bounds_top_k(top_k: int) -> None:
    with pytest.raises(ValueError):
        SearchPipelineInput(query="Qdrant", top_k=top_k)


def test_service_search_input_allows_positive_top_k_and_none() -> None:
    assert SearchPipelineInput(query="Qdrant", top_k=101).top_k == 101
    assert SearchPipelineInput(query="Qdrant", top_k=None).top_k is None


def test_service_search_input_rejects_legacy_search_strategy_key() -> None:
    with pytest.raises(ValueError):
        SearchPipelineInput.model_validate({"query": "Qdrant", "search_strategy": "schema"})


def test_add_record_mapper_uses_protocol_fields_only() -> None:
    ctx = make_context()
    submitted_at = datetime(2026, 5, 28, tzinfo=UTC)
    completed_at = datetime(2026, 5, 28, 0, 0, 1, tzinfo=UTC)

    point = to_add_record_point(
        AddPipelineInput(
            messages=[DialogueMessage(role="user", content="Remember Qdrant.")],
            timestamp=1770000000000,
            metadata={"source": "test"},
        ),
        AddPipelineSyncResult(
            status="ok",
            memories=[MemoryAddEventItem(operation="add", content="User wants Qdrant remembered.")],
        ),
        ctx=ctx,
        request_submitted_at=submitted_at,
        task_completed_at=completed_at,
        add_record_id="add-rec-1",
    )

    assert point.add_record_id == "add-rec-1"
    assert point.payload["project_id"] == "proj-1"
    assert "timestamp" not in point.payload
    assert "sources" not in point.payload
    assert "force_generation" not in point.payload
    assert point.payload["event_timestamp_ms"] == 1770000000000
    assert point.payload["event_time"] == datetime.fromtimestamp(1770000000000 / 1000, tz=UTC)
    assert point.payload["messages"][0]["content"] == "Remember Qdrant."
    assert point.payload["metadata"] == {"source": "test"}
    assert point.payload["status"] == "ok"
    assert point.payload["memories"][0]["operation"] == "add"
    assert "bm25_hit_count" not in point.payload
    assert "query_hash" not in point.payload


def test_add_record_mapper_preserves_add_input_timestamp() -> None:
    ctx = make_context()
    submitted_at = datetime(2026, 5, 28, tzinfo=UTC)
    completed_at = datetime(2026, 5, 28, 0, 0, 1, tzinfo=UTC)
    add_input = AddPipelineInput(messages=[{"text": "Remember default timestamp."}], timestamp=1770000000000)

    point = to_add_record_point(
        add_input,
        AddPipelineSyncResult(status="ok", memories=[]),
        ctx=ctx,
        request_submitted_at=submitted_at,
        task_completed_at=completed_at,
    )

    assert point.payload["event_timestamp_ms"] == add_input.timestamp
    assert "timestamp" not in point.payload
    assert "timestamp" not in point.payload["messages"][0]


def test_add_record_mapper_uses_explicit_request_timestamp_before_message_timestamp() -> None:
    ctx = make_context()
    submitted_at = datetime(2026, 5, 28, tzinfo=UTC)
    completed_at = datetime(2026, 5, 28, 0, 0, 1, tzinfo=UTC)
    add_input = AddPipelineInput(
        messages=[{"role": "user", "content": "Remember message time.", "timestamp": 1770000000000}],
        timestamp=1700000000000,
    )

    point = to_add_record_point(
        add_input,
        AddPipelineSyncResult(status="ok", memories=[]),
        ctx=ctx,
        request_submitted_at=submitted_at,
        task_completed_at=completed_at,
    )

    assert point.payload["event_timestamp_ms"] == 1700000000000
    assert "timestamp" not in point.payload


def test_search_record_mapper_uses_protocol_fields_only() -> None:
    ctx = make_context()
    submitted_at = datetime(2026, 5, 28, tzinfo=UTC)
    completed_at = datetime(2026, 5, 28, 0, 0, 1, tzinfo=UTC)

    point = to_search_record_point(
        SearchPipelineInput(
            query="Qdrant",
            filters={"entity_type": "database"},
            top_k=3,
            search_pipeline="schema",
            agentic=True,
            max_rounds=2,
            rerank=True,
        ),
        SearchPipelineResult(
            status="ok",
            memories=[
                MemorySearchItem(
                    id="mem-1",
                    memory="Project uses Qdrant.",
                    last_update_at="2026-05-28 00:00:00",
                )
            ],
        ),
        ctx=ctx,
        request_submitted_at=submitted_at,
        task_completed_at=completed_at,
        search_record_id="search-rec-1",
    )

    assert point.search_record_id == "search-rec-1"
    assert point.payload["query"] == "Qdrant"
    assert point.payload["filters"] == {"entity_type": "database"}
    assert point.payload["search_pipeline"] == "schema"
    assert point.payload["agentic"] is True
    assert point.payload["max_rounds"] == 2
    assert point.payload["rerank"] is True
    assert point.payload["memories"][0]["id"] == "mem-1"
    assert "semantic_hit_count" not in point.payload
    assert "returned_memory_ids" not in point.payload


def test_qdrant_hit_maps_to_business_search_hit() -> None:
    payload = {
        "memory_id": "mem-1",
        "project_id": "proj-1",
        "user_id": "user-1",
        "mem_type": "fact",
        "content": "Project uses Qdrant.",
        "status": "active",
        "metadata": {"quality_score": "0.9", "pinned": True},
        "validate_from": "2023-11-14T22:13:20+00:00",
        "validate_to": "2023-11-15T22:13:20+00:00",
    }
    hit = db.QdrantSearchRecord(point_id="mem-1", score=0.8, source="bm25", payload=payload, debug={"rank": 2})

    mapped = to_search_hit(hit)

    assert mapped.memory is not None
    assert mapped.memory.project_id == "proj-1"
    assert mapped.memory.metadata["quality_score"] == "0.9"
    assert mapped.memory.validate_from == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
    assert mapped.memory.validate_to == datetime(2023, 11, 15, 22, 13, 20, tzinfo=UTC)
    assert mapped.source == "bm25"
    assert mapped.rank == 2


def _ctx(**overrides) -> MemoryRequestContext:
    defaults = dict(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="sess-1",
    )
    defaults.update(overrides)
    return MemoryRequestContext(**defaults)


def test_tenant_conditions_always_includes_project_id() -> None:
    conditions = build_tenant_conditions(_ctx())
    fields = [c.field for c in conditions]
    assert "project_id" in fields
    assert fields.index("project_id") == 0


def test_tenant_conditions_includes_user_id_when_present() -> None:
    conditions = build_tenant_conditions(_ctx(user_id="user-42"))
    fields = [c.field for c in conditions]
    assert "user_id" in fields
    assert conditions[fields.index("user_id")].value == "user-42"


def test_tenant_conditions_includes_app_id_and_agent_id_when_present() -> None:
    conditions = build_tenant_conditions(_ctx(app_id="app-1", agent_id="agent-1"))
    fields = [c.field for c in conditions]
    assert "app_id" in fields
    assert "agent_id" in fields


def test_tenant_conditions_skips_none_fields() -> None:
    conditions = build_tenant_conditions(_ctx(app_id=None, agent_id=None))
    fields = [c.field for c in conditions]
    assert "app_id" not in fields
    assert "agent_id" not in fields
    assert "project_id" in fields
    assert "user_id" in fields


def test_tenant_conditions_skips_empty_string_fields() -> None:
    conditions = build_tenant_conditions(_ctx(user_id="", app_id=""))
    fields = [c.field for c in conditions]
    assert "user_id" not in fields
    assert "app_id" not in fields
    assert "project_id" in fields
