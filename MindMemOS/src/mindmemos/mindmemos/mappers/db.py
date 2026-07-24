"""Map business memory DTOs to low-level database primitives."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4, uuid5

from qdrant_client import models as qmodels

from ..infra import db
from ..infra.db import (
    FILTERABLE_ENTITY_FIELDS,
    FILTERABLE_MEMORY_FIELDS,
    build_filter,
    datetime_range,
    is_empty,
    is_null,
    match_any,
    match_except,
    match_text,
    match_value,
    number_range,
)
from ..typing import (
    REL_HAS_PROPERTY_MEMORY,
    REL_NEXT_IN_PROPERTY_TIMELINE,
    REL_RELATED_TO,
    REL_RELATES_TO,
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    EntityVectorWrite,
    EntityWrite,
    FieldCondition,
    GraphNodeRef,
    GraphRelationship,
    MemoryDbWritePlan,
    MemoryRequestContext,
    MemoryWrite,
    SearchFilter,
    SearchPipelineInput,
    SearchPipelineResult,
    SkillBinding,
    SourceWrite,
    VectorWrite,
)
from .errors import MappingError, ProjectIsolationError


def ensure_project(ctx: MemoryRequestContext, project_id: str) -> None:
    """Validate that a write belongs to the request project."""

    if project_id != ctx.project_id:
        raise ProjectIsolationError(f"project_id {project_id!r} does not match ctx.project_id {ctx.project_id!r}")


def search_filter_to_qdrant(
    ctx: MemoryRequestContext,
    sf: SearchFilter | None = None,
    *,
    target: str = "memory",
) -> qmodels.Filter:
    """Convert a ``SearchFilter`` tree to a project-scoped Qdrant filter."""

    filterable_fields = _filterable_fields_for_target(target)
    must: list[Any] = [match_value("project_id", ctx.project_id)]
    should: list[Any] = []
    must_not: list[Any] = []
    if sf is not None:
        must.extend(_to_condition(c, filterable_fields=filterable_fields) for c in sf.must)
        should.extend(_to_condition(c, filterable_fields=filterable_fields) for c in sf.should)
        must_not.extend(_to_condition(c, filterable_fields=filterable_fields) for c in sf.must_not)
    return build_filter(must=must, should=should, must_not=must_not)


def _filterable_fields_for_target(target: str) -> frozenset[str]:
    if target == "memory":
        return FILTERABLE_MEMORY_FIELDS
    if target == "entity":
        return FILTERABLE_ENTITY_FIELDS
    raise MappingError(f"unsupported search filter target {target!r}")


def _to_condition(c: FieldCondition | SearchFilter, *, filterable_fields: frozenset[str]) -> Any:
    """Translate one business condition (or nested filter) into a Qdrant condition."""

    if isinstance(c, SearchFilter):
        return build_filter(
            must=[_to_condition(x, filterable_fields=filterable_fields) for x in c.must],
            should=[_to_condition(x, filterable_fields=filterable_fields) for x in c.should],
            must_not=[_to_condition(x, filterable_fields=filterable_fields) for x in c.must_not],
        )
    if c.field not in filterable_fields:
        raise MappingError(f"field {c.field!r} is not filterable")
    match c.op:
        case "match":
            if c.value is None:
                raise MappingError(f"op 'match' on {c.field!r} requires 'value'")
            return match_value(c.field, c.value)
        case "any":
            return match_any(c.field, c.values or [])
        case "except":
            return match_except(c.field, c.values or [])
        case "text":
            if not isinstance(c.value, str):
                raise MappingError(f"op 'text' on {c.field!r} requires a string 'value'")
            return match_text(c.field, c.value)
        case "range":
            return number_range(c.field, gt=c.gt, gte=c.gte, lt=c.lt, lte=c.lte)
        case "datetime":
            return datetime_range(c.field, gt=c.gt, gte=c.gte, lt=c.lt, lte=c.lte)
        case "is_empty":
            return is_empty(c.field)
        case "is_null":
            return is_null(c.field)
    raise MappingError(f"unsupported filter op {c.op!r}")


def to_memory_payload(write: MemoryWrite, *, ctx: MemoryRequestContext | None = None) -> dict[str, Any]:
    """Convert MemoryWrite to the Qdrant ``memory_item_v1`` payload."""

    if ctx is not None:
        ensure_project(ctx, write.project_id)
    return write.model_dump(mode="python", exclude_none=True)


def to_entity_payload(write: EntityWrite, *, ctx: MemoryRequestContext | None = None) -> dict[str, Any]:
    """Convert EntityWrite to the Qdrant ``entity_item_v1`` payload."""

    if ctx is not None:
        ensure_project(ctx, write.project_id)
    return write.model_dump(mode="python", exclude_none=True)


def to_source_payload(write: SourceWrite, *, ctx: MemoryRequestContext | None = None) -> dict[str, Any]:
    """Convert SourceWrite to the Qdrant ``source_item_v1`` payload."""

    if ctx is not None:
        ensure_project(ctx, write.project_id)
    # `exclude` drops persist_payload (a routing flag, not payload); exclude_none then
    # strips optional None fields. Field-name exclusion and None filtering are independent.
    return write.model_dump(mode="python", exclude={"persist_payload"}, exclude_none=True)


def to_add_record_point(
    inp: AddPipelineInput,
    result: AddPipelineSyncResult | AddPipelineAsyncResult | None,
    *,
    ctx: MemoryRequestContext,
    request_submitted_at: Any,
    task_completed_at: Any,
    add_record_id: str | None = None,
    skill_bindings: list[SkillBinding] | None = None,
    score: float | None = None,
    task_id: str | None = None,
    status: str | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> db.AddRecordPoint:
    """Convert Add pipeline protocol values to an ``add_record_v1`` point.

    ``skill_bindings`` are the per-turn skill trace bindings (design §2.1); a
    pending binding carries ``version_id=None`` and is filled in later by rebind.
    The list is stored only when provided so existing add records are unaffected.
    ``score`` and ``task_id`` are trajectory annotations (evaluation score /
    rollout grouping); each is written only when supplied so older records and
    callers that omit them stay unchanged. ``status`` lets the caller seed an
    explicit lifecycle status (e.g. ``queued``/``processing``) when writing the
    input record before the output is known; a non-null ``result`` overrides it.
    """

    record_id = add_record_id or str(uuid4())
    payload = _context_payload(ctx)
    payload.update(
        {
            "add_record_id": record_id,
            "event_timestamp_ms": inp.event_timestamp,
            "event_time": inp.event_timestamp_utc,
            "request_submitted_at": request_submitted_at,
            "task_completed_at": task_completed_at,
            "messages": _model_list_dump(inp.messages),
            "mode": inp.mode,
            "feedback_processed": False,
            "metadata": dict(inp.metadata),
            "consolidation_status": "pending",
            "consolidated_at": None,
            "consolidation_run_id": None,
        }
    )
    if extra_payload:
        payload.update(extra_payload)
    if status is not None:
        payload["status"] = status
    if result is not None:
        payload["status"] = result.status
        if isinstance(result, AddPipelineSyncResult):
            payload["memories"] = _model_list_dump(result.memories)
    if skill_bindings is not None:
        payload["skill_bindings"] = [binding.model_dump(mode="json") for binding in skill_bindings]
    if score is not None:
        payload["score"] = score
    if task_id is not None:
        payload["task_id"] = task_id
    return db.AddRecordPoint(add_record_id=record_id, payload=payload)


def to_schema_add_buffer_point(
    inp: AddPipelineInput,
    *,
    ctx: MemoryRequestContext,
    request_submitted_at: Any,
    task_completed_at: Any,
    schema_buffer_record_id: str | None = None,
    source_add_record_id: str | None = None,
    force_generation: bool = False,
    extra_payload: dict[str, Any] | None = None,
) -> db.SchemaAddBufferPoint:
    """Convert one schema add buffer entry to a ``schema_add_buffer_v1`` point."""

    record_id = schema_buffer_record_id or str(uuid4())
    payload = _context_payload(ctx)
    payload.update(
        {
            "schema_buffer_record_id": record_id,
            "source_add_record_id": source_add_record_id,
            "timestamp": inp.event_timestamp,
            "event_timestamp_ms": inp.event_timestamp,
            "event_time": inp.event_timestamp_utc,
            "request_submitted_at": request_submitted_at,
            "task_completed_at": task_completed_at,
            "messages": _model_list_dump(inp.messages),
            "mode": inp.mode,
            "force_generation": force_generation,
            "metadata": dict(inp.metadata),
        }
    )
    if extra_payload:
        payload.update(extra_payload)
    return db.SchemaAddBufferPoint(schema_buffer_record_id=record_id, payload=payload)


def to_search_record_point(
    inp: SearchPipelineInput,
    result: SearchPipelineResult | None,
    *,
    ctx: MemoryRequestContext,
    request_submitted_at: Any,
    task_completed_at: Any,
    search_record_id: str | None = None,
) -> db.SearchRecordPoint:
    """Convert Search pipeline protocol values to a ``search_record_v1`` point."""

    payload = _context_payload(ctx)
    payload.update(
        {
            "request_submitted_at": request_submitted_at,
            "task_completed_at": task_completed_at,
            "query": inp.query,
            "filters": inp.filters if inp.filters else None,
            "top_k": inp.top_k,
            "search_pipeline": inp.search_pipeline,
            "agentic": inp.agentic,
            "max_rounds": inp.max_rounds,
            "rerank": inp.rerank,
        }
    )
    if result is not None:
        payload["status"] = result.status
        payload["memories"] = _model_list_dump(result.memories)
    return db.SearchRecordPoint(search_record_id=search_record_id or str(uuid4()), payload=payload)


def to_sparse_vector_data(vector: VectorWrite | None) -> db.SparseVectorData | None:
    """Convert vector DTO sparse fields to a Qdrant sparse vector primitive."""

    if vector is None or not vector.bm25_indices:
        return None
    return db.SparseVectorData(indices=list(vector.bm25_indices), values=list(vector.bm25_values))


def to_memory_point(
    write: MemoryWrite,
    *,
    vector: VectorWrite | None = None,
    ctx: MemoryRequestContext | None = None,
) -> db.MemoryPoint:
    """Convert MemoryWrite and optional VectorWrite to MemoryPoint."""

    if vector is not None and vector.memory_id != write.memory_id:
        raise MappingError(f"vector.memory_id {vector.memory_id!r} does not match memory_id {write.memory_id!r}")
    return db.MemoryPoint(
        memory_id=write.memory_id,
        payload=to_memory_payload(write, ctx=ctx),
        semantic_vector=vector.semantic_vector if vector else None,
        bm25_vector=to_sparse_vector_data(vector),
    )


def to_entity_point(
    write: EntityWrite,
    *,
    vector: EntityVectorWrite | None = None,
    ctx: MemoryRequestContext | None = None,
) -> db.EntityPoint:
    """Convert EntityWrite and optional EntityVectorWrite to EntityPoint."""

    if vector is not None and vector.entity_id != write.entity_id:
        raise MappingError(f"vector.entity_id {vector.entity_id!r} does not match entity_id {write.entity_id!r}")
    sparse = None
    if vector and vector.bm25_indices is not None and vector.bm25_values is not None:
        sparse = db.SparseVectorData(indices=vector.bm25_indices, values=vector.bm25_values)
    return db.EntityPoint(
        entity_id=write.entity_id,
        payload=to_entity_payload(write, ctx=ctx),
        vector=vector.semantic_vector if vector else None,
        bm25_vector=sparse,
    )


def to_source_point(write: SourceWrite, *, ctx: MemoryRequestContext | None = None) -> db.SourcePoint:
    """Convert SourceWrite to SourcePoint."""

    return db.SourcePoint(source_id=write.source_id, payload=to_source_payload(write, ctx=ctx))


def to_memory_node(write: MemoryWrite, *, ctx: MemoryRequestContext | None = None) -> db.MemoryNode:
    """Convert MemoryWrite to Neo4j Memory node input."""

    if ctx is not None:
        ensure_project(ctx, write.project_id)
    return db.MemoryNode(project_id=write.project_id, memory_id=write.memory_id, content=write.content)


def to_entity_node(write: EntityWrite, *, ctx: MemoryRequestContext | None = None) -> db.EntityNode:
    """Convert EntityWrite to Neo4j Entity node input."""

    if ctx is not None:
        ensure_project(ctx, write.project_id)
    return db.EntityNode(
        project_id=write.project_id,
        entity_id=write.entity_id,
        entity_name=write.entity_name,
        entity_type=write.entity_type,
        description=write.description,
    )


def to_source_node(write: SourceWrite, *, ctx: MemoryRequestContext | None = None) -> db.SourceNode:
    """Convert SourceWrite to Neo4j Source node input."""

    if ctx is not None:
        ensure_project(ctx, write.project_id)
    return db.SourceNode(
        project_id=write.project_id,
        source_id=write.source_id,
        parsed_content_path=write.parsed_content_path,
    )


def to_graph_relationship(rel: GraphRelationship, *, ctx: MemoryRequestContext | None = None) -> db.GraphRelationship:
    """Convert business graph relationship DTO to Neo4j relationship primitive."""

    if ctx is not None:
        ensure_project(ctx, rel.project_id)
    return db.GraphRelationship(
        source=_to_node_ref(rel.source),
        target=_to_node_ref(rel.target),
        rel_type=rel.rel_type,
        key=_relationship_key(rel),
        properties=_relationship_properties(rel),
    )


_SF_NAMESPACE = UUID("a1b2c3d4-0000-4000-8000-000000000000")


def _search_field_point_id(entity_id: str, index: int) -> str:
    """Generate a deterministic UUID for a search_field point from entity_id + index."""
    return str(uuid5(_SF_NAMESPACE, f"{entity_id}#sf{index}"))


def _to_search_field_entity_points(
    entities: list[EntityWrite],
    entity_vectors: dict[str, EntityVectorWrite],
    *,
    ctx: MemoryRequestContext | None = None,
) -> list[db.EntityPoint]:
    """Generate independent Qdrant points for each entity's search_fields (MaxSim multi-vector recall)."""
    points: list[db.EntityPoint] = []
    for write in entities:
        sfs = (write.metadata or {}).get("search_fields", [])
        for i, sf_text in enumerate(sfs):
            sf_lookup_key = f"{write.entity_id}#sf{i}"
            sf_vec = entity_vectors.get(sf_lookup_key)
            if not sf_vec or (not sf_vec.semantic_vector and not sf_vec.bm25_indices):
                continue
            sf_point_id = _search_field_point_id(write.entity_id, i)
            payload = to_entity_payload(write, ctx=ctx)
            if "metadata" not in payload or payload["metadata"] is None:
                payload["metadata"] = {}
            payload["metadata"]["is_search_field"] = True
            payload["metadata"]["search_field_content"] = str(sf_text)[:2000]
            payload["metadata"]["search_field_index"] = i
            sparse = None
            if sf_vec.bm25_indices is not None and sf_vec.bm25_values is not None:
                sparse = db.SparseVectorData(indices=sf_vec.bm25_indices, values=sf_vec.bm25_values)
            points.append(
                db.EntityPoint(
                    entity_id=sf_point_id,
                    payload=payload,
                    vector=sf_vec.semantic_vector,
                    bm25_vector=sparse,
                )
            )
    return points


def to_db_write_primitives(
    plan: MemoryDbWritePlan,
    *,
    ctx: MemoryRequestContext | None = None,
) -> tuple[list[db.MemoryPoint], list[db.EntityPoint], list[db.SourcePoint], list[db.GraphRelationship]]:
    """Convert a database write plan to Qdrant points and Neo4j relationships."""

    vectors = {vector.memory_id: vector for vector in plan.vectors}
    entity_vectors = {vector.entity_id: vector for vector in plan.entity_vectors}
    memory_points = [to_memory_point(memory, vector=vectors.get(memory.memory_id), ctx=ctx) for memory in plan.memories]
    entity_points = [
        to_entity_point(entity, vector=entity_vectors.get(entity.entity_id), ctx=ctx) for entity in plan.entities
    ]
    entity_points.extend(_to_search_field_entity_points(plan.entities, entity_vectors, ctx=ctx))
    source_points = [to_source_point(source, ctx=ctx) for source in plan.sources if source.persist_payload]
    relationships = [to_graph_relationship(rel, ctx=ctx) for rel in plan.relationships]
    return memory_points, entity_points, source_points, relationships


def _to_node_ref(ref: GraphNodeRef) -> db.NodeRef:
    if ref.kind == "Memory":
        return db.NodeRef(label="Memory", key={"project_id": ref.project_id, "memory_id": ref.node_id})
    if ref.kind == "Entity":
        return db.NodeRef(label="Entity", key={"project_id": ref.project_id, "entity_id": ref.node_id})
    return db.NodeRef(label="Source", key={"project_id": ref.project_id, "source_id": ref.node_id})


def _relationship_key(rel: GraphRelationship) -> dict[str, Any]:
    key: dict[str, Any] = {"project_id": rel.project_id}
    if rel.rel_type == REL_HAS_PROPERTY_MEMORY and rel.property_name:
        key["property_name"] = rel.property_name
    elif rel.rel_type == REL_NEXT_IN_PROPERTY_TIMELINE:
        if rel.entity_id:
            key["entity_id"] = rel.entity_id
        if rel.property_name:
            key["property_name"] = rel.property_name
    elif rel.rel_type == REL_RELATES_TO and rel.edge_type:
        key["edge_type"] = rel.edge_type
    elif rel.rel_type == REL_RELATED_TO and rel.relation_type:
        key["relation_type"] = rel.relation_type
    return key


def _relationship_properties(rel: GraphRelationship) -> dict[str, Any]:
    properties: dict[str, Any] = {"project_id": rel.project_id, "metadata": dict(rel.metadata)}
    for attr in (
        "property_name",
        "edge_type",
        "relation_type",
        "entity_id",
        "extraction_position",
        "mention_count",
    ):
        value = getattr(rel, attr)
        if value is not None:
            properties[attr] = value
    return properties


def _context_payload(ctx: MemoryRequestContext) -> dict[str, Any]:
    return ctx.model_dump(mode="python", exclude_none=True)


def _model_list_dump(values: list[Any]) -> list[Any]:
    dumped: list[Any] = []
    for value in values:
        if hasattr(value, "model_dump"):
            dumped.append(value.model_dump(mode="python", exclude_none=True))
        else:
            dumped.append(value)
    return dumped
