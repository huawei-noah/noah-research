"""Map low-level database records back to business memory DTOs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..infra import db
from ..typing import (
    EntitySearchHit,
    EntitySearchResult,
    EntityView,
    MemoryDbMutationResult,
    MemoryDbSearchHit,
    MemoryDbSearchResult,
    MemoryDbWriteSummary,
    MemoryView,
    MemoryWrite,
)


def to_entity_view(payload: dict[str, Any]) -> EntityView:
    """Convert a Qdrant entity payload to a business-visible EntityView."""

    return EntityView(
        entity_id=str(payload["entity_id"]),
        project_id=str(payload["project_id"]),
        entity_name=str(payload.get("entity_name") or ""),
        entity_type=payload.get("entity_type"),
        description=payload.get("description"),
        metadata=dict(payload.get("metadata") or {}),
        account_id=payload.get("account_id"),
        api_key_uuid=payload.get("api_key_uuid"),
        user_id=payload.get("user_id"),
        app_id=payload.get("app_id"),
        session_id=payload.get("session_id"),
        agent_id=payload.get("agent_id"),
        request_id=payload.get("request_id"),
        created_at=_dt_from_payload(payload.get("created_at")),
        update_at=_dt_from_payload(payload.get("update_at")),
    )


def to_entity_view_from_record(record: db.QdrantRecord) -> EntityView:
    """Convert a Qdrant record to EntityView."""

    return to_entity_view(record.payload)


def to_entity_search_hit(record: db.QdrantSearchRecord) -> EntitySearchHit:
    """Convert a Qdrant entity search record to an entity search hit."""

    payload = record.payload or {}
    metadata = dict(payload.get("metadata") or {})
    is_search_field = bool(metadata.get("is_search_field"))
    best_search_field = str(record.debug.get("best_search_field") or "")
    best_search_field_index = record.debug.get("best_search_field_index")
    best_search_field_score = record.debug.get("best_search_field_score")
    matched_point_role = str(record.debug.get("matched_point_role") or ("search_field" if is_search_field else "core"))
    if is_search_field and not best_search_field:
        best_search_field = str(metadata.get("search_field_content") or "")
    if is_search_field and best_search_field_index is None:
        best_search_field_index = metadata.get("search_field_index")
    if is_search_field and best_search_field_score is None:
        best_search_field_score = record.score
    if not is_search_field and not best_search_field:
        best_search_field = str(metadata.get("core_search_field") or "")
    if not is_search_field and best_search_field_score is None and best_search_field:
        best_search_field_score = record.score

    return EntitySearchHit(
        entity_id=str(payload.get("entity_id") or record.point_id),
        score=record.score,
        entity=to_entity_view(payload) if payload else None,
        source=record.source,
        rank=record.debug.get("rank") if isinstance(record.debug.get("rank"), int) else None,
        best_search_field=best_search_field,
        best_search_field_index=best_search_field_index if isinstance(best_search_field_index, int) else None,
        best_search_field_score=float(best_search_field_score) if best_search_field_score is not None else None,
        matched_point_role=matched_point_role,
    )


def to_entity_search_result(
    query: str,
    hits: list[db.QdrantSearchRecord],
    *,
    debug: dict[str, Any] | None = None,
) -> EntitySearchResult:
    """Convert Qdrant entity hits to a DB-layer entity search result."""

    mapped = [to_entity_search_hit(hit) for hit in hits]
    return EntitySearchResult(query=query, hits=mapped, total=len(mapped), debug=debug or {})


def to_memory_view(payload: dict[str, Any]) -> MemoryView:
    """Convert a Qdrant memory payload to a business-visible MemoryView."""

    return MemoryView(
        memory_id=str(payload["memory_id"]),
        project_id=str(payload["project_id"]),
        content=str(payload.get("content") or ""),
        mem_type=payload.get("mem_type", "fact"),
        mem_extract_type=payload.get("mem_extract_type"),
        mem_extract_version=payload.get("mem_extract_version"),
        status=payload.get("status", "active"),
        metadata=dict(payload.get("metadata") or {}),
        account_id=payload.get("account_id"),
        api_key_uuid=payload.get("api_key_uuid"),
        user_id=payload.get("user_id"),
        app_id=payload.get("app_id"),
        session_id=payload.get("session_id"),
        agent_id=payload.get("agent_id"),
        request_id=payload.get("request_id"),
        parent_ids=list(payload.get("parent_ids") or []),
        root_id=list(payload.get("root_id") or []),
        property_name=payload.get("property_name"),
        entity_id=payload.get("entity_id"),
        entity_type=payload.get("entity_type"),
        validate_from=_dt_from_payload(payload.get("validate_from")),
        validate_to=_dt_from_payload(payload.get("validate_to")),
        created_at=_dt_from_payload(payload.get("created_at")),
        update_at=_dt_from_payload(payload.get("update_at")),
    )


def to_memory_view_from_record(record: db.QdrantRecord) -> MemoryView:
    """Convert a Qdrant record to MemoryView."""

    return to_memory_view(record.payload)


def to_memory_write(payload: dict[str, Any]) -> MemoryWrite:
    """Convert a Qdrant memory payload back to a write DTO."""

    return MemoryWrite(
        memory_id=str(payload["memory_id"]),
        account_id=str(payload["account_id"]),
        project_id=str(payload["project_id"]),
        api_key_uuid=str(payload["api_key_uuid"]),
        user_id=str(payload["user_id"]),
        app_id=payload.get("app_id"),
        session_id=str(payload["session_id"]),
        agent_id=payload.get("agent_id"),
        request_id=payload.get("request_id"),
        content=str(payload.get("content") or ""),
        mem_type=payload.get("mem_type", "fact"),
        mem_extract_type=str(payload.get("mem_extract_type") or "vanilla"),
        mem_extract_version=str(payload.get("mem_extract_version") or "unknown"),
        metadata=dict(payload.get("metadata") or {}),
        validate_from=_dt_from_payload(payload.get("validate_from")),
        validate_to=_dt_from_payload(payload.get("validate_to")),
        status=payload.get("status", "active"),
        reinforcement_count=int(payload.get("reinforcement_count") or 0),
        created_at=_dt_from_payload(payload.get("created_at")) or datetime.now(UTC),
        update_at=_dt_from_payload(payload.get("update_at")),
        status_changed_at=_dt_from_payload(payload.get("status_changed_at")),
        parent_ids=_string_list(payload.get("parent_ids")),
        root_id=_string_list(payload.get("root_id")),
        property_name=payload.get("property_name"),
        entity_id=payload.get("entity_id"),
        entity_type=payload.get("entity_type"),
    )


def to_memory_write_from_record(record: db.QdrantRecord) -> MemoryWrite:
    """Convert a Qdrant record to a memory write DTO."""

    return to_memory_write(record.payload)


def to_search_hit(record: db.QdrantSearchRecord) -> MemoryDbSearchHit:
    """Convert a Qdrant search record to a DB-layer search hit."""

    return MemoryDbSearchHit(
        memory_id=record.point_id,
        score=record.score,
        memory=to_memory_view(record.payload) if record.payload else None,
        source=record.source,
        rank=record.debug.get("rank") if isinstance(record.debug.get("rank"), int) else None,
        debug=dict(record.debug),
    )


def to_search_result(
    query: str, hits: list[db.QdrantSearchRecord], *, debug: dict[str, Any] | None = None
) -> MemoryDbSearchResult:
    """Convert Qdrant hits to a DB-layer search result."""

    mapped = [to_search_hit(hit) for hit in hits]
    return MemoryDbSearchResult(query=query, hits=mapped, total=len(mapped), debug=debug or {})


def to_add_result(
    *,
    memory_ids: list[str],
    entity_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
    debug: dict[str, Any] | None = None,
) -> MemoryDbWriteSummary:
    """Build the DB-layer write summary from pipeline write output."""

    return MemoryDbWriteSummary(
        status="ok",
        memory_ids=list(memory_ids),
        entity_ids=list(entity_ids or []),
        source_ids=list(source_ids or []),
        debug=debug or {},
    )


def to_mutation_result(memory_id: str, *, changed: bool = True, hard: bool = False) -> MemoryDbMutationResult:
    """Build the DB-layer mutation result for delete/update operations."""

    return MemoryDbMutationResult(status="ok", memory_id=memory_id, changed=changed, hard=hard)


def _dt_from_payload(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]
