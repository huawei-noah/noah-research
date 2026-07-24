"""Qdrant filter and payload-index helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from qdrant_client import models as qmodels

from .models import PayloadIndexSpec


def match_value(key: str, value: str | int | bool) -> qmodels.FieldCondition:
    """Build a Qdrant exact-match condition."""

    return qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))


def match_any(key: str, values: list[str] | list[int]) -> qmodels.FieldCondition:
    """Build a Qdrant any-match condition."""

    return qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=values))


def match_except(key: str, values: list[str] | list[int]) -> qmodels.FieldCondition:
    """Build a Qdrant not-in condition."""

    return qmodels.FieldCondition(key=key, match=qmodels.MatchExcept(**{"except": values}))


def match_text(key: str, text: str) -> qmodels.FieldCondition:
    """Build a Qdrant full-text match condition."""

    return qmodels.FieldCondition(key=key, match=qmodels.MatchText(text=text))


def is_empty(key: str) -> qmodels.IsEmptyCondition:
    """Build a Qdrant is-empty condition."""

    return qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key=key))


def is_null(key: str) -> qmodels.IsNullCondition:
    """Build a Qdrant is-null condition."""

    return qmodels.IsNullCondition(is_null=qmodels.PayloadField(key=key))


def datetime_range(
    key: str,
    *,
    gt: datetime | None = None,
    gte: datetime | None = None,
    lt: datetime | None = None,
    lte: datetime | None = None,
) -> qmodels.FieldCondition:
    """Build a Qdrant datetime range condition."""

    return qmodels.FieldCondition(key=key, range=qmodels.DatetimeRange(gt=gt, gte=gte, lt=lt, lte=lte))


def number_range(
    key: str,
    *,
    gt: float | None = None,
    gte: float | None = None,
    lt: float | None = None,
    lte: float | None = None,
) -> qmodels.FieldCondition:
    """Build a Qdrant numeric range condition."""

    return qmodels.FieldCondition(key=key, range=qmodels.Range(gt=gt, gte=gte, lt=lt, lte=lte))


def build_filter(
    *,
    must: list[Any] | None = None,
    should: list[Any] | None = None,
    must_not: list[Any] | None = None,
) -> qmodels.Filter:
    """Build a Qdrant filter from already prepared primitive conditions."""

    return qmodels.Filter(must=must or None, should=should or None, must_not=must_not or None)


MEMORY_PAYLOAD_INDEX_SCHEMA: tuple[PayloadIndexSpec, ...] = (
    PayloadIndexSpec(field_name="memory_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="account_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="project_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="api_key_uuid", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="user_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="app_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="session_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="agent_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="request_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="mem_type", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="mem_extract_type", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="mem_extract_version", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="status", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="validate_from", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="validate_to", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="created_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="update_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="status_changed_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="reinforcement_count", field_schema=qmodels.PayloadSchemaType.INTEGER),
    PayloadIndexSpec(field_name="parent_ids", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="root_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="property_name", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="entity_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="entity_type", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="content", field_schema=qmodels.PayloadSchemaType.TEXT),
)

# Fields a caller-supplied SearchFilter may reference. Derived from the indexed
# memory payload so we never filter on an unindexed (slow) or isolation-critical
# field. ``project_id`` is intentionally excluded: it is force-injected by the
# mapper and must not be overridable by callers.
FILTERABLE_MEMORY_FIELDS: frozenset[str] = frozenset(
    spec.field_name for spec in MEMORY_PAYLOAD_INDEX_SCHEMA if spec.field_name != "project_id"
)


ENTITY_PAYLOAD_INDEX_SCHEMA: tuple[PayloadIndexSpec, ...] = (
    PayloadIndexSpec(field_name="entity_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="account_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="project_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="api_key_uuid", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="user_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="app_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="session_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="agent_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="request_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="entity_name", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="entity_type", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="status", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="created_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
)

FILTERABLE_ENTITY_FIELDS: frozenset[str] = frozenset(
    spec.field_name for spec in ENTITY_PAYLOAD_INDEX_SCHEMA if spec.field_name != "project_id"
)

SOURCE_PAYLOAD_INDEX_SCHEMA: tuple[PayloadIndexSpec, ...] = (
    PayloadIndexSpec(field_name="source_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="account_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="project_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="api_key_uuid", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="user_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="app_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="session_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="agent_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="request_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="source_type", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="file_path", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="file_name", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="is_parsed", field_schema=qmodels.PayloadSchemaType.BOOL),
    PayloadIndexSpec(field_name="status", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="created_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
)

ADD_RECORD_PAYLOAD_INDEX_SCHEMA: tuple[PayloadIndexSpec, ...] = (
    PayloadIndexSpec(field_name="add_record_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="account_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="project_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="api_key_uuid", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="user_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="app_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="session_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="agent_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="request_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="status", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="buffer_key", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="buffer_status", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="buffer_sequence", field_schema=qmodels.PayloadSchemaType.INTEGER),
    PayloadIndexSpec(field_name="buffered_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="split_attempted", field_schema=qmodels.PayloadSchemaType.BOOL),
    PayloadIndexSpec(field_name="split_attempted_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="episode_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="episode_queued_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="added_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="added_timestamp_ms", field_schema=qmodels.PayloadSchemaType.INTEGER),
    PayloadIndexSpec(field_name="event_time", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="event_timestamp_ms", field_schema=qmodels.PayloadSchemaType.INTEGER),
    PayloadIndexSpec(field_name="processed_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="consolidation_status", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="consolidated_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="consolidation_run_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="mode", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="task_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="score", field_schema=qmodels.PayloadSchemaType.FLOAT),
    PayloadIndexSpec(field_name="feedback_processed", field_schema=qmodels.PayloadSchemaType.BOOL),
    PayloadIndexSpec(field_name="request_submitted_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="task_completed_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
)

SCHEMA_ADD_BUFFER_PAYLOAD_INDEX_SCHEMA: tuple[PayloadIndexSpec, ...] = (
    PayloadIndexSpec(field_name="schema_buffer_record_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="source_add_record_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="account_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="project_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="api_key_uuid", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="user_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="app_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="session_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="agent_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="request_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="status", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="buffer_key", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="buffer_status", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="buffer_sequence", field_schema=qmodels.PayloadSchemaType.INTEGER),
    PayloadIndexSpec(field_name="buffered_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="split_attempted", field_schema=qmodels.PayloadSchemaType.BOOL),
    PayloadIndexSpec(field_name="split_attempted_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="episode_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="episode_queued_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="added_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="added_timestamp_ms", field_schema=qmodels.PayloadSchemaType.INTEGER),
    PayloadIndexSpec(field_name="event_time", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="event_timestamp_ms", field_schema=qmodels.PayloadSchemaType.INTEGER),
    PayloadIndexSpec(field_name="timestamp", field_schema=qmodels.PayloadSchemaType.INTEGER),
    PayloadIndexSpec(field_name="processed_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="mode", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="force_generation", field_schema=qmodels.PayloadSchemaType.BOOL),
)

SKILL_VERSION_PAYLOAD_INDEX_SCHEMA: tuple[PayloadIndexSpec, ...] = (
    PayloadIndexSpec(field_name="version_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="project_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="cloud_skill_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="skill_name", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="content_hash", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="parent_version_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="version_label", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="status", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="origin", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="created_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
)

SKILL_BLOB_PAYLOAD_INDEX_SCHEMA: tuple[PayloadIndexSpec, ...] = (
    PayloadIndexSpec(field_name="project_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="content_hash", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="created_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
)

SKILL_TRACE_PENDING_PAYLOAD_INDEX_SCHEMA: tuple[PayloadIndexSpec, ...] = (
    PayloadIndexSpec(field_name="trace_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="project_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="content_hash", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="base_version_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="created_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
)

SKILL_TRACE_SUMMARY_PAYLOAD_INDEX_SCHEMA: tuple[PayloadIndexSpec, ...] = (
    PayloadIndexSpec(field_name="summary_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="project_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="cloud_skill_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="add_record_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="consumed_version_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="task_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="score", field_schema=qmodels.PayloadSchemaType.FLOAT),
    PayloadIndexSpec(field_name="created_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
)

SEARCH_RECORD_PAYLOAD_INDEX_SCHEMA: tuple[PayloadIndexSpec, ...] = (
    PayloadIndexSpec(field_name="account_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="project_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="api_key_uuid", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="user_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="app_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="session_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="agent_id", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="request_id", field_schema=qmodels.PayloadSchemaType.UUID),
    PayloadIndexSpec(field_name="status", field_schema=qmodels.PayloadSchemaType.KEYWORD),
    PayloadIndexSpec(field_name="agentic", field_schema=qmodels.PayloadSchemaType.BOOL),
    PayloadIndexSpec(field_name="rerank", field_schema=qmodels.PayloadSchemaType.BOOL),
    PayloadIndexSpec(field_name="max_rounds", field_schema=qmodels.PayloadSchemaType.INTEGER),
    PayloadIndexSpec(field_name="request_submitted_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
    PayloadIndexSpec(field_name="task_completed_at", field_schema=qmodels.PayloadSchemaType.DATETIME),
)
