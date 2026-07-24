"""Mappers between skill DTOs and Qdrant point/record primitives (design §3).

Deterministic point ids implement Qdrant-side idempotency in place of the
missing unique constraint: the same dedup key always derives the same point id,
so a repeated upsert overwrites the same point instead of producing a duplicate
row.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from ..infra.db import (
    QdrantRecord,
    SkillBlobPoint,
    SkillTracePendingPoint,
    SkillTraceSummaryPoint,
    SkillVersionPoint,
)
from ..typing import (
    SkillBlob,
    SkillOrigin,
    SkillTracePending,
    SkillTraceSummary,
    SkillVersion,
    SkillVersionStatus,
)

# Fixed namespace for all skill-store deterministic ids (design §3). Kept stable
# forever; changing it would re-key every existing version/blob.
SKILL_ID_NAMESPACE = uuid.UUID("6f8d2a1e-3c4b-5a6d-8e9f-0a1b2c3d4e5f")


def skill_version_id(project_id: str, content_hash: str, parent_version_id: str | None) -> str:
    """Derive the deterministic ``version_id`` (== point id) for a version.

    A root version has ``parent_version_id=None``; the empty string is used in
    the key so root versions still get a stable id (design §3).
    """

    parent = parent_version_id or ""
    return str(uuid.uuid5(SKILL_ID_NAMESPACE, f"version|{project_id}|{content_hash}|{parent}"))


def skill_blob_id(project_id: str, content_hash: str) -> str:
    """Derive the deterministic ``skill_blob`` point id for content dedup."""

    return str(uuid.uuid5(SKILL_ID_NAMESPACE, f"blob|{project_id}|{content_hash}"))


def to_skill_version_point(version: SkillVersion) -> SkillVersionPoint:
    """Build the Qdrant point for one skill version."""

    return SkillVersionPoint(
        version_id=version.version_id,
        payload={
            "version_id": version.version_id,
            "project_id": version.project_id,
            "cloud_skill_id": version.cloud_skill_id,
            "skill_name": version.skill_name,
            "content_hash": version.content_hash,
            "parent_version_id": version.parent_version_id,
            "version_label": version.version_label,
            "status": version.status.value,
            "origin": version.origin.value,
            "created_at": version.created_at,
        },
    )


def skill_version_from_record(record: QdrantRecord) -> SkillVersion:
    """Reconstruct a ``SkillVersion`` from a Qdrant record."""

    payload = record.payload
    return SkillVersion(
        version_id=payload["version_id"],
        project_id=payload["project_id"],
        cloud_skill_id=payload["cloud_skill_id"],
        skill_name=payload["skill_name"],
        content_hash=payload["content_hash"],
        parent_version_id=payload.get("parent_version_id"),
        version_label=payload.get("version_label"),
        status=SkillVersionStatus(payload["status"]),
        origin=SkillOrigin(payload["origin"]),
        created_at=_parse_datetime(payload["created_at"]),
    )


def to_skill_blob_point(blob: SkillBlob) -> SkillBlobPoint:
    """Build the Qdrant point for one deduplicated bundle content row."""

    return SkillBlobPoint(
        blob_id=skill_blob_id(blob.project_id, blob.content_hash),
        payload={
            "project_id": blob.project_id,
            "content_hash": blob.content_hash,
            "content": blob.content,
            "created_at": blob.created_at,
        },
    )


def skill_blob_from_record(record: QdrantRecord) -> SkillBlob:
    """Reconstruct a ``SkillBlob`` from a Qdrant record."""

    payload = record.payload
    return SkillBlob(
        project_id=payload["project_id"],
        content_hash=payload["content_hash"],
        content=payload["content"],
        created_at=_parse_datetime(payload["created_at"]),
    )


def to_skill_trace_pending_point(trace: SkillTracePending) -> SkillTracePendingPoint:
    """Build the Qdrant point for one pending skill trace."""

    return SkillTracePendingPoint(
        trace_point_id=trace.trace_id,
        payload={
            "trace_id": trace.trace_id,
            "project_id": trace.project_id,
            "content_hash": trace.content_hash,
            "base_version_id": trace.base_version_id,
            "add_record_id": trace.add_record_id,
            "created_at": trace.created_at,
        },
    )


def skill_trace_pending_from_record(record: QdrantRecord) -> SkillTracePending:
    """Reconstruct a ``SkillTracePending`` from a Qdrant record."""

    payload = record.payload
    return SkillTracePending(
        trace_id=payload["trace_id"],
        project_id=payload["project_id"],
        content_hash=payload["content_hash"],
        base_version_id=payload.get("base_version_id", ""),
        add_record_id=payload.get("add_record_id", ""),
        created_at=_parse_datetime(payload["created_at"]),
    )


def to_skill_trace_summary_point(summary: SkillTraceSummary) -> SkillTraceSummaryPoint:
    """Build the Qdrant point for one trajectory summary.

    The point id equals ``summary_id`` (== originating ``add_record_id``) so a
    re-summarized trace overwrites in place.
    """

    return SkillTraceSummaryPoint(
        summary_id=summary.summary_id,
        payload={
            "summary_id": summary.summary_id,
            "project_id": summary.project_id,
            "cloud_skill_id": summary.cloud_skill_id,
            "add_record_id": summary.add_record_id,
            "skill_name": summary.skill_name,
            "summary": summary.summary,
            "created_at": summary.created_at,
            "consumed_version_id": summary.consumed_version_id,
            "score": summary.score,
            "task_id": summary.task_id,
        },
    )


def skill_trace_summary_from_record(record: QdrantRecord) -> SkillTraceSummary:
    """Reconstruct a ``SkillTraceSummary`` from a Qdrant record."""

    payload = record.payload
    return SkillTraceSummary(
        summary_id=payload["summary_id"],
        project_id=payload["project_id"],
        cloud_skill_id=payload["cloud_skill_id"],
        add_record_id=payload["add_record_id"],
        skill_name=payload.get("skill_name", ""),
        summary=payload.get("summary", ""),
        created_at=_parse_datetime(payload["created_at"]),
        consumed_version_id=payload.get("consumed_version_id"),
        score=payload.get("score"),
        task_id=payload.get("task_id"),
    )


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
