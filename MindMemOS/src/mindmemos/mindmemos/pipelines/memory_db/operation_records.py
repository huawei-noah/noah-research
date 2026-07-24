"""Memory DB boundary for public operation audit records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Awaitable, TypeVar

from ...infra.db import get_database_clients
from ...logging import get_logger
from ...mappers import to_add_record_point, to_search_record_point
from ...typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    MemoryAddEventItem,
    MemoryRequestContext,
    SearchPipelineInput,
    SearchPipelineResult,
    SkillBinding,
)
from .add_record_store import AddRecordStore

logger = get_logger(__name__)
T = TypeVar("T")


class MemoryOperationRecorder:
    """Persist public pipeline protocol inputs and outputs for audit/replay."""

    def __init__(self, *, add_record_store: AddRecordStore | None = None) -> None:
        self._add_records = add_record_store or AddRecordStore()

    async def record_add(
        self,
        inp: AddPipelineInput,
        result: AddPipelineSyncResult | AddPipelineAsyncResult | None,
        *,
        ctx: MemoryRequestContext,
        request_submitted_at: datetime,
        task_completed_at: datetime,
        add_record_id: str | None = None,
        skill_bindings: list[SkillBinding] | None = None,
        score: float | None = None,
        task_id: str | None = None,
        extra_payload: dict | None = None,
    ) -> str:
        """Write one add protocol record.

        ``add_record_id`` lets the caller pre-allocate the trace id so skill
        ``skill_bindings`` (design §2.1) and pending traces can reference the same
        record; both default to the previous behavior when omitted. ``score`` and
        ``task_id`` are trajectory annotations (rollout grouping / evaluation),
        stored as trace metadata and left unset when the caller omits them.
        """

        point = to_add_record_point(
            inp,
            result,
            ctx=ctx,
            request_submitted_at=request_submitted_at,
            task_completed_at=task_completed_at,
            add_record_id=add_record_id,
            skill_bindings=skill_bindings,
            score=score,
            task_id=task_id,
            extra_payload=extra_payload,
        )
        await self._add_records.append(point)
        return point.add_record_id

    async def record_add_input(
        self,
        inp: AddPipelineInput,
        *,
        ctx: MemoryRequestContext,
        request_submitted_at: datetime,
        add_record_id: str,
        status: str,
        skill_bindings: list[SkillBinding] | None = None,
        score: float | None = None,
        task_id: str | None = None,
    ) -> str:
        """Write the add input record up front, before the output is known.

        Captures the request payload, skill bindings, and trajectory annotations
        under ``add_record_id`` with an explicit lifecycle ``status`` (``queued``
        for async, ``processing`` for sync). The algorithm later writes the output
        back onto the same record via :meth:`mark_add_completed` or
        :meth:`append_add_output`.
        """

        point = to_add_record_point(
            inp,
            None,
            ctx=ctx,
            request_submitted_at=request_submitted_at,
            task_completed_at=None,
            add_record_id=add_record_id,
            skill_bindings=skill_bindings,
            score=score,
            task_id=task_id,
            status=status,
        )
        await self._add_records.append(point)
        return point.add_record_id

    async def mark_add_processing(self, ctx: MemoryRequestContext, add_record_id: str) -> None:
        """Mark an async add record as being processed by a worker."""

        await self._add_records.patch(
            ctx.project_id, add_record_id, {"status": "processing", "processing_at": utcnow()}
        )

    async def mark_add_completed(
        self,
        ctx: MemoryRequestContext,
        add_record_id: str,
        result: AddPipelineSyncResult,
    ) -> None:
        """Patch an async add record with final sync worker results."""
        if add_record_id is None:
            raise ValueError("Missing record id, cannot complete add record writeback")

        await self._add_records.patch(
            ctx.project_id,
            add_record_id,
            {
                "status": result.status,
                "task_completed_at": utcnow(),
                "memories": [memory.model_dump(mode="python") for memory in result.memories],
            },
        )

    async def append_add_output(
        self,
        ctx: MemoryRequestContext,
        add_record_id: str,
        events: list[MemoryAddEventItem],
    ) -> None:
        """Accumulate output events onto an existing add record (trigger binding).

        Used by the schema async path where one triggering request may produce
        several episodes over time. Reads the current ``memories`` and appends,
        so multiple episode completions for the same trigger accumulate instead
        of overwriting. Safe because drain/episode tasks for a buffer key are
        serialized onto one partition by ``dispatch_key``.
        """
        if add_record_id is None:
            raise ValueError("Missing record id, cannot complete partial add record writeback")

        records = await self._add_records.get_by_ids(ctx.project_id, [add_record_id])
        memories = list(records[0].payload.get("memories") or []) if records else []
        memories.extend(event.model_dump(mode="python") for event in events)
        await self._add_records.patch(
            ctx.project_id,
            add_record_id,
            {"status": "ok", "task_completed_at": utcnow(), "memories": memories},
        )

    async def mark_add_failed(self, ctx: MemoryRequestContext, add_record_id: str, error: str) -> None:
        """Patch an async add record with worker failure details."""

        await self._add_records.patch(
            ctx.project_id,
            add_record_id,
            {
                "status": "error",
                "error": error,
                "task_completed_at": utcnow(),
            },
        )

    async def record_search(
        self,
        inp: SearchPipelineInput,
        result: SearchPipelineResult | None,
        *,
        ctx: MemoryRequestContext,
        request_submitted_at: datetime,
        task_completed_at: datetime,
    ) -> None:
        """Write one search protocol record."""

        point = to_search_record_point(
            inp,
            result,
            ctx=ctx,
            request_submitted_at=request_submitted_at,
            task_completed_at=task_completed_at,
        )
        await get_database_clients().qdrant.upsert_search_record(point)


def utcnow() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(UTC)


async def suppress_recording_errors(awaitable: Awaitable[T], *, operation: str) -> T | None:
    """Run a recorder write without letting audit failures break add/search."""

    try:
        return await awaitable
    except Exception as exc:
        logger.warning("memory operation record write failed", operation=operation, error=str(exc))
    return None
