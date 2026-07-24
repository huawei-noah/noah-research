"""Memory DB boundary for durable add-record buffers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from qdrant_client import models as qmodels

from ...infra.db import DatabaseClients, QdrantRecord, SchemaAddBufferPoint, build_filter, match_any, match_value
from ...mappers import to_schema_add_buffer_point
from ...typing import AddPipelineInput, MemoryRequestContext, utc_millis_from_datetime
from .schema_add_buffer_store import SchemaAddBufferStore


@dataclass(slots=True)
class BufferedAddRecord:
    add_record_id: str
    payload: dict[str, Any]

    @property
    def buffer_sequence(self) -> int:
        """Return the record order inside its durable add buffer."""
        value = self.payload.get("buffer_sequence")
        return int(value) if value is not None else 0


@dataclass(slots=True)
class AddRecordBufferKey:
    project_id: str
    buffer_key: str


class AddRecordBuffer:
    """Persist and consume raw add requests as a durable buffer."""

    def __init__(self, *, store: SchemaAddBufferStore | None = None, clients: DatabaseClients | None = None) -> None:
        self._store = store or SchemaAddBufferStore(clients=clients)

    async def append(
        self,
        ctx: MemoryRequestContext,
        inp: AddPipelineInput,
        *,
        force_generation: bool,
        source_add_record_id: str | None = None,
    ) -> None:
        """Persist incoming add messages as queued buffer records.

        Args:
            ctx: Tenant, project, and actor context for hard isolation.
            inp: Add request whose messages should be buffered.
            force_generation: Whether the final buffered message should force episode generation.
        """
        added_at = datetime.now(UTC)
        points: list[SchemaAddBufferPoint] = []
        for index, message in enumerate(inp.messages):
            schema_buffer_record_id = str(uuid4())
            message_input = inp.model_copy(
                update={
                    "messages": [message],
                    "force_generation": force_generation and index == len(inp.messages) - 1,
                    "metadata": {**dict(inp.metadata), "buffer_message_index": index},
                }
            )
            point = to_schema_add_buffer_point(
                message_input,
                ctx=ctx,
                request_submitted_at=added_at,
                task_completed_at=added_at,
                schema_buffer_record_id=schema_buffer_record_id,
                source_add_record_id=source_add_record_id,
                force_generation=force_generation and index == len(inp.messages) - 1,
                extra_payload={
                    "status": "queued",
                    "buffer_status": "buffered",
                    "buffer_key": buffer_key(ctx),
                    "buffer_sequence": _sequence(added_at, index),
                    "buffered_at": added_at,
                    "added_at": added_at,
                    "added_timestamp_ms": utc_millis_from_datetime(added_at),
                    "force_generation": force_generation and index == len(inp.messages) - 1,
                },
            )
            points.append(point)
        await self._store.append_many(points)

    async def list_buffered(self, ctx: MemoryRequestContext, *, limit: int) -> list[BufferedAddRecord]:
        """List buffered records for the request context.

        Args:
            ctx: Tenant, project, and actor context for hard isolation.
            limit: Maximum number of records to return.

        Returns:
            Buffered records ordered by buffer sequence.
        """
        qfilter = build_filter(
            must=[
                match_value("buffer_key", buffer_key(ctx)),
                match_value("buffer_status", "buffered"),
            ]
        )
        records, _ = await self._store.list(
            ctx.project_id,
            filters=qfilter,
            limit=limit,
            order_by=_buffer_order(),
        )
        return [_to_buffered(record) for record in records]

    async def list_buffered_for_key(
        self,
        project_id: str,
        key: str,
        *,
        limit: int,
    ) -> list[BufferedAddRecord]:
        """Load unsplit records for one durable buffer key."""

        qfilter = build_filter(
            must=[
                match_value("buffer_key", key),
                match_value("buffer_status", "buffered"),
            ]
        )
        records, _ = await self._store.list(
            project_id,
            filters=qfilter,
            limit=limit,
            order_by=_buffer_order(),
        )
        return [_to_buffered(record) for record in records]

    async def list_buffer_keys_with_new_records(self, *, limit: int = 100) -> list[AddRecordBufferKey]:
        """Return buffer keys that have records waiting for drain."""

        qfilter = build_filter(
            must=[
                match_value("buffer_status", "buffered"),
            ],
            must_not=[
                match_value("split_attempted", True),
            ],
        )
        records, _ = await self._store.list_global(filters=qfilter, limit=limit, order_by=_buffer_order())
        keys: dict[tuple[str, str], AddRecordBufferKey] = {}
        for record in records:
            key = record.payload.get("buffer_key")
            project_id = record.payload.get("project_id")
            if not key or not project_id:
                continue
            keys.setdefault(
                (str(project_id), str(key)),
                AddRecordBufferKey(project_id=str(project_id), buffer_key=str(key)),
            )
        return list(keys.values())

    async def get_by_ids(self, ctx: MemoryRequestContext, add_record_ids: list[str]) -> list[BufferedAddRecord]:
        """Load buffered add records by ID."""

        records = await self._store.get_by_ids(ctx.project_id, add_record_ids)
        return _matching_context_records(ctx, [_to_buffered(record) for record in records])

    async def has_pending(self, ctx: MemoryRequestContext) -> bool:
        """Check whether any add records are still unfinished.

        Args:
            ctx: Tenant, project, and actor context for hard isolation.

        Returns:
            True when buffered, queued, or processing records remain.
        """
        qfilter = build_filter(
            must=[
                match_value("buffer_key", buffer_key(ctx)),
                match_any("buffer_status", ["buffered", "episode_queued", "processing"]),
            ]
        )
        records, _ = await self._store.list(ctx.project_id, filters=qfilter, limit=1, order_by=_buffer_order())
        return len(records) > 0

    async def mark_split_attempted(self, ctx: MemoryRequestContext, records: list[BufferedAddRecord]) -> None:
        """Mark buffered records as already considered for episode splitting.

        Args:
            ctx: Tenant, project, and actor context for hard isolation.
            records: Candidate records to patch after context validation.
        """
        now = datetime.now(UTC)
        for record in _matching_context_records(ctx, records):
            await self._store.patch(
                ctx.project_id,
                record.add_record_id,
                {"split_attempted": True, "split_attempted_at": now},
            )

    async def mark_episode_queued(
        self,
        ctx: MemoryRequestContext,
        records: list[BufferedAddRecord],
        *,
        episode_id: str,
    ) -> None:
        """Mark buffered records as queued for one episode extraction task.

        Args:
            ctx: Tenant, project, and actor context for hard isolation.
            records: Candidate records to patch after context validation.
            episode_id: Episode identifier assigned to the queued work.
        """
        now = datetime.now(UTC)
        for record in _matching_context_records(ctx, records):
            await self._store.patch(
                ctx.project_id,
                record.add_record_id,
                {
                    "buffer_status": "episode_queued",
                    "episode_id": episode_id,
                    "episode_queued_at": now,
                    "split_attempted": True,
                    "split_attempted_at": now,
                },
            )

    async def mark_processing(self, ctx: MemoryRequestContext, records: list[BufferedAddRecord]) -> None:
        """Mark records as actively being processed.

        Args:
            ctx: Tenant, project, and actor context for hard isolation.
            records: Candidate records to patch after context validation.
        """
        now = datetime.now(UTC)
        for record in _matching_context_records(ctx, records):
            await self._store.patch(
                ctx.project_id,
                record.add_record_id,
                {"buffer_status": "processing", "processing_at": now},
            )

    async def mark_processed(
        self,
        ctx: MemoryRequestContext,
        records: list[BufferedAddRecord],
        *,
        episode_id: str,
        events: list[dict[str, Any]],
    ) -> None:
        """Mark records as successfully processed.

        Args:
            ctx: Tenant, project, and actor context for hard isolation.
            records: Candidate records to patch after context validation.
            episode_id: Episode identifier that produced the events.
            events: Serialized add events produced by the episode.
        """
        now = datetime.now(UTC)
        for record in _matching_context_records(ctx, records):
            await self._store.patch(
                ctx.project_id,
                record.add_record_id,
                {
                    "status": "ok",
                    "buffer_status": "processed",
                    "episode_id": episode_id,
                    "processed_at": now,
                    "task_completed_at": now,
                    "memories": events,
                },
            )

    async def mark_failed(self, ctx: MemoryRequestContext, records: list[BufferedAddRecord], *, error: str) -> None:
        """Mark records as failed with an error message.

        Args:
            ctx: Tenant, project, and actor context for hard isolation.
            records: Candidate records to patch after context validation.
            error: Failure message to persist.
        """
        now = datetime.now(UTC)
        for record in _matching_context_records(ctx, records):
            await self._store.patch(
                ctx.project_id,
                record.add_record_id,
                {
                    "status": "error",
                    "buffer_status": "failed",
                    "error": error,
                    "task_completed_at": now,
                },
            )

    async def restore_buffered(
        self,
        ctx: MemoryRequestContext,
        records: list[BufferedAddRecord],
        *,
        error: str | None = None,
    ) -> None:
        """Return records to the buffered state after a recoverable failure.

        Args:
            ctx: Tenant, project, and actor context for hard isolation.
            records: Candidate records to patch after context validation.
            error: Optional last-error message for retry diagnostics.
        """
        for record in _matching_context_records(ctx, records):
            payload: dict[str, Any] = {"buffer_status": "buffered", "split_attempted": False}
            if error:
                payload["last_error"] = error
            await self._store.patch(ctx.project_id, record.add_record_id, payload)

    async def delete_processed(self, ctx: MemoryRequestContext, records: list[BufferedAddRecord]) -> None:
        """Delete buffer records that have been successfully processed.

        Args:
            ctx: Tenant, project, and actor context for hard isolation.
            records: Candidate records to delete after context validation.
        """
        matched = _matching_context_records(ctx, records)
        if matched:
            await self._store.delete_many([r.add_record_id for r in matched])


def buffer_key(ctx: MemoryRequestContext) -> str:
    return f"{ctx.project_id}:{ctx.user_id}"


def _matching_context_records(ctx: MemoryRequestContext, records: list[BufferedAddRecord]) -> list[BufferedAddRecord]:
    key = buffer_key(ctx)
    return [
        record
        for record in records
        if record.payload.get("project_id") == ctx.project_id and record.payload.get("buffer_key") == key
    ]


def context_from_record(record: BufferedAddRecord) -> MemoryRequestContext:
    payload = record.payload
    return MemoryRequestContext.model_validate(
        {
            "request_id": payload.get("request_id"),
            "account_id": payload.get("account_id"),
            "project_id": payload.get("project_id"),
            "api_key_uuid": payload.get("api_key_uuid"),
            "user_id": payload.get("user_id"),
            "app_id": payload.get("app_id"),
            "session_id": payload.get("session_id"),
            "agent_id": payload.get("agent_id"),
        }
    )


def _sequence(value: datetime, index: int = 0) -> int:
    return int(value.timestamp() * 1_000_000) + index


def _buffer_order() -> qmodels.OrderBy:
    return qmodels.OrderBy(key="buffer_sequence", direction=qmodels.Direction.ASC)


def _to_buffered(record: QdrantRecord) -> BufferedAddRecord:
    payload = dict(record.payload)
    payload.setdefault("schema_buffer_record_id", record.point_id)
    return BufferedAddRecord(add_record_id=record.point_id, payload=payload)
