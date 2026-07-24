"""Qdrant primitive store for the lightweight git-like skill version store."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from ....config import QdrantConfig
from ....logging import get_logger
from ..engine import QdrantEngine
from ..filters import datetime_range, is_empty, match_value
from ..models import (
    QdrantRecord,
    SkillBlobPoint,
    SkillTracePendingPoint,
    SkillTraceSummaryPoint,
    SkillVersionPoint,
)
from ..schema import (
    skill_blob_collection_spec,
    skill_trace_pending_collection_spec,
    skill_trace_summary_collection_spec,
    skill_version_collection_spec,
)

logger = get_logger(__name__)


class SkillVersionRepository:
    """Thin Qdrant adapter for the skill version store (no business logic).

    The skill collections live in the same Qdrant database as the memory tables,
    so this repository does not own a connection: it reuses a shared
    :class:`QdrantEngine` (typically ``QdrantStore.engine``) and never closes it.
    For standalone use it can be built from a borrowed ``AsyncQdrantClient``.
    """

    def __init__(
        self,
        cfg: QdrantConfig,
        *,
        client: AsyncQdrantClient | None = None,
        engine: QdrantEngine | None = None,
    ) -> None:
        if engine is None and client is None:
            raise ValueError("SkillVersionRepository requires either an engine or a client")
        self._cfg = cfg
        self._engine = engine if engine is not None else QdrantEngine(cfg, client=client)

    @property
    def version_collection(self) -> str:
        """Configured ``skill_version_v1`` collection name."""

        return self._cfg.skill_version_collection

    @property
    def blob_collection(self) -> str:
        """Configured ``skill_blob_v1`` collection name."""

        return self._cfg.skill_blob_collection

    @property
    def trace_pending_collection(self) -> str:
        """Configured ``skill_trace_pending_v1`` collection name."""

        return self._cfg.skill_trace_pending_collection

    @property
    def trace_summary_collection(self) -> str:
        """Configured ``skill_trace_summary_v1`` collection name."""

        return self._cfg.skill_trace_summary_collection

    async def ensure_schema(self) -> None:
        """Create the four skill collections and their payload indexes."""

        if not self._cfg.auto_create:
            return
        for spec in (
            skill_version_collection_spec(self._cfg),
            skill_blob_collection_spec(self._cfg),
            skill_trace_pending_collection_spec(self._cfg),
            skill_trace_summary_collection_spec(self._cfg),
        ):
            await self._engine.ensure_collection(spec)

    async def upsert_version(self, point: SkillVersionPoint) -> None:
        """Upsert one skill version point (idempotent via deterministic id)."""

        await self._engine.upsert(self.version_collection, [self._payload_point(point.version_id, point.payload)])

    async def upsert_blob(self, point: SkillBlobPoint) -> None:
        """Upsert one bundle content point (content dedup via deterministic id)."""

        await self._engine.upsert(self.blob_collection, [self._payload_point(point.blob_id, point.payload)])

    async def get_version(self, project_id: str, version_id: str) -> QdrantRecord | None:
        """Retrieve one version by id, scoped to ``project_id``."""

        records = await self._engine.retrieve(self.version_collection, [version_id])
        return self._engine.first_project_match(records, project_id)

    async def get_blob(self, project_id: str, content_hash: str) -> QdrantRecord | None:
        """Retrieve the content blob for ``(project_id, content_hash)``."""

        records, _ = await self._engine.scroll(
            self.blob_collection,
            scroll_filter=self._engine.project_filter(
                project_id, conditions=[match_value("content_hash", content_hash)]
            ),
            limit=1,
        )
        return records[0] if records else None

    async def published_head(self, project_id: str, cloud_skill_id: str) -> QdrantRecord | None:
        """Return the newest ``published`` version of a skill, or ``None``.

        One skill normally has a single published head; during gating several may
        coexist, so the most recently created one wins.
        """

        records, _ = await self._engine.scroll(
            self.version_collection,
            scroll_filter=self._engine.project_filter(
                project_id,
                conditions=[match_value("cloud_skill_id", cloud_skill_id), match_value("status", "published")],
            ),
            limit=1,
            order_by=qmodels.OrderBy(key="created_at", direction=qmodels.Direction.DESC),
        )
        return records[0] if records else None

    async def latest_version(self, project_id: str, cloud_skill_id: str) -> QdrantRecord | None:
        """Return the newest version metadata row for one cloud skill."""

        records, _ = await self._engine.scroll(
            self.version_collection,
            scroll_filter=self._engine.project_filter(
                project_id,
                conditions=[match_value("cloud_skill_id", cloud_skill_id)],
            ),
            limit=1,
            order_by=qmodels.OrderBy(key="created_at", direction=qmodels.Direction.DESC),
        )
        return records[0] if records else None

    async def list_versions(
        self,
        project_id: str,
        *,
        limit: int = 1000,
        cursor: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll all version metadata in one project (no bundle content)."""

        return await self._engine.scroll(
            self.version_collection,
            scroll_filter=self._engine.project_filter(project_id),
            limit=limit,
            offset=cursor,
            order_by=qmodels.OrderBy(key="created_at", direction=qmodels.Direction.ASC),
        )

    async def versions_since(
        self,
        project_id: str,
        cloud_skill_id: str,
        *,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[QdrantRecord]:
        """Return version metadata for a skill created after ``since`` (ascending).

        Version records never carry bundle content, so this is the incremental
        metadata feed backing ``.../versions?since=`` (design §5.4).
        """

        conditions: list[Any] = [match_value("cloud_skill_id", cloud_skill_id)]
        if since is not None:
            conditions.append(datetime_range("created_at", gt=since))
        records, _ = await self._engine.scroll(
            self.version_collection,
            scroll_filter=self._engine.project_filter(project_id, conditions=conditions),
            limit=limit,
            order_by=qmodels.OrderBy(key="created_at", direction=qmodels.Direction.ASC),
        )
        return records

    async def delete_versions(self, point_ids: list[str]) -> None:
        """Delete skill version metadata points by id."""

        await self._engine.delete(self.version_collection, point_ids)

    async def iter_lineage(self, project_id: str, version_id: str) -> list[QdrantRecord]:
        """Walk the parent chain from ``version_id`` up to the root version.

        Returned newest-first (the given version, then its parent, ...). The
        version chain is short, so following ``parent_version_id`` one hop at a
        time is acceptable (design §3).
        """

        lineage: list[QdrantRecord] = []
        seen: set[str] = set()
        current: str | None = version_id
        while current and current not in seen:
            seen.add(current)
            record = await self.get_version(project_id, current)
            if record is None:
                break
            lineage.append(record)
            current = record.payload.get("parent_version_id")
        return lineage

    async def add_pending_trace(self, point: SkillTracePendingPoint) -> None:
        """Park one trace whose content is not yet registered (design §2.1)."""

        await self._engine.upsert(
            self.trace_pending_collection, [self._payload_point(point.trace_point_id, point.payload)]
        )

    async def scroll_pending_traces(
        self,
        project_id: str,
        content_hash: str,
        *,
        limit: int = 200,
    ) -> list[QdrantRecord]:
        """Fetch all pending traces for ``(project_id, content_hash)`` (for rebind)."""

        records, _ = await self.scroll_pending_traces_page(project_id, content_hash, limit=limit)
        return records

    async def scroll_pending_traces_page(
        self,
        project_id: str,
        content_hash: str,
        *,
        base_version_id: str | None = None,
        limit: int = 200,
        cursor: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Fetch one pending-trace page for rebind."""

        conditions: list[Any] = [match_value("content_hash", content_hash)]
        if base_version_id is not None:
            conditions.append(match_value("base_version_id", base_version_id))
        records, next_cursor = await self._engine.scroll(
            self.trace_pending_collection,
            scroll_filter=self._engine.project_filter(project_id, conditions=conditions),
            limit=limit,
            offset=cursor,
        )
        return records, next_cursor

    async def delete_pending_traces(self, trace_point_ids: list[str]) -> None:
        """Delete pending trace points after they have been rebound."""

        await self._engine.delete(self.trace_pending_collection, trace_point_ids)

    async def upsert_summary(self, point: SkillTraceSummaryPoint) -> None:
        """Upsert one trajectory summary (idempotent: id == add_record_id)."""

        await self._engine.upsert(self.trace_summary_collection, [self._payload_point(point.summary_id, point.payload)])

    async def get_summary(self, project_id: str, add_record_id: str) -> QdrantRecord | None:
        """Retrieve the summary stored for one add trace, scoped to ``project_id``."""

        records = await self._engine.retrieve(self.trace_summary_collection, [add_record_id])
        return self._engine.first_project_match(records, project_id)

    async def scroll_summaries(
        self,
        project_id: str,
        cloud_skill_id: str,
        *,
        unconsumed_only: bool = False,
        limit: int = 500,
        cursor: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll trajectory summaries for one cloud skill, oldest-first.

        ``unconsumed_only`` filters to summaries not yet folded into an evolved
        version (``consumed_version_id`` empty/missing), the set that drives the
        evolution threshold.
        """

        conditions: list[Any] = [match_value("cloud_skill_id", cloud_skill_id)]
        if unconsumed_only:
            conditions.append(is_empty("consumed_version_id"))
        return await self._engine.scroll(
            self.trace_summary_collection,
            scroll_filter=self._engine.project_filter(project_id, conditions=conditions),
            limit=limit,
            offset=cursor,
            order_by=qmodels.OrderBy(key="created_at", direction=qmodels.Direction.ASC),
        )

    async def mark_summary_consumed(self, summary_id: str, version_id: str) -> None:
        """Stamp a summary with the evolved version that consumed it."""

        await self._engine.set_payload(self.trace_summary_collection, summary_id, {"consumed_version_id": version_id})

    def _payload_point(self, point_id: str, payload: dict[str, Any]) -> qmodels.PointStruct:
        return qmodels.PointStruct(id=point_id, vector={}, payload=self._engine.safe_payload(payload))
