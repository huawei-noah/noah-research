"""Repository for the ``add_record_v1`` collection (payload-only event log)."""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ..models import AddRecordPoint, QdrantRecord
from .base import CollectionRepository


class AddRecordRepository(CollectionRepository):
    """Typed adapter for ``add_record_v1``."""

    @property
    def collection(self) -> str:
        return self._cfg.add_record_collection

    async def upsert(self, points: list[AddRecordPoint]) -> None:
        """Upsert many add-record points."""

        await self._engine.upsert(
            self.collection,
            [self._payload_point(point.add_record_id, point.payload) for point in points],
        )

    async def get(self, project_id: str, add_record_id: str) -> QdrantRecord | None:
        """Retrieve one add-record by id, scoped to ``project_id``.

        Used by skill trace rebind to patch the ``skill_bindings`` of a specific
        trace once its skill content is registered (design §2.1).
        """

        records = await self._engine.retrieve(self.collection, [add_record_id])
        return self._engine.first_project_match(records, project_id)

    async def retrieve(self, project_id: str, add_record_ids: list[str]) -> list[QdrantRecord]:
        """Retrieve many add-records by id, keeping only records in ``project_id``."""

        return await self._retrieve_scoped(project_id, add_record_ids)

    async def scroll(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll add-record points in one project."""

        return await self._scroll_scoped(project_id, filter_=filter_, limit=limit, cursor=cursor, order_by=order_by)

    async def scroll_global(
        self,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll add-record points across projects for internal workers."""

        return await self._engine.scroll(
            self.collection,
            scroll_filter=filter_,
            limit=limit,
            offset=cursor,
            order_by=order_by,
        )

    async def patch(self, project_id: str, add_record_id: str, payload: dict[str, Any]) -> None:
        """Set payload fields after project ownership is checked."""

        record = await self.get(project_id, add_record_id)
        if record is None:
            return
        await self._engine.set_payload(self.collection, add_record_id, payload)
