"""Repository for the ``search_record_v1`` collection (payload-only event log)."""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ..models import QdrantRecord, SearchRecordPoint
from .base import CollectionRepository


class SearchRecordRepository(CollectionRepository):
    """Typed adapter for ``search_record_v1``."""

    @property
    def collection(self) -> str:
        return self._cfg.search_record_collection

    async def upsert(self, points: list[SearchRecordPoint]) -> None:
        """Upsert many search-record points."""

        await self._engine.upsert(
            self.collection,
            [self._payload_point(point.search_record_id, point.payload) for point in points],
        )

    async def scroll(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll search-record points in one project."""

        return await self._scroll_scoped(project_id, filter_=filter_, limit=limit, cursor=cursor, order_by=order_by)
