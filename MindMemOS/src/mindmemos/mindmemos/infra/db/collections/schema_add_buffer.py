"""Repository for the ``schema_add_buffer_v1`` collection."""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ..models import QdrantRecord, SchemaAddBufferPoint
from .base import CollectionRepository


class SchemaAddBufferRepository(CollectionRepository):
    """Typed adapter for schema-add durable buffer records."""

    @property
    def collection(self) -> str:
        return self._cfg.schema_add_buffer_collection

    async def upsert(self, points: list[SchemaAddBufferPoint]) -> None:
        """Upsert many schema buffer points."""

        await self._engine.upsert(
            self.collection,
            [self._payload_point(point.schema_buffer_record_id, point.payload) for point in points],
        )

    async def get(self, project_id: str, schema_buffer_record_id: str) -> QdrantRecord | None:
        """Retrieve one schema buffer record by id, scoped to ``project_id``."""

        records = await self._engine.retrieve(self.collection, [schema_buffer_record_id])
        return self._engine.first_project_match(records, project_id)

    async def retrieve(self, project_id: str, schema_buffer_record_ids: list[str]) -> list[QdrantRecord]:
        """Retrieve many schema buffer records by id, keeping only records in ``project_id``."""

        return await self._retrieve_scoped(project_id, schema_buffer_record_ids)

    async def scroll(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll schema buffer records in one project."""

        return await self._scroll_scoped(project_id, filter_=filter_, limit=limit, cursor=cursor, order_by=order_by)

    async def scroll_global(
        self,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll schema buffer records across projects for internal workers."""

        return await self._engine.scroll(
            self.collection,
            scroll_filter=filter_,
            limit=limit,
            offset=cursor,
            order_by=order_by,
        )

    async def delete_many(self, point_ids: list[str]) -> None:
        """Delete schema buffer points by id."""

        await self._engine.delete(self.collection, point_ids)

    async def patch(self, project_id: str, schema_buffer_record_id: str, payload: dict[str, Any]) -> None:
        """Set payload fields after project ownership is checked."""

        record = await self.get(project_id, schema_buffer_record_id)
        if record is None:
            return
        await self._engine.set_payload(self.collection, schema_buffer_record_id, payload)
