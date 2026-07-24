"""Memory DB boundary for the schema-add buffer collection."""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ...infra.db import DatabaseClients, QdrantRecord, SchemaAddBufferPoint, resolve_database_clients


class SchemaAddBufferStore:
    """Project-scoped schema add buffer storage operations."""

    def __init__(self, *, clients: DatabaseClients | None = None) -> None:
        self._clients = resolve_database_clients(clients)

    async def append_many(self, points: list[SchemaAddBufferPoint]) -> None:
        """Write schema buffer records in bulk."""

        await self._clients.qdrant.upsert_schema_add_buffer_records(points)

    async def list(
        self,
        project_id: str,
        *,
        filters: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """List schema buffer records for one project."""

        return await self._clients.qdrant.scroll_schema_add_buffer_records(
            project_id,
            filter_=filters,
            limit=limit,
            cursor=cursor,
            order_by=order_by,
        )

    async def list_global(
        self,
        *,
        filters: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """List schema buffer records across projects for internal background workers."""

        return await self._clients.qdrant.scroll_schema_add_buffer_records_global(
            filter_=filters,
            limit=limit,
            cursor=cursor,
            order_by=order_by,
        )

    async def get_by_ids(self, project_id: str, schema_buffer_record_ids: list[str]) -> list[QdrantRecord]:
        """Load schema buffer records by IDs with project ownership enforced."""

        return await self._clients.qdrant.get_schema_add_buffer_records_by_ids(project_id, schema_buffer_record_ids)

    async def delete_many(self, point_ids: list[str]) -> None:
        """Delete schema buffer records by id."""

        await self._clients.qdrant.delete_schema_add_buffer_records(point_ids)

    async def patch(self, project_id: str, schema_buffer_record_id: str, payload: dict[str, Any]) -> None:
        """Patch one schema buffer payload."""

        await self._clients.qdrant.patch_schema_add_buffer_record(project_id, schema_buffer_record_id, payload)
