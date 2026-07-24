"""Memory DB boundary for add-record collection access."""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ...infra.db import AddRecordPoint, DatabaseClients, QdrantRecord, resolve_database_clients


class AddRecordStore:
    """Project-scoped add-record storage operations."""

    def __init__(self, *, clients: DatabaseClients | None = None) -> None:
        self._clients = resolve_database_clients(clients)

    async def append(self, point: AddRecordPoint) -> None:
        """Write one add record."""

        await self.append_many([point])

    async def append_many(self, points: list[AddRecordPoint]) -> None:
        """Write add records in bulk."""

        await self._clients.qdrant.upsert_add_records(points)

    async def list(
        self,
        project_id: str,
        *,
        filters: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """List add records for one project."""

        return await self._clients.qdrant.scroll_add_records(
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
        """List add records across projects for internal background workers."""

        return await self._clients.qdrant.scroll_add_records_global(
            filter_=filters,
            limit=limit,
            cursor=cursor,
            order_by=order_by,
        )

    async def get_by_ids(self, project_id: str, add_record_ids: list[str]) -> list[QdrantRecord]:
        """Load add records by IDs with project ownership enforced."""

        return await self._clients.qdrant.get_add_records_by_ids(project_id, add_record_ids)

    async def patch(self, project_id: str, add_record_id: str, payload: dict[str, Any]) -> None:
        """Patch one add-record payload."""

        await self._clients.qdrant.patch_add_record(project_id, add_record_id, payload)
