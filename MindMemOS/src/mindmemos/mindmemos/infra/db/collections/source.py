"""Repository for the ``source_item_v1`` collection (single dense vector)."""

from __future__ import annotations

from ..models import QdrantRecord, SourcePoint
from .base import CollectionRepository


class SourceRepository(CollectionRepository):
    """Typed adapter for ``source_item_v1``."""

    @property
    def collection(self) -> str:
        return self._cfg.source_collection

    async def upsert(self, points: list[SourcePoint]) -> None:
        """Upsert many source points."""

        await self._engine.upsert(
            self.collection,
            [self._dense_point(point.source_id, point.vector, point.payload) for point in points],
        )

    async def get(self, project_id: str, source_id: str, *, with_vectors: bool = False) -> QdrantRecord | None:
        """Retrieve one source by project and id."""

        return await self._get_one_scoped(project_id, source_id, with_vectors=with_vectors)
