"""Shared base for per-collection Qdrant repositories.

Each table under ``collections/`` binds exactly one Qdrant collection and adds
its own typed upsert/read methods. The cross-cutting mechanics — project-scoped
retrieve/scroll and point-struct building — live here so the concrete
repositories stay small and free of duplication. All low-level work is delegated
to the shared :class:`QdrantEngine`.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ....config import QdrantConfig
from ..engine import QdrantEngine
from ..models import QdrantRecord


class CollectionRepository:
    """Typed adapter bound to a single Qdrant collection."""

    def __init__(self, engine: QdrantEngine, cfg: QdrantConfig) -> None:
        self._engine = engine
        self._cfg = cfg

    @property
    def collection(self) -> str:
        """Configured collection name (bound by the subclass)."""

        raise NotImplementedError

    @property
    def semantic_vector_name(self) -> str:
        """Configured dense vector name."""

        return self._cfg.semantic_vector_name

    @property
    def bm25_vector_name(self) -> str:
        """Configured sparse vector name."""

        return self._cfg.bm25_vector_name

    async def _get_one_scoped(
        self, project_id: str, point_id: str, *, with_vectors: bool = False
    ) -> QdrantRecord | None:
        """Retrieve one point and return it only if it belongs to ``project_id``."""

        records = await self._engine.retrieve(self.collection, [point_id], with_vectors=with_vectors)
        return self._engine.first_project_match(records, project_id)

    async def _retrieve_scoped(
        self, project_id: str, point_ids: list[str], *, with_vectors: bool = False
    ) -> list[QdrantRecord]:
        """Retrieve points by id, keeping only those owned by ``project_id``."""

        records = await self._engine.retrieve(self.collection, point_ids, with_vectors=with_vectors)
        return [record for record in records if record.payload.get("project_id") == project_id]

    async def _scroll_scoped(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int,
        cursor: Any | None = None,
        order_by: Any | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll the collection inside one project."""

        return await self._engine.scroll(
            self.collection,
            scroll_filter=self._engine.project_filter(project_id, filter_=filter_),
            limit=limit,
            offset=cursor,
            order_by=order_by,
            with_vectors=with_vectors,
        )

    def _dense_point(self, point_id: str, vector: list[float] | None, payload: dict[str, Any]) -> qmodels.PointStruct:
        """Build a point carrying a single dense vector (zero-filled when absent)."""

        return qmodels.PointStruct(
            id=point_id,
            vector={self.semantic_vector_name: vector or self._engine.zero_vector()},
            payload=self._engine.safe_payload(payload),
        )

    def _payload_point(self, point_id: str, payload: dict[str, Any]) -> qmodels.PointStruct:
        """Build a vector-less point (payload + filter only)."""

        return qmodels.PointStruct(id=point_id, vector={}, payload=self._engine.safe_payload(payload))
