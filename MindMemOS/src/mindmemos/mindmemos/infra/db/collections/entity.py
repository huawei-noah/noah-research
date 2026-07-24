"""Repository for the ``entity_item_v1`` collection."""

from __future__ import annotations

from qdrant_client import models as qmodels

from ..models import EntityPoint, QdrantRecord, QdrantSearchRecord, SparseVectorData
from .base import CollectionRepository


class EntityRepository(CollectionRepository):
    """Typed adapter for ``entity_item_v1``."""

    @property
    def collection(self) -> str:
        return self._cfg.entity_collection

    async def upsert(self, points: list[EntityPoint]) -> None:
        """Upsert many entity points."""

        await self._engine.upsert(
            self.collection,
            [self._point(point) for point in points],
        )

    async def get(self, project_id: str, entity_id: str, *, with_vectors: bool = False) -> QdrantRecord | None:
        """Retrieve one entity by project and id."""

        return await self._get_one_scoped(project_id, entity_id, with_vectors=with_vectors)

    async def search_dense(
        self,
        project_id: str,
        vector: list[float],
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search entities via dense semantic vector."""

        return await self._engine.query(
            self.collection,
            source="entity_semantic",
            query=vector,
            using=self.semantic_vector_name,
            query_filter=self._engine.project_filter(project_id, filter_=filter_),
            limit=limit,
            with_payload=with_payload,
            score_threshold=score_threshold,
        )

    async def search_sparse(
        self,
        project_id: str,
        vector: SparseVectorData,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search entities via sparse BM25 vector."""

        return await self._engine.query(
            self.collection,
            source="entity_bm25",
            query=self._engine.to_qdrant_sparse(vector),
            using=self.bm25_vector_name,
            query_filter=self._engine.project_filter(project_id, filter_=filter_),
            limit=limit,
            with_payload=with_payload,
        )

    async def search_rrf(
        self,
        project_id: str,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        dense_limit: int | None = None,
        sparse_limit: int | None = None,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Run Qdrant-side RRF over dense and sparse entity prefetches."""

        scoped_filter = self._engine.project_filter(project_id, filter_=filter_)
        return await self._engine.query(
            self.collection,
            source="entity_rrf",
            prefetch=[
                qmodels.Prefetch(
                    query=self._engine.to_qdrant_sparse(sparse_vector),
                    using=self.bm25_vector_name,
                    filter=scoped_filter,
                    limit=sparse_limit or max(limit * 3, 30),
                ),
                qmodels.Prefetch(
                    query=dense_vector,
                    using=self.semantic_vector_name,
                    filter=scoped_filter,
                    limit=dense_limit or max(limit * 3, 30),
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=limit,
            with_payload=with_payload,
        )

    def _point(self, point: EntityPoint) -> qmodels.PointStruct:
        return qmodels.PointStruct(
            id=point.entity_id,
            vector=self._vectors(point),
            payload=self._engine.safe_payload(point.payload),
        )

    def _vectors(self, point: EntityPoint) -> dict[str, list[float] | qmodels.SparseVector]:
        vectors: dict[str, list[float] | qmodels.SparseVector] = {
            self.semantic_vector_name: point.vector or self._engine.zero_vector()
        }
        if point.bm25_vector is not None:
            vectors[self.bm25_vector_name] = self._engine.to_qdrant_sparse(point.bm25_vector)
        return vectors
