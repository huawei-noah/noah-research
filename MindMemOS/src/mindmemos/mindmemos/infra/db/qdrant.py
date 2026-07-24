"""Qdrant facade for the documented MindMemOS memory tables.

``QdrantStore`` owns the shared :class:`QdrantEngine` and composes one
per-collection repository for each memory table (``memory_item_v1``,
``entity_item_v1``, ``source_item_v1``, ``add_record_v1``, ``search_record_v1``).
Its flat methods delegate straight to those repositories, so callers keep a
single entry point while each table's logic lives in its own module under
``collections/``. The repositories are also reachable directly via the
``memory``/``entity``/``source``/``add_record``/``search_record`` properties.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from ...config import QdrantConfig
from ...logging import get_logger
from .collections import (
    AddRecordRepository,
    EntityRepository,
    MemoryRepository,
    SchemaAddBufferRepository,
    SearchRecordRepository,
    SourceRepository,
)
from .engine import QdrantEngine
from .models import (
    AddRecordPoint,
    EntityPoint,
    MemoryPoint,
    QdrantCollectionSpec,
    QdrantRecord,
    QdrantSearchRecord,
    SchemaAddBufferPoint,
    SearchRecordPoint,
    SourcePoint,
    SparseVectorData,
)
from .schema import all_collection_specs

logger = get_logger(__name__)


class QdrantStore:
    """Thin Qdrant adapter with no memory business logic."""

    def __init__(self, cfg: QdrantConfig, *, client: AsyncQdrantClient | None = None) -> None:
        self._cfg = cfg
        self._engine = QdrantEngine(cfg, client=client)
        self._memory = MemoryRepository(self._engine, cfg)
        self._entity = EntityRepository(self._engine, cfg)
        self._source = SourceRepository(self._engine, cfg)
        self._add_record = AddRecordRepository(self._engine, cfg)
        self._schema_add_buffer = SchemaAddBufferRepository(self._engine, cfg)
        self._search_record = SearchRecordRepository(self._engine, cfg)

    @property
    def engine(self) -> QdrantEngine:
        """Shared engine, reused by other adapters on the same Qdrant database."""

        return self._engine

    @property
    def client(self) -> AsyncQdrantClient:
        """Underlying async Qdrant client (owned and closed by this store)."""

        return self._engine.client

    @property
    def memory(self) -> MemoryRepository:
        """Repository for ``memory_item_v1``."""

        return self._memory

    @property
    def entity(self) -> EntityRepository:
        """Repository for ``entity_item_v1``."""

        return self._entity

    @property
    def source(self) -> SourceRepository:
        """Repository for ``source_item_v1``."""

        return self._source

    @property
    def add_record(self) -> AddRecordRepository:
        """Repository for ``add_record_v1``."""

        return self._add_record

    @property
    def schema_add_buffer(self) -> SchemaAddBufferRepository:
        """Repository for ``schema_add_buffer_v1``."""

        return self._schema_add_buffer

    @property
    def search_record(self) -> SearchRecordRepository:
        """Repository for ``search_record_v1``."""

        return self._search_record

    @property
    def memory_collection(self) -> str:
        """Configured ``memory_item_v1`` collection name."""

        return self._cfg.memory_collection

    @property
    def entity_collection(self) -> str:
        """Configured ``entity_item_v1`` collection name."""

        return self._cfg.entity_collection

    @property
    def source_collection(self) -> str:
        """Configured ``source_item_v1`` collection name."""

        return self._cfg.source_collection

    @property
    def add_record_collection(self) -> str:
        """Configured ``add_record_v1`` collection name."""

        return self._cfg.add_record_collection

    @property
    def schema_add_buffer_collection(self) -> str:
        """Configured ``schema_add_buffer_v1`` collection name."""

        return self._cfg.schema_add_buffer_collection

    @property
    def search_record_collection(self) -> str:
        """Configured ``search_record_v1`` collection name."""

        return self._cfg.search_record_collection

    @property
    def semantic_vector_name(self) -> str:
        """Configured dense vector name."""

        return self._cfg.semantic_vector_name

    @property
    def bm25_vector_name(self) -> str:
        """Configured sparse vector name."""

        return self._cfg.bm25_vector_name

    async def ensure_schema(self) -> None:
        """Create all Qdrant collections and payload indexes."""

        if not self._cfg.auto_create:
            return
        for spec in all_collection_specs(self._cfg):
            await self._engine.ensure_collection(spec)

    async def ensure_collection(self, spec: QdrantCollectionSpec) -> None:
        """Create one collection and its payload indexes."""

        if not self._cfg.auto_create:
            return
        await self._engine.ensure_collection(spec)

    async def upsert_memory(self, point: MemoryPoint) -> None:
        """Upsert one point into ``memory_item_v1``."""

        await self._memory.upsert([point])

    async def upsert_memories(self, points: list[MemoryPoint]) -> None:
        """Upsert many points into ``memory_item_v1``."""

        await self._memory.upsert(points)

    async def upsert_entity(self, point: EntityPoint) -> None:
        """Upsert one point into ``entity_item_v1``."""

        await self._entity.upsert([point])

    async def upsert_entities(self, points: list[EntityPoint]) -> None:
        """Upsert many points into ``entity_item_v1``."""

        await self._entity.upsert(points)

    async def upsert_source(self, point: SourcePoint) -> None:
        """Upsert one point into ``source_item_v1``."""

        await self._source.upsert([point])

    async def upsert_sources(self, points: list[SourcePoint]) -> None:
        """Upsert many points into ``source_item_v1``."""

        await self._source.upsert(points)

    async def upsert_add_record(self, point: AddRecordPoint) -> None:
        """Upsert one point into ``add_record_v1``."""

        await self._add_record.upsert([point])

    async def upsert_add_records(self, points: list[AddRecordPoint]) -> None:
        """Upsert many points into ``add_record_v1``."""

        await self._add_record.upsert(points)

    async def patch_add_record(self, project_id: str, add_record_id: str, payload: dict[str, Any]) -> None:
        """Set payload fields for one add record after project ownership is checked."""

        await self._add_record.patch(project_id, add_record_id, payload)

    async def upsert_schema_add_buffer_records(self, points: list[SchemaAddBufferPoint]) -> None:
        """Upsert many points into ``schema_add_buffer_v1``."""

        await self._schema_add_buffer.upsert(points)

    async def patch_schema_add_buffer_record(
        self, project_id: str, schema_buffer_record_id: str, payload: dict[str, Any]
    ) -> None:
        """Set payload fields for one schema buffer record after project ownership is checked."""

        await self._schema_add_buffer.patch(project_id, schema_buffer_record_id, payload)

    async def delete_schema_add_buffer_records(self, point_ids: list[str]) -> None:
        """Delete points from ``schema_add_buffer_v1`` by id."""

        await self._schema_add_buffer.delete_many(point_ids)

    async def upsert_search_record(self, point: SearchRecordPoint) -> None:
        """Upsert one point into ``search_record_v1``."""

        await self._search_record.upsert([point])

    async def upsert_search_records(self, points: list[SearchRecordPoint]) -> None:
        """Upsert many points into ``search_record_v1``."""

        await self._search_record.upsert(points)

    async def get_memory(self, project_id: str, memory_id: str, *, with_vectors: bool = False) -> QdrantRecord | None:
        """Retrieve one memory by project and memory id."""

        return await self._memory.get(project_id, memory_id, with_vectors=with_vectors)

    async def get_memories(
        self,
        project_id: str,
        memory_ids: list[str],
        *,
        with_vectors: bool = False,
    ) -> list[QdrantRecord]:
        """Retrieve memories by project and ids."""

        return await self._memory.get_many(project_id, memory_ids, with_vectors=with_vectors)

    async def get_entity(self, project_id: str, entity_id: str, *, with_vectors: bool = False) -> QdrantRecord | None:
        """Retrieve one entity by project and entity id."""

        return await self._entity.get(project_id, entity_id, with_vectors=with_vectors)

    async def get_source(self, project_id: str, source_id: str, *, with_vectors: bool = False) -> QdrantRecord | None:
        """Retrieve one source by project and source id."""

        return await self._source.get(project_id, source_id, with_vectors=with_vectors)

    async def search_entity_dense(
        self,
        project_id: str,
        vector: list[float],
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search ``entity_item_v1`` via dense semantic vector."""

        return await self._entity.search_dense(
            project_id,
            vector,
            filter_=filter_,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=with_payload,
        )

    async def search_entity_sparse(
        self,
        project_id: str,
        vector: SparseVectorData,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search ``entity_item_v1`` via sparse BM25 vector."""

        return await self._entity.search_sparse(
            project_id, vector, filter_=filter_, limit=limit, with_payload=with_payload
        )

    async def search_entity_rrf(
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

        return await self._entity.search_rrf(
            project_id,
            dense_vector,
            sparse_vector,
            filter_=filter_,
            limit=limit,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
            with_payload=with_payload,
        )

    async def search_entity_hybrid(
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
        """Backward-compatible alias for Qdrant-side entity RRF search."""

        return await self.search_entity_rrf(
            project_id,
            dense_vector,
            sparse_vector,
            filter_=filter_,
            limit=limit,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
            with_payload=with_payload,
        )

    async def search_memory_dense(
        self,
        project_id: str,
        vector: list[float],
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search ``memory_item_v1`` via dense semantic vector."""

        return await self._memory.search_dense(
            project_id,
            vector,
            filter_=filter_,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=with_payload,
        )

    async def search_memory_sparse(
        self,
        project_id: str,
        vector: SparseVectorData,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search ``memory_item_v1`` via sparse BM25 vector."""

        return await self._memory.search_sparse(
            project_id, vector, filter_=filter_, limit=limit, with_payload=with_payload
        )

    async def search_memory_rrf(
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
        """Run Qdrant-side RRF over dense and sparse memory prefetches."""

        return await self._memory.search_rrf(
            project_id,
            dense_vector,
            sparse_vector,
            filter_=filter_,
            limit=limit,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
            with_payload=with_payload,
        )

    async def search_memory_hybrid(
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
        """Backward-compatible alias for Qdrant-side RRF hybrid search."""

        return await self.search_memory_rrf(
            project_id,
            dense_vector,
            sparse_vector,
            filter_=filter_,
            limit=limit,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
            with_payload=with_payload,
        )

    async def update_memory_payload(self, project_id: str, memory_id: str, payload: dict[str, Any]) -> None:
        """Set payload fields for one memory after project ownership is checked."""

        await self._memory.update_payload(project_id, memory_id, payload)

    async def patch_memory(
        self,
        project_id: str,
        memory_id: str,
        payload: dict[str, Any],
        *,
        dense_vector: list[float] | None = None,
        sparse_vector: SparseVectorData | None = None,
        record: QdrantRecord | None = None,
    ) -> None:
        """Apply a payload patch and optional vectors in one request."""

        await self._memory.patch(
            project_id,
            memory_id,
            payload,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            record=record,
        )

    async def delete_memory(self, project_id: str, memory_id: str) -> None:
        """Delete one memory point after project ownership is checked."""

        await self._memory.delete(project_id, memory_id)

    async def scroll_memories(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll memories in one project."""

        return await self._memory.scroll(
            project_id, filter_=filter_, limit=limit, cursor=cursor, with_vectors=with_vectors
        )

    async def scroll_add_records(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll ``add_record_v1`` points in one project."""

        return await self._add_record.scroll(project_id, filter_=filter_, limit=limit, cursor=cursor, order_by=order_by)

    async def get_add_records_by_ids(self, project_id: str, add_record_ids: list[str]) -> list[QdrantRecord]:
        """Retrieve add-record points by id inside one project."""

        return await self._add_record.retrieve(project_id, add_record_ids)

    async def scroll_add_records_global(
        self,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll ``add_record_v1`` points across projects for internal workers."""

        return await self._add_record.scroll_global(filter_=filter_, limit=limit, cursor=cursor, order_by=order_by)

    async def scroll_schema_add_buffer_records(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll ``schema_add_buffer_v1`` points in one project."""

        return await self._schema_add_buffer.scroll(
            project_id, filter_=filter_, limit=limit, cursor=cursor, order_by=order_by
        )

    async def get_schema_add_buffer_records_by_ids(
        self, project_id: str, schema_buffer_record_ids: list[str]
    ) -> list[QdrantRecord]:
        """Retrieve schema buffer points by id inside one project."""

        return await self._schema_add_buffer.retrieve(project_id, schema_buffer_record_ids)

    async def scroll_schema_add_buffer_records_global(
        self,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll ``schema_add_buffer_v1`` points across projects for internal workers."""

        return await self._schema_add_buffer.scroll_global(
            filter_=filter_, limit=limit, cursor=cursor, order_by=order_by
        )

    async def scroll_search_records(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll ``search_record_v1`` points in one project."""

        return await self._search_record.scroll(
            project_id, filter_=filter_, limit=limit, cursor=cursor, order_by=order_by
        )

    async def close(self) -> None:
        """Close the underlying client."""

        await self._engine.close()

    @staticmethod
    def _with_project_filter(project_id: str, filter_: qmodels.Filter | None) -> qmodels.Filter:
        """Backward-compatible project filter helper."""

        return QdrantEngine.project_filter(project_id, filter_=filter_)
