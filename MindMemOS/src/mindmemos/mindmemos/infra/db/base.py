"""Protocols for low-level database adapters."""

from __future__ import annotations

from typing import Any, Protocol

from .models import (
    AddRecordPoint,
    EntityNode,
    EntityPoint,
    GraphRelationship,
    MemoryNode,
    MemoryPoint,
    NodeRef,
    QdrantCollectionSpec,
    QdrantRecord,
    QdrantSearchRecord,
    SchemaAddBufferPoint,
    SearchRecordPoint,
    SourceNode,
    SourcePoint,
    SparseVectorData,
)


class QdrantStore(Protocol):
    """Qdrant primitive operations for the documented memory tables."""

    async def ensure_schema(self) -> None:
        """Create all Qdrant collections and payload indexes."""

    async def ensure_collection(self, spec: QdrantCollectionSpec) -> None:
        """Create one collection and payload indexes."""

    async def upsert_memory(self, point: MemoryPoint) -> None:
        """Upsert one memory point."""

    async def upsert_memories(self, points: list[MemoryPoint]) -> None:
        """Upsert memory points."""

    async def upsert_entity(self, point: EntityPoint) -> None:
        """Upsert one entity point."""

    async def upsert_entities(self, points: list[EntityPoint]) -> None:
        """Upsert entity points."""

    async def upsert_source(self, point: SourcePoint) -> None:
        """Upsert one source point."""

    async def upsert_sources(self, points: list[SourcePoint]) -> None:
        """Upsert source points."""

    async def upsert_add_record(self, point: AddRecordPoint) -> None:
        """Upsert one add record point."""

    async def upsert_add_records(self, points: list[AddRecordPoint]) -> None:
        """Upsert add record points."""

    async def patch_add_record(self, project_id: str, add_record_id: str, payload: dict[str, Any]) -> None:
        """Patch add-record payload fields after project ownership is checked."""

    async def upsert_schema_add_buffer_records(self, points: list[SchemaAddBufferPoint]) -> None:
        """Upsert schema add buffer points."""

    async def patch_schema_add_buffer_record(
        self, project_id: str, schema_buffer_record_id: str, payload: dict[str, Any]
    ) -> None:
        """Patch schema buffer payload fields after project ownership is checked."""

    async def upsert_search_record(self, point: SearchRecordPoint) -> None:
        """Upsert one search record point."""

    async def upsert_search_records(self, points: list[SearchRecordPoint]) -> None:
        """Upsert search record points."""

    async def get_memory(self, project_id: str, memory_id: str, *, with_vectors: bool = False) -> QdrantRecord | None:
        """Retrieve one memory."""

    async def get_memories(
        self,
        project_id: str,
        memory_ids: list[str],
        *,
        with_vectors: bool = False,
    ) -> list[QdrantRecord]:
        """Retrieve memories."""

    async def get_entity(self, project_id: str, entity_id: str, *, with_vectors: bool = False) -> QdrantRecord | None:
        """Retrieve one entity."""

    async def search_entity_dense(
        self,
        project_id: str,
        vector: list[float],
        *,
        filter_: Any | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search entities by dense vector."""

    async def search_entity_sparse(
        self,
        project_id: str,
        vector: Any,
        *,
        filter_: Any | None = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search entities by sparse BM25 vector."""

    async def search_entity_hybrid(
        self,
        project_id: str,
        dense_vector: list[float],
        sparse_vector: Any,
        *,
        filter_: Any | None = None,
        limit: int = 10,
        dense_limit: int | None = None,
        sparse_limit: int | None = None,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search entities by Qdrant-side dense/sparse hybrid fusion."""

    async def search_memory_dense(
        self,
        project_id: str,
        vector: list[float],
        *,
        filter_: Any | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search memories by dense vector."""

    async def search_memory_sparse(
        self,
        project_id: str,
        vector: SparseVectorData,
        *,
        filter_: Any | None = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search memories by sparse vector."""

    async def search_memory_hybrid(
        self,
        project_id: str,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        *,
        filter_: Any | None = None,
        limit: int = 10,
        dense_limit: int | None = None,
        sparse_limit: int | None = None,
        with_payload: bool = True,
    ) -> list[QdrantSearchRecord]:
        """Search memories by Qdrant-side dense/sparse hybrid fusion."""

    async def update_memory_payload(self, project_id: str, memory_id: str, payload: dict[str, Any]) -> None:
        """Set memory payload fields."""

    async def delete_memory(self, project_id: str, memory_id: str) -> None:
        """Delete one memory point."""

    async def scroll_memories(
        self,
        project_id: str,
        *,
        filter_: Any | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll memories."""

    async def scroll_add_records(
        self,
        project_id: str,
        *,
        filter_: Any | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll add record points."""

    async def get_add_records_by_ids(self, project_id: str, add_record_ids: list[str]) -> list[QdrantRecord]:
        """Retrieve add records by IDs."""

    async def scroll_add_records_global(
        self,
        *,
        filter_: Any | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll add records across projects for internal background workers."""

    async def scroll_schema_add_buffer_records(
        self,
        project_id: str,
        *,
        filter_: Any | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll schema add buffer records."""

    async def get_schema_add_buffer_records_by_ids(
        self, project_id: str, schema_buffer_record_ids: list[str]
    ) -> list[QdrantRecord]:
        """Retrieve schema add buffer records by IDs."""

    async def scroll_schema_add_buffer_records_global(
        self,
        *,
        filter_: Any | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll schema add buffer records across projects for internal background workers."""

    async def scroll_search_records(
        self,
        project_id: str,
        *,
        filter_: Any | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll search record points."""

    async def close(self) -> None:
        """Close the underlying client."""


class Neo4jStore(Protocol):
    """Neo4j primitive operations for graph mirror writes."""

    async def ensure_schema(self, statements: list[str] | None = None) -> None:
        """Create constraints and indexes."""

    async def upsert_memory_node(self, node: MemoryNode) -> None:
        """Upsert one memory node."""

    async def upsert_entity_node(self, node: EntityNode) -> None:
        """Upsert one entity node."""

    async def upsert_source_node(self, node: SourceNode) -> None:
        """Upsert one source node."""

    async def upsert_nodes(
        self,
        *,
        memories: list[MemoryNode] | None = None,
        entities: list[EntityNode] | None = None,
        sources: list[SourceNode] | None = None,
    ) -> None:
        """Upsert graph nodes in batches by node label."""

    async def upsert_relationship(self, rel: GraphRelationship) -> None:
        """Upsert one graph relationship."""

    async def upsert_relationships(self, relationships: list[GraphRelationship]) -> None:
        """Upsert graph relationships in batches by relationship shape."""

    async def delete_memory_node(self, project_id: str, memory_id: str, *, detach: bool = True) -> None:
        """Delete one memory node."""

    async def archive_memory_node(self, project_id: str, memory_id: str, *, reason: str | None = None) -> None:
        """Mark one memory node as archived."""

    async def get_related_memory_ids(
        self,
        project_id: str,
        memory_ids: list[str],
        *,
        limit_per_memory: int = 3,
        max_candidates: int = 10,
    ) -> list[dict[str, Any]]:
        """Return one-hop ``RELATES_TO`` memory neighbors for seed memories."""

    async def get_entity_neighbors(
        self,
        project_id: str,
        entity_id: str,
        *,
        direction: str = "both",
        rel_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return bounded one-hop entity neighbors."""

    async def delete_node(self, ref: NodeRef, *, detach: bool = True) -> None:
        """Delete one node."""

    async def run_read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Run a read query and return record dictionaries."""

    async def get_memory_lineage(self, project_id: str, memory_ids: list[str]) -> list[dict[str, Any]]:
        """Return memory_id -> ancestor IDs via outgoing DERIVED_FROM traversal."""

    async def close(self) -> None:
        """Close the underlying driver."""



class AddRecordStore(Protocol):
    """Low-level operations for the durable add-record collection."""

    async def upsert_add_record(self, point: AddRecordPoint) -> None:
        """Upsert one add record point."""

    async def upsert_add_records(self, points: list[AddRecordPoint]) -> None:
        """Upsert add record points."""

    async def patch_add_record(self, project_id: str, add_record_id: str, payload: dict[str, Any]) -> None:
        """Patch add-record payload fields after project ownership is checked."""

    async def scroll_add_records(
        self,
        project_id: str,
        *,
        filter_: Any | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll add records."""

    async def get_add_records_by_ids(self, project_id: str, add_record_ids: list[str]) -> list[QdrantRecord]:
        """Retrieve add records by IDs."""

    async def scroll_add_records_global(
        self,
        *,
        filter_: Any | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll add records across projects for internal background workers."""
