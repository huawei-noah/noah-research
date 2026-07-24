"""Low-level database models for Qdrant and Neo4j adapters.

The classes in this module are storage primitives only. They intentionally do
not contain memory operation semantics such as add, search, merge, feedback, or
retrieval ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

QdrantDistance = Literal["Cosine", "Euclid", "Dot", "Manhattan"]
GraphNodeLabel = Literal["Memory", "Entity", "Source"]


def utcnow() -> datetime:
    """Return the current UTC time."""

    return datetime.now(UTC)


@dataclass(kw_only=True)
class SparseVectorData:
    """Sparse vector accepted by Qdrant sparse vector fields."""

    indices: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            msg = "SparseVectorData indices and values length mismatch"
            raise ValueError(msg)


@dataclass(kw_only=True)
class PayloadIndexSpec:
    """Payload index definition for one Qdrant collection field."""

    field_name: str
    field_schema: Any


@dataclass(kw_only=True)
class QdrantCollectionSpec:
    """Collection schema required by QdrantStore.ensure_collection."""

    name: str
    vector_size: int
    dense_vector_name: str = "semantic"
    sparse_vector_name: str = "bm25"
    distance: QdrantDistance = "Cosine"
    enable_dense: bool = True
    enable_sparse: bool = False
    on_disk_payload: bool | None = None
    payload_indexes: list[PayloadIndexSpec] = field(default_factory=list)


@dataclass(kw_only=True)
class MemoryPoint:
    """Qdrant point for ``memory_item_v1``."""

    memory_id: str
    payload: dict[str, Any]
    semantic_vector: list[float] | None = None
    bm25_vector: SparseVectorData | None = None


@dataclass(kw_only=True)
class EntityPoint:
    """Qdrant point for ``entity_item_v1``."""

    entity_id: str
    payload: dict[str, Any]
    vector: list[float] | None = None
    bm25_vector: SparseVectorData | None = None


@dataclass(kw_only=True)
class SourcePoint:
    """Qdrant point for ``source_item_v1``."""

    source_id: str
    payload: dict[str, Any]
    vector: list[float] | None = None


@dataclass(kw_only=True)
class AddRecordPoint:
    """Qdrant point for ``add_record_v1``."""

    add_record_id: str
    payload: dict[str, Any]


@dataclass(kw_only=True)
class SchemaAddBufferPoint:
    """Qdrant point for ``schema_add_buffer_v1``."""

    schema_buffer_record_id: str
    payload: dict[str, Any]


@dataclass(kw_only=True)
class SearchRecordPoint:
    """Qdrant point for ``search_record_v1``."""

    search_record_id: str
    payload: dict[str, Any]


@dataclass(kw_only=True)
class SkillVersionPoint:
    """Qdrant point for ``skill_version_v1``.

    ``version_id`` is the point id, derived deterministically from
    ``(project_id, content_hash, parent_version_id)`` so repeated registration of
    the same key upserts the same point (idempotency, design §3 / §5.2).
    """

    version_id: str
    payload: dict[str, Any]


@dataclass(kw_only=True)
class SkillBlobPoint:
    """Qdrant point for ``skill_blob_v1``.

    ``blob_id`` is the point id, derived from ``(project_id, content_hash)`` so
    identical content is stored once per project.
    """

    blob_id: str
    payload: dict[str, Any]


@dataclass(kw_only=True)
class SkillTracePendingPoint:
    """Qdrant point for ``skill_trace_pending_v1`` (random ``uuid4`` point id)."""

    trace_point_id: str
    payload: dict[str, Any]


@dataclass(kw_only=True)
class SkillTraceSummaryPoint:
    """Qdrant point for ``skill_trace_summary_v1``.

    ``summary_id`` is the point id and equals the originating ``add_record_id``,
    so each injected trace has at most one stored summary (re-summarizing
    overwrites).
    """

    summary_id: str
    payload: dict[str, Any]


@dataclass(kw_only=True)
class QdrantRecord:
    """Point returned by Qdrant retrieve or scroll operations."""

    point_id: str
    payload: dict[str, Any]
    vectors: dict[str, Any] | None = None


@dataclass(kw_only=True)
class QdrantSearchRecord:
    """Search hit returned by QdrantStore without business DTO hydration."""

    point_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    vectors: dict[str, Any] | None = None
    source: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class NodeRef:
    """Reference to an existing Neo4j node."""

    label: GraphNodeLabel
    key: dict[str, Any]


@dataclass(kw_only=True)
class MemoryNode:
    """Neo4j ``Memory`` node upsert input."""

    project_id: str
    memory_id: str
    content: str

    @property
    def ref(self) -> NodeRef:
        """Return the canonical Neo4j node reference."""

        return NodeRef(label="Memory", key={"project_id": self.project_id, "memory_id": self.memory_id})


@dataclass(kw_only=True)
class EntityNode:
    """Neo4j ``Entity`` node upsert input."""

    project_id: str
    entity_id: str
    entity_name: str
    entity_type: str | None = None
    description: str | None = None

    @property
    def ref(self) -> NodeRef:
        """Return the canonical Neo4j node reference."""

        return NodeRef(label="Entity", key={"project_id": self.project_id, "entity_id": self.entity_id})


@dataclass(kw_only=True)
class SourceNode:
    """Neo4j ``Source`` node upsert input."""

    project_id: str
    source_id: str
    parsed_content_path: str | None = None

    @property
    def ref(self) -> NodeRef:
        """Return the canonical Neo4j node reference."""

        return NodeRef(label="Source", key={"project_id": self.project_id, "source_id": self.source_id})


@dataclass(kw_only=True)
class GraphRelationship:
    """Neo4j relationship upsert input."""

    source: NodeRef
    target: NodeRef
    rel_type: str
    key: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
