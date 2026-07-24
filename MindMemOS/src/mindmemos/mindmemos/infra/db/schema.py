"""Storage schema definitions derived from the MindMemOS v2 design."""

from __future__ import annotations

from ...config import QdrantConfig
from .filters import (
    ADD_RECORD_PAYLOAD_INDEX_SCHEMA,
    ENTITY_PAYLOAD_INDEX_SCHEMA,
    MEMORY_PAYLOAD_INDEX_SCHEMA,
    SCHEMA_ADD_BUFFER_PAYLOAD_INDEX_SCHEMA,
    SEARCH_RECORD_PAYLOAD_INDEX_SCHEMA,
    SKILL_BLOB_PAYLOAD_INDEX_SCHEMA,
    SKILL_TRACE_PENDING_PAYLOAD_INDEX_SCHEMA,
    SKILL_TRACE_SUMMARY_PAYLOAD_INDEX_SCHEMA,
    SKILL_VERSION_PAYLOAD_INDEX_SCHEMA,
    SOURCE_PAYLOAD_INDEX_SCHEMA,
)
from .models import QdrantCollectionSpec

MEMORY_COLLECTION = "memory_item_v1"
ENTITY_COLLECTION = "entity_item_v1"
SOURCE_COLLECTION = "source_item_v1"
ADD_RECORD_COLLECTION = "add_record_v1"
SCHEMA_ADD_BUFFER_COLLECTION = "schema_add_buffer_v1"
SEARCH_RECORD_COLLECTION = "search_record_v1"
SKILL_VERSION_COLLECTION = "skill_version_v1"
SKILL_BLOB_COLLECTION = "skill_blob_v1"
SKILL_TRACE_PENDING_COLLECTION = "skill_trace_pending_v1"
SKILL_TRACE_SUMMARY_COLLECTION = "skill_trace_summary_v1"


def memory_collection_spec(cfg: QdrantConfig) -> QdrantCollectionSpec:
    """Return the Qdrant spec for ``memory_item_v1``."""

    return QdrantCollectionSpec(
        name=cfg.memory_collection,
        vector_size=cfg.vector_size,
        dense_vector_name=cfg.semantic_vector_name,
        sparse_vector_name=cfg.bm25_vector_name,
        distance=cfg.distance,  # type: ignore[arg-type]
        enable_dense=True,
        enable_sparse=True,
        on_disk_payload=cfg.memory_on_disk_payload,
        payload_indexes=list(MEMORY_PAYLOAD_INDEX_SCHEMA),
    )


def entity_collection_spec(cfg: QdrantConfig) -> QdrantCollectionSpec:
    """Return the Qdrant spec for ``entity_item_v1``."""

    return QdrantCollectionSpec(
        name=cfg.entity_collection,
        vector_size=cfg.vector_size,
        dense_vector_name=cfg.semantic_vector_name,
        sparse_vector_name=cfg.bm25_vector_name,
        distance=cfg.distance,  # type: ignore[arg-type]
        enable_dense=True,
        enable_sparse=True,
        payload_indexes=list(ENTITY_PAYLOAD_INDEX_SCHEMA),
    )


def source_collection_spec(cfg: QdrantConfig) -> QdrantCollectionSpec:
    """Return the Qdrant spec for ``source_item_v1``."""

    return QdrantCollectionSpec(
        name=cfg.source_collection,
        vector_size=cfg.vector_size,
        dense_vector_name=cfg.semantic_vector_name,
        sparse_vector_name=cfg.bm25_vector_name,
        distance=cfg.distance,  # type: ignore[arg-type]
        enable_dense=True,
        enable_sparse=False,
        payload_indexes=list(SOURCE_PAYLOAD_INDEX_SCHEMA),
    )


def add_record_collection_spec(cfg: QdrantConfig) -> QdrantCollectionSpec:
    """Return the Qdrant spec for ``add_record_v1``."""

    return QdrantCollectionSpec(
        name=cfg.add_record_collection,
        vector_size=cfg.vector_size,
        dense_vector_name=cfg.semantic_vector_name,
        sparse_vector_name=cfg.bm25_vector_name,
        distance=cfg.distance,  # type: ignore[arg-type]
        enable_dense=False,
        enable_sparse=False,
        payload_indexes=list(ADD_RECORD_PAYLOAD_INDEX_SCHEMA),
    )


def schema_add_buffer_collection_spec(cfg: QdrantConfig) -> QdrantCollectionSpec:
    """Return the Qdrant spec for ``schema_add_buffer_v1``."""

    return QdrantCollectionSpec(
        name=cfg.schema_add_buffer_collection,
        vector_size=cfg.vector_size,
        dense_vector_name=cfg.semantic_vector_name,
        sparse_vector_name=cfg.bm25_vector_name,
        distance=cfg.distance,  # type: ignore[arg-type]
        enable_dense=False,
        enable_sparse=False,
        payload_indexes=list(SCHEMA_ADD_BUFFER_PAYLOAD_INDEX_SCHEMA),
    )


def search_record_collection_spec(cfg: QdrantConfig) -> QdrantCollectionSpec:
    """Return the Qdrant spec for ``search_record_v1``."""

    return QdrantCollectionSpec(
        name=cfg.search_record_collection,
        vector_size=cfg.vector_size,
        dense_vector_name=cfg.semantic_vector_name,
        sparse_vector_name=cfg.bm25_vector_name,
        distance=cfg.distance,  # type: ignore[arg-type]
        enable_dense=False,
        enable_sparse=False,
        payload_indexes=list(SEARCH_RECORD_PAYLOAD_INDEX_SCHEMA),
    )


def skill_version_collection_spec(cfg: QdrantConfig) -> QdrantCollectionSpec:
    """Return the Qdrant spec for ``skill_version_v1``.

    Payload + filter only: no dense or sparse vectors (design §3 physical model).
    """

    return QdrantCollectionSpec(
        name=cfg.skill_version_collection,
        vector_size=cfg.vector_size,
        dense_vector_name=cfg.semantic_vector_name,
        sparse_vector_name=cfg.bm25_vector_name,
        distance=cfg.distance,  # type: ignore[arg-type]
        enable_dense=False,
        enable_sparse=False,
        payload_indexes=list(SKILL_VERSION_PAYLOAD_INDEX_SCHEMA),
    )


def skill_blob_collection_spec(cfg: QdrantConfig) -> QdrantCollectionSpec:
    """Return the Qdrant spec for ``skill_blob_v1`` (payload + filter only)."""

    return QdrantCollectionSpec(
        name=cfg.skill_blob_collection,
        vector_size=cfg.vector_size,
        dense_vector_name=cfg.semantic_vector_name,
        sparse_vector_name=cfg.bm25_vector_name,
        distance=cfg.distance,  # type: ignore[arg-type]
        enable_dense=False,
        enable_sparse=False,
        payload_indexes=list(SKILL_BLOB_PAYLOAD_INDEX_SCHEMA),
    )


def skill_trace_pending_collection_spec(cfg: QdrantConfig) -> QdrantCollectionSpec:
    """Return the Qdrant spec for ``skill_trace_pending_v1`` (payload + filter only)."""

    return QdrantCollectionSpec(
        name=cfg.skill_trace_pending_collection,
        vector_size=cfg.vector_size,
        dense_vector_name=cfg.semantic_vector_name,
        sparse_vector_name=cfg.bm25_vector_name,
        distance=cfg.distance,  # type: ignore[arg-type]
        enable_dense=False,
        enable_sparse=False,
        payload_indexes=list(SKILL_TRACE_PENDING_PAYLOAD_INDEX_SCHEMA),
    )


def skill_trace_summary_collection_spec(cfg: QdrantConfig) -> QdrantCollectionSpec:
    """Return the Qdrant spec for ``skill_trace_summary_v1`` (payload + filter only).

    Holds trajectory summaries that feed skill self-evolution; like the other
    skill collections it carries no dense or sparse vectors.
    """

    return QdrantCollectionSpec(
        name=cfg.skill_trace_summary_collection,
        vector_size=cfg.vector_size,
        dense_vector_name=cfg.semantic_vector_name,
        sparse_vector_name=cfg.bm25_vector_name,
        distance=cfg.distance,  # type: ignore[arg-type]
        enable_dense=False,
        enable_sparse=False,
        payload_indexes=list(SKILL_TRACE_SUMMARY_PAYLOAD_INDEX_SCHEMA),
    )


def all_collection_specs(cfg: QdrantConfig) -> list[QdrantCollectionSpec]:
    """Return all configured Qdrant collection specs."""

    return [
        memory_collection_spec(cfg),
        entity_collection_spec(cfg),
        source_collection_spec(cfg),
        add_record_collection_spec(cfg),
        schema_add_buffer_collection_spec(cfg),
        search_record_collection_spec(cfg),
        skill_version_collection_spec(cfg),
        skill_blob_collection_spec(cfg),
        skill_trace_pending_collection_spec(cfg),
        skill_trace_summary_collection_spec(cfg),
    ]


NEO4J_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE CONSTRAINT memory_pk IF NOT EXISTS
    FOR (m:Memory)
    REQUIRE (m.project_id, m.memory_id) IS UNIQUE
    """,
    """
    CREATE CONSTRAINT entity_pk IF NOT EXISTS
    FOR (e:Entity)
    REQUIRE (e.project_id, e.entity_id) IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_pk IF NOT EXISTS
    FOR (s:Source)
    REQUIRE (s.project_id, s.source_id) IS UNIQUE
    """,
)
