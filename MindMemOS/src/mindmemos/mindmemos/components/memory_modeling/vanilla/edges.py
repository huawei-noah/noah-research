"""Graph relationship builders for Memory / Entity / Source edges.

Produces ``GraphRelationship`` DTOs for the four standard relationship types
used across add, update, merge, and dreaming pipelines. Each builder is a
pure function with no side effects.
"""

from __future__ import annotations

from ....typing import (
    REL_EXTRACTED_FROM,
    REL_MENTIONED_IN_SOURCE,
    REL_MENTIONS,
    REL_RELATES_TO,
    Entity,
    GraphNodeRef,
    GraphRelationship,
    MemoryRequestContext,
    SourceRef,
)
from ...chunker import SourceAwareSegment


def build_mentions_edge(
    memory_id: str,
    entity_id: str,
    entity: Entity,
    context: MemoryRequestContext,
) -> GraphRelationship:
    """Build a Memory --MENTIONS--> Entity relationship."""
    return GraphRelationship(
        source=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=memory_id),
        target=GraphNodeRef(kind="Entity", project_id=context.project_id, node_id=entity_id),
        rel_type=REL_MENTIONS,
        project_id=context.project_id,
        mention_count=1,
        metadata={
            "entity_name": entity.name,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "confidence": entity.confidence,
            "extractor": entity.extractor,
            "offsets": entity.offsets,
        },
    )


def build_extracted_from_edge(
    memory_id: str,
    source_ref: SourceRef,
    context: MemoryRequestContext,
    segment: SourceAwareSegment,
) -> GraphRelationship:
    """Build a Memory --EXTRACTED_FROM--> Source relationship."""
    if source_ref.source_id is None:
        raise ValueError("source_ref.source_id is required before writing source relationship")
    return GraphRelationship(
        source=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=memory_id),
        target=GraphNodeRef(kind="Source", project_id=context.project_id, node_id=source_ref.source_id),
        rel_type=REL_EXTRACTED_FROM,
        project_id=context.project_id,
        extraction_position={
            "message_index": segment.message_index,
            "start_offset": segment.start_offset,
            "end_offset": segment.end_offset,
        },
        metadata={
            "source_type": source_ref.source_type,
            "role": segment.role,
            "timestamp": segment.timestamp,
        },
    )


def build_mentioned_in_source_edge(
    entity_id: str,
    source_ref: SourceRef,
    entity: Entity,
    context: MemoryRequestContext,
) -> GraphRelationship:
    """Build an Entity --MENTIONED_IN_SOURCE--> Source relationship."""
    if source_ref.source_id is None:
        raise ValueError("source_ref.source_id is required before writing source relationship")
    return GraphRelationship(
        source=GraphNodeRef(kind="Entity", project_id=context.project_id, node_id=entity_id),
        target=GraphNodeRef(kind="Source", project_id=context.project_id, node_id=source_ref.source_id),
        rel_type=REL_MENTIONED_IN_SOURCE,
        project_id=context.project_id,
        mention_count=1,
        metadata={
            "entity_name": entity.name,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "confidence": entity.confidence,
            "extractor": entity.extractor,
        },
    )


def build_relates_to_edge(
    memory_id: str,
    related_memory_id: str,
    context: MemoryRequestContext,
    *,
    edge_type: str = "related_to",
    source: str = "add_related_recall",
) -> GraphRelationship:
    """Build a Memory --RELATES_TO--> Memory relationship."""
    return GraphRelationship(
        source=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=memory_id),
        target=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=related_memory_id),
        rel_type=REL_RELATES_TO,
        project_id=context.project_id,
        edge_type=edge_type,
        metadata={"source": source},
    )
