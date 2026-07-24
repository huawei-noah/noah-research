"""Shared DTO construction utilities for pipelines.

Pure field-mapping convenience functions that convert business objects and
request context into write-ready DTOs. No algorithm, no DB primitives —
just DTO shape construction that multiple pipelines (add, update, merge,
dreaming) can reuse.
"""

from __future__ import annotations

from datetime import datetime

from ...typing import (
    Entity,
    EntityWrite,
    MemoryRequestContext,
    SourceRef,
    SourceWrite,
)


def build_source_write(source_ref: SourceRef, context: MemoryRequestContext, now: datetime) -> SourceWrite:
    """Convert a SourceRef into a SourceWrite DTO."""
    source_id = source_ref.source_id
    if source_id is None:
        raise ValueError("source_ref.source_id is required before writing source")
    file_path = source_ref.file_path or source_ref.uri or source_ref.message_id or source_id
    return SourceWrite(
        source_id=source_id,
        account_id=context.account_id,
        project_id=context.project_id,
        api_key_uuid=context.api_key_uuid,
        user_id=context.user_id,
        app_id=context.app_id,
        session_id=context.session_id,
        agent_id=context.agent_id,
        request_id=context.request_id,
        source_type=source_ref.source_type,
        file_path=file_path,
        file_name=source_ref.file_name or source_ref.title or file_path,
        is_parsed=source_ref.is_parsed,
        parsed_content_path=source_ref.parsed_content_path,
        created_at=now,
        parsed_at=source_ref.parsed_at,
        parsed_cost=source_ref.parsed_cost,
        root_id=[source_id],
        metadata={
            "uri": source_ref.uri,
            "mime_type": source_ref.mime_type,
            "content_hash": source_ref.content_hash,
            "message_id": source_ref.message_id,
            "chunk_id": source_ref.chunk_id,
            "page": source_ref.page,
            "line_range": source_ref.line_range,
            "start_offset": source_ref.start_offset,
            "end_offset": source_ref.end_offset,
            **dict(source_ref.metadata),
        },
        persist_payload=source_ref.source_type != "message",
    )


def build_entity_write(
    entity: Entity,
    entity_id: str,
    context: MemoryRequestContext,
    now: datetime,
) -> EntityWrite:
    """Convert an Entity into an EntityWrite DTO."""
    return EntityWrite(
        entity_id=entity_id,
        account_id=context.account_id,
        project_id=context.project_id,
        api_key_uuid=context.api_key_uuid,
        user_id=context.user_id,
        app_id=context.app_id,
        session_id=context.session_id,
        agent_id=context.agent_id,
        request_id=context.request_id,
        entity_name=entity.canonical_name or entity.name,
        entity_type=entity.entity_type,
        description=entity.description,
        created_at=now,
        root_id=[entity_id],
        metadata={
            "aliases": list(entity.aliases),
            "confidence": entity.confidence,
            "extractor": entity.extractor,
            "offsets": entity.offsets,
            **dict(entity.metadata),
        },
    )
