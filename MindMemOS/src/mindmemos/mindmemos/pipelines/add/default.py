"""Default add pipeline for the first end-to-end memory write path."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from ...components.kafka import memory_add_dispatch_key
from ...components.text import SparseVectorEncoder, TextPreprocessor, get_text_preprocessor
from ...config import TextProcessingConfig, get_config
from ...infra.kafka import get_producer
from ...typing import (
    REL_MENTIONS,
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    Entity,
    EntityVectorWrite,
    EntityWrite,
    GraphNodeRef,
    GraphRelationship,
    MemoryAddEventItem,
    MemoryDbMutationPlan,
    MemoryDbWritePlan,
    MemoryRequestContext,
    MemoryWrite,
    VectorWrite,
)
from ..base import MemoryDbPipelineMixin
from ..memory_db import suppress_recording_errors
from ..registry import register

Consistency = Literal["fast", "strong"]
MEMORY_ADD_TOPIC = "memory.add"


@register(type="add", name="default_add")
class DefaultAddPipeline(MemoryDbPipelineMixin):
    """Write plain text memories with sparse vectors, entities, and mention edges."""

    def __init__(
        self,
        *,
        text_config: TextProcessingConfig | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        consistency: Consistency | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        cfg = text_config or get_config().algo_config.text_processing
        self._text_preprocessor = text_preprocessor or get_text_preprocessor(cfg)
        self._sparse_encoder = sparse_encoder or SparseVectorEncoder(cfg)
        self._consistency = consistency or _default_consistency()

    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        """Synchronously write normalized text memories and their mentioned entities."""

        plan, events = self._build_plan(inp, context)
        if plan.memories:
            await self.db_writer.apply_mutation_plan(
                context,
                MemoryDbMutationPlan.from_write_plan(plan),
                consistency=self._consistency,
            )
        result = AddPipelineSyncResult(status="ok", memories=events)
        await suppress_recording_errors(
            self.recorder.mark_add_completed(context, add_record_id, result),
            operation="add.default_add.sync",
        )
        return result

    async def add_async(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> AddPipelineAsyncResult:
        """Return an async acknowledgement; background execution is wired later."""

        message = {
            "context": context.model_dump(mode="json"),
            "input": inp.model_dump(mode="json", by_alias=True),
            "submitted_at": datetime.now(UTC).isoformat(),
        }
        if add_record_id is not None:
            message["add_record_id"] = add_record_id
        if record_metadata is not None:
            message["record_metadata"] = record_metadata

        await get_producer().send(
            MEMORY_ADD_TOPIC,
            value=message,
            dispatch_key=memory_add_dispatch_key(context),
        )
        return AddPipelineAsyncResult(status="queued")

    async def has_pending(self, context: MemoryRequestContext) -> bool:
        """Return whether the default add pipeline has queued work.

        Args:
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            False because this pipeline does not maintain an add buffer.
        """
        return False

    def _build_plan(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
    ) -> tuple[MemoryDbWritePlan, list[MemoryAddEventItem]]:
        now = datetime.now(UTC)
        event_time = inp.event_timestamp_utc
        memories: list[MemoryWrite] = []
        entities_by_id: dict[str, EntityWrite] = {}
        vectors: list[VectorWrite] = []
        entity_vectors: list[EntityVectorWrite] = []
        relationships: list[GraphRelationship] = []
        events: list[MemoryAddEventItem] = []

        for index, raw_text in enumerate(_iter_text_messages(inp), start=1):
            preprocessed = self._text_preprocessor.preprocess_text(raw_text, segment_id=f"segment-{index}")
            if not preprocessed.normalized_text:
                continue

            memory_id = str(uuid4())
            memory = MemoryWrite(
                memory_id=memory_id,
                account_id=context.account_id,
                project_id=context.project_id,
                api_key_uuid=context.api_key_uuid,
                user_id=context.user_id,
                app_id=context.app_id,
                session_id=context.session_id,
                agent_id=context.agent_id,
                request_id=context.request_id,
                content=preprocessed.normalized_text,
                mem_type="fact",
                mem_extract_type="vanilla",
                mem_extract_version="default_add_v1",
                metadata={
                    **dict(inp.metadata),
                    "content_hash": preprocessed.content_hash,
                    "bm25_text": preprocessed.bm25_text,
                    "tokens": list(preprocessed.tokens),
                    "lang": preprocessed.lang,
                    "source_message_index": index - 1,
                    "source_timestamp_ms": inp.event_timestamp,
                    "event_timestamp_ms": inp.event_timestamp,
                    "entity_count": len(preprocessed.entities),
                },
                validate_from=event_time,
                created_at=now,
                root_id=[memory_id],
            )
            memories.append(memory)
            events.append(MemoryAddEventItem(operation="add", content=memory.content))

            sparse = self._sparse_encoder.encode_document(preprocessed.tokens)
            vectors.append(
                VectorWrite(
                    memory_id=memory_id,
                    bm25_indices=list(sparse.indices),
                    bm25_values=list(sparse.values),
                )
            )

            for entity in _unique_entities(preprocessed.entities):
                entity_id = _entity_id(context.project_id, entity)
                entity_write = entities_by_id.setdefault(entity_id, _to_entity_write(entity, entity_id, context, now))
                _attach_search_fields(entity_write, [memory.content])
                relationships.append(_to_mentions_relationship(memory_id, entity_id, entity, context))

        for entity in entities_by_id.values():
            entity_vectors.extend(_to_entity_vectors(entity, self._text_preprocessor, self._sparse_encoder))

        return (
            MemoryDbWritePlan(
                memories=memories,
                entities=list(entities_by_id.values()),
                vectors=vectors,
                entity_vectors=entity_vectors,
                relationships=relationships,
            ),
            events,
        )


def _iter_text_messages(inp: AddPipelineInput):
    for message in inp.messages:
        text = getattr(message, "text", None)
        if text:
            yield text
            continue
        content = getattr(message, "content", None)
        if content:
            yield content


def _unique_entities(entities: list[Entity]) -> list[Entity]:
    unique: dict[tuple[str, str | None], Entity] = {}
    for entity in entities:
        name = entity.canonical_name or entity.name
        if not name:
            continue
        unique.setdefault((name, entity.entity_type), entity)
    return list(unique.values())


def _entity_id(project_id: str, entity: Entity) -> str:
    name = entity.canonical_name or entity.name
    key = f"{project_id}:{entity.entity_type or 'entity'}:{name}"
    return str(uuid5(NAMESPACE_URL, key))


def _to_entity_write(
    entity: Entity,
    entity_id: str,
    context: MemoryRequestContext,
    now: datetime,
) -> EntityWrite:
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


def _attach_search_fields(entity: EntityWrite, fields: list[str]) -> None:
    if not fields:
        return
    metadata = dict(entity.metadata or {})
    existing = [field for field in metadata.get("search_fields", []) if isinstance(field, str)]
    merged: list[str] = []
    for field in [*existing, *fields]:
        normalized = field.strip()
        if normalized and normalized not in merged:
            merged.append(normalized)
    metadata["search_fields"] = merged
    entity.metadata = metadata


def _to_entity_vectors(
    entity: EntityWrite,
    text_preprocessor: TextPreprocessor,
    sparse_encoder: SparseVectorEncoder,
) -> list[EntityVectorWrite]:
    core_text = _entity_embedding_text(entity)
    entity.metadata = {**dict(entity.metadata or {}), "core_search_field": core_text}
    texts: list[tuple[str, str]] = [(entity.entity_id, core_text)]
    for index, search_field in enumerate((entity.metadata or {}).get("search_fields", [])):
        if isinstance(search_field, str) and search_field.strip():
            texts.append((f"{entity.entity_id}#sf{index}", search_field.strip()[:2000]))
    vectors: list[EntityVectorWrite] = []
    for entity_id, text in texts:
        tokens = text_preprocessor.preprocess_text(text, include_entities=False).tokens
        sparse = sparse_encoder.encode_document(tokens)
        vectors.append(
            EntityVectorWrite(
                entity_id=entity_id,
                bm25_indices=list(sparse.indices),
                bm25_values=list(sparse.values),
            )
        )
    return vectors


def _entity_embedding_text(entity: EntityWrite) -> str:
    search_field_text = " ".join(
        str(field) for field in (entity.metadata or {}).get("search_fields", [])[:5] if isinstance(field, str)
    )
    return " ".join(
        part
        for part in [
            entity.entity_name,
            entity.entity_type or "",
            entity.description or "",
            search_field_text,
        ]
        if part
    )


def _to_mentions_relationship(
    memory_id: str,
    entity_id: str,
    entity: Entity,
    context: MemoryRequestContext,
) -> GraphRelationship:
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


def _default_consistency() -> Consistency:
    value = get_config().database.default_consistency
    return value if value in {"fast", "strong"} else "fast"
