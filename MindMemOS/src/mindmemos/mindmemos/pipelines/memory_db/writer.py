"""Database write orchestration for memory write plans and mutations."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from ...components.text import SparseVectorEncoder, TextPreprocessor, get_text_preprocessor
from ...config import TextProcessingConfig, get_config
from ...errors import MemoryUpdateError
from ...infra.db import (
    DatabaseClients,
    EntityPoint,
    GraphRelationship,
    MemoryPoint,
    QdrantRecord,
    SourcePoint,
    SparseVectorData,
    resolve_database_clients,
)
from ...llm import EmbedClient, get_embed_client
from ...logging import get_logger, traced, traced_awaitable
from ...mappers import to_db_write_primitives, to_entity_node, to_memory_node, to_mutation_result, to_source_node
from ...typing import (
    ConsistencyMode,
    EntityVectorWrite,
    EntityWrite,
    MemoryDbDeleteCommand,
    MemoryDbEntityUpdateCommand,
    MemoryDbMemoryDeleteCommand,
    MemoryDbMemoryUpdateCommand,
    MemoryDbMutationPlan,
    MemoryDbMutationResult,
    MemoryDbUpdateCommand,
    MemoryDbWritePlan,
    MemoryDbWriteResult,
    MemoryRequestContext,
)
from .add_record_store import AddRecordStore

logger = get_logger(__name__)


class MemoryDbWriter:
    """Coordinate Qdrant writes and Neo4j graph mirror writes."""

    def __init__(
        self,
        clients: DatabaseClients | None = None,
        add_record_store: AddRecordStore | None = None,
        *,
        text_config: TextProcessingConfig | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        embed_client: EmbedClient | None = None,
    ) -> None:
        self._clients = resolve_database_clients(clients)
        self._add_records = add_record_store or AddRecordStore(clients=self._clients)
        self._text_config = text_config
        self._text_preprocessor = text_preprocessor
        self._sparse_encoder = sparse_encoder
        self._embed_client = embed_client

    def _ensure_text_components(self) -> tuple[TextPreprocessor, SparseVectorEncoder]:
        if self._text_preprocessor is None or self._sparse_encoder is None:
            cfg = self._text_config or get_config().algo_config.text_processing
            self._text_preprocessor = self._text_preprocessor or get_text_preprocessor(cfg)
            self._sparse_encoder = self._sparse_encoder or SparseVectorEncoder(cfg)
        return self._text_preprocessor, self._sparse_encoder

    def _ensure_embed_client(self) -> EmbedClient:
        if self._embed_client is None:
            self._embed_client = get_embed_client()
        return self._embed_client

    @traced("memory_db.write")
    async def write(
        self,
        ctx: MemoryRequestContext,
        plan: MemoryDbWritePlan,
        *,
        consistency: ConsistencyMode = "fast",
    ) -> MemoryDbWriteResult:
        """Write one prepared database plan into Qdrant and Neo4j."""

        return await self.apply_mutation_plan(
            ctx,
            MemoryDbMutationPlan.from_write_plan(plan),
            consistency=consistency,
        )

    @traced("memory_db.apply_mutation_plan")
    async def apply_mutation_plan(
        self,
        ctx: MemoryRequestContext,
        plan: MemoryDbMutationPlan,
        *,
        consistency: ConsistencyMode = "fast",
    ) -> MemoryDbWriteResult:
        """Apply one unified mutation plan through the memory DB boundary."""

        write_result = (
            await self._write_plan(ctx, plan.to_write_plan(), consistency=consistency)
            if plan.has_writes()
            else MemoryDbWriteResult()
        )
        errors = list(write_result.errors)
        graph_pending = write_result.graph_pending
        mutations: list[MemoryDbMutationResult] = []
        prefetch_error: Exception | None = None
        try:
            memory_records = await self._prefetch_memory_records(ctx, plan)
        except Exception as exc:
            prefetch_error = exc
            memory_records = {}

        for command in plan.memory_updates:
            try:
                if prefetch_error is not None:
                    raise prefetch_error
                result = await self._update_memory_command(ctx, command, memory_records.get(command.memory_id))
                mutations.append(result)
                if not result.changed:
                    graph_pending = True
            except Exception as exc:
                if command.consistency == "strong":
                    raise
                graph_pending = True
                errors.append(str(exc))

        for command in plan.memory_deletes:
            try:
                if prefetch_error is not None and not command.hard:
                    raise prefetch_error
                result = await self._delete_memory_command(ctx, command, memory_records.get(command.memory_id))
                mutations.append(result)
                if not result.changed:
                    graph_pending = True
            except Exception as exc:
                if command.consistency == "strong":
                    raise
                graph_pending = True
                errors.append(str(exc))

        unsupported_count = (
            len([command for command in plan.entity_updates if command.entity is None])
            + len(plan.entity_deletes)
            + len(plan.source_updates)
            + len(plan.source_deletes)
            + len(plan.relationship_deletes)
        )
        if unsupported_count:
            graph_pending = True
            errors.append(f"unsupported mutation commands: {unsupported_count}")

        return MemoryDbWriteResult(
            memory_ids=write_result.memory_ids,
            entity_ids=write_result.entity_ids,
            source_ids=write_result.source_ids,
            mutations=mutations,
            graph_pending=graph_pending,
            errors=errors,
        )

    async def _prefetch_memory_records(
        self,
        ctx: MemoryRequestContext,
        plan: MemoryDbMutationPlan,
    ) -> dict[str, QdrantRecord]:
        memory_ids = list(
            dict.fromkeys(
                [command.memory_id for command in plan.memory_updates]
                + [command.memory_id for command in plan.memory_deletes if not command.hard]
            )
        )
        if not memory_ids:
            return {}
        records = await self._clients.qdrant.get_memories(ctx.project_id, memory_ids)
        return {record.point_id: record for record in records}

    async def _write_plan(
        self,
        ctx: MemoryRequestContext,
        plan: MemoryDbWritePlan,
        *,
        consistency: ConsistencyMode = "fast",
    ) -> MemoryDbWriteResult:
        memory_points, entity_points, source_points, relationships = to_db_write_primitives(plan, ctx=ctx)

        core_entity_points = [
            point for point in entity_points if not (point.payload or {}).get("metadata", {}).get("is_search_field")
        ]
        search_field_entity_points = [
            point for point in entity_points if (point.payload or {}).get("metadata", {}).get("is_search_field")
        ]

        from collections import Counter

        entity_type_counts = Counter(point.payload.get("entity_type", "?") for point in core_entity_points)
        logger.info(
            "db_writer_write",
            memory_count=len(memory_points),
            entity_count=len(core_entity_points),
            entity_types=dict(entity_type_counts),
            sf_entity_count=len(search_field_entity_points),
            source_count=len(source_points),
            rel_count=len(relationships),
        )

        graph_pending = False
        errors: list[str] = []

        if consistency == "fast":
            qdrant_result, neo4j_result = await asyncio.gather(
                self._write_qdrant(memory_points, core_entity_points, search_field_entity_points, source_points),
                self._write_neo4j(ctx, plan, relationships),
            )
            graph_pending = qdrant_result[0] or neo4j_result[0]
            errors.extend(qdrant_result[1])
            errors.extend(neo4j_result[1])
        else:
            qdrant_result = await self._write_qdrant(
                memory_points,
                core_entity_points,
                search_field_entity_points,
                source_points,
                strong=True,
            )
            neo4j_result = await self._write_neo4j(ctx, plan, relationships, strong=True)
            graph_pending = qdrant_result[0] or neo4j_result[0]
            errors.extend(qdrant_result[1])
            errors.extend(neo4j_result[1])

        return MemoryDbWriteResult(
            memory_ids=[memory.memory_id for memory in plan.memories],
            entity_ids=[entity.entity_id for entity in plan.entities],
            source_ids=[source.source_id for source in plan.sources],
            graph_pending=graph_pending,
            errors=errors,
        )

    @traced("memory_db.update_entity")
    async def update_entity(
        self,
        ctx: MemoryRequestContext,
        entity: EntityWrite,
        *,
        entity_vectors: Sequence[EntityVectorWrite] = (),
        consistency: ConsistencyMode = "fast",
    ) -> MemoryDbWriteResult:
        """Upsert one resolved entity update through the memory DB boundary."""

        command = MemoryDbEntityUpdateCommand(
            entity_id=entity.entity_id,
            entity=entity,
            core_vector=next((vector for vector in entity_vectors if vector.entity_id == entity.entity_id), None),
            search_field_vectors=[vector for vector in entity_vectors if vector.entity_id != entity.entity_id],
            consistency=consistency,
        )
        return await self.apply_mutation_plan(
            ctx,
            MemoryDbMutationPlan(entity_updates=[command]),
            consistency=consistency,
        )

    @traced("memory_db.update_add_record")
    async def patch_add_record(
        self,
        ctx: MemoryRequestContext,
        add_record_id: str,
        payload: dict[str, Any],
    ) -> MemoryDbMutationResult:
        """Patch one add record payload in the request project."""

        await self._add_records.patch(ctx.project_id, add_record_id, payload)
        return to_mutation_result(add_record_id, changed=True)

    @traced("memory_db.update_memory")
    async def update_memory(self, ctx: MemoryRequestContext, req: MemoryDbUpdateCommand) -> MemoryDbMutationResult:
        """Patch one memory in place in the request project."""

        result = await self.apply_mutation_plan(
            ctx,
            MemoryDbMutationPlan(memory_updates=[req]),
            consistency=req.consistency,
        )
        return result.mutations[0] if result.mutations else to_mutation_result(req.memory_id, changed=False)

    async def _update_memory_command(
        self,
        ctx: MemoryRequestContext,
        req: MemoryDbMemoryUpdateCommand,
        record: QdrantRecord | None,
    ) -> MemoryDbMutationResult:
        if record is None:
            return to_mutation_result(req.memory_id, changed=False)

        if req.dedup_metadata_key and req.dedup_metadata_key in req.metadata_patch:
            existing_metadata = dict(record.payload.get("metadata") or {})
            if existing_metadata.get(req.dedup_metadata_key) == req.metadata_patch[req.dedup_metadata_key]:
                return to_mutation_result(req.memory_id, changed=False)

        now = datetime.now(UTC)
        patch: dict[str, Any] = {**req.payload_patch, "update_at": now}
        metadata_patch = dict(req.metadata_patch)
        sparse: SparseVectorData | None = _sparse_from_command(req)
        dense: list[float] | None = _dense_from_command(req)

        if req.content is not None:
            preprocessor, encoder = self._ensure_text_components()
            preprocessed = preprocessor.preprocess_text(req.content, segment_id="update", include_entities=False)
            patch["content"] = preprocessed.normalized_text
            metadata_patch.update(
                {
                    "content_hash": preprocessed.content_hash,
                    "bm25_text": preprocessed.bm25_text,
                    "tokens": list(preprocessed.tokens),
                    "lang": preprocessed.lang,
                }
            )
            if sparse is None:
                encoded = encoder.encode_document(preprocessed.tokens)
                sparse = SparseVectorData(indices=list(encoded.indices), values=list(encoded.values))
            if dense is None:
                embed_resp = await self._ensure_embed_client().embed(
                    task="memory.update",
                    text=preprocessed.normalized_text,
                )
                if not embed_resp.embeddings or not embed_resp.embeddings[0]:
                    raise MemoryUpdateError("memory update embedding returned empty vector")
                dense = embed_resp.embeddings[0]

        if req.reinforcement_count is not None:
            patch["reinforcement_count"] = req.reinforcement_count
        if req.reinforcement_count_delta:
            current = int(record.payload.get("reinforcement_count") or 0)
            patch["reinforcement_count"] = current + req.reinforcement_count_delta
        if req.status is not None:
            patch["status"] = req.status
            patch["status_changed_at"] = now

        if metadata_patch:
            metadata = dict(record.payload.get("metadata") or {})
            metadata.update(metadata_patch)
            patch["metadata"] = metadata

        await self._clients.qdrant.patch_memory(
            ctx.project_id,
            req.memory_id,
            patch,
            dense_vector=dense,
            sparse_vector=sparse,
            record=record,
        )
        record.payload.update(patch)

        if req.content is not None or req.graph_content_sync:
            try:
                await self._clients.neo4j.update_memory_content(
                    ctx.project_id,
                    req.memory_id,
                    str(patch.get("content") or req.content or ""),
                )
            except Exception:
                if req.consistency == "strong":
                    raise
                logger.warning(
                    "memory graph content update failed",
                    project_id=ctx.project_id,
                    memory_id=req.memory_id,
                    exc_info=True,
                )

        if req.status == "archived":
            try:
                await self._clients.neo4j.archive_memory_node(
                    ctx.project_id,
                    req.memory_id,
                    reason=req.reason or "unknown",
                )
            except Exception:
                if req.consistency == "strong":
                    raise
                logger.warning(
                    "memory graph archive failed",
                    project_id=ctx.project_id,
                    memory_id=req.memory_id,
                    exc_info=True,
                )

        return to_mutation_result(req.memory_id, changed=True)

    @traced("memory_db.delete_memory")
    async def delete_memory(self, ctx: MemoryRequestContext, req: MemoryDbDeleteCommand) -> MemoryDbMutationResult:
        """Archive or hard delete one memory in the request project."""

        result = await self.apply_mutation_plan(
            ctx,
            MemoryDbMutationPlan(memory_deletes=[req]),
            consistency=req.consistency,
        )
        return result.mutations[0] if result.mutations else to_mutation_result(req.memory_id, changed=False)

    async def _delete_memory_command(
        self,
        ctx: MemoryRequestContext,
        req: MemoryDbMemoryDeleteCommand,
        record: QdrantRecord | None,
    ) -> MemoryDbMutationResult:
        if req.hard:
            await self._clients.qdrant.delete_memory(ctx.project_id, req.memory_id)
            try:
                await self._clients.neo4j.delete_memory_node(ctx.project_id, req.memory_id)
            except Exception:
                if req.consistency == "strong":
                    raise
                logger.warning(
                    "memory graph hard delete failed",
                    project_id=ctx.project_id,
                    memory_id=req.memory_id,
                    exc_info=True,
                )
            return to_mutation_result(req.memory_id, changed=True, hard=True)

        if record is None:
            return to_mutation_result(req.memory_id, changed=False, hard=False)

        now = datetime.now(UTC)
        metadata = dict(record.payload.get("metadata") or {})
        metadata["delete_reason"] = req.reason
        patch = {
            "status": "archived",
            "status_changed_at": now,
            "update_at": now,
            "metadata": metadata,
        }
        await self._clients.qdrant.patch_memory(
            ctx.project_id,
            req.memory_id,
            patch,
            record=record,
        )
        record.payload.update(patch)
        try:
            await self._clients.neo4j.archive_memory_node(ctx.project_id, req.memory_id, reason=req.reason)
        except Exception:
            if req.consistency == "strong":
                raise
            logger.warning(
                "memory graph archive failed",
                project_id=ctx.project_id,
                memory_id=req.memory_id,
                exc_info=True,
            )
        return to_mutation_result(req.memory_id, changed=True, hard=False)

    async def _write_qdrant(
        self,
        memory_points: Sequence[MemoryPoint],
        core_entity_points: Sequence[EntityPoint],
        search_field_entity_points: Sequence[EntityPoint],
        source_points: Sequence[SourcePoint],
        *,
        strong: bool = False,
    ) -> tuple[bool, list[str]]:
        qdrant_tasks = [
            traced_awaitable(
                "qdrant.upsert_memories",
                self._clients.qdrant.upsert_memories(list(memory_points)),
                attributes={"point_count": len(memory_points)},
            ),
            traced_awaitable(
                "qdrant.upsert_entities",
                self._clients.qdrant.upsert_entities(list(core_entity_points)),
                attributes={"point_count": len(core_entity_points), "entity_role": "core"},
            ),
            traced_awaitable(
                "qdrant.upsert_sources",
                self._clients.qdrant.upsert_sources(list(source_points)),
                attributes={"point_count": len(source_points)},
            ),
        ]
        if search_field_entity_points:
            qdrant_tasks.append(
                traced_awaitable(
                    "qdrant.upsert_entities",
                    self._clients.qdrant.upsert_entities(list(search_field_entity_points)),
                    attributes={"point_count": len(search_field_entity_points), "entity_role": "search_field"},
                )
            )

        qdrant_results = await asyncio.gather(*qdrant_tasks, return_exceptions=True)
        qdrant_errors = [result for result in qdrant_results if isinstance(result, Exception)]
        if qdrant_errors:
            if strong:
                raise qdrant_errors[0]
            logger.warning(
                "qdrant_partial_write_failure",
                error_count=len(qdrant_errors),
                errors=[str(error) for error in qdrant_errors],
            )
            return True, [str(error) for error in qdrant_errors]
        return False, []

    async def _write_neo4j(
        self,
        ctx: MemoryRequestContext,
        plan: MemoryDbWritePlan,
        relationships: Sequence[GraphRelationship],
        *,
        strong: bool = False,
    ) -> tuple[bool, list[str]]:
        try:
            memory_nodes = [to_memory_node(memory, ctx=ctx) for memory in plan.memories]
            entity_nodes = [to_entity_node(entity, ctx=ctx) for entity in plan.entities]
            source_nodes = [to_source_node(source, ctx=ctx) for source in plan.sources]
            if memory_nodes or entity_nodes or source_nodes:
                await traced_awaitable(
                    "neo4j.upsert_nodes",
                    self._clients.neo4j.upsert_nodes(
                        memories=memory_nodes,
                        entities=entity_nodes,
                        sources=source_nodes,
                    ),
                    attributes={
                        "memory_count": len(memory_nodes),
                        "entity_count": len(entity_nodes),
                        "source_count": len(source_nodes),
                    },
                )

            if relationships:
                await traced_awaitable(
                    "neo4j.upsert_relationships",
                    self._clients.neo4j.upsert_relationships(list(relationships)),
                    attributes={"relationship_count": len(relationships)},
                )
        except Exception as exc:
            if strong:
                raise
            return True, [str(exc)]
        return False, []


def _dense_from_command(req: MemoryDbUpdateCommand) -> list[float] | None:
    dense = req.embedding if req.embedding is not None else req.dense_vector
    if dense is None:
        return None
    if not dense:
        raise MemoryUpdateError("memory update embedding cannot be empty")
    return list(dense)


def _sparse_from_command(req: MemoryDbUpdateCommand) -> SparseVectorData | None:
    if req.bm25_indices is not None:
        return SparseVectorData(indices=list(req.bm25_indices), values=[1.0] * len(req.bm25_indices))
    if req.sparse_vectors is None:
        return None

    indices = req.sparse_vectors.get("bm25_indices")
    values = req.sparse_vectors.get("bm25_values")
    if (indices is None) != (values is None):
        raise ValueError("sparse_vectors bm25_indices and bm25_values must be updated together")
    if indices is None or values is None:
        return None
    return SparseVectorData(indices=list(indices), values=list(values))
