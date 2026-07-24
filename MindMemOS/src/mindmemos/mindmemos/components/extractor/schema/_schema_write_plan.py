"""Schema add write-plan assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from ....logging import get_logger
from ....typing import (
    EntityVectorWrite,
    EntityWrite,
    GraphRelationship,
    MemoryDbWritePlan,
    MemoryWrite,
    SparseVector,
    VectorWrite,
)
from ...text import SparseVectorEncoder, TextPreprocessor
from ._schema_utils import entity_write_embedding_text, memory_embedding_text
from .base import SchemaWritePlanBuilderProtocol

logger = get_logger(__name__)

EmbedTexts = Callable[[str, list[str]], Awaitable[list[list[float]]]]


@dataclass(slots=True)
class SchemaWritePlanBuilder(SchemaWritePlanBuilderProtocol):
    """Build DB write plans and vectors for resolved schema-add DTOs."""

    text_preprocessor: TextPreprocessor
    sparse_encoder: SparseVectorEncoder
    embed_texts: EmbedTexts

    async def build(
        self,
        *,
        memories: list[MemoryWrite],
        entities: list[EntityWrite],
        relationships: list[GraphRelationship],
        project_id: str,
        entity_context_memories: list[MemoryWrite] | None = None,
    ) -> MemoryDbWritePlan:
        """Build a complete memory DB write plan from resolved DTOs."""

        vector_writes = await self.build_memory_vectors(memories, project_id=project_id)
        entity_vector_writes = await self.build_entity_vectors(entities, entity_context_memories or memories)
        return MemoryDbWritePlan(
            memories=memories,
            entities=entities,
            vectors=vector_writes,
            entity_vectors=entity_vector_writes,
            relationships=relationships,
        )

    async def build_memory_vectors(self, memories: list[MemoryWrite], *, project_id: str) -> list[VectorWrite]:
        """Generate dense and sparse vectors for memory property points."""

        memory_texts = [memory_embedding_text(memory) for memory in memories]
        dense_vectors = await self.embed_texts("memory.add.property", memory_texts)
        return self._memory_vectors(memories, dense_vectors, project_id)

    async def build_entity_vectors(
        self,
        entities: list[EntityWrite],
        memories: list[MemoryWrite],
    ) -> list[EntityVectorWrite]:
        """Generate dense and sparse vectors for entities and their search fields."""

        entity_by_name = {entity.entity_name: entity for entity in entities}
        final_entity_texts = {
            name: entity_write_embedding_text(
                entity, [memory for memory in memories if memory.entity_id == entity.entity_id]
            )
            for name, entity in entity_by_name.items()
        }
        for name, entity in entity_by_name.items():
            entity.metadata = {**dict(entity.metadata or {}), "core_search_field": final_entity_texts[name]}
        final_entity_vectors = await self.embed_texts(
            "memory.add.entity",
            [final_entity_texts[name] for name in entity_by_name],
        )
        final_entity_vector_by_name = {
            name: vector for name, vector in zip(entity_by_name, final_entity_vectors, strict=True)
        }

        entity_vector_writes: list[EntityVectorWrite] = []
        for name, entity_write in entity_by_name.items():
            embedding_text = final_entity_texts[name]
            sparse = self._encode_sparse(embedding_text)
            entity_vector_writes.append(
                EntityVectorWrite(
                    entity_id=entity_write.entity_id,
                    semantic_vector=final_entity_vector_by_name.get(name),
                    bm25_indices=sparse.indices,
                    bm25_values=sparse.values,
                )
            )

        search_field_texts: list[str] = []
        search_field_mapping: list[tuple[str, int]] = []
        for entity_write in entity_by_name.values():
            search_fields = (entity_write.metadata or {}).get("search_fields", [])
            for index, search_field in enumerate(search_fields):
                if search_field and isinstance(search_field, str) and search_field.strip():
                    search_field_texts.append(str(search_field).strip()[:2000])
                    search_field_mapping.append((entity_write.entity_id, index))

        if search_field_texts:
            search_field_vectors = await self.embed_texts("entity.add.search_field", search_field_texts)
            for (entity_id, search_field_index), search_field_text, search_field_vector in zip(
                search_field_mapping,
                search_field_texts,
                search_field_vectors,
                strict=True,
            ):
                search_field_sparse = self._encode_sparse(search_field_text)
                entity_vector_writes.append(
                    EntityVectorWrite(
                        entity_id=f"{entity_id}#sf{search_field_index}",
                        semantic_vector=search_field_vector,
                        bm25_indices=search_field_sparse.indices,
                        bm25_values=search_field_sparse.values,
                    )
                )

        return entity_vector_writes

    def _memory_vectors(
        self,
        memories: list[MemoryWrite],
        dense_vectors: list[list[float]],
        project_id: str,
    ) -> list[VectorWrite]:
        if len(dense_vectors) != len(memories):
            logger.error(
                "embedding count mismatch; truncating to shorter length",
                project_id=project_id,
                memory_count=len(memories),
                embedding_count=len(dense_vectors),
            )
            min_len = min(len(memories), len(dense_vectors))
            memories = memories[:min_len]
            dense_vectors = dense_vectors[:min_len]
        vector_writes: list[VectorWrite] = []
        for memory, dense in zip(memories, dense_vectors, strict=True):
            sparse = self._encode_sparse(memory_embedding_text(memory))
            vector_writes.append(
                VectorWrite(
                    memory_id=memory.memory_id,
                    semantic_vector=dense,
                    bm25_indices=sparse.indices,
                    bm25_values=sparse.values,
                )
            )
        return vector_writes

    def _encode_sparse(self, text: str) -> SparseVector:
        preprocessed = self.text_preprocessor.preprocess_text(text, include_entities=False)
        return self.sparse_encoder.encode_document(preprocessed.tokens)
