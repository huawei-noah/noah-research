"""Memory vectorization — combine sparse BM25 and dense semantic vectors.

Generates ``VectorWrite`` DTOs with both sparse and dense vectors for a
single memory. Handles consistency-aware embedding failure: raises in
strong mode, marks pending in fast mode.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from ...errors import EmbeddingDimensionError
from ...logging import get_logger
from ...typing import EmbeddingResponse, EntityVectorWrite, EntityWrite, MemoryWrite, PreprocessedText, VectorWrite
from .sparse import SparseVectorEncoder

logger = get_logger(__name__)

MEMORY_EMBED_BATCH_SIZE = 10


class EmbedClient(Protocol):
    """Minimal protocol for dense embedding.

    Matches the real ``EmbedClient`` in ``mindmemos.llm.embedding`` which
    returns ``EmbeddingResponse`` (vector at ``response.embeddings[0]``).
    """

    async def embed(self, task: str, text: str | list[str], **kwargs) -> EmbeddingResponse: ...


class MemoryVectorizer:
    """Generate sparse + dense vectors for memory content.

    Stateless component: each call produces one VectorWrite independently.
    """

    def __init__(
        self,
        *,
        sparse_encoder: SparseVectorEncoder,
        embed_client: EmbedClient | None = None,
        text_preprocessor=None,
    ) -> None:
        self._sparse_encoder = sparse_encoder
        self._embed_client = embed_client
        self._text_preprocessor = text_preprocessor

    async def vectorize(
        self,
        memory_id: str,
        preprocessed: PreprocessedText,
        content: str,
        consistency: str = "fast",
    ) -> tuple[VectorWrite, bool]:
        """Generate sparse BM25 vector and optional dense semantic vector.

        Returns (VectorWrite, vector_pending). When dense embedding fails
        in fast mode, vector_pending=True and dense vector is None.
        """
        vector = await asyncio.to_thread(self._sparse_memory_vector, memory_id, preprocessed)
        vector_pending = False

        if self._embed_client is not None:
            try:
                resp = await self._embed_client.embed(
                    task="memory.add.embed",
                    text=content,
                )
                vector.semantic_vector = resp.embeddings[0] if resp.embeddings else None
                if not resp.embeddings:
                    if consistency == "strong":
                        msg = "Embedding response contained no vectors"
                        raise RuntimeError(msg)
                    vector_pending = True
            except EmbeddingDimensionError:
                raise
            except Exception:
                logger.warning("embed_failed", memory_id=memory_id, consistency=consistency)
                if consistency == "strong":
                    raise
                vector_pending = True

        return vector, vector_pending

    async def vectorize_many(
        self,
        items: list[tuple[str, PreprocessedText, str]],
        consistency: str = "fast",
    ) -> tuple[list[VectorWrite], list[bool]]:
        """Generate sparse vectors and batched dense vectors for memory contents."""
        vectors = await asyncio.to_thread(self._sparse_memory_vectors, items)
        vector_pending = [False] * len(items)

        if self._embed_client is None or not items:
            return vectors, vector_pending

        for start in range(0, len(items), MEMORY_EMBED_BATCH_SIZE):
            batch_items = items[start : start + MEMORY_EMBED_BATCH_SIZE]
            batch_vectors = vectors[start : start + MEMORY_EMBED_BATCH_SIZE]
            try:
                resp = await self._embed_client.embed(
                    task="memory.add.embed",
                    text=[content for _, _, content in batch_items],
                )
                embeddings = list(resp.embeddings)
            except EmbeddingDimensionError:
                raise
            except Exception:
                logger.warning("memory_embed_batch_failed", count=len(batch_items), consistency=consistency)
                await self._fallback_vectorize_batch(
                    batch_items,
                    batch_vectors,
                    vector_pending,
                    offset=start,
                    consistency=consistency,
                )
                continue

            if len(embeddings) > len(batch_items):
                embeddings = embeddings[: len(batch_items)]

            for vector, embedding in zip(batch_vectors, embeddings, strict=False):
                vector.semantic_vector = embedding

            if len(embeddings) < len(batch_items):
                missing_start = len(embeddings)
                await self._fallback_vectorize_batch(
                    batch_items[missing_start:],
                    batch_vectors[missing_start:],
                    vector_pending,
                    offset=start + missing_start,
                    consistency=consistency,
                )

        return vectors, vector_pending

    async def _fallback_vectorize_batch(
        self,
        items: list[tuple[str, PreprocessedText, str]],
        vectors: list[VectorWrite],
        vector_pending: list[bool],
        *,
        offset: int,
        consistency: str,
    ) -> None:
        for index, ((memory_id, _, content), vector) in enumerate(zip(items, vectors, strict=True)):
            try:
                resp = await self._embed_client.embed(
                    task="memory.add.embed",
                    text=content,
                )
                vector.semantic_vector = resp.embeddings[0] if resp.embeddings else None
                if not resp.embeddings:
                    if consistency == "strong":
                        msg = "Embedding response contained no vectors"
                        raise RuntimeError(msg)
                    vector_pending[offset + index] = True
            except EmbeddingDimensionError:
                raise
            except Exception:
                logger.warning("embed_failed", memory_id=memory_id, consistency=consistency)
                if consistency == "strong":
                    raise
                vector_pending[offset + index] = True

    def _sparse_memory_vector(self, memory_id: str, preprocessed: PreprocessedText) -> VectorWrite:
        sparse = self._sparse_encoder.encode_document(preprocessed.tokens)
        return VectorWrite(
            memory_id=memory_id,
            bm25_indices=list(sparse.indices),
            bm25_values=list(sparse.values),
        )

    async def vectorize_entities(
        self,
        entities: list[EntityWrite],
        *,
        memories_by_entity: dict[str, list[MemoryWrite]] | None = None,
        consistency: str = "fast",
    ) -> tuple[list[EntityVectorWrite], bool]:
        """Generate vectors for entities and their search fields in one embedding batch."""

        text_items: list[tuple[str, str]] = []
        memories_by_entity = memories_by_entity or {}
        for entity in entities:
            core_text = _entity_embedding_text(entity, memories_by_entity.get(entity.entity_id, []))
            entity.metadata = {**dict(entity.metadata or {}), "core_search_field": core_text}
            text_items.append((entity.entity_id, core_text))
            for index, search_field in enumerate((entity.metadata or {}).get("search_fields", [])):
                if isinstance(search_field, str) and search_field.strip():
                    text_items.append((f"{entity.entity_id}#sf{index}", search_field.strip()[:2000]))

        vector_pending = False
        semantic_vectors: list[list[float] | None] = [None] * len(text_items)
        if self._embed_client is not None and text_items:
            try:
                resp = await self._embed_client.embed(
                    task="memory_vectorizer.add.entity",
                    text=[text for _, text in text_items],
                )
                semantic_vectors = list(resp.embeddings)
                if len(semantic_vectors) < len(text_items):
                    vector_pending = True
                    semantic_vectors.extend([None] * (len(text_items) - len(semantic_vectors)))
                elif len(semantic_vectors) > len(text_items):
                    semantic_vectors = semantic_vectors[: len(text_items)]
            except EmbeddingDimensionError:
                raise
            except Exception:
                logger.warning("entity_embed_failed", entity_count=len(entities), consistency=consistency)
                if consistency == "strong":
                    raise
                vector_pending = True

        writes = await asyncio.to_thread(self._entity_vector_writes, text_items, semantic_vectors)
        return writes, vector_pending

    def _sparse_memory_vectors(self, items: list[tuple[str, PreprocessedText, str]]) -> list[VectorWrite]:
        return [self._sparse_memory_vector(memory_id, preprocessed) for memory_id, preprocessed, _ in items]

    def _entity_vector_writes(
        self,
        text_items: list[tuple[str, str]],
        semantic_vectors: list[list[float] | None],
    ) -> list[EntityVectorWrite]:
        writes: list[EntityVectorWrite] = []
        for index, (entity_id, text) in enumerate(text_items):
            sparse = self._sparse_encoder.encode_document(self._tokens_for_text(text))
            writes.append(
                EntityVectorWrite(
                    entity_id=entity_id,
                    semantic_vector=semantic_vectors[index],
                    bm25_indices=list(sparse.indices),
                    bm25_values=list(sparse.values),
                )
            )
        return writes

    def _tokens_for_text(self, text: str) -> list[str]:
        if self._text_preprocessor is not None:
            return self._text_preprocessor.preprocess_text(text, include_entities=False).tokens
        return [token for token in text.lower().split() if token]


def _entity_embedding_text(entity: EntityWrite, memories: list[MemoryWrite] | None = None) -> str:
    property_text = " ".join(memory.content for memory in (memories or [])[:5])
    search_field_text = " ".join(
        str(field) for field in (entity.metadata or {}).get("search_fields", [])[:5] if isinstance(field, str)
    )
    return " ".join(
        part
        for part in [
            entity.entity_name,
            entity.entity_type or "",
            entity.description or "",
            property_text,
            search_field_text,
        ]
        if part
    )
