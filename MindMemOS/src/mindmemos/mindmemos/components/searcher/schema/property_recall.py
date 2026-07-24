"""Reusable property-memory recall component."""

from __future__ import annotations

from typing import Any

from ....llm import EmbedClient
from ....typing import (
    FieldCondition,
    MemoryDbSearchQuery,
    MemoryDbSearchResult,
    MemoryRequestContext,
    SearchFilter,
    combine_search_filters,
)
from ...text import SparseVectorEncoder, TextPreprocessor
from ..rrf import reciprocal_rank_fusion


class PropertyRecall:
    """Recall entity property memories with dense + BM25 retrieval and RRF fusion."""

    def __init__(
        self,
        *,
        db_reader: Any,
        embed_client: EmbedClient | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.db_reader = db_reader
        self.embed_client = embed_client
        self.text_preprocessor = text_preprocessor
        self.sparse_encoder = sparse_encoder
        self.rrf_k = rrf_k

    async def recall_properties(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        entity_id: str | None = None,
        filters: SearchFilter | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Recall property memories and return fused result dicts."""

        vector_results = await self.recall_dense(ctx, query, entity_id=entity_id, filters=filters, limit=limit)
        bm25_results = await self.recall_sparse(ctx, query, entity_id=entity_id, filters=filters, limit=limit)
        return combine_property_results_rrf(vector_results, bm25_results, rrf_k=self.rrf_k, top_k=limit)

    async def recall_dense(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        entity_id: str | None = None,
        filters: SearchFilter | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Recall property memories using dense embeddings."""

        if not self.embed_client:
            return []
        resp = await self.embed_client.embed(task="search.property_recall", text=query)
        if not resp.embeddings or not resp.embeddings[0]:
            return []

        if entity_id and filters is None and hasattr(self.db_reader, "search_entity_property_memories"):
            result: MemoryDbSearchResult = await self.db_reader.search_entity_property_memories(
                ctx,
                query_vector=resp.embeddings[0],
                entity_id=entity_id,
                limit=limit,
            )
            return _memory_hits_to_dicts(result)

        entity_filter = (
            SearchFilter(must=[FieldCondition(field="entity_id", op="match", value=entity_id)]) if entity_id else None
        )
        result = await self.db_reader.search_dense(
            ctx,
            MemoryDbSearchQuery(
                query=query,
                top_k=limit,
                filters=combine_search_filters(entity_filter, filters),
                mode="semantic",
                ranking="score",
            ),
            query_vector=resp.embeddings[0],
        )
        return _memory_hits_to_dicts(result)

    async def recall_sparse(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        entity_id: str | None = None,
        filters: SearchFilter | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Recall property memories using BM25 sparse vectors."""

        if not (self.text_preprocessor and self.sparse_encoder):
            return []
        preprocessed = self.text_preprocessor.preprocess_query(query, include_entities=False)
        if not preprocessed.tokens:
            return []

        sparse = self.sparse_encoder.encode_query(preprocessed.tokens)
        if entity_id and filters is None and hasattr(self.db_reader, "search_entity_property_memories_sparse"):
            result: MemoryDbSearchResult = await self.db_reader.search_entity_property_memories_sparse(
                ctx,
                indices=list(sparse.indices),
                values=list(sparse.values),
                entity_id=entity_id,
                limit=limit,
            )
            return _memory_hits_to_dicts(result)

        entity_filter = (
            SearchFilter(must=[FieldCondition(field="entity_id", op="match", value=entity_id)]) if entity_id else None
        )
        result = await self.db_reader.search_sparse(
            ctx,
            MemoryDbSearchQuery(
                query=query,
                top_k=limit,
                filters=combine_search_filters(entity_filter, filters),
                mode="bm25",
                ranking="score",
            ),
            indices=list(sparse.indices),
            values=list(sparse.values),
        )
        return _memory_hits_to_dicts(result)


def combine_property_results_rrf(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """Fuse dense and BM25 property-memory results with RRF."""

    vector_with_id = [{**result, "property_id": _property_id(result)} for result in vector_results]
    bm25_with_id = [{**result, "property_id": _property_id(result)} for result in bm25_results]

    if vector_with_id and bm25_with_id:
        rrf_results = reciprocal_rank_fusion(
            result_lists=[vector_with_id, bm25_with_id],
            k=rrf_k,
            id_key="property_id",
            score_keys=["score"],
        )
    elif vector_with_id:
        rrf_results = [{**result, "rrf_score": result.get("score", 0.0)} for result in vector_with_id]
    elif bm25_with_id:
        rrf_results = [{**result, "rrf_score": result.get("score", 0.0)} for result in bm25_with_id]
    else:
        return []

    return rrf_results[:top_k]


def _memory_hits_to_dicts(result: MemoryDbSearchResult) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": hit.memory_id,
            "content": hit.memory.content if hit.memory else "",
            "score": hit.score,
            "metadata": {
                "entity_id": hit.memory.entity_id if hit.memory else "",
                "entity_type": hit.memory.entity_type if hit.memory else "",
                "property_name": hit.memory.property_name if hit.memory else "",
                "property_value": hit.memory.content if hit.memory else "",
                "entity_name": (hit.memory.metadata or {}).get("entity_name", "") if hit.memory else "",
                "timestamp": (hit.memory.metadata or {}).get("property_time", "") if hit.memory else "",
                "uid": hit.memory.memory_id if hit.memory else "",
            },
        }
        for hit in result.hits
        if hit.memory is not None
    ]


def _property_id(result: dict[str, Any]) -> str:
    metadata = result.get("metadata", {})
    entity_id = metadata.get("entity_id", "")
    property_name = metadata.get("property_name", "")
    timestamp = metadata.get("timestamp", "")
    uid = metadata.get("uid") or result.get("memory_id")
    if uid:
        return f"{entity_id}#{property_name}#{timestamp}#{uid}"
    property_value = metadata.get("property_value") or result.get("content", "")
    return f"{entity_id}#{property_name}#{timestamp}#{property_value}"
