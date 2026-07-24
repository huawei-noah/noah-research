"""Reusable entity-store recall component."""

from __future__ import annotations

from typing import Any

from ...llm import EmbedClient
from ...typing import (
    EntitySearchResult,
    FieldCondition,
    MemoryRequestContext,
    SearchFilter,
)
from ..text import SparseVectorEncoder, TextPreprocessor
from .rrf import reciprocal_rank_fusion


class EntityRecall:
    """Recall canonical entities with dense + BM25 retrieval and RRF fusion."""

    def __init__(
        self,
        *,
        db_reader: Any,
        embed_client: EmbedClient,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        rrf_k: int = 80,
    ) -> None:
        self.db_reader = db_reader
        self.embed_client = embed_client
        self.text_preprocessor = text_preprocessor
        self.sparse_encoder = sparse_encoder
        self.rrf_k = rrf_k

    async def recall_entities(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        filters: SearchFilter | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Recall canonical entities from dense and sparse stores."""

        vector_results = await self.recall_dense(ctx, query, filters=filters, limit=limit)
        bm25_results = await self.recall_sparse(ctx, query, filters=filters, limit=limit)
        return combine_entity_results_rrf(vector_results, bm25_results, rrf_k=self.rrf_k, top_k=limit)

    async def recall_dense(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        filters: SearchFilter | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Recall entity candidates using dense embeddings."""

        resp = await self.embed_client.embed(task="search.entity_recall", text=query)
        if not resp.embeddings or not resp.embeddings[0]:
            return []

        result: EntitySearchResult = await self.db_reader.search_entities_dense(
            ctx,
            query=query,
            query_vector=resp.embeddings[0],
            filters=filters,
            limit=limit,
        )
        return _entity_hits_to_dicts(result)[:limit]

    async def recall_sparse(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        filters: SearchFilter | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Recall entity candidates using BM25 sparse vectors."""

        if not (self.text_preprocessor and self.sparse_encoder):
            return []
        preprocessed = self.text_preprocessor.preprocess_query(query, include_entities=False)
        if not preprocessed.tokens:
            return []

        sparse = self.sparse_encoder.encode_query(preprocessed.tokens)
        result: EntitySearchResult = await self.db_reader.search_entities_sparse(
            ctx,
            indices=list(sparse.indices),
            values=list(sparse.values),
            filters=filters,
            limit=limit,
        )
        return _entity_hits_to_dicts(result)[:limit]


def build_entity_type_filter(entity_types: list[str] | None) -> SearchFilter | None:
    """Build an entity_type filter."""

    if not entity_types:
        return None
    if len(entity_types) == 1:
        return SearchFilter(must=[FieldCondition(field="entity_type", op="match", value=entity_types[0])])
    return SearchFilter(must=[FieldCondition(field="entity_type", op="any", values=entity_types)])


def combine_entity_results_rrf(
    vector_entities: list[dict[str, Any]],
    bm25_entities: list[dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """Fuse dense and BM25 entity results with RRF."""

    vector_results = [
        {"entity_id": e["entity_id"], "score": e.get("score", 1.0 / (i + 1)), **e}
        for i, e in enumerate(vector_entities)
        if e.get("entity_id")
    ]
    bm25_results = [
        {"entity_id": e["entity_id"], "score": e.get("score", 0.0), **e} for e in bm25_entities if e.get("entity_id")
    ]

    if vector_results and bm25_results:
        rrf_results = reciprocal_rank_fusion(
            result_lists=[vector_results, bm25_results],
            k=rrf_k,
            id_key="entity_id",
            score_keys=["score"],
        )
    elif vector_results:
        rrf_results = [{**r, "rrf_score": r["score"]} for r in vector_results]
    elif bm25_results:
        rrf_results = [{**r, "rrf_score": r["score"]} for r in bm25_results]
    else:
        return []

    best_search_fields = _best_entity_search_fields(vector_results, bm25_results)
    for result in rrf_results:
        entity_id = result["entity_id"]
        if entity_id in best_search_fields:
            result["best_search_field"] = best_search_fields[entity_id]["text"]
            result["best_search_field_score"] = best_search_fields[entity_id]["score"]
            result["best_search_field_source"] = best_search_fields[entity_id]["source"]

    return rrf_results[:top_k]


def _entity_hits_to_dicts(result: EntitySearchResult) -> list[dict[str, Any]]:
    return [
        {
            "entity_id": hit.entity_id,
            "score": hit.score,
            "entity_view": hit.entity,
            "best_search_field": hit.best_search_field,
            "best_search_field_index": hit.best_search_field_index,
            "best_search_field_score": hit.best_search_field_score
            if hit.best_search_field_score is not None
            else hit.score,
            "best_search_field_source": hit.source or "",
            "matched_point_role": hit.matched_point_role,
        }
        for hit in result.hits
        if hit.entity is not None
    ]


def _best_entity_search_fields(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for source, results in (("vector", vector_results), ("bm25", bm25_results)):
        for result in results:
            entity_id = result.get("entity_id")
            search_field = str(result.get("best_search_field") or "").strip()
            if not entity_id or not search_field:
                continue
            score = float(result.get("score") or 0.0)
            current = best.get(entity_id)
            if current is None or score > current["score"]:
                best[entity_id] = {"text": search_field, "score": score, "source": source}
    return best
