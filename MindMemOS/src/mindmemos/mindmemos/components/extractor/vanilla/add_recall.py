"""Add-time related memory recall component."""

from __future__ import annotations

import asyncio

from ....logging import get_logger
from ....typing import (
    FieldCondition,
    MemoryDbSearchQuery,
    MemoryRequestContext,
    MemoryView,
    PreprocessedText,
    RelatedMemoryCandidate,
    RelatedMemoryRecallResult,
    SearchFilter,
)
from ...text import SparseVectorEncoder
from ..protocols import AddRecallStrategy

logger = get_logger(__name__)


class RelatedMemoryRecall(AddRecallStrategy):
    """Recall small related-memory context for the add planner."""

    def __init__(
        self,
        *,
        db_reader,
        sparse_encoder: SparseVectorEncoder,
        top_k: int = 5,
        scan_limit: int = 100,
        fusion_weights: dict[str, float] | None = None,
        fusion_k: int = 60,
    ) -> None:
        self._db_reader = db_reader
        self._sparse_encoder = sparse_encoder
        self._top_k = top_k
        self._scan_limit = scan_limit
        self._fusion_weights = fusion_weights or {
            "semantic": 1.5,
            "bm25": 1.0,
            "entity": 1.2,
            "recent": 0.5,
            "schema_property": 2.0,
        }
        self._fusion_k = fusion_k

    async def recall(
        self,
        ctx: MemoryRequestContext,
        preprocessed: PreprocessedText,
        *,
        active_memories: list[MemoryView] | None = None,
    ) -> RelatedMemoryRecallResult:
        scanned = active_memories if active_memories is not None else await self._list_active_memories(ctx)
        hash_candidates = self._hash_candidates(preprocessed, scanned)
        duplicate = hash_candidates[0] if hash_candidates else None
        if not scanned:
            return RelatedMemoryRecallResult(duplicate=duplicate, candidates=[])

        candidates = [
            *hash_candidates,
            *self._entity_candidates(preprocessed, scanned),
            *await self._bm25_candidates(ctx, preprocessed),
        ]
        fused = (
            weighted_related_memory_rrf(candidates, top_k=self._top_k, weights=self._fusion_weights, k=self._fusion_k)
            if candidates
            else []
        )
        return RelatedMemoryRecallResult(duplicate=duplicate, candidates=fused)

    async def list_active_memories(self, ctx: MemoryRequestContext) -> list[MemoryView]:
        """Fetch active memories for a project. Exposed for batch-level caching."""
        return await self._list_active_memories(ctx)

    async def _list_active_memories(self, ctx: MemoryRequestContext) -> list[MemoryView]:
        memories, _ = await self._db_reader.list_memories(
            ctx,
            filters=SearchFilter(
                must=[*_context_conditions(ctx), FieldCondition(field="status", op="match", value="active")]
            ),
            limit=self._scan_limit,
        )
        return memories

    def _hash_candidates(
        self,
        preprocessed: PreprocessedText,
        memories: list[MemoryView],
    ) -> list[RelatedMemoryCandidate]:
        candidates: list[RelatedMemoryCandidate] = []
        for rank, memory in enumerate(memories, start=1):
            if memory.metadata.get("content_hash") != preprocessed.content_hash:
                continue
            candidates.append(
                RelatedMemoryCandidate(
                    memory_id=memory.memory_id,
                    score=1.0,
                    source="hash",
                    rank=rank,
                    memory=memory,
                    debug={"match": "content_hash"},
                )
            )
        return candidates

    def _entity_candidates(
        self,
        preprocessed: PreprocessedText,
        memories: list[MemoryView],
    ) -> list[RelatedMemoryCandidate]:
        query_entities = _entity_names(preprocessed)
        if not query_entities:
            return []
        candidates: list[RelatedMemoryCandidate] = []
        for rank, memory in enumerate(memories, start=1):
            overlap = query_entities & set(memory.metadata.get("entities") or [])
            if not overlap:
                continue
            candidates.append(
                RelatedMemoryCandidate(
                    memory_id=memory.memory_id,
                    score=len(overlap) / len(query_entities),
                    source="entity",
                    rank=rank,
                    memory=memory,
                    debug={"overlap": sorted(overlap)},
                )
            )
        return candidates

    async def _bm25_candidates(
        self,
        ctx: MemoryRequestContext,
        preprocessed: PreprocessedText,
    ) -> list[RelatedMemoryCandidate]:
        if not preprocessed.tokens:
            return []
        sparse = await asyncio.to_thread(self._sparse_encoder.encode_query, preprocessed.tokens)
        if not sparse.indices:
            return []
        req = MemoryDbSearchQuery(
            query=preprocessed.normalized_text,
            top_k=self._top_k,
            filters=SearchFilter(
                must=[*_context_conditions(ctx), FieldCondition(field="status", op="match", value="active")]
            ),
            mode="bm25",
            ranking="score",
        )
        result = await self._db_reader.search_sparse(
            ctx,
            req,
            indices=list(sparse.indices),
            values=list(sparse.values),
        )
        return [
            RelatedMemoryCandidate(
                memory_id=hit.memory_id,
                score=hit.score,
                source="bm25",
                rank=hit.rank or index,
                memory=hit.memory,
                debug=hit.debug,
            )
            for index, hit in enumerate(result.hits, start=1)
        ]


def weighted_related_memory_rrf(
    candidates: list[RelatedMemoryCandidate],
    *,
    top_k: int,
    weights: dict[str, float],
    k: int = 60,
) -> list[RelatedMemoryCandidate]:
    """Fuse ranked related-memory candidates with weighted reciprocal rank fusion."""
    merged: dict[str, RelatedMemoryCandidate] = {}
    scores: dict[str, float] = {}
    channels: dict[str, list[str]] = {}

    for fallback_rank, candidate in enumerate(candidates, start=1):
        rank = candidate.rank or fallback_rank
        weight = weights.get(candidate.source, 1.0)
        scores[candidate.memory_id] = scores.get(candidate.memory_id, 0.0) + weight / (k + rank)
        channels.setdefault(candidate.memory_id, []).append(candidate.source)
        current = merged.get(candidate.memory_id)
        if current is None or candidate.score > current.score:
            merged[candidate.memory_id] = candidate

    fused: list[RelatedMemoryCandidate] = []
    for memory_id, candidate in merged.items():
        fused.append(
            candidate.model_copy(
                update={
                    "score": scores[memory_id],
                    "source": "rrf",
                    "debug": {**candidate.debug, "channels": channels[memory_id]},
                }
            )
        )
    return sorted(fused, key=lambda item: item.score, reverse=True)[:top_k]


def _entity_names(preprocessed: PreprocessedText) -> set[str]:
    return {
        entity.canonical_name or entity.name for entity in preprocessed.entities if entity.canonical_name or entity.name
    }


def _context_conditions(ctx: MemoryRequestContext) -> list[FieldCondition]:
    conditions: list[FieldCondition] = []
    if ctx.user_id:
        conditions.append(FieldCondition(field="user_id", op="match", value=ctx.user_id))
    if ctx.app_id:
        conditions.append(FieldCondition(field="app_id", op="match", value=ctx.app_id))
    if ctx.agent_id:
        conditions.append(FieldCondition(field="agent_id", op="match", value=ctx.agent_id))
    return conditions
