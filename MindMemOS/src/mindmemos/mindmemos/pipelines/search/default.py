"""Default search engine using BM25 sparse retrieval."""

from __future__ import annotations

from datetime import datetime

from ...components.text import SparseVectorEncoder, TextPreprocessor, get_text_preprocessor
from ...config import TextProcessingConfig, get_config
from ...mappers import parse_search_dsl
from ...typing import (
    FieldCondition,
    MemoryDbSearchQuery,
    MemoryRequestContext,
    MemorySearchItem,
    SearchFilter,
    SearchPipelineInput,
)
from ..base import MemoryDbPipelineMixin
from .base import SearchEngineOptions


class DefaultSearchEngine(MemoryDbPipelineMixin):
    """Retrieve memories with Qdrant sparse BM25 search."""

    name = "default"

    def __init__(
        self,
        *,
        text_config: TextProcessingConfig | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        cfg = text_config or get_config().algo_config.text_processing
        self._text_preprocessor = text_preprocessor or get_text_preprocessor(cfg)
        self._sparse_encoder = sparse_encoder or SparseVectorEncoder(cfg)

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        """Search memories with the query sparse vector."""

        preprocessed = self._text_preprocessor.preprocess_query(inp.query, include_entities=False)
        if not preprocessed.tokens:
            return []

        sparse = self._sparse_encoder.encode_query(preprocessed.tokens)
        query = MemoryDbSearchQuery(
            query=inp.query,
            top_k=inp.top_k or get_config().algo_config.search.default.top_k,
            filters=_request_filter(inp, context),
            mode="bm25",
            ranking="score",
        )
        result = await self.db_reader.search_sparse(
            context,
            query,
            indices=list(sparse.indices),
            values=list(sparse.values),
        )
        return [
            MemorySearchItem(
                id=hit.memory_id,
                memory=hit.memory.content if hit.memory else "",
                memory_type=hit.memory.mem_type if hit.memory else "fact",
                last_update_at=_format_time((hit.memory.update_at or hit.memory.created_at) if hit.memory else None),
            )
            for hit in result.hits
        ]


def _request_filter(inp: SearchPipelineInput, ctx: MemoryRequestContext) -> SearchFilter:
    """Combine user DSL with the always-on active-memory scope."""
    base = parse_search_dsl(inp.filters)
    defaults = _default_scope(ctx)
    return SearchFilter(
        must=[*defaults, *base.must],
        should=base.should,
        must_not=base.must_not,
    )


def _default_scope(ctx: MemoryRequestContext) -> list[FieldCondition]:
    """Build always-on scope conditions for public search pipelines."""
    return [FieldCondition(field="status", op="match", value="active")]


def _format_time(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")
