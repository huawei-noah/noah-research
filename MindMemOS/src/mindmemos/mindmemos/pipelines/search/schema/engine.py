"""Schema-aware single-pass search engine."""

from __future__ import annotations

from typing import Any

from ....components.memory_modeling.schema import EntityManager, get_entity_manager
from ....components.searcher.schema import SchemaSearchExpander, SchemaSearchQueryBuilder
from ....components.text import SparseVectorEncoder, TextPreprocessor, detect_prompt_language, get_text_preprocessor
from ....config import get_config
from ....config.algo.search import SearchConfig
from ....llm import EmbedClient, LLMClient, RerankClient, get_embed_client, get_llm_client
from ....mappers import parse_schema_search_filters
from ....prompts import SearchPromptSet, get_search_prompts
from ....typing import (
    MemoryDbSearchHit,
    MemoryDbSearchQuery,
    MemoryRequestContext,
    MemorySearchItem,
    SearchFilter,
    SearchPipelineInput,
)
from ...base import MemoryDbPipelineMixin
from ...utils import format_datetime, format_memory_event_time, format_source_timestamp
from ..base import SearchEngineOptions


class SchemaSearchEngine(MemoryDbPipelineMixin):
    """Run one schema-aware entity/property retrieval pass."""

    name = "schema"

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        embed_client: EmbedClient | None = None,
        rerank_client: RerankClient | None = None,
        entity_manager: EntityManager | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        search_config: SearchConfig | None = None,
        prompts: SearchPromptSet | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        cfg = get_config()
        self._search_config = search_config or cfg.algo_config.search
        self._schema_config = self._search_config.schema_search
        self._llm = llm_client or get_llm_client()
        self._embed_client = embed_client or get_embed_client()
        self._rerank_client = rerank_client if rerank_client is not None else _optional_rerank_client()
        self._entity_manager = entity_manager or get_entity_manager()
        text_config = cfg.algo_config.text_processing
        self._text_preprocessor = text_preprocessor or get_text_preprocessor()
        self._sparse_encoder = sparse_encoder or SparseVectorEncoder(text_config)
        self._prompts = prompts or get_search_prompts(cfg.algo_config.common.prompt_language)
        self._expander = SchemaSearchExpander(
            db_reader=self.db_reader,
            embed_client=self._embed_client,
            rerank_client=self._rerank_client,
            text_preprocessor=self._text_preprocessor,
            sparse_encoder=self._sparse_encoder,
            entity_manager=self._entity_manager,
            config=self._search_config.schema_search,
        )
        self._query_builder = SchemaSearchQueryBuilder(
            llm=self._llm,
            prompts=self._prompts,
            entity_schema=self._entity_manager.get_all_dicts(),
            current_time_mode=self._schema_config.current_time_mode,
            min_time_window_days=self._schema_config.min_time_window_days,
        )

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        """Search schema entities and project them to public memory items."""

        detected_lang = detect_prompt_language(
            inp.query,
            fallback=get_config().algo_config.common.prompt_language,
        )
        request_prompts = get_search_prompts(detected_lang)

        parsed_filters = parse_schema_search_filters(inp.filters, context)
        property_filter = self._query_builder.all_property_filter()
        initial_time_window = None
        if not parsed_filters.has_time_filter:
            initial_time_window = await self._query_builder.extract_time_from_query(
                inp.query, prompts=request_prompts
            )
        entities = await self._expander.search(
            ctx=parsed_filters.context,
            query=inp.query,
            entity_types=list(property_filter.keys()) or None,
            property_filter=property_filter,
            time_window=initial_time_window,
            search_filter=parsed_filters.memory_filter,
            entity_search_filter=parsed_filters.entity_filter,
            num_hops=options.num_hops if options and options.num_hops is not None else self._schema_config.multi_hop,
            use_reranker=options.use_reranker if options else None,
            top_k=options.recall_top_k if options else None,
            top_n=options.result_top_n if options and options.result_top_n is not None else inp.top_k,
        )
        if not entities:
            return await self._search_memory_fallback(
                inp, parsed_filters.context, parsed_filters.memory_filter, options
            )

        return [
            MemorySearchItem(
                id=entity.entity_id,
                memory=entity.format_entity_prompt(
                    ignore_edge_num=self._schema_config.output_max_edge_num,
                    include_description=False,
                    include_edges=self._schema_config.include_edges,
                ),
                memory_type="fact",
                last_update_at="",
            )
            for entity in entities
        ]

    async def _search_memory_fallback(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        filters: SearchFilter | None,
        options: SearchEngineOptions | None,
    ) -> list[MemorySearchItem]:
        """Fallback to direct memory recall when schema entity recall is empty."""

        preprocessed = self._text_preprocessor.preprocess_query(inp.query, include_entities=False)
        if not preprocessed.tokens:
            return []

        sparse = self._sparse_encoder.encode_query(preprocessed.tokens)
        top_k = _memory_fallback_top_k(inp, options)
        query = MemoryDbSearchQuery(
            query=inp.query,
            top_k=top_k or self._search_config.default.top_k,
            filters=filters,
            mode="bm25",
            ranking="score",
        )
        result = await self.db_reader.search_sparse(
            context,
            query,
            indices=list(sparse.indices),
            values=list(sparse.values),
        )
        return [_to_memory_search_item(hit) for hit in result.hits]


def _optional_rerank_client() -> RerankClient | None:
    try:
        from ....llm import get_rerank_client

        return get_rerank_client()
    except Exception:
        return None


def _to_memory_search_item(hit: MemoryDbSearchHit) -> MemorySearchItem:
    memory = hit.memory
    return MemorySearchItem(
        id=hit.memory_id,
        memory=memory.content if memory else "",
        memory_type=memory.mem_type if memory else "fact",
        last_update_at=format_datetime((memory.update_at or memory.created_at) if memory else None),
        event_time=format_memory_event_time(memory, fallback_to_source_timestamp=True) if memory else None,
        source_timestamp=format_source_timestamp(memory) if memory else None,
    )


def _memory_fallback_top_k(inp: SearchPipelineInput, options: SearchEngineOptions | None) -> int | None:
    if options and options.recall_top_k is not None:
        return options.recall_top_k
    if options and options.result_top_n is not None:
        return options.result_top_n
    return inp.top_k
