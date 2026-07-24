"""Agentic search wrapper around a selected search engine."""

from __future__ import annotations

from copy import copy
from dataclasses import is_dataclass, replace
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ....components.memory_modeling.schema import TemporalEntity
from ....components.searcher.schema import SchemaSearchRanker
from ....components.text import detect_prompt_language
from ....config import get_config
from ....config.algo.search import AgenticConfig
from ....llm import LLMClient, get_llm_client
from ....mappers import parse_schema_search_filters
from ....prompts import SearchPromptSet, get_search_prompts
from ....typing import MemoryRequestContext, MemorySearchItem, SearchPipelineInput
from ..base import SearchEngine, SearchEngineOptions
from .base import SearchTool, SearchToolRequest, SearchToolResult
from .loop import AgenticLoop
from .planner import LLMAgenticPlanner
from .sufficiency import LLMSufficiencyEvaluator
from .tool_router import DefaultToolRouter


class AgenticSearchWrapper:
    """Run multi-round planning around a single search engine."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        config: AgenticConfig | DictConfig | dict[str, Any] | None = None,
        prompts: SearchPromptSet | None = None,
    ) -> None:
        cfg = get_config()
        self._config = config or cfg.algo_config.search.agentic
        self._llm = llm_client or get_llm_client()
        self._prompts = prompts or get_search_prompts(cfg.algo_config.common.prompt_language)
        self._ranker = SchemaSearchRanker()

    async def run(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        engine: SearchEngine,
    ) -> list[MemorySearchItem]:
        """Run the agentic loop and return merged candidates."""

        detected_lang = detect_prompt_language(
            inp.query,
            fallback=get_config().algo_config.common.prompt_language,
        )
        request_prompts = get_search_prompts(detected_lang)

        agentic_config = _agentic_config_with_max_rounds(self._config, inp.max_rounds)
        tool = EngineSearchTool(
            engine=engine,
            template=inp,
            recall_top_k=agentic_config.top_k_per_round,
            result_top_n=agentic_config.top_n_per_round,
            use_reranker=agentic_config.use_rerank,
        )
        router = DefaultToolRouter(
            tools={tool.name: tool},
            enabled_tools=[tool.name],
            default_tool=tool.name,
        )
        planner = LLMAgenticPlanner(
            llm=self._llm,
            prompts=request_prompts,
            format_entities=lambda entities: self._ranker.format_entities_for_prompt(
                entities,
                include_edges=agentic_config.include_edges,
            ),
            enforce_min_time_window=lambda value: value,
        )
        sufficiency = LLMSufficiencyEvaluator(
            llm=self._llm,
            prompts=request_prompts,
            format_entities=lambda entities, max_edge_num=None: self._ranker.format_entities_for_prompt(
                entities,
                max_edge_num=max_edge_num,
                include_edges=agentic_config.include_edges,
            ),
            output_max_edge_num=agentic_config.output_max_edge_num,
        )
        loop = AgenticLoop(
            config=agentic_config,
            tool_router=router,
            planner=planner,
            sufficiency=sufficiency,
            ranker=self._ranker,
        )
        parsed_filters = parse_schema_search_filters(inp.filters, context)
        entities = await loop.run(
            query=inp.query,
            context=parsed_filters.context,
            filters=inp.filters,
            search_filter=parsed_filters.memory_filter,
            entity_search_filter=parsed_filters.entity_filter,
            allow_time_extraction=False,
        )
        return [
            MemorySearchItem(
                id=entity.entity_id,
                memory=entity.description
                or entity.format_entity_prompt(
                    ignore_edge_num=agentic_config.output_max_edge_num,
                    include_description=False,
                    include_edges=agentic_config.include_edges,
                ),
                memory_type="fact",
                last_update_at="",
            )
            for entity in entities
        ]


class EngineSearchTool(SearchTool):
    """Adapt a SearchEngine to the AgenticLoop tool protocol."""

    def __init__(
        self,
        *,
        engine: SearchEngine,
        template: SearchPipelineInput,
        recall_top_k: int,
        result_top_n: int,
        use_reranker: bool,
    ) -> None:
        self._engine = engine
        self._template = template
        self._recall_top_k = recall_top_k
        self._result_top_n = result_top_n
        self._use_reranker = use_reranker
        self.name = engine.name

    async def search(self, request: SearchToolRequest) -> SearchToolResult:
        """Run the wrapped search engine for one agentic tool request.

        Args:
            request: Query, filters, context, and retrieval options.

        Returns:
            Search tool results with ranked candidates.
        """
        filters = request.filters if self._engine.name == "schema" else _drop_schema_only_filter_fields(request.filters)
        inp = self._template.model_copy(
            update={
                "query": request.query,
                "filters": filters,
                "top_k": self._result_top_n,
                "agentic": False,
                "rerank": False,
            }
        )
        candidates = await self._engine.search_candidates(
            inp,
            request.context,
            options=SearchEngineOptions(
                num_hops=request.num_hops,
                recall_top_k=self._recall_top_k,
                result_top_n=self._result_top_n,
                use_reranker=self._use_reranker,
            ),
        )
        entities: list[TemporalEntity] = []
        for item in candidates:
            entity = TemporalEntity(entity_id=item.id, name=item.id, entity_type="memory", description=item.memory)
            entity.modify_property("search_result", item.memory, item.last_update_at or "", uid=item.id)
            entities.append(entity)
        return SearchToolResult(entities=entities, debug={"tool": self.name})


def _drop_schema_only_filter_fields(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    cleaned = _drop_field(filters, "project_id")
    return cleaned if isinstance(cleaned, dict) and cleaned else None


def _drop_field(node: Any, field: str) -> Any:
    if isinstance(node, list):
        return [item for value in node if (item := _drop_field(value, field))]
    if not isinstance(node, dict):
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key == field:
            continue
        if key in {"AND", "OR", "NOT"}:
            nested = _drop_field(value, field)
            if nested:
                cleaned[key] = nested
            continue
        cleaned[key] = value
    return cleaned


def _agentic_config_with_max_rounds(config: Any, max_rounds: int) -> Any:
    if is_dataclass(config) and not isinstance(config, type):
        return replace(config, max_rounds=max_rounds)
    if isinstance(config, DictConfig):
        cloned = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cloned.max_rounds = max_rounds
        return cloned
    if isinstance(config, dict):
        cloned = OmegaConf.create(config)
        cloned.max_rounds = max_rounds
        return cloned
    cloned = copy(config)
    cloned.max_rounds = max_rounds
    return cloned
