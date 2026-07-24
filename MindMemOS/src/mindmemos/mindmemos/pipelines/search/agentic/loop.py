"""Multi-round agentic search loop."""

from __future__ import annotations

import time as _time
from typing import Any

from ....components.memory_modeling.schema import TemporalEntity
from ....components.searcher.schema import SchemaSearchRanker
from ....config.algo.search import AgenticConfig
from ....logging import get_logger
from ....typing import MemoryRequestContext, SearchFilter
from .base import (
    AgenticPlanner,
    AgenticQuery,
    SearchToolRequest,
    SearchToolRouter,
    SufficiencyEvaluator,
)
from .planner import simple_rephrase

logger = get_logger(__name__)


class AgenticLoop:
    """Coordinate round control, tool invocation, sufficiency checks, and query planning."""

    def __init__(
        self,
        *,
        config: AgenticConfig,
        tool_router: SearchToolRouter,
        planner: AgenticPlanner,
        sufficiency: SufficiencyEvaluator,
        ranker: SchemaSearchRanker | None = None,
    ) -> None:
        self._config = config
        self._tool_router = tool_router
        self._planner = planner
        self._sufficiency = sufficiency
        self._ranker = ranker or SchemaSearchRanker()

    async def run(
        self,
        *,
        query: str,
        context: MemoryRequestContext,
        initial_time_window: tuple[str, str] | None = None,
        filters: dict[str, Any] | None = None,
        search_filter: SearchFilter | None = None,
        entity_search_filter: SearchFilter | None = None,
        allow_time_extraction: bool = True,
    ) -> list[TemporalEntity]:
        """Run the configured multi-round agentic search."""

        all_entities: dict[str, TemporalEntity] = {}
        query_history: list[str] = [query]
        current_queries = [
            AgenticQuery(
                query=query,
                time_window=initial_time_window,
                num_hops=2,
                allow_time_extraction=allow_time_extraction,
            ),
        ]

        round_num = 1
        while round_num <= self._config.max_rounds and current_queries:
            round_t0 = _time.monotonic()
            logger.info(
                "agentic_round_start",
                round=round_num,
                queries=[(item.query, item.time_window, item.tool_name) for item in current_queries],
            )

            round_entities = await self._run_queries(
                query,
                context,
                current_queries,
                filters=filters,
                search_filter=search_filter,
                entity_search_filter=entity_search_filter,
            )
            for entity in round_entities:
                if entity.entity_id not in all_entities:
                    all_entities[entity.entity_id] = entity
                else:
                    all_entities[entity.entity_id] = self._ranker.merge_entity_properties_and_edges(
                        all_entities[entity.entity_id],
                        entity,
                    )

            total_props = sum(sum(len(timeline) for timeline in e._properties.values()) for e in all_entities.values())
            logger.info(
                "agentic_round_summary",
                round=round_num,
                new_entities=len(round_entities),
                total_entities=len(all_entities),
                total_properties=total_props,
                search_time_s=round(_time.monotonic() - round_t0, 2),
            )

            if round_num >= self._config.max_rounds:
                break

            is_sufficient, reasoning, missing = await self._sufficiency.evaluate_sufficiency(
                user_query=query,
                retrieved_entities=list(all_entities.values()),
            )
            logger.info(
                "agentic_round_sufficiency",
                round=round_num,
                is_sufficient=is_sufficient,
                reasoning=reasoning,
                missing=missing,
            )
            if is_sufficient or not missing:
                break

            next_queries_info = await self._planner.generate_next_queries(
                original_query=query,
                all_entities=list(all_entities.values()),
                missing_info=missing,
                query_history=query_history,
            )
            current_queries = self._next_round_queries(
                query,
                query_history,
                next_queries_info,
                round_num=round_num,
                original_time_window=initial_time_window,
                allow_time_extraction=allow_time_extraction,
            )
            if not current_queries:
                break
            round_num += 1

        final_entities = list(all_entities.values())
        if final_entities and self._config.use_relevance_filter:
            final_entities, relevance_filtered = await self._sufficiency.filter_entities_by_relevance(
                final_entities,
                query,
            )
            logger.info("agentic_relevance_filter", kept=len(final_entities), removed=len(relevance_filtered))

        final_entities, empty_entities = self._ranker.filter_empty_properties(final_entities)
        if empty_entities:
            logger.info("agentic_empty_filter", removed=len(empty_entities), after=len(final_entities))
        final_entities.sort(key=lambda e: (e.entity_type == "episodes",))
        return final_entities

    async def _run_queries(
        self,
        original_query: str,
        context: MemoryRequestContext,
        queries: list[AgenticQuery],
        *,
        filters: dict[str, Any] | None = None,
        search_filter: SearchFilter | None = None,
        entity_search_filter: SearchFilter | None = None,
    ) -> list[TemporalEntity]:
        results: list[TemporalEntity] = []
        for query in queries:
            tool = self._tool_router.select(query.tool_name)
            tool_result = await tool.search(
                SearchToolRequest(
                    query=query.query,
                    original_query=original_query,
                    time_window=query.time_window,
                    num_hops=query.num_hops,
                    context=context,
                    allow_time_extraction=query.allow_time_extraction,
                    filters=filters,
                    search_filter=search_filter,
                    entity_search_filter=entity_search_filter,
                )
            )
            results.extend(tool_result.entities)
        return results

    def _next_round_queries(
        self,
        original_query: str,
        query_history: list[str],
        next_queries_info: list[dict],
        *,
        round_num: int,
        original_time_window: tuple[str, str] | None,
        allow_time_extraction: bool,
    ) -> list[AgenticQuery]:
        next_queries: list[AgenticQuery] = []
        for query_info in next_queries_info:
            q = query_info["query"]
            if q in query_history:
                continue
            query_history.append(q)
            next_queries.append(
                AgenticQuery(
                    query=q,
                    time_window=query_info.get("time_range") if allow_time_extraction else None,
                    num_hops=1,
                    allow_time_extraction=allow_time_extraction,
                ),
            )
        if next_queries:
            self._append_time_relaxed_query(
                next_queries,
                original_query=original_query,
                query_history=query_history,
                round_num=round_num,
                original_time_window=original_time_window,
            )
            return next_queries
        fallback = simple_rephrase(original_query, query_history)
        if not fallback:
            return []
        query_history.append(fallback["query"])
        next_queries = [
            AgenticQuery(
                query=fallback["query"],
                time_window=fallback.get("time_range") if allow_time_extraction else None,
                num_hops=1,
                allow_time_extraction=allow_time_extraction,
            )
        ]
        self._append_time_relaxed_query(
            next_queries,
            original_query=original_query,
            query_history=query_history,
            round_num=round_num,
            original_time_window=original_time_window,
        )
        return next_queries

    def _append_time_relaxed_query(
        self,
        queries: list[AgenticQuery],
        *,
        original_query: str,
        query_history: list[str],
        round_num: int,
        original_time_window: tuple[str, str] | None,
    ) -> None:
        if round_num < 2 or original_time_window is None or not queries:
            return
        if any(query.time_window is None and not query.allow_time_extraction for query in queries):
            return
        relaxed_key = f"{original_query} (time-relaxed)"
        if relaxed_key in query_history:
            return
        query_history.append(relaxed_key)
        queries.append(
            AgenticQuery(
                query=original_query,
                time_window=None,
                num_hops=1,
                allow_time_extraction=False,
            )
        )
        logger.info("agentic_time_relaxed_query_added", next_round=round_num + 1)
