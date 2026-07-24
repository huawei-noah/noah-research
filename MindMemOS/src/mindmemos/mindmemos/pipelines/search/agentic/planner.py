"""Query planning helpers for agentic search."""

from __future__ import annotations

import json
from typing import Any, Callable

from ....components.memory_modeling.schema import TemporalEntity
from ....llm import LLMClient
from ....logging import get_logger
from ....prompts import SearchPromptSet
from .base import ask_json

logger = get_logger(__name__)


class LLMAgenticPlanner:
    """Generate follow-up queries for the next agentic retrieval round."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: SearchPromptSet,
        format_entities: Callable[[list[TemporalEntity]], str],
        enforce_min_time_window: Callable[[tuple[str, str] | None], tuple[str, str] | None],
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._format_entities = format_entities
        self._enforce_min_time_window = enforce_min_time_window

    async def generate_next_queries(
        self,
        *,
        original_query: str,
        all_entities: list[TemporalEntity],
        missing_info: list[str],
        query_history: list[str],
    ) -> list[dict[str, Any]]:
        """Generate improved queries for the next retrieval round."""

        if not missing_info:
            return []

        memory_text = self._format_entities(all_entities)
        prompt = self._prompts.multi_query_generation.format(
            original_query=original_query,
            retrieved_docs=memory_text,
            missing_info=json.dumps(missing_info, ensure_ascii=False),
        )

        try:
            result = await ask_json(self._llm, "search.multi_query", prompt)
            queries = result.get("queries", []) if isinstance(result, dict) else []
            valid_queries = self._normalize_queries(queries, query_history)
            new_queries = [query for query in valid_queries if query["is_new"]]
            selected = new_queries or valid_queries
            return [{"query": query["query"], "time_range": query["time_range"]} for query in selected]
        except Exception as exc:
            logger.error("agentic_query_generation_failed", error=str(exc))
            fallback = simple_rephrase(original_query, query_history)
            return [fallback] if fallback else []

    def _normalize_queries(self, queries: list[dict[str, Any]], query_history: list[str]) -> list[dict[str, Any]]:
        valid_queries: list[dict[str, Any]] = []
        for item in queries:
            query_str = item.get("query") if isinstance(item, dict) else None
            time_range = item.get("time_range") if isinstance(item, dict) else None
            if not query_str:
                continue
            if time_range and isinstance(time_range, list) and len(time_range) == 2:
                time_range_tuple = self._enforce_min_time_window((time_range[0], time_range[1]))
            else:
                time_range_tuple = None
            valid_queries.append(
                {
                    "query": query_str,
                    "time_range": time_range_tuple,
                    "is_new": query_str not in query_history,
                }
            )
        return valid_queries


def simple_rephrase(query: str, history: list[str]) -> dict[str, Any] | None:
    """Simple query rephrasing as a fallback when LLM query generation fails."""

    variations = [
        f"{query} specific time",
        f"{query} related people",
        f"{query} detailed events",
        query.replace("why", "reason").replace("how", "method"),
        f"more information about {query}",
    ]
    for value in variations:
        if value not in history:
            return {"query": value, "time_range": None}
    return None
