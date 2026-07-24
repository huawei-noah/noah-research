"""Query planning helpers for schema-aware search."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ....llm import LLMClient
from ....logging import get_logger
from ....prompts import SearchPromptSet
from ...extractor.schema._schema_utils import parse_json_object

logger = get_logger(__name__)


class SchemaSearchQueryBuilder:
    """Build schema-aware property filters and time windows."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: SearchPromptSet,
        entity_schema: list[dict[str, Any]],
        current_time_mode: str,
        min_time_window_days: int | None,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._entity_schema = entity_schema
        self._current_time_mode = current_time_mode
        self._min_time_window_days = min_time_window_days

    async def select_property_filter(
        self,
        user_query: str,
        *,
        prompts: SearchPromptSet | None = None,
    ) -> dict[str, list[str]]:
        """Use the LLM to select relevant entity types and properties."""

        entity_all_props: dict[str, list[str]] = {}
        for entity_info in self._entity_schema:
            entity_type = entity_info.get("entity_type", "")
            all_props = list(entity_info.get("static_property", {}).keys()) + list(
                entity_info.get("dynamic_property", {}).keys()
            )
            entity_all_props[entity_type] = all_props

        prompts_to_use = prompts or self._prompts
        non_episode_schema = [e for e in self._entity_schema if e.get("entity_type") != "episodes"]
        prompt = prompts_to_use.property_filter_selection.replace("{query}", user_query).replace(
            "{entity_schema}",
            str(non_episode_schema),
        )

        try:
            result = await self._ask_json("search.property_filter", prompt)
            if "selected_entities" not in result:
                return result
            converted: dict[str, list[str]] = {}
            for item in result.get("selected_entities", []):
                entity_type = item.get("entity_type")
                if not entity_type or entity_type == "episodes":
                    continue
                props = item.get("relevant_properties", [])
                if "all" in props:
                    props = entity_all_props.get(entity_type, [])
                if "default_property" not in props and "default_property" in entity_all_props.get(entity_type, []):
                    props.append("default_property")
                converted[entity_type] = props
            converted["episodes"] = entity_all_props.get("episodes", [])
            logger.info("schema_search_property_filter", entity_types=list(converted))
            return converted
        except Exception as exc:
            logger.error("schema_search_property_filter_failed", error=str(exc))
            return {}

    def all_property_filter(self) -> dict[str, list[str]]:
        """Return all entity types and all their properties."""

        result: dict[str, list[str]] = {}
        for entity_info in self._entity_schema:
            entity_type = entity_info.get("entity_type", "")
            all_props = list(entity_info.get("static_property", {}).keys()) + list(
                entity_info.get("dynamic_property", {}).keys()
            )
            result[entity_type] = all_props
        return result

    async def extract_time_from_query(
        self,
        user_query: str,
        *,
        prompts: SearchPromptSet | None = None,
    ) -> tuple[str, str] | None:
        """Extract and normalize temporal constraints from the user query."""

        try:
            if self._current_time_mode == "system":
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                current_time = "unknown"

            prompts_to_use = prompts or self._prompts
            prompt = prompts_to_use.time_extraction.format(query=user_query, current_time=current_time)
            result = await self._ask_json("search.time_extraction", prompt)
            time_range = result.get("time_range")
            if time_range and isinstance(time_range, list) and len(time_range) == 2:
                raw_range = (time_range[0], time_range[1])
                return self.enforce_min_time_window(raw_range)
            return None
        except Exception as exc:
            logger.warning("schema_search_time_extraction_failed", error=str(exc))
            return None

    def enforce_min_time_window(self, time_range: tuple[str, str] | None) -> tuple[str, str] | None:
        """Ensure the time window is at least the configured minimum width."""

        if time_range is None or not self._min_time_window_days:
            return time_range
        try:
            start = datetime.strptime(time_range[0], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(time_range[1], "%Y-%m-%d %H:%M:%S")
            min_delta = timedelta(days=self._min_time_window_days)
            actual_delta = end - start
            if actual_delta < min_delta:
                mid = start + actual_delta / 2
                new_start = mid - min_delta / 2
                new_end = mid + min_delta / 2
                return (new_start.strftime("%Y-%m-%d %H:%M:%S"), new_end.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as exc:
            logger.warning("schema_search_time_window_expand_failed", error=str(exc))
        return time_range

    async def _ask_json(self, task: str, prompt: str) -> Any:
        response = await self._llm.chat(
            task=task,
            messages=[{"role": "user", "content": prompt}],
            format_parser=parse_json_object,
        )
        return response.parsed
