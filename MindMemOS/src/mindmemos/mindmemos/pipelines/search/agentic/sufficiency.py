"""Sufficiency and relevance checks for agentic search."""

from __future__ import annotations

import asyncio
from typing import Callable

from ....components.extractor.schema import parse_json_object
from ....components.memory_modeling.schema import TemporalEntity
from ....llm import LLMClient
from ....logging import get_logger
from ....prompts import SearchPromptSet
from .base import ask_json

logger = get_logger(__name__)


class LLMSufficiencyEvaluator:
    """Evaluate whether retrieved information is enough for the original query."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: SearchPromptSet,
        format_entities: Callable[[list[TemporalEntity], int | None], str],
        output_max_edge_num: int,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._format_entities = format_entities
        self._output_max_edge_num = output_max_edge_num

    async def evaluate_sufficiency(
        self,
        *,
        user_query: str,
        retrieved_entities: list[TemporalEntity],
    ) -> tuple[bool, str, list[str]]:
        """Evaluate whether retrieved information is sufficient to answer the query."""

        memory_text = self._format_entities(retrieved_entities, self._output_max_edge_num)

        prompt = self._prompts.sufficiency_check.format(
            query=user_query,
            retrieved_docs=memory_text,
        )

        try:
            result = await ask_json(self._llm, "search.sufficiency_check", prompt)
            is_sufficient: bool = result["is_sufficient"]
            reasoning: str = result["reasoning"]
            missing: list[str] = result["missing_information"]
            return is_sufficient, reasoning, missing
        except Exception as exc:
            logger.error("agentic_sufficiency_eval_failed", error=str(exc))
            return False, "Evaluation failed", ["Unable to determine if information is sufficient"]

    async def filter_entities_by_relevance(
        self,
        entities: list[TemporalEntity],
        query: str,
    ) -> tuple[list[TemporalEntity], list[TemporalEntity]]:
        """Use LLM to concurrently filter out irrelevant entities."""

        if not entities or not query:
            return entities, []

        logger.info("agentic_relevance_filter_start", entity_count=len(entities))

        try:
            relevant, filtered_out = await self._concurrent_filter_entities(entities, query)
            logger.info(
                "agentic_relevance_filter_complete",
                total=len(entities),
                relevant=len(relevant),
                filtered=len(filtered_out),
            )
            return relevant, filtered_out
        except Exception as exc:
            logger.error("agentic_relevance_filter_failed", error=str(exc))
            return entities, []

    async def _concurrent_filter_entities(
        self,
        entities: list[TemporalEntity],
        query: str,
    ) -> tuple[list[TemporalEntity], list[TemporalEntity]]:
        """Concurrently judge entity relevance with the LLM."""

        relevance_tasks = [self._judge_entity_relevance_enhanced(entity, query, i) for i, entity in enumerate(entities)]

        relevance_results = await asyncio.gather(*relevance_tasks, return_exceptions=True)

        relevant_entities: list[TemporalEntity] = []
        filtered_out_entities: list[TemporalEntity] = []

        for index, (entity, result) in enumerate(zip(entities, relevance_results)):
            if isinstance(result, Exception):
                logger.warning(
                    "agentic_relevance_judge_failed",
                    index=index + 1,
                    entity=entity.name,
                    error=str(result),
                )
                relevant_entities.append(entity)
            elif result is True:
                relevant_entities.append(entity)
            else:
                filtered_out_entities.append(entity)

        return relevant_entities, filtered_out_entities

    async def _judge_entity_relevance_enhanced(
        self,
        entity: TemporalEntity,
        query: str,
        entity_index: int,
    ) -> bool:
        """Judge a single entity's relevance via LLM; default to keeping on failure."""

        try:
            entity_description = entity.format_entity_prompt()

            prompt = self._prompts.entity_relevance_filter.format(
                query=query,
                entity_description=entity_description,
            )

            resp = await self._llm.chat(
                task="search.relevance_filter",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )

            response = resp.parsed if isinstance(resp.parsed, dict) else {}
            relevance = response.get("relevance", "yes").lower()
            reasoning = response.get("reasoning", "")

            if relevance == "no":
                logger.info(
                    "agentic_relevance_removed",
                    index=entity_index + 1,
                    entity=entity.name,
                    reason=reasoning,
                )
                return False

            logger.info(
                "agentic_relevance_kept",
                index=entity_index + 1,
                entity=entity.name,
                reason=reasoning,
            )
            return True

        except Exception as exc:
            logger.warning(
                "agentic_relevance_judge_error",
                index=entity_index + 1,
                entity=entity.name,
                error=str(exc),
            )
            return True
