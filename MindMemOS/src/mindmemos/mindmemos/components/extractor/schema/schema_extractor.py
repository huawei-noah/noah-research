"""Prompt-driven schema add extraction operators."""

from __future__ import annotations

import copy
import json
from typing import Any

from ....llm import LLMClient
from ....logging import get_logger
from ....prompts import AddPromptSet
from ...memory_modeling.schema import EntitySchemaProvider
from ._schema_utils import (
    build_filtered_schema,
    format_schema_summary,
    has_unique_entity_names,
    parse_json_object,
    strip_for_generation,
)
from .base import SchemaEpisodeExtractor
from .schema_normalizer import SchemaExtractionNormalizer

logger = get_logger(__name__)


class SchemaAddExtractor(SchemaEpisodeExtractor):
    """Run prompt-driven schema add extraction steps."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        prompt_set: AddPromptSet,
        entity_manager: EntitySchemaProvider,
        enable_schema_selection: bool,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_set = prompt_set
        self.entity_manager = entity_manager
        self.enable_schema_selection = enable_schema_selection
        self.normalizer = SchemaExtractionNormalizer(entity_manager=entity_manager)

    async def extract_episode(
        self,
        *,
        conversation_text: str,
        dialogue_timestamp: str,
    ) -> dict[str, Any]:
        schema = self.schema_for_generation()
        selected_schema = await self.select_schema(conversation_text, schema)
        raw_memory = await self.extract_memory(
            entity_schema=selected_schema,
            dialogue_timestamp=dialogue_timestamp,
            conversation_text=conversation_text,
        )
        return self.prepare_raw_memory(raw_memory, dialogue_timestamp)

    async def select_schema(
        self,
        conversation_text: str,
        full_schema: list[dict[str, Any]],
        *,
        prompt_set: AddPromptSet | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enable_schema_selection:
            return full_schema
        try:
            return await self._select_schema(conversation_text, full_schema, prompt_set=prompt_set)
        except Exception:
            logger.warning("schema selection failed; using full schema", exc_info=True)
            return full_schema

    async def _select_schema(
        self,
        conversation_text: str,
        full_schema: list[dict[str, Any]],
        *,
        prompt_set: AddPromptSet | None = None,
    ) -> list[dict[str, Any]]:
        prompts = prompt_set or self.prompt_set
        schema_summary = format_schema_summary(full_schema)
        prompt = prompts.schema_selection_for_generation.format(
            dialogue_text=conversation_text[:2000],
            entity_schema=schema_summary,
        )
        response = await self.llm_client.chat(
            task="memory.add.schema_selection",
            messages=[{"role": "user", "content": prompt}],
            format_parser=parse_json_object,
        )
        selected = response.parsed.get("selected_entities", []) if isinstance(response.parsed, dict) else []
        filtered = build_filtered_schema(full_schema, selected)
        return filtered or full_schema

    async def extract_memory(
        self,
        *,
        entity_schema: list[dict[str, Any]],
        dialogue_timestamp: str,
        conversation_text: str,
        prompt_set: AddPromptSet | None = None,
    ) -> dict[str, Any]:
        prompts = prompt_set or self.prompt_set
        prompt = (
            prompts.entity_generation.replace("{entity_schema}", str(entity_schema))
            .replace("{dialogue_timestamp}", dialogue_timestamp)
            .replace("{chat_chunk}", conversation_text)
        )

        last_memory: dict[str, Any] | None = None
        for _ in range(3):
            response = await self.llm_client.chat(
                task="memory.add.entity_generation",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )
            raw_memory = response.parsed
            if not isinstance(raw_memory, dict):
                raw_memory = {"entities": [], "edges": []}
            last_memory = raw_memory
            validation_error = self.validate_memory(raw_memory)
            if not validation_error and has_unique_entity_names(raw_memory):
                return raw_memory
            prompt += (
                "\nPrevious answer: "
                + json.dumps(raw_memory, ensure_ascii=False)
                + f"\nERROR: {validation_error or 'There are entities with duplicate names. Please merge them.'}"
            )
        return last_memory or {"entities": [], "edges": []}

    async def objectify_conversation(
        self,
        conversation_text: str,
        conversation_timestamp: str,
        *,
        prompt_set: AddPromptSet | None = None,
    ) -> str:
        prompts = prompt_set or self.prompt_set
        prompt = prompts.episode_objectify.replace("{conversation_text}", conversation_text).replace(
            "{conversation_timestamp}", conversation_timestamp
        )
        try:
            response = await self.llm_client.chat(
                task="memory.add.episode_objectify",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content.strip()
        except Exception:
            logger.warning("episode objectify failed; using original conversation", exc_info=True)
            return conversation_text
        if "**OBJECTIVE DESCRIPTION:**" in content:
            content = content.split("**OBJECTIVE DESCRIPTION:**")[-1].strip()
        if len(content) < 20:
            return conversation_text
        return content

    async def generate_episode_description(
        self,
        conversation_text: str,
        conversation_timestamp: str,
        *,
        prompt_set: AddPromptSet | None = None,
    ) -> str:
        prompts = prompt_set or self.prompt_set
        prompt = prompts.episode_description.replace("{conversation_text}", conversation_text).replace(
            "{conversation_timestamp}", conversation_timestamp
        )
        try:
            response = await self.llm_client.chat(
                task="memory.add.episode_description",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )
            parsed = response.parsed
            if isinstance(parsed, dict) and "title" in parsed and "content" in parsed:
                return f"{parsed['title']}\n{parsed['content']}"
        except Exception:
            logger.warning("episode description generation failed; using original conversation", exc_info=True)
        return conversation_text

    def schema_for_generation(self) -> list[dict[str, Any]]:
        schema = copy.deepcopy(self.entity_manager.get_all_dicts())
        return strip_for_generation(schema)

    def prepare_raw_memory(self, raw_memory: dict[str, Any], dialogue_timestamp: str) -> dict[str, Any]:
        return self.normalizer.normalize(raw_memory, dialogue_timestamp)

    def validate_memory(self, raw_memory: dict[str, Any]) -> str | None:
        return self.normalizer.validate(raw_memory)
