"""Schema-mode search-field generation."""

from __future__ import annotations

from typing import Any

from ....llm import LLMClient
from ....logging import get_logger
from ....prompts import AddPromptSet
from ._schema_utils import dedupe_non_empty, parse_json_object
from .base import SchemaSearchFieldExtractorProtocol

logger = get_logger(__name__)


class SchemaSearchFieldExtractor(SchemaSearchFieldExtractorProtocol):
    """Generate compact search fields from schema entity attributes."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        prompt_set: AddPromptSet | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._prompt_set = prompt_set

    async def extract_search_fields(
        self,
        *,
        entities: list[dict[str, Any]],
        context_text: str,
        max_fields: int,
        augment: bool = False,
        augment_count: int = 0,
        fallback_text: str | None = None,
        prompt_set: AddPromptSet | None = None,
    ) -> list[str]:
        """Build deduplicated schema search fields from entity properties."""

        raw_fields = _entity_field_values(entities)
        if not raw_fields and fallback_text:
            raw_fields.append(fallback_text)
        fields = dedupe_non_empty(raw_fields)[:max_fields]
        if not fields:
            return []

        if augment and self._llm_client is not None and augment_count > 0:
            effective_prompts = prompt_set or self._prompt_set
            if effective_prompts is not None:
                fields.extend(
                    await self._augment_fields(
                        fields, context_text=context_text, augment_count=augment_count, prompt_set=effective_prompts
                    )
                )

        return dedupe_non_empty(fields)[:max_fields]

    async def _augment_fields(
        self,
        fields: list[str],
        *,
        context_text: str,
        augment_count: int,
        prompt_set: AddPromptSet | None = None,
    ) -> list[str]:
        prompts = prompt_set or self._prompt_set
        if prompts is None:
            return []
        prompt = (
            prompts.episode_search_field_augment.replace("{episode_text}", context_text[:4000])
            .replace("{existing_fields}", "\n".join(f"- {field}" for field in fields))
            .replace("{augment_count}", str(augment_count))
        )
        try:
            response = await self._llm_client.chat(
                task="memory.add.search_field_augment",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )
        except Exception:
            logger.warning("search_field_augmentation_failed", exc_info=True)
            return []
        parsed = response.parsed
        if not isinstance(parsed, list):
            return []
        return [str(field) for field in parsed if field]


def _entity_field_values(entities: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for entity in entities:
        if entity.get("entity_type") == "episodes":
            continue
        properties = _entity_properties(entity)
        if properties:
            values.extend(str(prop.get("value") or "")[:150] for prop in properties if prop.get("value"))
            continue
        description = str(entity.get("description") or "")
        if description:
            values.append(description[:200])
    return values


def _entity_properties(entity: dict[str, Any]) -> list[dict[str, Any]]:
    raw_properties = entity.get("properties", [])
    if not isinstance(raw_properties, list):
        return []
    return [
        prop
        for prop in raw_properties
        if isinstance(prop, dict) and prop.get("property_name") and prop.get("property_name") != "input_messages"
    ]
