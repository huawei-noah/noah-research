"""Property-level similarity search for schema add merge decisions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ....logging import get_logger
from ....typing import MemoryRequestContext, MemoryView

logger = get_logger(__name__)

EmbedTexts = Callable[[str, list[str]], Awaitable[list[list[float]]]]


@dataclass(slots=True)
class PropertyMemoryMatch:
    """One stored property memory and its retrieval score."""

    memory: MemoryView
    score: float


@dataclass(slots=True)
class PropertySimilarityMatch:
    """One new property and its similar stored property matches."""

    property: dict[str, Any]
    matches: list[PropertyMemoryMatch]


@dataclass(slots=True)
class SchemaPropertySimilaritySearcher:
    """Search stored property memories for values similar to new schema properties."""

    db_reader: Any
    embed_texts: EmbedTexts

    async def find_for_merge(
        self,
        *,
        context: MemoryRequestContext,
        entity_id: str,
        properties: list[dict[str, Any]],
        limit: int = 5,
    ) -> list[PropertySimilarityMatch]:
        """Find active stored memories similar to each new property."""

        prop_texts = [self._property_text(prop) for prop in properties]
        prop_vectors = await self.embed_texts("memory.add.property_merge", prop_texts)
        results = await _gather_matches(
            [
                self._find_property_matches(
                    context=context,
                    entity_id=entity_id,
                    prop=prop,
                    vector=vector,
                    limit=limit,
                )
                for prop, vector in zip(properties, prop_vectors, strict=True)
            ]
        )
        return results

    async def find_delete_candidates(
        self,
        *,
        context: MemoryRequestContext,
        entity_id: str,
        prop: dict[str, Any],
        limit: int = 5,
    ) -> list[PropertyMemoryMatch]:
        """Find active stored memories similar to one delete operation."""

        value = str(prop.get("value") or prop.get("property_name") or "")
        if not value:
            return []
        try:
            vectors = await self.embed_texts("memory.add.property_delete", [value])
            if not vectors:
                return []
            return await self._search_active_memories(
                context=context,
                entity_id=entity_id,
                vector=vectors[0],
                limit=limit,
            )
        except Exception:
            logger.warning("property delete similarity search failed", exc_info=True)
            return []

    async def _find_property_matches(
        self,
        *,
        context: MemoryRequestContext,
        entity_id: str,
        prop: dict[str, Any],
        vector: list[float],
        limit: int,
    ) -> PropertySimilarityMatch:
        if not vector or not prop.get("value"):
            return PropertySimilarityMatch(property=prop, matches=[])
        try:
            matches = await self._search_active_memories(
                context=context,
                entity_id=entity_id,
                vector=vector,
                limit=limit,
            )
            return PropertySimilarityMatch(property=prop, matches=matches)
        except Exception:
            logger.warning("property similarity search failed", exc_info=True)
            return PropertySimilarityMatch(property=prop, matches=[])

    async def _search_active_memories(
        self,
        *,
        context: MemoryRequestContext,
        entity_id: str,
        vector: list[float],
        limit: int,
    ) -> list[PropertyMemoryMatch]:
        result = await self.db_reader.search_entity_property_memories(
            context,
            query_vector=vector,
            entity_id=entity_id,
            limit=limit,
        )
        return [
            PropertyMemoryMatch(memory=hit.memory, score=hit.score)
            for hit in result.hits
            if hit.memory and hit.memory.status == "active"
        ]

    @staticmethod
    def _property_text(prop: dict[str, Any]) -> str:
        return f"{prop.get('property_name', '')}: {prop.get('value', '')}"


async def _gather_matches(tasks: list[Awaitable[PropertySimilarityMatch]]) -> list[PropertySimilarityMatch]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, PropertySimilarityMatch)]
