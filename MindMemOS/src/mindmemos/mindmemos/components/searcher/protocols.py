"""Searcher component protocols."""

from __future__ import annotations

from typing import Any, Protocol

from ...typing import MemoryRequestContext, SearchFilter
from ..memory_modeling.schema import TemporalEntity


class EntityRecallStrategy(Protocol):
    """Protocol for reusable entity-store recall."""

    async def recall_entities(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        filters: SearchFilter | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]: ...


class SearchStrategy(Protocol):
    """Protocol for end-to-end search components."""

    async def search(self, ctx: MemoryRequestContext, query: str, **kwargs: Any) -> list[TemporalEntity]: ...


class EntityHydrator(Protocol):
    """Protocol for entity hydration components."""

    async def hydrate(self, ctx: MemoryRequestContext, entity_ids: list[str]) -> list[TemporalEntity]: ...
