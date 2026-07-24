"""Schema add component protocols."""

from __future__ import annotations

from typing import Any, Protocol

from ....typing import EntityWrite, GraphRelationship, MemoryDbWritePlan, MemoryWrite


class SchemaEpisodeExtractor(Protocol):
    """Protocol for prompt-driven schema episode extraction."""

    async def extract_episode(self, *, conversation_text: str, dialogue_timestamp: str) -> dict[str, Any]: ...


class SchemaExtractionNormalizerProtocol(Protocol):
    """Protocol for schema extraction result normalization."""

    def normalize(self, raw_memory: dict[str, Any], dialogue_timestamp: str) -> dict[str, Any]: ...

    def validate(self, raw_memory: dict[str, Any]) -> str | None: ...


class SchemaMergePolicyProtocol(Protocol):
    """Protocol for schema merge policy components."""

    async def prepare(
        self,
        *,
        raw_entities: list[dict[str, Any]],
        raw_edges: list[dict[str, Any]],
        episode_entity: dict[str, Any],
    ) -> Any: ...


class SchemaWritePlanBuilderProtocol(Protocol):
    """Protocol for turning schema merge decisions into a memory DB write plan."""

    async def build(
        self,
        *,
        memories: list[MemoryWrite],
        entities: list[EntityWrite],
        relationships: list[GraphRelationship],
        project_id: str,
        entity_context_memories: list[MemoryWrite] | None = None,
    ) -> MemoryDbWritePlan: ...


class SchemaSearchFieldExtractorProtocol(Protocol):
    """Protocol for generating schema query-oriented entity search fields."""

    async def extract_search_fields(
        self,
        *,
        entities: list[dict[str, Any]],
        context_text: str,
        max_fields: int,
        augment: bool = False,
        augment_count: int = 0,
        fallback_text: str | None = None,
    ) -> list[str]: ...
