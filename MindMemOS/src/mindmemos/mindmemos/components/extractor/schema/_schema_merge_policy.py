"""Schema add merge policy facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ....typing import EntityWrite
from .base import SchemaMergePolicyProtocol


@dataclass(slots=True)
class SchemaMergeContext:
    """Intermediate schema add merge state consumed by the write-plan builder."""

    raw_entities: list[dict[str, Any]]
    raw_edges: list[dict[str, Any]]
    episode_entity: dict[str, Any]
    entity_by_name: dict[str, EntityWrite] = field(default_factory=dict)
    pending_archives: list[str] = field(default_factory=list)


class SchemaMergePolicy(SchemaMergePolicyProtocol):
    """Lightweight merge-policy boundary for schema add.

    The current implementation preserves the migrated algorithm's behavior by
    delegating entity/property decisions to the write-plan builder. The class is
    intentionally explicit so future merge policy changes do not grow inside the
    pipeline or DTOs.
    """

    async def prepare(
        self,
        *,
        raw_entities: list[dict[str, Any]],
        raw_edges: list[dict[str, Any]],
        episode_entity: dict[str, Any],
    ) -> SchemaMergeContext:
        """Create the merge context passed to the write-plan builder."""

        return SchemaMergeContext(
            raw_entities=list(raw_entities),
            raw_edges=list(raw_edges),
            episode_entity=dict(episode_entity),
        )
