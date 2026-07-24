"""Entity-to-entity edge model used by modeling components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....typing import (
    REL_RELATED_TO,
    EntityView,
    EntityWrite,
    GraphNodeRef,
    GraphRelationship,
)


@dataclass(slots=True)
class Edge:
    """Relationship between two modeled entity instances."""

    link_entity1_id: str = "unknown"
    link_entity1_name: str = "unknown"
    link_entity2_id: str = "unknown"
    link_entity2_name: str = "unknown"
    link_description: str = "unknown"

    def get_entity1_id(self) -> str:
        return self.link_entity1_id

    def get_entity2_id(self) -> str:
        return self.link_entity2_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize the edge as a dictionary."""
        return {
            "link_entity1_id": self.link_entity1_id,
            "link_entity1_name": self.link_entity1_name,
            "link_entity2_id": self.link_entity2_id,
            "link_entity2_name": self.link_entity2_name,
            "link_description": self.link_description,
        }

    def to_graph_relationship(self, *, project_id: str, rel_type: str = REL_RELATED_TO) -> GraphRelationship:
        """Convert the edge to a graph relationship DTO.

        Args:
            project_id: Project id for graph isolation.
            rel_type: Relationship type to write.

        Returns:
            Graph relationship DTO ready for persistence.
        """
        return GraphRelationship(
            source=GraphNodeRef(kind="Entity", project_id=project_id, node_id=self.link_entity1_id),
            target=GraphNodeRef(kind="Entity", project_id=project_id, node_id=self.link_entity2_id),
            rel_type=rel_type,
            project_id=project_id,
            relation_type=self.link_description,
            metadata={
                "link_entity1_name": self.link_entity1_name,
                "link_entity2_name": self.link_entity2_name,
                "link_description": self.link_description,
            },
        )

    @classmethod
    def from_entity_dtos(
        cls,
        source: EntityView | EntityWrite,
        target: EntityView | EntityWrite,
        *,
        description: str = "related_to",
    ) -> Edge:
        """Build an edge from two entity DTOs.

        Args:
            source: Source entity DTO.
            target: Target entity DTO.
            description: Relationship description.

        Returns:
            Edge instance connecting the two entities.
        """
        return cls(
            link_entity1_id=source.entity_id,
            link_entity1_name=source.entity_name,
            link_entity2_id=target.entity_id,
            link_entity2_name=target.entity_name,
            link_description=description,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        """Deserialize an edge from a dictionary."""
        return cls(
            link_entity1_id=str(data.get("link_entity1_id", "unknown")),
            link_entity1_name=str(data.get("link_entity1_name", "unknown")),
            link_entity2_id=str(data.get("link_entity2_id", "unknown")),
            link_entity2_name=str(data.get("link_entity2_name", "unknown")),
            link_description=str(data.get("link_description", "unknown")),
        )
