"""Root configuration for schema-aware search."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema_search_dual_path import DualPathConfig
from .schema_search_edge import EdgeSearchConfig
from .schema_search_entity import EntitySearchConfig
from .schema_search_entity_weights import EntityWeightsConfig
from .schema_search_property import PropertySearchConfig


@dataclass
class SchemaSearchConfig:
    """Configuration for schema-aware entity/property search components."""

    entity: EntitySearchConfig = field(default_factory=EntitySearchConfig)
    property: PropertySearchConfig = field(default_factory=PropertySearchConfig)
    dual_path: DualPathConfig = field(default_factory=DualPathConfig)
    entity_weights: EntityWeightsConfig = field(default_factory=EntityWeightsConfig)
    edge: EdgeSearchConfig = field(default_factory=EdgeSearchConfig)

    multi_hop: int = field(default=2)
    """Number of graph expansion hops."""

    use_entity_agent_search: bool = field(default=True)
    """Whether schema entity-level disclosure is enabled."""

    current_time_mode: str = field(default="unknown")
    """Time mode hint for schema query time extraction."""

    min_time_window_days: int | None = field(default=30)
    """Minimum time window expansion in days."""

    include_edges: bool = field(default=False)
    """Whether formatted schema search results include entity edges."""

    output_max_edge_num: int = field(default=10)
    """Maximum edges shown per entity in schema search output."""
