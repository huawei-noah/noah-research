"""Schema add merge policy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchemaAddMergeConfig:
    """Entity and property merge configuration for schema add."""

    enable_entity_merge_decision: bool = field(default=True)
    """Whether schema add asks the LLM to decide create/update for recalled candidates."""

    entity_recall_top_k: int = field(default=15)
    """Number of entity candidates recalled before the entity merge decision."""

    max_merge_retries: int = field(default=8)
    """Maximum LLM retries for entity merge decisions."""

    use_property_merge: bool = field(default=False)
    """Whether schema add runs the property merge/delete decision prompt."""

    secondary_search_limit: int = field(default=30)
    """Entity-name fallback search limit when an LLM update target is not in primary recall."""

    secondary_search_retries: int = field(default=3)
    """Retry count for entity-name fallback search."""
