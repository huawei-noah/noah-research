"""Schema add extraction configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchemaAddExtractionConfig:
    """LLM extraction and episode search-field configuration."""

    enable_schema_selection: bool = field(default=True)
    """Whether schema add asks the LLM to select relevant entity schemas before extraction."""

    use_search_fields: bool = field(default=True)
    """Whether schema add stores episode search fields derived from extracted properties."""

    search_fields_max: int = field(default=10)
    """Maximum number of episode search fields stored on the episode entity metadata."""

    episode_search_fields_augment: bool = field(default=True)
    """Whether schema add asks the LLM to augment episode search fields."""

    episode_augment_count: int = field(default=4)
    """Maximum number of LLM-generated supplemental episode search fields."""
