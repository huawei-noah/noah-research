"""Add-operation algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import (
    DrainConfig,
    EpisodesChunkerConfig,
    SchemaAddConfig,
    SchemaAddEpisodeEdgeConfig,
    SchemaAddExtractionConfig,
    SchemaAddHigherOrderConfig,
    SchemaAddMergeConfig,
)
from .vanilla import VanillaAddConfig


@dataclass
class AddAlgoConfig:
    """Configuration for add-operation algorithms."""

    schema: SchemaAddConfig = field(default_factory=SchemaAddConfig)
    vanilla: VanillaAddConfig = field(default_factory=VanillaAddConfig)


__all__ = [
    "AddAlgoConfig",
    "DrainConfig",
    "EpisodesChunkerConfig",
    "SchemaAddConfig",
    "SchemaAddEpisodeEdgeConfig",
    "SchemaAddExtractionConfig",
    "SchemaAddHigherOrderConfig",
    "SchemaAddMergeConfig",
    "VanillaAddConfig",
]
