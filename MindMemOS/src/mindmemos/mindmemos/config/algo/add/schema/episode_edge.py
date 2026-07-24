"""Schema add episode edge configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchemaAddEpisodeEdgeConfig:
    """Episode-to-episode edge generation configuration."""

    top_k: int = field(default=10)
    """Number of existing episodes recalled for episode edge creation."""
