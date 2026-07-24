"""Schema search entity weighting configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EntityWeightsConfig:
    """Episode / non-episode weight balancing."""

    force_balanced_split: bool = field(default=True)
    """Separate episode and non-episode retrieval paths."""

    episode_weight: float = field(default=0.7)
    """Episode entities weight ratio."""

    non_episode_weight: float = field(default=0.3)
    """Non-episode entities weight ratio."""
