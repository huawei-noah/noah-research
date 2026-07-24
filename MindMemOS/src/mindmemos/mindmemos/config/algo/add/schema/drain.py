"""Schema add drain configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DrainConfig:
    """Configuration for draining schema add buffered records."""

    episode_generation_max_retries: int = field(default=3)
    """Max retry attempts for a single episode generation task."""

    cleanup_processed_buffer: bool = field(default=True)
    """Whether to delete buffer records after successful episode generation."""
