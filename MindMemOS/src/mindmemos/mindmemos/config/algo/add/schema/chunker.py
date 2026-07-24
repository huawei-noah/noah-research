"""Episode chunker configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodesChunkerConfig:
    """Configuration for add-message episode boundary detection."""

    split_mode: str = field(default="llm")
    """Boundary detection mode. ``llm`` uses the prompt, ``rule`` uses deterministic cuts."""

    min_episode_length: int = 1
    """Minimum buffered message count before boundary detection runs."""

    max_episode_length: int = 15
    """Maximum message count in one rule-split episode."""

    max_buffer_size: int = 1000
    """Maximum number of raw messages kept in one project buffer."""

    split_on_user_speaker: bool = field(default=True)
    """Whether rule mode starts a new episode when the current speaker is user."""

    max_minutes_from_first: int = field(default=30)
    """Rule mode cuts when a message is farther than this from episode start."""
