"""Default search configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DefaultSearchConfig:
    """Configuration for the non-agentic default search pipeline."""

    top_k: int = field(default=10)
    """Default number of memories returned when the request does not override it."""
