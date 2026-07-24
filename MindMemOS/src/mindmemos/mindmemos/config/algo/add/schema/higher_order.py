"""Schema add higher-order memory configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchemaAddHigherOrderConfig:
    """Higher-order property generation configuration."""

    enabled: bool = field(default=True)
    """Whether updated non-episode entities trigger higher-order property generation."""

    top_k: int = field(default=10)
    """Number of first-order property memories used as evidence."""

    min_evidence_count: int = field(default=2)
    """Minimum evidence count inserted into the higher-order generation prompt."""
