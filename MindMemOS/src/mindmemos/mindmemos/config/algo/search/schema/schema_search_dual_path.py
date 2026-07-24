"""Schema search dual-path configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DualPathConfig:
    """Parallel entity + property path concurrent search."""

    enabled: bool = field(default=True)
    """Enable dual concurrent retrieval paths."""

    property_recall_size: int = field(default=300)
    """Property path initial recall size."""

    property_rrf_k: int = field(default=80)
    """RRF parameter for property path."""

    property_top_k: int = field(default=80)
    """Post-RRF count for property path."""

    property_top_n: int = field(default=25)
    """Final count after reranking for property path."""
