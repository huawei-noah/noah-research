"""Schema search property disclosure configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PropertySearchConfig:
    """Per-entity property shrinking parameters."""

    recall_size: int = field(default=45)
    """Initial property recall per entity."""

    rrf_k: int = field(default=60)
    """RRF smoothing parameter for property fusion."""

    top_k: int = field(default=20)
    """RRF fusion result count."""

    top_n: int = field(default=16)
    """Final properties kept per entity."""

    alloc_min_factor: float = field(default=0.5)
    """Dynamic allocation lower bound factor."""

    alloc_max_factor: float = field(default=1.5)
    """Dynamic allocation upper bound factor."""

    use_property_extension: bool = field(default=True)
    """Whether temporal context expansion is applied after property shrink."""

    extension_step: int = field(default=3)
    """Forward/backward expansion steps for temporal extension."""

    higher_order_ratio: float = field(default=0.4)
    """Budget split ratio for higher-order properties."""
