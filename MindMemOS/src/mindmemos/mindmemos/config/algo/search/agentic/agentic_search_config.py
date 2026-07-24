"""Agentic search configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgenticConfig:
    """Multi-round agentic retrieval parameters."""

    max_rounds: int = field(default=3)
    """Maximum search iterations."""

    top_k_per_round: int = field(default=10)
    """Entities recalled per round."""

    top_n_per_round: int = field(default=5)
    """Entities kept per round after filtering."""

    num_hops: int = field(default=2)
    """Multi-hop expansion hops."""

    use_rerank: bool = field(default=True)
    """Whether reranker is used in the agentic loop."""

    use_relevance_filter: bool = field(default=False)
    """Whether LLM relevance filtering is applied to final results."""

    use_property_filter: bool = field(default=False)
    """Whether LLM selects entity types and properties before schema search."""

    current_time_mode: str = field(default="unknown")
    """Time mode hint for prompt generation."""

    min_time_window_days: int | None = field(default=30)
    """Minimum time window expansion in days."""

    include_edges: bool = field(default=False)
    """Whether edges are included in formatted entity prompts."""

    output_max_edge_num: int = field(default=10)
    """Maximum edges shown per entity in search output and sufficiency prompts."""
