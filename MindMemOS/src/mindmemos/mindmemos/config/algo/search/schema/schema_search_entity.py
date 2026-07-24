"""Schema search entity recall configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EntitySearchConfig:
    """Entity-level hybrid retrieval parameters."""

    recall_size: int = field(default=800)
    """Initial recall count per method (vector / BM25)."""

    rrf_k: int = field(default=80)
    """RRF smoothing parameter."""

    top_k: int = field(default=40)
    """RRF fusion result count."""

    top_n: int = field(default=16)
    """Final entities after reranking."""

    use_reranker: bool = field(default=True)
    """Whether to apply reranker after RRF fusion."""

    max_rerank_candidates: int = field(default=100)
    """Maximum candidates sent to the reranker."""

    use_maxsim_rescore: bool = field(default=False)
    """Whether to apply MaxSim rescoring after RRF fusion."""

    maxsim_weight: float = field(default=0.3)
    """Weight for MaxSim score in combined scoring."""

    search_field_overfetch_factor: int = field(default=3)
    """Raw Qdrant point over-fetch factor before canonical search-field dedup."""
