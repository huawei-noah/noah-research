"""Schema search edge expansion configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EdgeSearchConfig:
    """Edge filtering parameters used by schema entity disclosure."""

    top_k: int = field(default=2)
    """Maximum edges kept per entity after filtering."""

    neighbor_fetch_limit: int = field(default=20)
    """Maximum raw graph neighbors fetched per entity before rerank filtering."""

    min_relevance_score: float = field(default=0.1)
    """Reranker threshold for edge relevance."""
