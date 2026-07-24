"""Vanilla search engine configuration."""

from dataclasses import dataclass, field


@dataclass
class VanillaSearchConfig:
    """Vanilla hybrid search parameters.

    Purpose:
        Configure the vanilla search engine recall phase for flat memory retrieval
        using dense + sparse hybrid search.

    Used in:
        - ``MemoryConfig.algo_config.search.vanilla`` for YAML-driven configuration
        - ``VanillaSearchEngine`` constructor for dependency injection
    """

    recall_size: int = field(default=20)
    """Over-retrieval count for RRF fusion before final filtering."""

    hybrid_prefetch_factor: int = field(default=3)
    """Multiplier applied to recall_size for dense/sparse prefetch in hybrid RRF search."""

    hybrid_prefetch_min: int = field(default=30)
    """Floor for dense/sparse prefetch counts in hybrid RRF search."""

    use_reranker: bool = field(default=True)
    """Whether final reranking is allowed for vanilla search candidates.

    Rerank truncation limits are owned by ``algo.search.rerank`` (``max_query_length`` /
    ``max_doc_length``) and applied by the shared rerank client, not duplicated here.
    """

    graph_enabled: bool = field(default=False)
    """Whether to supplement vanilla search with Neo4j one-hop related memories."""

    shared_entity_graph_enabled: bool = field(default=False)
    """Whether to supplement vanilla search with memories that mention the same entities as base hits."""

    graph_seed_memory_limit: int = field(default=5)
    """Maximum base memory hits used as Neo4j graph expansion seeds."""

    graph_related_per_seed: int = field(default=3)
    """Maximum ``RELATES_TO`` neighbors kept per seed memory."""

    shared_entity_graph_limit_per_entity: int = field(default=3)
    """Maximum memories kept per shared entity scope."""

    graph_max_candidates: int = field(default=10)
    """Maximum related memory IDs hydrated from graph expansion."""

    graph_decay: float = field(default=0.5)
    """Score multiplier applied to the originating seed memory for graph candidates."""

    graph_score: float = field(default=0.01)
    """Fallback score assigned to graph candidates when seed score is unavailable."""
