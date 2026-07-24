"""Search components: recall, rerank, and fusion operators."""

from .entity_recall import EntityRecall, build_entity_type_filter, combine_entity_results_rrf
from .final_filter import SearchFinalFilter
from .protocols import EntityHydrator, EntityRecallStrategy, SearchStrategy
from .rerank import rerank, rerank_with_scores
from .rrf import reciprocal_rank_fusion

__all__ = [
    "EntityHydrator",
    "EntityRecall",
    "EntityRecallStrategy",
    "SearchStrategy",
    "SearchFinalFilter",
    "build_entity_type_filter",
    "combine_entity_results_rrf",
    "reciprocal_rank_fusion",
    "rerank",
    "rerank_with_scores",
]
