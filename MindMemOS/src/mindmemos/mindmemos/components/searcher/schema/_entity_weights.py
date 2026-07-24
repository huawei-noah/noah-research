"""Entity type weight loading and application for search ranking."""

from __future__ import annotations

from typing import Any

from ....logging import get_logger
from ...memory_modeling.schema import EntityManager

logger = get_logger(__name__)


def schema_search_load_entity_weights(entity_manager: EntityManager) -> dict[str, float]:
    """Load entity-type search weights from the entity manager."""
    try:
        weights: dict[str, float] = {}
        for entity_type_obj in entity_manager.get_all():
            weights[entity_type_obj.entity_type] = entity_type_obj.search_weight
        logger.info("entity_weights_loaded", weights=weights)
        return weights
    except Exception:
        logger.warning("entity_weights_load_failed", exc_info=True)
        return {}


def schema_search_apply_entity_weights(
    entities_with_scores: list[tuple[Any, float]],
    weights: dict[str, float],
) -> list[tuple[Any, float]]:
    """Apply entity weights to search scores."""
    weighted_results = []
    for entity, score in entities_with_scores:
        entity_type = entity.entity_type if hasattr(entity, "entity_type") else entity.get("entity_type", "")
        weight = weights.get(entity_type, 1.0)
        weighted_results.append((entity, score * weight))

    weighted_results.sort(key=lambda x: x[1], reverse=True)
    return weighted_results


def schema_search_apply_weights_to_ranked(
    ranked_entities: list[Any],
    weights: dict[str, float],
) -> list[Any]:
    """Convert ranks to scores, apply weights, and return reranked entities."""
    if not weights or not ranked_entities:
        return ranked_entities

    entities_with_scores = [(entity, 1.0 / (i + 1)) for i, entity in enumerate(ranked_entities)]
    weighted_results = schema_search_apply_entity_weights(entities_with_scores, weights)
    return [entity for entity, _ in weighted_results]
