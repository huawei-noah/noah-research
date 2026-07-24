"""Generic reciprocal-rank fusion helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...logging import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    id_key: str = "entity_id",
    score_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists with the RRF formula."""
    if not result_lists:
        return []

    if score_keys is None:
        score_keys = ["score"]

    rrf_scores: dict[str, float] = defaultdict(float)
    result_info: dict[str, dict[str, Any]] = {}
    original_scores: dict[str, dict[str, Any]] = defaultdict(dict)

    for list_idx, result_list in enumerate(result_lists):
        for rank, result in enumerate(result_list):
            result_id = result.get(id_key)
            if result_id is None:
                continue

            rrf_scores[str(result_id)] += 1.0 / (k + rank + 1)
            result_info.setdefault(str(result_id), result.copy())

            for score_key in score_keys:
                if score_key in result:
                    original_scores[str(result_id)][f"{score_key}_list_{list_idx}"] = result[score_key]

    fused_results = []
    for result_id, rrf_score in rrf_scores.items():
        if result_id in result_info:
            result = result_info[result_id].copy()
            result["rrf_score"] = rrf_score
            result.update(original_scores[result_id])
            fused_results.append(result)

    fused_results.sort(key=lambda item: item["rrf_score"], reverse=True)

    if fused_results:
        logger.debug(
            "rrf_fusion_complete",
            list_count=len(result_lists),
            fused_count=len(fused_results),
        )

    return fused_results
