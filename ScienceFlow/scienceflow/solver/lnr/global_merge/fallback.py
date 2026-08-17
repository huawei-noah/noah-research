# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

from typing import Any


LOWER_METRIC_HINTS = (
    "lower is better",
    "lower better",
    "smaller is better",
    "smaller better",
    "minimize",
    "minimise",
    "rmse",
    "rmsle",
    "mae",
    "mse",
    "logloss",
    "log loss",
    "cross entropy",
    "loss",
    "error",
    "levenshtein",
    "edit distance",
    "negative log likelihood",
    "nll",
)
HIGHER_METRIC_HINTS = (
    "higher is better",
    "higher better",
    "larger is better",
    "larger better",
    "maximize",
    "maximise",
    "kendall",
    "kendall tau",
    "modified laplace log likelihood",
    "laplace log likelihood",
    "auc",
    "auroc",
    "average precision",
    "mean average precision",
    "map@",
    " map ",
    "accuracy",
    "balanced accuracy",
    "f1",
    "dice",
    "iou",
    "jaccard",
    "mcc",
    "kappa",
    "precision",
    "recall",
    "spearman",
    "pearson",
)
ACCEPTED_METRIC_VALIDITY = {"high", "medium"}
REJECTED_METRIC_VALIDITY = {"low", "invalid", "false", "0", "no"}
REJECTED_REASON_CODES = {
    "invalid_submission",
    "missing_submission",
    "missing_artifact",
    "schema_invalid",
    "metric_parse_failed",
}
RISKY_REASON_CODES = {"same_validation_meta_fit"}


def metric_float(candidate: dict[str, Any]) -> float | None:
    try:
        value = float(candidate.get("metric_value"))
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return value


def hard_gate_reason(candidate: dict[str, Any]) -> str:
    if candidate.get("candidate_ready") is False:
        return "candidate_not_ready"
    if not str(candidate.get("submission_sha") or candidate.get("artifact_sha") or "").strip():
        return "missing_artifact_sha"
    if candidate.get("validation_ok") is False:
        return "sdk_validation_failed"
    selection_eligible = _coerce_bool(candidate.get("selection_eligible"))
    if selection_eligible is False:
        return "selection_not_eligible"
    validity = str(candidate.get("metric_validity") or "").strip().lower()
    if validity in REJECTED_METRIC_VALIDITY:
        return "low_metric_validity"
    if validity and validity not in ACCEPTED_METRIC_VALIDITY:
        return "unknown_metric_validity"
    reason_code = str(candidate.get("metric_validity_reason_code") or "").strip().lower()
    if reason_code in REJECTED_REASON_CODES:
        return reason_code
    if metric_float(candidate) is None:
        return "invalid_metric"
    return ""


def eligible_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in candidates if not hard_gate_reason(c)]


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _metric_direction_hint(text: str) -> bool | None:
    normalized = f" {text.lower().replace('_', ' ').replace('-', ' ')} "
    has_lower = any(hint in normalized for hint in LOWER_METRIC_HINTS)
    has_higher = any(hint in normalized for hint in HIGHER_METRIC_HINTS)
    if has_lower and not has_higher:
        return True
    if has_higher and not has_lower:
        return False
    return None


def infer_lower_is_better(candidates: list[dict[str, Any]]) -> bool:
    hint_votes: list[bool] = []
    for candidate in candidates:
        hint = _metric_direction_hint(str(candidate.get("metric_name") or ""))
        if hint is not None:
            hint_votes.append(hint)
    if hint_votes and hint_votes.count(True) != hint_votes.count(False):
        return hint_votes.count(True) > hint_votes.count(False)

    explicit_votes = [
        vote
        for vote in (_coerce_bool(candidate.get("lower_is_better")) for candidate in candidates)
        if vote is not None
    ]
    if explicit_votes and explicit_votes.count(True) != explicit_votes.count(False):
        return explicit_votes.count(True) > explicit_votes.count(False)
    if explicit_votes:
        return explicit_votes[0]
    return True


def _risk_rank(candidate: dict[str, Any]) -> int:
    if candidate.get("selection_eligible") is False:
        return 3
    validity = str(candidate.get("metric_validity") or "").strip().lower()
    if validity == "medium" or not validity:
        validity_rank = 1
    elif validity == "high":
        validity_rank = 0
    else:
        validity_rank = 2
    reason_code = str(candidate.get("metric_validity_reason_code") or "").strip().lower()
    if reason_code in RISKY_REASON_CODES:
        validity_rank = max(2, validity_rank)
    val_type = str(candidate.get("val_score_type") or "").strip().lower()
    if val_type == "training_loss":
        return max(2, validity_rank)
    if val_type == "post_fulltrain_replay":
        return max(1, validity_rank)
    if str(candidate.get("duplicate_submission_of_stage") or "").strip():
        return max(1, validity_rank)
    return validity_rank


def ranked_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = eligible_candidates(candidates)
    if not valid:
        return []
    lower = infer_lower_is_better(valid)

    def sort_key(candidate: dict[str, Any]) -> tuple[int, float, str]:
        metric = float(metric_float(candidate) or 0.0)
        score = metric if lower else -metric
        return (
            _risk_rank(candidate),
            score,
            str(candidate.get("candidate_id") or ""),
        )

    return sorted(valid, key=sort_key)
