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

import re
from collections.abc import Mapping
from typing import Any


METRIC_VALIDITY_LEVELS = {"high", "medium", "low"}

_LOW_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "metric_text_reports_leakage",
        re.compile(r"\b(target[- ]?leak|target leakage|target-leaking|leaky|leakage risk|data leakage)\b", re.I),
    ),
    (
        "metric_text_reports_overfit",
        re.compile(r"\b(overfit|overfits|overfitting|unreliable|not generalizable|inflated|inflates|massively overfits|severely overfits)\b", re.I),
    ),
    (
        "metric_text_reports_target_derived_feature",
        re.compile(
            r"\b(benign_malignant|diagnosis feature|pat_malig|pat_mal|mal_ratio|pos_rate|positive rate|"
            r"target-derived|patient positive)\b",
            re.I,
        ),
    ),
    (
        "metric_text_reports_same_validation_overfit",
        re.compile(
            r"\b(trained and evaluated on the same|evaluated on the same|same .*validation set|"
            r"meta-learner overfit|honest estimate is|honest score would|real honest score|"
            r"not held[- ]?out valid|not held[- ]?out|"
            r"(?:meta[- ]?stacking|meta model|stacking|calibration|isotonic).{0,80}(?:same validation|not held[- ]?out)|"
            r"(?:same validation|not held[- ]?out).{0,80}(?:meta[- ]?stacking|meta model|stacking|calibration|isotonic))\b",
            re.I,
        ),
    ),
    (
        "metric_text_reports_fulltrain_validation_reuse",
        re.compile(
            r"\b("
            r"trained on train\+val.*(?:re-evaluated|reevaluated|evaluated).*val|"
            r"train\+val.*(?:re-evaluated|reevaluated|evaluated).*same val|"
            r"final model.*(?:re-evaluated|reevaluated|evaluated).*same val|"
            r"model trained on train\+val.*(?:loading validation features|validation features for score computation)|"
            r"train\+val.*validation features for score computation|"
            r"trivial auc\s*=\s*1(?:\.0+)?|"
            r"used validation rows for training"
            r")\b",
            re.I,
        ),
    ),
    (
        "metric_text_reports_schema_fix_needed",
        re.compile(r"\b(schema_fix_needed|schema fix needed|submission schema invalid|invalid submission)\b", re.I),
    ),
)

_MEDIUM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "metric_text_reports_diagnostic_or_proxy",
        re.compile(
            r"\b(diagnostic checkpoint|diagnostic only|not used for selection|not comparable|proxy score|"
            r"train[- ]only proxy|cheap probe|replay metric kept|kept but not used for selection)\b",
            re.I,
        ),
    ),
    (
        "metric_text_reports_post_fulltrain_reference",
        re.compile(r"\b(post[_ -]?fulltrain[_ -]?replay|fulltrain replay|replay metric)\b", re.I),
    ),
)

_OPT_SOLVER_TASK_PROFILES = {"opt_solver", "optimization_solver", "artifact_solver"}
_LOW_REASON_CODES = {"same_validation_meta_fit", "train_val_reuse", "leakage_risk"}


def normalize_metric_validity(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in METRIC_VALIDITY_LEVELS else ""


def infer_metric_validity(fields: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(metric_validity, reason)`` using one generic high/medium/low contract.

    ``high`` means the metric has the strongest comparability evidence.
    ``medium`` means useful route evidence or an evaluator-verified score with weaker provenance notes.
    ``low`` means untrustworthy or non-comparable unless an explicit evaluator event verified the artifact.
    """
    validation_ok = _boolish(fields.get("validation_ok"), default=None)
    submission_validation_ok = _boolish(fields.get("submission_validation_ok"), default=None)
    submission_status = str(fields.get("submission_status") or "").strip()
    val_score_type = str(fields.get("val_score_type") or fields.get("metric_type") or "").strip().lower()
    task_profile = _normalize_task_profile(fields.get("task_profile"))
    evaluator_status = str(fields.get("evaluator_status") or "").strip().lower()
    evaluator_backend = str(fields.get("evaluator_backend") or "").strip().lower()
    selection_eligible = _boolish(fields.get("selection_eligible"), default=False) is True
    candidate_ready = _boolish(fields.get("candidate_ready"), default=True) is True
    explicit = normalize_metric_validity(fields.get("metric_validity"))
    text = _combined_text(fields)
    reason_code = str(fields.get("metric_validity_reason_code") or "").strip().lower()

    if validation_ok is False:
        return "low", str(fields.get("validation_issue") or "validation_ok=false")
    if submission_validation_ok is False or submission_status == "invalid_submission":
        return "low", "invalid_submission"
    if reason_code in _LOW_REASON_CODES:
        return "low", reason_code
    lowered_text = text.lower()
    if "train+val" in lowered_text and "validation features" in lowered_text:
        return "low", "metric_text_reports_fulltrain_validation_reuse"
    for reason, pattern in _LOW_PATTERNS:
        if pattern.search(text):
            return "low", reason

    if _boolish(fields.get("metric_authoritative"), default=False) and evaluator_backend:
        if evaluator_status == "ok":
            if not _metric_value_present(fields.get("metric_value")):
                return "medium", "authoritative_evaluator_metric_missing"
            if not _direction_known(fields.get("lower_is_better")):
                return "medium", "authoritative_evaluator_metric_direction_missing"
            if candidate_ready and validation_ok is not False and selection_eligible:
                return "high", "authoritative_evaluator_ok"
        elif evaluator_status:
            return "low", "authoritative_evaluator_not_ok"

    if task_profile in _OPT_SOLVER_TASK_PROFILES and evaluator_backend:
        if evaluator_status == "ok":
            if not _metric_value_present(fields.get("metric_value")):
                return "medium", "opt_solver_metric_missing"
            if not _direction_known(fields.get("lower_is_better")):
                return "medium", "opt_solver_metric_direction_missing"
            if candidate_ready and validation_ok is not False:
                return "high", "opt_solver_evaluator_ok"
        elif evaluator_status:
            return "low", "opt_solver_evaluator_not_ok"

    if explicit:
        return explicit, f"explicit_metric_validity_{explicit}"

    for reason, pattern in _MEDIUM_PATTERNS:
        if pattern.search(text):
            return "medium", reason
    if val_score_type in {"post_fulltrain_replay", "training_loss", "unknown"}:
        return "medium", f"{val_score_type}_metric_not_final_best"
    if selection_eligible and candidate_ready and val_score_type in {"holdout", "cv"}:
        return "high", "selection_eligible_comparable_metric"
    if selection_eligible and candidate_ready:
        return "high", "selection_eligible_stage_metric"
    return "medium", "observed_metric_not_selection_best"


def _combined_text(fields: Mapping[str, Any]) -> str:
    keys = (
        "metric_validity_note",
        "metric_validity_reason_code",
        "metric_validity_source",
        "validation_issue",
        "selection_note",
        "metric_source_note",
        "brief",
        "why",
        "route_evidence",
        "metric_protocol",
        "val_score_type",
        "metric_type",
        "metric_note",
        "task_profile",
        "evaluator_backend",
        "evaluator_status",
        "metric_authoritative",
        "stdout_tail",
        "stderr_tail",
    )
    return " ".join(str(fields.get(key) or "") for key in keys)


def _normalize_task_profile(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _metric_value_present(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return bool(text and text not in {"none", "null", "nan"})


def _direction_known(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if value is None:
        return False
    text = str(value).strip().lower()
    return bool(text and text not in {"none", "null", "nan", "unknown"})


def _boolish(value: Any, *, default: bool | None = None) -> bool | None:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "ok"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default
