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

import csv
import math
import re
from pathlib import Path
from typing import Any

from scienceflow.solver.lnr.stage.score_summary import score_record_from_stage_performance_row


EXECUTION_SCALES = {"pilot", "direct_full", "full_after_pilot"}
COMPARABLE_METRIC_TYPES = {"holdout", "cv"}
VALIDATION_PROTOCOL_ALIASES = {
    "cv": "cv",
    "cross_validation": "cv",
    "group_cv": "cv",
    "group_kfold": "cv",
    "kfold": "cv",
    "oof": "cv",
    "oof_cv": "cv",
    "repeated_cv": "cv",
    "holdout": "holdout",
    "heldout": "holdout",
    "single_holdout": "holdout",
    "validation_holdout": "holdout",
}


def normalize_execution_scale(value: Any) -> str:
    scale = str(value or "").strip().lower().replace("-", "_")
    return scale if scale in EXECUTION_SCALES else "unknown"


def normalize_route_key(value: Any) -> str:
    route = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip()).strip("_")
    return route[:160]


def normalize_validation_protocol(value: Any) -> str:
    protocol = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return VALIDATION_PROTOCOL_ALIASES.get(protocol, "")


def validation_protocols_comparable(current: Any, observed_best: Any) -> bool:
    current_protocol = normalize_validation_protocol(current)
    best_protocol = normalize_validation_protocol(observed_best)
    return bool(current_protocol and current_protocol == best_protocol)


def route_metric_evidence(path: str | Path, route_key: str) -> dict[str, Any]:
    route = normalize_route_key(route_key)
    result = {
        "route_key": route,
        "comparable_metric_count": 0,
        "route_metric_proven": False,
        "latest_stage_id": "",
        "latest_metric_value": None,
        "source": "stage_performance_csv",
    }
    p = Path(path)
    if not route or not p.is_file():
        return result
    try:
        with p.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return result
    for row in rows:
        row_route = normalize_route_key(row.get("route_id"))
        if not row_route and row.get("solution_sha"):
            row_route = f"source:{str(row.get('solution_sha'))[:16]}"
        if row_route != route:
            continue
        record = score_record_from_stage_performance_row(row)
        if record is None:
            continue
        comparable = bool(
            record.valid_comparable
            or (
                record.selection_eligible
                and record.val_score_type in COMPARABLE_METRIC_TYPES
                and record.metric_validity != "low"
                and record.validation_ok is not False
            )
        )
        if not comparable:
            continue
        result["comparable_metric_count"] += 1
        result["route_metric_proven"] = True
        result["latest_stage_id"] = record.stage_id
        result["latest_metric_value"] = record.value
    return result


def is_comparable_live_metric(metric: dict[str, Any] | None, *, saw_final_score: bool = False) -> bool:
    if saw_final_score:
        return True
    item = metric if isinstance(metric, dict) else {}
    name = str(item.get("metric_name") or "").strip().lower()
    phase = str(item.get("phase") or "").strip().lower()
    if not name:
        return False
    if name in {"loss", "train_loss", "training_loss"} or "train_loss" in name:
        return False
    metric_tokens = ("valid", "val_", "cv", "oof", "heldout", "held_out")
    phase_tokens = ("valid", "eval", "score", "oof", "heldout", "held_out")
    return any(token in name for token in metric_tokens) or any(token in phase for token in phase_tokens)


def build_research_cadence_facts(
    *,
    enabled: bool,
    eligible: bool,
    elapsed_sec: float,
    observe_sec: float,
    first_metric_budget_sec: float,
    proven_route_metric_budget_sec: float,
    execution_scale: str,
    route_key: str,
    route_evidence: dict[str, Any] | None,
    current_comparable_metric: bool,
    eta_to_next_comparable_metric_sec: float | None,
    eta_confidence: str,
) -> dict[str, Any]:
    elapsed = max(0.0, _float(elapsed_sec, 0.0))
    observe = max(1.0, _float(observe_sec, 300.0))
    first_budget = max(observe, _float(first_metric_budget_sec, 900.0))
    proven_budget = max(first_budget, _float(proven_route_metric_budget_sec, 1800.0))
    scale = normalize_execution_scale(execution_scale)
    evidence = dict(route_evidence or {})
    route_proven = bool(evidence.get("route_metric_proven"))
    budget_kind = "first_comparable_metric"
    budget = first_budget
    if scale != "pilot" and route_proven:
        budget_kind = "proven_route_metric"
        budget = proven_budget
    eta = _finite_or_none(eta_to_next_comparable_metric_sec)
    confidence = str(eta_confidence or "low").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    projected_metric_elapsed = elapsed + eta if eta is not None else None
    violation = False
    reason_code = ""
    if enabled and eligible and not current_comparable_metric and elapsed >= observe:
        if eta is not None and confidence in {"medium", "high"} and projected_metric_elapsed > budget:
            violation = True
            reason_code = "next_comparable_metric_eta_exceeds_budget"
        elif elapsed >= budget:
            violation = True
            reason_code = (
                "proven_route_metric_budget_exceeded"
                if budget_kind == "proven_route_metric"
                else "first_comparable_metric_budget_exceeded"
            )
    if not enabled:
        state = "disabled"
    elif not eligible:
        state = "not_applicable"
    elif current_comparable_metric:
        state = "metric_observed"
    elif violation:
        state = "cadence_exceeded"
    elif elapsed < observe:
        state = "observing"
    else:
        state = "within_budget"
    return {
        "enabled": bool(enabled),
        "eligible": bool(eligible),
        "state": state,
        "violation": violation,
        "reason_code": reason_code,
        "execution_scale": scale,
        "route_key": normalize_route_key(route_key),
        "route_metric_proven": route_proven,
        "route_metric_evidence": evidence,
        "current_comparable_metric": bool(current_comparable_metric),
        "elapsed_sec": elapsed,
        "observe_sec": observe,
        "metric_budget_kind": budget_kind,
        "metric_budget_sec": budget,
        "eta_to_next_comparable_metric_sec": eta,
        "projected_metric_elapsed_sec": projected_metric_elapsed,
        "eta_confidence": confidence,
    }


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, out) if math.isfinite(out) else None
