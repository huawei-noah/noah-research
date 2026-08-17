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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from scienceflow.solver.lnr.global_merge.fallback import HIGHER_METRIC_HINTS, LOWER_METRIC_HINTS
from scienceflow.solver.lnr.stage.metric_validity import infer_metric_validity


@dataclass(frozen=True)
class ScoreRecord:
    value: float
    metric_name: str = "Final Validation Score"
    lower_is_better: bool = True
    source: str = "stage_performance_csv"
    validity: str = "observed_not_official"
    reason: str = ""
    metric_source_note: str = ""
    metric_validity: str = "medium"
    metric_validity_note: str = ""
    worker_id: str = ""
    stage_id: str = ""
    candidate_id: str = ""
    artifact_path: str = ""
    snapshot_path: str = ""
    selection_score: float | None = None
    selection_eligible: bool = False
    val_score_type: str = ""
    candidate_ready: bool = False
    submission_status: str = ""
    validation_ok: bool | None = None
    evaluator_backend: str = ""
    evaluator_status: str = ""
    task_profile: str = ""
    row_order: int = 0
    method_text: str = ""
    route_id: str = ""
    execution_scale: str = ""
    solution_sha: str = ""

    @property
    def evaluator_verified(self) -> bool:
        return (
            bool(self.evaluator_backend)
            and self.evaluator_status == "ok"
            and self.validation_ok is True
            and self.selection_eligible
            and self.candidate_ready
            and self.submission_status != "invalid_submission"
        )

    @property
    def valid_comparable(self) -> bool:
        return self.validity == "valid_comparable"

    def to_public_dict(self, *, include_paths: bool = False) -> dict[str, Any]:
        out = {
            "value": self.value,
            "metric_name": self.metric_name,
            "lower_is_better": self.lower_is_better,
            "source": self.source,
            "validity": self.validity,
            "reason": self.reason,
            "metric_source_note": self.metric_source_note,
            "metric_validity": self.metric_validity,
            "metric_validity_note": self.metric_validity_note,
            "worker_id": self.worker_id,
            "stage_id": self.stage_id,
            "candidate_id": self.candidate_id,
            "selection_score": self.selection_score,
            "selection_eligible": self.selection_eligible,
            "val_score_type": self.val_score_type,
            "candidate_ready": self.candidate_ready,
            "submission_status": self.submission_status,
            "validation_ok": self.validation_ok,
            "evaluator_backend": self.evaluator_backend,
            "evaluator_status": self.evaluator_status,
            "task_profile": self.task_profile,
            "row_order": self.row_order,
            "route_id": self.route_id,
            "execution_scale": self.execution_scale,
        }
        if include_paths:
            out["artifact_path"] = self.artifact_path
            out["snapshot_path"] = self.snapshot_path
        return out


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_zero(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _boolish(value: Any, *, default: bool | None = None) -> bool | None:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "ok"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def metric_lower_is_better_hint(text: str) -> bool | None:
    """Return a direction hint from metric/task text.

    The constants intentionally match select-best audit hints so score context,
    peer summaries, and final selection do not diverge on common metrics.
    """
    normalized = f" {str(text or '').lower().replace('_', ' ').replace('-', ' ')} "
    has_lower = any(hint in normalized for hint in LOWER_METRIC_HINTS)
    has_higher = any(hint in normalized for hint in HIGHER_METRIC_HINTS)
    if has_lower and not has_higher:
        return True
    if has_higher and not has_lower:
        return False
    return None


def _infer_lower_is_better(records: list["ScoreRecord"]) -> bool:
    explicit_votes = [rec.lower_is_better for rec in records]
    if explicit_votes and explicit_votes.count(True) != explicit_votes.count(False):
        return explicit_votes.count(True) > explicit_votes.count(False)
    if explicit_votes:
        return explicit_votes[0]
    return True


def normalize_score_record_directions(records: list["ScoreRecord"]) -> list["ScoreRecord"]:
    if not records:
        return []
    lower = _infer_lower_is_better(records)
    return [rec if rec.lower_is_better == lower else replace(rec, lower_is_better=lower) for rec in records]


def infer_stage_rows_lower_is_better(rows: list[dict[str, Any]]) -> bool:
    records = [rec for row in rows if (rec := score_record_from_stage_performance_row(row)) is not None]
    return _infer_lower_is_better(records)


def _is_better(candidate: ScoreRecord, current: ScoreRecord | None, *, use_selection_score: bool = False) -> bool:
    if current is None:
        return True
    cand_value = candidate.selection_score if use_selection_score and candidate.selection_score is not None else candidate.value
    cur_value = current.selection_score if use_selection_score and current.selection_score is not None else current.value
    return cand_value < cur_value if candidate.lower_is_better else cand_value > cur_value

def _validity_from_row(row: dict[str, Any]) -> tuple[str, str]:
    validation_ok = _boolish(row.get("validation_ok"), default=None)
    selection_eligible = _boolish(row.get("selection_eligible"), default=False) is True
    candidate_ready = _candidate_ready_from_row(row)
    submission_status = _submission_status_from_row(row)
    validation_issue = str(row.get("validation_issue") or "").strip()
    metric_validity, metric_validity_note = infer_metric_validity(row)
    evaluator_verified = _evaluator_verified_row(row)
    if validation_ok is False:
        return "validation_failed", validation_issue or "validation_ok=false"
    if submission_status == "invalid_submission":
        return "produced_invalid", "submission invalid or failed validation"
    if metric_validity == "low":
        return "risk_flagged_metric", metric_validity_note or "metric_validity=low"
    if selection_eligible and candidate_ready and evaluator_verified:
        return "valid_comparable", metric_validity_note or "evaluator verified metric"
    if selection_eligible and candidate_ready and metric_validity == "high":
        return "valid_comparable", metric_validity_note or "selection_eligible stage-performance row"
    if candidate_ready:
        return "observed_not_comparable", metric_validity_note or "candidate ready but not selection eligible"
    return "observed_not_official", metric_validity_note or submission_status or "stage row not ready"


def _evaluator_verified_row(row: dict[str, Any]) -> bool:
    evaluator_backend = _evaluator_backend_from_row(row)
    evaluator_status = _evaluator_status_from_row(row)
    validation_ok = _boolish(row.get("validation_ok"), default=None)
    selection_eligible = _boolish(row.get("selection_eligible"), default=False) is True
    candidate_ready = _candidate_ready_from_row(row)
    submission_status = _submission_status_from_row(row)
    return (
        bool(evaluator_backend)
        and evaluator_status == "ok"
        and validation_ok is True
        and selection_eligible
        and candidate_ready
        and _direction_known(row.get("lower_is_better"))
        and submission_status != "invalid_submission"
    )


def _legacy_shifted_evaluator_row(row: dict[str, Any]) -> bool:
    """Detect historical CSV rows written before artifact/evaluator columns settled."""
    if str(row.get("submission_changed") or "").strip().lower() != "ok":
        return False
    if _boolish(row.get("submission_status"), default=False) is not True:
        return False
    backend_hint = str(row.get("evaluator_status") or "").strip().lower()
    artifact_hint = str(row.get("artifact_sha") or "").strip()
    return bool(backend_hint and artifact_hint and "/" in artifact_hint)


def _evaluator_backend_from_row(row: dict[str, Any]) -> str:
    if _legacy_shifted_evaluator_row(row):
        return str(row.get("evaluator_status") or "").strip()
    return str(row.get("evaluator_backend") or "").strip()


def _evaluator_status_from_row(row: dict[str, Any]) -> str:
    if _legacy_shifted_evaluator_row(row):
        return str(row.get("submission_changed") or "").strip().lower()
    return str(row.get("evaluator_status") or "").strip().lower()


def _candidate_ready_from_row(row: dict[str, Any]) -> bool:
    ready = _boolish(row.get("candidate_ready"), default=None)
    if ready is not None and ready is True:
        return True
    if _legacy_shifted_evaluator_row(row):
        return _boolish(row.get("submission_status"), default=False) is True
    return ready is True


def _submission_status_from_row(row: dict[str, Any]) -> str:
    if _legacy_shifted_evaluator_row(row):
        return str(row.get("duplicate_submission_of_stage") or "").strip()
    return str(row.get("submission_status") or "").strip()


def _artifact_path_from_row(row: dict[str, Any]) -> str:
    artifact_path = str(row.get("artifact_path") or "").strip()
    if artifact_path:
        return artifact_path
    if _legacy_shifted_evaluator_row(row):
        return str(row.get("artifact_sha") or "").strip()
    return str(row.get("submission_snapshot") or "").strip()


def _artifact_sha_from_row(row: dict[str, Any]) -> str:
    if _legacy_shifted_evaluator_row(row):
        return str(row.get("artifact_kind") or "").strip()
    return str(row.get("artifact_sha") or "").strip()


def _direction_known(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if value is None:
        return False
    text = str(value).strip().lower()
    return bool(text and text not in {"none", "null", "nan", "unknown"})


def score_record_from_stage_performance_row(row: dict[str, Any]) -> ScoreRecord | None:
    metric = _float_or_none(row.get("metric_value"))
    if metric is None:
        return None
    lower = _boolish(row.get("lower_is_better"), default=True)
    selection_score = _float_or_none(row.get("selection_score"))
    validity, reason = _validity_from_row(row)
    metric_validity, metric_validity_note = infer_metric_validity(row)
    return ScoreRecord(
        value=metric,
        metric_name=str(row.get("metric_name") or "Final Validation Score"),
        lower_is_better=bool(lower is not False),
        source="stage_performance_csv",
        validity=validity,
        reason=reason,
        metric_source_note=str(row.get("metric_source_note") or ""),
        metric_validity=metric_validity,
        metric_validity_note=metric_validity_note,
        worker_id=str(row.get("worker_id") or ""),
        stage_id=str(row.get("stage_id") or ""),
        candidate_id=str(row.get("candidate_id") or ""),
        artifact_path=_artifact_path_from_row(row),
        snapshot_path=str(row.get("snapshot_path") or ""),
        selection_score=selection_score,
        selection_eligible=_boolish(row.get("selection_eligible"), default=False) is True,
        val_score_type=str(row.get("val_score_type") or ""),
        candidate_ready=_candidate_ready_from_row(row),
        submission_status=_submission_status_from_row(row),
        validation_ok=_boolish(row.get("validation_ok"), default=None),
        evaluator_backend=_evaluator_backend_from_row(row),
        evaluator_status=_evaluator_status_from_row(row),
        task_profile=str(row.get("task_profile") or ""),
        row_order=_int_or_zero(row.get("row_order")),
        method_text=" ".join(str(row.get(key) or "") for key in ("brief", "why", "selection_note", "metric_source_note", "metric_protocol")),
        route_id=str(row.get("route_id") or ""),
        execution_scale=str(row.get("execution_scale") or ""),
        solution_sha=str(row.get("solution_sha") or ""),
    )


def score_record_from_stage_payload(payload: dict[str, Any], *, worker_id: str = "") -> ScoreRecord | None:
    metric_event = payload.get("metric_event") if isinstance(payload.get("metric_event"), dict) else payload
    if not isinstance(metric_event, dict):
        return None
    row = {
        "metric_value": metric_event.get("metric_value"),
        "metric_name": metric_event.get("metric_name"),
        "lower_is_better": metric_event.get("lower_is_better"),
        "validation_ok": metric_event.get("validation_ok"),
        "validation_issue": metric_event.get("validation_issue"),
        "selection_eligible": metric_event.get("selection_eligible"),
        "selection_score": metric_event.get("selection_score"),
        "val_score_type": metric_event.get("val_score_type"),
        "candidate_ready": metric_event.get("candidate_ready"),
        "submission_status": metric_event.get("submission_status"),
        "evaluator_backend": metric_event.get("evaluator_backend"),
        "evaluator_status": metric_event.get("evaluator_status"),
        "task_profile": metric_event.get("task_profile"),
        "worker_id": worker_id or metric_event.get("worker_id") or payload.get("worker_id"),
        "stage_id": payload.get("stage_id") or metric_event.get("visible_stage_id") or metric_event.get("stage_id"),
        "candidate_id": metric_event.get("node_uid") or payload.get("candidate_id"),
        "submission_snapshot": metric_event.get("submission_snapshot"),
        "snapshot_path": metric_event.get("snapshot_path"),
        "brief": metric_event.get("brief"),
        "why": metric_event.get("why"),
        "route_evidence": metric_event.get("route_evidence"),
        "selection_note": metric_event.get("selection_note"),
        "metric_source_note": metric_event.get("metric_source_note"),
        "metric_validity": metric_event.get("metric_validity"),
        "metric_validity_note": metric_event.get("metric_validity_note"),
        "metric_protocol": metric_event.get("metric_protocol"),
        "route_id": metric_event.get("route_id"),
        "execution_scale": metric_event.get("execution_scale"),
        "solution_sha": metric_event.get("solution_sha"),
    }
    rec = score_record_from_stage_performance_row(row)
    if rec is None:
        return None
    return ScoreRecord(**{**rec.__dict__, "source": "stage_payload"})


def read_stage_performance_records(path: str | Path) -> list[ScoreRecord]:
    p = Path(path)
    if not p.is_file():
        return []
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []
    records = [rec for row in rows if (rec := score_record_from_stage_performance_row(row)) is not None]
    return records


def _cheap_signal_from_record(rec: ScoreRecord) -> bool:
    text = " ".join([rec.candidate_id, rec.stage_id, rec.val_score_type, rec.reason, rec.method_text]).lower()
    return any(word in text for word in ("logit", "metadata", "embedding", "oof", "cached"))


def _public_score(rec: ScoreRecord | None, *, current_worker_id: str = "") -> dict[str, Any]:
    if rec is None:
        return {}
    include_paths = bool(current_worker_id and rec.worker_id == current_worker_id)
    return rec.to_public_dict(include_paths=include_paths)


def build_score_summary(records: list[ScoreRecord], *, current_worker_id: str = "") -> dict[str, Any]:
    records = normalize_score_record_directions(records)
    best: ScoreRecord | None = None
    valid_best: ScoreRecord | None = None
    for rec in records:
        if _is_better(rec, best):
            best = rec
        if rec.valid_comparable and _is_better(rec, valid_best, use_selection_score=True):
            valid_best = rec
    capture_gap = bool(best is not None and (not best.valid_comparable) and _is_better(best, valid_best))
    recommended_action = ""
    if capture_gap:
        recommended_action = "metric_source_review" if best and best.validity == "risk_flagged_metric" else "score_contract_repair"
    cheap_records = [rec for rec in records if _cheap_signal_from_record(rec)]
    cheap_valid_records = [rec for rec in cheap_records if rec.valid_comparable]
    return {
        "best_score": _public_score(best, current_worker_id=current_worker_id),
        "valid_best_score": _public_score(valid_best, current_worker_id=current_worker_id),
        "capture_gap": capture_gap,
        "recommended_action": recommended_action,
        "record_count": len(records),
        "valid_record_count": sum(1 for rec in records if rec.valid_comparable),
        "cheap_signal_available": bool(cheap_records),
        "cheap_signal_valid_record_count": len(cheap_valid_records),
        "cheap_signal_best_score": _public_score(_best_of(cheap_valid_records), current_worker_id=current_worker_id),
        "worker_isolation": "global_scores_hide_cross_worker_paths; local_worker_scores_may_include_paths",
    }


def _clip(value: Any, limit: int = 180) -> str:
    text = str(value or "")
    return text[:limit]


def format_score_summary_context(score_summary: dict[str, Any]) -> list[str]:
    """Return prompt-safe score summary lines.

    Public resource context is shared across workers. Keep it score-centric and
    never expose artifact or snapshot paths even if the source summary contains
    local-worker path fields.
    """
    if not isinstance(score_summary, dict):
        return []
    has_context = bool((score_summary.get("best_score") or {}) or score_summary.get("cheap_signal_available"))
    if not has_context:
        return []
    best = score_summary.get("best_score") if isinstance(score_summary.get("best_score"), dict) else {}
    valid = score_summary.get("valid_best_score") if isinstance(score_summary.get("valid_best_score"), dict) else {}
    lines = ["score_summary:"]
    if best:
        best_line = "  - best_score: value={value}; source={source}; validity={validity}; metric_validity={metric_validity}; reason={reason}".format(
            value=best.get("value"),
            source=best.get("source") or "unknown",
            validity=best.get("validity") or "unknown",
            metric_validity=best.get("metric_validity") or "unknown",
            reason=_clip(best.get("reason")),
        )
        if best.get("metric_source_note"):
            best_line += "; metric_source_note=" + _clip(best.get("metric_source_note"), 220)
        lines.append(best_line)
    if valid:
        valid_line = "  - valid_best_score: value={value}; source={source}; validity={validity}; metric_validity={metric_validity}; reason={reason}".format(
            value=valid.get("value"),
            source=valid.get("source") or "unknown",
            validity=valid.get("validity") or "unknown",
            metric_validity=valid.get("metric_validity") or "unknown",
            reason=_clip(valid.get("reason")),
        )
        if valid.get("metric_source_note"):
            valid_line += "; metric_source_note=" + _clip(valid.get("metric_source_note"), 220)
        lines.append(valid_line)
    else:
        lines.append("  - valid_best_score: none")
    lines.append(f"  - capture_gap: {bool(score_summary.get('capture_gap'))}")
    action = str(score_summary.get("recommended_action") or "")
    if action:
        lines.append(f"  - recommended_action: {action}")
    cheap_best = (
        score_summary.get("cheap_signal_best_score")
        if isinstance(score_summary.get("cheap_signal_best_score"), dict)
        else {}
    )
    if cheap_best:
        lines.append(
            "  - cheap_signal_best_score: value={value}; worker_id={worker}; stage_id={stage}".format(
                value=cheap_best.get("value"),
                worker=cheap_best.get("worker_id") or "",
                stage=cheap_best.get("stage_id") or "",
            )
        )
    return lines


def _best_of(records: list[ScoreRecord]) -> ScoreRecord | None:
    best: ScoreRecord | None = None
    for rec in records:
        if _is_better(rec, best):
            best = rec
    return best


def build_stage_performance_score_summary(
    path: str | Path,
    *,
    current_worker_id: str = "",
    fallback_stage_payloads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = read_stage_performance_records(path)
    if not records and fallback_stage_payloads:
        records = [
            rec
            for payload in fallback_stage_payloads
            if (rec := score_record_from_stage_payload(payload, worker_id=current_worker_id)) is not None
        ]
    return build_score_summary(records, current_worker_id=current_worker_id)
