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

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scienceflow.solver.lnr.stage.metric_validity import infer_metric_validity, normalize_metric_validity


VALID_REASON_CODES = {
    "comparable_holdout",
    "oof_cv",
    "comparable_cv",
    "same_validation_meta_fit",
    "train_val_reuse",
    "proxy_metric",
    "invalid_submission",
    "leakage_risk",
    "unknown_protocol",
    "uncertain",
    "opt_solver_evaluator_ok",
    "authoritative_evaluator_ok",
}

HIGH_REASON_CODES = {
    "comparable_holdout",
    "oof_cv",
    "comparable_cv",
    "opt_solver_evaluator_ok",
    "authoritative_evaluator_ok",
}
VALID_CONFIDENCE = {"high", "medium", "low"}
_VALIDITY_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class MetricValidityJudgment:
    metric_validity: str
    selection_eligible: bool | None
    reason_code: str
    reason: str
    confidence: str
    source: str
    expected_lower_is_better: bool | None = None
    lower_is_better_ok: bool | None = None
    metric_direction_reason: str = ""


@dataclass(frozen=True)
class MetricOutputInterpretation:
    metric_found: bool
    metric_name: str
    metric_value: float | None
    split: str
    is_final: bool
    evidence_line: str
    confidence: str
    reason: str


def build_metric_validity_fields(
    *,
    metric_event: Mapping[str, Any],
    card_fields: Mapping[str, Any],
) -> dict[str, Any]:
    fields = dict(metric_event)
    for key in ("brief", "why", "route_evidence", "metric_note"):
        if card_fields.get(key) not in (None, ""):
            fields[key] = card_fields.get(key)
    if card_fields.get("metric_validity") not in (None, ""):
        fields["metric_validity"] = card_fields.get("metric_validity")
    return fields


def parse_metric_validity_judgment_text(text: str) -> MetricValidityJudgment | None:
    data = _parse_json_object(text)
    if not data:
        return None
    validity = normalize_metric_validity(data.get("metric_validity"))
    if not validity:
        return None
    reason_code = str(data.get("reason_code") or "uncertain").strip().lower()
    if reason_code not in VALID_REASON_CODES:
        reason_code = "uncertain"
    confidence = str(data.get("confidence") or "low").strip().lower()
    if confidence not in VALID_CONFIDENCE:
        confidence = "low"
    selection_raw = data.get("selection_eligible")
    selection_eligible = None
    if isinstance(selection_raw, bool):
        selection_eligible = selection_raw
    elif str(selection_raw).strip().lower() in {"true", "1", "yes"}:
        selection_eligible = True
    elif str(selection_raw).strip().lower() in {"false", "0", "no"}:
        selection_eligible = False
    expected_lower = _boolish_or_none(data.get("expected_lower_is_better"))
    direction_ok = _boolish_or_none(data.get("lower_is_better_ok"))
    return MetricValidityJudgment(
        metric_validity=validity,
        selection_eligible=selection_eligible,
        reason_code=reason_code,
        reason=str(data.get("reason") or "")[:600],
        confidence=confidence,
        source="metric_validity_llm",
        expected_lower_is_better=expected_lower,
        lower_is_better_ok=direction_ok,
        metric_direction_reason=str(data.get("metric_direction_reason") or "")[:600],
    )


def parse_metric_output_interpretation_text(text: str) -> MetricOutputInterpretation | None:
    data = _parse_json_object(text)
    if not data:
        return None
    metric_found = _boolish_or_none(data.get("metric_found"))
    is_final = _boolish_or_none(data.get("is_final"))
    confidence = str(data.get("confidence") or "low").strip().lower()
    if confidence not in VALID_CONFIDENCE:
        confidence = "low"
    raw_value = data.get("metric_value")
    metric_value: float | None = None
    if raw_value not in (None, ""):
        try:
            metric_value = float(raw_value)
        except (TypeError, ValueError):
            metric_value = None
    return MetricOutputInterpretation(
        metric_found=metric_found is True,
        metric_name=str(data.get("metric_name") or "")[:160],
        metric_value=metric_value,
        split=str(data.get("split") or "unknown").strip().lower()[:40],
        is_final=is_final is True,
        evidence_line=str(data.get("evidence_line") or "")[:1200],
        confidence=confidence,
        reason=str(data.get("reason") or "")[:600],
    )


def adjudicate_metric_validity(
    fields: Mapping[str, Any],
    *,
    llm_judgment: MetricValidityJudgment | None = None,
) -> dict[str, Any]:
    base_validity, base_note = infer_metric_validity(fields)
    no_explicit = dict(fields)
    no_explicit["metric_validity"] = ""
    system_validity, system_note = infer_metric_validity(no_explicit)

    final = base_validity
    notes = [base_note]
    source = "metric_validity_system"
    reason_code = _reason_code_from_note(base_note)
    confidence = "medium"
    authoritative_system_high = _is_authoritative_system_high(system_validity, system_note)
    evaluator_verified = _evaluator_verified(fields)

    if system_validity == "low" and not evaluator_verified:
        final = "low"
        notes = [system_note]
        source = "metric_validity_system_veto"
        reason_code = _reason_code_from_note(system_note)
        confidence = "high"
    elif authoritative_system_high:
        final = "high"
        notes = [system_note]
        source = "metric_validity_system"
        reason_code = _reason_code_from_note(system_note)
        confidence = "high"
    elif llm_judgment is not None:
        source = llm_judgment.source
        reason_code = llm_judgment.reason_code
        confidence = llm_judgment.confidence
        notes = [f"{llm_judgment.reason_code}: {llm_judgment.reason}".strip()]
        final = _merge_system_and_llm(
            base_validity=base_validity,
            system_validity=system_validity,
            judgment=llm_judgment,
        )
        if final != llm_judgment.metric_validity:
            notes.append(f"conservative_merge_from_{llm_judgment.metric_validity}_to_{final}")

    selection_eligible = _boolish(fields.get("selection_eligible"), default=False)
    if evaluator_verified:
        selection_eligible = True
    elif authoritative_system_high and final == "high":
        selection_eligible = True
    elif llm_judgment is not None and llm_judgment.selection_eligible is not None:
        selection_eligible = bool(llm_judgment.selection_eligible)
    if final != "high" and not evaluator_verified:
        selection_eligible = False

    return {
        "metric_validity": final,
        "metric_validity_note": "; ".join(note for note in notes if note)[:700],
        "metric_validity_reason_code": reason_code,
        "metric_validity_confidence": confidence,
        "metric_validity_source": source,
        "selection_eligible": selection_eligible,
    }


def _evaluator_verified(fields: Mapping[str, Any]) -> bool:
    evaluator_backend = str(fields.get("evaluator_backend") or "").strip()
    evaluator_status = str(fields.get("evaluator_status") or "").strip().lower()
    validation_ok = _boolish(fields.get("validation_ok"), default=None)
    selection_eligible = _boolish(fields.get("selection_eligible"), default=False) is True
    candidate_ready = _boolish(fields.get("candidate_ready"), default=False) is True
    submission_status = str(fields.get("submission_status") or "").strip()
    return (
        bool(evaluator_backend)
        and evaluator_status == "ok"
        and validation_ok is True
        and selection_eligible
        and candidate_ready
        and _direction_known(fields.get("lower_is_better"))
        and submission_status != "invalid_submission"
    )


def _direction_known(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if value is None:
        return False
    text = str(value).strip().lower()
    return bool(text and text not in {"none", "null", "nan", "unknown"})


def _merge_system_and_llm(
    *,
    base_validity: str,
    system_validity: str,
    judgment: MetricValidityJudgment,
) -> str:
    if judgment.metric_validity == "low":
        return "low"
    if judgment.metric_validity == "medium":
        return _min_validity(base_validity, "medium")
    if judgment.confidence == "low":
        return _min_validity(base_validity, "medium")
    if judgment.metric_validity == "high":
        if judgment.reason_code in HIGH_REASON_CODES and system_validity == "high":
            return "high"
        if judgment.reason_code in HIGH_REASON_CODES and system_validity == "medium":
            return "medium"
        return _min_validity(base_validity, "medium")
    return _min_validity(base_validity, "medium")


def _min_validity(a: str, b: str) -> str:
    return a if _VALIDITY_RANK.get(a, 1) <= _VALIDITY_RANK.get(b, 1) else b


def _is_authoritative_system_high(system_validity: str, system_note: str) -> bool:
    return system_validity == "high" and _reason_code_from_note(system_note) in {
        "opt_solver_evaluator_ok",
        "authoritative_evaluator_ok",
    }


def _reason_code_from_note(note: str) -> str:
    text = str(note or "").lower()
    if "authoritative_evaluator_ok" in text:
        return "authoritative_evaluator_ok"
    if "opt_solver_evaluator_ok" in text:
        return "opt_solver_evaluator_ok"
    if "invalid_submission" in text or "schema" in text:
        return "invalid_submission"
    if "leak" in text or "target" in text:
        return "leakage_risk"
    if "fulltrain" in text or "train+val" in text:
        return "train_val_reuse"
    if "same_validation" in text or "meta" in text or "overfit" in text:
        return "same_validation_meta_fit"
    if "unknown" in text or "insufficient" in text:
        return "unknown_protocol"
    if "proxy" in text or "diagnostic" in text:
        return "proxy_metric"
    return "uncertain"


def _boolish(value: Any, *, default: bool = False) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "ok"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _boolish_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            data, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else {}
    return {}
