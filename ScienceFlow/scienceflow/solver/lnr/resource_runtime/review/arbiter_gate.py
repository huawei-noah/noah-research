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

from .progress import LOW_PROGRESS_SIGNALS

KILL_AND_REPLAN_ACTIONS = {"KILL_AND_REPLAN"}
TERMINATING_SAFE_ACTIONS = {"STOP_BOUNDARY_VIOLATION"}
STOP_ADVISORY_PREFERENCES = {
    "safe_to_stop",
    "timebox_expired_without_progress",
    "kill_and_replan",
    "stop",
    "stop_and_replan",
    "replan",
}
DETERMINISTIC_DECISION_REASON_CODES = {
    "negotiated_timebox_missed_commitment",
    "repeated_failed_proof_windows",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _confidence(decision: dict[str, Any]) -> str:
    raw = str(decision.get("confidence") or "medium").strip().lower()
    return raw if raw in {"low", "medium", "high"} else "medium"


def _progress(proposal: dict[str, Any]) -> dict[str, Any]:
    raw = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
    return dict(raw or {})


def _reason(proposal: dict[str, Any]) -> str:
    return str(proposal.get("reason_code") or "").lower()


def proposal_has_phase_completion_protection(proposal: dict[str, Any]) -> bool:
    progress = _progress(proposal)
    return bool(
        progress.get("phase_completion_protected")
        and progress.get("structured_progress_recent")
        and str(progress.get("eta_confidence") or "").strip().lower() == "high"
        and 0.0 <= _float(progress.get("eta_to_current_phase_end_sec"), -1.0) <= 300.0
        and not progress.get("deadline_event")
    )


def _resource_metric_value(proposal: dict[str, Any]) -> dict[str, Any]:
    raw = proposal.get("resource_metric_value")
    if isinstance(raw, dict) and raw:
        return dict(raw)
    preview = proposal.get("decision_preview")
    if isinstance(preview, dict) and isinstance(preview.get("resource_metric_value"), dict):
        return dict(preview["resource_metric_value"])
    return {}


def proposal_has_live_best_metric_protection(proposal: dict[str, Any]) -> bool:
    """Protect active, expensive-to-stop work that already beats the formal best."""

    metric = _resource_metric_value(proposal)
    progress = _progress(proposal)
    execution = proposal.get("execution_facts") if isinstance(proposal.get("execution_facts"), dict) else {}
    eta = _float(progress.get("eta_to_current_phase_end_sec"), -1.0)
    remaining = _float(progress.get("remaining_useful_budget_sec"), -1.0)
    progress_signal = str(progress.get("progress_signal") or "").strip().lower()
    active_progress = bool(
        progress_signal in {"active", "progressing"}
        and progress.get("structured_progress_recent")
    )
    return bool(
        metric.get("useful") is True
        and str(metric.get("status") or "").strip().lower() == "above_observed_best"
        and active_progress
        and execution.get("assigned_resource_idle") is False
        and str(progress.get("stop_cost") or "").strip().lower() == "high"
        and not progress.get("deadline_event")
        and eta > 0.0
        and remaining > 0.0
        and eta <= remaining
    )


def _quick_probe_hard_review_sec(progress: dict[str, Any]) -> float:
    quick_probe = progress.get("quick_probe") if isinstance(progress.get("quick_probe"), dict) else {}
    return _float(
        quick_probe.get("quick_probe_hard_review_sec")
        or progress.get("quick_probe_hard_review_sec"),
        900.0,
    )


def _artifact_update_is_log_like(item: Any) -> bool:
    if isinstance(item, dict):
        path = str(item.get("path") or item.get("name") or "").strip().lower()
        if bool(item.get("recoverable_artifact_on_disk") or item.get("run_state_confirmed") or item.get("done_marker")):
            return False
    else:
        path = str(item or "").strip().lower()
    if not path:
        return False
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".log"):
        return True
    return bool(name.endswith(".txt") and any(token in name for token in ("log", "stdout", "stderr", "trace")))


def _has_recent_artifact_value_signal(progress: dict[str, Any]) -> bool:
    updates = progress.get("artifact_updates") if isinstance(progress.get("artifact_updates"), list) else []
    if not updates:
        return False
    artifact_age = _float(progress.get("artifact_last_update_age_sec"), 1e9)
    if artifact_age >= 300.0:
        return False
    return not all(_artifact_update_is_log_like(item) for item in updates)


def _has_recent_stdout_value_signal(progress: dict[str, Any]) -> bool:
    observation = progress.get("stdout_observation") if isinstance(progress.get("stdout_observation"), dict) else {}
    stream = observation.get("stdout_stream") if isinstance(observation.get("stdout_stream"), dict) else {}
    if observation:
        if not stream.get("present"):
            return False
        if not stream.get("fresh"):
            return False
    stdout_lines = int(_float(progress.get("stdout_lines"), 0.0))
    stdout_bytes = int(_float(progress.get("stdout_bytes"), 0.0))
    has_stdout = bool(progress.get("meaningful_stdout") or stdout_lines > 0 or stdout_bytes > 0)
    if not has_stdout:
        return False

    age_raw = progress.get("stdout_last_line_age_sec", progress.get("stdout_age_sec"))
    if age_raw is None:
        return True
    stdout_age = _float(age_raw, 0.0)
    signal = str(progress.get("progress_signal") or "").strip().lower()
    confidence = str(progress.get("progress_confidence") or "").strip().lower()
    stale_limit = max(300.0, min(_quick_probe_hard_review_sec(progress), 900.0))
    if stdout_age >= stale_limit:
        return False
    if signal in LOW_PROGRESS_SIGNALS and confidence == "high" and stdout_age >= stale_limit:
        return False
    return True


def _iter_advisory_payloads(proposal: dict[str, Any]):
    for key in ("main_agent_advisory", "advisory", "advisory_support"):
        raw = proposal.get(key)
        if isinstance(raw, dict):
            yield raw


def _normalized_fact(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def proposal_advisory_fact_conflicts(
    proposal: dict[str, Any],
    advisory: dict[str, Any],
) -> list[dict[str, str]]:
    conflicts: list[dict[str, str]] = []
    progress = _progress(proposal)
    metric = _resource_metric_value(proposal)
    execution = proposal.get("execution_facts") if isinstance(proposal.get("execution_facts"), dict) else {}
    phase_claim = _normalized_fact(advisory.get("observed_phase"))
    phase_observed = _normalized_fact(progress.get("known_stage") or metric.get("phase"))
    if phase_claim not in {"", "unknown"} and phase_observed and phase_claim != phase_observed:
        conflicts.append({"field": "phase", "claimed": phase_claim, "observed": phase_observed})
    checkpoint_claim = _normalized_fact(advisory.get("observed_checkpoint"))
    if checkpoint_claim in {"yes", "no"}:
        checkpoint_observed = "yes" if progress.get("recoverable_artifact_on_disk") else "no"
        if checkpoint_claim != checkpoint_observed:
            conflicts.append({"field": "checkpoint", "claimed": checkpoint_claim, "observed": checkpoint_observed})
    resource_claim = _normalized_fact(advisory.get("observed_resource_state"))
    if resource_claim in {"active", "idle"} and isinstance(execution.get("assigned_resource_idle"), bool):
        resource_observed = "idle" if execution["assigned_resource_idle"] else "active"
        if resource_claim != resource_observed:
            conflicts.append({"field": "resource_state", "claimed": resource_claim, "observed": resource_observed})
    metric_claim = _normalized_fact(advisory.get("observed_metric_status"))
    metric_observed = _normalized_fact(metric.get("status"))
    if metric_claim not in {"", "unknown"} and metric_observed and metric_claim != metric_observed:
        conflicts.append({"field": "metric_status", "claimed": metric_claim, "observed": metric_observed})
    return conflicts


def proposal_advisory_gate_status(proposal: dict[str, Any]) -> dict[str, Any]:
    """Return the best-effort advisory support state used in kill gates."""

    fallback = {
        "captured": False,
        "status": "",
        "preference": "",
        "confidence": "",
        "supports_stop_preference": False,
        "fact_conflicts": [],
        "support": False,
    }
    for raw in _iter_advisory_payloads(proposal):
        status = str(raw.get("advisory_status") or raw.get("status") or "").strip().lower()
        preference = str(raw.get("preference") or raw.get("action") or "").strip().lower()
        confidence = str(raw.get("confidence") or "").strip().lower()
        conflicts = proposal_advisory_fact_conflicts(proposal, raw)
        captured = not status or status in {"captured", "captured_with_parse_issue"}
        supports_stop = preference in STOP_ADVISORY_PREFERENCES
        state = {
            "captured": captured,
            "status": status,
            "preference": preference,
            "confidence": confidence,
            "supports_stop_preference": supports_stop,
            "fact_conflicts": conflicts,
            "support": bool(captured and supports_stop and confidence == "high" and not conflicts),
        }
        if captured and preference:
            return state
        if captured:
            fallback = state
    return fallback


def proposal_has_advisory_support(proposal: dict[str, Any]) -> bool:
    return bool(proposal_advisory_gate_status(proposal).get("support"))


def proposal_has_captured_advisory(proposal: dict[str, Any]) -> bool:
    state = proposal_advisory_gate_status(proposal)
    return bool(state.get("captured") and state.get("preference"))


def _blocked_kill_reason(kill_class: str, advisory_state: dict[str, Any]) -> str:
    if kill_class != "discretionary":
        return "missing_support_evidence"
    if not advisory_state.get("captured"):
        return "owning_agent_advisory_missing"
    if not advisory_state.get("preference"):
        return "owning_agent_advisory_missing"
    if not advisory_state.get("supports_stop_preference"):
        return "owning_agent_advisory_not_stop_support"
    if str(advisory_state.get("confidence") or "").strip().lower() != "high":
        return "owning_agent_advisory_confidence_not_high"
    return "missing_support_evidence"

def proposal_has_multi_window_progress_support(proposal: dict[str, Any], *, min_windows: int = 2) -> bool:
    progress = _progress(proposal)
    signal = str(progress.get("progress_signal") or "").strip().lower()
    confidence = str(progress.get("progress_confidence") or "").strip().lower()
    try:
        windows = int(progress.get("progress_signal_windows") or 0)
    except (TypeError, ValueError):
        windows = 0
    if (
        signal in LOW_PROGRESS_SIGNALS
        and confidence == "high"
        and (windows >= max(1, int(min_windows or 1)) or bool(progress.get("multi_window_low_progress")))
    ):
        return True
    return False


def _quick_probe_has_no_value_signal(proposal: dict[str, Any]) -> bool:
    progress = _progress(proposal)
    execution = proposal.get("execution_facts") if isinstance(proposal.get("execution_facts"), dict) else {}
    metric_count = int(_float(progress.get("metric_history_line_count"), 0.0))
    return not bool(
        progress.get("near_submission")
        or _has_recent_stdout_value_signal(progress)
        or metric_count > 0
        or _has_recent_artifact_value_signal(progress)
        or progress.get("recoverable_artifact_on_disk")
        or execution.get("has_recent_metric_or_submission")
        or execution.get("has_recent_useful_artifact")
    )


def _quick_probe_runtime_exceeded_without_value(proposal: dict[str, Any]) -> bool:
    progress = _progress(proposal)
    quick_probe = progress.get("quick_probe") if isinstance(progress.get("quick_probe"), dict) else {}
    reason = _reason(proposal)
    is_quick_probe = bool(quick_probe.get("quick_probe_candidate")) or "quick_probe_runtime_exceeded" in reason
    if not is_quick_probe:
        return False
    runtime = _float(progress.get("runtime_sec"), 0.0)
    hard_review = max(1.0, _quick_probe_hard_review_sec(progress))
    return runtime >= hard_review and _quick_probe_has_no_value_signal(proposal)


def _is_false(value: Any) -> bool:
    return value is False or str(value).strip().lower() == "false"


def proposal_has_unaffordable_finish_without_output(proposal: dict[str, Any]) -> bool:
    progress = _progress(proposal)
    execution = proposal.get("execution_facts") if isinstance(proposal.get("execution_facts"), dict) else {}
    confidence = str(progress.get("eta_confidence") or "low").strip().lower()
    eta = max(
        _float(progress.get("eta_to_deliverable_sec"), 0.0),
        _float(progress.get("eta_to_current_phase_end_sec"), 0.0),
    )
    metric_count = int(_float(progress.get("metric_history_line_count"), 0.0))
    deadline_event = bool(progress.get("deadline_event")) or "deadline_finalization" in _reason(proposal)
    has_metric_or_submission = bool(
        progress.get("near_submission")
        or metric_count > 0
        or execution.get("has_recent_metric_or_submission")
    )
    has_recoverable_or_useful_artifact = bool(
        progress.get("recoverable_artifact_on_disk")
        or execution.get("has_recent_useful_artifact")
    )
    return bool(
        _is_false(progress.get("finish_feasible"))
        and confidence in {"medium", "high"}
        and eta > 0.0
        and deadline_event
        and not has_metric_or_submission
        and not has_recoverable_or_useful_artifact
    )


def proposal_has_deterministic_kill_support(proposal: dict[str, Any]) -> bool:
    reason = _reason(proposal)
    progress = _progress(proposal)
    blocker = proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {}
    suspect = blocker.get("active_lease_suspect") if isinstance(blocker.get("active_lease_suspect"), dict) else {}
    execution = proposal.get("execution_facts") if isinstance(proposal.get("execution_facts"), dict) else {}
    if "boundary" in reason:
        return True
    if "invalid_training_metrics" in reason:
        return True
    if "repeated_failed_proof_windows" in reason:
        return True
    if _quick_probe_runtime_exceeded_without_value(proposal):
        return True
    if proposal_has_unaffordable_finish_without_output(proposal):
        return True
    if bool(suspect.get("resource_suspect")):
        return True
    runtime = _float(progress.get("runtime_sec"), 0.0)
    no_value_signal = not bool(
        progress.get("near_submission")
        or _has_recent_stdout_value_signal(progress)
        or int(_float(progress.get("metric_history_line_count"), 0.0)) > 0
        or execution.get("has_recent_metric_or_submission")
    )
    deadline_event = bool(progress.get("deadline_event")) or "deadline_finalization" in reason
    resource_idle_or_mismatch = bool(
        execution.get("intent_device_mismatch")
        or (execution.get("assigned_resource_idle") and execution.get("cpu_busy"))
        or execution.get("activity_without_metric_submission_or_stdout")
    )
    if no_value_signal and resource_idle_or_mismatch and (deadline_event or runtime >= 1800.0):
        return True
    return False


def decision_has_deterministic_kill_support(decision: dict[str, Any]) -> bool:
    reason_code = str(decision.get("reason_code") or "").strip().lower()
    source = str(decision.get("source") or "").strip().lower()
    if source == "proof_window_override" and reason_code in DETERMINISTIC_DECISION_REASON_CODES:
        return True
    stall_escalation = decision.get("stall_escalation")
    return bool(
        isinstance(stall_escalation, dict)
        and str(stall_escalation.get("reason") or "").strip().lower()
        == "repeated_high_confidence_no_work_stall"
    )


def proposal_has_final_resource_review_support(proposal: dict[str, Any]) -> bool:
    escalation = proposal.get("resource_budget_escalation") if isinstance(proposal.get("resource_budget_escalation"), dict) else {}
    if str(escalation.get("kind") or "").strip().lower() != "final_resource_review":
        return False
    failed = int(_float(escalation.get("failed_proof_window_count"), 0.0))
    max_windows = max(1, int(_float(escalation.get("max_proof_windows"), 1.0)))
    return failed >= max_windows


def proposal_kill_class(proposal: dict[str, Any]) -> str:
    raw = str(proposal.get("kill_class") or "").strip().lower()
    if raw in {"hard_safety", "discretionary", "none"}:
        return raw
    reason = _reason(proposal)
    trigger_text = " ".join(str(x) for x in (proposal.get("trigger_reasons") or [])).lower()
    text = f"{reason} {trigger_text}"
    if any(token in text for token in ("boundary", "oom", "out_of_memory", "hard_fuse", "user_stop", "orphan", "runaway")):
        return "hard_safety"
    return "discretionary"


def proposal_has_hard_safety_support(proposal: dict[str, Any]) -> bool:
    if proposal_kill_class(proposal) == "hard_safety":
        return True
    reason = _reason(proposal)
    trigger_text = " ".join(str(x) for x in (proposal.get("trigger_reasons") or [])).lower()
    text = f"{reason} {trigger_text}"
    return any(token in text for token in ("boundary", "oom", "out_of_memory", "hard_fuse", "user_stop", "orphan", "runaway"))


def stop_after_deliverable_support_detail(proposal: dict[str, Any]) -> dict[str, Any]:
    """Explain whether STOP_AFTER has direct deliverable-complete support.

    Missing evidence means a proposal path failed to populate the required
    deliverable fields. Unknown evidence means the fields were present but do
    not prove validity or ownership yet. Keeping these separate makes gate
    downgrades debuggable.
    """

    reason = _reason(proposal)
    progress = _progress(proposal)
    blocker = proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {}
    near = blocker.get("near_deliverable") if isinstance(blocker.get("near_deliverable"), dict) else {}
    trigger_text = " ".join(str(x) for x in (proposal.get("trigger_reasons") or [])).lower()

    validity_sources = {
        "progress.deliverable_validity": progress.get("deliverable_validity") if "deliverable_validity" in progress else None,
        "blocker.deliverable_validity": blocker.get("deliverable_validity") if "deliverable_validity" in blocker else None,
        "blocker.near_deliverable.deliverable_validity": near.get("deliverable_validity") if "deliverable_validity" in near else None,
    }
    seen_fields = [key for key, value in validity_sources.items() if value is not None]
    validities = {key: str(value or "").strip().lower() for key, value in validity_sources.items() if value is not None}
    if any(value == "produced_invalid" for value in validities.values()) or "produced_invalid_deliverable" in trigger_text:
        return {
            "supported": False,
            "blocked_reason": "stop_after_evidence_invalid",
            "evidence_state": "invalid",
            "observed_fields": seen_fields,
        }
    if any(value == "produced_valid" for value in validities.values()):
        return {
            "supported": True,
            "blocked_reason": "",
            "evidence_state": "valid",
            "observed_fields": seen_fields,
        }

    state_sources = {
        "progress.deliverable_completion_state": progress.get("deliverable_completion_state"),
        "blocker.deliverable_completion_state": blocker.get("deliverable_completion_state"),
        "blocker.near_deliverable": near,
    }
    state_fields: list[str] = []
    unknown_seen = False
    for key, state in state_sources.items():
        if not isinstance(state, dict) or not state:
            continue
        state_fields.append(key)
        state_validity = str(state.get("deliverable_validity") or state.get("validity") or "").strip().lower()
        if state_validity == "produced_invalid":
            return {
                "supported": False,
                "blocked_reason": "stop_after_evidence_invalid",
                "evidence_state": "invalid",
                "observed_fields": seen_fields + state_fields,
            }
        if bool(state.get("complete")) and state_validity == "produced_valid":
            return {
                "supported": True,
                "blocked_reason": "",
                "evidence_state": "valid",
                "observed_fields": seen_fields + state_fields,
            }
        unknown_seen = True

    observed = seen_fields + state_fields
    if not observed:
        return {
            "supported": False,
            "blocked_reason": "stop_after_evidence_missing",
            "evidence_state": "missing",
            "observed_fields": [],
            "reason_had_completion_text": "deliverable_complete" in reason or "submission_complete" in reason,
        }
    return {
        "supported": False,
        "blocked_reason": "stop_after_evidence_unknown" if unknown_seen or observed else "stop_after_evidence_missing",
        "evidence_state": "unknown" if unknown_seen or observed else "missing",
        "observed_fields": observed,
        "reason_had_completion_text": "deliverable_complete" in reason or "submission_complete" in reason,
    }


def proposal_has_stop_after_deliverable_support(proposal: dict[str, Any]) -> bool:
    """Return whether STOP_AFTER has direct deliverable-complete support."""

    return bool(stop_after_deliverable_support_detail(proposal).get("supported"))


def enforce_arbiter_kill_gate(
    decision: dict[str, Any],
    proposal: dict[str, Any],
    *,
    min_windows: int = 2,
    require_high_confidence: bool = True,
) -> dict[str, Any]:
    """Apply V4 kill-like action gates to an arbiter decision.

    Advisory support is evidence only. It can satisfy the support side of the
    gate, but it cannot replace the arbiter's own high confidence requirement.
    """

    out = dict(decision or {})
    action = str(out.get("action") or "OBSERVE_MORE").strip().upper()
    if action == "STOP_AFTER_DELIVERABLE_AND_RELEASE":
        out.update({
            "action": "OBSERVE_MORE",
            "reason": (
                str(out.get("reason") or "deliverable state is not resource-kill evidence").rstrip(".")
                + "; downgraded from STOP_AFTER_DELIVERABLE_AND_RELEASE because deliverable state belongs to selection, not resource kill"
            ),
            "gate": {
                "allowed": False,
                "blocked_reason": "deliverable_not_resource_kill_evidence",
                "original_action": action,
            },
        })
        return out
    if action not in KILL_AND_REPLAN_ACTIONS and action not in TERMINATING_SAFE_ACTIONS:
        return out

    if action == "STOP_BOUNDARY_VIOLATION":
        out.setdefault("gate", {"allowed": True, "reason": "deterministic_boundary_violation"})
        return out

    if require_high_confidence and _confidence(out) != "high":
        original = action
        out.update({
            "action": "OBSERVE_MORE",
            "confidence": _confidence(out),
            "reason": (
                str(out.get("reason") or "kill-like action lacked high arbiter confidence").rstrip(".")
                + f"; downgraded from {original} because arbiter confidence was not high"
            ),
            "gate": {"allowed": False, "blocked_reason": "arbiter_confidence_not_high", "original_action": original},
        })
        return out

    hard_safety_support = proposal_has_hard_safety_support(proposal)
    advisory_state = proposal_advisory_gate_status(proposal)
    advisory_conflicts = list(advisory_state.get("fact_conflicts") or [])
    if (
        advisory_state.get("supports_stop_preference")
        and advisory_conflicts
        and not hard_safety_support
    ):
        original = action
        out.update({
            "action": "OBSERVE_MORE",
            "observe_more_sec": 120.0,
            "reason": "owning-agent stop advisory conflicts with machine-observed execution facts",
            "gate": {
                "allowed": False,
                "blocked_reason": "advisory_fact_conflict",
                "original_action": original,
                "advisory_fact_conflicts": advisory_conflicts,
            },
        })
        return out
    if proposal_has_phase_completion_protection(proposal) and not hard_safety_support:
        original = action
        eta = max(0.0, _float(_progress(proposal).get("eta_to_current_phase_end_sec"), 0.0))
        out.update({
            "action": "OBSERVE_MORE",
            "observe_more_sec": max(60.0, min(300.0, eta * 1.5 if eta > 0.0 else 120.0)),
            "reason": (
                str(out.get("reason") or "phase completion is close and progress is recent").rstrip(".")
                + f"; downgraded from {original} because current phase ETA is bounded and stopping would discard near-complete work"
            ),
            "gate": {
                "allowed": False,
                "blocked_reason": "near_phase_completion",
                "original_action": original,
                "phase_completion_protected": True,
                "eta_to_current_phase_end_sec": eta,
            },
        })
        return out

    if proposal_has_live_best_metric_protection(proposal) and not hard_safety_support:
        original = action
        progress = _progress(proposal)
        metric = _resource_metric_value(proposal)
        eta = max(0.0, _float(progress.get("eta_to_current_phase_end_sec"), 0.0))
        remaining = max(0.0, _float(progress.get("remaining_useful_budget_sec"), 0.0))
        out.update({
            "action": "OBSERVE_MORE",
            "observe_more_sec": max(60.0, min(300.0, eta)),
            "reason": (
                str(out.get("reason") or "live metric is better than the observed stage best").rstrip(".")
                + f"; downgraded from {original} because the improved active phase can finish within the remaining useful budget"
            ),
            "gate": {
                "allowed": False,
                "blocked_reason": "live_metric_above_observed_best",
                "original_action": original,
                "live_best_metric_protected": True,
                "metric_name": str(metric.get("metric_name") or ""),
                "metric_value": metric.get("value"),
                "observed_best_value": metric.get("best_value"),
                "eta_to_current_phase_end_sec": eta,
                "remaining_useful_budget_sec": remaining,
            },
        })
        return out

    multi_window_support = proposal_has_multi_window_progress_support(proposal, min_windows=min_windows)
    advisory_support = bool(advisory_state.get("support"))
    deterministic_support = (
        proposal_has_deterministic_kill_support(proposal)
        or decision_has_deterministic_kill_support(out)
        or proposal_has_final_resource_review_support(proposal)
    )
    structured_support = bool(
        proposal.get("structured_opportunity_cost_support")
        or ((proposal.get("review_history") if isinstance(proposal.get("review_history"), dict) else {}).get("structured_opportunity_cost_support"))
    )
    kill_class = proposal_kill_class(proposal)
    if hard_safety_support or advisory_support or deterministic_support:
        out["gate"] = {
            "allowed": True,
            "arbiter_high_confidence": _confidence(out) == "high",
            "kill_class": "hard_safety" if hard_safety_support else "deterministic_resource" if deterministic_support and not advisory_support else kill_class,
            "hard_safety_support": hard_safety_support,
            "multi_window_progress": multi_window_support,
            "advisory_support": advisory_support,
            "captured_advisory": proposal_has_captured_advisory(proposal),
            "deterministic_support": deterministic_support,
            "structured_opportunity_cost_support": structured_support,
            "advisory_fact_conflicts": advisory_conflicts,
        }
        return out

    progress = _progress(proposal)
    signal = str(progress.get("progress_signal") or "").strip().lower()
    original = action
    replacement = "OBSERVE_MORE"
    if signal in LOW_PROGRESS_SIGNALS and not (multi_window_support or structured_support or deterministic_support):
        replacement = "MARK_STALLED_NO_KILL"
    advisory_state = proposal_advisory_gate_status(proposal)
    blocked_reason = _blocked_kill_reason(kill_class, advisory_state)
    out.update({
        "action": replacement,
        "reason": (
            str(out.get("reason") or "kill-like action lacked support evidence").rstrip(".")
            + f"; downgraded from {original} because {blocked_reason}"
        ),
        "gate": {
            "allowed": False,
            "blocked_reason": blocked_reason,
            "original_action": original,
            "kill_class": kill_class,
            "hard_safety_support": hard_safety_support,
            "multi_window_progress": multi_window_support,
            "advisory_support": advisory_support,
            "captured_advisory": bool(advisory_state.get("captured") and advisory_state.get("preference")),
            "advisory_confidence": advisory_state.get("confidence") or "",
            "advisory_preference": advisory_state.get("preference") or "",
            "advisory_supports_stop_preference": bool(advisory_state.get("supports_stop_preference")),
            "deterministic_support": deterministic_support,
            "structured_opportunity_cost_support": structured_support,
        },
    })
    return out
