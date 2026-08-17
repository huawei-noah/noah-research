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
import re
import time
import uuid
from typing import Any

from .arbiter_gate import (
    proposal_has_advisory_support,
    proposal_has_deterministic_kill_support,
    proposal_has_multi_window_progress_support,
    proposal_has_unaffordable_finish_without_output,
)
from scienceflow.solver.lnr.resource_feedback_guidance import resource_feedback_guidance_value

CANONICAL_ACTIONS = {
    "CONTINUE",
    "DENY_KILL",
    "OBSERVE_MORE",
    "MARK_STALLED_NO_KILL",
    "KILL_AND_REPLAN",
    "RELEASE_IDLE_LEASE",
    "GRANT_SHARED_GPU_LEASE",
    "DENY_SHARE_USE_CPU_SUPPORT",
    "CONTINUE_SHARED_OBSERVE",
    "STOP_SECONDARY_SHARED_JOB",
    "REVOKE_SHARED_LEASE",
}
INTERNAL_DETERMINISTIC_ACTIONS = {"STOP_BOUNDARY_VIOLATION"}
VALID_ACTIONS = CANONICAL_ACTIONS | INTERNAL_DETERMINISTIC_ACTIONS
TERMINATING_ACTIONS = {"KILL_AND_REPLAN", "STOP_BOUNDARY_VIOLATION"}
LEGACY_ACTION_ALIASES = {"APPROVE_KILL", "APPROVE_KILL_STALLED"}
CANONICAL_REVIEW_OUTCOMES = {"KILL", "NO_ACTION", "TIMEBOX"}
VALID_REVIEW_CONFIDENCE = {"low", "medium", "high"}
VALID_TIMEBOX_CLEAR_ON = {
    "metric_update",
    "artifact_growth",
    "progress_advance",
    "active_work_recovered",
    "opportunity_cost_cleared",
    "useful_progress",
}


def normalize_action_value(action: Any, proposal: dict[str, Any] | None = None, *, default: str = "OBSERVE_MORE") -> tuple[str, str]:
    raw = str(action or default).strip().upper()
    reason_code = str((proposal or {}).get("reason_code") or "").lower()
    if raw == "STOP_AFTER_DELIVERABLE_AND_RELEASE":
        fallback = str(default or "OBSERVE_MORE").strip().upper()
        return (fallback if fallback in VALID_ACTIONS else "OBSERVE_MORE"), raw
    if raw in {"APPROVE_KILL", "APPROVE_KILL_STALLED"}:
        if raw == "APPROVE_KILL" and ("deliverable_complete" in reason_code or "submission_complete" in reason_code):
            fallback = str(default or "OBSERVE_MORE").strip().upper()
            return (fallback if fallback in VALID_ACTIONS else "OBSERVE_MORE"), raw
        if raw == "APPROVE_KILL" and "boundary" in reason_code:
            return "STOP_BOUNDARY_VIOLATION", raw
        return "KILL_AND_REPLAN", raw
    if raw not in VALID_ACTIONS:
        fallback = str(default or "OBSERVE_MORE").strip().upper()
        return (fallback if fallback in VALID_ACTIONS else "OBSERVE_MORE"), raw
    return raw, raw


def allowed_actions_for_proposal(proposal_type: str) -> set[str]:
    kind = str(proposal_type or "").strip().lower()
    if kind == "kill_proposal":
        return {"DENY_KILL", "OBSERVE_MORE", "MARK_STALLED_NO_KILL", "KILL_AND_REPLAN", "RELEASE_IDLE_LEASE"}
    if kind == "resource_contention_review":
        return {"CONTINUE", "DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN", "RELEASE_IDLE_LEASE"}
    if kind == "periodic_efficiency_review":
        return {"CONTINUE", "DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN", "RELEASE_IDLE_LEASE"}
    if kind == "quick_probe_review":
        return {"DENY_KILL", "OBSERVE_MORE", "MARK_STALLED_NO_KILL", "KILL_AND_REPLAN"}
    if kind == "deadline_finalize_review":
        return {"OBSERVE_MORE", "KILL_AND_REPLAN"}
    if kind == "task_gpu_share_review":
        return {"GRANT_SHARED_GPU_LEASE", "DENY_SHARE_USE_CPU_SUPPORT", "CONTINUE_SHARED_OBSERVE"}
    if kind == "shared_runtime_review":
        return {"CONTINUE_SHARED_OBSERVE", "STOP_SECONDARY_SHARED_JOB", "REVOKE_SHARED_LEASE"}
    return set(VALID_ACTIONS)


def fallback_action_for_proposal(proposal_type: str) -> str:
    kind = str(proposal_type or "").strip().lower()
    if kind in {"task_gpu_share_review", "shared_runtime_review"}:
        return "CONTINUE_SHARED_OBSERVE"
    return "OBSERVE_MORE"


def enforce_proposal_action_allowlist(decision: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    out = dict(decision or {})
    action = str(out.get("action") or "OBSERVE_MORE").strip().upper()
    allowed = allowed_actions_for_proposal(str(proposal.get("proposal_type") or ""))
    if action in allowed:
        return out
    fallback = fallback_action_for_proposal(str(proposal.get("proposal_type") or ""))
    if fallback not in allowed:
        fallback = "OBSERVE_MORE" if "OBSERVE_MORE" in allowed else sorted(allowed or {"OBSERVE_MORE"})[0]
    out.update({
        "action": fallback,
        "reason": (
            str(out.get("reason") or "arbiter action was not allowed for proposal type").rstrip(".")
            + f"; downgraded from {action} to {fallback} because proposal_type={proposal.get('proposal_type') or 'unknown'} does not allow it"
        ),
        "gate": {
            **(out.get("gate") if isinstance(out.get("gate"), dict) else {}),
            "allowed": False,
            "blocked_reason": "proposal_action_not_allowed",
            "original_action": action,
            "fallback_action": fallback,
            "proposal_type": str(proposal.get("proposal_type") or ""),
        },
    })
    return out


def build_resource_arbiter_prompt(proposal: dict[str, Any]) -> str:
    """Build the isolated arbiter prompt for one resource proposal."""
    compact = {
        "proposal_id": proposal.get("proposal_id"),
        "proposal_type": proposal.get("proposal_type"),
        "reason_code": proposal.get("reason_code"),
        "severity": proposal.get("severity"),
        "trigger_reasons": proposal.get("trigger_reasons"),
        "suggested_actions": proposal.get("suggested_actions"),
        "resource_snapshot": proposal.get("resource_snapshot"),
        "progress_snapshot": proposal.get("progress_snapshot"),
        "execution_facts": proposal.get("execution_facts"),
        "research_cadence": proposal.get("research_cadence"),
        "blocker": proposal.get("blocker"),
        "waiters": proposal.get("waiters"),
        "share_observation": proposal.get("share_observation"),
        "share_decision_facts": proposal.get("share_decision_facts"),
        "main_agent_advisory": proposal.get("main_agent_advisory"),
        "review_history": proposal.get("review_history"),
        "budget_priority": proposal.get("budget_priority"),
        "budget_context": proposal.get("budget_context"),
        "resource_budget_escalation": proposal.get("resource_budget_escalation"),
        "contention_context": proposal.get("contention_context"),
        "clear_on_cadence": proposal.get("clear_on_cadence"),
        "structured_opportunity_cost_support": proposal.get("structured_opportunity_cost_support"),
        "score_context": proposal.get("score_context"),
        "state_generation": proposal.get("state_generation"),
        "decision_preview": proposal.get("decision_preview"),
    }
    return (
        "You are Resource Arbiter, an isolated control-plane agent. "
        "Decide resource control using only the provided proposal. "
        "Use KILL_AND_REPLAN when evidence supports stopping this command and replanning. "
        "Use CONTINUE or DENY_KILL when there is recent progress or the command is near submission. "
        "Use OBSERVE_MORE when evidence is insufficient or the command is still in a fresh warmup window. "
        "execution_facts is task-agnostic; if task_device_mode=cpu_only, do not cite GPU intent, no GPU assigned, or assigned GPU idle as a problem. If intent_device_mismatch=true, declared GPU work is observed as CPU-only while assigned GPU is idle. Treat CPU busy, log churn, or artifact mtimes as activity facts, not automatic metric/submission progress proof. "
        "If progress_snapshot.resource_efficiency.review_required=true, perform the bounded efficiency review even when the process is active. Treat agent_reported progress or ETA as review input only; it cannot authorize a kill without corroborating system resource facts, no verified value progress, and the existing kill gate. "
        "Recent stdout, tqdm, CPU activity, or log growth proves liveness only; if progress_snapshot.finish_feasible=false under deadline/finalization pressure and no recoverable artifact or metric/submission exists, do not use recent progress alone as DENY_KILL support. "
        f"{resource_feedback_guidance_value('budget_deliverable_value_note')} "
        f"{resource_feedback_guidance_value('research_cadence_note')} "
        "If resource_budget_escalation.kind=final_resource_review, main-agent advisory has already been used or exhausted; decide the final resource-budget action without requiring another advisory. "
        "Use execution_facts as additional evidence for your decision; do not apply task-specific assumptions. "
        "For task_gpu_share_review, answer with action, not outcome: choose GRANT_SHARED_GPU_LEASE when share_decision_facts/share_observation show share_eligible=true, hard gates pass, GPU utilization is low, memory headroom covers the secondary estimate, and the secondary is revocable or otherwise policy-allowed. Choose DENY_SHARE_USE_CPU_SUPPORT when memory, CPU, or primary-progress facts make sharing risky. Choose CONTINUE_SHARED_OBSERVE only when required facts are missing or contradictory. "
        "For shared_runtime_review, answer with action, not outcome: choose CONTINUE_SHARED_OBSERVE, STOP_SECONDARY_SHARED_JOB, or REVOKE_SHARED_LEASE from observed primary impact facts. "
        "If progress_confidence=high, progress_signal=stalled, runtime is long, stdout/artifact/metric ages are all stale, and process_tree_cpu shows no active child work, treat repeated OBSERVE_MORE or DENY_KILL as exhausted evidence and prefer KILL_AND_REPLAN. "
        "A tiny old artifact, static warning, unchanged one-line log, or artifact update without metric/stdout signal is not metric/submission progress. "
        "If progress_snapshot.metric_history_text is present, compare the target-metric trajectory across observations; CPU/GPU activity and epoch advancement prove liveness, not route value. After a prior TIMEBOX or repeated OBSERVE_MORE, if the next comparable target metric still does not improve over the observed best and the remaining route is expensive, prefer KILL_AND_REPLAN over another open-ended observation. If metric direction or comparability is unclear, use a bounded TIMEBOX instead of inferring degradation. "
        "If progress_snapshot.phase_completion_protected=true, recent structured progress has a high-confidence phase ETA within five minutes; do not kill unless hard-safety or deadline evidence overrides it. "
        "Submission or deliverable availability is selection-layer state, not resource-kill evidence. "
        "For kill/value-review proposals, preferred output uses outcome values: KILL | NO_ACTION | TIMEBOX. "
        "For TIMEBOX, choose clear_on only: metric_update | artifact_growth | progress_advance | active_work_recovered | opportunity_cost_cleared; prefer progress_advance for prediction/inference commands that emit monotonic structured progress. Do not choose timebox_sec. "
        "Allowed canonical action values: CONTINUE | DENY_KILL | OBSERVE_MORE | MARK_STALLED_NO_KILL | KILL_AND_REPLAN | RELEASE_IDLE_LEASE | GRANT_SHARED_GPU_LEASE | DENY_SHARE_USE_CPU_SUPPORT | CONTINUE_SHARED_OBSERVE | STOP_SECONDARY_SHARED_JOB | REVOKE_SHARED_LEASE. These legacy action values are still accepted for non-value paths. "
        "Do not emit legacy APPROVE_KILL or APPROVE_KILL_STALLED; they are accepted only as normalizer aliases. "
        "KILL_AND_REPLAN requires arbiter confidence=high and hard-safety support, deterministic resource support, or captured owning-agent advisory support. "
        "Deterministic resource support includes repeated failed proof windows or quick-probe runtime overrun with stale stdout and no metric/artifact/submission progress. "
        "Multi-window low-progress and structured opportunity-cost evidence increase urgency, but they do not by themselves authorize discretionary kill. "
        "Return JSON only. For task_gpu_share_review and shared_runtime_review, use keys action, reason, confidence, ttl_sec or observe_more_sec. For kill/value-review proposals, prefer keys outcome, reason_code, reason, confidence, next_review_after_sec, clear_on.\n\n"
        f"PROPOSAL_JSON:\n{json.dumps(compact, ensure_ascii=True, sort_keys=True)}"
    )


def parse_arbiter_decision_text(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _proposal_process_liveness(proposal: dict[str, Any]) -> dict[str, Any]:
    for parent_key in ("progress_snapshot", "resource_snapshot", "blocker"):
        parent = proposal.get(parent_key) if isinstance(proposal.get(parent_key), dict) else {}
        live = parent.get("process_liveness") if isinstance(parent.get("process_liveness"), dict) else {}
        if live:
            return dict(live)
    return {}


def _apply_liveness_ttl_guard(
    *,
    action: str,
    reason: str,
    confidence: str,
    ttl_sec: float,
    observe_more_sec: float,
    proposal: dict[str, Any],
) -> tuple[str, str, float, float, dict[str, Any]]:
    liveness = _proposal_process_liveness(proposal)
    status = str(liveness.get("status") or "").strip().lower()
    guard: dict[str, Any] = {}
    reason_lower = str(reason or "").lower()
    completed_conflict = bool(
        status in {"alive", "inconsistent"}
        and ("completed" in reason_lower or "no active process" in reason_lower)
    )
    if status == "inconsistent" or completed_conflict:
        ttl_sec = min(float(ttl_sec or 0.0), 60.0)
        observe_more_sec = min(float(observe_more_sec or 0.0), 60.0)
        guard = {
            "process_liveness_status": status or "unknown",
            "force_process_rescan": True,
            "cooldown_break_allowed": True,
            "ttl_expiry_action": "reobserve_and_repropose",
            "completed_liveness_conflict": completed_conflict,
        }
        if completed_conflict:
            confidence = "low"
            reason = (
                str(reason or "process liveness conflict").rstrip(".")
                + "; process_liveness does not support completed/no-active-process"
            )
    return reason, confidence, max(1.0, ttl_sec), max(1.0, observe_more_sec), guard


def _canonical_review_decision(data: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    outcome = str(data.get("outcome") or "").strip().upper()
    if not outcome:
        return None
    original_outcome = outcome
    if outcome in TERMINATING_ACTIONS:
        outcome = "KILL"
    elif outcome in {"CONTINUE", "DENY_KILL", "OBSERVE_MORE", "MARK_STALLED_NO_KILL"}:
        outcome = "NO_ACTION"
    decision_id = str(data.get("decision_id") or f"rd_{uuid.uuid4().hex[:12]}")
    if outcome not in CANONICAL_REVIEW_OUTCOMES:
        return {
            "decision_id": decision_id,
            "action": "OBSERVE_MORE",
            "canonical_outcome": "NO_ACTION",
            "schema_violation": True,
            "schema_violation_reason": "unknown_outcome",
            "reason_code": "invalid_resource_review_schema",
            "reason": "Resource arbiter returned an unknown canonical outcome.",
            "confidence": "low",
            "ttl_sec": 60.0,
            "observe_more_sec": 60.0,
            "created_at": time.time(),
            "source": source,
            "original_action": original_outcome,
            "action_normalized": True,
        }
    reason = str(data.get("reason") or "").strip()
    reason_code = str(data.get("reason_code") or "").strip().lower()
    confidence = str(data.get("confidence") or "").strip().lower()
    if not reason or not reason_code or confidence not in VALID_REVIEW_CONFIDENCE:
        return {
            "decision_id": decision_id,
            "action": "OBSERVE_MORE",
            "canonical_outcome": "NO_ACTION",
            "schema_violation": True,
            "schema_violation_reason": "missing_reason_or_confidence",
            "reason_code": "invalid_resource_review_schema",
            "reason": "Resource arbiter response was missing required reason_code, reason, or confidence.",
            "confidence": "low",
            "ttl_sec": 60.0,
            "observe_more_sec": 60.0,
            "created_at": time.time(),
            "source": source,
            "original_action": outcome,
            "action_normalized": True,
        }
    if outcome == "KILL":
        action = "KILL_AND_REPLAN"
    else:
        action = "OBSERVE_MORE"
    out = {
        "decision_id": decision_id,
        "action": action,
        "canonical_outcome": outcome,
        "reason_code": reason_code,
        "reason": reason,
        "confidence": confidence,
        "ttl_sec": max(1.0, _float(data.get("next_review_after_sec"), 60.0)),
        "observe_more_sec": max(1.0, _float(data.get("next_review_after_sec"), 60.0)),
        "created_at": time.time(),
        "source": source,
        "original_action": original_outcome,
        "action_normalized": True,
        **({"outcome_alias_normalized": True} if original_outcome != outcome else {}),
    }
    if outcome == "TIMEBOX":
        clear_on = str(data.get("clear_on") or "").strip().lower()
        if clear_on not in VALID_TIMEBOX_CLEAR_ON:
            clear_on = "useful_progress"
            out.update({
                "schema_violation": True,
                "schema_violation_reason": "missing_or_unknown_clear_on",
                "reason_code": reason_code or "timebox_clear_on_defaulted",
                "reason": (
                    str(reason or "TIMEBOX response was missing a known clear_on signal.").rstrip(".")
                    + "; defaulted clear_on=useful_progress so the bounded proof window can close."
                ),
                "confidence": "low",
                "clear_on_defaulted": True,
            })
        out["clear_on"] = clear_on
        if "timebox_sec" in data:
            out["ignored_timebox_sec"] = data.get("timebox_sec")
    return out


def normalize_arbiter_decision(
    raw: dict[str, Any] | None,
    *,
    proposal: dict[str, Any],
    default_action: str = "OBSERVE_MORE",
    source: str = "resource_arbiter",
) -> dict[str, Any]:
    data = dict(raw or {})
    proposal_type = str(proposal.get("proposal_type") or "").strip().lower()
    action_text = str(data.get("action") or "").strip()
    canonical = None
    if not (proposal_type in {"task_gpu_share_review", "shared_runtime_review"} and action_text):
        canonical = _canonical_review_decision(data, source=source)
    if canonical is not None:
        canonical["proposal_id"] = str(proposal.get("proposal_id") or "")
        return canonical
    action, original_action = normalize_action_value(data.get("action") or default_action, proposal, default=default_action)
    try:
        ttl_sec = float(data.get("ttl_sec") or 900.0)
    except (TypeError, ValueError):
        ttl_sec = 900.0
    try:
        observe_more_sec = float(data.get("observe_more_sec") or 300.0)
    except (TypeError, ValueError):
        observe_more_sec = 300.0
    reason = str(data.get("reason") or _default_reason_for_action(action, proposal)).strip()
    confidence = str(data.get("confidence") or "medium").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    decision_id = str(data.get("decision_id") or f"rd_{uuid.uuid4().hex[:12]}")
    reason, confidence, ttl_sec, observe_more_sec, liveness_guard = _apply_liveness_ttl_guard(
        action=action,
        reason=reason,
        confidence=confidence,
        ttl_sec=ttl_sec,
        observe_more_sec=observe_more_sec,
        proposal=proposal,
    )
    return {
        "decision_id": decision_id,
        "proposal_id": str(proposal.get("proposal_id") or ""),
        "action": action,
        "reason": reason,
        "confidence": confidence,
        "ttl_sec": max(1.0, ttl_sec),
        "observe_more_sec": max(1.0, observe_more_sec),
        "created_at": time.time(),
        "source": source,
        "original_action": original_action,
        "action_normalized": action != original_action,
        **({"liveness_guard": liveness_guard} if liveness_guard else {}),
    }


def proposal_has_severe_stalled_no_work(
    proposal: dict[str, Any],
    *,
    min_runtime_sec: float = 900.0,
    stale_age_sec: float = 600.0,
    require_prior_review: bool = True,
) -> bool:
    """Return whether a proposal proves a long stall with no useful work.

    This is intentionally narrower than generic low progress. It avoids killing
    silent-but-busy batch work by requiring high-confidence stalled evidence,
    stale stdout/artifact/metric facts, and a process-tree snapshot with no busy
    children.
    """

    progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
    if bool(progress.get("near_submission")):
        return False
    if str(progress.get("progress_signal") or "").strip().lower() != "stalled":
        return False
    if str(progress.get("progress_confidence") or "").strip().lower() != "high":
        return False
    runtime = _float(progress.get("runtime_sec"), 0.0)
    if runtime < max(1.0, float(min_runtime_sec or 0.0)):
        return False

    stdout_age = _float(progress.get("stdout_last_line_age_sec", progress.get("stdout_age_sec")), runtime)
    artifact_age = _float(progress.get("artifact_last_update_age_sec"), runtime)
    metric_age = _float(progress.get("metric_last_update_age_sec"), runtime)
    stale_limit = max(1.0, float(stale_age_sec or 0.0))
    if stdout_age < stale_limit or artifact_age < stale_limit or metric_age < stale_limit:
        return False

    cpu = progress.get("process_tree_cpu") if isinstance(progress.get("process_tree_cpu"), dict) else {}
    if not cpu:
        return False
    busy_children = int(_float(cpu.get("busy_child_count"), 0.0))
    total_cpu = _float(cpu.get("total_cpu_pct"), 0.0)
    child_cpu = _float(cpu.get("child_cpu_pct"), 0.0)
    if busy_children > 0 or total_cpu >= 5.0 or child_cpu >= 5.0:
        return False

    if not require_prior_review:
        return True
    history = proposal.get("review_history") if isinstance(proposal.get("review_history"), dict) else {}
    if str(history.get("last_action") or "").upper() in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL", "DENY_KILL"}:
        return True
    if int(_float(history.get("observe_count"), 0.0)) > 0:
        return True
    return runtime >= 1200.0


def enforce_repeated_stall_escalation(decision: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    """Escalate repeated high-confidence no-work stalls out of observe loops."""

    out = dict(decision or {})
    action = str(out.get("action") or "OBSERVE_MORE").strip().upper()
    if action not in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL", "DENY_KILL"}:
        return out
    if not proposal_has_severe_stalled_no_work(proposal):
        return out
    progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
    runtime = _float(progress.get("runtime_sec"), 0.0)
    original = action
    out.update({
        "action": "KILL_AND_REPLAN",
        "confidence": "high",
        "reason": (
            str(out.get("reason") or "high-confidence stalled command kept consuming review windows").rstrip(".")
            + f"; escalated from {original} because runtime_sec={runtime:.0f} with stale stdout/artifact/metric and no active child work"
        ),
        "ttl_sec": min(_float(out.get("ttl_sec"), 900.0), 60.0),
        "observe_more_sec": min(_float(out.get("observe_more_sec"), 300.0), 60.0),
        "stall_escalation": {
            "from_action": original,
            "runtime_sec": runtime,
            "reason": "repeated_high_confidence_no_work_stall",
        },
    })
    return out


def fallback_policy_decision(proposal: dict[str, Any]) -> dict[str, Any]:
    proposal_type = str(proposal.get("proposal_type") or "").strip().lower()
    if proposal_type == "task_gpu_share_review":
        return {
            "action": "CONTINUE_SHARED_OBSERVE",
            "reason": "no valid arbiter grant was received, so task-local GPU sharing keeps observing instead of granting a secondary lease without LLM approval",
            "confidence": "medium",
            "ttl_sec": 600,
            "observe_more_sec": 300,
        }
    if proposal_type == "shared_runtime_review":
        return {
            "action": "CONTINUE_SHARED_OBSERVE",
            "reason": "shared runtime has no deterministic degradation signal",
            "confidence": "medium",
            "ttl_sec": 300,
            "observe_more_sec": 60,
        }
    if proposal_type == "quick_probe_review":
        if proposal_has_deterministic_kill_support(proposal):
            return {
                "action": "KILL_AND_REPLAN",
                "reason": "quick/probe exceeded its hard runtime envelope with no stdout, metric, submission, or useful artifact progress",
                "confidence": "high",
                "ttl_sec": 60,
                "observe_more_sec": 60,
            }
        return {
            "action": "OBSERVE_MORE",
            "reason": "quick/probe runtime exceeded its hard review window but still has progress or recoverability signals; policy fallback requests a short rescan",
            "confidence": "low",
            "ttl_sec": 60,
            "observe_more_sec": 60,
        }
    progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
    resource = proposal.get("resource_snapshot") if isinstance(proposal.get("resource_snapshot"), dict) else {}
    liveness = _proposal_process_liveness(proposal)
    if str(liveness.get("status") or "").lower() == "inconsistent":
        return {
            "action": "OBSERVE_MORE",
            "reason": "process liveness facts are inconsistent, so the process must be rescanned before a kill/deny decision is reused",
            "confidence": "low",
            "ttl_sec": 60,
            "observe_more_sec": 60,
        }
    confidence = str(progress.get("progress_confidence") or "low").lower()
    progress_signal = str(progress.get("progress_signal") or "").lower()
    near_submission = bool(progress.get("near_submission"))
    artifact_updates = progress.get("artifact_updates") or []
    stdout_age = _float(
        progress.get("stdout_last_line_age_sec", progress.get("stdout_age_sec")),
        0.0,
    )
    runtime = _float(progress.get("runtime_sec"), 0.0)
    artifact_age = _float(progress.get("artifact_last_update_age_sec"), runtime)
    artifact_recent = bool(artifact_updates) and artifact_age < 300.0
    artifact_recent_useful = bool(artifact_recent and not _artifact_updates_are_log_only(artifact_updates))
    pressure = str(resource.get("pressure") or resource.get("pressure_reason") or "").lower()
    reason_code = str(proposal.get("reason_code") or "")
    blocker = proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {}
    suspect = blocker.get("active_lease_suspect") if isinstance(blocker.get("active_lease_suspect"), dict) else {}
    deterministic_kill_support = proposal_has_deterministic_kill_support(proposal)
    advisory_kill_support = proposal_has_advisory_support(proposal)
    multi_window_progress_support = proposal_has_multi_window_progress_support(proposal)
    no_metric_submission_or_artifact = _proposal_has_no_metric_submission_or_useful_artifact(proposal)
    active_but_unaffordable = proposal_has_unaffordable_finish_without_output(proposal)
    structured_support = bool(
        proposal.get("structured_opportunity_cost_support")
        or ((proposal.get("review_history") if isinstance(proposal.get("review_history"), dict) else {}).get("structured_opportunity_cost_support"))
    )
    if advisory_kill_support and multi_window_progress_support and no_metric_submission_or_artifact and not near_submission:
        action = "KILL_AND_REPLAN"
        reason = resource_feedback_guidance_value("resource_execution_stop_reason")
        conf = "high"
    elif proposal_has_severe_stalled_no_work(proposal, require_prior_review=False):
        action = "OBSERVE_MORE"
        reason = "long-running command has high-confidence stalled evidence, but discretionary stop requires owning-agent advisory before execution"
        conf = "medium"
    elif bool(suspect.get("resource_suspect")):
        action = "OBSERVE_MORE"
        reason = "active lease is blocking a waiter with low-resource and poor-progress evidence; request advisory before discretionary stop"
        conf = "medium"
    elif bool(suspect.get("unknown_progress_suspect")) and structured_support:
        action = "OBSERVE_MORE"
        reason = "active lease has repeated unknown-progress windows while blocking queued work; request advisory before discretionary stop"
        conf = "medium"
    elif bool(suspect.get("unknown_progress_suspect")):
        action = "OBSERVE_MORE"
        reason = "active lease has unknown progress without useful output, but evidence must be collected before stopping"
        conf = "medium"
    elif bool(suspect.get("value_suspect")) and not bool(suspect.get("resource_suspect")):
        action = "OBSERVE_MORE"
        reason = "route value is suspect but resource-stall evidence is insufficient; lower priority or extend proof window instead of stopping"
        conf = "medium"
    elif "invalid_training_metrics" in reason_code:
        action = "OBSERVE_MORE"
        reason = "training metrics appear invalid or stuck; request advisory before discretionary stop/replan"
        conf = "medium"
    elif deterministic_kill_support:
        action = "KILL_AND_REPLAN"
        reason = "deterministic resource facts show long runtime without metric/submission progress; CPU/log activity is not useful progress"
        conf = "high"
    elif near_submission:
        action = "DENY_KILL"
        reason = "near submission or final scoring signal is present"
        conf = "high"
    elif "dataloader_bottleneck" in reason_code and not artifact_recent_useful:
        action = "OBSERVE_MORE"
        reason = "GPU compute is starved by input preprocessing while holding a lease; request advisory before stopping"
        conf = "medium"
    elif active_but_unaffordable:
        action = "OBSERVE_MORE"
        reason = "recent progress shows liveness, but ETA exceeds remaining useful budget and no recoverable metric/submission artifact exists"
        conf = "medium"
    elif (confidence in {"high", "medium"} and progress_signal not in {"stalled", "degraded"}) or artifact_recent_useful:
        action = "DENY_KILL"
        reason = "recent progress or artifact updates are still visible"
        conf = "high" if confidence == "high" else "medium"
    elif runtime < 900:
        action = "OBSERVE_MORE"
        reason = "runtime is still within the warmup window and evidence is incomplete"
        conf = "medium"
    elif stdout_age < 600 and "low_progress" not in reason_code:
        action = "OBSERVE_MORE"
        reason = "stdout is not stale enough to prove low value"
        conf = "medium"
    elif "red" in pressure or "under_pressure" in reason_code or "low_progress" in reason_code or "stalled" in reason_code or stdout_age >= 900:
        action = "OBSERVE_MORE"
        reason = "long-running expensive command has no recent progress evidence; request advisory before discretionary stop"
        conf = "medium"
    else:
        action = "OBSERVE_MORE"
        reason = "insufficient evidence to kill"
        conf = "low"
    return {
        "action": action,
        "reason": reason,
        "confidence": conf,
        "ttl_sec": 900,
        "observe_more_sec": 300,
    }


def main_agent_feedback(decision: dict[str, Any], proposal: dict[str, Any] | None = None) -> str:
    action = str(decision.get("action") or "").upper()
    reason = str(decision.get("reason") or "resource evidence").strip().rstrip(".")
    if action == "KILL_AND_REPLAN":
        message = f"Resource arbiter approved kill because {reason}."
    elif action == "STOP_BOUNDARY_VIOLATION":
        message = f"Resource arbiter stopped the command because {reason}."
    elif action in {"DENY_KILL", "CONTINUE"}:
        message = f"Resource arbiter denied kill because {reason}."
    elif action == "MARK_STALLED_NO_KILL":
        message = f"Resource arbiter marked the command stalled but requested another observation window because {reason}."
    elif action == "RELEASE_IDLE_LEASE":
        message = f"Resource arbiter recommended releasing an idle lease because {reason}."
    elif action == "GRANT_SHARED_GPU_LEASE":
        message = f"Resource arbiter granted a revocable shared GPU lease because {reason}."
    elif action == "DENY_SHARE_USE_CPU_SUPPORT":
        message = f"Resource arbiter denied shared GPU use and requested CPU support work because {reason}."
    elif action == "CONTINUE_SHARED_OBSERVE":
        message = f"Resource arbiter requested another shared-GPU observation window because {reason}."
    elif action == "STOP_SECONDARY_SHARED_JOB":
        message = f"Resource arbiter stopped the shared secondary job because {reason}."
    elif action == "REVOKE_SHARED_LEASE":
        message = f"Resource arbiter revoked the shared secondary lease because {reason}."
    else:
        message = f"Resource arbiter requested more observation because {reason}."
    return message + "\n" + resource_research_signal(decision, proposal=proposal)


def resource_research_signal(decision: dict[str, Any], proposal: dict[str, Any] | None = None) -> str:
    """Compact, fact-only resource signal for main-agent handoff."""
    action = str(decision.get("action") or "").upper()
    confidence = _signal_token(decision.get("confidence") or "medium").lower()
    fields = {
        "outcome": _signal_token(decision.get("execution_outcome") or action or "resource_review"),
        "confidence": confidence if confidence in {"low", "medium", "high"} else "medium",
    }
    for key, value in (
        ("review_facts", _review_history_tokens(proposal)),
        ("resource_facts", _resource_fact_tokens(proposal)),
        ("progress_facts", _progress_fact_tokens(proposal)),
        ("budget_facts", _budget_fact_tokens(proposal)),
        ("value_facts", _value_fact_tokens(proposal)),
        ("execution_focus", _execution_focus_token(action, proposal)),
    ):
        if value:
            fields[key] = value
    return "RESOURCE_RESEARCH_SIGNAL: " + "; ".join(f"{k}={v}" for k, v in fields.items()) + "."


def _review_history_tokens(proposal: dict[str, Any] | None) -> str:
    history = _nested_dict(proposal, "review_history")
    if not history:
        return ""
    tokens: list[str] = []
    observe_count = int(max(0.0, _float(history.get("observe_count"), 0.0)))
    unchanged_windows = int(max(0.0, _float(history.get("unchanged_bad_fact_windows"), 0.0)))
    if observe_count:
        tokens.append(f"prior_observe_count={observe_count}")
    if unchanged_windows:
        tokens.append(f"unchanged_windows={unchanged_windows}")
    last_action = _signal_token(history.get("last_action") or "")
    if last_action != "unknown":
        tokens.append(f"last_action={last_action}")
    for source_key, token in (
        ("repeated_observe_support", "repeated_observe=true"),
        ("structured_opportunity_cost_support", "opportunity_cost=true"),
        ("advisory_opportunity_support", "advisory_stop_support=true"),
    ):
        if bool(history.get(source_key)):
            tokens.append(token)
    advisory_preference = _signal_token(history.get("advisory_preference") or "")
    advisory_confidence = _signal_token(history.get("advisory_confidence") or "")
    if advisory_preference != "unknown":
        tokens.append(f"advisory={advisory_preference}")
    if advisory_confidence != "unknown":
        tokens.append(f"advisory_conf={advisory_confidence}")
    return ",".join(tokens[:8])


def _resource_fact_tokens(proposal: dict[str, Any] | None) -> str:
    facts = _nested_dict(proposal, "execution_facts")
    resource = _nested_dict(proposal, "resource_snapshot")
    tokens: list[str] = []
    raw_gpu_expected = facts.get("gpu_expected")
    gpu_expected = True if raw_gpu_expected is None else bool(raw_gpu_expected)
    device_mode = _signal_token(facts.get("task_device_mode") or "")
    if device_mode != "unknown":
        tokens.append(f"device_mode={device_mode}")
    if bool(facts.get("intent_device_mismatch")) and gpu_expected:
        tokens.append("intent_device_mismatch")
    declared = _signal_token(facts.get("declared_device") or facts.get("declared_resource_class") or "")
    observed = _signal_token(facts.get("observed_device") or resource.get("primary_resource_type") or "")
    if gpu_expected and (declared != "unknown" or observed != "unknown"):
        tokens.append(f"device={declared}_to_{observed}")
    if bool(facts.get("assigned_resource_idle")) and gpu_expected:
        tokens.append("assigned_resource_idle=true")
    if facts.get("assigned_gpu_max_util_pct") is not None:
        tokens.append(f"gpu_util={_number_token(facts.get('assigned_gpu_max_util_pct'))}")
    if facts.get("assigned_gpu_max_mem_mb") is not None:
        tokens.append(f"gpu_mem_mb={_number_token(facts.get('assigned_gpu_max_mem_mb'))}")
    if facts.get("process_cpu_pct") is not None:
        tokens.append(f"process_cpu_pct={_number_token(facts.get('process_cpu_pct'))}")
    if bool(facts.get("cpu_busy")):
        tokens.append("cpu_busy=true")
    return ",".join(tokens[:8])


def _progress_fact_tokens(proposal: dict[str, Any] | None) -> str:
    progress = _nested_dict(proposal, "progress_snapshot")
    facts = _nested_dict(proposal, "execution_facts")
    tokens: list[str] = []
    for key in ("progress_signal", "progress_confidence", "output_pattern", "stop_cost", "finish_feasible"):
        value = _fact_value_token(progress.get(key))
        if value != "unknown":
            tokens.append(f"{key}={value}")
    if facts.get("has_recent_metric_or_submission") is not None:
        tokens.append(f"recent_metric_or_submission={str(bool(facts.get('has_recent_metric_or_submission'))).lower()}")
    if facts.get("has_recent_useful_artifact") is not None:
        tokens.append(f"recent_useful_artifact={str(bool(facts.get('has_recent_useful_artifact'))).lower()}")
    if progress.get("runtime_sec") is not None:
        tokens.append(f"runtime_sec={_number_token(progress.get('runtime_sec'))}")
    return ",".join(tokens[:8])


def _budget_fact_tokens(proposal: dict[str, Any] | None) -> str:
    progress = _nested_dict(proposal, "progress_snapshot")
    budget = _nested_dict(proposal, "budget_context")
    tokens: list[str] = []
    cadence = _nested_dict(proposal, "research_cadence")
    for key in (
        "metric_budget_sec",
        "eta_to_next_comparable_metric_sec",
        "projected_metric_elapsed_sec",
    ):
        if cadence.get(key) is not None:
            tokens.append(f"{key}={_number_token(cadence.get(key))}")
    for key in (
        "eta_to_deliverable_sec",
        "eta_to_current_phase_end_sec",
        "remaining_useful_budget_sec",
        "deadline_remaining_sec",
        "finalization_reserve_sec",
    ):
        if progress.get(key) is not None:
            tokens.append(f"{key}={_number_token(progress.get(key))}")
    for key in ("eta_confidence", "deadline_event"):
        value = _fact_value_token(progress.get(key))
        if value != "unknown":
            tokens.append(f"{key}={value}")
    if budget.get("remaining_budget_sec") is not None:
        tokens.append(f"remaining_budget_sec={_number_token(budget.get('remaining_budget_sec'))}")
    return ",".join(tokens[:8])


def _value_fact_tokens(proposal: dict[str, Any] | None) -> str:
    cadence = _nested_dict(proposal, "research_cadence")
    score_context = _nested_dict(proposal, "score_context")
    tokens: list[str] = []
    for key in ("state", "reason_code", "execution_scale", "route_key"):
        value = _signal_token(cadence.get(key) or "")
        if value != "unknown":
            tokens.append(f"{key}={value}")
    if cadence:
        tokens.append(f"route_metric_proven={str(bool(cadence.get('route_metric_proven'))).lower()}")
    if not score_context:
        tokens.append("score_context=unavailable")
        return ",".join(tokens[:8])
    valid_best = score_context.get("valid_best_score") if isinstance(score_context.get("valid_best_score"), dict) else {}
    tokens.extend([
        f"record_count={_number_token(score_context.get('record_count') or 0)}",
        f"valid_record_count={_number_token(score_context.get('valid_record_count') or 0)}",
    ])
    if valid_best:
        if valid_best.get("value") is not None:
            tokens.append(f"valid_best={_number_token(valid_best.get('value'))}")
        validity = _signal_token(valid_best.get("validity") or "")
        if validity != "unknown":
            tokens.append(f"validity={validity}")
    if bool(score_context.get("capture_gap")):
        tokens.append("capture_gap=true")
    recommended = _signal_token(score_context.get("recommended_action") or "")
    if recommended != "unknown":
        tokens.append(f"score_action={recommended}")
    return ",".join(tokens[:8])


def _execution_focus_token(action: str, proposal: dict[str, Any] | None = None) -> str:
    if action not in {"KILL_AND_REPLAN", "STOP_BOUNDARY_VIOLATION", "OBSERVE_MORE", "MARK_STALLED_NO_KILL"}:
        return ""
    facts = _nested_dict(proposal, "execution_facts")
    if (
        facts.get("gpu_expected") is False
        and bool(facts.get("cpu_busy"))
        and bool(facts.get("activity_without_metric_submission_or_stdout"))
        and not bool(facts.get("has_recent_metric_or_submission"))
        and not bool(facts.get("has_recent_useful_artifact"))
    ):
        return "add_flushed_scienceflow_hb_or_metric_callback_then_resume"
    return _signal_token(
        resource_feedback_guidance_value(
            "execution_optimization_focus",
            "change_method_search_space_schedule_validation_target_or_stopping_condition",
        )
    )


def _nested_dict(parent: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(parent, dict):
        return {}
    value = parent.get(key)
    return value if isinstance(value, dict) else {}


def _number_token(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _signal_token(value)
    if number != number:
        return "unknown"
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")

def _signal_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip()).strip("_") or "unknown"


def _fact_value_token(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return _signal_token(value)


def _artifact_updates_are_log_only(updates: Any) -> bool:
    if not isinstance(updates, list) or not updates:
        return False
    return all(_artifact_update_is_log_only(item) for item in updates)


def _artifact_update_is_log_only(item: Any) -> bool:
    if isinstance(item, dict):
        if bool(item.get("recoverable_artifact_on_disk") or item.get("run_state_confirmed") or item.get("done_marker")):
            return False
        path = str(item.get("path") or item.get("name") or "").strip().lower()
    else:
        path = str(item or "").strip().lower()
    if not path:
        return False
    name = path.rsplit("/", 1)[-1]
    if name in {"submission.csv", "predictions.csv"}:
        return False
    if name.endswith((".pt", ".pth", ".ckpt", ".pkl", ".joblib", ".npy", ".npz", ".parquet", ".feather")):
        return False
    if name.endswith(".log"):
        return True
    if name.endswith(".txt") and any(token in name for token in ("log", "stdout", "stderr", "trace")):
        return True
    return False


def _proposal_has_no_metric_submission_or_useful_artifact(proposal: dict[str, Any]) -> bool:
    progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
    execution = proposal.get("execution_facts") if isinstance(proposal.get("execution_facts"), dict) else {}
    raw_updates = progress.get("artifact_updates")
    artifact_updates = raw_updates if isinstance(raw_updates, list) else ([raw_updates] if raw_updates else [])
    metric_count = int(_float(progress.get("metric_history_line_count"), 0.0))
    has_metric_or_submission = bool(
        progress.get("near_submission")
        or progress.get("submission_updated")
        or progress.get("submission_ready")
        or progress.get("saw_final_score")
        or metric_count > 0
        or execution.get("has_recent_metric_or_submission")
    )
    has_useful_artifact = bool(
        progress.get("recoverable_artifact_on_disk")
        or execution.get("has_recent_useful_artifact")
        or (artifact_updates and not _artifact_updates_are_log_only(artifact_updates))
    )
    return not (has_metric_or_submission or has_useful_artifact)


def _default_reason_for_action(action: str, proposal: dict[str, Any]) -> str:
    reason = str(proposal.get("reason_code") or "resource risk")
    if action == "KILL_AND_REPLAN":
        return f"{reason} is supported by low progress evidence"
    if action == "STOP_BOUNDARY_VIOLATION":
        return f"{reason} shows command placement outside the task resource boundary"
    if action == "DENY_KILL":
        return f"{reason} is not supported by enough evidence"
    if action == "GRANT_SHARED_GPU_LEASE":
        return f"{reason} supports revocable shared GPU work"
    if action == "DENY_SHARE_USE_CPU_SUPPORT":
        return f"{reason} does not support safe GPU sharing"
    if action == "CONTINUE_SHARED_OBSERVE":
        return f"{reason} needs another shared-GPU observation window"
    return f"{reason} needs another observation window"


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
