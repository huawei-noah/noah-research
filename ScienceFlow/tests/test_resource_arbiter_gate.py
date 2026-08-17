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

from scienceflow.solver.lnr.resource_runtime.review.arbiter_gate import enforce_arbiter_kill_gate


def test_arbiter_gate_treats_replan_advisory_as_stop_support() -> None:
    proposal = {
        "progress_snapshot": {
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 2,
        },
        "main_agent_advisory": {
            "preference": "replan",
            "confidence": "high",
            "advisory_status": "captured",
            "reason": "current command is likely stuck and should be replaced",
        },
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "high", "reason": "advisory says replan"},
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["advisory_support"] is True


def test_arbiter_gate_allows_internal_proof_window_override_without_advisory() -> None:
    proposal = {
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:low_progress_heartbeat_stalled",
        "progress_snapshot": {
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 1,
        },
    }

    gated = enforce_arbiter_kill_gate(
        {
            "action": "KILL_AND_REPLAN",
            "confidence": "high",
            "reason_code": "negotiated_timebox_missed_commitment",
            "reason": "missed negotiated proof window",
            "source": "proof_window_override",
        },
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["advisory_support"] is False
    assert gated["gate"]["deterministic_support"] is True
    assert gated["gate"]["kill_class"] == "deterministic_resource"


def test_arbiter_gate_allows_final_resource_review_without_advisory() -> None:
    proposal = {
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_timebox_expired",
        "resource_budget_escalation": {
            "kind": "final_resource_review",
            "failed_proof_window_count": 3,
            "max_proof_windows": 3,
        },
    }

    gated = enforce_arbiter_kill_gate(
        {
            "action": "KILL_AND_REPLAN",
            "confidence": "high",
            "reason_code": "final_resource_budget_review",
            "reason": "proof windows are exhausted",
            "source": "resource_arbiter_llm",
        },
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["advisory_support"] is False
    assert gated["gate"]["deterministic_support"] is True
    assert gated["gate"]["kill_class"] == "deterministic_resource"


def test_arbiter_gate_observes_near_complete_phase_instead_of_killing() -> None:
    proposal = {
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_timebox_expired",
        "progress_snapshot": {
            "progress_signal": "progressing",
            "progress_confidence": "high",
            "structured_progress_recent": True,
            "phase_completion_protected": True,
            "eta_confidence": "high",
            "eta_to_current_phase_end_sec": 28.0,
            "deadline_event": False,
        },
        "resource_budget_escalation": {
            "kind": "final_resource_review",
            "failed_proof_window_count": 3,
            "max_proof_windows": 3,
        },
    }

    gated = enforce_arbiter_kill_gate(
        {
            "action": "KILL_AND_REPLAN",
            "confidence": "high",
            "reason": "proof windows are exhausted",
            "source": "resource_arbiter_llm",
        },
        proposal,
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["observe_more_sec"] == 60.0
    assert gated["gate"]["allowed"] is False
    assert gated["gate"]["blocked_reason"] == "near_phase_completion"


def _live_best_metric_proposal() -> dict:
    return {
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_timebox_expired",
        "progress_snapshot": {
            "progress_signal": "active",
            "progress_confidence": "high",
            "structured_progress_recent": True,
            "eta_to_current_phase_end_sec": 425.6,
            "remaining_useful_budget_sec": 8364.7,
            "stop_cost": "high",
            "deadline_event": False,
        },
        "execution_facts": {
            "assigned_resource_idle": False,
        },
        "resource_metric_value": {
            "metric_name": "val_w_auc",
            "value": 0.544124,
            "best_value": 0.490066,
            "lower_is_better": False,
            "status": "above_observed_best",
            "useful": True,
        },
        "main_agent_advisory": {
            "preference": "safe_to_stop",
            "confidence": "high",
            "advisory_status": "captured",
        },
    }


def test_arbiter_gate_protects_active_live_metric_above_observed_best() -> None:
    gated = enforce_arbiter_kill_gate(
        {
            "action": "KILL_AND_REPLAN",
            "confidence": "high",
            "reason": "no useful progress for five windows",
        },
        _live_best_metric_proposal(),
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["observe_more_sec"] == 300.0
    assert gated["gate"]["allowed"] is False
    assert gated["gate"]["blocked_reason"] == "live_metric_above_observed_best"
    assert gated["gate"]["metric_value"] == 0.544124
    assert gated["gate"]["observed_best_value"] == 0.490066


def test_arbiter_gate_live_best_protection_does_not_override_hard_safety() -> None:
    proposal = _live_best_metric_proposal()
    proposal["kill_class"] = "hard_safety"
    proposal["reason_code"] = "out_of_memory"

    gated = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "high", "reason": "OOM"},
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["hard_safety_support"] is True


def test_arbiter_gate_live_best_protection_requires_feasible_phase_eta() -> None:
    proposal = _live_best_metric_proposal()
    proposal["progress_snapshot"]["eta_to_current_phase_end_sec"] = 9000.0

    gated = enforce_arbiter_kill_gate(
        {
            "action": "KILL_AND_REPLAN",
            "confidence": "high",
            "reason": "no useful progress for five windows",
        },
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["advisory_support"] is True


def test_arbiter_gate_live_best_protection_requires_recent_progress() -> None:
    proposal = _live_best_metric_proposal()
    proposal["progress_snapshot"]["structured_progress_recent"] = False

    gated = enforce_arbiter_kill_gate(
        {
            "action": "KILL_AND_REPLAN",
            "confidence": "high",
            "reason": "no useful progress for five windows",
        },
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True


def test_arbiter_gate_allows_structured_repeated_no_work_stall_without_advisory() -> None:
    proposal = {
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "progress_snapshot": {
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "runtime_sec": 30663.0,
        },
    }
    decision = {
        "action": "KILL_AND_REPLAN",
        "confidence": "high",
        "reason": "LLM budget exhausted; repeated no-work stall",
        "source": "llm_budget_guard",
        "stall_escalation": {
            "from_action": "OBSERVE_MORE",
            "runtime_sec": 30663.0,
            "reason": "repeated_high_confidence_no_work_stall",
        },
    }

    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["deterministic_support"] is True
    assert gated["gate"]["advisory_support"] is False


def test_arbiter_gate_does_not_trust_external_proof_window_reason_code() -> None:
    proposal = {
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:low_progress_heartbeat_stalled",
        "progress_snapshot": {
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 1,
        },
    }

    gated = enforce_arbiter_kill_gate(
        {
            "action": "KILL_AND_REPLAN",
            "confidence": "high",
            "reason_code": "negotiated_timebox_missed_commitment",
            "reason": "external arbiter tried to stop",
            "source": "resource_arbiter_llm",
        },
        proposal,
    )

    assert gated["action"] in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL"}
    assert gated["gate"]["allowed"] is False
    assert gated["gate"]["deterministic_support"] is False
    assert gated["gate"]["blocked_reason"] == "owning_agent_advisory_missing"


def _quick_probe_overrun_proposal(*, stdout_age_sec: float) -> dict:
    return {
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:low_progress_heartbeat_stalled",
        "progress_snapshot": {
            "runtime_sec": 1355.0,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 25,
            "multi_window_low_progress": True,
            "quick_probe": {
                "quick_probe_candidate": True,
                "quick_probe_expected_runtime_sec": 180.0,
                "quick_probe_hard_review_sec": 420.0,
            },
            "meaningful_stdout": True,
            "stdout_lines": 8,
            "stdout_bytes": 242,
            "stdout_last_line_age_sec": stdout_age_sec,
            "metric_history_line_count": 0,
            "artifact_updates": [],
            "near_submission": False,
            "recoverable_artifact_on_disk": False,
        },
        "execution_facts": {
            "intent_device_mismatch": True,
            "assigned_resource_idle": True,
            "cpu_busy": True,
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
    }


def test_arbiter_gate_allows_quick_probe_overrun_with_stale_stdout_without_advisory() -> None:
    proposal = _quick_probe_overrun_proposal(stdout_age_sec=900.0)

    gated = enforce_arbiter_kill_gate(
        {
            "action": "KILL_AND_REPLAN",
            "confidence": "high",
            "reason": "quick probe exceeded hard review with no useful output",
            "source": "resource_arbiter_llm",
        },
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["deterministic_support"] is True
    assert gated["gate"]["advisory_support"] is False
    assert gated["gate"]["kill_class"] == "deterministic_resource"


def test_arbiter_gate_keeps_recent_stdout_quick_probe_as_advisory_gated() -> None:
    proposal = _quick_probe_overrun_proposal(stdout_age_sec=30.0)

    gated = enforce_arbiter_kill_gate(
        {
            "action": "KILL_AND_REPLAN",
            "confidence": "high",
            "reason": "quick probe exceeded but stdout is still fresh",
            "source": "resource_arbiter_llm",
        },
        proposal,
    )

    assert gated["action"] in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL"}
    assert gated["gate"]["allowed"] is False
    assert gated["gate"]["deterministic_support"] is False
    assert gated["gate"]["blocked_reason"] == "owning_agent_advisory_missing"

def test_arbiter_gate_allows_quick_probe_overrun_with_only_stale_redirected_log() -> None:
    proposal = _quick_probe_overrun_proposal(stdout_age_sec=30.0)
    progress = proposal["progress_snapshot"]
    progress.update({
        "meaningful_stdout": False,
        "stdout_lines": 0,
        "stdout_bytes": 0,
        "stdout_observation": {
            "primary_channel": "redirected_log",
            "stdout_stream": {"present": False, "fresh": False},
            "redirected_log": {"present": True, "fresh": False, "age_sec": 1800.0, "path": "tmp/train_run.log"},
        },
        "artifact_updates": [{"path": "tmp/train_run.log", "age_sec": 1800.0, "artifact_log_like": True}],
        "artifact_last_update_age_sec": 1800.0,
    })

    gated = enforce_arbiter_kill_gate(
        {
            "action": "KILL_AND_REPLAN",
            "confidence": "high",
            "reason": "quick probe exceeded with only stale redirected log output",
            "source": "resource_arbiter_llm",
        },
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["deterministic_support"] is True


def test_arbiter_gate_rejects_stop_advisory_with_machine_fact_conflict() -> None:
    proposal = {
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_timebox_expired",
        "progress_snapshot": {
            "known_stage": "train",
            "recoverable_artifact_on_disk": False,
            "progress_signal": "stalled",
            "progress_confidence": "high",
        },
        "execution_facts": {"assigned_resource_idle": False},
        "resource_metric_value": {"status": "below_live_best", "phase": "train"},
        "resource_budget_escalation": {
            "kind": "final_resource_review",
            "failed_proof_window_count": 2,
            "max_proof_windows": 2,
        },
        "main_agent_advisory": {
            "preference": "safe_to_stop",
            "confidence": "high",
            "advisory_status": "captured",
            "observed_phase": "train",
            "observed_metric_status": "below_live_best",
            "observed_checkpoint": "no",
            "observed_resource_state": "idle",
        },
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "high", "reason": "stop route"},
        proposal,
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["gate"]["blocked_reason"] == "advisory_fact_conflict"
    assert gated["gate"]["advisory_fact_conflicts"] == [
        {"field": "resource_state", "claimed": "idle", "observed": "active"}
    ]


def test_arbiter_gate_hard_safety_ignores_advisory_fact_conflict() -> None:
    proposal = {
        "kill_class": "hard_safety",
        "reason_code": "out_of_memory",
        "progress_snapshot": {"known_stage": "train", "recoverable_artifact_on_disk": False},
        "execution_facts": {"assigned_resource_idle": False},
        "main_agent_advisory": {
            "preference": "safe_to_stop",
            "confidence": "high",
            "advisory_status": "captured",
            "observed_resource_state": "idle",
        },
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "high", "reason": "OOM"},
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["hard_safety_support"] is True
