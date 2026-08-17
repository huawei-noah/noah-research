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

import asyncio
from dataclasses import replace
import json
import time
from pathlib import Path

from scienceflow.core.tools.bash_tool import _is_gpu_visibility_probe
from scienceflow.solver.lnr.resource_advisory import build_inline_resource_advisory_prompt
from scienceflow.solver.lnr.resource_feedback_guidance import resource_feedback_guidance_value
from scienceflow.solver.lnr.resource_runtime.review.arbiter import (
    build_resource_arbiter_prompt,
    enforce_proposal_action_allowlist,
    enforce_repeated_stall_escalation,
    fallback_policy_decision,
    main_agent_feedback,
    normalize_arbiter_decision,
    proposal_has_severe_stalled_no_work,
)
from scienceflow.solver.lnr.resource_runtime.review.arbiter_gate import enforce_arbiter_kill_gate
from scienceflow.solver.lnr.resource_runtime.review.execution_facts import build_execution_facts
from scienceflow.solver.lnr.resource_runtime.review.progress import classify_progress_signal
from tests.lnr_resource_test_utils import make_observer
from scienceflow.solver.lnr.resource_observer import LHRResourceObserver


def _heavy_job(observer, tmp_path: Path) -> str:
    (tmp_path / "train.py").write_text("import torch\ntorch.cuda.is_available()\nprint(\'train\')\n", encoding="utf-8")
    job_id = observer.job_created(
        command="python3 train.py --device cuda --epochs 10",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=100.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    return job_id


def _resource_events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "resource" / "resource_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_resource_feedback_key_collapses_llm_reworded_pending_reasons() -> None:
    base = {
        "status": "PENDING",
        "scope": "per_gpu",
        "resource_mode": "YELLOW",
        "blocked_class": "heavy_gpu_train",
        "gpu_ids": ["0"],
        "allowed_classes": ["gpu_tt_light", "pure_tt_cpu"],
        "unlock_condition": "holder_released",
        "blocked_until_unlock": True,
    }
    key_a = LHRResourceObserver._resource_feedback_state_key(
        reason="GPU slot unavailable due to high utilization and resource class contention in YELLOW mode",
        **base,
    )
    key_b = LHRResourceObserver._resource_feedback_state_key(
        reason="GPU slot unavailable and lease not grantable by LLM; wait for availability",
        **base,
    )

    assert key_a == key_b
    assert "gpu_resource_wait" in key_a


def test_resource_arbiter_prompt_lists_allowed_action_values() -> None:
    prompt = build_resource_arbiter_prompt({"proposal_id": "p1"})

    assert "Allowed canonical action values:" in prompt
    assert "CONTINUE | DENY_KILL | OBSERVE_MORE | MARK_STALLED_NO_KILL | KILL_AND_REPLAN | RELEASE_IDLE_LEASE | GRANT_SHARED_GPU_LEASE | DENY_SHARE_USE_CPU_SUPPORT | CONTINUE_SHARED_OBSERVE | STOP_SECONDARY_SHARED_JOB | REVOKE_SHARED_LEASE" in prompt
    assert "finish_feasible=false" in prompt
    assert "liveness only" in prompt
    assert "Do not equate low-level liveness with research value" in prompt
    assert "root submission.csv" in prompt
    assert "STOP_AFTER_DELIVERABLE_AND_RELEASE" not in prompt


def test_resource_feedback_guidance_loads_from_markdown() -> None:
    assert resource_feedback_guidance_value("execution_optimization_focus") == "change_method_search_space_schedule_validation_target_or_stopping_condition"
    assert "method" in resource_feedback_guidance_value("advisory_stop_boundary_note")
    assert "search space" in resource_feedback_guidance_value("advisory_stop_boundary_note")
    assert "low-level liveness" in resource_feedback_guidance_value("budget_deliverable_value_note")
    assert "valid task deliverable" in resource_feedback_guidance_value("budget_deliverable_value_note")


def test_resource_advisory_prompt_separates_stop_from_route_abandonment() -> None:
    prompt = build_inline_resource_advisory_prompt({
        "proposal_id": "p-low-progress",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:low_progress_heartbeat_stalled",
    })

    assert "safe_to_stop means the current command can be interrupted" in prompt
    assert "does not mean abandon the route" in prompt
    assert "change the method" in prompt
    assert "search space" in prompt
    assert "Do not equate low-level liveness with research value" in prompt
    assert "next valid deliverable" in prompt


def test_resource_arbiter_prompt_tells_share_reviews_to_return_action() -> None:
    prompt = build_resource_arbiter_prompt({
        "proposal_id": "p-share",
        "proposal_type": "task_gpu_share_review",
        "share_decision_facts": {"share_eligible": True, "primary_gpu_util_p90_pct": 0.0},
    })

    assert "For task_gpu_share_review, answer with action, not outcome" in prompt
    assert "share_decision_facts" in prompt
    assert "GRANT_SHARED_GPU_LEASE" in prompt


def test_task_gpu_share_explicit_action_wins_over_canonical_outcome() -> None:
    proposal = {"proposal_id": "p-share", "proposal_type": "task_gpu_share_review"}

    decision = normalize_arbiter_decision(
        {
            "outcome": "NO_ACTION",
            "action": "GRANT_SHARED_GPU_LEASE",
            "reason": "low util and enough headroom",
            "confidence": "high",
        },
        proposal=proposal,
    )

    assert decision["action"] == "GRANT_SHARED_GPU_LEASE"
    assert decision["original_action"] == "GRANT_SHARED_GPU_LEASE"
    assert "canonical_outcome" not in decision


def test_main_agent_feedback_includes_fact_only_resource_signal() -> None:
    feedback = main_agent_feedback(
        {
            "action": "KILL_AND_REPLAN",
            "reason": "active_intervention:dataloader_bottleneck_low_gpu_high_cpu",
            "confidence": "high",
        },
        proposal={
            "execution_facts": {
                "intent_device_mismatch": True,
                "declared_device": "gpu",
                "observed_device": "cpu",
                "assigned_resource_idle": True,
                "assigned_gpu_max_util_pct": 0.0,
                "assigned_gpu_max_mem_mb": 0.0,
                "process_cpu_pct": 640.0,
                "cpu_busy": True,
                "has_recent_metric_or_submission": False,
                "has_recent_useful_artifact": False,
            },
            "progress_snapshot": {
                "progress_signal": "stalled",
                "progress_confidence": "high",
                "output_pattern": "unknown",
                "stop_cost": "high",
                "finish_feasible": False,
                "eta_confidence": "high",
                "eta_to_deliverable_sec": 3600.0,
                "remaining_useful_budget_sec": 1200.0,
                "runtime_sec": 900.0,
            },
            "budget_context": {
                "remaining_budget_sec": 2400.0,
            },
            "score_context": {
                "record_count": 3,
                "valid_record_count": 1,
                "valid_best_score": {"value": 0.44, "validity": "valid_comparable"},
            },
            "review_history": {
                "observe_count": 2,
                "last_action": "OBSERVE_MORE",
                "unchanged_bad_fact_windows": 2,
                "repeated_observe_support": True,
                "structured_opportunity_cost_support": True,
                "advisory_preference": "safe_to_stop",
                "advisory_confidence": "medium",
            },
        },
    )

    assert feedback.startswith("Resource arbiter approved kill because")
    assert "RESOURCE_RESEARCH_SIGNAL:" in feedback
    assert "outcome=KILL_AND_REPLAN" in feedback
    assert "review_facts=" in feedback
    assert "prior_observe_count=2" in feedback
    assert "unchanged_windows=2" in feedback
    assert "last_action=OBSERVE_MORE" in feedback
    assert "opportunity_cost=true" in feedback
    assert "resource_facts=" in feedback
    assert "intent_device_mismatch" in feedback
    assert "device=gpu_to_cpu" in feedback
    assert "progress_facts=" in feedback
    assert "progress_signal=stalled" in feedback
    assert "finish_feasible=false" in feedback
    assert "budget_facts=" in feedback
    assert "eta_to_deliverable_sec=3600" in feedback
    assert "remaining_useful_budget_sec=1200" in feedback
    assert "value_facts=" in feedback
    assert "valid_best=0.44" in feedback
    assert "execution_focus=change_method_search_space_schedule_validation_target_or_stopping_condition" in feedback
    assert "category=" not in feedback
    assert "decision_hint=" not in feedback
    assert "route_action=" not in feedback
    assert "DataLoader" not in feedback


def test_main_agent_feedback_cpu_only_omits_gpu_mismatch_tokens() -> None:
    feedback = main_agent_feedback(
        {
            "action": "KILL_AND_REPLAN",
            "reason": "active_intervention:sm_timebox_expired",
            "confidence": "high",
        },
        proposal={
            "execution_facts": {
                "task_device_mode": "cpu_only",
                "gpu_expected": False,
                "declared_device": "gpu",
                "observed_device": "cpu",
                "intent_device_mismatch": False,
                "assigned_resource_idle": False,
                "process_cpu_pct": 640.0,
                "cpu_busy": True,
                "activity_without_metric_submission_or_stdout": True,
                "has_recent_metric_or_submission": False,
                "has_recent_useful_artifact": False,
            },
            "progress_snapshot": {
                "progress_signal": "stalled",
                "progress_confidence": "high",
                "runtime_sec": 1800.0,
            },
        },
    )

    assert "device_mode=cpu_only" in feedback
    assert "intent_device_mismatch" not in feedback
    assert "device=gpu_to_cpu" not in feedback
    assert "assigned_resource_idle=true" not in feedback
    assert "progress_facts=" in feedback
    assert "progress_signal=stalled" in feedback
    assert "value_facts=" in feedback
    assert "score_context=unavailable" in feedback
    assert "execution_focus=add_flushed_scienceflow_hb_or_metric_callback_then_resume" in feedback
    assert "change_method_search_space_schedule_validation_target_or_stopping_condition" not in feedback


def test_execution_facts_detect_gpu_intent_running_cpu_only() -> None:
    facts = build_execution_facts(
        command="SCIENCEFLOW_RESOURCE_INTENT=gpu_tt python3 score.py",
        assigned_gpu_ids=["2"],
        gpu_expected=True,
        resource_snapshot={
            "resource_class_declared": "gpu_tt_light",
            "lease": {"active": True, "resource_ids": ["2"]},
            "resources": [
                {"resource_type": "gpu", "id": "2", "utilization_gpu_pct": 0.0, "memory_used_mb": 256.0},
                {"resource_type": "cpu", "id": "process", "process_tree_cpu": {"total_cpu_pct": 640.0}},
            ],
        },
        progress_snapshot={
            "runtime_sec": 1800.0,
            "artifact_updates": [{"path": "tmp/score.log"}],
            "artifact_last_update_age_sec": 10.0,
            "metric_history_line_count": 0,
            "meaningful_stdout": False,
            "near_submission": False,
            "process_tree_cpu": {"total_cpu_pct": 640.0},
        },
    )

    assert facts["declared_device"] == "gpu"
    assert facts["observed_device"] == "cpu"
    assert facts["intent_device_mismatch"] is True
    assert facts["activity_without_metric_submission_or_stdout"] is True


def test_execution_facts_cpu_only_task_suppresses_gpu_mismatch() -> None:
    facts = build_execution_facts(
        command="SCIENCEFLOW_RESOURCE_INTENT=gpu_tt python3 score.py",
        assigned_gpu_ids=["2"],
        gpu_expected=False,
        resource_snapshot={
            "resource_class_declared": "gpu_tt_light",
            "resources": [
                {"resource_type": "gpu", "id": "2", "utilization_gpu_pct": 0.0, "memory_used_mb": 256.0},
                {"resource_type": "cpu", "id": "process", "process_tree_cpu": {"total_cpu_pct": 640.0}},
            ],
        },
        progress_snapshot={
            "runtime_sec": 1800.0,
            "metric_history_line_count": 0,
            "meaningful_stdout": False,
            "near_submission": False,
            "process_tree_cpu": {"total_cpu_pct": 640.0},
        },
    )

    assert facts["task_device_mode"] == "cpu_only"
    assert facts["gpu_expected"] is False
    assert facts["declared_device"] == "gpu"
    assert facts["observed_device"] == "cpu"
    assert facts["intent_device_mismatch"] is False
    assert facts["assigned_resource_idle"] is False
    assert facts["activity_without_metric_submission_or_stdout"] is True


def test_observer_execution_facts_do_not_expect_gpu_for_cpu_job(tmp_path: Path) -> None:
    observer, _ = make_observer(tmp_path, gpu_pool=["0"])
    job_id = observer.job_created(
        command="SCIENCEFLOW_RESOURCE_INTENT=cpu_support python3 train_s13_ensemble.py",
        inferred_class="heavy_cpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800.0,
        workspace_dir=tmp_path,
    )

    assert job_id is not None
    job = observer._jobs[job_id]
    facts = observer._execution_facts_for_job(
        job,
        resource_snapshot={
            "resource_class_declared": "heavy_cpu_candidate",
            "resources": [
                {"resource_type": "gpu", "id": "0", "utilization_gpu_pct": 0.0, "memory_used_mb": 0.0},
                {"resource_type": "cpu", "id": "process", "process_tree_cpu": {"total_cpu_pct": 758.0}},
            ],
        },
        progress_snapshot={
            "runtime_sec": 786.0,
            "metric_history_line_count": 0,
            "meaningful_stdout": False,
            "near_submission": False,
            "process_tree_cpu": {"total_cpu_pct": 758.0},
        },
    )

    assert job.gpu_ids == ["0"]
    assert facts["gpu_expected"] is False
    assert facts["task_device_mode"] == "cpu_only"
    assert facts["assigned_resource_idle"] is False


def test_execution_facts_do_not_treat_zero_metric_age_without_metric_as_progress() -> None:
    facts = build_execution_facts(
        command="SCIENCEFLOW_RESOURCE_INTENT=gpu_train python3 train.py",
        assigned_gpu_ids=["6"],
        resource_snapshot={
            "resource_class_declared": "heavy_gpu_train",
            "lease": {"active": True, "resource_ids": ["6"]},
            "resources": [
                {"resource_type": "gpu", "id": "6", "utilization_gpu_pct": 0.0, "memory_used_mb": 1200.0},
                {"resource_type": "cpu", "id": "process", "process_tree_cpu": {"total_cpu_pct": 240.0}},
            ],
        },
        progress_snapshot={
            "runtime_sec": 2400.0,
            "artifact_updates": [{"path": "tmp/train_log.txt", "recoverable_artifact_on_disk": False}],
            "artifact_last_update_age_sec": 0.0,
            "metric_last_update_age_sec": 0.0,
            "metric_history_line_count": 0,
            "meaningful_stdout": False,
            "near_submission": False,
            "process_tree_cpu": {"total_cpu_pct": 240.0},
        },
    )

    assert facts["has_recent_metric_or_submission"] is False
    assert facts["artifact_updates_log_only"] is True
    assert facts["has_recent_useful_artifact"] is False
    assert facts["activity_without_metric_submission_or_stdout"] is True


def test_resource_arbiter_prompt_includes_execution_facts() -> None:
    prompt = build_resource_arbiter_prompt({
        "proposal_id": "p-exec",
        "execution_facts": {"intent_device_mismatch": True, "observed_device": "cpu"},
    })

    assert "execution_facts" in prompt
    assert "intent_device_mismatch" in prompt
    assert "CPU busy" in prompt


def test_resource_arbiter_prompt_includes_review_history_for_stall_decisions() -> None:
    prompt = build_resource_arbiter_prompt({
        "proposal_id": "p1",
        "review_history": {"last_action": "OBSERVE_MORE", "observe_count": 2},
        "budget_priority": {"level": "high"},
    })

    assert "review_history" in prompt
    assert "budget_priority" in prompt
    assert "repeated OBSERVE_MORE or DENY_KILL" in prompt


def test_arbiter_normalizes_legacy_kill_alias_to_canonical() -> None:
    proposal = {"proposal_id": "p1", "proposal_type": "kill_proposal", "reason_code": "stalled_stdout"}

    decision = normalize_arbiter_decision(
        {"action": "APPROVE_KILL_STALLED", "confidence": "high", "reason": "legacy"},
        proposal=proposal,
    )

    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["original_action"] == "APPROVE_KILL_STALLED"
    assert decision["action_normalized"] is True


def test_arbiter_downgrades_action_not_allowed_for_proposal_type() -> None:
    proposal = {
        "proposal_id": "p1",
        "proposal_type": "task_gpu_share_review",
        "reason_code": "task_gpu_share_review:wrong_action",
        "progress_snapshot": {
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 2,
            "deliverable_validity": "produced_valid",
        },
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "STOP_AFTER_DELIVERABLE_AND_RELEASE", "confidence": "high", "reason": "wrong proposal"},
        proposal,
    )
    gated = enforce_proposal_action_allowlist(gated, proposal)

    assert gated["action"] == "CONTINUE_SHARED_OBSERVE"
    assert gated["gate"]["blocked_reason"] == "proposal_action_not_allowed"
    assert gated["gate"]["fallback_action"] == "CONTINUE_SHARED_OBSERVE"


def test_kill_proposal_allows_idle_lease_release_action() -> None:
    proposal = {
        "proposal_id": "p-idle-release",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
    }

    gated = enforce_proposal_action_allowlist(
        {"action": "RELEASE_IDLE_LEASE", "confidence": "high", "reason": "idle gpu lease should be released"},
        proposal,
    )

    assert gated["action"] == "RELEASE_IDLE_LEASE"
    assert gated.get("gate", {}).get("blocked_reason") != "proposal_action_not_allowed"


def test_contention_review_emits_non_terminating_arbiter_proposal(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        assignment="env_only",
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_contention_review_enabled=True,
        arbiter_contention_min_runtime_sec=0.0,
        arbiter_contention_min_waiter_age_sec=0.0,
        arbiter_contention_min_interval_sec=1.0,
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    blocker = _heavy_job(observer, tmp_path)
    acquired = observer.queue_try_acquire(blocker, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert acquired["acquired"] is True
    (tmp_path / "train_waiter.py").write_text("import torch\ntorch.cuda.is_available()\nprint('waiter')\n", encoding="utf-8")
    waiter = observer.job_created(
        command="python3 train_waiter.py --device cuda --epochs 3",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=100.0,
        workspace_dir=tmp_path,
    )
    assert waiter is not None
    queued = observer.queue_try_acquire(waiter, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert queued["acquired"] is False

    decision = observer.active_intervention_decision(
        blocker,
        elapsed_sec=1200.0,
        stdout_age_sec=1.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
    )

    assert decision["arbiter_review"] is True
    assert decision["would_terminate"] is False
    proposal = decision["proposal"]
    assert proposal["proposal_type"] == "resource_contention_review"
    assert proposal["waiters"][0]["job_id"] == waiter
    assert "near_deliverable" not in proposal["blocker"]
    assert "deliverable_completion_state" not in proposal["blocker"]

    arbiter = asyncio.run(observer.arbiter_decide(blocker, proposal=proposal, decision_preview=decision))

    assert arbiter["enabled"] is True
    assert arbiter["terminate"] is False
    events = _resource_events(tmp_path)
    types = [event["event_type"] for event in events]
    assert "resource_review_proposal" in types
    assert "arbiter_input" in types
    assert "decision" in types


def test_main_agent_advisory_is_attached_as_evidence(tmp_path) -> None:
    async def advisory(_proposal):
        return {
            "preference": "safe_to_stop",
            "confidence": "high",
            "reason": "route is no longer useful",
            "advisory_mode": "inline_memory_edit",
            "memory_edit_applied": True,
            "_audit": {
                "event_version": 1,
                "advisory_mode": "inline_memory_edit",
                "status": "captured",
                "raw_response": "RESOURCE_ADVISORY_RESPONSE_BEGIN...",
            },
        }

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
    )
    proposal = {
        "proposal_id": "rp_advisory",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "periodic_efficiency:low_progress_review",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "unknown", "progress_confidence": "low"},
        "resource_snapshot": {},
    }

    arbiter = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))

    assert arbiter["enabled"] is True
    events = _resource_events(tmp_path)
    audit_event = next(event for event in events if event["event_type"] == "resource_advisory_audit")
    assert audit_event["payload"]["advisory_mode"] == "inline_memory_edit"
    assert audit_event["payload"]["job_id"] == "W00:bash:00001"
    advisory_event = next(event for event in events if event["event_type"] == "main_agent_advisory")
    assert advisory_event["payload"]["advisory"]["preference"] == "safe_to_stop"
    assert advisory_event["payload"]["advisory"]["memory_edit_applied"] is True
    decision_event = next(event for event in events if event["event_type"] == "decision")
    assert decision_event["payload"]["decision"]["proposal_id"] == "rp_advisory"


def test_state_machine_bad_route_kill_requires_main_agent_advisory(tmp_path) -> None:
    async def advisory(_proposal):
        return {
            "preference": "safe_to_stop",
            "confidence": "high",
            "reason": "training is CPU busy but not producing useful metric or artifact progress",
            "advisory_mode": "inline_memory_edit",
            "memory_edit_applied": False,
        }

    async def arbiter(_proposal):
        return {"action": "KILL_AND_REPLAN", "reason": "low-value route after review", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        kill_mode="auto",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter,
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=0.0,
        review_state_enabled=True,
        review_heartbeat_sec=60.0,
        review_warmup_windows=0,
        review_inactive_windows=5,
        review_value_windows=2,
    )
    job_id = _heavy_job(observer, tmp_path)

    first = observer.active_intervention_decision(
        job_id,
        elapsed_sec=60.0,
        stdout_age_sec=60.0,
        stdout_lines=0,
        stdout_bytes=0,
        process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
    )
    assert not first.get("would_terminate")

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=120.0,
        stdout_age_sec=120.0,
        stdout_lines=0,
        stdout_bytes=0,
        process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
    )
    assert decision["would_terminate"] is True
    assert decision["terminate"] is False
    assert decision["requires_llm_decision"] is True
    assert decision["resource_review_boundary"]["kind"] == "route_value"

    arbiter_decision = asyncio.run(observer.arbiter_decide(job_id, proposal=decision["proposal"], decision_preview=decision))

    assert arbiter_decision["action"] == "KILL_AND_REPLAN"
    assert arbiter_decision["raw_action"] == "KILL_AND_REPLAN"
    assert arbiter_decision["execution_outcome"] == "KILL"
    assert arbiter_decision["terminate"] is True
    kill_intent = arbiter_decision["decision"]["kill_intent_snapshot"]
    assert kill_intent["snapshot_id"]
    assert kill_intent["metric_scope_key"] == ""
    assert arbiter_decision["suppress_main_agent_feedback"] is False
    assert "approved kill" in arbiter_decision["feedback"]
    assert observer._review_states[job_id].job_state_bucket == "ACTIONED"
    events = _resource_events(tmp_path)
    event_types = [event.get("event_type") for event in events]
    assert "main_agent_advisory" in event_types
    outcome_event = next(event for event in events if event.get("event_type") == "resource_review_outcome")
    assert outcome_event["payload"]["action"] == "KILL"
    assert outcome_event["payload"]["execution_outcome"] == "KILL"
    assert outcome_event["payload"]["raw_action"] == "KILL_AND_REPLAN"
    assert kill_intent["kill_basis"] == "value_stagnation"
    pending_event = next(
        event
        for event in events
        if event.get("event_type") == "execution"
        and event.get("payload", {}).get("execution_status") == "pending"
    )
    proposal_id = str(pending_event.get("proposal_id") or "")
    resource_state = json.loads((tmp_path / "resource" / "resource_state.json").read_text(encoding="utf-8"))
    assert resource_state["active_proposals"][proposal_id]["execution_status"] == "pending"

    observer.resource_guard_action(
        job_id,
        action="terminate_by_arbiter",
        reason="approved_test_kill",
        elapsed_sec=121.0,
    )

    resource_state = json.loads((tmp_path / "resource" / "resource_state.json").read_text(encoding="utf-8"))
    assert resource_state.get("active_proposals") == {}
    executed_event = next(
        event
        for event in reversed(_resource_events(tmp_path))
        if event.get("event_type") == "execution"
        and event.get("payload", {}).get("execution_status") == "executed"
    )
    assert executed_event["payload"]["raw_action"] == "terminate_by_arbiter"

def test_arbiter_job_lookup_does_not_reuse_mismatched_active_proposal(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
    )
    assert observer.resource_runtime is not None
    observer.resource_runtime.update_active_resource_proposal(
        "rp_W00:bash:00001_share",
        {
            "proposal_id": "rp_W00:bash:00001_share",
            "proposal_type": "task_gpu_share_review",
            "reason_code": "task_gpu_share_review:eligible_secondary_waiter",
            "command_id": "W00:bash:00001",
            "decision_preview": {
                "job_id": "W00:bash:00001",
                "reason": "task_gpu_share_review:eligible_secondary_waiter",
            },
            "progress_snapshot": {"progress_signal": "active", "progress_confidence": "high"},
            "resource_snapshot": {},
            "share_observation": {"share_eligible": True},
        },
    )

    result = asyncio.run(
        observer.arbiter_decide(
            "W00:bash:00001",
            decision_preview={
                "job_id": "W00:bash:00001",
                "reason": "active_intervention:dataloader_bottleneck_low_gpu_high_cpu",
                "would_terminate": True,
            },
        )
    )

    assert result["enabled"] is False
    assert result["reason"] == "proposal_missing"


def test_sync_llm_arbiter_timeout_records_fallback_event(tmp_path) -> None:
    def slow(_proposal):
        time.sleep(1.2)
        return {"action": "KILL_AND_REPLAN", "reason": "too late", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=slow,
        arbiter_timeout_sec=0.05,
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
    )
    proposal = {
        "proposal_id": "rp_timeout",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "unknown", "progress_confidence": "low"},
        "resource_snapshot": {},
    }

    decision = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))

    assert decision["action"] == "OBSERVE_MORE"
    assert decision["decision"]["reason"] == "arbiter_llm_timeout"
    events = _resource_events(tmp_path)
    assert any(event["event_type"] == "resource_arbiter_timeout" for event in events)
    assert any(event["event_type"] == "decision" for event in events)


def test_llm_arbiter_budget_exhaustion_downgrades_to_observe_more(tmp_path) -> None:
    async def deny(_proposal):
        return {"action": "DENY_KILL", "reason": "still useful", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=deny,
        arbiter_job_llm_call_cap=1,
        arbiter_job_token_cap=100000,
    )
    proposal = {
        "proposal_id": "rp_budget",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
    }

    first = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))
    second = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))

    assert first["action"] == "DENY_KILL"
    assert second["action"] == "OBSERVE_MORE"
    assert "llm_budget_exhausted" in second["decision"]["reason"]
    assert any(event["event_type"] == "llm_budget_exhausted" for event in _resource_events(tmp_path))


def test_llm_budget_exhaustion_kills_repeated_high_confidence_no_work_stall(tmp_path) -> None:
    async def deny(_proposal):
        return {"action": "DENY_KILL", "reason": "still useful", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W01",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=deny,
        arbiter_job_llm_call_cap=1,
        arbiter_job_token_cap=100000,
    )
    job_id = "W01:bash:00151"
    observer._job_llm_call_count[job_id] = 1
    proposal = {
        "proposal_id": "rp_budget_stalled",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "command_id": job_id,
        "progress_snapshot": {
            "runtime_sec": 30663.0,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "stdout_last_line_age_sec": 30663.0,
            "artifact_last_update_age_sec": 30663.0,
            "metric_last_update_age_sec": 30663.0,
            "process_tree_cpu": {
                "available": True,
                "busy_child_count": 0,
                "total_cpu_pct": 0.0,
                "child_cpu_pct": 0.0,
            },
        },
        "execution_facts": {
            "assigned_resource_idle": True,
            "cpu_busy": False,
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
        "resource_snapshot": {},
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["action"] == "KILL_AND_REPLAN"
    assert result["terminate"] is True
    assert result["decision"]["gate"]["allowed"] is True
    assert result["decision"]["gate"]["deterministic_support"] is True
    assert "llm_budget_exhausted" in result["decision"]["reason"]


def test_stalled_guard_includes_cpu_fact_and_policy_kills_no_work(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=300.0,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        review_state_enabled=False,
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.stalled_guard_decision(
        job_id,
        elapsed_sec=1800.0,
        stdout_age_sec=1800.0,
        process_tree_cpu={"available": True, "busy_child_count": 0, "total_cpu_pct": 0.0, "child_cpu_pct": 0.0},
    )
    proposal = decision["proposal"]

    assert proposal["progress_snapshot"]["process_tree_cpu"]["busy_child_count"] == 0
    assert proposal["progress_snapshot"]["progress_signal"] == "stalled"

    arbiter = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview=decision))

    assert arbiter["action"] == "KILL_AND_REPLAN"
    assert arbiter["terminate"] is True
    assert arbiter["decision"]["gate"]["allowed"] is True
    assert arbiter["decision"]["gate"]["deterministic_support"] is True


def test_policy_arbiter_requests_advisory_for_low_progress_kill(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=0.1,
        low_progress_no_artifact_sec=0.1,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
    )
    job_id = _heavy_job(observer, tmp_path)
    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=1200.0,
        stdout_age_sec=1200.0,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="training",
    )
    assert decision["would_terminate"] is True
    assert decision["terminate"] is False
    assert decision["arbiter_enabled"] is True

    arbiter = asyncio.run(observer.arbiter_decide(job_id, proposal=decision["proposal"], decision_preview=decision))

    assert arbiter["action"] == "OBSERVE_MORE"
    assert arbiter["terminate"] is False
    assert "gate" not in arbiter["decision"]
    events = _resource_events(tmp_path)
    types = [event["event_type"] for event in events]
    assert "kill_proposal" in types
    assert "arbiter_input" in types
    assert "decision" in types
    assert "execution" in types
    execution = next(event for event in events if event["event_type"] == "execution")
    assert execution["payload"]["action"] == "NO_ACTION"
    assert execution["payload"]["execution_outcome"] == "NO_ACTION"
    assert execution["payload"]["raw_action"] == "OBSERVE_MORE"


def test_llm_arbiter_callback_can_deny_kill(tmp_path) -> None:
    async def deny(_proposal):
        return {"action": "DENY_KILL", "reason": "artifact chunks are still updating", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=0.1,
        low_progress_no_artifact_sec=0.1,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=deny,
    )
    job_id = _heavy_job(observer, tmp_path)
    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=1200.0,
        stdout_age_sec=1200.0,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="training",
    )

    arbiter = asyncio.run(observer.arbiter_decide(job_id, proposal=decision["proposal"], decision_preview=decision))

    assert arbiter["action"] == "DENY_KILL"
    assert arbiter["raw_action"] == "DENY_KILL"
    assert arbiter["execution_outcome"] == "NO_ACTION"
    assert arbiter["terminate"] is False
    assert arbiter["value_review_outcome"] == "NO_ACTION"
    assert arbiter["suppress_main_agent_feedback"] is True
    assert arbiter["feedback"] == ""


def test_cpu_sidecar_backfill_writes_review_report(tmp_path) -> None:
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        sidecar_enabled=True,
        sidecar_min_parent_runtime_sec=1.0,
    )
    job_id = _heavy_job(observer, tmp_path)

    result = observer.maybe_run_sidecar_backfill(job_id, elapsed_sec=5.0, parent_state="healthy_running")

    assert result["started"] is True
    report_path = tmp_path / "resource" / result["report"]["report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["quality_gate"] == "review"
    assert report["sanity_checks"]["sidecar_writes_isolated"] is True
    types = [event["event_type"] for event in _resource_events(tmp_path)]
    assert "sidecar_forked" in types
    assert "sidecar_joined" in types


def test_estra_magent_sidecar_writes_join_packet_and_resource_hint(tmp_path) -> None:
    from scienceflow.solver.lnr.estra_magent.join_gate import load_join_packets
    from scienceflow.solver.lnr.estra_magent.prompt_blocks import format_magent_recommendations

    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        estra_magent_enabled=True,
        estra_magent_sidecar_enabled=True,
        estra_magent_min_parent_runtime_sec=1.0,
    )
    job_id = _heavy_job(observer, tmp_path)

    result = observer.maybe_run_sidecar_backfill(job_id, elapsed_sec=5.0, parent_state="healthy_running")

    assert result["started"] is True
    join_packet = result["join_packet"]
    assert join_packet["quality_gate"] == "review"
    packet_path = tmp_path / "resource" / join_packet["packet_path"]
    assert packet_path.exists()
    packets = load_join_packets(tmp_path / "resource", parent_worker_id="W00")
    block = format_magent_recommendations(packets)
    assert "magent_recommendations:" in block
    assert "parent_summary:" in block
    assert "submission_checker" in block
    types = [event["event_type"] for event in _resource_events(tmp_path)]
    assert "magent_train_observed" in types
    assert "magent_fork_considered" in types
    assert "magent_sidecar_started" in types
    assert "magent_join_packet_ready" in types


def test_estra_magent_recommendations_do_not_fallback_to_estra_summary() -> None:
    from scienceflow.solver.lnr.estra_magent.prompt_blocks import format_magent_recommendations

    block = format_magent_recommendations([
        {
            "sidecar_id": "sc1",
            "quality_gate": "review",
            "summary_for_parent": "",
            "summary_for_estra": "route evidence only",
        }
    ])

    assert "route evidence only" not in block
    assert "parent_summary:" not in block


def test_estra_magent_recommendations_skip_whole_items_over_char_budget() -> None:
    from scienceflow.solver.lnr.estra_magent.prompt_blocks import format_magent_recommendations

    packets = [
        {
            "sidecar_id": "sc1",
            "quality_gate": "review",
            "summary_for_parent": "usable parent advice",
        },
        {
            "sidecar_id": "sc2",
            "quality_gate": "review",
            "summary_for_parent": "second item should not be partially rendered",
            "artifact_refs": [{"type": "report", "path": "reports/very_long_second_item.md"}],
        },
    ]

    block = format_magent_recommendations(packets, max_items=2, max_chars=110)

    assert "sc1" in block
    assert "usable parent advice" in block
    assert "sc2" not in block
    assert "second item" not in block
    assert not block.endswith("...")
    assert format_magent_recommendations(packets, max_items=2, max_chars=0) == ""


def test_checkpoint_to_submission_guard_records_gap(tmp_path) -> None:
    observer, _sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])
    job_id = _heavy_job(observer, tmp_path)
    gap = {
        "reason": "checkpoint_without_submission",
        "checkpoint_artifacts": [{"path": "model.pt", "size_bytes": 4}],
        "submission_updated": False,
    }

    feedback = observer.checkpoint_to_submission_guard(job_id, gap=gap, elapsed_sec=12.0)

    assert feedback.startswith("RESOURCE_FEEDBACK: checkpoint_without_submission because")
    event = next(event for event in _resource_events(tmp_path) if event["event_type"] == "checkpoint_to_submission_guard")
    assert event["payload"]["checkpoint_artifacts"][0]["path"] == "model.pt"


def test_checkpoint_guard_uses_configured_candidate_artifact_feedback(tmp_path) -> None:
    observer, _sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])
    job_id = _heavy_job(observer, tmp_path)
    gap = {
        "reason": "checkpoint_without_candidate_artifact",
        "checkpoint_artifacts": [{"path": "tmp/model.npy", "size_bytes": 4}],
        "artifact_path": "artifacts/best_solution.json",
        "artifact_updated": False,
    }

    feedback = observer.checkpoint_to_submission_guard(job_id, gap=gap, elapsed_sec=12.0)

    assert feedback.startswith("RESOURCE_FEEDBACK: checkpoint_without_candidate_artifact because")
    assert "artifacts/best_solution.json" in feedback
    assert "root submission.csv" not in feedback


def test_policy_fallback_uses_stdout_last_line_age_field() -> None:
    proposal = {
        "reason_code": "stalled_stdout",
        "severity": "red",
        "progress_snapshot": {
            "runtime_sec": 1200.0,
            "stdout_last_line_age_sec": 1200.0,
            "progress_confidence": "low",
            "artifact_updates": [],
            "near_submission": False,
        },
        "resource_snapshot": {},
    }

    decision = fallback_policy_decision(proposal)

    assert decision["action"] == "OBSERVE_MORE"
    assert "advisory" in decision["reason"]


def test_policy_fallback_kills_replan_advisory_with_stalled_no_output() -> None:
    proposal = {
        "proposal_id": "p-advisory-replan",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_route_value",
        "main_agent_advisory": {
            "preference": "replan",
            "confidence": "high",
            "advisory_status": "captured",
        },
        "progress_snapshot": {
            "runtime_sec": 3527.0,
            "stdout_last_line_age_sec": 3100.0,
            "stdout_lines": 16,
            "stdout_bytes": 575,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
            "multi_window_low_progress": True,
            "metric_history_line_count": 0,
            "artifact_updates": [],
            "near_submission": False,
            "recoverable_artifact_on_disk": False,
        },
        "execution_facts": {
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
        "resource_snapshot": {"pressure": "green"},
    }

    decision = fallback_policy_decision(proposal)
    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["confidence"] == "high"
    assert "main-agent advisory" in decision["reason"]
    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["advisory_support"] is True


def test_policy_fallback_keeps_replan_advisory_from_killing_with_progress_artifact() -> None:
    proposal = {
        "proposal_id": "p-advisory-artifact",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_route_value",
        "main_agent_advisory": {
            "preference": "replan",
            "confidence": "high",
            "advisory_status": "captured",
        },
        "progress_snapshot": {
            "runtime_sec": 1800.0,
            "stdout_last_line_age_sec": 1200.0,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
            "multi_window_low_progress": True,
            "metric_history_line_count": 0,
            "artifact_updates": [{"path": "artifacts/best_model.pt", "recoverable_artifact_on_disk": True}],
            "near_submission": False,
            "recoverable_artifact_on_disk": True,
        },
        "execution_facts": {
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": True,
        },
        "resource_snapshot": {"pressure": "green"},
    }

    decision = fallback_policy_decision(proposal)

    assert decision["action"] != "KILL_AND_REPLAN"


def test_policy_fallback_keeps_replan_advisory_from_killing_near_submission() -> None:
    proposal = {
        "proposal_id": "p-advisory-near-submission",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_route_value",
        "main_agent_advisory": {
            "preference": "replan",
            "confidence": "high",
            "advisory_status": "captured",
        },
        "progress_snapshot": {
            "runtime_sec": 1800.0,
            "stdout_last_line_age_sec": 1200.0,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
            "multi_window_low_progress": True,
            "metric_history_line_count": 0,
            "artifact_updates": [],
            "near_submission": True,
            "recoverable_artifact_on_disk": False,
        },
        "execution_facts": {
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
        "resource_snapshot": {"pressure": "green"},
    }

    decision = fallback_policy_decision(proposal)

    assert decision["action"] == "DENY_KILL"


def test_gpu_visibility_probe_allows_only_low_cost_cuda_checks() -> None:
    assert _is_gpu_visibility_probe('python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"')
    assert _is_gpu_visibility_probe("nvidia-smi --query-gpu=index --format=csv")
    assert not _is_gpu_visibility_probe("python3 train.py --epochs 10")
    assert not _is_gpu_visibility_probe('python3 -c "import torch; model.train(); loss.backward(); optimizer.step()"')


def test_policy_fallback_approves_dataloader_bottleneck_without_artifact_progress() -> None:
    proposal = {
        "reason_code": "active_intervention:dataloader_bottleneck_low_gpu_high_cpu",
        "severity": "red",
        "progress_snapshot": {
            "runtime_sec": 1200.0,
            "stdout_last_line_age_sec": 20.0,
            "progress_confidence": "medium",
            "artifact_updates": [],
            "near_submission": False,
        },
        "resource_snapshot": {},
    }

    decision = fallback_policy_decision(proposal)

    assert decision["action"] == "OBSERVE_MORE"
    assert "input preprocessing" in decision["reason"]
    assert "advisory" in decision["reason"]


def test_policy_arbiter_ignores_completed_deliverable_as_kill_reason() -> None:
    decision = fallback_policy_decision(
        {
            "reason_code": "active_intervention:deliverable_complete_resource_hold",
            "progress_snapshot": {"progress_confidence": "high", "near_submission": True, "runtime_sec": 3600},
            "resource_snapshot": {"pressure": "green"},
        }
    )

    assert decision["action"] == "DENY_KILL"
    assert "near submission" in decision["reason"]


def test_policy_arbiter_approves_invalid_training_metrics() -> None:
    decision = fallback_policy_decision(
        {
            "reason_code": "active_intervention:invalid_training_metrics_nan_inf",
            "progress_snapshot": {"progress_confidence": "high", "artifact_updates": ["checkpoint.pt"], "runtime_sec": 1200},
            "resource_snapshot": {"pressure": "green"},
        }
    )

    assert decision["action"] == "OBSERVE_MORE"
    assert "metrics appear invalid" in decision["reason"]
    assert "advisory" in decision["reason"]


def test_policy_fallback_kills_late_log_only_gpu_idle_cpu_busy_work() -> None:
    proposal = {
        "proposal_id": "p-log-only",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_timebox_expired",
        "progress_snapshot": {
            "runtime_sec": 2400.0,
            "progress_signal": "active",
            "progress_confidence": "medium",
            "artifact_updates": [{"path": "tmp/train_log.txt", "recoverable_artifact_on_disk": False}],
            "artifact_last_update_age_sec": 0.0,
            "metric_history_line_count": 0,
            "meaningful_stdout": False,
            "near_submission": False,
        },
        "execution_facts": {
            "intent_device_mismatch": True,
            "assigned_resource_idle": True,
            "cpu_busy": True,
            "activity_without_metric_submission_or_stdout": True,
            "has_recent_metric_or_submission": False,
        },
        "resource_snapshot": {"pressure": "green"},
    }

    decision = fallback_policy_decision(proposal)
    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert decision["action"] == "KILL_AND_REPLAN"
    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["deterministic_support"] is True
    assert gated["gate"]["kill_class"] == "deterministic_resource"


def test_progress_signal_marks_multi_window_stalled() -> None:
    state = classify_progress_signal(
        {
            "elapsed_sec": 2400.0,
            "stdout_age_sec": 1300.0,
            "stdout_lines": 0,
            "saw_training_progress": False,
            "saw_final_score": False,
        },
        progress_age_sec=2400.0,
        artifact_age_sec=2400.0,
        low_progress_warmup_sec=600.0,
        stalled_stdout_sec=600.0,
        no_progress_sec=600.0,
        no_artifact_sec=600.0,
        min_confidence_windows=2,
    )

    assert state["progress_signal"] == "stalled"
    assert state["progress_confidence"] == "high"
    assert state["multi_window_low_progress"] is True


def test_progress_signal_does_not_treat_unadvanced_structured_heartbeat_as_active() -> None:
    state = classify_progress_signal(
        {
            "elapsed_sec": 240.0,
            "stdout_age_sec": 10.0,
            "stdout_lines": 13,
            "saw_training_progress": True,
            "structured_progress": {"current": 0.0, "total": 103.0, "advanced": False},
            "structured_progress_advanced": False,
        },
        progress_age_sec=10.0,
        artifact_age_sec=240.0,
        low_progress_warmup_sec=0.0,
        stalled_stdout_sec=600.0,
        no_progress_sec=600.0,
        no_artifact_sec=600.0,
        idle_samples=3,
        min_confidence_windows=1,
    )

    assert state["progress_signal"] == "degraded"
    assert "recent_progress_heartbeat" not in state["progress_signal_reason"]
    assert "idle_gpu_samples" in state["progress_signal_reason"]


def test_progress_signal_keeps_advanced_structured_heartbeat_active() -> None:
    state = classify_progress_signal(
        {
            "elapsed_sec": 240.0,
            "stdout_age_sec": 10.0,
            "stdout_lines": 13,
            "saw_training_progress": True,
            "structured_progress": {"current": 25.0, "total": 103.0, "advanced": True},
            "structured_progress_advanced": True,
        },
        progress_age_sec=10.0,
        artifact_age_sec=240.0,
        low_progress_warmup_sec=0.0,
        stalled_stdout_sec=600.0,
        no_progress_sec=600.0,
        no_artifact_sec=600.0,
        idle_samples=3,
        min_confidence_windows=1,
    )

    assert state["progress_signal"] == "active"
    assert "recent_progress_heartbeat" in state["progress_signal_reason"]


def _route_value_review_payload(no_useful_windows: int) -> tuple[dict, dict]:
    boundary = {"kind": "route_value", "reason": "no_useful_progress_windows>=2"}
    review_state = {
        "job_state_bucket": "RUNNING_NO_PROGRESS",
        "no_useful_progress_windows": no_useful_windows,
        "blocked_worker_count": 0,
        "active_waiter_pressure": False,
    }
    signal = {
        "elapsed_sec": 300.0,
        "stdout_lines": 5,
        "stdout_bytes": 200,
        "process_tree_cpu": {"total_cpu_pct": 600.0, "busy_child_count": 4},
        "resource_review_boundary": boundary,
        "resource_review_state": review_state,
        "metric_history_text": "",
        "metric_history_line_count": 0,
    }
    decision = {
        "enabled": True,
        "terminate": False,
        "would_terminate": True,
        "requires_llm_decision": True,
        "arbiter_enabled": True,
        "reason": "active_intervention:sm_route_value",
        "resource_review_boundary": boundary,
        "resource_review_state": review_state,
    }
    return decision, signal


def test_active_resource_proposals_coalesce_same_command_type_reason(tmp_path: Path) -> None:
    observer, _ = make_observer(tmp_path, gpu_pool=["0"])
    runtime = observer.resource_runtime
    assert runtime is not None

    first = {
        "proposal_id": "rp_old",
        "proposal_type": "kill_proposal",
        "command_id": "W00:bash:00034",
        "reason_code": "active_intervention:sm_timebox_expired",
    }
    second = {
        "proposal_id": "rp_new",
        "proposal_type": "kill_proposal",
        "command_id": "W00:bash:00034",
        "reason_code": "active_intervention:sm_timebox_expired",
        "boundary_state": {"timebox_windows": 7},
    }

    runtime.update_active_resource_proposal("rp_old", first)
    runtime.update_active_resource_proposal("rp_new", second)

    state = json.loads((tmp_path / "resource" / "resource_state.json").read_text(encoding="utf-8"))
    active = state.get("active_proposals") or {}
    assert list(active) == ["rp_new"]
    assert active["rp_new"]["boundary_state"]["timebox_windows"] == 7

def test_kill_proposal_cooldown_suppresses_unchanged_boundary(tmp_path: Path) -> None:
    observer, _ = make_observer(
        tmp_path,
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    observer.kill_proposal_cooldown_sec = 600.0
    job_id = _heavy_job(observer, tmp_path)
    job = observer._jobs[job_id]

    first_decision, first_signal = _route_value_review_payload(2)
    second_decision, second_signal = _route_value_review_payload(2)

    first = observer._record_kill_proposal_event(job, first_decision, first_signal, source="resource_review_state")
    second = observer._record_kill_proposal_event(job, second_decision, second_signal, source="resource_review_state")

    assert first is not None
    assert second is None
    events = _resource_events(tmp_path)
    assert sum(1 for event in events if event.get("event_type") == "kill_proposal") == 1
    suppressed = [event for event in events if event.get("event_type") == "suppressed_proposal"]
    assert suppressed
    assert suppressed[-1]["payload"]["suppressed_reason"] == "cooldown_same_unchanged_boundary"


def test_kill_proposal_carries_resource_metric_value_at_top_level(tmp_path: Path) -> None:
    observer, _ = make_observer(
        tmp_path,
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)
    job = observer._jobs[job_id]
    decision, signal = _route_value_review_payload(2)
    signal["resource_metric_value"] = {
        "metric_name": "val_w_auc",
        "value": 0.544124,
        "best_value": 0.490066,
        "status": "above_observed_best",
        "useful": True,
    }

    proposal = observer._record_kill_proposal_event(
        job,
        decision,
        signal,
        source="resource_review_state",
    )

    assert proposal is not None
    assert proposal["resource_metric_value"] == signal["resource_metric_value"]


def test_kill_proposal_cooldown_allows_no_progress_escalation(tmp_path: Path) -> None:
    observer, _ = make_observer(
        tmp_path,
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_proposal_coalesce_window_sec=0.0,
        review_value_windows=2,
    )
    observer.kill_proposal_cooldown_sec = 600.0
    job_id = _heavy_job(observer, tmp_path)
    job = observer._jobs[job_id]

    first_decision, first_signal = _route_value_review_payload(2)
    second_decision, second_signal = _route_value_review_payload(3)

    first = observer._record_kill_proposal_event(job, first_decision, first_signal, source="resource_review_state")
    second = observer._record_kill_proposal_event(job, second_decision, second_signal, source="resource_review_state")

    assert first is not None
    assert second is not None
    assert second["cooldown_bypassed_by_escalation"] is True
    assert second["boundary_state"]["no_useful_progress_windows"] == 3
    events = _resource_events(tmp_path)
    proposals = [event for event in events if event.get("event_type") == "kill_proposal"]
    assert len(proposals) == 2
    assert proposals[-1]["payload"]["cooldown_bypassed_by_escalation"] is True


def test_kill_proposal_cooldown_allows_repeated_timebox_expired(tmp_path: Path) -> None:
    observer, _ = make_observer(
        tmp_path,
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    observer.kill_proposal_cooldown_sec = 600.0
    job_id = _heavy_job(observer, tmp_path)
    job = observer._jobs[job_id]
    boundary = {"kind": "timebox_expired", "reason": "timebox_expired"}
    review_state = {
        "job_state_bucket": "TIMEBOX_ACTIVE",
        "no_useful_progress_windows": 0,
        "timebox_id": "tb_test",
        "timebox_windows": 2,
        "blocked_worker_count": 0,
        "active_waiter_pressure": False,
    }
    signal = {
        "elapsed_sec": 360.0,
        "stdout_lines": 5,
        "stdout_bytes": 200,
        "process_tree_cpu": {"total_cpu_pct": 600.0, "busy_child_count": 4},
        "resource_review_boundary": boundary,
        "resource_review_state": review_state,
        "metric_history_text": "",
        "metric_history_line_count": 0,
    }
    decision = {
        "enabled": True,
        "terminate": False,
        "would_terminate": True,
        "requires_llm_decision": True,
        "arbiter_enabled": True,
        "reason": "active_intervention:sm_timebox_expired",
        "resource_review_boundary": boundary,
        "resource_review_state": review_state,
    }

    first = observer._record_kill_proposal_event(job, dict(decision), dict(signal), source="resource_review_state")
    second = observer._record_kill_proposal_event(job, dict(decision), dict(signal), source="resource_review_state")

    assert first is not None
    assert second is not None
    assert second["cooldown_bypassed_by_escalation"] is True
    events = _resource_events(tmp_path)
    proposals = [event for event in events if event.get("event_type") == "kill_proposal"]
    assert len(proposals) == 2
    assert not [event for event in events if event.get("event_type") == "suppressed_proposal"]


def test_value_review_observe_more_suppresses_main_agent_feedback(tmp_path: Path) -> None:
    observer, _ = make_observer(
        tmp_path,
        min_register_sec=0.0,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_proposal_coalesce_window_sec=0.0,
        review_warmup_windows=0,
    )
    job_id = _heavy_job(observer, tmp_path)
    proposal = {
        "proposal_id": "rp_suppress_no_action",
        "proposal_type": "kill_proposal",
        "command_id": job_id,
        "reason_code": "active_intervention:sm_progress_window",
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
        "progress_snapshot": {
            "progress_signal": "active",
            "progress_confidence": "medium",
            "progress_signal_windows": 1,
        },
        "decision_preview": {"job_id": job_id, "reason": "active_intervention:sm_progress_window"},
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview=proposal["decision_preview"]))

    assert result["enabled"] is True
    assert result["action"] in {"OBSERVE_MORE", "DENY_KILL", "CONTINUE", "MARK_STALLED_NO_KILL"}
    assert result["raw_action"] == result["action"]
    assert result["execution_outcome"] == "NO_ACTION"
    assert result["value_review_outcome"] == "NO_ACTION"
    assert result["suppress_main_agent_feedback"] is True
    assert result["feedback"] == ""
    events = _resource_events(tmp_path)
    outcome_event = next(e for e in events if e.get("event_type") == "resource_review_outcome")
    assert outcome_event["payload"]["action"] == "NO_ACTION"
    assert outcome_event["payload"]["execution_outcome"] == "NO_ACTION"
    assert outcome_event["payload"]["raw_action"] == result["action"]
    assert any(e.get("event_type") == "decision" for e in events)

def test_review_history_escalates_repeated_safe_to_stop_advisory_to_structured_support(tmp_path: Path) -> None:
    observer, _ = make_observer(tmp_path)
    proposal = {
        "command_id": "W00:bash:00123",
        "proposal_id": "rp1",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "progress_snapshot": {
            "progress_signal": "unknown",
            "progress_confidence": "low",
            "deadline_event": False,
        },
        "main_agent_advisory": {"preference": "safe_to_stop", "confidence": "low"},
    }

    first = observer._attach_review_history(proposal, now=1.0)
    observer._note_review_decision_history(first, {"action": "OBSERVE_MORE"})
    second = observer._attach_review_history(proposal, now=2.0)
    observer._note_review_decision_history(second, {"action": "OBSERVE_MORE"})
    third = observer._attach_review_history(proposal, now=3.0)

    history = third["review_history"]
    assert history["repeated_observe_support"] is True
    assert history["advisory_opportunity_support"] is True
    assert history["structured_opportunity_cost_support"] is True
    assert third["structured_opportunity_cost_support"] is True


def test_arbiter_gate_downgrades_low_confidence_kill_even_with_advisory() -> None:
    proposal = {
        "progress_snapshot": {
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
        },
        "main_agent_advisory": {"preference": "safe_to_stop", "confidence": "high"},
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "medium", "reason": "advisory agrees"},
        proposal,
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["gate"]["blocked_reason"] == "arbiter_confidence_not_high"


def test_arbiter_gate_requires_advisory_for_discretionary_kill() -> None:
    proposal = {
        "progress_snapshot": {
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 2,
        }
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "high", "reason": "stalled"},
        proposal,
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["gate"]["allowed"] is False
    assert gated["gate"]["blocked_reason"] == "owning_agent_advisory_missing"
    assert gated["gate"]["multi_window_progress"] is True

    with_advisory = dict(proposal)
    with_advisory["main_agent_advisory"] = {
        "preference": "safe_to_stop",
        "confidence": "high",
        "advisory_status": "captured",
    }
    allowed = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "high", "reason": "advisory agrees"},
        with_advisory,
    )
    assert allowed["action"] == "KILL_AND_REPLAN"
    assert allowed["gate"]["allowed"] is True
    assert allowed["gate"]["advisory_support"] is True


def test_arbiter_gate_allows_hard_safety_without_advisory() -> None:
    proposal = {
        "kill_class": "hard_safety",
        "reason_code": "gpu_boundary_violation",
        "progress_snapshot": {"progress_signal": "stalled", "progress_confidence": "high"},
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "high", "reason": "GPU boundary violation"},
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["kill_class"] == "hard_safety"


def test_repeated_high_confidence_stall_escalates_observe_to_kill() -> None:
    proposal = {
        "proposal_id": "rp_stall",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "progress_snapshot": {
            "runtime_sec": 1800.0,
            "stdout_last_line_age_sec": 1800.0,
            "artifact_last_update_age_sec": 1800.0,
            "metric_last_update_age_sec": 1800.0,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
            "multi_window_low_progress": True,
            "process_tree_cpu": {"available": True, "busy_child_count": 0, "total_cpu_pct": 0.0, "child_cpu_pct": 0.0},
        },
        "review_history": {"last_action": "OBSERVE_MORE", "observe_count": 2},
    }

    assert proposal_has_severe_stalled_no_work(proposal) is True
    decision = enforce_repeated_stall_escalation(
        {"action": "OBSERVE_MORE", "confidence": "medium", "reason": "collect one more window"},
        proposal,
    )
    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["confidence"] == "high"
    assert decision["stall_escalation"]["from_action"] == "OBSERVE_MORE"
    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["deterministic_support"] is True


def test_repeated_stall_escalation_respects_active_cpu_counterevidence() -> None:
    proposal = {
        "proposal_id": "rp_busy",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "progress_snapshot": {
            "runtime_sec": 1800.0,
            "stdout_last_line_age_sec": 1800.0,
            "artifact_last_update_age_sec": 1800.0,
            "metric_last_update_age_sec": 1800.0,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
            "multi_window_low_progress": True,
            "process_tree_cpu": {"available": True, "busy_child_count": 1, "total_cpu_pct": 120.0, "child_cpu_pct": 120.0},
        },
        "review_history": {"last_action": "OBSERVE_MORE", "observe_count": 2},
    }

    assert proposal_has_severe_stalled_no_work(proposal) is False
    decision = enforce_repeated_stall_escalation(
        {"action": "OBSERVE_MORE", "confidence": "medium", "reason": "silent but busy"},
        proposal,
    )

    assert decision["action"] == "OBSERVE_MORE"


def test_arbiter_gate_downgrades_stop_after_for_valid_deliverable_without_low_progress() -> None:
    proposal = {
        "reason_code": "active_intervention:dataloader_bottleneck_low_gpu_high_cpu",
        "progress_snapshot": {
            "progress_signal": "active",
            "progress_confidence": "medium",
            "deliverable_validity": "produced_valid",
        },
    }

    gated = enforce_arbiter_kill_gate(
        {
            "action": "STOP_AFTER_DELIVERABLE_AND_RELEASE",
            "confidence": "high",
            "reason": "valid deliverable already exists; release resources",
        },
        proposal,
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["gate"]["blocked_reason"] == "deliverable_not_resource_kill_evidence"


def test_arbiter_gate_downgrades_stop_after_for_invalid_deliverable() -> None:
    proposal = {
        "reason_code": "active_intervention:deliverable_complete_resource_hold",
        "progress_snapshot": {
            "progress_signal": "active",
            "progress_confidence": "medium",
            "deliverable_validity": "produced_invalid",
        },
    }

    gated = enforce_arbiter_kill_gate(
        {
            "action": "STOP_AFTER_DELIVERABLE_AND_RELEASE",
            "confidence": "high",
            "reason": "deliverable exists",
        },
        proposal,
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["gate"]["blocked_reason"] == "deliverable_not_resource_kill_evidence"


def test_arbiter_gate_distinguishes_stop_after_missing_from_unknown_evidence() -> None:
    missing = {
        "reason_code": "active_intervention:deliverable_complete_resource_hold",
        "progress_snapshot": {"progress_signal": "active", "progress_confidence": "medium"},
    }
    unknown = {
        "reason_code": "active_intervention:deliverable_complete_resource_hold",
        "progress_snapshot": {
            "progress_signal": "active",
            "progress_confidence": "medium",
            "deliverable_validity": "produced_unknown",
        },
    }

    missing_gated = enforce_arbiter_kill_gate(
        {"action": "STOP_AFTER_DELIVERABLE_AND_RELEASE", "confidence": "high", "reason": "complete"},
        missing,
    )
    unknown_gated = enforce_arbiter_kill_gate(
        {"action": "STOP_AFTER_DELIVERABLE_AND_RELEASE", "confidence": "high", "reason": "complete"},
        unknown,
    )

    assert missing_gated["action"] == "OBSERVE_MORE"
    assert missing_gated["gate"]["blocked_reason"] == "deliverable_not_resource_kill_evidence"
    assert unknown_gated["action"] == "OBSERVE_MORE"
    assert unknown_gated["gate"]["blocked_reason"] == "deliverable_not_resource_kill_evidence"

def test_lhr_observer_progress_signal_enters_proposal(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=600.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=600.0,
        low_progress_no_artifact_sec=600.0,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=2400.0,
        stdout_age_sec=1300.0,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="training",
    )

    progress = decision["proposal"]["progress_snapshot"]
    assert progress["progress_signal"] == "stalled"
    assert progress["progress_confidence"] == "high"
    assert progress["multi_window_low_progress"] is True


def test_resource_arbiter_prompt_includes_score_context() -> None:
    prompt = build_resource_arbiter_prompt(
        {
            "proposal_id": "p-score",
            "score_context": {
                "valid_best_score": {"value": 0.89468, "lower_is_better": False},
                "capture_gap": False,
            },
        }
    )

    assert "score_context" in prompt
    assert "0.89468" in prompt


def test_resource_observer_score_context_fact_card_hides_paths(tmp_path: Path) -> None:
    (tmp_path / "lhr_stage_performance.csv").write_text(
        "row_order,candidate_id,worker_id,stage_id,metric_value,metric_name,lower_is_better,validation_ok,val_score_type,selection_eligible,selection_score,metric_source_note,brief,why,submission_snapshot,candidate_ready,submission_status,snapshot_path\n"
        "1,W00:L01:S01,W00,S01,0.894680,Final Validation Score,0,1,holdout,1,0.894680,type=holdout,valid auc,ok,snapshots/W00/s01.csv,1,ready,/abs/snapshots/W00/S01\n",
        encoding="utf-8",
    )
    observer, _ = make_observer(tmp_path)

    card = observer._score_context_fact_card()

    assert card["valid_best_score"]["value"] == 0.89468
    assert card["valid_best_score"]["lower_is_better"] is False
    assert "artifact_path" not in card["valid_best_score"]
    assert "snapshot_path" not in card["valid_best_score"]

def test_resource_intervention_summary_uses_trusted_valid_best(tmp_path: Path) -> None:
    (tmp_path / "lhr_stage_performance.csv").write_text(
        "row_order,candidate_id,worker_id,stage_id,metric_value,metric_name,lower_is_better,validation_ok,val_score_type,selection_eligible,selection_score,metric_source_note,brief,why,submission_snapshot,candidate_ready,submission_status,snapshot_path\n"
        "1,W01:L01:S10,W01,S10,0.894680,Final Validation Score,0,1,holdout,1,0.894680,type=holdout,valid auc,ok,snapshots/W01/s10.csv,1,ready,/abs/snapshots/W01/S10\n",
        encoding="utf-8",
    )
    observer, _ = make_observer(tmp_path)

    feedback = observer._append_resource_intervention_summary(
        "RESOURCE_FEEDBACK: terminated_low_progress_command because no heartbeat.\n",
        action="KILL_AND_REPLAN",
        reason="active_intervention:low_progress",
    )

    assert "RESOURCE_INTERVENTION_SUMMARY:" in feedback
    assert "action=KILL_AND_REPLAN" in feedback
    assert "current_valid_best=0.89468" in feedback
    assert "current_valid_best_source=lhr_stage_performance.csv" in feedback
    assert "current_valid_best_validity=valid_comparable" in feedback
    assert "lower_is_better=false" in feedback
    assert "best_worker=W01" in feedback
    assert "best_stage=S10" in feedback


def test_resource_intervention_summary_skips_same_validation_meta_best(tmp_path: Path) -> None:
    (tmp_path / "lhr_stage_performance.csv").write_text(
        "row_order,candidate_id,worker_id,stage_id,metric_value,metric_name,lower_is_better,validation_ok,"
        "val_score_type,selection_eligible,selection_score,metric_validity,metric_validity_note,"
        "metric_validity_reason_code,metric_source_note,evaluator_backend,evaluator_status,brief,why,"
        "submission_snapshot,candidate_ready,submission_status,snapshot_path\n"
        "1,W00:L02:S24,W00,S24,0.708300,Final Validation Score,0,1,holdout,1,0.708300,high,"
        "comparable_holdout: clean holdout,comparable_holdout,Submission is valid.,local_csv,ok,valid ensemble,"
        "clean held-out score,snapshots/W00/s24.csv,1,ready,/abs/snapshots/W00/S24\n"
        "2,W00:L05:S12,W00,S12,0.967800,Final Validation Score,0,1,holdout,1,0.967800,medium,"
        "same_validation_meta_fit: validation set tuned ensemble weights,same_validation_meta_fit,"
        "Submission is valid.,local_csv,ok,contaminated high score,validation reuse inflated score,"
        "snapshots/W00/s12.csv,1,ready,/abs/snapshots/W00/S12\n",
        encoding="utf-8",
    )
    observer, _ = make_observer(tmp_path)

    feedback = observer._append_resource_intervention_summary(
        "RESOURCE_FEEDBACK: terminated_low_progress_command because no heartbeat.\n",
        action="KILL_AND_REPLAN",
        reason="active_intervention:low_progress",
    )

    assert "RESOURCE_INTERVENTION_SUMMARY:" in feedback
    assert "current_valid_best=0.7083" in feedback
    assert "best_stage=S24" in feedback
    assert "current_valid_best=0.9678" not in feedback
    assert "best_stage=S12" not in feedback


def test_resource_intervention_summary_marks_untrusted_best(tmp_path: Path) -> None:
    observer, _ = make_observer(tmp_path)

    feedback = observer._append_resource_intervention_summary(
        "RESOURCE_FEEDBACK: terminated_low_progress_command because no heartbeat.\n",
        action="KILL_AND_REPLAN",
        reason="active_intervention:low_progress",
    )

    assert "RESOURCE_INTERVENTION_SUMMARY:" in feedback
    assert "current_valid_best=unknown" in feedback
    assert "current_valid_best_status=best_unreliable" in feedback
    assert "best_unreliable_reason=metric_direction_or_source_uncertain" in feedback


def test_budget_priority_marks_waiter_low_progress_as_high(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        arbiter_contention_min_waiter_age_sec=300.0,
    )
    priority = observer._budget_priority_for_proposal({
        "proposal_id": "rp_budget_priority",
        "proposal_type": "resource_contention_review",
        "reason_code": "low_progress_with_waiter",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {"progress_signal": "unknown", "progress_confidence": "low"},
        "waiters": [{"job_id": "W01:bash:00002", "queue_age_sec": 480.0}],
        "suggested_actions": ["OBSERVE_MORE", "KILL_AND_REPLAN"],
    })

    assert priority["level"] == "high"
    assert "waiter_blocked" in priority["reasons"]
    assert "reason_indicates_low_value_or_stall" in priority["reasons"]


def test_low_priority_repeated_llm_review_is_deferred(tmp_path) -> None:
    calls: list[dict] = []

    async def deny(proposal):
        calls.append(dict(proposal))
        return {"action": "DENY_KILL", "reason": "active and useful", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=deny,
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
    )
    proposal = {
        "proposal_id": "rp_low_budget",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_holder_observe",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE"],
    }

    first = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))
    second = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))

    assert first["action"] == "DENY_KILL"
    assert second["action"] == "OBSERVE_MORE"
    assert "llm_budget_deferred" in second["decision"]["reason"]
    assert len(calls) == 1
    events = _resource_events(tmp_path)
    deferred = [event for event in events if event["event_type"] == "llm_budget_deferred"]
    assert deferred
    assert deferred[-1]["payload"]["budget_priority"]["level"] == "low"


def test_new_metric_history_triggers_fresh_llm_review(tmp_path) -> None:
    calls: list[dict] = []

    async def decide(proposal):
        calls.append(dict(proposal))
        return {"action": "DENY_KILL", "reason": "reviewed latest metric", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=decide,
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
    )
    base = {
        "proposal_id": "rp_metric_generation",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_holder_observe",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {
            "runtime_sec": 1200,
            "progress_signal": "active",
            "progress_confidence": "high",
            "metric_history_text": "Epoch 1 val_auc=0.800",
        },
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
    }

    first = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=base))
    repeated = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=base))
    updated = dict(base)
    updated["progress_snapshot"] = dict(base["progress_snapshot"])
    updated["progress_snapshot"]["metric_history_text"] += "\nEpoch 2 val_auc=0.790"
    third = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=updated))

    assert first["action"] == "DENY_KILL"
    assert repeated["action"] == "OBSERVE_MORE"
    assert third["action"] == "DENY_KILL"
    assert len(calls) == 2
    assert calls[-1]["budget_priority"]["level"] == "low"
    assert "new_metric_history_since_llm_review" in calls[-1]["budget_priority"]["reasons"]


def test_advisory_continue_commitment_becomes_negotiated_timebox(tmp_path: Path) -> None:
    async def advisory(_proposal):
        return {
            "preference": "continue",
            "confidence": "medium",
            "reason": "Need one clean held-out validation metric before judging route value.",
            "commitment": "print clean held-out validation RMSLE",
            "expected_next_artifact": "tmp/train_s03.log with validation RMSLE",
        }

    arbiter_calls: list[dict] = []

    async def arbiter(proposal):
        arbiter_calls.append(dict(proposal))
        return {"action": "DENY_KILL", "reason": "active and useful", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter,
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)
    from scienceflow.safety.resource import new_review_state

    observer._review_states[job_id] = new_review_state(job_id)
    proposal = {
        "proposal_id": "rp_negotiated_timebox",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_holder_observe",
        "command_id": job_id,
        "resource_review_boundary": {"kind": "progress_window", "reason": "progress_window_elapsed>=1"},
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE"],
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["execution_outcome"] == "TIMEBOX"
    assert result["value_review_outcome"] == "TIMEBOX"
    assert result["suppress_main_agent_feedback"] is True
    assert result["decision"]["reason_code"] == "main_agent_advisory_commitment_timebox"
    assert result["decision"]["clear_on"] == "metric_update"
    assert result["computed_timebox_sec"] is not None
    assert observer._review_states[job_id].job_state_bucket == "TIMEBOX_ACTIVE"
    assert observer._review_states[job_id].timebox_success_condition == "metric_update"
    assert arbiter_calls == []
    events = _resource_events(tmp_path)
    outcome_event = next(event for event in events if event["event_type"] == "resource_review_outcome")
    assert outcome_event["payload"]["action"] == "TIMEBOX"
    assert outcome_event["payload"]["reason_code"] == "main_agent_advisory_commitment_timebox"
    timebox_id = observer._review_states[job_id].timebox_id
    proposal_with_advisory = dict(proposal)
    proposal_with_advisory["main_agent_advisory"] = {
        "preference": "continue",
        "confidence": "medium",
        "commitment": "print clean held-out validation RMSLE",
        "expected_next_artifact": "tmp/train_s03.log with validation RMSLE",
    }

    second = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal_with_advisory, decision_preview={"job_id": job_id}))

    assert second["execution_outcome"] == "NO_ACTION"
    assert second["decision"]["reason_code"] == "advisory_timebox_already_active"
    assert observer._review_states[job_id].timebox_id == timebox_id

    third = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert third["execution_outcome"] == "NO_ACTION"
    assert third["decision"].get("active_timebox_preserved") is True
    assert observer._review_states[job_id].timebox_id == timebox_id


def test_advisory_commitment_missed_timebox_advances_proof_window_without_llm(tmp_path: Path) -> None:
    async def advisory(_proposal):
        raise AssertionError("missed proof-window retry must not ask main agent")

    async def arbiter(_proposal):
        raise AssertionError("missed proof-window retry must not call arbiter LLM")

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter,
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)
    from scienceflow.safety.resource import new_review_state

    observer._review_states[job_id] = new_review_state(job_id)
    proposal = {
        "proposal_id": "rp_expired_timebox",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_intervention:sm_timebox_expired",
        "command_id": job_id,
        "resource_review_boundary": {"kind": "timebox_expired", "reason": "timebox_expired"},
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE"],
    }

    observer._review_states[job_id] = replace(observer._review_states[job_id], failed_proof_window_count=1, proof_window_index=1)
    observer._review_states[job_id] = replace(
        observer._review_states[job_id],
        proof_window_source="negotiated",
        timebox_success_condition="metric_update",
        timebox_id="tb_missed",
        timebox_failure_recorded=True,
    )

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["execution_outcome"] == "TIMEBOX"
    assert result["terminate"] is False
    assert result["decision"].get("reason_code") == "proof_window_retry"
    assert observer._review_states[job_id].job_state_bucket == "TIMEBOX_ACTIVE"
    assert observer._review_states[job_id].proof_window_index == 2
    assert observer._review_states[job_id].failed_proof_window_count == 1
    assert observer._review_states[job_id].proof_window_source == "proof_retry"


def test_recorded_negotiated_failure_skips_advisory_but_uses_resource_arbiter(tmp_path: Path) -> None:
    async def advisory(_proposal):
        raise AssertionError("final resource review must not ask main agent")

    arbiter_seen: dict[str, object] = {}

    async def arbiter(proposal):
        arbiter_seen.update(proposal)
        return {
            "outcome": "KILL",
            "reason_code": "final_resource_budget_review",
            "reason": "proof windows are exhausted and resource budget no longer supports another retry",
            "confidence": "high",
        }

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter,
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)
    from scienceflow.safety.resource import new_review_state

    observer._review_states[job_id] = replace(
        new_review_state(job_id),
        failed_proof_window_count=3,
        proof_window_index=3,
        proof_window_source="negotiated",
        timebox_id="tb_missed",
        timebox_success_condition="metric_update",
        timebox_failure_recorded=True,
    )
    proposal = {
        "proposal_id": "rp_later_low_progress",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:low_progress_heartbeat_stalled",
        "command_id": job_id,
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "stalled", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["execution_outcome"] == "KILL"
    assert result["terminate"] is True
    assert result["decision"].get("reason_code") == "final_resource_budget_review"
    assert result["decision"].get("source") == "resource_arbiter_llm"
    assert arbiter_seen["resource_budget_escalation"]["kind"] == "final_resource_review"
    assert observer._review_states[job_id].job_state_bucket == "ACTIONED"


def test_vague_advisory_commitment_does_not_create_timebox(tmp_path: Path) -> None:
    async def advisory(_proposal):
        return {
            "preference": "continue",
            "confidence": "medium",
            "reason": "I think it may work.",
            "commitment": "it should improve soon",
        }

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=lambda _proposal: {"action": "OBSERVE_MORE", "reason": "uncertain", "confidence": "medium"},
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)
    from scienceflow.safety.resource import new_review_state

    observer._review_states[job_id] = replace(new_review_state(job_id), failed_proof_window_count=1, proof_window_index=1)
    proposal = {
        "proposal_id": "rp_vague_commitment",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_intervention:sm_timebox_expired",
        "command_id": job_id,
        "resource_review_boundary": {"kind": "timebox_expired", "reason": "timebox_expired"},
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE"],
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["execution_outcome"] == "NO_ACTION"
    assert result["decision"].get("reason_code") != "main_agent_advisory_commitment_timebox"
    assert observer._review_states[job_id].proof_window_source == "no_action"


def test_repeated_failed_proof_windows_final_review_can_timebox(tmp_path: Path) -> None:
    async def advisory(_proposal):
        raise AssertionError("final resource review must not ask main agent")

    async def arbiter(proposal):
        assert proposal["resource_budget_escalation"]["kind"] == "final_resource_review"
        return {
            "outcome": "TIMEBOX",
            "reason_code": "final_resource_review_one_last_window",
            "reason": "resource arbiter allows one last bounded observation window",
            "confidence": "medium",
            "clear_on": "progress_advance",
        }

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter,
        arbiter_job_llm_call_cap=0,
        arbiter_job_token_cap=0,
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=0.0,
        review_timebox_windows=1,
    )
    job_id = _heavy_job(observer, tmp_path)
    from scienceflow.safety.resource import new_review_state

    observer._review_states[job_id] = replace(
        new_review_state(job_id),
        failed_proof_window_count=3,
        proof_window_index=3,
    )
    proposal = {
        "proposal_id": "rp_final_proof_override",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_intervention:sm_timebox_expired",
        "command_id": job_id,
        "resource_review_boundary": {"kind": "timebox_expired", "reason": "timebox_expired"},
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["execution_outcome"] == "TIMEBOX"
    assert result["terminate"] is False
    assert result["decision"]["reason_code"] == "final_resource_review_one_last_window"
    assert result["decision"].get("source") == "resource_arbiter_llm"
    assert observer._review_states[job_id].job_state_bucket == "TIMEBOX_ACTIVE"


def test_policy_fallback_kills_quick_probe_overrun_without_value_signal() -> None:
    proposal = {
        "proposal_id": "p-quick-probe",
        "proposal_type": "quick_probe_review",
        "reason_code": "quick_probe_runtime_exceeded",
        "progress_snapshot": {
            "runtime_sec": 640.0,
            "quick_probe_hard_review_sec": 420.0,
            "quick_probe_candidate": True,
            "progress_confidence": "high",
            "progress_signal": "stalled",
            "metric_history_line_count": 0,
            "stdout_lines": 0,
            "stdout_bytes": 0,
            "artifact_updates": [],
            "near_submission": False,
            "meaningful_stdout": False,
            "recoverable_artifact_on_disk": False,
        },
        "execution_facts": {
            "assigned_resource_idle": True,
            "cpu_busy": True,
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
    }

    decision = fallback_policy_decision(proposal)
    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["confidence"] == "high"
    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["deterministic_support"] is True


def test_policy_fallback_kills_active_but_unaffordable_under_finalization() -> None:
    proposal = {
        "proposal_id": "p-active-unaffordable",
        "proposal_type": "kill_proposal",
        "reason_code": "deadline_finalization_reserve",
        "progress_snapshot": {
            "runtime_sec": 5820.0,
            "progress_confidence": "high",
            "progress_signal": "active",
            "metric_history_line_count": 0,
            "stdout_lines": 100,
            "stdout_bytes": 4096,
            "artifact_updates": [],
            "near_submission": False,
            "meaningful_stdout": True,
            "recoverable_artifact_on_disk": False,
            "deadline_event": True,
            "deadline_remaining_sec": 800.0,
            "finalization_reserve_sec": 900.0,
            "eta_confidence": "high",
            "eta_to_current_phase_end_sec": 63760.0,
            "eta_to_deliverable_sec": 63760.0,
            "finish_feasible": False,
        },
        "execution_facts": {
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
    }

    decision = fallback_policy_decision(proposal)
    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["confidence"] == "high"
    assert "deterministic resource facts" in decision["reason"]
    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["deterministic_support"] is True


def test_policy_fallback_observes_quick_probe_overrun_with_stdout_signal() -> None:
    proposal = {
        "proposal_id": "p-quick-probe-stdout",
        "proposal_type": "quick_probe_review",
        "reason_code": "quick_probe_runtime_exceeded",
        "progress_snapshot": {
            "runtime_sec": 640.0,
            "quick_probe_hard_review_sec": 420.0,
            "quick_probe_candidate": True,
            "metric_history_line_count": 0,
            "stdout_lines": 3,
            "stdout_bytes": 128,
            "artifact_updates": [],
            "near_submission": False,
            "meaningful_stdout": True,
            "recoverable_artifact_on_disk": False,
        },
    }

    decision = fallback_policy_decision(proposal)

    assert decision["action"] == "OBSERVE_MORE"
    assert decision["confidence"] == "low"


def test_high_confidence_stop_advisory_is_reused_until_value_changes(tmp_path: Path) -> None:
    calls: list[dict] = []

    async def advisory(proposal):
        calls.append(dict(proposal))
        return {
            "preference": "safe_to_stop",
            "confidence": "high",
            "reason": "route remains below the observed best",
            "ttl_sec": 600.0,
            "observed_phase": "train",
        }

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=600.0,
        arbiter_job_advisory_call_cap=1,
    )
    base = {
        "proposal_id": "rp_advisory_first",
        "proposal_type": "kill_proposal",
        "reason_code": "periodic_efficiency:low_progress_review",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {
            "known_stage": "train",
            "metric_scope_key": "route|train|auc",
            "near_submission": False,
            "metric_history_text": "val_auc=0.70",
        },
        "resource_metric_value": {
            "metric_scope_key": "route|train|auc",
            "useful": False,
            "status": "below_observed_best",
        },
    }

    first = asyncio.run(observer._maybe_attach_main_agent_advisory(base, job_key="W00:bash:00001"))
    repeated = dict(base)
    repeated["proposal_id"] = "rp_advisory_repeated"
    second = asyncio.run(observer._maybe_attach_main_agent_advisory(repeated, job_key="W00:bash:00001"))

    assert first["main_agent_advisory"]["source"] == "main_agent_advisory"
    assert second["main_agent_advisory"]["source"] == "main_agent_advisory_cache"
    assert second["main_agent_advisory"]["reused"] is True
    assert len(calls) == 1
    assert "main_agent_advisory_reused" in {
        event["event_type"] for event in _resource_events(tmp_path)
    }

    improved = dict(repeated)
    improved["proposal_id"] = "rp_advisory_improved"
    improved["progress_snapshot"] = dict(repeated["progress_snapshot"])
    improved["progress_snapshot"]["metric_history_text"] = "val_auc=0.70\nval_auc=0.71"
    improved["resource_metric_value"] = {
        "metric_scope_key": "route|train|auc",
        "useful": False,
        "status": "below_observed_best",
    }
    third = asyncio.run(observer._maybe_attach_main_agent_advisory(improved, job_key="W00:bash:00001"))

    assert "main_agent_advisory" not in third
    assert len(calls) == 1
