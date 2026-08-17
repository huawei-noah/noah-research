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
from pathlib import Path

from scienceflow.safety.resource.artifacts import submission_completion_state
from scienceflow.safety.resource.completion import deliverable_completion_state
from scienceflow.solver.lnr.resource_runtime.review.arbiter import fallback_policy_decision
from scienceflow.solver.lnr.resource_runtime.review.arbiter_gate import enforce_arbiter_kill_gate
from scienceflow.solver.lnr.resource_runtime.review.lease_suspect import (
    LeaseSuspectThresholds,
    classify_active_lease_suspect,
)
from tests.lnr_resource_test_utils import make_observer, payloads


def _events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "resource" / "resource_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_submission_pair(tmp_path: Path, *, submission_header: str) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "sample_submission.csv").write_text("id,CVC - Normal\n1,0\n2,0\n", encoding="utf-8")
    (tmp_path / "submission.csv").write_text(f"{submission_header}\n1,0\n2,0\n", encoding="utf-8")


def _heavy_job(observer, tmp_path: Path) -> str:
    (tmp_path / "train.py").write_text("import torch\ntorch.cuda.is_available()\nprint('train')\n", encoding="utf-8")
    job_id = observer.job_created(
        command="python3 train.py --device cuda --epochs 10",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    return job_id


def _waiter_job(observer, tmp_path: Path) -> str:
    (tmp_path / "waiter.py").write_text("import torch\ntorch.cuda.is_available()\nprint('waiter')\n", encoding="utf-8")
    job_id = observer.job_created(
        command="python3 waiter.py --device cuda --epochs 3",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    return job_id


def _contention_pair(tmp_path: Path):
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
        gpu_idle_lease_warmup_sec=0.0,
        gpu_idle_lease_util_pct=1.0,
        gpu_idle_lease_mem_gb=1.0,
        gpu_share_tt_with_train=False,
    )
    blocker = _heavy_job(observer, tmp_path)
    assert observer.queue_try_acquire(blocker, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    waiter = _waiter_job(observer, tmp_path)
    assert observer.queue_try_acquire(waiter, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is False
    observer._promote(observer._jobs[blocker], reason="test", elapsed_sec=1200.0)
    return observer, blocker, waiter


def test_submission_missing_required_column_is_invalid_not_complete(tmp_path: Path) -> None:
    _write_submission_pair(tmp_path, submission_header="id,Normal")

    state = submission_completion_state(tmp_path, settle_sec=0.0)
    combined = deliverable_completion_state(tmp_path, terminal_signal_seen=True, settle_sec=0.0)

    assert state["complete"] is False
    assert state["deliverable_validity"] == "produced_invalid"
    assert state["schema"]["missing_columns"] == ["CVC - Normal"]
    assert combined["complete"] is False
    assert combined["deliverable_validity"] == "produced_invalid"


def test_submission_matching_sample_schema_is_valid_complete(tmp_path: Path) -> None:
    _write_submission_pair(tmp_path, submission_header="id,CVC - Normal")

    state = submission_completion_state(tmp_path, settle_sec=0.0)

    assert state["complete"] is True
    assert state["deliverable_validity"] == "produced_valid"
    assert state["schema"]["valid"] is True


def test_completed_valid_deliverable_does_not_short_circuit_resource_preflight(tmp_path: Path) -> None:
    _write_submission_pair(tmp_path, submission_header="id,CVC - Normal")
    observer, _sm = make_observer(
        tmp_path,
        deliverable_completion_guard_enabled=True,
        deliverable_completion_settle_sec=0.0,
    )
    job_id = observer.job_created(
        command="python3 solution.py",
        inferred_class="heavy_cpu_candidate",
        gpu_ids=[],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

    decision = observer.resource_preflight_decision(job_id, inferred_class="heavy_cpu_candidate", gpu_ids=[])

    assert decision["allowed"] is True
    assert "status" not in decision
    state = observer._jobs[job_id].last_deliverable_completion_check["state"]
    assert state["complete"] is True
    events = _events(tmp_path)
    facts = [row for row in events if row.get("event") == "resource_completed_deliverable_preflight_fact"]
    assert facts == []


def test_completed_deliverable_preflight_records_valid_state_without_feedback(tmp_path: Path) -> None:
    _write_submission_pair(tmp_path, submission_header="id,CVC - Normal")
    observer, sm = make_observer(
        tmp_path,
        deliverable_completion_guard_enabled=True,
        deliverable_completion_settle_sec=0.0,
    )
    first_job = observer.job_created(
        command="python3 train.py --epochs 10",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert first_job is not None

    first = observer.resource_preflight_decision(first_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])

    assert first["allowed"] is True

    second_job = observer.job_created(
        command="python3 extract_features.py --device cuda",
        inferred_class="gpu_feature_extract",
        gpu_ids=["0"],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert second_job is not None

    second = observer.resource_preflight_decision(second_job, inferred_class="gpu_feature_extract", gpu_ids=["0"])

    assert second["allowed"] is True
    assert observer._jobs[first_job].last_deliverable_completion_check["state"]["complete"] is True
    assert observer._jobs[second_job].last_deliverable_completion_check["state"]["complete"] is True
    assert payloads(sm, "resource_completed_deliverable_preflight_fact") == []
    assert not payloads(sm, "resource_feedback_suppressed")


def test_completed_deliverable_preflight_allows_light_validation(tmp_path: Path) -> None:
    _write_submission_pair(tmp_path, submission_header="id,CVC - Normal")
    observer, _sm = make_observer(
        tmp_path,
        deliverable_completion_guard_enabled=True,
        deliverable_completion_settle_sec=0.0,
    )
    job_id = observer.job_created(
        command="python3 predict.py --check",
        inferred_class="pure_tt_cpu",
        gpu_ids=[],
        timeout_sec=300.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

    decision = observer.resource_preflight_decision(job_id, inferred_class="pure_tt_cpu", gpu_ids=[])

    assert decision["allowed"] is True


def test_completed_deliverable_preflight_allows_readonly_with_inherited_gpu_id(tmp_path: Path) -> None:
    _write_submission_pair(tmp_path, submission_header="id,CVC - Normal")
    observer, _sm = make_observer(
        tmp_path,
        deliverable_completion_guard_enabled=True,
        deliverable_completion_settle_sec=0.0,
    )
    job_id = observer.job_created(
        command="pwd 2>&1",
        inferred_class="readonly_cpu",
        gpu_ids=["0"],
        timeout_sec=30.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

    decision = observer.resource_preflight_decision(job_id, inferred_class="readonly_cpu", gpu_ids=["0"])

    assert decision["allowed"] is True


def test_completed_deliverable_preflight_allows_scratch_notes_even_if_misclassified_heavy(tmp_path: Path) -> None:
    _write_submission_pair(tmp_path, submission_header="id,CVC - Normal")
    observer, _sm = make_observer(
        tmp_path,
        deliverable_completion_guard_enabled=True,
        deliverable_completion_settle_sec=0.0,
    )
    job_id = observer.job_created(
        command="cat > tmp/experiment_notes.md << 'EOF'\nFinal Validation Score: 0.9\nEOF",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=30.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

    decision = observer.resource_preflight_decision(job_id, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])

    assert decision["allowed"] is True


def test_completed_deliverable_preflight_allows_solution_rerun_without_resource_fact(tmp_path: Path) -> None:
    _write_submission_pair(tmp_path, submission_header="id,CVC - Normal")
    observer, _sm = make_observer(
        tmp_path,
        deliverable_completion_guard_enabled=True,
        deliverable_completion_settle_sec=0.0,
    )
    job_id = observer.job_created(
        command="python3 solution.py 2>&1",
        inferred_class="heavy_cpu_candidate",
        gpu_ids=[],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

    decision = observer.resource_preflight_decision(job_id, inferred_class="heavy_cpu_candidate", gpu_ids=[])

    assert decision["allowed"] is True


def test_invalid_deliverable_blocks_new_heavy_preflight(tmp_path: Path) -> None:
    _write_submission_pair(tmp_path, submission_header="id,Normal")
    observer, _sm = make_observer(
        tmp_path,
        deliverable_completion_guard_enabled=True,
        deliverable_completion_settle_sec=0.0,
    )
    job_id = observer.job_created(
        command="python3 train.py",
        inferred_class="heavy_cpu_candidate",
        gpu_ids=[],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

    decision = observer.resource_preflight_decision(job_id, inferred_class="heavy_cpu_candidate", gpu_ids=[])

    assert decision["allowed"] is False
    assert decision["reason"] == "invalid_deliverable_schema_preflight"
    assert decision["deliverable_validity"] == "produced_invalid"


def test_active_lease_value_suspect_does_not_become_resource_stop_candidate() -> None:
    out = classify_active_lease_suspect(
        elapsed_sec=2400.0,
        has_waiter=True,
        has_active_lease=True,
        resource_class="heavy_gpu_candidate",
        progress_snapshot={"progress_signal": "active", "progress_confidence": "high", "progress_signal_windows": 3},
        gpu_memory={
            "sample_available": True,
            "gpu_util_p90_pct": 20.0,
            "gpu_mem_current_gb": 3.0,
            "gpu_mem_peak_gb": 3.0,
        },
        deliverable_validity="none",
        route_viability={"route_viability_unproven": True},
        thresholds=LeaseSuspectThresholds(warmup_sec=0.0, min_progress_windows=2),
    )

    assert out["active_lease_suspect"] is True
    assert out["value_suspect"] is True
    assert out["resource_suspect"] is False
    assert out["stop_review_priority"] == "low"


def test_active_lease_resource_suspect_requires_progress_evidence() -> None:
    common = {
        "elapsed_sec": 2400.0,
        "has_waiter": True,
        "has_active_lease": True,
        "resource_class": "heavy_gpu_candidate",
        "gpu_memory": {
            "sample_available": True,
            "gpu_util_p90_pct": 0.0,
            "gpu_mem_current_gb": 3.0,
            "gpu_mem_peak_gb": 3.0,
        },
        "deliverable_validity": "none",
        "route_viability": {},
        "thresholds": LeaseSuspectThresholds(warmup_sec=0.0, min_progress_windows=2),
    }

    weak = classify_active_lease_suspect(
        progress_snapshot={"progress_signal": "stalled", "progress_confidence": "low", "progress_signal_windows": 3},
        **common,
    )
    strong = classify_active_lease_suspect(
        progress_snapshot={"progress_signal": "stalled", "progress_confidence": "high", "progress_signal_windows": 3},
        **common,
    )

    assert weak["resource_suspect"] is False
    assert strong["resource_suspect"] is True
    assert strong["stop_review_priority"] == "high"




def test_unknown_progress_suspect_requires_no_active_work_counterevidence() -> None:
    common = {
        "elapsed_sec": 1800.0,
        "has_waiter": True,
        "has_active_lease": True,
        "resource_class": "heavy_gpu_candidate",
        "gpu_memory": {
            "sample_available": True,
            "gpu_util_p90_pct": 0.0,
            "gpu_mem_current_gb": 4.0,
            "gpu_mem_peak_gb": 4.0,
        },
        "deliverable_validity": "none",
        "route_viability": {},
        "thresholds": LeaseSuspectThresholds(
            warmup_sec=0.0,
            min_progress_windows=2,
            unknown_progress_grace_sec=300.0,
            unknown_progress_stale_output_sec=300.0,
            active_work_cpu_pct=20.0,
        ),
    }

    stalled_unknown = classify_active_lease_suspect(
        progress_snapshot={
            "progress_signal": "unknown",
            "progress_confidence": "low",
            "progress_signal_windows": 3,
            "stdout_last_line_age_sec": 900.0,
            "metric_last_update_age_sec": 900.0,
            "artifact_last_update_age_sec": 900.0,
            "process_tree_cpu": {"available": True, "total_cpu_pct": 0.8, "child_cpu_pct": 0.0, "busy_child_count": 0},
        },
        **common,
    )
    active_unknown = classify_active_lease_suspect(
        progress_snapshot={
            "progress_signal": "unknown",
            "progress_confidence": "low",
            "progress_signal_windows": 3,
            "stdout_last_line_age_sec": 900.0,
            "metric_last_update_age_sec": 900.0,
            "artifact_last_update_age_sec": 900.0,
            "process_tree_cpu": {"available": True, "total_cpu_pct": 55.0, "child_cpu_pct": 55.0, "busy_child_count": 1},
        },
        **common,
    )

    assert stalled_unknown["unknown_progress_suspect"] is True
    assert stalled_unknown["resource_suspect"] is False
    assert stalled_unknown["stop_review_priority"] == "medium"
    assert active_unknown["unknown_progress_suspect"] is False
    assert active_unknown["active_work_counterevidence"]["active"] is True

def test_contention_review_records_resource_suspect_and_policy_can_kill(tmp_path: Path) -> None:
    observer, blocker, waiter = _contention_pair(tmp_path)
    job = observer._jobs[blocker]
    job.last_gpu_util_sample = {
        "sample": {
            "available": True,
            "gpus": [{"gpu_id": "0", "utilization_gpu_pct": 0.0, "memory_used_mb": 4096.0, "memory_total_mb": 49152.0}],
        }
    }
    job.progress_signal = "stalled"
    job.progress_signal_windows = 3
    job.progress_signal_state = {
        "progress_signal": "stalled",
        "progress_confidence": "high",
        "progress_signal_windows": 3,
        "multi_window_low_progress": True,
    }

    decision = observer._contention_review_decision(job, {"elapsed_sec": 1200.0, "stdout_lines": 0, "stdout_bytes": 0, "stdout_age_sec": 1200.0})
    proposal = decision["proposal"]
    fallback = fallback_policy_decision(proposal)
    gated = enforce_arbiter_kill_gate(fallback, proposal)

    assert proposal["waiters"][0]["job_id"] == waiter
    assert proposal["blocker"]["active_lease_suspect"]["resource_suspect"] is True
    assert "KILL_AND_REPLAN" in proposal["suggested_actions"]
    assert fallback["action"] == "OBSERVE_MORE"
    assert gated["action"] == "OBSERVE_MORE"
    assert any(event["event_type"] == "active_lease_suspect" for event in _events(tmp_path))


def test_value_suspect_only_contention_does_not_suggest_kill(tmp_path: Path) -> None:
    observer, blocker, _waiter = _contention_pair(tmp_path)
    job = observer._jobs[blocker]
    job.last_gpu_util_sample = {
        "sample": {
            "available": True,
            "gpus": [{"gpu_id": "0", "utilization_gpu_pct": 20.0, "memory_used_mb": 4096.0, "memory_total_mb": 49152.0}],
        }
    }
    job.progress_signal = "active"
    job.progress_signal_windows = 3
    job.progress_signal_state = {"progress_signal": "active", "progress_confidence": "high", "progress_signal_windows": 3}

    decision = observer._contention_review_decision(job, {"elapsed_sec": 2400.0, "stdout_lines": 10, "stdout_bytes": 100, "stdout_age_sec": 1.0})
    proposal = decision["proposal"]
    fallback = fallback_policy_decision(proposal)

    assert proposal["blocker"]["active_lease_suspect"]["value_suspect"] is True
    assert proposal["blocker"]["active_lease_suspect"]["resource_suspect"] is False
    assert "KILL_AND_REPLAN" not in proposal["suggested_actions"]
    assert fallback["action"] == "OBSERVE_MORE"

def test_observe_more_decision_schedules_requested_rereview(tmp_path: Path) -> None:
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
        arbiter_periodic_min_interval_sec=999.0,
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)
    job = observer._jobs[job_id]
    proposal_type = "periodic_efficiency_review"
    observer._mark_review_proposal_emitted(job, proposal_type, now=1000.0)

    proposal = {
        "proposal_id": "rp_test_observe_more",
        "proposal_type": proposal_type,
        "command_id": job_id,
    }
    decision = {
        "decision_id": "rd_test_observe_more",
        "action": "OBSERVE_MORE",
        "reason": "collect one more window",
        "confidence": "medium",
        "observe_more_sec": 5.0,
    }
    observer._record_arbiter_decision_events(proposal, decision, decision_preview={"job_id": job_id})

    key = f"{job_id}:{proposal_type}"
    assert key in observer._review_observe_more_until
    assert observer._review_cooldown_active(job, proposal_type, min_interval_sec=999.0, now=observer._review_observe_more_until[key] - 0.1)
    assert not observer._review_cooldown_active(job, proposal_type, min_interval_sec=999.0, now=observer._review_observe_more_until[key] + 0.1)
    assert key not in observer._review_observe_more_until

    scheduled = [e for e in _events(tmp_path) if e.get("event_type") == "resource_review_observe_more_scheduled"]
    assert scheduled
    assert scheduled[-1]["payload"]["delay_field"] == "observe_more_sec"
    assert scheduled[-1]["payload"]["observe_more_sec"] == 5.0


def test_deny_kill_ttl_schedules_requested_rereview(tmp_path: Path) -> None:
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
        arbiter_periodic_min_interval_sec=999.0,
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)
    job = observer._jobs[job_id]
    proposal_type = "periodic_efficiency_review"
    observer._mark_review_proposal_emitted(job, proposal_type, now=2000.0)

    proposal = {
        "proposal_id": "rp_test_deny_ttl",
        "proposal_type": proposal_type,
        "command_id": job_id,
    }
    decision = {
        "decision_id": "rd_test_deny_ttl",
        "action": "DENY_KILL",
        "reason": "productive dataloader bottleneck",
        "confidence": "medium",
        "ttl_sec": 7.0,
    }
    observer._record_arbiter_decision_events(proposal, decision, decision_preview={"job_id": job_id})

    key = f"{job_id}:{proposal_type}"
    assert key in observer._review_observe_more_until
    assert observer._review_cooldown_active(job, proposal_type, min_interval_sec=999.0, now=observer._review_observe_more_until[key] - 0.1)
    assert not observer._review_cooldown_active(job, proposal_type, min_interval_sec=999.0, now=observer._review_observe_more_until[key] + 0.1)

    scheduled = [e for e in _events(tmp_path) if e.get("event_type") == "resource_review_observe_more_scheduled"]
    assert scheduled[-1]["payload"]["action"] == "NO_ACTION"
    assert scheduled[-1]["payload"]["raw_action"] == "DENY_KILL"
    assert scheduled[-1]["payload"]["delay_field"] == "ttl_sec"
    assert scheduled[-1]["payload"]["ttl_sec"] == 7.0


def test_unknown_progress_observe_escalation_enables_kill_support(tmp_path: Path) -> None:
    observer, blocker, _waiter = _contention_pair(tmp_path)
    job = observer._jobs[blocker]
    job.last_gpu_util_sample = {
        "sample": {
            "available": True,
            "gpus": [{"gpu_id": "0", "utilization_gpu_pct": 0.0, "memory_used_mb": 4096.0, "memory_total_mb": 49152.0}],
        }
    }
    job.progress_signal = "unknown"
    job.progress_signal_windows = 3
    job.progress_signal_state = {"progress_signal": "unknown", "progress_confidence": "low", "progress_signal_windows": 3}
    signal = {
        "elapsed_sec": 1800.0,
        "stdout_lines": 0,
        "stdout_bytes": 0,
        "stdout_age_sec": 1800.0,
        "process_tree_cpu": {"available": True, "total_cpu_pct": 0.5, "child_cpu_pct": 0.0, "busy_child_count": 0},
    }

    first = observer._contention_review_decision(job, signal)
    proposal = first["proposal"]
    assert proposal["blocker"]["active_lease_suspect"]["unknown_progress_suspect"] is True
    assert proposal["blocker"]["active_lease_suspect"]["resource_suspect"] is False
    assert "KILL_AND_REPLAN" not in proposal["suggested_actions"]
    assert fallback_policy_decision(proposal)["action"] == "OBSERVE_MORE"

    observe = {"decision_id": "rd_obs1", "action": "OBSERVE_MORE", "reason": "collect evidence", "confidence": "medium", "observe_more_sec": 1.0}
    observer._record_arbiter_decision_events(proposal, observe, decision_preview={"job_id": blocker})
    observer._record_arbiter_decision_events(proposal, {**observe, "decision_id": "rd_obs2"}, decision_preview={"job_id": blocker})

    escalated = observer._attach_review_history(dict(proposal), now=999999.0)
    assert escalated["review_history"]["repeated_observe_support"] is True
    assert escalated["structured_opportunity_cost_support"] is True
    fallback = fallback_policy_decision(escalated)
    gated = enforce_arbiter_kill_gate(fallback, escalated)
    assert fallback["action"] == "OBSERVE_MORE"
    assert gated["action"] == "OBSERVE_MORE"


def test_unknown_progress_with_active_cpu_counterevidence_does_not_become_suspect_in_contention(tmp_path: Path) -> None:
    observer, blocker, _waiter = _contention_pair(tmp_path)
    job = observer._jobs[blocker]
    job.last_gpu_util_sample = {
        "sample": {
            "available": True,
            "gpus": [{"gpu_id": "0", "utilization_gpu_pct": 0.0, "memory_used_mb": 4096.0, "memory_total_mb": 49152.0}],
        }
    }
    job.progress_signal = "unknown"
    job.progress_signal_windows = 3
    job.progress_signal_state = {"progress_signal": "unknown", "progress_confidence": "low", "progress_signal_windows": 3}

    decision = observer._contention_review_decision(
        job,
        {
            "elapsed_sec": 1800.0,
            "stdout_lines": 0,
            "stdout_bytes": 0,
            "stdout_age_sec": 1800.0,
            "process_tree_cpu": {"available": True, "total_cpu_pct": 80.0, "child_cpu_pct": 80.0, "busy_child_count": 1},
        },
    )
    suspect = decision["proposal"]["blocker"]["active_lease_suspect"]

    assert suspect["unknown_progress_suspect"] is False
    assert suspect["active_work_counterevidence"]["active"] is True
    assert "KILL_AND_REPLAN" not in decision["proposal"]["suggested_actions"]
