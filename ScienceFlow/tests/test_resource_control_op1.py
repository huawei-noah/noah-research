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

from scienceflow.solver.lnr.context_hygiene import evaluate_context_hygiene_compact
from scienceflow.solver.lnr.resource_runtime import process_liveness
from scienceflow.solver.lnr.resource_runtime.review.arbiter import fallback_policy_decision, normalize_arbiter_decision
from scienceflow.solver.lnr.resource_runtime.gpu_lease_store import GPULeaseStore
from scienceflow.solver.lnr.resource_runtime.quick_probe import classify_quick_probe_command
from tests.lnr_resource_test_utils import FakeStateMachine, make_observer, payloads
from scienceflow.solver.lnr.resource_runtime.review.state_generation import build_resource_state_generation


def _resource_event_types(tmp_path) -> list[str]:
    path = tmp_path / "resource" / "resource_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line).get("event_type", "") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _proc_row(*, pid: int, ppid: int = 1, pgid: int | None = None, session: int = 1, state: str = "S", comm: str = "python", start: float = 100.0) -> dict:
    return {
        "pid": pid,
        "ppid": ppid,
        "pgid": pgid if pgid is not None else pid,
        "session": session,
        "state": state,
        "comm": comm,
        "start_time_epoch": start,
    }


def test_process_liveness_does_not_mark_reparented_child_completed(monkeypatch) -> None:
    monkeypatch.setattr(
        process_liveness,
        "_proc_rows",
        lambda: {101: _proc_row(pid=101, ppid=1, pgid=100, session=7, comm="python")},
    )

    fact = process_liveness.build_process_liveness(
        root_pid=100,
        root_pgid=100,
        exit_code_seen=True,
        terminal_signal_seen=True,
    )

    assert fact["status"] == "inconsistent"
    assert fact["process_group_alive"] is True
    assert fact["process_group_member_count"] == 1
    assert fact["python_child_count"] == 1


def test_process_liveness_requires_terminal_signal_for_completed(monkeypatch) -> None:
    monkeypatch.setattr(process_liveness, "_proc_rows", lambda: {})

    unknown = process_liveness.build_process_liveness(root_pid=100, root_pgid=100)
    exited = process_liveness.build_process_liveness(root_pid=100, root_pgid=100, terminal_signal_seen=True)

    assert unknown["status"] == "unknown"
    assert exited["status"] == "exited"


def _state_snapshot(*, runtime_sec: float = 60.0, progress_confidence: str = "low", stdout_age_sec: float = 10.0, blocked_until_unlock: bool = False, unlock_condition: str = "", lease_active: bool = True, released_idle: bool = False) -> dict:
    return {
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "blocked_until_unlock": blocked_until_unlock,
        "unlock_condition": unlock_condition,
        "progress_snapshot": {
            "runtime_sec": runtime_sec,
            "progress_signal": "unknown",
            "progress_confidence": progress_confidence,
            "stdout_last_line_age_sec": stdout_age_sec,
            "artifact_last_update_age_sec": 10.0,
            "metric_last_update_age_sec": 10.0,
            "process_liveness": {"status": "alive"},
        },
        "resource_snapshot": {
            "gpu_ids": ["0"],
            "lease": {"active": lease_active, "released_idle": released_idle, "resource_ids": ["0"]},
            "resources": [
                {
                    "resource_type": "gpu",
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 512.0,
                }
            ],
        },
    }


def test_state_generation_separates_control_noise_from_feedback_capability() -> None:
    base = build_resource_state_generation(_state_snapshot(runtime_sec=60, progress_confidence="low", stdout_age_sec=10))
    noisy = build_resource_state_generation(_state_snapshot(runtime_sec=8200, progress_confidence="high", stdout_age_sec=1200))

    assert noisy.control_generation_key != base.control_generation_key
    assert noisy.feedback_generation_key == base.feedback_generation_key

    capability_changed = build_resource_state_generation(
        _state_snapshot(blocked_until_unlock=True, unlock_condition="submission_schema_valid")
    )
    assert capability_changed.feedback_generation_key != base.feedback_generation_key
    assert capability_changed.feedback_fields["unlock_condition"] == "submission_schema_valid"


def test_state_generation_changes_when_metric_history_changes() -> None:
    first_snapshot = _state_snapshot()
    first_snapshot["progress_snapshot"]["metric_history_text"] = "Epoch 1 val_auc=0.800"
    second_snapshot = _state_snapshot()
    second_snapshot["progress_snapshot"]["metric_history_text"] = "Epoch 1 val_auc=0.800\nEpoch 2 val_auc=0.790"

    first = build_resource_state_generation(first_snapshot)
    repeated = build_resource_state_generation(first_snapshot)
    second = build_resource_state_generation(second_snapshot)

    assert repeated.control_generation_key == first.control_generation_key
    assert second.control_generation_key != first.control_generation_key
    assert second.feedback_generation_key == first.feedback_generation_key
    assert second.control_fields["metric_history_digest"]


def test_state_generation_feedback_key_tracks_execution_capability() -> None:
    base = build_resource_state_generation(_state_snapshot())
    mismatch_snapshot = _state_snapshot()
    mismatch_snapshot["execution_facts"] = {
        "declared_device": "gpu",
        "observed_device": "cpu",
        "intent_device_mismatch": True,
        "assigned_resource_idle": True,
        "artifact_update_without_metric_or_stdout": True,
        "activity_without_metric_submission_or_stdout": True,
    }

    mismatch = build_resource_state_generation(mismatch_snapshot)

    assert mismatch.feedback_generation_key != base.feedback_generation_key
    assert mismatch.feedback_fields["intent_device_mismatch"] is True
    assert mismatch.feedback_fields["observed_device"] == "cpu"


def test_arbiter_liveness_guard_clamps_inconsistent_decision_ttl() -> None:
    proposal = {
        "proposal_id": "rp_live",
        "proposal_type": "kill_proposal",
        "progress_snapshot": {"process_liveness": {"status": "inconsistent"}},
    }

    decision = normalize_arbiter_decision(
        {"action": "DENY_KILL", "reason": "still useful", "confidence": "high", "ttl_sec": 900, "observe_more_sec": 300},
        proposal=proposal,
    )

    assert decision["action"] == "DENY_KILL"
    assert decision["ttl_sec"] <= 60
    assert decision["observe_more_sec"] <= 60
    assert decision["liveness_guard"]["force_process_rescan"] is True


def test_arbiter_completed_reason_conflicts_with_alive_process() -> None:
    proposal = {
        "proposal_id": "rp_completed_conflict",
        "proposal_type": "kill_proposal",
        "progress_snapshot": {"process_liveness": {"status": "alive"}},
    }

    decision = normalize_arbiter_decision(
        {"action": "DENY_KILL", "reason": "Command already completed; no active process", "confidence": "high", "ttl_sec": 900},
        proposal=proposal,
    )

    assert decision["confidence"] == "low"
    assert decision["ttl_sec"] <= 60
    assert "process_liveness does not support completed/no-active-process" in decision["reason"]
    assert decision["liveness_guard"]["completed_liveness_conflict"] is True


def test_policy_fallback_observes_more_for_inconsistent_liveness() -> None:
    decision = fallback_policy_decision(
        {
            "proposal_type": "kill_proposal",
            "progress_snapshot": {"process_liveness": {"status": "inconsistent"}},
        }
    )

    assert decision["action"] == "OBSERVE_MORE"
    assert decision["ttl_sec"] <= 60
    assert decision["confidence"] == "low"


def test_strict_exclusive_idle_release_reserves_gpu_until_reacquire_or_final_release(tmp_path) -> None:
    store = GPULeaseStore(tmp_path / "gpu_leases.json", idle_release_admission_mode="strict_exclusive")

    ok, first = store.try_acquire(
        job_id="job-a",
        worker_id="W00",
        gpu_ids=["0"],
        max_heavy_per_gpu=2,
        resource_class="heavy_gpu_candidate",
        capacity_slots=2.0,
        metadata={"resource_class": "heavy_gpu_candidate"},
    )
    assert ok is True, first

    released = store.release_idle(job_id="job-a", resident_mem_gb=1.25, elapsed_sec=120, reason="idle_probe")
    assert released["released"] is True
    snapshot = store.snapshot_active()
    assert "job-a" not in snapshot["leases"]
    assert snapshot["released_idle"]["job-a"]["metadata"]["released_idle_gpu_resident_mem_gb"] == 1.25

    ok, blocked = store.try_acquire(
        job_id="job-b",
        worker_id="W01",
        gpu_ids=["0"],
        max_heavy_per_gpu=2,
        resource_class="heavy_gpu_candidate",
        capacity_slots=2.0,
        metadata={"resource_class": "heavy_gpu_candidate"},
    )
    assert ok is False
    details = blocked["blocker_details"]["0"]
    assert details["idle_release_admission_mode"] == "strict_exclusive"
    assert details["released_idle_owners"] == ["job-a"]
    assert details["released_idle_gpu_resident_mem_gb"] == 1.25
    assert "released_idle_residue_reserved" in details["reasons"]

    ok, reacquired = store.try_acquire(
        job_id="job-a",
        worker_id="W00",
        gpu_ids=["0"],
        max_heavy_per_gpu=2,
        resource_class="heavy_gpu_candidate",
        capacity_slots=2.0,
        metadata={"resource_class": "heavy_gpu_candidate"},
    )
    assert ok is True
    assert reacquired["reason"] == "reacquired_released_idle"
    snapshot = store.snapshot_active()
    assert "job-a" in snapshot["leases"]
    assert "job-a" not in snapshot["released_idle"]

    final = store.release(job_id="job-a")
    assert final["released"] is True
    ok, acquired = store.try_acquire(
        job_id="job-b",
        worker_id="W01",
        gpu_ids=["0"],
        max_heavy_per_gpu=2,
        resource_class="heavy_gpu_candidate",
        capacity_slots=2.0,
        metadata={"resource_class": "heavy_gpu_candidate"},
    )
    assert ok is True, acquired


def test_quick_probe_classifier_detects_small_feature_extract() -> None:
    facts = classify_quick_probe_command(
        "python3 extract_features.py # quick test: extract features for 20 products",
        expected_runtime_sec=300,
        hard_review_sec=900,
    )

    assert facts.candidate is True
    assert facts.reason == "small_feature_extract"
    assert facts.scope_count == 20
    assert facts.output_pattern == "terminal_write_expected"


def test_quick_probe_classifier_reads_small_entrypoint_scope(tmp_path) -> None:
    (tmp_path / "extract_data.py").write_text(
        "# quick sample extractor\nN_PRODUCTS = 100\nprint('extracting sample')\n",
        encoding="utf-8",
    )

    facts = classify_quick_probe_command(
        "python3 extract_data.py",
        expected_runtime_sec=300,
        hard_review_sec=600,
        workspace_dir=tmp_path,
    )

    assert facts.candidate is True
    assert facts.reason == "small_scope_entrypoint"
    assert facts.scope_count == 100
    assert facts.hard_review_sec == 600


def test_quick_probe_classifier_does_not_flag_large_bounded_extract(tmp_path) -> None:
    (tmp_path / "extract_data.py").write_text(
        "\"\"\"Extract 100K subset images from BSON and build a training dataset.\"\"\"\nN_PRODUCTS = 100000\n",
        encoding="utf-8",
    )

    facts = classify_quick_probe_command(
        "python3 extract_data.py",
        expected_runtime_sec=300,
        hard_review_sec=600,
        workspace_dir=tmp_path,
    )

    assert facts.candidate is False
    assert facts.hard_review_sec == 600


def test_quick_probe_classifier_does_not_flag_plain_training() -> None:
    facts = classify_quick_probe_command("python3 train.py --epochs 20 --folds 5")

    assert facts.candidate is False


def test_quick_probe_runtime_exceeded_forces_arbiter_review(tmp_path) -> None:
    sm = FakeStateMachine()
    (tmp_path / "extract_features.py").write_text("print('done')\n", encoding="utf-8")
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        quick_probe_guard_enabled=True,
        quick_probe_expected_runtime_sec=30.0,
        quick_probe_hard_review_sec=60.0,
    )
    job_id = observer.job_created(
        command="python3 extract_features.py # quick test: extract features for 20 products",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    observer._jobs[job_id].visible = True
    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=75.0,
        stdout_age_sec=75.0,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
    )

    assert decision["arbiter_review"] is True
    assert decision["reason"] == "quick_probe_runtime_exceeded"
    proposal = decision["proposal"]
    assert proposal["proposal_type"] == "quick_probe_review"
    assert proposal["severity"] == "red"
    assert proposal["requires_llm_decision"] is True
    assert proposal["progress_snapshot"]["quick_probe_candidate"] is True
    assert proposal["control_generation_key"]
    assert payloads(sm, "resource_quick_probe_detected")
    assert "quick_probe_forced_review" in _resource_event_types(tmp_path)


def test_context_hygiene_requires_evidence_not_only_safety() -> None:
    decision = evaluate_context_hygiene_compact(
        stage_count_since_last_compact=30,
        total_input_tokens_since_last_compact=10_000_000,
        cache_rates=[0.95, 0.94, 0.93, 0.92, 0.91],
        large_file_touch_counts={},
        large_tool_output_count=0,
        seconds_since_last_compact=3600.0,
        active_high_risk_bash=False,
        snapshot_can_preserve_current_state=True,
    )

    assert decision.should_compact is False
    assert decision.facts["primary_gate"] is True
    assert decision.facts["evidence_gate"] is False


def test_context_hygiene_compacts_on_low_cache_with_safety() -> None:
    decision = evaluate_context_hygiene_compact(
        stage_count_since_last_compact=25,
        total_input_tokens_since_last_compact=6_000_000,
        cache_rates=[0.3, 0.4, 0.5, 0.6, 0.7],
        large_file_touch_counts={},
        large_tool_output_count=0,
        seconds_since_last_compact=3600.0,
        active_high_risk_bash=False,
        snapshot_can_preserve_current_state=True,
    )

    assert decision.should_compact is True
    assert "low_incremental_cache" in decision.facts["evidence_reasons"]


def test_context_hygiene_blocks_when_snapshot_cannot_preserve_state() -> None:
    decision = evaluate_context_hygiene_compact(
        stage_count_since_last_compact=25,
        total_input_tokens_since_last_compact=6_000_000,
        cache_rates=[0.3, 0.4, 0.5, 0.6, 0.7],
        large_file_touch_counts={},
        large_tool_output_count=0,
        seconds_since_last_compact=3600.0,
        active_high_risk_bash=False,
        snapshot_can_preserve_current_state=False,
    )

    assert decision.should_compact is False
    assert decision.facts["safety"]["snapshot_can_preserve_current_state"] is False
