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

from dataclasses import replace

from scienceflow.safety.resource.review_boundary import PROGRESS_WINDOW, RESOURCE_PRESSURE, ROUTE_VALUE, STALL, TIMEBOX_EXPIRED
from scienceflow.safety.resource.review_outcome import KILL, NO_ACTION, TIMEBOX, normalize_value_review_outcome
from scienceflow.safety.resource.review_signal import build_review_signal
from scienceflow.safety.resource.review_state import (
    TIMEBOX_ACTIVE,
    ResourceReviewConfig,
    advance_review_state,
    apply_no_action,
    apply_timebox,
    compute_timebox_sec,
    mark_review_emitted,
    new_review_state,
    next_review_boundary,
)
from scienceflow.solver.lnr.resource_runtime.review.arbiter import normalize_arbiter_decision

from tests.lnr_resource_test_utils import make_observer


def test_warmup_active_no_progress_does_not_trigger_route_value() -> None:
    cfg = ResourceReviewConfig(warmup_windows=10, inactive_windows=3, value_windows=5)
    state = new_review_state("job-1")
    signal = build_review_signal(elapsed_sec=60, process_tree_cpu={"total_cpu_pct": 300.0})

    for _ in range(5):
        state = advance_review_state(state, signal, config=cfg)

    assert state.no_useful_progress_windows == 0
    assert next_review_boundary(state, config=cfg) is None


def test_terminal_signal_only_resets_progress_on_new_event() -> None:
    state = new_review_state("job-terminal-edge")
    first = build_review_signal(
        elapsed_sec=60,
        terminal_signal_events=1,
        previous_terminal_signal_events=state.last_terminal_signal_events,
    )
    state = advance_review_state(state, first, config=ResourceReviewConfig(warmup_windows=0))
    repeated = build_review_signal(
        elapsed_sec=120,
        terminal_signal_events=1,
        previous_terminal_signal_events=state.last_terminal_signal_events,
    )

    assert first.terminal_signal_advanced is True
    assert first.useful_progress is True
    assert first.proof_reset_progress is True
    assert state.last_terminal_signal_events == 1
    assert repeated.terminal_signal_advanced is False
    assert repeated.useful_progress is False
    assert repeated.proof_reset_progress is False


def test_timebox_counts_active_no_useful_progress_until_expiry() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3, value_windows=2, timebox_windows=4)
    state = new_review_state("job-1")
    signal = build_review_signal(elapsed_sec=60, process_tree_cpu={"total_cpu_pct": 300.0})

    for _ in range(2):
        state = advance_review_state(state, signal, config=cfg)
    boundary = next_review_boundary(state, config=cfg)
    assert boundary and boundary.kind == ROUTE_VALUE

    state = apply_no_action(state, observe_more_sec=240, heartbeat_sec=60)
    assert state.job_state_bucket == TIMEBOX_ACTIVE
    frozen = state.no_useful_progress_windows

    for _ in range(3):
        state = advance_review_state(state, signal, config=cfg)
        boundary = next_review_boundary(state, config=cfg)
        assert boundary is None
        assert state.no_useful_progress_windows > frozen
        frozen = state.no_useful_progress_windows

    state = advance_review_state(state, signal, config=cfg)
    boundary = next_review_boundary(state, config=cfg)
    assert boundary and boundary.kind == TIMEBOX_EXPIRED


def test_metric_timebox_ignores_unrelated_artifact_growth() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3, timebox_windows=3)
    state = apply_timebox(new_review_state("job-1"), timebox_sec=180, heartbeat_sec=60, clear_on="metric_update")

    artifact_only = build_review_signal(
        elapsed_sec=60,
        artifact_recent=True,
        artifact_grew=True,
        process_tree_cpu={"total_cpu_pct": 100.0},
    )
    state = advance_review_state(state, artifact_only, config=cfg)

    assert state.job_state_bucket == TIMEBOX_ACTIVE
    assert state.timebox_success_condition == "metric_update"


def test_progress_advance_timebox_clears_on_monotonic_structured_progress() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3, timebox_windows=3)
    state = apply_timebox(new_review_state("job-1"), timebox_sec=180, heartbeat_sec=60, clear_on="progress_advance")

    signal = build_review_signal(
        elapsed_sec=60,
        process_tree_cpu={"total_cpu_pct": 100.0},
        structured_progress_advanced=True,
        metric_value_useful=True,
    )
    state = advance_review_state(state, signal, config=cfg)

    assert state.job_state_bucket != TIMEBOX_ACTIVE
    assert state.timebox_id == ""


def test_unknown_metric_structured_progress_is_not_useful_progress() -> None:
    signal = build_review_signal(
        elapsed_sec=60,
        process_tree_cpu={"total_cpu_pct": 100.0},
        structured_progress_advanced=True,
        metric_value_useful=None,
    )

    assert signal.active_work is True
    assert "structured_progress" in signal.active_channels
    assert signal.useful_progress is False


def test_observer_review_signal_ignores_unadvanced_structured_heartbeat(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        resource_control_profile="high",
        kill_mode="arbiter",
        arbiter_enabled=True,
        low_progress_enabled=True,
        review_warmup_windows=10,
        review_value_windows=5,
        stalled_stdout_sec=600.0,
    )
    job_id = observer.job_created(
        command="python3 scan.py",
        inferred_class="light_cpu",
        gpu_ids=["0"],
        workspace_dir=tmp_path,
    )
    assert job_id

    observer.progress_heartbeat(
        job_id,
        elapsed_sec=60,
        phase="analysis",
        signals={"phase": "analysis", "items": {"current": 0, "total": 103, "source": "tqdm"}},
        stdout_lines=13,
        stdout_bytes=656,
    )
    observer.active_intervention_decision(
        job_id,
        elapsed_sec=180.0,
        stdout_age_sec=5.0,
        stdout_lines=13,
        stdout_bytes=656,
        saw_training_progress=True,
        current_phase="analysis",
        process_tree_cpu={"total_cpu_pct": 680.0, "busy_child_count": 1},
    )

    review_signal = observer._jobs[job_id].last_signal["resource_review_signal"]
    assert review_signal["structured_progress_advanced"] is False
    assert "process" not in review_signal["active_channels"]



def test_below_best_metric_heartbeat_does_not_count_as_useful_progress() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=5, value_windows=2)
    state = new_review_state("job-1")

    for idx in range(2):
        signal = build_review_signal(
            elapsed_sec=(idx + 1) * 60,
            stdout_bytes=100 + idx,
            previous_stdout_bytes=idx,
            metric_history_text=f"SCIENCEFLOW_HB phase=val progress={idx + 1}/10 unit=nb kt=0.750",
            previous_metric_history_text=f"SCIENCEFLOW_HB phase=val progress={idx}/10 unit=nb kt=0.749",
            saw_training_progress=True,
            process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
            structured_progress_advanced=True,
            metric_value_useful=False,
            metric_value_status="below_observed_best",
            metric_value_delta_to_best=0.02,
        )
        assert signal.raw_metric_changed is True
        assert signal.metric_changed is False
        assert signal.useful_progress is False
        state = advance_review_state(state, signal, config=cfg)

    timeboxed = apply_timebox(new_review_state("job-2"), timebox_sec=180, heartbeat_sec=60, clear_on="progress_advance")
    timeboxed = advance_review_state(timeboxed, signal, config=cfg)
    assert timeboxed.job_state_bucket == TIMEBOX_ACTIVE

    boundary = next_review_boundary(state, config=cfg)
    assert boundary and boundary.kind == ROUTE_VALUE


def test_observer_marks_below_best_structured_metric_as_low_value(tmp_path) -> None:
    stage_csv = tmp_path / "lhr_stage_performance.csv"
    stage_csv.write_text(
        "row_order,candidate_id,worker_id,stage_id,metric_value,metric_name,lower_is_better,"
        "validation_ok,selection_eligible,metric_validity,candidate_ready,submission_status,val_score_type\n"
        "1,W00:L01:S01,W00,S01,0.780000,Final Validation Score,0,1,1,high,1,ready,holdout\n",
        encoding="utf-8",
    )
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        resource_control_profile="high",
        kill_mode="arbiter",
        arbiter_enabled=True,
        low_progress_enabled=True,
        review_warmup_windows=0,
        review_value_windows=2,
        stalled_stdout_sec=600.0,
    )
    job_id = observer.job_created(
        command="python3 validate.py",
        inferred_class="pure_tt_cpu",
        gpu_ids=["0"],
        workspace_dir=tmp_path,
    )
    assert job_id

    observer.progress_heartbeat(
        job_id,
        elapsed_sec=60,
        phase="val",
        signals={
            "phase": "val",
            "heartbeat": {"validation_protocol": "holdout"},
            "nb": {"current": 100, "total": 1000},
            "metrics": {"kt": 0.750},
        },
        stdout_lines=10,
        stdout_bytes=500,
    )
    observer.active_intervention_decision(
        job_id,
        elapsed_sec=120.0,
        stdout_age_sec=5.0,
        stdout_lines=10,
        stdout_bytes=500,
        saw_training_progress=True,
        current_phase="val",
        process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
    )

    signal = observer._jobs[job_id].last_signal
    assert signal["resource_metric_value"]["status"] == "below_observed_best"
    review_signal = signal["resource_review_signal"]
    assert review_signal["raw_metric_changed"] is True
    assert review_signal["metric_changed"] is False
    assert review_signal["useful_progress"] is False


def test_observer_marks_worse_live_metric_as_low_value_without_stage_score(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        resource_control_profile="high",
        kill_mode="arbiter",
        arbiter_enabled=True,
        low_progress_enabled=True,
        review_warmup_windows=0,
        review_value_windows=2,
        stalled_stdout_sec=600.0,
    )
    job_id = observer.job_created(
        command="python3 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        workspace_dir=tmp_path,
    )
    assert job_id

    observer.progress_heartbeat(
        job_id,
        elapsed_sec=60,
        phase="val",
        signals={"phase": "val", "epochs": {"current": 1, "total": 20}, "metrics": {"val_w_auc": 0.301356}},
        stdout_lines=10,
        stdout_bytes=500,
    )
    observer.active_intervention_decision(
        job_id,
        elapsed_sec=60.0,
        stdout_age_sec=5.0,
        stdout_lines=10,
        stdout_bytes=500,
        saw_training_progress=True,
        current_phase="val",
        process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
    )
    assert observer._jobs[job_id].last_signal["resource_metric_value"]["status"] == "live_metric_baseline_established"

    observer.progress_heartbeat(
        job_id,
        elapsed_sec=120,
        phase="val",
        signals={"phase": "val", "epochs": {"current": 2, "total": 20}, "metrics": {"val_w_auc": 0.299184}},
        stdout_lines=20,
        stdout_bytes=1000,
    )
    observer.active_intervention_decision(
        job_id,
        elapsed_sec=120.0,
        stdout_age_sec=5.0,
        stdout_lines=20,
        stdout_bytes=1000,
        saw_training_progress=True,
        current_phase="val",
        process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
    )

    signal = observer._jobs[job_id].last_signal
    assert signal["resource_metric_value"]["status"] == "below_live_best"
    assert signal["resource_metric_value"]["useful"] is False
    assert signal["resource_review_signal"]["metric_changed"] is False
    assert signal["resource_review_signal"]["useful_progress"] is False


def test_opportunity_timebox_clears_when_waiter_pressure_drops() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3, timebox_windows=3)
    state = apply_timebox(
        new_review_state("job-1"),
        timebox_sec=180,
        heartbeat_sec=60,
        clear_on="opportunity_cost_cleared",
        active_waiter_pressure=True,
    )

    cleared = build_review_signal(
        elapsed_sec=60,
        process_tree_cpu={"total_cpu_pct": 100.0},
        active_waiter_pressure=False,
        blocked_worker_count=0,
    )
    state = advance_review_state(state, cleared, config=cfg)

    assert state.job_state_bucket != TIMEBOX_ACTIVE
    assert state.timebox_id == ""


def test_new_waiter_pressure_interrupts_timebox_started_without_waiter() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3, timebox_windows=10)
    state = apply_timebox(new_review_state("job-1"), timebox_sec=600, heartbeat_sec=60, clear_on="metric_update")

    waiter_signal = build_review_signal(
        elapsed_sec=60,
        process_tree_cpu={"total_cpu_pct": 100.0},
        active_waiter_pressure=True,
        blocked_worker_count=1,
    )
    state = advance_review_state(state, waiter_signal, config=cfg)
    boundary = next_review_boundary(state, config=cfg)

    assert boundary and boundary.kind == RESOURCE_PRESSURE


def test_timebox_sizing_uses_contention_cap() -> None:
    state = new_review_state("job-1")
    state = advance_review_state(
        state,
        build_review_signal(
            elapsed_sec=1200,
            metric_history_text="val_auc=0.80",
            previous_metric_history_text="",
            process_tree_cpu={"total_cpu_pct": 100.0},
        ),
        config=ResourceReviewConfig(warmup_windows=0),
    )
    state = advance_review_state(
        state,
        build_review_signal(
            elapsed_sec=2400,
            metric_history_text="val_auc=0.81",
            previous_metric_history_text="val_auc=0.80",
            process_tree_cpu={"total_cpu_pct": 100.0},
        ),
        config=ResourceReviewConfig(warmup_windows=0),
    )

    no_contention = compute_timebox_sec(
        state,
        clear_on="metric_update",
        heartbeat_sec=60,
        fallback_next_review_sec=300,
        max_timebox_sec=1800,
        remaining_budget_sec=14400,
        active_waiter_pressure=False,
    )
    with_contention = compute_timebox_sec(
        state,
        clear_on="metric_update",
        heartbeat_sec=60,
        fallback_next_review_sec=300,
        max_timebox_sec=1800,
        remaining_budget_sec=14400,
        active_waiter_pressure=True,
    )

    assert no_contention > 300
    assert with_contention == 300


def test_metric_every_heartbeat_is_rate_limited_by_progress_window() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, progress_event_min_windows=3)
    state = new_review_state("job-1")

    for idx in range(2):
        signal = build_review_signal(
            elapsed_sec=(idx + 1) * 60,
            stdout_bytes=100 + idx,
            previous_stdout_bytes=idx,
            metric_history_text=f"val_auc={0.8 + idx * 0.01:.3f}",
            previous_metric_history_text=f"val_auc={0.79 + idx * 0.01:.3f}",
            process_tree_cpu={"total_cpu_pct": 100.0},
        )
        state = advance_review_state(state, signal, config=cfg)
        assert next_review_boundary(state, config=cfg) is None

    signal = build_review_signal(
        elapsed_sec=180,
        stdout_bytes=200,
        previous_stdout_bytes=150,
        metric_history_text="val_auc=0.830",
        previous_metric_history_text="val_auc=0.820",
        process_tree_cpu={"total_cpu_pct": 100.0},
    )
    state = advance_review_state(state, signal, config=cfg)
    boundary = next_review_boundary(state, config=cfg)
    assert boundary and boundary.kind == PROGRESS_WINDOW

    state = mark_review_emitted(state, boundary)
    assert state.progress_event_count_since_review == 0
    assert next_review_boundary(state, config=cfg) is None


def test_true_inactive_reaches_stall_boundary() -> None:
    cfg = ResourceReviewConfig(warmup_windows=10, inactive_windows=2)
    state = new_review_state("job-1")
    inactive = build_review_signal(elapsed_sec=60)
    state = advance_review_state(state, inactive, config=cfg)
    assert next_review_boundary(state, config=cfg) is None
    state = advance_review_state(state, inactive, config=cfg)
    boundary = next_review_boundary(state, config=cfg)
    assert boundary and boundary.kind == STALL


def test_value_review_no_action_suppresses_main_agent_feedback() -> None:
    outcome = normalize_value_review_outcome("OBSERVE_MORE", {"proposal_type": "kill_proposal"})
    assert outcome.outcome == NO_ACTION
    assert outcome.suppress_main_agent_feedback is True

    share_outcome = normalize_value_review_outcome("CONTINUE_SHARED_OBSERVE", {"proposal_type": "task_gpu_share_review"})
    assert share_outcome.outcome == NO_ACTION
    assert share_outcome.suppress_main_agent_feedback is False


def test_active_work_without_useful_progress_requires_arbiter_before_kill(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        min_register_sec=0.0,
        kill_mode="auto",
        arbiter_enabled=True,
        arbiter_mode="policy",
        review_warmup_windows=0,
        review_inactive_windows=5,
        review_value_windows=2,
    )
    job_id = observer.job_created(
        command="python3 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

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

    assert decision["enabled"] is True
    assert decision["terminate"] is False
    assert decision["would_terminate"] is True
    assert decision["requires_llm_decision"] is True
    assert decision["arbiter_enabled"] is True
    assert decision["resource_review_boundary"]["kind"] == ROUTE_VALUE
    assert decision.get("proposal", {}).get("proposal_type") == "kill_proposal"


def test_observer_review_state_advances_only_on_heartbeat(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        min_register_sec=0.0,
        kill_mode="auto",
        arbiter_enabled=True,
        arbiter_mode="policy",
        review_heartbeat_sec=60.0,
        review_warmup_windows=0,
        review_inactive_windows=5,
        review_value_windows=2,
    )
    job_id = observer.job_created(
        command="python3 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

    first = observer.active_intervention_decision(
        job_id,
        elapsed_sec=60.0,
        stdout_age_sec=60.0,
        stdout_lines=0,
        stdout_bytes=0,
        process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
    )
    assert not first.get("would_terminate")
    state = observer._review_states[job_id]
    assert state.heartbeat_index == 1

    too_soon = observer.active_intervention_decision(
        job_id,
        elapsed_sec=70.0,
        stdout_age_sec=70.0,
        stdout_lines=0,
        stdout_bytes=0,
        process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
    )
    assert not too_soon.get("would_terminate")
    assert observer._review_states[job_id].heartbeat_index == 1

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=120.0,
        stdout_age_sec=120.0,
        stdout_lines=0,
        stdout_bytes=0,
        process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
    )
    assert decision["resource_review_boundary"]["kind"] == ROUTE_VALUE
    assert observer._review_states[job_id].heartbeat_index == 2


def test_canonical_kill_outcome_accepts_action_alias() -> None:
    decision = normalize_arbiter_decision(
        {
            "outcome": "KILL_AND_REPLAN",
            "reason_code": "low_progress_stalled",
            "reason": "No useful progress and deterministic resource evidence supports stopping.",
            "confidence": "high",
        },
        proposal={"proposal_id": "p1", "proposal_type": "kill_proposal"},
    )

    assert decision["canonical_outcome"] == KILL
    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["original_action"] == "KILL_AND_REPLAN"
    assert decision["outcome_alias_normalized"] is True


def test_canonical_timebox_outcome_keeps_clear_on_without_timebox_sec() -> None:
    decision = normalize_arbiter_decision(
        {
            "outcome": "TIMEBOX",
            "reason_code": "plateau_needs_observation",
            "reason": "Metric progress is flat but active work continues.",
            "confidence": "medium",
            "clear_on": "metric_update",
            "timebox_sec": 9999,
        },
        proposal={"proposal_id": "p1", "proposal_type": "kill_proposal"},
    )

    assert decision["canonical_outcome"] == TIMEBOX
    assert decision["action"] == "OBSERVE_MORE"
    assert decision["clear_on"] == "metric_update"
    assert decision["ignored_timebox_sec"] == 9999


def test_canonical_timebox_accepts_progress_advance_clear_on() -> None:
    decision = normalize_arbiter_decision(
        {
            "outcome": "TIMEBOX",
            "reason_code": "predict_progress_needs_observation",
            "reason": "Observe monotonic prediction progress.",
            "confidence": "medium",
            "clear_on": "progress_advance",
        },
        proposal={"proposal_id": "progress-advance", "proposal_type": "kill_proposal"},
    )

    assert decision["canonical_outcome"] == TIMEBOX
    assert decision["clear_on"] == "progress_advance"
    assert not decision.get("clear_on_defaulted")


def test_canonical_timebox_missing_clear_on_defaults_to_useful_progress() -> None:
    decision = normalize_arbiter_decision(
        {
            "outcome": "TIMEBOX",
            "reason_code": "plateau_needs_observation",
            "reason": "Metric progress is flat but active work continues.",
            "confidence": "medium",
        },
        proposal={"proposal_id": "p1", "proposal_type": "kill_proposal"},
    )

    assert decision["canonical_outcome"] == TIMEBOX
    assert decision["action"] == "OBSERVE_MORE"
    assert decision["clear_on"] == "useful_progress"
    assert decision["clear_on_defaulted"] is True
    assert decision["schema_violation"] is True
    assert decision["confidence"] == "low"


def test_metric_clear_closes_timebox_but_does_not_reset_proof_debt() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3, timebox_windows=3)
    state = apply_timebox(new_review_state("job-1"), timebox_sec=180, heartbeat_sec=60, clear_on="metric_update")
    state = replace(state, failed_proof_window_count=2, proof_window_index=2)

    signal = build_review_signal(
        elapsed_sec=60,
        metric_history_text="val_auc=0.800",
        previous_metric_history_text="val_auc=0.799",
        process_tree_cpu={"total_cpu_pct": 100.0},
    )
    state = advance_review_state(state, signal, config=cfg)

    assert state.job_state_bucket != TIMEBOX_ACTIVE
    assert state.failed_proof_window_count == 2


def test_hard_progress_resets_proof_debt() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3)
    state = replace(new_review_state("job-1"), failed_proof_window_count=2, proof_window_index=2)

    signal = build_review_signal(
        elapsed_sec=60,
        saw_final_score=True,
        metric_history_text="Final Validation Score: 0.900",
        previous_metric_history_text="",
        process_tree_cpu={"total_cpu_pct": 100.0},
    )
    state = advance_review_state(state, signal, config=cfg)

    assert state.failed_proof_window_count == 0
    assert state.proof_window_index == 0


def test_full_timebox_expiry_increments_proof_debt_once() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3, timebox_windows=1)
    state = apply_timebox(new_review_state("job-1"), timebox_sec=60, heartbeat_sec=60, clear_on="metric_update")
    signal = build_review_signal(elapsed_sec=60, process_tree_cpu={"total_cpu_pct": 100.0})

    state = advance_review_state(state, signal, config=cfg)
    boundary = next_review_boundary(state, config=cfg)
    assert boundary and boundary.kind == TIMEBOX_EXPIRED

    state = mark_review_emitted(state, boundary)
    assert state.failed_proof_window_count == 1
    state = mark_review_emitted(state, boundary)
    assert state.failed_proof_window_count == 1


def test_no_action_resource_proof_window_can_count_failed_proof_debt() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3, timebox_windows=1)
    state = apply_no_action(
        new_review_state("job-1"),
        observe_more_sec=60,
        heartbeat_sec=60,
        counts_as_proof_failure=True,
    )
    signal = build_review_signal(elapsed_sec=60, process_tree_cpu={"total_cpu_pct": 100.0})

    state = advance_review_state(state, signal, config=cfg)
    boundary = next_review_boundary(state, config=cfg)
    assert boundary and boundary.kind == TIMEBOX_EXPIRED

    state = mark_review_emitted(state, boundary)
    assert state.proof_window_source == "no_action"
    assert state.failed_proof_window_count == 1


def test_contention_shortened_timebox_expiry_does_not_count_route_failure() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3, timebox_windows=1)
    state = apply_timebox(
        new_review_state("job-1"),
        timebox_sec=60,
        heartbeat_sec=60,
        clear_on="metric_update",
        active_waiter_pressure=True,
    )
    signal = build_review_signal(
        elapsed_sec=60,
        process_tree_cpu={"total_cpu_pct": 100.0},
        active_waiter_pressure=True,
        blocked_worker_count=1,
    )

    state = advance_review_state(state, signal, config=cfg)
    boundary = next_review_boundary(state, config=cfg)
    assert boundary and boundary.kind == TIMEBOX_EXPIRED
    state = mark_review_emitted(state, boundary)

    assert state.timebox_counts_as_proof_failure is False
    assert state.failed_proof_window_count == 0


def test_metric_scope_change_resets_only_old_scope_proof_debt() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=3)
    state = replace(
        new_review_state("job-scope"),
        metric_scope_key="route_a|validation|fold_0|auc|group_cv",
        failed_proof_window_count=2,
        proof_window_index=2,
    )
    state = apply_timebox(state, timebox_sec=600, heartbeat_sec=60, clear_on="metric_update")

    same_scope = build_review_signal(
        elapsed_sec=60,
        metric_scope_key="route_a|validation|fold_0|auc|group_cv",
        process_tree_cpu={"total_cpu_pct": 100.0},
    )
    state = advance_review_state(state, same_scope, config=cfg)
    assert state.failed_proof_window_count == 2
    assert state.in_timebox is True

    next_scope = build_review_signal(
        elapsed_sec=120,
        metric_scope_key="route_a|validation|fold_1|auc|group_cv",
        process_tree_cpu={"total_cpu_pct": 100.0},
    )
    state = advance_review_state(state, next_scope, config=cfg)

    assert state.metric_scope_key.endswith("fold_1|auc|group_cv")
    assert state.metric_scope_change_count == 1
    assert state.failed_proof_window_count == 0
    assert state.proof_window_index == 0
    assert state.in_timebox is False
    assert state.no_useful_progress_windows == 1


def test_metric_scope_return_restores_scope_local_debt_and_cadence() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=99, value_windows=5)
    state = replace(
        new_review_state("job-scope-return"),
        metric_scope_key="route_a|validation|fold_0|auc|cv",
        no_useful_progress_windows=3,
        failed_proof_window_count=2,
        proof_window_index=2,
        proof_window_source="system_fixed",
        last_metric_update_elapsed_sec=600.0,
        metric_update_interval_sec=300.0,
    )

    switch_to_b = build_review_signal(
        elapsed_sec=1200.0,
        metric_scope_key="route_b|validation|fold_0|auc|cv",
        metric_value_useful=False,
        process_tree_cpu={"total_cpu_pct": 100.0},
    )
    state = advance_review_state(state, switch_to_b, config=cfg)

    assert state.failed_proof_window_count == 0
    assert state.no_useful_progress_windows == 1
    assert state.metric_update_interval_sec == 0.0
    assert state.last_metric_update_elapsed_sec == 1200.0

    state = replace(
        state,
        no_useful_progress_windows=2,
        failed_proof_window_count=1,
        proof_window_index=1,
        proof_window_source="negotiated",
        last_metric_update_elapsed_sec=1400.0,
        metric_update_interval_sec=200.0,
    )
    switch_back_to_a = build_review_signal(
        elapsed_sec=1600.0,
        metric_scope_key="route_a|validation|fold_0|auc|cv",
        metric_value_useful=False,
        process_tree_cpu={"total_cpu_pct": 100.0},
    )
    state = advance_review_state(state, switch_back_to_a, config=cfg)

    assert state.failed_proof_window_count == 2
    assert state.proof_window_index == 2
    assert state.proof_window_source == "system_fixed"
    assert state.no_useful_progress_windows == 4
    assert state.metric_update_interval_sec == 300.0
    assert state.last_metric_update_elapsed_sec == 1600.0


def test_metric_scope_memory_is_bounded_and_new_scope_churn_stops_resetting_debt() -> None:
    cfg = ResourceReviewConfig(warmup_windows=0, inactive_windows=99, value_windows=5)
    state = new_review_state("job-scope-churn")

    for index in range(40):
        signal = build_review_signal(
            elapsed_sec=float((index + 1) * 60),
            metric_scope_key=f"route_{index}|validation|fold_0|auc|cv",
            metric_value_useful=False,
            process_tree_cpu={"total_cpu_pct": 100.0},
        )
        state = advance_review_state(state, signal, config=cfg)

    boundary = next_review_boundary(state, config=cfg)
    assert len(state.metric_scope_memory) == 32
    assert state.no_useful_progress_windows >= cfg.value_windows
    assert boundary is not None
    assert boundary.kind == ROUTE_VALUE
