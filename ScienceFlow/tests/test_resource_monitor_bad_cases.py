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

from scienceflow.safety.resource.review_boundary import RESOURCE_PRESSURE, TIMEBOX_EXPIRED
from scienceflow.safety.resource.review_outcome import TIMEBOX
from scienceflow.safety.resource.review_signal import build_review_signal
from scienceflow.safety.resource.review_state import (
    TIMEBOX_ACTIVE,
    ResourceReviewConfig,
    advance_review_state,
    apply_timebox,
    compute_timebox_sec,
    new_review_state,
    next_review_boundary,
)
from scienceflow.solver.lnr.resource_runtime.review.arbiter import normalize_arbiter_decision


def _state_with_metric_cadence(interval_sec: float = 1200.0):
    cfg = ResourceReviewConfig(warmup_windows=0)
    state = new_review_state("job-long-epoch")
    state = advance_review_state(
        state,
        build_review_signal(
            elapsed_sec=interval_sec,
            metric_history_text="val_auc=0.800",
            previous_metric_history_text="",
            process_tree_cpu={"total_cpu_pct": 200.0},
        ),
        config=cfg,
    )
    state = advance_review_state(
        state,
        build_review_signal(
            elapsed_sec=interval_sec * 2,
            metric_history_text="val_auc=0.805",
            previous_metric_history_text="val_auc=0.800",
            process_tree_cpu={"total_cpu_pct": 200.0},
        ),
        config=cfg,
    )
    return state


def test_bad_case_long_epoch_is_not_killed_by_short_review_horizon() -> None:
    """A 20-minute/epoch job should not fail a 10-minute value-review horizon."""

    state = _state_with_metric_cadence(interval_sec=1200.0)
    computed = compute_timebox_sec(
        state,
        clear_on="metric_update",
        heartbeat_sec=60.0,
        fallback_next_review_sec=300.0,
        max_timebox_sec=1800.0,
        remaining_budget_sec=12 * 3600.0,
        timebox_budget_fraction=0.10,
        active_waiter_pressure=False,
    )
    assert computed == 1800.0

    cfg = ResourceReviewConfig(warmup_windows=0, timebox_windows=10)
    state = apply_timebox(state, timebox_sec=computed, heartbeat_sec=60.0, clear_on="metric_update")
    active = build_review_signal(elapsed_sec=60.0, process_tree_cpu={"total_cpu_pct": 200.0})

    for _ in range(10):
        state = advance_review_state(state, active, config=cfg)
        assert next_review_boundary(state, config=cfg) is None
        assert state.job_state_bucket == TIMEBOX_ACTIVE

    for _ in range(20):
        state = advance_review_state(state, active, config=cfg)
    boundary = next_review_boundary(state, config=cfg)
    assert boundary and boundary.kind == TIMEBOX_EXPIRED


def test_bad_case_min_timebox_bound_is_configurable() -> None:
    """Very small remaining budget still respects the configured timebox lower bound."""

    state = _state_with_metric_cadence(interval_sec=1200.0)
    computed = compute_timebox_sec(
        state,
        clear_on="metric_update",
        heartbeat_sec=60.0,
        fallback_next_review_sec=300.0,
        min_timebox_sec=120.0,
        max_timebox_sec=1800.0,
        remaining_budget_sec=5 * 60.0,
        timebox_budget_fraction=0.10,
        active_waiter_pressure=False,
    )

    assert computed == 120.0


def test_bad_case_remaining_budget_shrinks_timebox_near_deadline() -> None:
    """The same signal cadence gets a shorter observation window near deadline."""

    state = _state_with_metric_cadence(interval_sec=1200.0)
    early = compute_timebox_sec(
        state,
        clear_on="metric_update",
        heartbeat_sec=60.0,
        fallback_next_review_sec=300.0,
        max_timebox_sec=1800.0,
        remaining_budget_sec=12 * 3600.0,
        timebox_budget_fraction=0.10,
        active_waiter_pressure=False,
    )
    late = compute_timebox_sec(
        state,
        clear_on="metric_update",
        heartbeat_sec=60.0,
        fallback_next_review_sec=300.0,
        max_timebox_sec=1800.0,
        remaining_budget_sec=30 * 60.0,
        timebox_budget_fraction=0.10,
        active_waiter_pressure=False,
    )

    assert early == 1800.0
    assert late == 180.0
    assert late < early


def test_bad_case_waiter_caps_long_timebox_to_avoid_starvation() -> None:
    """A long healthy holder should not get a cadence-sized window while a waiter is blocked."""

    state = _state_with_metric_cadence(interval_sec=1200.0)
    computed = compute_timebox_sec(
        state,
        clear_on="metric_update",
        heartbeat_sec=60.0,
        fallback_next_review_sec=300.0,
        max_timebox_sec=1800.0,
        remaining_budget_sec=12 * 3600.0,
        timebox_budget_fraction=0.10,
        active_waiter_pressure=True,
    )
    assert computed == 300.0


def test_bad_case_new_waiter_interrupts_existing_no_contention_timebox() -> None:
    """A waiter appearing after a long timebox starts is a resource-pressure event."""

    cfg = ResourceReviewConfig(warmup_windows=0, timebox_windows=30)
    state = apply_timebox(
        new_review_state("job-holder"),
        timebox_sec=1800.0,
        heartbeat_sec=60.0,
        clear_on="metric_update",
        active_waiter_pressure=False,
    )
    waiter = build_review_signal(
        elapsed_sec=60.0,
        process_tree_cpu={"total_cpu_pct": 200.0},
        active_waiter_pressure=True,
        blocked_worker_count=1,
    )
    state = advance_review_state(state, waiter, config=cfg)
    boundary = next_review_boundary(state, config=cfg)
    assert boundary and boundary.kind == RESOURCE_PRESSURE


def test_bad_case_checkpoint_growth_does_not_clear_metric_wait_timebox() -> None:
    """A checkpoint write is not proof for a metric-update timebox."""

    cfg = ResourceReviewConfig(warmup_windows=0, timebox_windows=3)
    state = apply_timebox(
        new_review_state("job-checkpoint"),
        timebox_sec=180.0,
        heartbeat_sec=60.0,
        clear_on="metric_update",
    )
    checkpoint_only = build_review_signal(
        elapsed_sec=60.0,
        artifact_recent=True,
        artifact_grew=True,
        process_tree_cpu={"total_cpu_pct": 200.0},
    )
    state = advance_review_state(state, checkpoint_only, config=cfg)
    assert state.job_state_bucket == TIMEBOX_ACTIVE
    assert state.timebox_success_condition == "metric_update"


def test_bad_case_bad_timebox_schema_fails_closed_to_no_action() -> None:
    """Malformed TIMEBOX output must not create an unbounded or ambiguous observation window."""

    decision = normalize_arbiter_decision(
        {
            "outcome": "TIMEBOX",
            "reason_code": "plateau_needs_observation",
            "reason": "Need more observation, but the clear condition is missing.",
            "confidence": "medium",
        },
        proposal={"proposal_id": "bad-schema", "proposal_type": "kill_proposal"},
    )
    assert decision["canonical_outcome"] == TIMEBOX
    assert decision["clear_on"] == "useful_progress"
    assert decision["clear_on_defaulted"] is True
    assert decision["schema_violation"] is True


def test_bad_case_llm_timebox_duration_is_ignored() -> None:
    """The LLM may choose what to observe, but runtime owns the observation duration."""

    decision = normalize_arbiter_decision(
        {
            "outcome": "TIMEBOX",
            "reason_code": "plateau_needs_observation",
            "reason": "Observe the next metric line.",
            "confidence": "medium",
            "clear_on": "metric_update",
            "timebox_sec": 999999,
        },
        proposal={"proposal_id": "bad-duration", "proposal_type": "kill_proposal"},
    )
    assert decision["canonical_outcome"] == TIMEBOX
    assert decision["clear_on"] == "metric_update"
    assert decision["ignored_timebox_sec"] == 999999
