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

from dataclasses import replace

from scienceflow.core.tools.bash_tool import (
    _kill_revalidation_allows_termination,
    _resource_call,
)
from scienceflow.solver.lnr.resource_runtime.review.kill_intent import (
    build_kill_intent_snapshot,
    revalidate_kill_intent,
)
from tests.lnr_resource_test_utils import make_observer


def _snapshot(
    *,
    scope: str = "route|validation|fold0|auc|group_cv",
    metric_value: float = 0.80,
    metric_useful: bool | None = False,
    metric_lines: int = 1,
    recoverable: bool = False,
    artifact_path: str = "",
    progress_done: float = 2.0,
    near_submission: bool = False,
    kill_basis: str = "",
) -> dict:
    progress = {
        "metric_scope_key": scope,
        "metric_history_line_count": metric_lines,
        "recoverable_artifact_on_disk": recoverable,
        "recoverability": {"artifact_path": artifact_path},
        "progress_unit": "epoch",
        "progress_units_done": progress_done,
        "progress_units_total": 10.0,
        "structured_progress_recent": True,
        "known_stage": "validation",
        "near_submission": near_submission,
    }
    metric = {
        "metric_scope_key": scope,
        "value": metric_value,
        "status": "above_live_best" if metric_useful else "below_live_best",
        "useful": metric_useful,
    }
    return build_kill_intent_snapshot(progress, metric, kill_basis=kill_basis)


def test_useful_metric_after_decision_invalidates_kill_intent() -> None:
    original = _snapshot()
    fresh = _snapshot(metric_value=0.83, metric_useful=True, metric_lines=2)

    result = revalidate_kill_intent(original, fresh)

    assert result["allow_kill"] is False
    assert result["protective_changes"] == ["useful_metric_advanced"]


def test_worse_metric_and_ordinary_snapshot_change_do_not_block_kill() -> None:
    original = _snapshot()
    fresh = _snapshot(metric_value=0.76, metric_useful=False, metric_lines=2)

    result = revalidate_kill_intent(original, fresh)

    assert result["allow_kill"] is True
    assert result["protective_changes"] == []


def test_new_scope_checkpoint_and_progress_are_protective() -> None:
    original = _snapshot()
    fresh = _snapshot(
        scope="route|validation|fold1|auc|group_cv",
        recoverable=True,
        artifact_path="models/fold1.pt",
        progress_done=3.0,
    )

    result = revalidate_kill_intent(original, fresh)

    assert result["allow_kill"] is False
    assert "metric_scope_changed" in result["protective_changes"]
    assert "recoverable_checkpoint_created" in result["protective_changes"]
    assert "structured_progress_advanced" in result["protective_changes"]


def test_value_stagnation_kill_rebases_liveness_progress() -> None:
    original = _snapshot(kill_basis="value_stagnation")
    fresh = _snapshot(
        recoverable=True,
        artifact_path="models/latest.pt",
        progress_done=3.0,
        kill_basis="value_stagnation",
    )

    result = revalidate_kill_intent(original, fresh)

    assert result["allow_kill"] is True
    assert result["reason"] == "kill_intent_rebased"
    assert result["protective_changes"] == []
    assert result["rebased_changes"] == [
        "recoverable_checkpoint_created",
        "structured_progress_advanced",
    ]


def test_value_stagnation_kill_still_protects_metric_improvement() -> None:
    original = _snapshot(kill_basis="value_stagnation")
    fresh = _snapshot(
        metric_value=0.83,
        metric_useful=True,
        metric_lines=2,
        progress_done=3.0,
        kill_basis="value_stagnation",
    )

    result = revalidate_kill_intent(original, fresh)

    assert result["allow_kill"] is False
    assert result["protective_changes"] == ["useful_metric_advanced"]
    assert result["rebased_changes"] == ["structured_progress_advanced"]


def test_hard_safety_and_legacy_snapshot_preserve_existing_kill_behavior() -> None:
    fresh = _snapshot(metric_value=0.83, metric_useful=True, metric_lines=2)

    legacy = revalidate_kill_intent({}, fresh)
    hard_safety = revalidate_kill_intent(_snapshot(), fresh, hard_safety=True)

    assert legacy == {
        "allow_kill": True,
        "reason": "legacy_snapshot_missing",
        "protective_changes": [],
    }
    assert hard_safety == {
        "allow_kill": True,
        "reason": "hard_safety",
        "protective_changes": [],
    }


def test_observer_revalidation_denies_stale_kill_and_clears_proof(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
    )
    job_id = observer.job_created(
        command="python3 train.py",
        inferred_class="heavy_gpu_train",
        gpu_ids=["0"],
        workspace_dir=tmp_path,
    )
    assert job_id

    base_signals = {
        "phase": "validation",
        "heartbeat": {
            "phase": "validation",
            "route_id": "cnn_v2",
            "fold": "0",
            "validation_protocol": "group_cv",
        },
        "metrics": {"val_auc": 0.80},
        "epoch": {"current": 2, "total": 10},
    }
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=120,
        phase="validation",
        signals=base_signals,
        stdout_lines=10,
        stdout_bytes=400,
        metric_history_text="val_auc=0.80",
        metric_history_line_count=1,
    )
    job = observer._jobs[job_id]
    initial_signal = observer._intervention_signal(
        elapsed_sec=120,
        stdout_age_sec=1,
        current_phase="validation",
        metric_history_text="val_auc=0.80",
        metric_history_line_count=1,
    )
    observer._attach_resource_metric_value_assessment(job, initial_signal)
    original = build_kill_intent_snapshot(
        observer._progress_snapshot_for_job(job, initial_signal, elapsed_sec=120),
        initial_signal["resource_metric_value"],
    )
    observer._review_states[job_id] = replace(
        observer._review_states[job_id],
        failed_proof_window_count=2,
        proof_window_index=2,
    )

    improved_signals = dict(base_signals)
    improved_signals["metrics"] = {"val_auc": 0.85}
    improved_signals["epoch"] = {"current": 3, "total": 10}
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=180,
        phase="validation",
        signals=improved_signals,
        stdout_lines=12,
        stdout_bytes=500,
        metric_history_text="val_auc=0.80\nval_auc=0.85",
        metric_history_line_count=2,
    )
    result = observer.revalidate_kill_intent(
        job_id,
        arbiter_decision={
            "kill_intent_snapshot": original,
            "gate": {"kill_class": "discretionary"},
        },
        elapsed_sec=180,
        stdout_age_sec=1,
        stdout_lines=12,
        stdout_bytes=500,
        metric_history_text="val_auc=0.80\nval_auc=0.85",
        metric_history_line_count=2,
        saw_training_progress=True,
        current_phase="validation",
    )

    assert result["allow_kill"] is False
    assert "useful_metric_advanced" in result["protective_changes"]
    assert observer._review_states[job_id].failed_proof_window_count == 0
    assert observer._review_states[job_id].last_outcome == "STALE_KILL_INTENT"


def test_discretionary_kill_revalidation_fails_closed() -> None:
    assert _kill_revalidation_allows_termination(None, hard_safety=False) is False
    assert _kill_revalidation_allows_termination({}, hard_safety=False) is False
    assert (
        _kill_revalidation_allows_termination(
            {"enabled": False, "allow_kill": True},
            hard_safety=False,
        )
        is False
    )
    assert (
        _kill_revalidation_allows_termination(
            {"enabled": True, "allow_kill": False},
            hard_safety=False,
        )
        is False
    )
    assert (
        _kill_revalidation_allows_termination(
            {"enabled": True, "allow_kill": True},
            hard_safety=False,
        )
        is True
    )


def test_hard_safety_kill_does_not_depend_on_revalidation_availability() -> None:
    assert _kill_revalidation_allows_termination(None, hard_safety=True) is True


def test_kill_intent_snapshot_tolerates_malformed_legacy_values() -> None:
    snapshot = build_kill_intent_snapshot(
        {
            "metric_history_line_count": "not-an-int",
            "recoverable_artifact_on_disk": "false",
            "structured_progress_recent": "yes",
            "near_submission": "no",
            "progress_units_done": "inf",
        },
        {"value": "nan"},
    )

    assert snapshot["metric_history_line_count"] == 0
    assert snapshot["recoverable_artifact_on_disk"] is False
    assert snapshot["structured_progress_recent"] is True
    assert snapshot["near_submission"] is False
    assert snapshot["progress_units_done"] is None
    assert snapshot["metric_value"] is None


def test_observer_revalidation_exception_cannot_authorize_discretionary_kill() -> None:
    class BrokenObserver:
        def revalidate_kill_intent(self):
            raise RuntimeError("transient observer failure")

    result = _resource_call(BrokenObserver(), "revalidate_kill_intent")

    assert result is None
    assert _kill_revalidation_allows_termination(result, hard_safety=False) is False
