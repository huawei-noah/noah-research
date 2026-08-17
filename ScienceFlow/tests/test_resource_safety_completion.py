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

from pathlib import Path

from scienceflow.core.tools.resource_classifier import RESOURCE_HEAVY_CPU_CANDIDATE
from scienceflow.safety.resource.completion import deliverable_completion_state
from scienceflow.safety.resource.signals import scan_output_health
from tests.lnr_resource_test_utils import make_observer


def _heavy_cpu_job(observer, tmp_path: Path) -> str:
    (tmp_path / "solve.py").write_text("print('solve')\n", encoding="utf-8")
    job_id = observer.job_created(
        command="python3 solve.py",
        inferred_class=RESOURCE_HEAVY_CPU_CANDIDATE,
        gpu_ids=[],
        timeout_sec=100.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    return job_id


def test_scan_output_health_detects_terminal_signal_without_bare_score_zero() -> None:
    health = scan_output_health("Final validation score: 0.713\nscore=0 can be valid for some optimizers")

    assert health["terminal_signal_events"] == 1
    assert "zero_score_events" not in health


def test_deliverable_completion_state_supports_train_only_artifacts(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "best_model.pkl").write_text("model", encoding="utf-8")

    state = deliverable_completion_state(
        tmp_path,
        terminal_signal_seen=True,
        terminal_signal_kind="final_metric",
        settle_sec=0.0,
    )

    assert state["complete"] is True
    assert state["mode"] == "train_artifact"
    assert state["artifact_path"] == "artifacts/best_model.pkl"


def test_deliverable_completion_state_requires_terminal_signal_for_train_artifact(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "best_model.pkl").write_text("model", encoding="utf-8")

    state = deliverable_completion_state(tmp_path, terminal_signal_seen=False, settle_sec=0.0)

    assert state["complete"] is False
    assert state["reason"] == "terminal_signal_missing"


def test_deliverable_completion_state_uses_configured_candidate_artifact(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "best_solution.json").write_text('{"x": [1, 2, 3]}', encoding="utf-8")

    state = deliverable_completion_state(
        tmp_path,
        terminal_signal_seen=False,
        settle_sec=0.0,
        candidate_artifact="artifacts/best_solution.json",
    )

    assert state["complete"] is True
    assert state["mode"] == "artifact"
    assert state["artifact_path"] == "artifacts/best_solution.json"
    assert state["reason"] == "candidate_artifact_complete"


def test_deliverable_completion_state_does_not_require_submission_for_configured_artifact(tmp_path) -> None:
    state = deliverable_completion_state(
        tmp_path,
        terminal_signal_seen=False,
        settle_sec=0.0,
        candidate_artifact="artifacts/best_solution.json",
    )

    assert state["complete"] is False
    assert state["reason"] == "missing_candidate_artifact"
    assert state["deliverable_validity"] == "none"


def test_deliverable_completion_guard_supports_train_only_resource_hold(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "best_model.pkl").write_text("model", encoding="utf-8")
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        kill_mode="auto",
        deliverable_completion_guard_enabled=True,
        deliverable_completion_warmup_sec=0.0,
        deliverable_completion_settle_sec=0.0,
        deliverable_completion_scan_interval_sec=0.0,
        deliverable_completion_quiet_sec=0.0,
    )
    job_id = _heavy_cpu_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=10.0,
        stdout_age_sec=120.0,
        stdout_lines=10,
        stdout_bytes=100,
        current_phase="training",
        terminal_signal_events=1,
        terminal_signal_kind="final_metric",
        last_terminal_signal_text="Final validation score: 0.71",
    )

    assert decision["terminate"] is False
    state = observer._jobs[job_id].last_deliverable_completion_check["state"]
    assert state["mode"] == "train_artifact"


def test_deliverable_completion_guard_supports_solver_solution_resource_hold(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "best_route.txt").write_text("1 2 3 4\n", encoding="utf-8")
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        kill_mode="auto",
        deliverable_completion_guard_enabled=True,
        deliverable_completion_warmup_sec=0.0,
        deliverable_completion_settle_sec=0.0,
        deliverable_completion_scan_interval_sec=0.0,
        deliverable_completion_quiet_sec=0.0,
    )
    job_id = _heavy_cpu_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=10.0,
        stdout_age_sec=120.0,
        stdout_lines=10,
        stdout_bytes=100,
        current_phase="optimization",
        terminal_signal_events=1,
        terminal_signal_kind="solver_terminal",
        last_terminal_signal_text="converged best objective 123.4 solution saved",
    )

    assert decision["terminate"] is False
    state = observer._jobs[job_id].last_deliverable_completion_check["state"]
    assert state["mode"] == "solver_solution"


def test_deliverable_preflight_uses_candidate_artifact_unlock_condition(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "best_solution.json").write_text("", encoding="utf-8")
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        min_register_sec=0.0,
        deliverable_completion_guard_enabled=True,
        deliverable_completion_settle_sec=0.0,
    )
    job_id = observer.job_created(
        command="python3 solve.py",
        inferred_class=RESOURCE_HEAVY_CPU_CANDIDATE,
        gpu_ids=[],
        timeout_sec=100.0,
        workspace_dir=tmp_path,
        candidate_artifact="artifacts/best_solution.json",
    )
    assert job_id is not None

    result = observer._completed_deliverable_preflight_gate(observer._jobs[job_id])

    assert result["blocked"] is True
    assert result["candidate_artifact"] == "artifacts/best_solution.json"
    assert "candidate_artifact_valid" in result["feedback"]
    assert "submission_schema_valid" not in result["feedback"]
