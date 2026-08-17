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
from types import SimpleNamespace

import pytest

from scienceflow.core.agent.run_control.embedded_fullrun import _evaluate_embedded_candidate
from scienceflow.safety.execution_policy import EnsureFullRunResult
from scienceflow.gates import GateService
from scienceflow.gates.evaluator import (
    EvalContext,
    EvaluationRequest,
    EvaluatorManager,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _evaluate(
    tmp_path: Path,
    *,
    task_id: str,
    task_profile: str,
    metric_event: dict[str, object] | None = None,
):
    ctx = EvalContext(
        task_profile=task_profile,
        task_id=task_id,
        task_root=tmp_path,
        workspace=tmp_path,
        worker_id="W00",
        stage_id="S01",
        cfg={},
        metadata={"metric_event": dict(metric_event or {})},
    )
    outcomes = GateService.default().evaluate(
        EvaluationRequest(context=ctx, trigger="stage_end")
    )
    assert len(outcomes) == 1
    return outcomes[0]


def test_unified_entry_accepts_nomad2018_submission(tmp_path: Path) -> None:
    _write(tmp_path / "dataset" / "sample_submission.csv", "id,target\n1,0\n")
    _write(tmp_path / "submission.csv", "id,target\n1,0.5\n")

    outcome = _evaluate(
        tmp_path,
        task_id="nomad2018-predict-transparent-conductors",
        task_profile="mlebench",
        metric_event={
            "metric_value": 0.5,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
        },
    )

    assert outcome.event.metric_name == "Final Validation Score"
    assert outcome.event.validation_ok is True
    assert outcome.decision.action == "accept"
    assert outcome.decision.selection_eligible is True


def test_stage_accepts_valid_complete_run_result_without_submission(tmp_path: Path) -> None:
    outcome = _evaluate(
        tmp_path,
        task_id="nomad2018-predict-transparent-conductors",
        task_profile="mlebench",
        metric_event={
            "metric_value": 0.5,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "validation_ok": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "val_score_type": "holdout",
            "execution_scale": "direct_full",
            "bash_cmd": "python3 solution.py",
            "solution_sha": "full-run-source",
        },
    )

    assert outcome.event.evaluator_backend == "result_signal"
    assert outcome.event.artifact_path == ""
    assert outcome.event.extra["artifact_ready"] is False
    assert outcome.decision.action == "accept"
    assert outcome.decision.candidate_ready is True


@pytest.mark.parametrize(
    ("bash_cmd", "execution_scale"),
    [
        ("QUICK_TEST_ROWS=20 python3 solution.py", "unknown"),
        ("python3 train.py", "pilot"),
    ],
)
def test_stage_rejects_non_full_result_signal(
    tmp_path: Path,
    bash_cmd: str,
    execution_scale: str,
) -> None:
    outcome = _evaluate(
        tmp_path,
        task_id="nomad2018-predict-transparent-conductors",
        task_profile="mlebench",
        metric_event={
            "metric_value": 0.5,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "validation_ok": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "val_score_type": "holdout",
            "execution_scale": execution_scale,
            "bash_cmd": bash_cmd,
        },
    )

    assert outcome.event.evaluator_backend == "result_signal"
    assert outcome.decision.action == "retry"
    assert outcome.decision.reason_code == "candidate_not_ready"
    assert outcome.decision.candidate_ready is False


def test_stage_does_not_infer_non_full_run_from_script_name(tmp_path: Path) -> None:
    outcome = _evaluate(
        tmp_path,
        task_id="nomad2018-predict-transparent-conductors",
        task_profile="mlebench",
        metric_event={
            "metric_value": 0.5,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "validation_ok": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "val_score_type": "holdout",
            "execution_scale": "unknown",
            "bash_cmd": "python3 train_fast.py",
        },
    )

    assert outcome.event.evaluator_backend == "result_signal"
    assert outcome.decision.action == "accept"
    assert outcome.decision.candidate_ready is True


def test_final_gate_does_not_use_result_signal_without_artifact(tmp_path: Path) -> None:
    ctx = EvalContext(
        task_profile="mlebench",
        task_id="nomad2018-predict-transparent-conductors",
        task_root=tmp_path,
        workspace=tmp_path,
        worker_id="W00",
        stage_id="S01",
        cfg={},
        metadata={
            "metric_event": {
                "metric_value": 0.5,
                "metric_name": "Final Validation Score",
                "lower_is_better": False,
                "validation_ok": True,
                "selection_eligible": True,
                "metric_validity": "high",
                "val_score_type": "holdout",
                "execution_scale": "direct_full",
                "bash_cmd": "python3 solution.py",
            }
        },
    )

    outcomes = GateService.default().evaluate(
        EvaluationRequest(context=ctx, trigger="global_merge")
    )

    assert outcomes == []


def test_unified_entry_accepts_ratio_minimization_candidate(tmp_path: Path) -> None:
    _write(
        tmp_path / "dataset" / "problem.json",
        json.dumps({"num_points": 3, "dimension": 2}),
    )
    _write(
        tmp_path / "artifacts" / "best_solution.json",
        json.dumps({"points": [[0, 0], [1, 0], [0.5, 0.8660254038]]}),
    )

    outcome = _evaluate(
        tmp_path,
        task_id="ratio-minimization",
        task_profile="opt_solver",
    )

    assert outcome.event.metric_name == "inv_ratio_squared"
    assert outcome.event.metric_value == pytest.approx(1.0)
    assert outcome.event.evaluator_backend == "task_package"
    assert outcome.decision.action == "accept"
    assert outcome.event.extra["gate"]["policy"] == "default"
    assert outcome.event.extra["gate"]["version"] == "1"
    assert outcome.event.extra["gate"]["decision"]["action"] == "accept"


def test_unified_entry_normalizes_backend_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(
        tmp_path / "artifacts" / "best_solution.json",
        json.dumps({"points": [[0, 0], [1, 0], [0.5, 0.8660254038]]}),
    )
    manager = EvaluatorManager.default()
    backend = manager.get("task_package")
    assert backend is not None

    def _raise(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("simulator unavailable")

    monkeypatch.setattr(backend, "evaluate", _raise)
    ctx = EvalContext(
        task_profile="opt_solver",
        task_id="ratio-minimization",
        task_root=tmp_path,
        workspace=tmp_path,
        worker_id="W00",
        stage_id="S01",
        cfg={},
    )

    outcomes = GateService(manager).evaluate(
        EvaluationRequest(context=ctx, trigger="stage_end")
    )

    assert len(outcomes) == 1
    assert outcomes[0].event.evaluator_status == "evaluator_exception"
    assert outcomes[0].decision.action == "retry"
    assert outcomes[0].decision.reason_code == "validation_failed"


def test_embedded_fullrun_uses_unified_service_for_nomad2018(tmp_path: Path) -> None:
    _write(tmp_path / "dataset" / "sample_submission.csv", "id,target\n1,0\n")
    _write(tmp_path / "submission.csv", "id,target\n1,0.5\n")
    agent = SimpleNamespace(
        _workspace_dir=tmp_path,
        _scienceflow_evaluation_service=GateService.default(),
        _scienceflow_evaluation_cfg=SimpleNamespace(
            exp_id="nomad2018-predict-transparent-conductors"
        ),
        _scienceflow_task_root=tmp_path,
        _scienceflow_task_profile="mlebench",
        _scienceflow_worker_id="W00",
        _mlebench_exp_id="nomad2018-predict-transparent-conductors",
    )
    result = EnsureFullRunResult(
        executed=True,
        skipped=False,
        reason="test",
        exit_code=0,
        metric_value=0.5,
        metric_name="Final Validation Score",
        lower_is_better=False,
        stdout="Final Validation Score: 0.5",
    )

    outcome = _evaluate_embedded_candidate(agent, result)

    assert outcome is not None
    assert outcome.event.validation_ok is True
    assert outcome.event.metric_value == 0.5
    assert outcome.decision.action == "accept"
