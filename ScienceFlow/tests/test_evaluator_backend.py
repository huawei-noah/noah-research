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

from scienceflow.gates.evaluator import EvalContext, EvaluatorManager


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ctx(tmp_path: Path, *, metric_value: float | None = 0.42) -> EvalContext:
    metric_event = {
        "metric_value": metric_value,
        "metric_name": "Final Validation Score",
        "lower_is_better": False,
        "val_score_type": "holdout",
        "wall_sec": 12.5,
    }
    return EvalContext(
        task_profile="mlebench",
        task_id="nomad2018-predict-transparent-conductors",
        task_root=tmp_path,
        workspace=tmp_path,
        worker_id="W00",
        stage_id="S01",
        cfg={},
        metadata={"metric_event": metric_event},
    )


def test_mlebench_backend_light_validation_emits_metric_event(tmp_path: Path) -> None:
    _write(tmp_path / "dataset" / "sample_submission.csv", "id,target\n1,0\n2,0\n")
    _write(tmp_path / "submission.csv", "id,target\n1,0.2\n2,0.8\n")

    events = EvaluatorManager.default().evaluate_workspace(_ctx(tmp_path))

    assert len(events) == 1
    event = events[0]
    assert event.candidate_id == "W00:S01"
    assert event.artifact_path == "submission.csv"
    assert event.artifact_sha
    assert event.metric_value == 0.42
    assert event.lower_is_better is False
    assert event.validation_ok is True
    assert event.candidate_ready is True
    assert event.selection_eligible is True
    assert event.metric_validity == "high"
    assert event.evaluator_backend == "task_package"
    assert event.extra["artifact_kind"] == "submission_csv"
    assert event.extra["deliverable_role"] == "submission_csv"


def test_mlebench_backend_invalid_submission_is_not_selection_eligible(tmp_path: Path) -> None:
    _write(tmp_path / "dataset" / "sample_submission.csv", "id,target\n1,0\n2,0\n")
    _write(tmp_path / "submission.csv", "id,wrong\n1,0.2\n2,0.8\n")

    events = EvaluatorManager.default().evaluate_workspace(_ctx(tmp_path))

    assert len(events) == 1
    event = events[0]
    assert event.validation_ok is False
    assert event.candidate_ready is False
    assert event.selection_eligible is False
    assert event.metric_validity == "low"
    assert event.evaluator_status == "invalid_submission"
    assert event.metric_validity_reason_code == "invalid_submission"


def test_mlebench_backend_no_submission_emits_no_event(tmp_path: Path) -> None:
    _write(tmp_path / "dataset" / "sample_submission.csv", "id,target\n1,0\n")

    assert EvaluatorManager.default().evaluate_workspace(_ctx(tmp_path)) == []


def test_mlebench_backend_valid_submission_without_metric_is_observational(tmp_path: Path) -> None:
    _write(tmp_path / "dataset" / "sample_submission.csv", "id,target\n1,0\n")
    _write(tmp_path / "submission.csv", "id,target\n1,0.2\n")

    events = EvaluatorManager.default().evaluate_workspace(_ctx(tmp_path, metric_value=None))

    assert len(events) == 1
    assert events[0].validation_ok is True
    assert events[0].candidate_ready is True
    assert events[0].selection_eligible is False
    assert events[0].metric_validity == "medium"
    assert events[0].evaluator_status == "validated_no_metric"
