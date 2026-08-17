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
from types import SimpleNamespace

import pytest

from scienceflow.core.task_package import (
    description_path_for_task,
    find_task_package,
    prepare_task_runtime,
)
from scienceflow.gates.evaluator import EvalContext, EvaluatorManager
from scienceflow.gates.evaluator.backends.task_package import _runner_config


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_task_package_registry_finds_mlebench_description() -> None:
    spec = find_task_package("nomad2018-predict-transparent-conductors")

    assert spec is not None
    assert spec.category == "ml"
    assert spec.provider == "mlebench"
    assert spec.artifact_path == "submission.csv"
    assert description_path_for_task(spec.task_id) == spec.source_dir / "description_lite.md"


def test_task_runtime_copy_registers_shared_evaluator(tmp_path: Path) -> None:
    runtime = prepare_task_runtime("nomad2018-predict-transparent-conductors", task_root=tmp_path)

    assert runtime.runtime_root == (tmp_path / "task_runtime").resolve()
    assert runtime.runtime_task_dir.is_dir()
    assert runtime.entrypoint_path.name == "evaluator.py"
    assert runtime.entrypoint_sha256
    assert runtime.package_sha256
    assert runtime.manifest_path.is_file()
    assert not str(runtime.entrypoint_path).startswith(str(tmp_path / "workers"))


def test_task_package_merges_shared_defaults_with_task_overrides(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    _write(
        tasks_root / "ml" / "provider" / "_shared" / "task_defaults.yaml",
        """
category: ml
provider: provider
profile: shared-profile
artifact:
  path: shared.csv
  kind: shared_csv
metric:
  name: shared-score
  lower_is_better: null
  type: holdout
evaluator:
  entrypoint: ../_shared/evaluator.py:evaluate
""",
    )
    _write(
        tasks_root / "ml" / "provider" / "demo" / "task.yaml",
        """
id: demo
metric:
  lower_is_better: true
""",
    )

    spec = find_task_package("demo", tasks_root=tasks_root)

    assert spec is not None
    assert spec.profile == "shared-profile"
    assert spec.artifact_path == "shared.csv"
    assert spec.artifact_kind == "shared_csv"
    assert spec.metric_name == "shared-score"
    assert spec.metric_type == "holdout"
    assert spec.lower_is_better is True
    assert spec.evaluator_entrypoint == "../_shared/evaluator.py:evaluate"


def test_task_package_backend_evaluates_mlebench_submission(tmp_path: Path) -> None:
    _write(tmp_path / "dataset" / "sample_submission.csv", "id,target\n1,0\n")
    _write(tmp_path / "submission.csv", "id,target\n1,0.5\n")
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
            }
        },
    )

    events = EvaluatorManager.default().evaluate_workspace(ctx)

    assert len(events) == 1
    event = events[0]
    assert event.evaluator_backend == "task_package"
    assert event.validation_ok is True
    assert event.metric_value == 0.5
    assert event.extra["deliverable_role"] == "submission_csv"


def test_task_package_runner_passes_trusted_worker_query_context(tmp_path: Path) -> None:
    spec = SimpleNamespace(
        task_id="demo-task",
        profile="demo",
        config={"id": "demo-task"},
    )
    ctx = EvalContext(
        task_profile="mlebench",
        task_id=spec.task_id,
        task_root=tmp_path,
        workspace=tmp_path,
        worker_id="W01",
        stage_id="S03",
        cfg=SimpleNamespace(
            evaluator=SimpleNamespace(query_budget_scope="worker"),
        ),
    )

    config = _runner_config(ctx, spec, "package-sha")

    assert config["evaluation_context"] == {
        "worker_id": "W01",
        "stage_id": "S03",
        "query_budget_scope": "worker",
    }


def test_task_package_runner_rejects_worker_scope_without_worker_id(tmp_path: Path) -> None:
    spec = SimpleNamespace(
        task_id="demo-task",
        profile="demo",
        config={"id": "demo-task"},
    )
    ctx = EvalContext(
        task_profile="mlebench",
        task_id=spec.task_id,
        task_root=tmp_path,
        workspace=tmp_path,
        worker_id="",
        cfg=SimpleNamespace(
            evaluator=SimpleNamespace(query_budget_scope="worker"),
        ),
    )

    with pytest.raises(ValueError, match="requires a trusted worker_id"):
        _runner_config(ctx, spec, "package-sha")
