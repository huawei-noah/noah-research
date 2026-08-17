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
import shlex
import sys
from pathlib import Path

from scienceflow.gates.evaluator import EvalContext, EvaluatorManager
from scienceflow.gates.evaluator.backends.command_env import task_command_env


def _ctx(tmp_path: Path, cfg: dict) -> EvalContext:
    return EvalContext(
        task_profile="opt_solver",
        task_id="toy",
        task_root=tmp_path,
        workspace=tmp_path,
        worker_id="W00",
        stage_id="S01",
        cfg=cfg,
    )


def _base_cfg(command: str) -> dict:
    return {
        "evaluator": {
            "backend": "artifact_command",
            "task_profile": "opt_solver",
            "candidate": {
                "artifact": "artifacts/best_solution.json",
                "artifact_kind": "json_solution",
                "require_sha": False,
            },
            "metric": {
                "name": "cost",
                "lower_is_better": True,
                "type": "benchmark",
                "json_path": "metric.value",
                "regex": "",
            },
            "command": {
                "evaluator_command": command,
                "timeout_sec": 5,
            },
        }
    }


def test_artifact_command_backend_parses_json_metric(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "best_solution.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"route": [1, 2]}', encoding="utf-8")
    py = shlex.quote(sys.executable)
    command = f"{py} -c 'import json; print(json.dumps({{\"metric\": {{\"value\": 12.5}}}}))'"

    events = EvaluatorManager.default().evaluate_workspace(_ctx(tmp_path, _base_cfg(command)))

    assert len(events) == 1
    event = events[0]
    assert event.evaluator_backend == "artifact_command"
    assert event.evaluator_status == "ok"
    assert event.task_profile == "opt_solver"
    assert event.metric_value == 12.5
    assert event.lower_is_better is True
    assert event.selection_eligible is True
    assert event.artifact_sha
    assert event.extra["artifact_kind"] == "json_solution"


def test_artifact_command_backend_parses_regex_metric(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "best_solution.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    py = shlex.quote(sys.executable)
    cfg = _base_cfg(f"{py} -c 'print(\"score: 0.77\")'")
    cfg["evaluator"]["metric"]["json_path"] = ""
    cfg["evaluator"]["metric"]["regex"] = r"score:\s*([0-9.]+)"
    cfg["evaluator"]["metric"]["lower_is_better"] = False

    events = EvaluatorManager.default().evaluate_workspace(_ctx(tmp_path, cfg))

    assert len(events) == 1
    assert events[0].metric_value == 0.77
    assert events[0].lower_is_better is False
    assert events[0].selection_eligible is True


def test_artifact_command_backend_missing_artifact_can_emit_rejection(tmp_path: Path) -> None:
    cfg = _base_cfg("python -c 'print(1)'")
    cfg["evaluator"]["candidate"]["emit_missing_artifact_event"] = True

    events = EvaluatorManager.default().evaluate_workspace(_ctx(tmp_path, cfg))

    assert len(events) == 1
    assert events[0].evaluator_status == "missing_artifact"
    assert events[0].selection_eligible is False
    assert events[0].metric_validity == "low"


def test_artifact_command_backend_rejects_sha_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "best_solution.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    (tmp_path / ".candidate_commit.json").write_text(
        json.dumps({"artifact_path": "artifacts/best_solution.json", "artifact_sha": "wrong"}),
        encoding="utf-8",
    )
    py = shlex.quote(sys.executable)
    cfg = _base_cfg(f"{py} -c 'print(\"{{}}\")'")
    cfg["evaluator"]["candidate"]["commit_file"] = ".candidate_commit.json"
    cfg["evaluator"]["candidate"]["require_sha"] = True

    events = EvaluatorManager.default().evaluate_workspace(_ctx(tmp_path, cfg))

    assert len(events) == 1
    assert events[0].evaluator_status == "artifact_sha_mismatch"
    assert events[0].selection_eligible is False


def test_artifact_command_backend_reuses_sha_cache(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "best_solution.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    counter = tmp_path / "counter.txt"
    evaluator = tmp_path / "eval_counter.py"
    evaluator.write_text(
        """
from pathlib import Path

p = Path("counter.txt")
n = int(p.read_text()) if p.exists() else 0
p.write_text(str(n + 1))
print('{"metric": {"value": 3.0}}')
""",
        encoding="utf-8",
    )
    py = shlex.quote(sys.executable)
    command = f"{py} {shlex.quote(str(evaluator))}"
    ctx = _ctx(tmp_path, _base_cfg(command))

    first = EvaluatorManager.default().evaluate_workspace(ctx)
    second = EvaluatorManager.default().evaluate_workspace(ctx)

    assert first[0].metric_value == 3.0
    assert second[0].metric_value == 3.0
    assert counter.read_text(encoding="utf-8") == "1"


def test_artifact_command_backend_uses_configured_python_executable(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "best_solution.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    cfg = _base_cfg("{python} -c 'import json, sys; print(json.dumps({\"metric\": {\"value\": 9.0}, \"python\": sys.executable}))'")
    cfg["evaluator"]["command"]["python_executable"] = sys.executable

    events = EvaluatorManager.default().evaluate_workspace(_ctx(tmp_path, cfg))

    assert len(events) == 1
    assert events[0].metric_value == 9.0
    assert events[0].evaluator_status == "ok"


def test_artifact_command_backend_derives_python_from_environment_path(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "best_solution.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    cfg = _base_cfg("{python} -c 'import json; print(json.dumps({\"metric\": {\"value\": 8.0}}))'")
    cfg["evaluator"]["command"]["environment_path"] = str(Path(sys.executable).parent.parent)

    events = EvaluatorManager.default().evaluate_workspace(_ctx(tmp_path, cfg))

    assert len(events) == 1
    assert events[0].metric_value == 8.0
    assert events[0].evaluator_status == "ok"


def test_task_command_env_replaces_controller_virtualenv(monkeypatch, tmp_path: Path) -> None:
    env_root = tmp_path / "conda_env"
    python = env_root / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv("VIRTUAL_ENV", "/controller/.venv")
    monkeypatch.setenv("CONDA_PREFIX", "/controller/conda")

    env = task_command_env(_ctx(tmp_path, _base_cfg("{python} -c pass")), python=python)

    assert env["CONDA_PREFIX"] == str(env_root)
    assert "VIRTUAL_ENV" not in env
    assert env["PATH"].split(":")[0] == str(python.parent)
