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

import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import ModuleType

import pytest

from scienceflow.core.task_package import find_task_package
from scienceflow.gates import GateService
from scienceflow.gates.evaluator import EvalContext, EvaluationRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "tasks" / "sci_modeling_bench" / "tfbind8-black-box-v1"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_public_dataset(adapter: ModuleType, root: Path) -> None:
    data_path = root / "offline_data.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text("sequence,normalized_e_score\nAAAAAAAA,0.0\n", encoding="utf-8")
    manifest = {
        "task_id": adapter.TASK_ID,
        "repo_id": adapter.REPO_ID,
        "config_name": adapter.CONFIG_NAME,
        "split": adapter.SPLIT,
        "revision": adapter.REVISION,
        "protocol_id": adapter.PROTOCOL_ID,
        "offline_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class _FakeOfficialResult:
    def model_dump_json(self) -> str:
        return json.dumps(
            {
                "task_id": "design-bench/tfbind8-black-box-optimization-v1",
                "submission_valid": True,
                "metrics": {"top_1_normalized_e_score": 0.75},
                "expected_candidates": 128,
                "submitted_candidates": 128,
                "valid_candidates": 127,
                "invalid_candidates": 1,
                "all_candidates_valid": False,
                "best_candidate_index": 4,
                "best_objective_output": {"normalized_e_score": 0.75},
                "aggregation": "max",
            }
        )


class _FakeOfficialTask:
    def evaluate(self, candidates: list[object]) -> _FakeOfficialResult:
        assert len(candidates) == 128
        return _FakeOfficialResult()


def test_tfbind8_task_package_contract() -> None:
    spec = find_task_package("sci-modeling-bench-tfbind8-v1")

    assert spec is not None
    assert spec.category == "scientific_design"
    assert spec.provider == "sci_modeling_bench"
    assert spec.profile == "scientific_design"
    assert spec.artifact_path == "artifacts/submission.json"
    assert spec.artifact_kind == "candidate_batch"
    assert spec.metric_name == "top_1_normalized_e_score"
    assert spec.lower_is_better is False
    assert spec.metric_authoritative is True


def test_tfbind8_adapter_normalizes_official_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _load_module("tfbind8_adapter_unit", TASK_DIR / "evaluator.py")
    _write_public_dataset(adapter, tmp_path / "dataset")
    artifact = tmp_path / "artifacts" / "submission.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps({"candidates": [{"sequence": "AAAAAAAA"}] * 128}),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "_load_task", lambda: _FakeOfficialTask())

    payload = adapter.evaluate(
        artifact_path=artifact,
        workspace_dir=tmp_path,
        task_dir=TASK_DIR,
        dataset_dir=tmp_path / "dataset",
        config={},
    )

    assert payload["valid"] is True
    assert payload["candidate_ready"] is True
    assert payload["metric"] == {
        "name": "top_1_normalized_e_score",
        "value": 0.75,
        "lower_is_better": False,
    }
    assert payload["extra"]["valid_candidates"] == 127
    assert payload["extra"]["invalid_candidates"] == 1
    assert payload["extra"]["dataset_revision"] == adapter.REVISION


def test_tfbind8_adapter_rejects_tampered_public_data(tmp_path: Path) -> None:
    adapter = _load_module("tfbind8_adapter_tamper", TASK_DIR / "evaluator.py")
    dataset_dir = tmp_path / "dataset"
    _write_public_dataset(adapter, dataset_dir)
    (dataset_dir / "offline_data.csv").write_text("tampered\n", encoding="utf-8")
    artifact = tmp_path / "artifacts" / "submission.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("[]", encoding="utf-8")

    payload = adapter.evaluate(
        artifact_path=artifact,
        workspace_dir=tmp_path,
        task_dir=TASK_DIR,
        dataset_dir=dataset_dir,
        config={},
    )

    assert payload["valid"] is False
    assert payload["reason_code"] == "tfbind8_input_invalid"
    assert "does not match" in payload["feedback"]


@pytest.mark.skipif(
    os.environ.get("SCIENCEFLOW_RUN_SCI_MODELING_BENCH_INTEGRATION") != "1",
    reason="requires the official package and pinned Hugging Face dataset",
)
def test_tfbind8_official_evaluator_through_unified_service(tmp_path: Path) -> None:
    prepare_module = _load_module("tfbind8_prepare_integration", TASK_DIR / "prepare_data.py")
    prepared_root = tmp_path / "prepared"
    summary = prepare_module.prepare(prepared_root)

    workspace = tmp_path / "workspace"
    shutil.copytree(prepared_root / "public", workspace / "dataset")
    artifact = workspace / "artifacts" / "submission.json"
    artifact.parent.mkdir(parents=True)
    shutil.copy2(summary["baseline_submission"], artifact)

    context = EvalContext(
        task_profile="scientific_design",
        task_id="sci-modeling-bench-tfbind8-v1",
        task_root=tmp_path / "runtime",
        workspace=workspace,
        worker_id="W00",
        stage_id="S01",
        cfg={},
    )
    outcomes = GateService.default().evaluate(
        EvaluationRequest(context=context, trigger="stage_end")
    )

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.event.evaluator_backend == "task_package"
    assert outcome.event.metric_name == "top_1_normalized_e_score"
    assert outcome.event.metric_value == pytest.approx(
        summary["visible_max_normalized_e_score"]
    )
    assert outcome.event.validation_ok is True
    assert outcome.event.extra["valid_candidates"] == 128
    assert outcome.event.extra["dataset_revision"] == (
        "2ee2856f4255bb6a64c11b6c2660a6f41418e654"
    )
    assert outcome.decision.action == "accept"
