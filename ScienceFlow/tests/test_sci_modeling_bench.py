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

import copy
import importlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import pytest
import yaml
from datasets import Dataset as HFDataset
from sci_modeling_bench import (
    AgentInputBundle,
    AgentInputField,
    AgentInputManifest,
    AgentInputView,
)

from scienceflow.config.settings import load_cfg
from scienceflow.core.task_package import (
    find_task_package,
    iter_task_packages,
    prepare_task_runtime,
)
from scienceflow.gates import GateService
from scienceflow.gates.evaluator import EvalContext, EvaluationRequest
from scienceflow.core.parallel_runner import ParallelRunner
from scienceflow.solver.lnr.prompts import build_first_user_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = REPO_ROOT / "tasks" / "sci_modeling_bench"
SHARED_EVALUATOR = SUITE_ROOT / "_shared" / "evaluator.py"
SHARED_PREPARER = SUITE_ROOT / "_shared" / "prepare_data.py"
RUN_CONFIG_ROOT = REPO_ROOT / "scienceflow" / "config" / "runs" / "sci_modeling_bench"
PROFILE = REPO_ROOT / "scienceflow" / "config" / "sci_modeling_bench.yaml"

TASK_CONTRACTS = {
    "sci-modeling-bench-tfbind8": (32, 5, "best_k_mean", 10, "tfbind8"),
    "sci-modeling-bench-tfbind10-pho4": (
        128,
        16,
        "normalized_enrichment",
        10,
        "tfbind10-pho4",
    ),
    "sci-modeling-bench-superconductor": (32, 5, "global_ndcg", 10, "superconductor"),
    "sci-modeling-bench-gfp": (128, 16, "normalized_enrichment", 10, "gfp"),
    "sci-modeling-bench-utr-mrl": (128, 16, "normalized_enrichment", 5, "utr-mrl"),
    "sci-modeling-bench-hopper-controller": (32, 5, "global_ndcg", 5, "hopper-controller"),
    "sci-modeling-bench-drugmatrix-mchc": (16, 5, "global_ndcg", 4, "drugmatrix-mchc"),
    "sci-modeling-bench-drugmatrix-mch": (16, 5, "global_ndcg", 4, "drugmatrix-mch"),
    "sci-modeling-bench-drugmatrix-creatinine": (
        16,
        5,
        "global_ndcg",
        4,
        "drugmatrix-creatinine",
    ),
    "sci-modeling-bench-drugmatrix-sodium": (16, 5, "global_ndcg", 4, "drugmatrix-sodium"),
    "sci-modeling-bench-drugmatrix-chloride": (
        16,
        5,
        "global_ndcg",
        4,
        "drugmatrix-chloride",
    ),
    "sci-modeling-bench-drugmatrix-phosphorus": (
        16,
        5,
        "global_ndcg",
        4,
        "drugmatrix-phosphorus",
    ),
}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tfbind8_config(*, worker_id: str = "W00", max_queries: int = 1) -> dict[str, Any]:
    spec = find_task_package("sci-modeling-bench-tfbind8")
    assert spec is not None
    config = copy.deepcopy(dict(spec.config))
    config["evaluator"]["max_queries"] = max_queries
    config["evaluation_context"] = {
        "worker_id": worker_id,
        "stage_id": "S01",
        "query_budget_scope": "worker",
    }
    return config


def _manifest(config: Mapping[str, Any], *, num_rows: int = 1) -> AgentInputManifest:
    source = config["source"]
    candidate_field = config["submission"]["candidate_fields"][0]
    target_field = (
        "normalized_e_score"
        if source["config_name"] == "tfbind8"
        else "visible_score"
    )
    return AgentInputManifest(
        dataset_id=f"design-bench/{source['config_name']}",
        dataset_version="1.0.0",
        dataset_description="Tiny disclosure-scoped test input.",
        dataset_license="other",
        repo_id=source["repo_id"],
        resolved_revision=source["revision"],
        config_name=source["config_name"],
        split=source["split"],
        split_description="Tiny test split.",
        protocol_id=source["protocol_id"],
        task_id=source["benchmark_task_id"],
        views=(
            AgentInputView(
                name="observations",
                role="observations",
                description="Visible observations.",
                num_rows=num_rows,
                fields=(
                    AgentInputField(
                        name=candidate_field,
                        role="input",
                        description="Candidate input.",
                        physical_type="Value('string')",
                    ),
                    AgentInputField(
                        name=target_field,
                        role="target",
                        description="Visible score.",
                        physical_type="Value('float64')",
                    ),
                ),
            ),
        ),
    )


def _write_public_dataset(root: Path, config: Mapping[str, Any]) -> None:
    manifest = _manifest(config)
    view = root / "views" / "observations.parquet"
    view.parent.mkdir(parents=True, exist_ok=True)
    view.write_bytes(b"test parquet placeholder")
    (root / "dataset_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (root / "dataset_files.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": config["id"],
                "benchmark_task_id": config["source"]["benchmark_task_id"],
                "files": [
                    {
                        "view": "observations",
                        "role": "observations",
                        "path": "views/observations.parquet",
                        "sha256": "not-checked-by-default",
                        "num_rows": 1,
                        "columns": [field.name for field in manifest.views[0].fields],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "task_contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": config["id"],
                "benchmark_task_id": config["source"]["benchmark_task_id"],
                "artifact": config["artifact"],
                "submission": config["submission"],
                "metric": config["metric"],
                "evaluator_feedback": {
                    "max_queries": config["evaluator"]["max_queries"],
                    "feedback_metrics": config["evaluator"]["feedback_metrics"],
                    "candidate_outputs_exposed": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


class _FakeOfficialResult:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "task_id": "design-bench/tfbind8-black-box-optimization-v3",
            "submission_valid": True,
            "metrics": {
                "best_k_mean": 0.75,
                "best_k_mean_regret": 0.1,
                "global_ndcg": 0.8,
            },
            "metric_directions": {
                "best_k_mean": "maximize",
                "best_k_mean_regret": "minimize",
                "global_ndcg": "maximize",
            },
            "primary_metric": "best_k_mean",
            "metric_direction": "maximize",
            "expected_candidates": 32,
            "submitted_candidates": 32,
            "valid_candidates": 32,
            "invalid_candidates": 0,
            "all_candidates_valid": True,
            "evaluation_eligible": True,
            "summary_size": 5,
            "reference_scope": "full_domain",
            "reference_size": 65536,
            "submission_validation": {"violations": []},
            "candidates": [],
        }


class _FakeOfficialTask:
    task_id = "design-bench/tfbind8-black-box-optimization-v3"
    submission_size = 32
    summary_size = 5
    primary_metric = "best_k_mean"
    dataset = SimpleNamespace(
        resolved_revision="d8b9ea3c78ed33edf0869a427940a0651eb49f52",
        knowledge={},
    )

    def prepare(self) -> tuple[()]:
        return ()

    def evaluate(self, candidates: list[object]) -> _FakeOfficialResult:
        assert len(candidates) == 32
        return _FakeOfficialResult()


def _artifact(path: Path, sequence: str, *, compact: bool = False) -> Path:
    payload = {"candidates": [{"sequence": sequence}] * 32}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, separators=(",", ":") if compact else None),
        encoding="utf-8",
    )
    return path


def test_canonical_task_contracts_and_descriptions() -> None:
    specs = {
        spec.task_id: spec
        for spec in iter_task_packages()
        if spec.provider == "sci_modeling_bench"
    }
    assert set(specs) == set(TASK_CONTRACTS)
    assert find_task_package("sci-modeling-bench-tfbind8-v1") is None

    for task_id, (size, summary_size, metric, query_limit, _slug) in TASK_CONTRACTS.items():
        spec = specs[task_id]
        assert spec.category == "scientific_design"
        assert spec.profile == "sci_modeling_bench"
        assert spec.artifact_path == "artifacts/submission.json"
        assert spec.artifact_kind == "ordered_candidate_batch"
        assert spec.metric_authoritative is True
        assert spec.metric_name == metric
        assert spec.lower_is_better is False
        assert spec.evaluator_entrypoint == "../_shared/evaluator.py:evaluate"
        assert spec.config["source"]["package"] == "sci-modeling-bench==0.10.0"
        assert spec.config["submission"]["size"] == size
        assert spec.config["submission"]["summary_size"] == summary_size
        assert spec.config["evaluator"]["max_queries"] == query_limit
        module_name, class_name = spec.config["source"]["task_factory"].rsplit(
            ":", 1
        )
        factory = getattr(importlib.import_module(module_name), class_name)
        assert callable(getattr(factory, "from_hub", None))
        description = (spec.source_dir / spec.description_relpath).read_text(
            encoding="utf-8"
        )
        assert "dataset/dataset_manifest.json" in description
        assert "artifacts/submission.json" in description
        assert metric in description
        assert str(size) in description
        for field in spec.config["submission"]["candidate_fields"]:
            assert f'"{field}"' in description


def test_shared_runtime_and_canonical_run_configs(tmp_path: Path) -> None:
    runtime = prepare_task_runtime(
        "sci-modeling-bench-drugmatrix-mchc", task_root=tmp_path
    )
    assert runtime.entrypoint_path.parent.name == "_shared"
    assert runtime.runtime_task_dir.name == "drugmatrix-mchc-ranking"
    for name in ("common.py", "evaluator.py", "prepare_data.py", "query_budget.py"):
        assert (runtime.entrypoint_path.parent / name).is_file()

    profile = load_cfg(PROFILE, cli_args=False)
    assert profile.evaluator.query_budget_scope == "worker"
    assert profile.evaluator.stop_on_query_budget_exhausted is True
    assert profile.evaluator.expose_wall_clock_remaining_sec is True
    assert profile.lnr.final_artifact_mode == "best_stage"
    assert profile.lnr.protected_eda_mode == "agent"
    assert profile.lnr.stage_commit_context_mode == "inherit"
    assert profile.lnr.stage_commit_output_format == "json"

    manifests = sorted(RUN_CONFIG_ROOT.glob("*.yaml"))
    assert {path.stem for path in manifests} == {
        contract[4] for contract in TASK_CONTRACTS.values()
    }
    for path in manifests:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        task = data["tasks"][0]
        assert data["time_limit"] == 7800
        assert data["defaults"]["config"] == "scienceflow/config/sci_modeling_bench.yaml"
        assert data["defaults"]["tool"]["scienceflow_bash_timeout_sec"] == 600
        assert task["lnr"]["wall_clock_budget_sec"] == 7200
        assert task["lnr"]["num_workers"] == 2
        assert task["lnr"]["omp_threads_cap"] == 4
        assert task["lnr"]["resource_gpu_pool"] == []
        assert task["gpu_list"] == "cpu"
        assert "cpu_list" not in task
        assert "api_key" not in path.read_text(encoding="utf-8")

        dataset_dir = tmp_path / path.stem / "public"
        dataset_dir.mkdir(parents=True)
        task["input_data_dir"] = str(dataset_dir)
        runtime_manifest = tmp_path / f"run-{path.name}"
        runtime_manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
        runner = ParallelRunner(runtime_manifest, max_concurrent=1)
        assert len(runner._tasks) == 1
        assert runner._tasks[0].exp_id == task["exp_id"]


def test_sci_modeling_prompt_uses_candidate_design_contract() -> None:
    prompt = build_first_user_prompt(
        "Select candidates and save artifacts/submission.json.",
        wall_clock_budget_sec=7200,
        task_profile="sci_modeling_bench",
    )

    assert "offline scientific candidate-design task" in " ".join(prompt.split())
    assert "limited query budget" in prompt
    assert "dataset/" in prompt
    assert "ML-competition submission" in prompt


def test_shared_preparer_materializes_manifest_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _load_module("smb_shared_preparer_unit", SHARED_PREPARER)
    config = _tfbind8_config(max_queries=10)
    rows = {
        "sequence": [f"SEQ-{index:03d}" for index in range(40)],
        "normalized_e_score": [index / 40 for index in range(40)],
    }
    table = HFDataset.from_dict(rows)
    bundle = AgentInputBundle(data=table, manifest=_manifest(config, num_rows=40))
    fake_task = _FakeOfficialTask()
    fake_task.build_input = lambda: bundle  # type: ignore[attr-defined]
    monkeypatch.setattr(preparer, "load_official_task", lambda _config: fake_task)

    result = preparer.prepare(config["id"], tmp_path)

    assert result["public_input_cache_hit"] is False
    assert (tmp_path / "public" / "dataset_manifest.json").is_file()
    assert (tmp_path / "public" / "views" / "observations.parquet").is_file()
    validation = json.loads(
        (tmp_path / "validation" / "validation_submission.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(validation["candidates"]) == 32
    assert validation["candidates"][0] == {"sequence": "SEQ-039"}

    fake_task.build_input = lambda: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        AssertionError("valid cache must bypass build_input")
    )
    assert preparer.prepare(config["id"], tmp_path)["public_input_cache_hit"] is True


def test_shared_evaluator_uses_worker_budget_and_terminal_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _load_module("smb_shared_evaluator_unit", SHARED_EVALUATOR)
    config_w00 = _tfbind8_config(worker_id="W00", max_queries=1)
    config_w01 = _tfbind8_config(worker_id="W01", max_queries=1)
    dataset_dir = tmp_path / "dataset"
    _write_public_dataset(dataset_dir, config_w00)
    monkeypatch.setattr(
        evaluator,
        "_load_or_create_official_task",
        lambda **_kwargs: (_FakeOfficialTask(), False),
    )
    task_dir = tmp_path / "task_runtime" / "sci_modeling_bench" / "_shared"
    first = _artifact(tmp_path / "first.json", "AAAAAAAA")

    scored = evaluator.evaluate(
        artifact_path=first,
        workspace_dir=tmp_path,
        task_dir=task_dir,
        dataset_dir=dataset_dir,
        config=config_w00,
    )
    duplicate = evaluator.evaluate(
        artifact_path=_artifact(tmp_path / "duplicate.json", "AAAAAAAA", compact=True),
        workspace_dir=tmp_path,
        task_dir=task_dir,
        dataset_dir=dataset_dir,
        config=config_w00,
    )
    exhausted = evaluator.evaluate(
        artifact_path=_artifact(tmp_path / "second.json", "AAAAAAAC"),
        workspace_dir=tmp_path,
        task_dir=task_dir,
        dataset_dir=dataset_dir,
        config=config_w00,
    )
    independent = evaluator.evaluate(
        artifact_path=first,
        workspace_dir=tmp_path,
        task_dir=task_dir,
        dataset_dir=dataset_dir,
        config=config_w01,
    )

    assert scored["valid"] is True
    assert scored["metric"] == {
        "name": "best_k_mean",
        "value": 0.75,
        "lower_is_better": False,
    }
    assert scored["extra"]["queries_remaining"] == 0
    assert scored["extra"]["queries_terminal"] is True
    assert duplicate["extra"]["query_cache_hit"] is True
    assert duplicate["extra"]["queries_used"] == 1
    assert exhausted["valid"] is False
    assert exhausted["reason_code"] == "scientific_design_query_budget_exhausted"
    assert exhausted["extra"]["queries_terminal"] is True
    assert independent["valid"] is True


@pytest.mark.parametrize(
    ("task_id", "candidate"),
    [
        ("sci-modeling-bench-tfbind8", {"sequence": "AAAAAAAA"}),
        (
            "sci-modeling-bench-superconductor",
            {"composition": "YBa2Cu3O7"},
        ),
        (
            "sci-modeling-bench-drugmatrix-mchc",
            {"condition_id": "condition-001"},
        ),
    ],
)
def test_shared_evaluator_maps_supported_candidate_schemas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
    candidate: dict[str, Any],
) -> None:
    evaluator = _load_module(f"smb_schema_{task_id}", SHARED_EVALUATOR)
    spec = find_task_package(task_id)
    assert spec is not None
    config = copy.deepcopy(dict(spec.config))
    config["evaluator"]["max_queries"] = 1
    config["evaluation_context"] = {
        "worker_id": "W00",
        "stage_id": "S01",
        "query_budget_scope": "worker",
    }
    source = config["source"]
    submission = config["submission"]
    primary_metric = config["metric"]["name"]
    expected_size = submission["size"]

    class FakeResult:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            metrics = {
                "best_k_mean_regret": 0.1,
                "normalized_enrichment": 0.7,
                "global_ndcg": 0.8,
            }
            metrics[primary_metric] = 0.75
            metric_directions = {
                "best_k_mean_regret": "minimize",
                "normalized_enrichment": "maximize",
                "global_ndcg": "maximize",
            }
            metric_directions[primary_metric] = "maximize"
            return {
                "task_id": source["benchmark_task_id"],
                "submission_valid": True,
                "metrics": metrics,
                "metric_directions": metric_directions,
                "primary_metric": primary_metric,
                "metric_direction": "maximize",
                "expected_candidates": expected_size,
                "submitted_candidates": expected_size,
                "valid_candidates": expected_size,
                "invalid_candidates": 0,
                "all_candidates_valid": True,
                "evaluation_eligible": True,
                "summary_size": submission["summary_size"],
                "reference_scope": "test_reference",
                "reference_size": 1000,
                "submission_validation": {"violations": []},
                "candidates": [],
            }

    class FakeTask:
        task_id = source["benchmark_task_id"]
        submission_size = expected_size
        summary_size = submission["summary_size"]
        primary_metric = config["metric"]["name"]
        dataset = SimpleNamespace(resolved_revision=source["revision"], knowledge={})

        def evaluate(self, candidates: list[object]) -> FakeResult:
            assert candidates == [candidate] * expected_size
            return FakeResult()

    dataset_dir = tmp_path / "dataset"
    _write_public_dataset(dataset_dir, config)
    monkeypatch.setattr(
        evaluator,
        "_load_or_create_official_task",
        lambda **_kwargs: (FakeTask(), False),
    )
    artifact = tmp_path / "submission.json"
    artifact.write_text(
        json.dumps({"candidates": [candidate] * expected_size}),
        encoding="utf-8",
    )

    payload = evaluator.evaluate(
        artifact_path=artifact,
        workspace_dir=tmp_path,
        task_dir=tmp_path / "task_runtime" / task_id / "_shared",
        dataset_dir=dataset_dir,
        config=config,
    )

    assert payload["valid"] is True
    assert payload["metric"] == {
        "name": primary_metric,
        "value": 0.75,
        "lower_is_better": False,
    }
    assert payload["extra"]["official_task_id"] == source["benchmark_task_id"]


def test_shared_evaluator_rejects_manifest_identity_mismatch(tmp_path: Path) -> None:
    evaluator = _load_module("smb_shared_evaluator_tamper", SHARED_EVALUATOR)
    config = _tfbind8_config()
    dataset_dir = tmp_path / "dataset"
    _write_public_dataset(dataset_dir, config)
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_id"] = "tampered-protocol"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    payload = evaluator.evaluate(
        artifact_path=_artifact(tmp_path / "submission.json", "AAAAAAAA"),
        workspace_dir=tmp_path,
        task_dir=tmp_path / "task_runtime" / "sci_modeling_bench" / "_shared",
        dataset_dir=dataset_dir,
        config=config,
    )

    assert payload["valid"] is False
    assert payload["reason_code"] == "scientific_design_input_invalid"
    assert "protocol_id mismatch" in payload["feedback"]


def test_shared_evaluator_accepts_runner_symlinked_public_dataset(tmp_path: Path) -> None:
    evaluator = _load_module("smb_shared_evaluator_runner_links", SHARED_EVALUATOR)
    config = _tfbind8_config()
    source = tmp_path / "prepared" / "public"
    _write_public_dataset(source, config)

    dataset_dir = tmp_path / "workspace" / "dataset"
    dataset_dir.mkdir(parents=True)
    for child in source.iterdir():
        (dataset_dir / child.name).symlink_to(child.resolve())

    manifest = evaluator._validate_public_dataset(dataset_dir, config=config)

    assert manifest.task_id == config["source"]["benchmark_task_id"]


def test_shared_evaluator_rejects_dataset_view_parent_traversal(tmp_path: Path) -> None:
    evaluator = _load_module("smb_shared_evaluator_parent_traversal", SHARED_EVALUATOR)
    config = _tfbind8_config()
    dataset_dir = tmp_path / "dataset"
    _write_public_dataset(dataset_dir, config)
    files_path = dataset_dir / "dataset_files.json"
    files = json.loads(files_path.read_text(encoding="utf-8"))
    files["files"][0]["path"] = "../outside.parquet"
    files_path.write_text(json.dumps(files), encoding="utf-8")
    (tmp_path / "outside.parquet").write_bytes(b"not a dataset view")

    with pytest.raises(ValueError, match="escapes dataset directory"):
        evaluator._validate_public_dataset(dataset_dir, config=config)


@pytest.mark.skipif(
    os.environ.get("SCIENCEFLOW_RUN_SCI_MODELING_BENCH_INTEGRATION") != "1",
    reason="requires the official package and pinned Hugging Face dataset",
)
def test_tfbind8_official_evaluator_through_gate_service(tmp_path: Path) -> None:
    preparer = _load_module("smb_shared_preparer_integration", SHARED_PREPARER)
    prepared_root = tmp_path / "prepared"
    summary = preparer.prepare("sci-modeling-bench-tfbind8", prepared_root)

    workspace = tmp_path / "workspace"
    shutil.copytree(prepared_root / "public", workspace / "dataset")
    artifact = workspace / "artifacts" / "submission.json"
    artifact.parent.mkdir(parents=True)
    shutil.copy2(summary["validation_submission"], artifact)
    context = EvalContext(
        task_profile="sci_modeling_bench",
        task_id="sci-modeling-bench-tfbind8",
        task_root=tmp_path / "run",
        workspace=workspace,
        worker_id="W00",
        stage_id="S01",
        cfg={},
    )

    outcomes = GateService.default().evaluate(
        EvaluationRequest(context=context, trigger="stage_end")
    )

    assert len(outcomes) == 1
    event = outcomes[0].event
    assert event.evaluator_backend == "task_package"
    assert event.metric_name == "best_k_mean"
    assert event.validation_ok is True
    assert event.candidate_ready is True
    assert event.extra["package_version"] == "0.10.0"
