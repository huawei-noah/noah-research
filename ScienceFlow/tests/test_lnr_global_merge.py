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

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from scienceflow.gates.evaluator import EvaluatorManager
from scienceflow.solver.lnr.global_merge import runner as global_merge_runner
from scienceflow.solver.lnr.global_merge.candidate_pack import pack_candidates
from scienceflow.solver.lnr.global_merge.runner import (
    _evaluate_finals,
    run_global_merge,
)
from scienceflow.solver.lnr.global_merge.final_artifacts import (
    materialize_best_stage_final,
)
from scienceflow.solver.lnr.submission_links import (
    canonicalize_worker_workspace_artifacts,
    refresh_submission_links,
)
from scienceflow.solver.lnr.solver import LnrSolver


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialize_best_stage_final_selects_verified_historical_best(
    tmp_path: Path,
) -> None:
    low = tmp_path / "snap-low"
    high = tmp_path / "snap-high"
    _write(low / "artifacts" / "submission.json", '{"candidate":"low"}\n')
    _write(high / "artifacts" / "submission.json", '{"candidate":"high"}\n')
    candidates = [
        {
            "candidate_id": "W00:L01:S01",
            "node_uid": "W00:L01:S01",
            "snapshot_path": str(low),
            "metric_value": 0.4,
            "lower_is_better": False,
            "validation_ok": True,
            "candidate_ready": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "metric_authoritative": True,
            "artifact_sha": _sha256(low / "artifacts" / "submission.json"),
        },
        {
            "candidate_id": "W01:L02:S04",
            "node_uid": "W01:L02:S04",
            "snapshot_path": str(high),
            "metric_value": 0.9,
            "lower_is_better": False,
            "validation_ok": True,
            "candidate_ready": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "metric_authoritative": True,
            "artifact_sha": _sha256(high / "artifacts" / "submission.json"),
        },
    ]

    manifest = materialize_best_stage_final(
        merge_dir=tmp_path / "merge",
        candidates=candidates,
        artifact_path="artifacts/submission.json",
        ledger_filename=".run_results.md",
    )

    final = tmp_path / "merge" / "finals" / "final_00" / "artifacts" / "submission.json"
    assert manifest["status"] == "success"
    assert manifest["selected_candidate"]["node_uid"] == "W01:L02:S04"
    assert manifest["artifact_sha"] == candidates[1]["artifact_sha"]
    assert final.is_symlink()
    assert json.loads(final.read_text(encoding="utf-8"))["candidate"] == "high"
    assert json.loads(
        (tmp_path / "merge" / "best_stage_manifest.json").read_text(encoding="utf-8")
    )["selected_candidate"]["node_uid"] == "W01:L02:S04"


def test_materialize_best_stage_final_rejects_sha_mismatch(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write(snapshot / "submission.json", '{"candidate":"actual"}\n')

    manifest = materialize_best_stage_final(
        merge_dir=tmp_path / "merge",
        candidates=[
            {
                "candidate_id": "W00:L01:S01",
                "snapshot_path": str(snapshot),
                "metric_value": 0.9,
                "lower_is_better": False,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
                "metric_authoritative": True,
                "artifact_sha": "0" * 64,
            }
        ],
        artifact_path="submission.json",
        ledger_filename=".run_results.md",
    )

    assert manifest["status"] == "artifact_sha_mismatch"
    assert manifest["final_count"] == 0
    assert not (tmp_path / "merge" / "finals" / "final_00").exists()


def test_materialize_best_stage_final_supports_lower_metric_and_legacy_submission_sha(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write(first / "submission.csv", "id,value\n1,first\n")
    _write(second / "submission.csv", "id,value\n1,second\n")

    manifest = materialize_best_stage_final(
        merge_dir=tmp_path / "merge",
        candidates=[
            {
                "candidate_id": "W00:L01:S01",
                "snapshot_path": str(first),
                "metric_value": 0.3,
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
                "metric_authoritative": True,
                "submission_sha": _sha256(first / "submission.csv"),
            },
            {
                "candidate_id": "W00:L01:S02",
                "snapshot_path": str(second),
                "metric_value": 0.1,
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
                "metric_authoritative": True,
                "submission_sha": _sha256(second / "submission.csv"),
            },
        ],
        artifact_path="submission.csv",
        ledger_filename=".run_results.md",
    )

    assert manifest["status"] == "success"
    assert manifest["selected_candidate"]["candidate_id"] == "W00:L01:S02"


def test_materialize_best_stage_prefers_authoritative_direction_over_metric_name(
    tmp_path: Path,
) -> None:
    low = tmp_path / "low"
    high = tmp_path / "high"
    _write(low / "submission.json", '{"candidate":"low"}\n')
    _write(high / "submission.json", '{"candidate":"high"}\n')

    def candidate(candidate_id: str, snapshot: Path, metric: float) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "snapshot_path": str(snapshot),
            "metric_name": "validation_loss",
            "metric_value": metric,
            "lower_is_better": False,
            "validation_ok": True,
            "candidate_ready": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "metric_authoritative": True,
            "artifact_sha": _sha256(snapshot / "submission.json"),
        }

    manifest = materialize_best_stage_final(
        merge_dir=tmp_path / "merge",
        candidates=[
            candidate("W00:L01:S01", low, 0.1),
            candidate("W00:L01:S02", high, 0.9),
        ],
        artifact_path="submission.json",
        ledger_filename=".run_results.md",
    )

    assert manifest["status"] == "success"
    assert manifest["selected_candidate"]["candidate_id"] == "W00:L01:S02"


def test_materialize_best_stage_rejects_submission_sha_for_json_artifact(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _write(snapshot / "submission.json", '{"candidate":"value"}\n')

    manifest = materialize_best_stage_final(
        merge_dir=tmp_path / "merge",
        candidates=[
            {
                "candidate_id": "W00:L01:S01",
                "snapshot_path": str(snapshot),
                "metric_value": 0.9,
                "lower_is_better": False,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
                "metric_authoritative": True,
                "submission_sha": _sha256(snapshot / "submission.json"),
            }
        ],
        artifact_path="submission.json",
        ledger_filename=".run_results.md",
    )

    assert manifest["status"] == "no_valid_candidate"
    assert manifest["final_count"] == 0


def test_pack_candidates_links_artifact_metadata_and_ledger(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    _write(snap / "submission.csv", "id,target\n1,0.1\n")
    _write(snap / ".run_results.md", "# stage\n")

    packed = pack_candidates(
        merge_dir=tmp_path / "merge",
        merge_workspace=tmp_path / "merge_ws",
        candidates=[
            {
                "candidate_id": "W00:L01:S03",
                "snapshot_path": str(snap),
                "metric_value": 0.1,
                "submission_sha": "abc",
                "metric_validity": "high",
                "selection_eligible": True,
            }
        ],
        artifact_path="submission.csv",
        ledger_filename=".run_results.md",
    )

    assert len(packed) == 1
    assert (
        tmp_path / "merge" / "candidates" / "W00-L01-S03" / "submission.csv"
    ).is_symlink()
    assert (
        tmp_path / "merge_ws" / "candidates" / "W00-L01-S03" / "metadata.json"
    ).is_file()


def test_pack_candidates_excludes_low_confidence_candidates(tmp_path: Path) -> None:
    low = tmp_path / "low"
    high = tmp_path / "high"
    _write(low / "submission.csv", "id,target\n1,0.1\n")
    _write(high / "submission.csv", "id,target\n1,0.2\n")

    packed = pack_candidates(
        merge_dir=tmp_path / "merge",
        merge_workspace=tmp_path / "merge_ws",
        candidates=[
            {
                "candidate_id": "W00:S01",
                "snapshot_path": str(low),
                "metric_value": 0.0,
                "submission_sha": "low",
                "metric_validity": "low",
                "selection_eligible": False,
            },
            {
                "candidate_id": "W01:S02",
                "snapshot_path": str(high),
                "metric_value": 0.2,
                "submission_sha": "high",
                "metric_validity": "medium",
                "selection_eligible": True,
            },
        ],
        artifact_path="submission.csv",
        ledger_filename=".run_results.md",
    )

    assert [p["candidate_id"] for p in packed] == ["W01:S02"]
    assert not (tmp_path / "merge" / "candidates" / "W00-S01").exists()


def test_submission_links_include_worker_workspace_and_snapshots(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "workers" / "w00"
    _write(worker / "workspace" / "submission.csv", "id,target\n1,0.1\n")
    _write(
        worker / "snapshots" / "W00-L01-S01-abc" / "submission.csv",
        "id,target\n1,0.2\n",
    )
    _write(tmp_path / "submissions" / "submission.csv", "old\n")
    _write(tmp_path / "submissions" / "selected_submission.csv", "old\n")

    links = refresh_submission_links(
        submission_dir=tmp_path / "submissions",
        artifact_path="submission.csv",
        worker_roots=[worker],
    )

    assert len(links) == 2
    workspace_link = tmp_path / "submissions" / "workers" / "w00" / "workspace.csv"
    snapshot_link = (
        tmp_path
        / "submissions"
        / "workers"
        / "w00"
        / "snapshots"
        / "W00-L01-S01-abc.csv"
    )
    assert workspace_link.is_symlink()
    assert snapshot_link.is_symlink()
    assert not os.readlink(workspace_link).startswith("/")
    assert not os.readlink(snapshot_link).startswith("/")
    assert not (tmp_path / "submissions" / "submission.csv").exists()
    assert not (tmp_path / "submissions" / "selected_submission.csv").exists()


def test_canonicalize_worker_workspace_artifacts_links_only_matching_snapshot(
    tmp_path: Path,
) -> None:
    matching_worker = tmp_path / "workers" / "w00"
    unmatched_worker = tmp_path / "workers" / "w01"
    matching_snapshot = matching_worker / "snapshots" / "W00-L01-S01-abc"
    unmatched_snapshot = unmatched_worker / "snapshots" / "W01-L01-S01-def"
    _write(matching_worker / "workspace" / "submission.csv", "id,target\\n1,0.1\\n")
    _write(matching_snapshot / "submission.csv", "id,target\\n1,0.1\\n")
    _write(unmatched_worker / "workspace" / "submission.csv", "id,target\\n1,0.2\\n")
    _write(unmatched_snapshot / "submission.csv", "id,target\\n1,0.3\\n")

    records = canonicalize_worker_workspace_artifacts(
        worker_roots=[matching_worker, unmatched_worker],
        candidates=[
            {
                "worker_id": "W00",
                "snapshot_path": str(matching_snapshot),
            },
            {
                "worker_id": "W01",
                "snapshot_path": str(unmatched_snapshot),
            },
        ],
        artifact_path="submission.csv",
    )

    matching_artifact = matching_worker / "workspace" / "submission.csv"
    unmatched_artifact = unmatched_worker / "workspace" / "submission.csv"
    assert len(records) == 1
    assert matching_artifact.is_symlink()
    assert matching_artifact.read_text(encoding="utf-8") == "id,target\\n1,0.1\\n"
    assert not os.readlink(matching_artifact).startswith("/")
    assert unmatched_artifact.is_file()
    assert not unmatched_artifact.is_symlink()


def test_global_merge_collects_multiple_finals_without_single_winner(
    tmp_path: Path,
) -> None:
    snap = tmp_path / "snap"
    _write(snap / "submission.csv", "id,target\n1,0.2\n")
    _write(tmp_path / "data" / "sample_submission.csv", "id,target\n1,0\n")

    calls: list[tuple[str, float, list[Path]]] = []

    async def merge_executor(
        workspace: Path,
        prompt: str,
        budget: float,
        read_roots: list[Path],
    ) -> None:
        calls.append((prompt, budget, read_roots))
        _write(
            workspace / "finals" / "final_00" / "submission.csv", "id,target\n1,0.3\n"
        )
        _write(workspace / "finals" / "final_00" / "merge_report.md", "copy\n")
        _write(
            workspace / "finals" / "final_01" / "submission.csv", "id,target\n1,0.3\n"
        )
        _write(workspace / "finals" / "final_01" / "merge_report.md", "duplicate\n")
        _write(
            workspace / "finals" / "final_02" / "submission.csv", "id,target\n1,0.4\n"
        )
        _write(workspace / "tmp" / "merge_work" / "submission.csv", "scratch\\n")
        _write(workspace / "finals" / "final_02" / "merge_report.md", "blend\n")

    manifest = asyncio.run(
        run_global_merge(
            merge_dir=tmp_path / "merge",
            candidates=[
                {
                    "candidate_id": "W00:S01",
                    "snapshot_path": str(snap),
                    "metric_value": 0.2,
                    "lower_is_better": True,
                    "validation_ok": True,
                    "submission_sha": "abc",
                    "metric_validity": "high",
                    "selection_eligible": True,
                }
            ],
            worker_results=[],
            task_desc="write a valid submission",
            artifact_path="submission.csv",
            ledger_filename=".run_results.md",
            wall_clock_sec=120,
            evaluator_manager=EvaluatorManager.default(),
            cfg=SimpleNamespace(
                evaluator=SimpleNamespace(enabled=False),
                task_profile="mlebench",
                submission_dir=tmp_path / "submissions",
            ),
            task_profile="mlebench",
            task_id="dummy",
            task_root=tmp_path,
            dataset_source=tmp_path / "data",
            merge_executor=merge_executor,
        )
    )

    assert manifest["agent_status"] == "completed"
    assert len(calls) == 1
    assert "exploration phase is complete" in calls[0][0].lower()
    assert "route-diverse uniform" in calls[0][0]
    assert "do not assume that increasing" in calls[0][0]
    assert "exactly 3 distinct final artifacts" in calls[0][0]
    assert (
        tmp_path / "merge" / "global_merge_workspace" / "evidence" / "index.json"
    ).is_file()
    assert manifest["status"] == "success"
    assert manifest["required_final_count"] == 3
    assert manifest["requirement_met"] is True
    assert manifest["final_count"] == 3
    assert manifest["valid_final_count"] == 3
    assert manifest["fallback_final_sources"][0]["candidate_id"] == "W00:S01"
    assert manifest["merge_mode"] == "worker_reduce"
    assert "selected" not in manifest
    assert "selection_mode" not in manifest
    assert not (tmp_path / "merge" / "selected_submission.csv").exists()
    assert not (tmp_path / "merge" / "submission.csv").exists()
    assert not (tmp_path / "submissions" / "submission.csv").exists()
    assert not (tmp_path / "submissions" / "selected_submission.csv").exists()
    assert (tmp_path / "submissions" / "candidates" / "W00-S01.csv").is_symlink()
    assert (tmp_path / "submissions" / "finals" / "final_00.csv").is_symlink()
    assert (tmp_path / "merge" / "finals" / "final_00" / "submission.csv").is_symlink()
    assert (tmp_path / "merge" / "finals" / "final_01" / "submission.csv").is_symlink()
    assert (tmp_path / "merge" / "finals" / "final_02" / "submission.csv").is_symlink()
    assert (
        len(
            list((tmp_path / "merge" / "submission_snapshots").glob("*/submission.csv"))
        )
        == 2
    )
    assert manifest["submission_links"]
    assert not (tmp_path / "merge" / "global_merge_workspace" / "tmp").exists()


def test_global_merge_writes_fallback_finals_when_agent_writes_no_final(
    tmp_path: Path,
) -> None:
    snap = tmp_path / "snap"
    _write(snap / "submission.csv", "id,target\n1,0.2\n")

    async def no_final(
        _workspace: Path,
        _prompt: str,
        _budget: float,
        _read_roots: list[Path],
    ) -> None:
        return None

    manifest = asyncio.run(
        run_global_merge(
            merge_dir=tmp_path / "merge",
            candidates=[
                {
                    "candidate_id": "W00:S01",
                    "snapshot_path": str(snap),
                    "metric_value": 0.2,
                    "lower_is_better": True,
                    "validation_ok": True,
                    "submission_sha": "abc",
                    "metric_validity": "medium",
                    "selection_eligible": True,
                }
            ],
            worker_results=[],
            task_desc="write a valid submission",
            artifact_path="submission.csv",
            ledger_filename=".run_results.md",
            wall_clock_sec=120,
            evaluator_manager=EvaluatorManager.default(),
            cfg=SimpleNamespace(
                evaluator=SimpleNamespace(enabled=False),
                task_profile="mlebench",
                submission_dir=tmp_path / "submissions",
            ),
            task_profile="mlebench",
            task_id="dummy",
            task_root=tmp_path,
            dataset_source=None,
            merge_executor=no_final,
        )
    )

    assert manifest["agent_status"] == "completed"
    assert manifest["status"] == "insufficient_finals"
    assert manifest["required_final_count"] == 3
    assert manifest["requirement_met"] is False
    assert manifest["final_count"] == 1
    assert manifest["valid_final_count"] == 1
    assert manifest["merge_mode"] == "worker_reduce"
    assert "selected" not in manifest
    assert manifest["fallback_final_sources"][0]["candidate_id"] == "W00:S01"
    assert (tmp_path / "merge" / "finals" / "final_00" / "submission.csv").is_symlink()
    assert not (tmp_path / "merge" / "selected_submission.csv").exists()
    assert not (tmp_path / "submissions" / "submission.csv").exists()
    assert (tmp_path / "submissions" / "candidates" / "W00-S01.csv").is_symlink()
    assert (tmp_path / "submissions" / "finals" / "final_00.csv").is_symlink()


def test_global_merge_exposes_dataset_to_task_package_final_evaluator(
    tmp_path: Path,
) -> None:
    snap = tmp_path / "snap"
    points = [
        [math.cos(2 * math.pi * index / 16), math.sin(2 * math.pi * index / 16)]
        for index in range(16)
    ]
    _write(
        snap / "artifacts" / "best_solution.json",
        json.dumps({"points": points}),
    )
    dataset = tmp_path / "data"
    _write(
        dataset / "problem.json",
        json.dumps({"num_points": 16, "dimension": 2}),
    )

    manifest = asyncio.run(
        run_global_merge(
            merge_dir=tmp_path / "merge",
            candidates=[
                {
                    "candidate_id": "W00:S01",
                    "snapshot_path": str(snap),
                    "metric_value": 0.01,
                    "lower_is_better": False,
                    "validation_ok": True,
                    "submission_sha": "abc",
                    "metric_validity": "high",
                    "selection_eligible": True,
                }
            ],
            worker_results=[],
            task_desc="place 16 points",
            artifact_path="artifacts/best_solution.json",
            ledger_filename=".run_results.md",
            wall_clock_sec=0,
            evaluator_manager=EvaluatorManager.default(),
            cfg=SimpleNamespace(
                evaluator=SimpleNamespace(enabled=True, backend="task_package"),
                gate=SimpleNamespace(policy="default", params={}),
                submission_dir=tmp_path / "submissions",
            ),
            task_profile="opt_solver",
            task_id="ratio-minimization",
            task_root=tmp_path,
            dataset_source=dataset,
            required_finals=1,
            max_finals=1,
        )
    )

    final = manifest["finals"][0]
    assert manifest["status"] == "success"
    assert (tmp_path / "merge" / "finals" / "final_00" / "dataset").is_symlink()
    assert final["validation_ok"] is True
    assert final["gate_decision"]["action"] == "accept"
    assert final["metric_event"]["extra"]["gate"]["policy"] == "default"
    assert final["metric_event"]["extra"]["gate"]["trigger"] == "global_merge"


def test_global_final_records_rejected_gate_without_changing_validation_facts(
    tmp_path: Path,
) -> None:
    finals_dir = tmp_path / "finals"
    points = [
        [math.cos(2 * math.pi * index / 16), math.sin(2 * math.pi * index / 16)]
        for index in range(16)
    ]
    _write(
        finals_dir / "final_00" / "artifacts" / "best_solution.json",
        json.dumps({"points": points}),
    )
    dataset = tmp_path / "data"
    _write(dataset / "problem.json", json.dumps({"num_points": 16, "dimension": 2}))

    finals = _evaluate_finals(
        finals_dir=finals_dir,
        artifact_path="artifacts/best_solution.json",
        evaluator_manager=EvaluatorManager.default(),
        cfg=SimpleNamespace(
            evaluator=SimpleNamespace(enabled=True, backend="task_package"),
            gate=SimpleNamespace(policy="default", params={"unexpected": True}),
        ),
        task_profile="opt_solver",
        task_id="ratio-minimization",
        task_root=tmp_path,
        dataset_source=dataset,
    )

    assert finals[0]["validation_ok"] is True
    assert finals[0]["gate_decision"]["accepted"] is False
    assert finals[0]["gate_decision"]["reason_code"] == "gate_invalid_config"


def test_global_final_fails_closed_when_evaluator_returns_multiple_outcomes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    finals_dir = tmp_path / "finals"
    _write(finals_dir / "final_00" / "artifact.json", "{}")

    class FakeGateService:
        def evaluate(self, _request):
            return [object(), object()]

    monkeypatch.setattr(
        global_merge_runner,
        "GateService",
        lambda _manager: FakeGateService(),
    )
    finals = _evaluate_finals(
        finals_dir=finals_dir,
        artifact_path="artifact.json",
        evaluator_manager=object(),
        cfg=SimpleNamespace(),
        task_profile="default",
        task_id="dummy",
        task_root=tmp_path,
        dataset_source=None,
    )

    assert finals[0]["gate_decision"]["accepted"] is False
    assert finals[0]["gate_decision"]["reason_code"] == "evaluator_multiple_outcomes"


def test_pack_candidates_links_optional_merge_payload_with_size_caps(
    tmp_path: Path,
) -> None:
    snap = tmp_path / "snap"
    _write(snap / "submission.csv", "id,target\n1,0.1\n")
    _write(snap / "merge_payload" / "probabilities.csv", "id,p0,p1\n1,0.2,0.8\n")
    _write(
        snap / "merge_payload" / "oversized.bin",
        "0123456789012345678901234567890123456789",
    )

    packed = pack_candidates(
        merge_dir=tmp_path / "merge",
        merge_workspace=tmp_path / "merge_ws",
        candidates=[
            {
                "candidate_id": "W00:S01",
                "snapshot_path": str(snap),
                "metric_value": 0.1,
                "submission_sha": "abc",
                "metric_validity": "high",
                "selection_eligible": True,
            }
        ],
        artifact_path="submission.csv",
        ledger_filename=".run_results.md",
        max_prediction_file_bytes=30,
        max_prediction_total_bytes=30,
    )

    payload = tmp_path / "merge_ws" / "candidates" / "W00-S01" / "merge_payload"
    assert (payload / "probabilities.csv").is_symlink()
    assert not (payload / "oversized.bin").exists()
    assert packed[0]["merge_payload_files"] == ["merge_payload/probabilities.csv"]


def test_live_merge_reuses_the_existing_owner_agent(tmp_path: Path) -> None:
    calls: list[str | None] = []

    class Agent:
        llm = object()
        availableTools = SimpleNamespace(tool_map={})

        def swap_workspace(self, **kwargs: object) -> None:
            self.workspace = Path(str(kwargs["workspace_dir"]))

        async def run(self, prompt: str | None) -> str:
            calls.append(prompt)
            _write(
                self.workspace / "finals" / "final_00" / "submission.csv",
                "id,target\n1,0.3\n",
            )
            return "done"

    agent = Agent()
    owner = SimpleNamespace(
        _live_agent=agent,
        worker_id="W00",
        orchestrator=SimpleNamespace(make_llm_call_tracer=lambda **_kwargs: None),
        cfg=SimpleNamespace(exp_id="test"),
        worker_extra_env={},
        skill_registry=None,
        skill_task_category="",
        skill_allow_names=(),
        skill_tool_mode="all",
        skill_allow_generic_wildcard=True,
        skill_visible_max=0,
        resource_observer=None,
        state_machine=SimpleNamespace(mark_run_status=lambda *_args, **_kwargs: None),
        _task_runtime_extra_env=lambda: {},
        _accumulate_main_run_tokens=lambda _agent: None,
        _evaluator_candidate_artifact=lambda: "submission.csv",
    )
    coordinator = SimpleNamespace(
        deadline=10**12,
        _merge_owner_solver=owner,
    )

    asyncio.run(
        LnrSolver._run_live_merge_agent(
            coordinator,
            tmp_path / "merge_workspace",
            "final reducer query",
            60.0,
            [],
        )
    )

    assert owner._live_agent is agent
    assert calls == ["final reducer query"]
