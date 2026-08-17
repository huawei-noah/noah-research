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
from pathlib import Path

from scienceflow.solver.lnr.global_merge.candidate_evidence import (
    apply_candidate_evidence,
    load_peer_candidate_evidence,
    recover_candidate_artifact,
)
from scienceflow.solver.lnr.global_merge.candidate_pack import pack_candidates
from scienceflow.solver.lnr.global_merge.fallback import ranked_candidates


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_peer_evidence_recovers_failed_worker_submission(tmp_path: Path) -> None:
    worker = tmp_path / "workers" / "w01"
    submission = (
        worker / "logs" / "submission_snapshots" / "iter_0006_s03_submission.csv"
    )
    _write(submission, "id,target\n1,0.3\n")
    sha = hashlib.sha256(submission.read_bytes()).hexdigest()
    peer_csv = tmp_path / "task_logs" / "lhr_stage_performance.csv"
    _write(
        peer_csv,
        "worker_id,stage_id,metric_value,metric_validity,lineage_id,submission_sha,"
        "submission_snapshot,candidate_ready,selection_eligible\n"
        f"W01,S03,0.3,medium,L01,{sha},"
        "../logs/submission_snapshots/iter_0006_s03_submission.csv,1,1\n",
    )

    evidence = load_peer_candidate_evidence(peer_csv, worker_id="W01")
    candidate = apply_candidate_evidence(
        {
            "candidate_id": "W01:S03",
            "worker_id": "W01",
            "stage_id": "S03",
            "worker_root": str(worker),
            "snapshot_path": str(worker / "snapshots" / "missing"),
        },
        evidence["S03"],
    )
    recovered = recover_candidate_artifact(candidate, artifact_path="submission.csv")

    assert recovered["candidate_ready"] is True
    assert recovered["submission_sha"] == sha
    assert Path(recovered["artifact_source"]) == submission.resolve()


def test_candidate_without_sha_does_not_claim_current_workspace_artifact(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "workers" / "w01"
    _write(worker / "workspace" / "submission.csv", "id,target\n1,current\n")

    recovered = recover_candidate_artifact(
        {
            "candidate_id": "W01:S03",
            "worker_id": "W01",
            "stage_id": "S03",
            "worker_root": str(worker),
            "snapshot_path": str(worker / "snapshots" / "missing"),
        },
        artifact_path="submission.csv",
    )

    assert recovered["candidate_ready"] is False
    assert "artifact_source" not in recovered


def test_json_recovery_prefers_artifact_sha_over_conflicting_submission_sha(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    artifact = snapshot / "artifacts" / "submission.json"
    _write(artifact, '{"candidate":"value"}\n')
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()

    recovered = recover_candidate_artifact(
        {
            "candidate_id": "W00:L01:S01",
            "snapshot_path": str(snapshot),
            "artifact_path": "artifacts/submission.json",
            "artifact_sha": artifact_sha,
            "submission_sha": "0" * 64,
        },
        artifact_path="artifacts/submission.json",
    )

    assert recovered["candidate_ready"] is True
    assert recovered["artifact_sha"] == artifact_sha


def test_json_recovery_does_not_use_legacy_submission_sha(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    artifact = snapshot / "submission.json"
    _write(artifact, '{"candidate":"value"}\n')

    recovered = recover_candidate_artifact(
        {
            "candidate_id": "W00:L01:S01",
            "snapshot_path": str(snapshot),
            "submission_sha": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        artifact_path="submission.json",
    )

    assert recovered["candidate_ready"] is False
    assert "artifact_source" not in recovered


def test_pack_candidates_deduplicates_and_preserves_diversity(tmp_path: Path) -> None:
    specs = [
        ("W00:S01", "W00", "L01", 0.9, "a"),
        ("W00:S02", "W00", "L01", 0.8, "a"),
        ("W00:S03", "W00", "L02", 0.7, "b"),
        ("W01:S01", "W01", "L01", 0.1, "c"),
    ]
    candidates = []
    for candidate_id, worker_id, lineage_id, metric, value in specs:
        snap = tmp_path / candidate_id.replace(":", "-")
        _write(snap / "submission.csv", f"id,target\n1,{value}\n")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "worker_id": worker_id,
                "lineage_id": lineage_id,
                "snapshot_path": str(snap),
                "metric_value": metric,
                "lower_is_better": False,
                "submission_sha": "placeholder",
                "metric_validity": "medium",
                "selection_eligible": True,
            }
        )

    packed = pack_candidates(
        merge_dir=tmp_path / "merge",
        merge_workspace=tmp_path / "merge_ws",
        candidates=candidates,
        artifact_path="submission.csv",
        ledger_filename=".run_results.md",
        max_candidates=3,
    )

    assert {candidate["worker_id"] for candidate in packed} == {"W00", "W01"}
    assert {candidate["lineage_id"] for candidate in packed} == {"L01", "L02"}
    assert len({candidate["submission_sha"] for candidate in packed}) == 3


def test_meta_fit_candidate_ranks_after_comparable_candidate() -> None:
    ranked = ranked_candidates(
        [
            {
                "candidate_id": "W00:S01",
                "candidate_ready": True,
                "submission_sha": "meta-fit",
                "selection_eligible": True,
                "metric_validity": "medium",
                "metric_validity_reason_code": "same_validation_meta_fit",
                "metric_value": 0.9,
                "lower_is_better": False,
            },
            {
                "candidate_id": "W01:S01",
                "candidate_ready": True,
                "submission_sha": "comparable",
                "selection_eligible": True,
                "metric_validity": "medium",
                "metric_value": 0.5,
                "lower_is_better": False,
            },
        ]
    )

    assert [candidate["candidate_id"] for candidate in ranked] == [
        "W01:S01",
        "W00:S01",
    ]
