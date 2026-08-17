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

import csv
from pathlib import Path

from scienceflow.solver.lnr.stage.score_summary import (
    build_stage_performance_score_summary,
    format_score_summary_context,
    metric_lower_is_better_hint,
)


FIELDS = [
    "row_order",
    "candidate_id",
    "worker_id",
    "stage_id",
    "metric_value",
    "metric_name",
    "lower_is_better",
    "validation_ok",
    "val_score_type",
    "selection_eligible",
    "selection_score",
    "metric_source_note",
    "metric_validity",
    "metric_validity_note",
    "metric_validity_source",
    "brief",
    "why",
    "route_evidence",
    "submission_snapshot",
    "candidate_ready",
    "submission_status",
    "artifact_sha",
    "artifact_kind",
    "evaluator_backend",
    "evaluator_status",
    "submission_changed",
    "duplicate_submission_of_stage",
    "task_profile",
    "snapshot_path",
]


def _write_perf(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_stage_performance_summary_exposes_best_and_valid_best_with_gap(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "stage_id": "S01",
                "metric_value": "0.80",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "0.80",
                "brief": "valid baseline",
                "why": "valid submission",
                "submission_snapshot": "snapshots/W00/s01.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S01",
            },
            {
                "row_order": 2,
                "candidate_id": "W01:L01:S01",
                "worker_id": "W01",
                "stage_id": "S01",
                "metric_value": "0.90",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "train_only_or_training_metric",
                "selection_eligible": "0",
                "selection_score": "",
                "brief": "observed better log score",
                "why": "not yet official comparable",
                "submission_snapshot": "snapshots/W01/s01.csv",
                "candidate_ready": "0",
                "submission_status": "missing_submission",
                "snapshot_path": "/abs/snapshots/W01/S01",
            },
        ],
    )

    summary = build_stage_performance_score_summary(perf, current_worker_id="W02")

    assert summary["best_score"]["value"] == 0.90
    assert summary["best_score"]["validity"] == "observed_not_official"
    assert summary["valid_best_score"]["value"] == 0.80
    assert summary["valid_best_score"]["validity"] == "valid_comparable"
    assert summary["capture_gap"] is True
    assert summary["recommended_action"] == "score_contract_repair"
    assert "artifact_path" not in summary["best_score"]
    assert "snapshot_path" not in summary["valid_best_score"]


def test_legacy_shifted_evaluator_row_can_be_valid_best(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "stage_id": "S01",
                "metric_value": "519.647260",
                "metric_name": "mission_duration_days",
                "lower_is_better": "1",
                "validation_ok": "1",
                "val_score_type": "benchmark",
                "selection_eligible": "1",
                "selection_score": "519.647260",
                "metric_validity": "high",
                "candidate_ready": "1",
                "submission_status": "ok",
            },
            {
                "row_order": 2,
                "candidate_id": "W01:L09:S112",
                "worker_id": "W01",
                "stage_id": "S112",
                "metric_value": "393.234125",
                "metric_name": "mission_duration_days",
                "lower_is_better": "1",
                "validation_ok": "1",
                "val_score_type": "benchmark",
                "selection_eligible": "1",
                "selection_score": "393.234125",
                "metric_validity": "high",
                "metric_validity_note": "opt_solver_evaluator_ok",
                "metric_validity_source": "metric_validity_system",
                "artifact_sha": "artifacts/best_solution.json",
                "artifact_kind": "5e1055963bfc3fe58d3696a81c9493b6e3e007de839f9c7f6b27c5b2f496fda8",
                "evaluator_backend": "json_solution",
                "evaluator_status": "artifact_command",
                "submission_changed": "ok",
                "candidate_ready": "0",
                "submission_status": "1",
                "duplicate_submission_of_stage": "ok",
                "snapshot_path": "W01-L09-S112-95c4b94f8f",
            },
        ],
    )

    summary = build_stage_performance_score_summary(perf)

    assert summary["valid_best_score"]["value"] == 393.234125
    assert summary["valid_best_score"]["validity"] == "valid_comparable"
    assert summary["valid_best_score"]["candidate_ready"] is True
    assert summary["valid_best_score"]["evaluator_backend"] == "artifact_command"
    assert summary["valid_best_score"]["evaluator_status"] == "ok"


def test_evaluator_verified_medium_metric_can_be_valid_best(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "stage_id": "S01",
                "metric_value": "0.80",
                "metric_name": "solver_score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "benchmark",
                "selection_eligible": "1",
                "selection_score": "0.80",
                "metric_validity": "high",
                "brief": "baseline",
                "why": "verified baseline",
                "submission_snapshot": "snapshots/W00/s01.json",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S01",
            },
            {
                "row_order": 2,
                "candidate_id": "W01:L01:S02",
                "worker_id": "W01",
                "stage_id": "S02",
                "metric_value": "0.90",
                "metric_name": "solver_score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "benchmark",
                "selection_eligible": "1",
                "selection_score": "0.90",
                "metric_validity": "medium",
                "metric_validity_note": "evaluator verified but provenance note is medium",
                "brief": "better evaluator result",
                "why": "artifact evaluator accepted the solution",
                "submission_snapshot": "snapshots/W01/s02.json",
                "candidate_ready": "1",
                "submission_status": "ready",
                "evaluator_backend": "artifact_command",
                "evaluator_status": "ok",
                "task_profile": "custom_solver",
                "snapshot_path": "/abs/snapshots/W01/S02",
            },
        ],
    )

    summary = build_stage_performance_score_summary(perf, current_worker_id="W00")

    assert summary["valid_best_score"]["value"] == 0.90
    assert summary["valid_best_score"]["validity"] == "valid_comparable"
    assert summary["valid_best_score"]["metric_validity"] == "medium"
    assert summary["valid_best_score"]["evaluator_status"] == "ok"
    assert summary["valid_record_count"] == 2
    assert summary["capture_gap"] is False


def test_stage_performance_summary_keeps_local_paths_only_for_current_worker(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W00:L01:S07",
                "worker_id": "W00",
                "stage_id": "S07",
                "metric_value": "0.889245",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "cv",
                "selection_eligible": "1",
                "selection_score": "0.889245",
                "brief": "Direct sigmoid of precomputed logits",
                "why": "logits dominate tabular features",
                "submission_snapshot": "snapshots/W00/s07.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S07",
            }
        ],
    )

    local = build_stage_performance_score_summary(perf, current_worker_id="W00")
    remote = build_stage_performance_score_summary(perf, current_worker_id="W01")

    assert local["valid_best_score"]["artifact_path"] == "snapshots/W00/s07.csv"
    assert local["valid_best_score"]["snapshot_path"] == "/abs/snapshots/W00/S07"
    assert "artifact_path" not in remote["valid_best_score"]
    assert "snapshot_path" not in remote["valid_best_score"]
    assert remote["worker_isolation"].startswith("global_scores_hide_cross_worker_paths")


def test_stage_performance_summary_detects_cheap_signal_from_brief(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W03:L01:S02",
                "worker_id": "W03",
                "stage_id": "S02",
                "metric_value": "0.876728",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "0.876728",
                "brief": "Direct sigmoid of pre-computed CNN logits",
                "why": "Strongest cheap signal; no tabular blend needed",
                "submission_snapshot": "snapshots/W03/s02.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W03/S02",
            }
        ],
    )

    summary = build_stage_performance_score_summary(perf, current_worker_id="W00")

    assert summary["cheap_signal_available"] is True
    assert summary["cheap_signal_best_score"]["value"] == 0.876728
    assert summary["cheap_signal_best_score"]["worker_id"] == "W03"
    assert "artifact_path" not in summary["cheap_signal_best_score"]


def test_score_summary_context_hides_paths_even_for_local_summary(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W00:L01:S07",
                "worker_id": "W00",
                "stage_id": "S07",
                "metric_value": "0.889245",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "cv",
                "selection_eligible": "1",
                "selection_score": "0.889245",
                "brief": "Direct sigmoid of cached logits",
                "why": "valid comparable route",
                "submission_snapshot": "snapshots/W00/s07.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S07",
            }
        ],
    )

    summary = build_stage_performance_score_summary(perf, current_worker_id="W00")
    assert summary["valid_best_score"]["artifact_path"] == "snapshots/W00/s07.csv"

    context = '\n'.join(format_score_summary_context(summary))

    assert "score_summary:" in context
    assert "0.889245" in context
    assert "snapshots/W00/s07.csv" not in context
    assert "/abs/snapshots/W00/S07" not in context



def test_stage_performance_summary_excludes_risk_flagged_metric_from_valid_best(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "stage_id": "S01",
                "metric_value": "0.998160",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "cv",
                "selection_eligible": "1",
                "selection_score": "0.998160",
                "metric_source_note": "type=cv; risk=overfit_suspected",
                "brief": "XGB stacking meta-model on validation set",
                "why": "This severely overfits the validation set and is not generalizable.",
                "submission_snapshot": "snapshots/W00/s01.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S01",
            },
            {
                "row_order": 2,
                "candidate_id": "W01:L01:S02",
                "worker_id": "W01",
                "stage_id": "S02",
                "metric_value": "0.896611",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "cv",
                "selection_eligible": "1",
                "selection_score": "0.896611",
                "metric_source_note": "type=cv; eval=oof; train_data=cv_folds; execution=train_only_or_training_metric",
                "brief": "OOF image-feature ensemble with logits blend",
                "why": "Reusable model with comparable CV score.",
                "submission_snapshot": "snapshots/W01/s02.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W01/S02",
            },
        ],
    )

    summary = build_stage_performance_score_summary(perf, current_worker_id="W02")

    assert summary["best_score"]["value"] == 0.998160
    assert summary["best_score"]["validity"] == "risk_flagged_metric"
    assert summary["best_score"]["reason"] == "metric_text_reports_overfit"
    assert summary["valid_best_score"]["value"] == 0.896611
    assert summary["valid_best_score"]["validity"] == "valid_comparable"
    assert summary["cheap_signal_best_score"]["value"] == 0.896611
    assert summary["capture_gap"] is True
    assert summary["recommended_action"] == "metric_source_review"

    context = '\n'.join(format_score_summary_context(summary))
    assert "metric_source_note=type=cv; risk=overfit_suspected" in context
    assert "snapshots/W00/s01.csv" not in context


def test_metric_direction_hint_recognizes_auc_as_higher_is_better() -> None:
    assert metric_lower_is_better_hint("Submissions are evaluated on ROC AUC") is False
    assert metric_lower_is_better_hint("root mean squared error") is True


def test_stage_performance_summary_direction_uses_votes_not_metric_text(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "stage_id": "S01",
                "metric_value": "0.20",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "0.20",
                "brief": "AUC style model",
                "why": "higher better text must not override explicit merge votes",
                "submission_snapshot": "snapshots/W00/s01.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S01",
            },
            {
                "row_order": 2,
                "candidate_id": "W01:L01:S01",
                "worker_id": "W01",
                "stage_id": "S01",
                "metric_value": "0.30",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "0.30",
                "brief": "AUC style model",
                "why": "higher better text must not override explicit merge votes",
                "submission_snapshot": "snapshots/W01/s01.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W01/S01",
            },
        ],
    )

    summary = build_stage_performance_score_summary(perf)

    assert summary["valid_best_score"]["value"] == 0.20
    assert summary["valid_best_score"]["lower_is_better"] is True


def test_stage_performance_summary_normalizes_mixed_direction_votes(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "stage_id": "S01",
                "metric_value": "0.894680",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "0.894680",
                "brief": "valid AUC baseline",
                "why": "submission validated",
                "submission_snapshot": "snapshots/W00/s01.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S01",
            },
            {
                "row_order": 2,
                "candidate_id": "W01:L01:S01",
                "worker_id": "W01",
                "stage_id": "S01",
                "metric_value": "0.891799",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "0.891799",
                "brief": "valid lower AUC",
                "why": "submission validated",
                "submission_snapshot": "snapshots/W01/s01.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W01/S01",
            },
            {
                "row_order": 3,
                "candidate_id": "W02:L01:S01",
                "worker_id": "W02",
                "stage_id": "S01",
                "metric_value": "0.876728",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "0.876728",
                "brief": "row has stale lower flag from result card",
                "why": "should not invert the task-level direction",
                "submission_snapshot": "snapshots/W02/s01.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W02/S01",
            },
        ],
    )

    summary = build_stage_performance_score_summary(perf, current_worker_id="W03")

    assert summary["valid_best_score"]["value"] == 0.894680
    assert summary["valid_best_score"]["lower_is_better"] is False
    assert summary["best_score"]["value"] == 0.894680
