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

from scienceflow.solver.lnr.resource_runtime.metric_history import (
    extract_metric_history_lines,
    metric_history_line_from_progress_signals,
    metric_history_text,
    update_metric_history_lines,
)

from tests.lnr_resource_test_utils import make_observer


def test_metric_history_keeps_target_metrics_and_drops_loss_only_noise() -> None:
    text = """
Epoch 1 train_loss=0.6921
Epoch 1 val_auc=0.7812 train_loss=0.6921
WARNING: dataloader worker is slow
100%|##########| 100/100 [00:10<00:00, 9.9it/s]
Final Validation Score: 0.812345
"""

    lines = extract_metric_history_lines(text)

    assert "Epoch 1 train_loss=0.6921" not in lines
    assert "Epoch 1 val_auc=0.7812 train_loss=0.6921" in lines
    assert "Final Validation Score: 0.812345" in lines
    assert all("WARNING" not in line for line in lines)
    assert all("100%" not in line for line in lines)


def test_structured_heartbeat_metric_becomes_metric_history_line() -> None:
    line = metric_history_line_from_progress_signals({
        "phase": "eval",
        "nb": {"current": 1437, "total": 11925},
        "metrics": {"kt": 0.770032, "loss": 0.21},
    })

    assert line == "SCIENCEFLOW_HB phase=eval progress=1437/11925 unit=nb kt=0.770032"


def test_metric_history_dedups_and_keeps_recent_lines() -> None:
    lines = update_metric_history_lines(
        ["Epoch 1 val_auc=0.70"],
        "\n".join([
            "Epoch 1 val_auc=0.70",
            "Epoch 2 val_auc=0.72",
            "Epoch 3 val_auc=0.73",
            "Epoch 4 val_auc=0.74",
        ]),
        max_lines=3,
    )

    assert lines == [
        "Epoch 2 val_auc=0.72",
        "Epoch 3 val_auc=0.73",
        "Epoch 4 val_auc=0.74",
    ]
    assert metric_history_text(lines) == "Epoch 2 val_auc=0.72\nEpoch 3 val_auc=0.73\nEpoch 4 val_auc=0.74"


def test_observer_progress_heartbeat_promotes_structured_metric_signals(tmp_path) -> None:
    observer, _ = make_observer(tmp_path, gpu_pool=["0"])
    job_id = observer.job_created(
        command="python train.py",
        inferred_class="heavy_gpu_train",
        gpu_ids=["0"],
        workspace_dir=tmp_path,
    )
    assert job_id

    observer.progress_heartbeat(
        job_id,
        elapsed_sec=120,
        phase="eval",
        signals={
            "phase": "eval",
            "nb": {"current": 1437, "total": 11925},
            "metrics": {"kt": 0.770032, "loss": 0.21},
        },
        stdout_lines=4,
        stdout_bytes=120,
    )

    job = observer._jobs[job_id]

    assert job.metric_history_text == "SCIENCEFLOW_HB phase=eval progress=1437/11925 unit=nb kt=0.770032"
    assert job.metric_history_line_count == 1


def test_observer_marks_only_monotonic_structured_progress_as_advanced(tmp_path) -> None:
    observer, _ = make_observer(tmp_path, gpu_pool=["0"])
    job_id = observer.job_created(
        command="python predict.py",
        inferred_class="heavy_gpu_train",
        gpu_ids=["0"],
        workspace_dir=tmp_path,
    )
    assert job_id

    observer.progress_heartbeat(
        job_id,
        elapsed_sec=60,
        phase="predict",
        signals={"phase": "predict", "nb": {"current": 1000, "total": 20000}},
    )
    first = observer._jobs[job_id].last_progress["structured_progress"]
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=120,
        phase="predict",
        signals={"phase": "predict", "nb": {"current": 1500, "total": 20000}},
    )
    second = observer._jobs[job_id].last_progress["structured_progress"]

    assert first["advanced"] is False
    assert second["advanced"] is True
    assert second["previous_current"] == 1000.0


def test_observer_treats_stable_prediction_artifact_as_value_bearing(tmp_path) -> None:
    observer, _ = make_observer(tmp_path, gpu_pool=["0"])

    assert observer._artifact_update_is_value_bearing({
        "path": "tmp/preds_argmax.pkl",
        "size_bytes": 1024,
        "stability": "stable",
        "artifact_scope": "workspace_recent",
    })
    assert not observer._artifact_update_is_value_bearing({
        "path": "tmp/train.log",
        "size_bytes": 1024,
        "stability": "stable",
        "artifact_scope": "workspace_recent",
    })
    assert not observer._artifact_update_is_value_bearing({
        "path": "tmp/preds_argmax.pkl",
        "size_bytes": 1024,
        "stability": "candidate",
        "artifact_scope": "workspace_recent",
    })


def test_observer_progress_snapshot_carries_metric_history(tmp_path) -> None:
    observer, _ = make_observer(tmp_path, gpu_pool=["0"])
    job_id = observer.job_created(
        command="python train.py",
        inferred_class="heavy_gpu_train",
        gpu_ids=["0"],
        workspace_dir=tmp_path,
    )
    assert job_id

    observer.progress_heartbeat(
        job_id,
        elapsed_sec=120,
        phase="training",
        signals={"metrics": {"val_auc": 0.81}},
        stdout_lines=4,
        stdout_bytes=120,
        metric_history_text="Epoch 1 val_auc=0.78\nEpoch 2 val_auc=0.81",
        metric_history_line_count=2,
    )

    job = observer._jobs[job_id]
    snapshot = observer._progress_snapshot_for_job(job, {"stdout_lines": 4, "stdout_bytes": 120}, elapsed_sec=180)

    assert snapshot["metric_history_text"] == "Epoch 1 val_auc=0.78\nEpoch 2 val_auc=0.81"
    assert snapshot["metric_history_line_count"] == 2
