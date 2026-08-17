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

import pytest

from tests.lnr_resource_test_utils import make_observer


def _job_with_redirected_log(observer, tmp_path):
    job_id = observer.job_created(
        command="python3 train.py > tmp/train_run.log 2>&1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=7200.0,
        workspace_dir=tmp_path,
    )
    assert job_id
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=60.0,
        phase="training",
        signals={
            "artifact": {
                "path": "tmp/train_run.log",
                "age_sec": 53.0,
                "candidate_artifact": True,
                "size_bytes": 986,
                "artifact_scope": "current_run",
            }
        },
        stdout_lines=0,
        stdout_bytes=0,
    )
    return job_id


def test_recoverability_and_artifact_update_age_refresh_with_current_elapsed(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        gpu_pool=["0"],
        stalled_stdout_sec=300.0,
        low_progress_no_artifact_sec=300.0,
    )
    job_id = _job_with_redirected_log(observer, tmp_path)
    job = observer._jobs[job_id]
    signal = {
        "elapsed_sec": 1500.0,
        "stdout_age_sec": 1500.0,
        "stdout_lines": 0,
        "stdout_bytes": 0,
    }

    progress = observer._progress_snapshot_for_job(job, signal, elapsed_sec=1500.0)
    resource = observer._resource_snapshot_for_job(job, signal)

    assert progress["artifact_last_update_age_sec"] == pytest.approx(1493.0)
    assert progress["artifact_updates"][0]["age_sec"] == pytest.approx(1493.0)
    assert progress["recoverability"]["artifact_age_sec"] == pytest.approx(1493.0)
    assert resource["recoverability"]["artifact_age_sec"] == pytest.approx(1493.0)


def test_stale_redirected_log_is_not_stdout_progress(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        gpu_pool=["0"],
        stalled_stdout_sec=300.0,
        low_progress_no_artifact_sec=300.0,
    )
    job_id = _job_with_redirected_log(observer, tmp_path)
    job = observer._jobs[job_id]
    signal = {
        "elapsed_sec": 1500.0,
        "stdout_age_sec": 1500.0,
        "stdout_lines": 0,
        "stdout_bytes": 0,
        "saw_training_progress": True,
    }

    observer._update_job_progress_signal(job, signal, elapsed_sec=1500.0)
    progress = observer._progress_snapshot_for_job(job, signal, elapsed_sec=1500.0)
    stdout = progress["stdout_observation"]

    assert stdout["primary_channel"] == "redirected_log"
    assert stdout["stdout_stream"]["present"] is False
    assert stdout["redirected_log"]["present"] is True
    assert stdout["redirected_log"]["fresh"] is False
    assert progress["meaningful_stdout"] is False
    assert progress["artifact_updates"][0]["artifact_log_like"] is True
