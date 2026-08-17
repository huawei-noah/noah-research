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

import pytest

from scienceflow.core.tools.bash_tool import _parse_progress_signals
from tests.lnr_resource_test_utils import make_observer


def _monitor_signal(elapsed_sec: float) -> dict:
    return {
        "current_phase": "validation_or_inference",
        "process_tree_cpu": {"available": True, "total_cpu_pct": 100.0},
        "deadline_remaining_sec": 43200.0 - elapsed_sec,
        "finalization_reserve_sec": 900.0,
    }


def _make_slow_cpu_job(tmp_path: Path):
    observer, _ = make_observer(
        tmp_path,
        arbiter_enabled=True,
        arbiter_periodic_review_enabled=False,
        review_heartbeat_sec=60.0,
    )
    job_id = observer.job_created(
        command="python3 predict.py",
        inferred_class="heavy_cpu_candidate",
        gpu_ids=[],
        cpu_set="0-15",
        timeout_sec=43200.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    job = observer._jobs[job_id]
    observer._promote(job, reason="elapsed_threshold", elapsed_sec=1800.0)
    return observer, job_id, job


def _progress(observer, job_id: str, *, current: int, elapsed_sec: float) -> None:
    text = f"Processed {current}/302644, elapsed={elapsed_sec:.1f}s"
    if current == 50000:
        text = "Validation Results: avg_levenshtein=9.3403\n" + text
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=elapsed_sec,
        phase="validation_or_inference",
        signals=_parse_progress_signals(text),
        stdout_lines=max(1, current // 50000),
        stdout_bytes=max(100, current // 1000),
    )


def test_billion_word_unknown_eta_reaches_early_efficiency_review(
    tmp_path: Path,
) -> None:
    observer, _job_id, job = _make_slow_cpu_job(tmp_path)
    first = observer._update_resource_efficiency_state(
        job,
        _monitor_signal(1800.0),
        elapsed_sec=1800.0,
    )
    sustained = observer._update_resource_efficiency_state(
        job,
        _monitor_signal(1860.0),
        elapsed_sec=1860.0,
    )
    decision = observer._periodic_efficiency_review_decision(
        job,
        {
            **_monitor_signal(1860.0),
            "elapsed_sec": 1860.0,
            "resource_efficiency": sustained,
        },
    )

    assert first["status"] == "inefficient_unknown_eta"
    assert first["unknown_eta_persistent"] is True
    assert first["mismatch_windows"] == 1
    assert sustained["mismatch_windows"] == 2
    assert sustained["review_required"] is True
    assert decision["enabled"] is True
    assert decision["proposal"]["reason_code"] == (
        "resource_efficiency:sustained_resource_mismatch"
    )


def test_billion_word_first_mature_progress_sample_has_actionable_eta(
    tmp_path: Path,
) -> None:
    observer, job_id, job = _make_slow_cpu_job(tmp_path)
    _progress(observer, job_id, current=50000, elapsed_sec=3658.8)

    assessment = observer._update_resource_efficiency_state(
        job,
        _monitor_signal(3658.8),
        elapsed_sec=3658.8,
    )

    assert assessment["progress_interval_samples"] == 0
    assert assessment["eta_reliable"] is True
    assert assessment["eta_to_deliverable_sec"] == pytest.approx(18487.477344)
    assert assessment["long_eta"] is True
    assert assessment["mismatch_windows"] == 1


def test_billion_word_near_complete_inference_remains_protected(
    tmp_path: Path,
) -> None:
    observer, job_id, job = _make_slow_cpu_job(tmp_path)
    _progress(observer, job_id, current=250000, elapsed_sec=18464.1)
    observer._update_resource_efficiency_state(
        job,
        _monitor_signal(18464.1),
        elapsed_sec=18464.1,
    )
    _progress(observer, job_id, current=300000, elapsed_sec=22165.7)
    assessment = observer._update_resource_efficiency_state(
        job,
        _monitor_signal(22165.7),
        elapsed_sec=22165.7,
    )

    assert assessment["eta_to_deliverable_sec"] == pytest.approx(195.740608)
    assert assessment["phase_completion_protected"] is True
    assert assessment["review_required"] is False
