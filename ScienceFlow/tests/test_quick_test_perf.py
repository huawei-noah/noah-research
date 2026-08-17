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

"""Tests for quick-test stdout parsing and full-run time extrapolation."""

from __future__ import annotations

from scienceflow.utils.quick_test_perf import (
    estimate_full_run_seconds,
    is_plausible_epoch_duration_sec,
    parse_epoch_line,
    parse_quick_test_stdout,
    strip_bash_tool_prefix,
)


def test_strip_bash_tool_prefix() -> None:
    raw = "[exit=0, 1.2s]\n[DATA] loaded in 0.5s rows=20\n"
    assert "[DATA]" in strip_bash_tool_prefix(raw)


def test_parse_and_extrapolate() -> None:
    stdout = """
[DATASET] total_train_rows=1000 quick_test_rows=20
[DATA] loaded in 1.0s rows=20
[EPOCH] 1/2 time=2.0s train_loss=0.1 val_metric=0.5
[EPOCH] 2/2 time=2.0s train_loss=0.05 val_metric=0.4
Final Validation Score: 0.4
"""
    p = parse_quick_test_stdout(stdout)
    assert p.total_train_rows == 1000
    assert p.quick_test_rows == 20
    assert p.data_load_sec == 1.0
    assert p.num_epochs == 2
    assert p.epoch_times_sec == [2.0, 2.0]

    est = estimate_full_run_seconds(
        p,
        quick_test_rows=20,
        total_rows=1000,
    )
    assert est is not None
    # scale 50: data 1*50 + avg_epoch 2*50 * 2 epochs = 50 + 200 = 250
    assert abs(est - 250.0) < 0.01


def test_extrapolate_without_epochs_returns_none() -> None:
    p = parse_quick_test_stdout("hello\n")
    assert estimate_full_run_seconds(p, quick_test_rows=20, total_rows=100) is None


def test_extrapolate_separates_first_epoch_overhead() -> None:
    """First epoch much slower than later ones: fixed part is not scaled by ``scale``."""
    stdout = """
[DATASET] total_train_rows=1000 quick_test_rows=20
[DATA] loaded in 1.0s rows=20
[EPOCH] 1/2 time=12.0s train_loss=0.1 val_metric=0.5
[EPOCH] 2/2 time=2.0s train_loss=0.05 val_metric=0.4
Final Validation Score: 0.4
"""
    p = parse_quick_test_stdout(stdout)
    assert p.epoch_times_sec == [12.0, 2.0]
    est = estimate_full_run_seconds(
        p,
        quick_test_rows=20,
        total_rows=1000,
        separate_fixed_epoch_overhead=True,
    )
    assert est is not None
    # fixed_epoch = max(0, 12-2)=10; avg_epoch=2; scale=50; n_ep=2
    # data: 1*50=50; epoch: 10 + 2*50*2 = 10+200=210 → 260
    assert abs(est - 260.0) < 0.01


def test_extrapolate_legacy_flag_matches_uniform_epochs() -> None:
    stdout = """
[DATASET] total_train_rows=1000 quick_test_rows=20
[DATA] loaded in 1.0s rows=20
[EPOCH] 1/2 time=2.0s
[EPOCH] 2/2 time=2.0s
"""
    p = parse_quick_test_stdout(stdout)
    est_new = estimate_full_run_seconds(
        p, quick_test_rows=20, total_rows=1000, separate_fixed_epoch_overhead=True,
    )
    est_old = estimate_full_run_seconds(
        p, quick_test_rows=20, total_rows=1000, separate_fixed_epoch_overhead=False,
    )
    assert est_new is not None and est_old is not None
    assert abs(est_new - est_old) < 0.01

def test_unix_timestamp_as_epoch_time_is_ignored_for_extrapolation() -> None:
    """Bug: solution prints time.time() as duration → huge bogus estimate; must not parse as epoch sec."""
    stdout = """
[DATASET] total_train_rows=1000 quick_test_rows=20
[DATA] loaded in 1.0s rows=20
[EPOCH] 1/1 time=1775373847.3s train_loss=0.0 val_metric=0.19
Final Validation Score: 0.19
"""
    p = parse_quick_test_stdout(stdout)
    assert p.epoch_times_sec is None
    assert estimate_full_run_seconds(p, quick_test_rows=20, total_rows=1000) is None


def test_parse_epoch_line_rejects_unix_timestamp() -> None:
    assert parse_epoch_line("[EPOCH] 1/10 time=1775373847.3s") is None
    assert parse_epoch_line("[EPOCH] 1/10 time=2.5s") == (1, 10, 2.5)


def test_is_plausible_epoch_duration_sec() -> None:
    assert is_plausible_epoch_duration_sec(2.5) is True
    assert is_plausible_epoch_duration_sec(1e9) is False
    assert is_plausible_epoch_duration_sec(1e9, strict_quick_test=False) is False
    assert is_plausible_epoch_duration_sec(100000.0, strict_quick_test=True) is False
    assert is_plausible_epoch_duration_sec(100000.0, strict_quick_test=False) is True

