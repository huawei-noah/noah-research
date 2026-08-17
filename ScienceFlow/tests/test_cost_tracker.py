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

"""Tests for full-run cost report parsing and ``Scale:`` line formatting."""

from __future__ import annotations

from scienceflow.utils.cost_tracker import (
    format_fullrun_cost_report,
    parse_fullrun_cost_summary,
)


def test_scale_line_from_scale_only() -> None:
    stdout = (
        "[SCALE] trained_rows=836000 total_train_rows=839000 batch_size=128 epochs=10\n"
        "Final Validation Score: 0.5\n"
    )
    s = parse_fullrun_cost_summary(
        stdout,
        wall_sec=100.0,
        exit_code=0,
        metric_value=0.5,
        metric_name="Final Validation Score",
        lower_is_better=False,
    )
    report = format_fullrun_cost_report(s)
    assert "  Scale: " in report
    assert "trained_rows=" in report
    assert "836.0k/839.0k" in report or "836" in report
    assert "batch_size=128" in report
    assert "epochs=10" in report
    assert s.trained_rows == 836000
    assert s.total_train_rows == 839000
    assert s.batch_size == 128
    assert s.epochs_from_scale == 10


def test_scale_line_from_dataset_and_data() -> None:
    stdout = (
        "[DATASET] total_train_rows=1000 quick_test_rows=20\n"
        "[DATA] loaded in 1.2s rows=1000\n"
    )
    s = parse_fullrun_cost_summary(stdout, wall_sec=50.0, exit_code=0)
    report = format_fullrun_cost_report(s)
    assert "  Scale: " in report
    assert "100.0%" in report or "100%" in report
    assert s.total_train_rows == 1000
    assert s.trained_rows == 1000


def test_no_scale_signals_no_scale_line() -> None:
    stdout = "Some log without markers\n[EPOCH] 1/5 time=2.0s\n"
    s = parse_fullrun_cost_summary(stdout, wall_sec=30.0, exit_code=0)
    report = format_fullrun_cost_report(s)
    assert "  Scale: " not in report
    assert "Epoch stats:" in report


def test_dataset_and_data_last_wins() -> None:
    stdout = (
        "[DATASET] total_train_rows=100 quick_test_rows=10\n"
        "[DATA] loaded in 0.1s rows=50\n"
        "[DATASET] total_train_rows=200 quick_test_rows=10\n"
        "[DATA] loaded in 0.2s rows=180\n"
    )
    s = parse_fullrun_cost_summary(stdout, wall_sec=1.0, exit_code=0)
    assert s.total_train_rows == 200
    assert s.trained_rows == 180


def test_dataset_data_plus_epoch_adds_epochs_to_scale_line() -> None:
    stdout = (
        "[DATASET] total_train_rows=500 quick_test_rows=20\n"
        "[DATA] loaded in 0.5s rows=500\n"
        "[EPOCH] 1/2 time=1.0s\n"
        "[EPOCH] 2/2 time=1.0s\n"
    )
    s = parse_fullrun_cost_summary(stdout, wall_sec=5.0, exit_code=0)
    report = format_fullrun_cost_report(s)
    assert "  Scale: " in report
    assert "epochs=2" in report


def test_scale_line_before_epoch_stats() -> None:
    stdout = (
        "[SCALE] batch_size=32 epochs=3\n"
        "[EPOCH] 1/3 time=1.0s\n"
        "[EPOCH] 2/3 time=1.0s\n"
        "[EPOCH] 3/3 time=1.0s\n"
    )
    s = parse_fullrun_cost_summary(stdout, wall_sec=10.0, exit_code=0)
    report = format_fullrun_cost_report(s)
    idx_scale = report.find("  Scale: ")
    idx_epoch = report.find("  Epoch stats:")
    assert idx_scale != -1 and idx_epoch != -1
    assert idx_scale < idx_epoch
    assert "batch_size=32" in report
    assert "epochs=3" in report
