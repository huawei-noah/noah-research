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

"""Monitor UI: ParallelRunner ``logs/state.json`` overlay (DONE on budget/time end etc.)."""

from __future__ import annotations

from scienceflow.ui.monitor import (
    _multi_status_and_progress,
    _parallel_done_for_row,
)


def test_parallel_done_for_row_run_id_mismatch() -> None:
    assert not _parallel_done_for_row(
        "run-a",
        {"run_id": "run-b", "status": "timeout"},
    )
    assert _parallel_done_for_row(
        "run-a",
        {"run_id": "run-a", "status": "timeout"},
    )


def test_parallel_done_legacy_state_without_run_id() -> None:
    assert _parallel_done_for_row("any", {"status": "timeout"})
    assert _parallel_done_for_row("any", {"status": "budget_done"})
    assert _parallel_done_for_row(None, {"status": "completed"})


def test_multi_status_lnr_terminal_wins() -> None:
    st, force = _multi_status_and_progress(
        "r1",
        "budget_expired",
        {},
        {"run_id": "r1", "status": "timeout"},
    )
    assert force is True
    assert "budget_expired" in st


def test_multi_status_parallel_timeout_while_monitor_running() -> None:
    st, force = _multi_status_and_progress(
        "r1",
        "running",
        {"timestamp": "2020-01-01T00:00:00", "status": "running"},
        {"run_id": "r1", "status": "timeout"},
    )
    assert force is True
    assert "DONE" in st
    assert "timeout" in st


def test_multi_status_parallel_budget_done_while_monitor_running() -> None:
    st, force = _multi_status_and_progress(
        "r1",
        "running",
        {"timestamp": "2020-01-01T00:00:00", "status": "running"},
        {"run_id": "r1", "status": "budget_done"},
    )
    assert force is True
    assert "DONE" in st
    assert "budget_done" in st


def test_multi_status_legacy_timeout_budget_exhaustion_is_labeled_budget_done() -> None:
    st, force = _multi_status_and_progress(
        "r1",
        "running",
        {"timestamp": "2020-01-01T00:00:00", "status": "running"},
        {"run_id": "r1", "status": "timeout", "error": "exceeded 43158s"},
    )
    assert force is True
    assert "DONE" in st
    assert "budget_done" in st


def test_multi_status_no_parallel_state() -> None:
    st, force = _multi_status_and_progress(
        "r1",
        "running",
        {"status": "running"},
        {},
    )
    assert force is False
    assert "run" in st or "STALE" in st
