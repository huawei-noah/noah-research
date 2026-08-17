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

from deepcraft_core.tool import ToolResult

from scienceflow.core.agent.tools.tool_guards import (
    EditFailureGuard,
    GuardManager,
    NoProgressHardStopGuard,
    ToolGuard,
)


def test_no_progress_hard_stop_guard_triggers_after_threshold() -> None:
    state = {
        "initial_sha": "abc",
        "current_sha": "abc",
        "metric_token": None,
    }
    guard = NoProgressHardStopGuard(
        threshold=3,
        get_current_solution_sha=lambda: state["current_sha"],
        get_initial_solution_sha=lambda: state["initial_sha"],
        get_embedded_metric_token=lambda: state["metric_token"],
    )
    guard.reset()

    assert guard.on_round_complete(["bash"]) == []
    assert guard.on_round_complete(["bash"]) == []
    warn = guard.on_round_complete(["bash"])
    assert guard.should_terminate_run() is False
    assert len(warn) == 1
    assert warn[0].startswith("[Guard]")
    assert "No meaningful progress detected" in warn[0]
    assert "next round" in warn[0].lower()

    assert guard.on_round_complete(["bash"]) == []
    assert guard.on_round_complete(["bash"]) == []
    final = guard.on_round_complete(["bash"])
    assert guard.should_terminate_run() is True
    assert len(final) == 1
    assert final[0].startswith("[Guard]")
    assert "Stopping this run early" in final[0]


def test_no_progress_hard_stop_cross_round_grace() -> None:
    state = {
        "initial_sha": "abc",
        "current_sha": "abc",
        "metric_token": None,
    }
    guard = NoProgressHardStopGuard(
        threshold=3,
        get_current_solution_sha=lambda: state["current_sha"],
        get_initial_solution_sha=lambda: state["initial_sha"],
        get_embedded_metric_token=lambda: state["metric_token"],
        grace_rounds_after_peer_inject=1,
    )
    guard.reset()
    assert guard.on_round_complete(["bash"]) == []
    assert guard.on_round_complete(["bash"]) == []
    guard.arm_cross_round_grace(1)
    assert guard.on_round_complete(["bash"]) == []
    assert guard.should_terminate_run() is False
    assert guard.on_round_complete(["bash"]) == []
    assert guard.on_round_complete(["bash"]) == []
    warn = guard.on_round_complete(["bash"])
    assert guard.should_terminate_run() is False
    assert len(warn) == 1
    assert "next round" in warn[0].lower()
    assert guard.on_round_complete(["bash"]) == []
    assert guard.on_round_complete(["bash"]) == []
    final = guard.on_round_complete(["bash"])
    assert guard.should_terminate_run() is True
    assert len(final) == 1


class _PeerInjectGuard(ToolGuard):
    name = "peer_inject_guard"

    def on_tool_result(self, tool_name: str, args: dict, result: ToolResult) -> str | None:
        return None

    def on_round_complete(self, tool_names: list[str] | None, **kwargs) -> list[str]:
        return ["[peer] nudge"]


def test_guard_manager_peer_inject_prevents_no_progress_same_round() -> None:
    state = {
        "initial_sha": "abc",
        "current_sha": "abc",
        "metric_token": None,
    }
    np_guard = NoProgressHardStopGuard(
        threshold=1,
        get_current_solution_sha=lambda: state["current_sha"],
        get_initial_solution_sha=lambda: state["initial_sha"],
        get_embedded_metric_token=lambda: state["metric_token"],
        grace_rounds_after_peer_inject=1,
    )
    np_guard.reset()
    gm = GuardManager([_PeerInjectGuard(), np_guard])
    out = gm.process_round_complete(["bash"])
    assert any("[peer]" in m for m in out)
    assert not any("No meaningful progress" in m for m in out)
    assert np_guard.should_terminate_run() is False


def test_guard_manager_productivity_counters_only_count_success() -> None:
    gm = GuardManager([])
    gm.process_tool_result("write", {}, ToolResult(output="ok"))
    gm.process_tool_result("edit", {}, ToolResult(output="ok"))
    gm.process_tool_result("write", {}, ToolResult(output="", error="disk full"))

    counts = gm.productivity_counters()
    assert counts["write_success_count"] == 1
    assert counts["edit_success_count"] == 1


def test_no_progress_hard_stop_warning_cleared_after_edit() -> None:
    state = {
        "initial_sha": "abc",
        "current_sha": "abc",
        "metric_token": None,
    }
    guard = NoProgressHardStopGuard(
        threshold=2,
        get_current_solution_sha=lambda: state["current_sha"],
        get_initial_solution_sha=lambda: state["initial_sha"],
        get_embedded_metric_token=lambda: state["metric_token"],
    )
    guard.reset()
    assert guard.on_round_complete(["bash"]) == []
    warn = guard.on_round_complete(["bash"])
    assert len(warn) == 1
    assert guard.should_terminate_run() is False
    assert guard.on_round_complete(["edit"], write_edit_success=True) == []
    assert guard.on_round_complete(["bash"]) == []
    warn2 = guard.on_round_complete(["bash"])
    assert len(warn2) == 1
    assert guard.should_terminate_run() is False


def test_no_progress_failed_edit_does_not_reset_streak() -> None:
    state = {
        "initial_sha": "abc",
        "current_sha": "abc",
        "metric_token": None,
    }
    guard = NoProgressHardStopGuard(
        threshold=3,
        get_current_solution_sha=lambda: state["current_sha"],
        get_initial_solution_sha=lambda: state["initial_sha"],
        get_embedded_metric_token=lambda: state["metric_token"],
    )
    guard.reset()
    assert guard.on_round_complete(["bash"], write_edit_success=False) == []
    assert guard.on_round_complete(["bash"], write_edit_success=False) == []
    # Failed edit used to reset no-progress streak (because ``edit`` was in tool_names).
    warn = guard.on_round_complete(["edit"], write_edit_success=False)
    assert len(warn) == 1
    assert warn[0].startswith("[Guard]")
    assert "No meaningful progress detected" in warn[0]
    assert guard.should_terminate_run() is False


def test_edit_failure_guard_counts_blocked_edit() -> None:
    g = EditFailureGuard(soft_threshold=2, hard_threshold=4, short_old_str_chars=40)
    g.reset()
    m1 = g.on_tool_result("edit", {"old_str": "x"}, ToolResult(error="Edit blocked: re-read `solution.py`"))
    assert m1 and "blocked" in (m1 or "").lower()
    assert g.failure_streak == 1
    m2 = g.on_tool_result("edit", {"old_str": "x"}, ToolResult(error="Edit blocked: re-read `solution.py`"))
    assert m2 and "2+" in m2
    assert g.failure_streak == 2
