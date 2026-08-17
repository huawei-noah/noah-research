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

"""Tests for BashPythonSourceDumpGuard."""

from __future__ import annotations

from deepcraft_core.tool import ToolResult

from scienceflow.core.agent.tools.tool_guards import BashPythonSourceDumpGuard


def test_parallel_bash_py_dumps_inject_once() -> None:
    g = BashPythonSourceDumpGuard(parallel_threshold=2, consecutive_rounds=2)
    cmd_a = "cat -n solution.py | head -60"
    cmd_b = "cat -n solution.py | sed -n '60,130p'"
    g.on_tool_result("bash", {"command": cmd_a}, ToolResult(output="x", error=None))
    g.on_tool_result("bash", {"command": cmd_b}, ToolResult(output="y", error=None))
    msgs = g.on_round_complete(["bash", "bash"])
    assert len(msgs) == 1
    assert "compact" in msgs[0].lower()
    assert "bash" in msgs[0].lower()


def test_single_bash_two_rounds_injects() -> None:
    g = BashPythonSourceDumpGuard(parallel_threshold=2, consecutive_rounds=2)
    cmd = "cat solution.py"
    g.on_tool_result("bash", {"command": cmd}, ToolResult(output="a", error=None))
    assert g.on_round_complete(["bash"]) == []
    g.on_tool_result("bash", {"command": cmd}, ToolResult(output="b", error=None))
    msgs = g.on_round_complete(["bash"])
    assert msgs
    assert "targeted" in msgs[0].lower()
