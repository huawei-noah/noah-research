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

from scienceflow.core.agent.tools.tool_guards import NoSuccessfulSolutionRunGuard


def test_no_success_soft_then_hard_and_reset_on_success() -> None:
    g = NoSuccessfulSolutionRunGuard(soft_threshold=2, hard_threshold=4)
    cmd = "python3 solution.py"

    g.on_tool_result("bash", {"command": cmd}, ToolResult(output="err", error="fail"))
    assert g.on_round_complete([]) == []
    soft_msgs = g.on_round_complete([])
    assert soft_msgs != []
    soft = soft_msgs[0]
    assert soft.startswith("[Guard]")
    assert "No successful `python3 solution.py`" in soft
    assert "NROWS=20" in soft or "python3 -c" in soft

    assert g.on_round_complete([]) == []
    hard_msgs = g.on_round_complete([])
    assert hard_msgs != []
    hard = hard_msgs[0]
    assert hard.startswith("[Guard]")
    assert "minimal runnable" in hard.lower() or "minimal" in hard

    g.on_tool_result("bash", {"command": cmd}, ToolResult(output="ok", error=None))
    assert g.on_round_complete([]) == []

    g.on_tool_result("bash", {"command": cmd}, ToolResult(output="err", error="fail"))
    assert g.on_round_complete([]) == []
    assert g.on_round_complete([]) != []
