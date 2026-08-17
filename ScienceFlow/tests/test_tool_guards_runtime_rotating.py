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

from scienceflow.core.agent.tools.tool_guards import RuntimeErrorGuard


def _bash_fail(cmd: str, output: str) -> ToolResult:
    return ToolResult(output=output, error="nonzero")


def test_runtime_rotating_coaching_once_on_third_distinct_type_failure() -> None:
    g = RuntimeErrorGuard(rotating_runtime_error_threshold=3, rotating_distinct_types_min=2)
    cmd = "python3 solution.py"
    out_ve = "Traceback (most recent call last):\nValueError: bad\n"
    out_ic = "Traceback (most recent call last):\npandas.errors.IntCastingNaNError: nan\n"

    m1 = g.on_tool_result("bash", {"command": cmd}, _bash_fail(cmd, out_ve))
    assert m1 is not None
    assert "Rotating errors" not in m1

    m2 = g.on_tool_result("bash", {"command": cmd}, _bash_fail(cmd, out_ic))
    assert m2 is not None
    assert "Rotating errors" not in m2

    m3 = g.on_tool_result("bash", {"command": cmd}, _bash_fail(cmd, out_ve))
    assert m3 is not None
    assert "Rotating errors detected" in m3

    m4 = g.on_tool_result("bash", {"command": cmd}, _bash_fail(cmd, out_ic))
    assert m4 is not None
    assert m4.count("Rotating errors detected") == 0

    ok = ToolResult(output="ok", error=None)
    assert g.on_tool_result("bash", {"command": cmd}, ok) is None


def test_dtype_diagnostic_hint_appended() -> None:
    g = RuntimeErrorGuard()
    cmd = "python3 solution.py"
    out = "Traceback (most recent call last):\nValueError: dtypes must be int for column x\n"
    msg = g.on_tool_result("bash", {"command": cmd}, _bash_fail(cmd, out))
    assert msg is not None
    assert "dtype / casting hint" in msg
    assert "pd.read_csv" in msg
