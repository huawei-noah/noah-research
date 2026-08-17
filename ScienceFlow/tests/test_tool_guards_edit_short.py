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

from scienceflow.core.agent.tools.tool_guards import EditFailureGuard


def test_edit_short_old_str_hint_on_not_found() -> None:
    g = EditFailureGuard(soft_threshold=2, hard_threshold=4, short_old_str_chars=40)
    msg = g.on_tool_result(
        "edit",
        {"old_str": "x" * 10},
        ToolResult(output="", error="old_str not found"),
    )
    assert msg is not None
    assert "very short" in msg
    assert "≥2 lines" in msg or "2 lines" in msg


def test_edit_short_hint_combined_with_soft_streak() -> None:
    g = EditFailureGuard(soft_threshold=2, hard_threshold=4, short_old_str_chars=40)
    err = ToolResult(output="", error="old_str not found")
    g.on_tool_result("edit", {"old_str": "short"}, err)
    m2 = g.on_tool_result("edit", {"old_str": "short"}, err)
    assert m2 is not None
    assert "very short" in m2
    assert "2+ times" in m2 or "failed 2+" in m2
