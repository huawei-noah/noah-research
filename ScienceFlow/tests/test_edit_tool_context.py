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

"""Tests for EditTool._change_context (post-edit preview)."""

from __future__ import annotations

import pytest

from scienceflow.core.tools.edit_tool import EditTool


def test_change_context_shows_new_content() -> None:
    new = "line1\nnew_line\nline3\n"
    ctx = EditTool._change_context(new, "new_line")
    assert "new_line" in ctx
    assert "old_line" not in ctx
    assert ">>" in ctx


def test_change_context_hint_offset_disambiguates() -> None:
    text = "foo\nbar\nfoo\nbaz\n"
    ctx1 = EditTool._change_context(text, "foo")
    assert "1|foo" in ctx1.replace(" ", "")
    # Second "foo" starts after "foo\nbar\n" = 8
    ctx2 = EditTool._change_context(text, "foo", hint_offset=8)
    assert "3|foo" in ctx2.replace(" ", "")


@pytest.mark.asyncio
async def test_edit_not_found_returns_copyable_anchor(tmp_path) -> None:
    path = tmp_path / "solution.py"
    path.write_text(
        "\n".join(
            [
                "def engineer_features(df):",
                "    df = df.copy()",
                "    df['ratio'] = df['a'] / df['b']",
                "    return df",
            ],
        ),
        encoding="utf-8",
    )
    tool = EditTool(workspace_dir=tmp_path)

    result = await tool.execute(
        path="solution.py",
        old_str="    df['ratio'] = df['a'] / df['missing']",
        new_str="    df['ratio'] = 1",
    )

    assert result.error is not None
    assert "[edit-anchor suggestion: solution.py lines" in result.error
    assert "Enclosing symbol: function engineer_features lines 1-4" in result.error
    assert "Suggested old_str anchor (copy exactly between markers):" in result.error
    assert "    df['ratio'] = df['a'] / df['b']" in result.error
