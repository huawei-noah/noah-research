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

import shutil

import pytest

from scienceflow.core.tools.grep_tool import GrepTool


@pytest.mark.asyncio
async def test_grep_tool_handles_long_matching_line(tmp_path):
    if shutil.which("rg") is None:
        pytest.skip("rg is not installed")

    target = tmp_path / "long.log"
    target.write_text("needle " + "x" * 100_000 + "\n", encoding="utf-8")
    tool = GrepTool(workspace_dir=tmp_path)

    result = await tool.execute(pattern="needle", path="long.log")

    assert not result.error
    assert "long.log:1:" in (result.output or "")
    assert len(result.output or "") < 30_000
