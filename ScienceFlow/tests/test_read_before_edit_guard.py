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

"""Tests for read-before-edit enforcement on ``ScienceAgent._execute_tool_maybe_edit_guard``."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from deepcraft_core.tool import ToolResult


def _bare_agent_for_edit_guard(tmp_path: Path):
    from scienceflow.core.agent import ScienceAgent

    agent = ScienceAgent.__new__(ScienceAgent)
    agent._workspace_dir = tmp_path
    agent._sandbox = True
    agent._path_guard_extra_roots = ()
    agent._recent_read_history = deque(maxlen=3)
    agent._last_read_sha_by_path = {}
    execute_mock = AsyncMock(return_value=ToolResult(output="edited"))
    tools = MagicMock()
    tools.execute = execute_mock
    # ScienceAgent is a Pydantic model; instances from __new__ without __init__
    # must bypass normal setattr for arbitrary assignments.
    object.__setattr__(agent, "availableTools", tools)
    return agent, execute_mock


@pytest.mark.asyncio
async def test_edit_blocked_without_prior_read(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text("a = 1\n", encoding="utf-8")
    agent, execute_mock = _bare_agent_for_edit_guard(tmp_path)
    res = await agent._execute_tool_maybe_edit_guard(
        "edit",
        {"path": "solution.py", "old_str": "a", "new_str": "b"},
    )
    assert res.error
    assert "Edit blocked" in (res.error or "")
    assert "re-read" in (res.error or "")
    execute_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_allowed_after_read_record(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("x\n", encoding="utf-8")
    agent, execute_mock = _bare_agent_for_edit_guard(tmp_path)
    agent._record_read_for_edit_guard("foo.py")
    res = await agent._execute_tool_maybe_edit_guard(
        "edit",
        {"path": "foo.py", "old_str": "x", "new_str": "y"},
    )
    assert not res.error
    execute_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_edit_blocked_after_file_changed_after_read(tmp_path: Path) -> None:
    p = tmp_path / "bar.py"
    p.write_text("v1\n", encoding="utf-8")
    agent, execute_mock = _bare_agent_for_edit_guard(tmp_path)
    agent._record_read_for_edit_guard("bar.py")
    p.write_text("v2\n", encoding="utf-8")
    res = await agent._execute_tool_maybe_edit_guard(
        "edit",
        {"path": "bar.py", "old_str": "v2", "new_str": "v3"},
    )
    assert res.error
    assert "Edit blocked" in (res.error or "")
    execute_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_edit_refreshes_guard_for_followup_edit(tmp_path: Path) -> None:
    p = tmp_path / "chain.py"
    p.write_text("v1\n", encoding="utf-8")
    agent, execute_mock = _bare_agent_for_edit_guard(tmp_path)
    agent._record_read_for_edit_guard("chain.py")

    writes = iter(["v2\n", "v3\n"])

    async def _apply_edit(*_args, **_kwargs):
        p.write_text(next(writes), encoding="utf-8")
        return ToolResult(output="edited")

    execute_mock.side_effect = _apply_edit
    first = await agent._execute_tool_maybe_edit_guard(
        "edit",
        {"path": "chain.py", "old_str": "v1", "new_str": "v2"},
    )
    second = await agent._execute_tool_maybe_edit_guard(
        "edit",
        {"path": "chain.py", "old_str": "v2", "new_str": "v3"},
    )

    assert not first.error
    assert not second.error
    assert execute_mock.await_count == 2


@pytest.mark.asyncio
async def test_successful_write_seeds_guard_for_followup_edit(tmp_path: Path) -> None:
    p = tmp_path / "new_solution.py"
    agent, execute_mock = _bare_agent_for_edit_guard(tmp_path)

    async def _apply_tool(*_args, **kwargs):
        tool_input = kwargs.get("tool_input") or {}
        if tool_input.get("path") == "new_solution.py" and "content" in tool_input:
            p.write_text(str(tool_input["content"]), encoding="utf-8")
            return ToolResult(output="written")
        p.write_text("y = 2\n", encoding="utf-8")
        return ToolResult(output="edited")

    execute_mock.side_effect = _apply_tool
    written = await agent._execute_tool_maybe_edit_guard(
        "write",
        {"path": "new_solution.py", "content": "x = 1\n"},
    )
    edited = await agent._execute_tool_maybe_edit_guard(
        "edit",
        {"path": "new_solution.py", "old_str": "x = 1", "new_str": "y = 2"},
    )

    assert not written.error
    assert not edited.error
    assert execute_mock.await_count == 2


@pytest.mark.asyncio
async def test_non_edit_tools_bypass_guard(tmp_path: Path) -> None:
    agent, execute_mock = _bare_agent_for_edit_guard(tmp_path)
    execute_mock.return_value = ToolResult(output="read content")
    res = await agent._execute_tool_maybe_edit_guard(
        "read",
        {"path": "solution.py"},
    )
    assert not res.error
    execute_mock.assert_awaited_once()
