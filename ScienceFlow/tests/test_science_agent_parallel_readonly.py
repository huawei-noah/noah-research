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

"""ScienceAgent: parallel read-only file tool calls (P8)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from deepcraft_core import Memory

from scienceflow.core.agent import ScienceAgent


class _FakeLLMStream:
    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self._last_call_input_tokens: int | None = 100
        self._last_call_output_tokens: int | None = 7
        self.model = "fake-model"

    async def ask_tool_stream(self, *, handle: Any, **kwargs: Any) -> Any:
        if not self._replies:
            raise RuntimeError("no fake replies left")
        msg = self._replies.pop(0)
        content = getattr(msg, "content", None) or ""
        if content:
            await handle.put(content)
        handle.finish()
        return msg


def _tc(tc_id: str, name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args),
        ),
    )


@pytest.mark.asyncio
async def test_parallel_read_two_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    dual = SimpleNamespace(
        tool_calls=[
            _tc("id_a", "read", {"path": "a.txt"}),
            _tc("id_b", "read", {"path": "b.txt"}),
        ],
        content="",
    )
    llm = _FakeLLMStream(
        [dual, SimpleNamespace(tool_calls=[], content="summary")],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )
    out = await agent.run("inspect both files")
    assert "summary" in out or out == "summary"
    msgs = agent.memory.messages
    roles = [m.role for m in msgs]
    assert "assistant" in roles
    tool_msgs = [m for m in msgs if m.role == "tool"]
    assert len(tool_msgs) == 2
    ids = sorted([getattr(m, "tool_call_id", None) or "" for m in tool_msgs])
    assert ids == ["id_a", "id_b"]


@pytest.mark.asyncio
async def test_parallel_glob_and_ls(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.csv").write_text("x\n1\n", encoding="utf-8")
    dual = SimpleNamespace(
        tool_calls=[
            _tc("id_glob", "glob", {"pattern": "**/*.csv"}),
            _tc("id_ls", "ls", {"path": "."}),
        ],
        content="",
    )
    llm = _FakeLLMStream(
        [dual, SimpleNamespace(tool_calls=[], content="summary")],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )
    await agent.run("inspect files")
    tool_msgs = [m for m in agent.memory.messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    by_name = {getattr(m, "name", ""): m.content or "" for m in tool_msgs}
    assert "data/train.csv" in by_name["glob"]
    assert "DIR data/" in by_name["ls"]


@pytest.mark.asyncio
async def test_mixed_read_write_bundle_blocked_then_single_write_succeeds(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("only read this", encoding="utf-8")
    mixed = SimpleNamespace(
        tool_calls=[
            _tc("id_r", "read", {"path": "a.txt"}),
            _tc("id_w", "write", {"path": "out.txt", "content": "blocked"}),
        ],
        content="",
    )
    single_write = SimpleNamespace(
        tool_calls=[
            _tc("id_w2", "write", {"path": "out.txt", "content": "allowed"}),
        ],
        content="",
    )
    llm = _FakeLLMStream(
        [mixed, single_write, SimpleNamespace(tool_calls=[], content="done")],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )
    await agent.run("go")
    msgs = agent.memory.messages
    asst_with_tools = [m for m in msgs if m.role == "assistant" and getattr(m, "tool_calls", None)]
    first_batch = asst_with_tools[0].tool_calls or []
    assert len(first_batch) == 1
    names: list[str] = []
    for tc in first_batch:
        if isinstance(tc, dict):
            fn = tc.get("function")
            if isinstance(fn, dict):
                names.append(str(fn.get("name", "")))
            elif hasattr(fn, "name"):
                names.append(str(getattr(fn, "name", "")))
        elif hasattr(tc, "function"):
            f = getattr(tc, "function", None)
            if f is not None and hasattr(f, "name"):
                names.append(str(getattr(f, "name", "")))
    assert names == ["write"]
    tool_msgs = [m for m in msgs if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert getattr(tool_msgs[0], "name", "") == "write"
    assert getattr(tool_msgs[0], "tool_call_id", "") == "id_w2"
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "allowed"
    user_text = "\n".join(str(m.content or "") for m in msgs if m.role == "user")
    assert "No tools were executed" in user_text
    assert "exactly one `write` or exactly one `edit`" in user_text


@pytest.mark.asyncio
async def test_single_bash_still_one_round(tmp_path: Path) -> None:
    llm = _FakeLLMStream(
        [
            SimpleNamespace(
                tool_calls=[_tc("b1", "bash", {"command": "echo hi"})],
                content="",
            ),
            SimpleNamespace(tool_calls=[], content="ok"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )
    await agent.run("run")
    msgs = agent.memory.messages
    assert any(m.role == "tool" for m in msgs)
