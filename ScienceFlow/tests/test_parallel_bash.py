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

"""Tests for parallel bash tool-call classification and safety heuristics."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from deepcraft_core import Memory

from scienceflow.core.agent import (
    ScienceAgent,
    _bash_command_parallel_safe,
    _default_system_prompt,
    _tool_call_args_from_tc,
)


def test_default_system_prompt_parallel_bash_allows_multi_tool() -> None:
    on = _default_system_prompt(parallel_bash_enabled=True).lower()
    off = _default_system_prompt(parallel_bash_enabled=False).lower()
    assert "multiple tool calls" in on
    assert "never" in on
    assert "&&" in on
    assert "at most one" in off
    assert "exactly one `write` or one `edit`" in on
    assert "exactly one `write` or one `edit`" in off
    assert "uv pip" in on
    assert "avoid bare" in on
    assert "uv pip" in off


@pytest.mark.parametrize(
    ("cmd", "safe"),
    [
        ("ls -la", True),
        ("head -n 5 dataset/train.csv", True),
        ("grep pip requirements.txt", True),
        ("python3 solution.py", False),
        ("python solution.py", False),
        ("pip install numpy", False),
        ("pip3 show pandas", False),
        ("uv pip install x", False),
        ("uv run python script.py", False),
        ("git status", False),
        ("rm -rf /tmp/x", False),
        ("sudo ls", False),
        ("wget http://x", False),
        ("", False),
    ],
)
def test_bash_command_parallel_safe(cmd: str, safe: bool) -> None:
    assert _bash_command_parallel_safe(cmd) is safe


def test_tool_call_args_from_tc_roundtrip() -> None:
    tc = SimpleNamespace(
        function=SimpleNamespace(
            name="bash",
            arguments='{"command": "ls", "thought": "x"}',
        ),
    )
    args = _tool_call_args_from_tc(tc)
    assert args.get("command") == "ls"
    assert "thought" in args  # stripped later by agent


def _make_tc(name: str, command: str) -> SimpleNamespace:
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps({"command": command, "thought": "t"}),
        ),
    )


def _agent(tmp_path, **kwargs: object) -> ScienceAgent:
    return ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        **kwargs,
    )


def test_normalize_parallel_bash_mode(tmp_path) -> None:
    agent = _agent(tmp_path, parallel_bash_enabled=True)
    calls = [
        _make_tc("bash", "ls"),
        _make_tc("bash", "wc -l dataset/train.csv"),
    ]
    mode, out = agent._normalize_tool_calls_for_execution(calls)
    assert mode == "parallel_bash"
    assert len(out) == 2


def test_normalize_parallel_bash_disabled_sequential(tmp_path) -> None:
    agent = _agent(tmp_path, parallel_bash_enabled=False)
    calls = [
        _make_tc("bash", "ls"),
        _make_tc("bash", "wc -l dataset/train.csv"),
    ]
    mode, _ = agent._normalize_tool_calls_for_execution(calls)
    assert mode == "sequential"


def test_normalize_python_bash_sequential(tmp_path) -> None:
    agent = _agent(tmp_path, parallel_bash_enabled=True)
    calls = [
        _make_tc("bash", "ls"),
        _make_tc("bash", "python3 -c 'print(1)'"),
    ]
    mode, _ = agent._normalize_tool_calls_for_execution(calls)
    assert mode == "sequential"


@pytest.mark.parametrize(
    "names",
    [
        ["write", "write"],
        ["edit", "edit"],
        ["read", "write"],
        ["write", "bash"],
        ["edit", "bash"],
    ],
)
def test_normalize_blocks_multi_tool_bundle_containing_write_or_edit(
    tmp_path,
    names: list[str],
) -> None:
    agent = _agent(tmp_path, parallel_bash_enabled=True)
    calls = [_make_tc(name, "ls") for name in names]
    mode, out = agent._normalize_tool_calls_for_execution(calls)
    assert mode == "blocked_write_edit_bundle"
    assert out == calls


@pytest.mark.parametrize("name", ["write", "edit"])
def test_normalize_allows_single_write_or_edit(tmp_path, name: str) -> None:
    agent = _agent(tmp_path, parallel_bash_enabled=True)
    calls = [_make_tc(name, "ls")]
    mode, out = agent._normalize_tool_calls_for_execution(calls)
    assert mode == "single"
    assert out == calls
