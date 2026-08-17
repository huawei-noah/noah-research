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

"""Tests for required ``thought`` tool parameter injection and memory stripping."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scienceflow.core.agent import (
    _compress_tool_call_for_memory,
    inject_thought_into_tool_params,
)
from scienceflow.core.agent.memory.memory_utils import _THOUGHT_TOOL_PARAM
from scienceflow.core.tools import create_tool_collection


def test_inject_thought_adds_required_property() -> None:
    raw = create_tool_collection(".").to_params()
    out = inject_thought_into_tool_params(raw)
    assert len(out) == len(raw)
    for item in out:
        assert item.get("type") == "function"
        fn = item["function"]
        props = fn["parameters"]["properties"]
        assert "thought" in props
        assert props["thought"]["type"] == "string"
        assert "required" in fn["parameters"]
        assert "thought" in fn["parameters"]["required"]


def test_thought_schema_uses_short_required_reason() -> None:
    desc = _THOUGHT_TOOL_PARAM["description"]
    assert "One to two concise sentences" in desc
    assert "what this tool call will do and why" in desc
    assert "multi-paragraph" in desc
    assert "Two to three" not in desc
    assert "2-3" not in desc


def test_inject_thought_does_not_mutate_original() -> None:
    raw = create_tool_collection(".").to_params()
    before = json.dumps(raw, sort_keys=True)
    inject_thought_into_tool_params(raw)
    after = json.dumps(raw, sort_keys=True)
    assert before == after


def test_compress_tool_call_drops_thought_from_memory() -> None:
    tc = SimpleNamespace(
        id="c1",
        type="function",
        function=SimpleNamespace(
            name="bash",
            arguments=json.dumps(
                {
                    "command": "echo hi",
                    "thought": "I need to verify the shell works.",
                },
            ),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    args = json.loads(out.function.arguments)
    assert "thought" not in args
    assert args.get("command") == "echo hi"


def test_workspace_interaction_log_thought_head_style() -> None:
    from scienceflow.utils.workspace_interaction_log import _head_style_for_first_line

    assert _head_style_for_first_line("[thought] probe") is not None
