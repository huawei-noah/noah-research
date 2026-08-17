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

"""Sliding-window tool-turn sanitization (multi-tool assistant + tool messages)."""

from __future__ import annotations

from deepcraft_core import Memory
from deepcraft_core import Message

from scienceflow.core.agent.run_control.embedded_fullrun import EmbeddedFullRunMixin
from scienceflow.core.agent.tool_exec.single import SingleToolExecMixin
from scienceflow.core.mem.memory_context import (
    _ensure_complete_tool_turn_prefix,
    _finalize_messages_for_llm,
    _sanitize_orphan_tool_messages,
)


class _RunControlDeferralDummy(SingleToolExecMixin, EmbeddedFullRunMixin):
    pass


def test_complete_two_tool_turn_unchanged() -> None:
    msgs = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "grep", "arguments": "{}"},
                },
            ],
        ),
        Message(role="tool", content="ra", name="read", tool_call_id="a"),
        Message(role="tool", content="rb", name="grep", tool_call_id="b"),
        Message(role="user", content="next"),
    ]
    out = _ensure_complete_tool_turn_prefix(msgs)
    assert len(out) == 4
    assert out[0].role == "assistant"


def test_incomplete_multi_tool_turn_dropped() -> None:
    """Only one tool response after assistant with two tool_calls — strip broken prefix."""
    msgs = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
            ],
        ),
        Message(role="tool", content="only one", name="read", tool_call_id="a"),
        Message(role="user", content="keep"),
    ]
    out = _ensure_complete_tool_turn_prefix(msgs)
    assert len(out) == 1
    assert out[0].role == "user"
    assert out[0].content == "keep"


def test_orphan_tool_at_start_stripped() -> None:
    out = _ensure_complete_tool_turn_prefix(
        [
            Message(role="tool", content="x", name="read", tool_call_id="z"),
            Message(role="user", content="u"),
        ],
    )
    assert len(out) == 1
    assert out[0].role == "user"


def test_mismatched_tool_call_id_dropped() -> None:
    msgs = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
            ],
        ),
        Message(role="tool", content="wrong", name="read", tool_call_id="other"),
    ]
    out = _ensure_complete_tool_turn_prefix(msgs)
    assert out == []


def test_empty_messages() -> None:
    assert _ensure_complete_tool_turn_prefix([]) == []


def test_extra_tool_after_complete_turn_downgraded_to_user() -> None:
    """Assistant declares one tool_call but two tool outputs (e.g. legacy synthetic auto-qt)."""
    msgs = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "write_id",
                    "type": "function",
                    "function": {"name": "write", "arguments": "{}"},
                },
            ],
        ),
        Message(role="tool", content="wrote", name="write", tool_call_id="write_id"),
        Message(
            role="tool",
            content="bash out",
            name="bash",
            tool_call_id="write_id_auto_qt",
        ),
        Message(role="user", content="next"),
    ]
    out = _sanitize_orphan_tool_messages(msgs)
    assert len(out) == 4
    assert out[0].role == "assistant"
    assert out[1].role == "tool"
    assert out[2].role == "user"
    assert "downgraded" in (out[2].content or "")
    assert "bash out" in (out[2].content or "")
    assert out[3].role == "user"


def test_interleaved_user_between_multi_tool_outputs_is_sanitized() -> None:
    """Legacy guard injection can split a multi-tool turn; LLM view must stay valid."""
    msgs = [
        Message(role="user", content="prior context"),
        Message(
            role="assistant",
            content="checking",
            tool_calls=[
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                },
            ],
        ),
        Message(role="tool", content="first", name="bash", tool_call_id="a"),
        Message(role="user", content="[MLEBENCH-INVALID] fix submission"),
        Message(role="tool", content="second", name="bash", tool_call_id="b"),
    ]
    out = _finalize_messages_for_llm(msgs)
    assert out[1].role == "assistant"
    assert not getattr(out[1], "tool_calls", None)
    assert "converted to text" in (out[1].content or "")
    assert [m.role for m in out] == ["user", "assistant", "user", "user", "user"]
    assert "first" in (out[2].content or "")
    assert "second" in (out[4].content or "")


def test_run_control_user_message_defers_inside_tool_bundle() -> None:
    dummy = _RunControlDeferralDummy()
    dummy.memory = Memory(max_messages=10)
    dummy._run_control_user_injections = 0
    dummy._tool_bundle_deferred_messages = []

    dummy._inject_run_control_user_message("[MLEBENCH-INVALID] fix submission")

    assert dummy.memory.messages == []
    assert len(dummy._tool_bundle_deferred_messages) == 1
    assert dummy._tool_bundle_deferred_messages[0].role == "user"
    assert dummy._run_control_user_injections == 1
