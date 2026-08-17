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

"""Same-round retries when ask_tool_stream raises empty streaming tool response."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from deepcraft_core import Memory, Message

from scienceflow.core.agent import ScienceAgent
from scienceflow.core.agent.runtime.run_loop import _is_retryable_empty_tool_stream_error
from scienceflow.core.agent.memory.reasoning_replay import (
    messages_with_synthetic_reasoning_replay,
)


async def _sleep_noop(*_a: object, **_k: object) -> None:
    return None


def test_incomplete_streaming_tool_call_is_retryable() -> None:
    assert _is_retryable_empty_tool_stream_error(
        ValueError("Incomplete streaming tool call(s) from LLM; finish_reason=length")
    )


def test_transport_read_error_is_retryable() -> None:
    assert _is_retryable_empty_tool_stream_error(httpx.ReadError("stream disconnected"))
    assert not _is_retryable_empty_tool_stream_error(httpx.ReadTimeout("deadline"))


@pytest.mark.asyncio
async def test_ask_tool_stream_retries_empty_response_same_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First call raises empty-stream ValueError; second succeeds without advancing round."""
    monkeypatch.setattr(asyncio, "sleep", _sleep_noop)

    class _FlakyLLM:
        model = "fake-retry"
        _last_call_input_tokens = 1
        _last_call_output_tokens = 2

        def __init__(self) -> None:
            self.calls = 0

        async def ask_tool_stream(self, **kwargs: object) -> object:
            self.calls += 1
            handle = kwargs.get("handle")
            if self.calls == 1:
                if handle is not None and not getattr(handle, "interrupted", False):
                    handle.finish()
                raise ValueError("Empty response from streaming tool LLM")
            if handle is not None and not getattr(handle, "interrupted", False):
                handle.finish()
            return SimpleNamespace(tool_calls=[], content="recovered")

    llm = _FlakyLLM()
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
        llm_tool_stream_max_attempts=5,
        llm_tool_stream_retry_base_delay_sec=0.0,
        llm_tool_stream_retry_max_delay_sec=0.0,
    )
    out = await agent.run("hi")
    assert llm.calls == 2
    assert "recovered" in out


@pytest.mark.asyncio
async def test_ask_tool_stream_exhausts_retries_then_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _sleep_noop)

    class _AlwaysEmpty:
        model = "fake-empty"

        def __init__(self) -> None:
            self.calls = 0

        async def ask_tool_stream(self, **kwargs: object) -> object:
            self.calls += 1
            handle = kwargs.get("handle")
            if handle is not None and not getattr(handle, "interrupted", False):
                handle.finish()
            raise ValueError("Empty response from streaming tool LLM")

    llm = _AlwaysEmpty()
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=2,
        llm_tool_stream_max_attempts=3,
        llm_tool_stream_retry_base_delay_sec=0.0,
        llm_tool_stream_retry_max_delay_sec=0.0,
    )
    with pytest.raises(ValueError, match="Empty response from streaming tool LLM"):
        await agent.run("x")
    assert llm.calls == 3


def test_synthetic_reasoning_replay_is_non_mutating() -> None:
    prior = Message.assistant_message("old assistant turn")
    user = Message.user_message("next")

    projected, count = messages_with_synthetic_reasoning_replay(
        [prior, user],
        purpose="unit test",
    )

    assert count == 1
    assert projected[0] is not prior
    assert projected[1] is user
    assert prior.reasoning_content is None
    assert projected[0].reasoning_content == "Synthetic replay placeholder for unit test."


@pytest.mark.asyncio
async def test_ask_tool_stream_retries_missing_reasoning_content_same_round(
    tmp_path: Path,
) -> None:
    class _ThinkingModeLLM:
        model = "deepseek-v4-flash"
        _last_call_input_tokens = 1
        _last_call_output_tokens = 2

        def __init__(self) -> None:
            self.calls = 0
            self.assistant_reasoning_by_call: list[list[str | None]] = []

        async def ask_tool_stream(self, **kwargs: object) -> object:
            self.calls += 1
            handle = kwargs.get("handle")
            messages = list(kwargs.get("messages") or [])
            self.assistant_reasoning_by_call.append([
                getattr(m, "reasoning_content", None)
                for m in messages
                if getattr(m, "role", None) == "assistant"
            ])
            if handle is not None and not getattr(handle, "interrupted", False):
                handle.finish()
            if self.calls == 1:
                raise RuntimeError(
                    "BadRequestError: Error code: 400 - "
                    "{'error': {'message': 'The `reasoning_content` in the "
                    "thinking mode must be passed back to the API.'}}"
                )
            return SimpleNamespace(tool_calls=[], content="resumed")

    memory = Memory(max_messages=50)
    prior = Message.assistant_message("prior assistant without provider metadata")
    memory.add_message(Message.user_message("prior task"))
    memory.add_message(prior)

    llm = _ThinkingModeLLM()
    agent = ScienceAgent(
        llm=llm,
        memory=memory,
        workspace_dir=tmp_path,
        max_steps=3,
        llm_tool_stream_max_attempts=2,
        llm_tool_stream_retry_base_delay_sec=0.0,
        llm_tool_stream_retry_max_delay_sec=0.0,
    )

    out = await agent.run("continue")

    assert "resumed" in out
    assert llm.calls == 2
    assert llm.assistant_reasoning_by_call[0] == [None]
    assert llm.assistant_reasoning_by_call[1] == [
        "Synthetic replay placeholder for thinking-mode resume compatibility.",
    ]
    assert prior.reasoning_content is None

@pytest.mark.asyncio
async def test_reasoning_replay_is_reused_after_first_provider_error(
    tmp_path: Path,
) -> None:
    class _StrictThinkingModeLLM:
        model = "deepseek-v4-flash"
        _last_call_input_tokens = 1
        _last_call_output_tokens = 2

        def __init__(self) -> None:
            self.calls = 0
            self.assistant_reasoning_by_call: list[list[str | None]] = []

        async def ask_tool_stream(self, **kwargs: object) -> object:
            self.calls += 1
            handle = kwargs.get("handle")
            messages = list(kwargs.get("messages") or [])
            assistant_reasoning = [
                getattr(m, "reasoning_content", None)
                for m in messages
                if getattr(m, "role", None) == "assistant"
            ]
            self.assistant_reasoning_by_call.append(assistant_reasoning)
            if handle is not None and not getattr(handle, "interrupted", False):
                handle.finish()
            if any(not rc for rc in assistant_reasoning):
                raise RuntimeError(
                    "BadRequestError: Error code: 400 - "
                    "{'error': {'message': 'The `reasoning_content` in the "
                    "thinking mode must be passed back to the API.'}}"
                )
            return SimpleNamespace(tool_calls=[], content=f"resumed-{self.calls}")

    memory = Memory(max_messages=50)
    memory.add_message(Message.user_message("prior task"))
    memory.add_message(Message.assistant_message("prior assistant without provider metadata"))

    llm = _StrictThinkingModeLLM()
    agent = ScienceAgent(
        llm=llm,
        memory=memory,
        workspace_dir=tmp_path,
        max_steps=3,
        llm_tool_stream_max_attempts=3,
        llm_tool_stream_retry_base_delay_sec=0.0,
        llm_tool_stream_retry_max_delay_sec=0.0,
    )

    first = await agent.run("continue")
    second = await agent.run("continue again")

    assert "resumed-2" in first
    assert "resumed-3" in second
    assert llm.calls == 3
    assert llm.assistant_reasoning_by_call[0] == [None]
    assert all(llm.assistant_reasoning_by_call[1])
    assert all(llm.assistant_reasoning_by_call[2])
    assert len(llm.assistant_reasoning_by_call[2]) >= 2

