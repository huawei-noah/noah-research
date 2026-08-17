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

import asyncio
import types

from deepcraft_core import Message

from scienceflow.solver.lnr.resource_advisory import (
    RESOURCE_ADVISORY_BEGIN,
    RESOURCE_ADVISORY_END,
    parse_inline_resource_advisory_response,
)
from scienceflow.solver.lnr.solver import LnrSolver


class _FakeMemoryCtx:
    def build_messages_for_llm(self):
        return [Message.user_message("stable main memory")]


class _FakeLLM:
    pass


class _FakeStreamingLLM:
    def __init__(self):
        self.calls = []

    async def ask_tool_stream(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            content=(
                f"{RESOURCE_ADVISORY_BEGIN}\n"
                "preference: safe_to_stop\n"
                "confidence: high\n"
                "reason: route has no useful progress and can be replanned\n"
                f"{RESOURCE_ADVISORY_END}"
            )
        )


class _FakeAgent:
    def __init__(self, *, tool_call: bool = False):
        self.llm = _FakeLLM()
        self._memory_ctx = _FakeMemoryCtx()
        self._tools_with_thought = [{"type": "function", "function": {"name": "bash"}}]
        self._parallel_llm_tool_calls = True
        self.state = types.SimpleNamespace(name="IDLE")
        self.memory = types.SimpleNamespace(add_message=lambda *_args, **_kwargs: None)
        self.tool_call = tool_call
        self.run_calls = []

    async def run(self, request=None, *, first_round_tool_choice=None):
        self.run_calls.append((request, first_round_tool_choice))
        assert request is None
        assert self._lnr_transient_tool_choice_none is False
        assert self._lnr_transient_turn_kind == "inline_resource_advisory"
        assert "RESOURCE_ADVISORY_REQUEST" in self._lnr_transient_user_prompt
        callback = self._lnr_text_only_callback
        if self.tool_call:
            self._lnr_resource_advisory_tool_call_rejected = True
            return await callback(
                agent=self,
                assistant_text="RESOURCE_ADVISORY_TOOL_CALL_REJECTED: bash",
                round_idx=0,
                max_steps=1,
            )
        return await callback(
            agent=self,
            assistant_text=(
                f"{RESOURCE_ADVISORY_BEGIN}\n"
                "preference: timebox_continue\n"
                "confidence: medium\n"
                "reason: metric is still improving but below current best\n"
                "commitment: produce a new validation metric within 30m\n"
                "expected_next_artifact: submission.csv\n"
                f"{RESOURCE_ADVISORY_END}"
            ),
            round_idx=0,
            max_steps=1,
        )


def _solver_with_agent(agent):
    solver = object.__new__(LnrSolver)
    solver.lhr = types.SimpleNamespace(
        resource_main_agent_advisory_enabled=True,
        resource_main_agent_advisory_timeout_sec=60.0,
        resource_advisory_mode="inline_memory_edit",
    )
    solver._resource_main_agent_ref = agent
    return solver


def test_inline_resource_advisory_response_parser():
    parsed, block, reason = parse_inline_resource_advisory_response(
        f"prefix\n{RESOURCE_ADVISORY_BEGIN}\n"
        "preference: continue\n"
        "confidence: high\n"
        "reason: route still has useful signal\n"
        f"{RESOURCE_ADVISORY_END}\ntrailer"
    )
    assert reason == ""
    assert parsed["preference"] == "continue"
    assert parsed["confidence"] == "high"
    assert "RESOURCE_ADVISORY_RESPONSE_BEGIN" in block


def test_resource_advisory_uses_inline_main_agent_turn_and_memory_edit():
    agent = _FakeAgent()
    solver = _solver_with_agent(agent)

    decider = solver._make_resource_main_agent_advisory_decider()
    result = asyncio.run(decider({"proposal_id": "p1", "proposal_type": "kill_proposal", "reason_code": "low_value"}))

    assert agent.run_calls == [(None, None)]
    assert result["preference"] == "timebox_continue"
    assert result["confidence"] == "medium"
    assert result["advisory_mode"] == "inline_memory_edit"
    assert result["memory_edit_applied"] is True
    assert result["advisory_tool_call_rejected"] is False
    assert result["_audit"]["event_version"] == 1
    assert getattr(agent, "_lnr_transient_user_prompt") == ""
    assert getattr(agent, "_lnr_resource_advisory_text_pending") is False


def test_resource_advisory_tool_call_is_rejected_not_executed():
    agent = _FakeAgent(tool_call=True)
    solver = _solver_with_agent(agent)

    decider = solver._make_resource_main_agent_advisory_decider()
    result = asyncio.run(decider({"proposal_id": "p2", "proposal_type": "kill_proposal"}))

    assert result["preference"] == "unknown"
    assert result["confidence"] == "low"
    assert result["advisory_tool_call_rejected"] is True
    assert result["_audit"]["tool_call_rejected"] is True


def test_resource_advisory_defers_when_agent_not_idle_without_stream_llm():
    agent = _FakeAgent()
    agent.state = types.SimpleNamespace(name="RUNNING")
    solver = _solver_with_agent(agent)

    decider = solver._make_resource_main_agent_advisory_decider()
    result = asyncio.run(decider({"proposal_id": "p3", "proposal_type": "kill_proposal"}))

    assert result["preference"] == "unknown"
    assert result["reason"].startswith("inline_resource_advisory_deferred:")
    assert agent.run_calls == []


def test_resource_advisory_captures_blocked_state_with_no_tools_stream_call():
    agent = _FakeAgent()
    agent.llm = _FakeStreamingLLM()
    agent.state = types.SimpleNamespace(name="RUNNING")
    solver = _solver_with_agent(agent)

    decider = solver._make_resource_main_agent_advisory_decider()
    result = asyncio.run(decider({"proposal_id": "p4", "proposal_type": "kill_proposal", "reason_code": "low_progress"}))

    assert agent.run_calls == []
    assert len(agent.llm.calls) == 1
    assert agent.llm.calls[0]["tools"] == []
    assert agent.llm.calls[0]["tool_choice"] == "none"
    assert result["preference"] == "safe_to_stop"
    assert result["confidence"] == "high"
    assert result["blocked_state"] is True
    assert result["advisory_status"] == "captured"
    assert result["_audit"]["status"] == "captured"
    assert getattr(agent, "_lnr_transient_user_prompt", "") == ""


def test_resource_advisory_safe_state_uses_cache_friendly_direct_no_tools_call():
    agent = _FakeAgent()
    agent.llm = _FakeStreamingLLM()
    agent.state = types.SimpleNamespace(name="IDLE")
    solver = _solver_with_agent(agent)

    decider = solver._make_resource_main_agent_advisory_decider()
    result = asyncio.run(decider({"proposal_id": "p5", "proposal_type": "kill_proposal", "reason_code": "low_value"}))

    assert agent.run_calls == []
    assert len(agent.llm.calls) == 1
    call = agent.llm.calls[0]
    assert call["tools"] == []
    assert call["tool_choice"] == "none"
    assert call["parallel_tool_calls"] is False
    assert "RESOURCE_ADVISORY_REQUEST" in call["messages"][-1].content
    assert result["preference"] == "safe_to_stop"
    assert result["confidence"] == "high"
    assert result["blocked_state"] is False
    assert result["advisory_status"] == "captured"
    assert result["_audit"]["defer_reason"] == "cache_friendly_direct_advisory"
    assert getattr(agent, "_lnr_transient_user_prompt", "") == ""
