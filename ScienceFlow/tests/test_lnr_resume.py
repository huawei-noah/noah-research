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

import json
from types import SimpleNamespace

import pytest
from deepcraft_core import Message
from deepcraft_core.tool import ToolCall
from deepcraft_core.tool.base import Function

from scienceflow.core.tools.bash_tool import BashTool
from tests.lnr_resource_test_utils import event_types, make_observer, payloads

from scienceflow.solver.lnr.resume import (
    inspect_agent_resume_state,
    inspect_resume_memory_messages,
    resume_loaded_agent_from_memory,
)


def _bash_call(call_id: str = "call_1", command: str = "python train.py") -> ToolCall:
    return ToolCall(
        id=call_id,
        function=Function(name="bash", arguments=json.dumps({"command": command})),
    )


class _Record:
    def __init__(self, message: Message):
        self.memory_record = SimpleNamespace(message=message)


class _Memory:
    def __init__(self, messages: list[Message]):
        self.messages = list(messages)
        self.chat_history_memory = self

    def retrieve(self, window_size=None):
        _ = window_size
        return [_Record(m) for m in self.messages]

    def add_message(self, message: Message) -> None:
        self.messages.append(message)


class _ResumeObserver:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def resume_monitor_event(self, event: str, **kwargs) -> None:
        self.events.append({"event": event, **kwargs})


class _Agent:
    def __init__(self, messages: list[Message], resource_observer=None):
        self.memory = _Memory(messages)
        self.calls: list[tuple[ToolCall, object, int | None, int | None, bool]] = []
        self.completed: list[list[str]] = []
        self._resource_observer = resource_observer

    async def _run_one_tool_after_assistant_logged(
        self,
        tc,
        assistant_msg,
        effective_max,
        initial_max,
        *,
        log_assistant_thinking=True,
    ):
        self.calls.append((tc, assistant_msg, effective_max, initial_max, log_assistant_thinking))
        self.memory.add_message(Message.tool_message("ok", tc.function.name, tc.id))
        return None, effective_max

    def _lnr_on_round_complete(self, names):
        self.completed.append(list(names))


def test_resume_memory_detects_pending_single_tool_call() -> None:
    tc = _bash_call()
    assistant = Message(role="assistant", content="", tool_calls=[tc])

    state = inspect_resume_memory_messages([assistant])

    assert state.action == "execute_pending_tool"
    assert state.reason == "assistant_tool_call_without_tool_result"
    assert state.pending_tool_calls[0].tool_call_id == "call_1"
    assert state.pending_tool_calls[0].tool_name == "bash"


def test_resume_memory_continues_after_tool_result() -> None:
    tc = _bash_call()
    assistant = Message(role="assistant", content="", tool_calls=[tc])
    tool = Message.tool_message("ok", "bash", "call_1")

    state = inspect_resume_memory_messages([assistant, tool])

    assert state.action == "continue_llm"
    assert state.reason == "last_tool_result_ready_for_next_llm_round"


def test_resume_memory_continues_after_user_when_llm_failed_before_response() -> None:
    state = inspect_resume_memory_messages([Message.user_message("continue task")])

    assert state.action == "continue_llm"
    assert state.reason == "last_user_request_waiting_for_llm"


@pytest.mark.asyncio
async def test_resume_loaded_agent_executes_pending_tool_without_relogging_assistant() -> None:
    tc = _bash_call(command="python model.py")
    assistant = Message(role="assistant", content="run it", tool_calls=[tc])
    agent = _Agent([assistant])

    result = await resume_loaded_agent_from_memory(agent, effective_max=9, initial_max=9)

    assert result.executed_tool is True
    assert result.state.action == "execute_pending_tool"
    assert len(agent.calls) == 1
    called_tc, called_assistant, effective_max, initial_max, log_thinking = agent.calls[0]
    assert called_tc is tc
    assert called_assistant is assistant
    assert effective_max == 9
    assert initial_max == 9
    assert log_thinking is False
    assert agent.completed == [["bash"]]
    assert inspect_agent_resume_state(agent).action == "continue_llm"

@pytest.mark.asyncio
async def test_resume_loaded_agent_emits_resource_resume_monitor_events() -> None:
    observer = _ResumeObserver()
    tc = _bash_call(command="python3 -u predict.py 2>&1 | tail -20")
    assistant = Message(role="assistant", content="run predict", tool_calls=[tc])
    agent = _Agent([assistant], resource_observer=observer)

    result = await resume_loaded_agent_from_memory(agent, effective_max=7, initial_max=9)

    assert result.executed_tool is True
    names = [event["event"] for event in observer.events]
    assert names == ["pending_tool_resume_started", "pending_tool_resume_finished"]
    assert observer.events[0]["tool_name"] == "bash"
    assert observer.events[0]["command"] == "python3 -u predict.py 2>&1 | tail -20"
    assert observer.events[0]["status"] == "started"
    assert observer.events[1]["status"] == "finished"

@pytest.mark.asyncio
async def test_resume_pending_bash_tool_keeps_resource_monitor_heartbeat(tmp_path) -> None:
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=[],
        min_register_sec=0.0,
        stalled_stdout_sec=999.0,
        bash_monitor_all_enabled=True,
        kill_mode="recommend",
        review_state_enabled=True,
        review_heartbeat_sec=0.1,
        review_warmup_windows=0,
        review_value_windows=2,
        check_interval_sec=0.1,
    )
    command = 'python3 -u -c "import time; end=time.time()+1.8;\nwhile time.time()<end: pass" 2>&1 | tail -20'
    tc = _bash_call(command=command)
    assistant = Message(role="assistant", content="resume bash", tool_calls=[tc])

    class AgentWithBash(_Agent):
        def __init__(self):
            super().__init__([assistant], resource_observer=observer)
            self.tool = BashTool(
                workspace_dir=tmp_path,
                bash_timeout_sec=3.0,
                bash_timeout_slow_sec=3.0,
                resource_observer=observer,
            )

        async def _run_one_tool_after_assistant_logged(
            self,
            tc,
            assistant_msg,
            effective_max,
            initial_max,
            *,
            log_assistant_thinking=True,
        ):
            args = json.loads(tc.function.arguments)
            result = await self.tool.execute(args["command"])
            assert result.error is None
            self.calls.append((tc, assistant_msg, effective_max, initial_max, log_assistant_thinking))
            self.memory.add_message(Message.tool_message(result.output or "ok", tc.function.name, tc.id))
            return None, effective_max

    result = await resume_loaded_agent_from_memory(AgentWithBash(), effective_max=5, initial_max=5)

    assert result.executed_tool is True
    assert "resource_resume_monitor_event" in event_types(sm)
    heartbeat_payloads = payloads(sm, "resource_monitor_heartbeat")
    assert len(heartbeat_payloads) >= 2
    assert {payload["cpu_bucket"] for payload in heartbeat_payloads} & {"active", "heavy"}
