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

"""Inspect persisted agent memory and choose an LNR resume point."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PendingToolCall:
    tool_call: Any
    assistant_message: Any
    tool_call_id: str
    tool_name: str
    arguments_json: str


@dataclass(frozen=True)
class ResumeMemoryState:
    action: str
    reason: str
    message_count: int
    last_role: str
    pending_tool_calls: tuple[PendingToolCall, ...] = ()

    def to_event(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "message_count": self.message_count,
            "last_role": self.last_role,
            "pending_tool_calls": [
                {
                    "tool_call_id": p.tool_call_id,
                    "tool_name": p.tool_name,
                    "arguments_json": p.arguments_json[:1000],
                }
                for p in self.pending_tool_calls
            ],
        }


def _record_message(record: Any) -> Any:
    try:
        return record.memory_record.message
    except Exception:
        pass
    if isinstance(record, dict):
        msg = record.get("message")
        return msg if isinstance(msg, dict) else record
    return record


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "").strip()
    return str(getattr(message, "role", "") or "").strip()


def _message_tool_call_id(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("tool_call_id") or "").strip()
    return str(getattr(message, "tool_call_id", "") or "").strip()


def _message_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, dict):
        calls = message.get("tool_calls") or []
    else:
        calls = getattr(message, "tool_calls", None) or []
    return list(calls) if isinstance(calls, (list, tuple)) else []


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or "").strip()
    return str(getattr(tool_call, "id", "") or "").strip()


def _tool_call_function(tool_call: Any) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get("function") or {}
    return getattr(tool_call, "function", None)


def _tool_call_name(tool_call: Any) -> str:
    fn = _tool_call_function(tool_call)
    if isinstance(fn, dict):
        return str(fn.get("name") or "").strip()
    return str(getattr(fn, "name", "") or "").strip()


def _tool_call_arguments(tool_call: Any) -> str:
    fn = _tool_call_function(tool_call)
    if isinstance(fn, dict):
        args = fn.get("arguments")
    else:
        args = getattr(fn, "arguments", "")
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(args or "")


def _pending_tool_calls_from_tail(messages: list[Any]) -> tuple[PendingToolCall, ...]:
    if not messages:
        return ()
    last = messages[-1]
    if _message_role(last) != "assistant":
        return ()
    calls = _message_tool_calls(last)
    if not calls:
        return ()
    # If the last message is the assistant tool-call message, no following tool result
    # exists yet. This is the precise interruption point we can replay safely.
    pending: list[PendingToolCall] = []
    for call in calls:
        pending.append(
            PendingToolCall(
                tool_call=call,
                assistant_message=last,
                tool_call_id=_tool_call_id(call),
                tool_name=_tool_call_name(call),
                arguments_json=_tool_call_arguments(call),
            )
        )
    return tuple(pending)


def inspect_resume_memory_messages(messages_in: list[Any]) -> ResumeMemoryState:
    messages = [_record_message(record) for record in messages_in]
    messages = [m for m in messages if m is not None]
    if not messages:
        return ResumeMemoryState(
            action="fresh_start",
            reason="empty_memory",
            message_count=0,
            last_role="",
        )

    last = messages[-1]
    last_role = _message_role(last)
    pending = _pending_tool_calls_from_tail(messages)
    if len(pending) == 1:
        return ResumeMemoryState(
            action="execute_pending_tool",
            reason="assistant_tool_call_without_tool_result",
            message_count=len(messages),
            last_role=last_role,
            pending_tool_calls=pending,
        )
    if len(pending) > 1:
        return ResumeMemoryState(
            action="pending_tool_bundle_unsupported",
            reason="assistant_multiple_tool_calls_without_tool_results",
            message_count=len(messages),
            last_role=last_role,
            pending_tool_calls=pending,
        )
    if last_role == "tool":
        reason = "last_tool_result_ready_for_next_llm_round"
    elif last_role == "user":
        reason = "last_user_request_waiting_for_llm"
    elif last_role == "assistant":
        reason = "last_assistant_message_complete"
    else:
        reason = "memory_tail_complete"
    return ResumeMemoryState(
        action="continue_llm",
        reason=reason,
        message_count=len(messages),
        last_role=last_role,
    )


def inspect_agent_resume_state(agent: Any) -> ResumeMemoryState:
    try:
        records = agent.memory.chat_history_memory.retrieve(window_size=None)
    except Exception:
        return ResumeMemoryState(
            action="fresh_start",
            reason="memory_unreadable",
            message_count=0,
            last_role="",
        )
    return inspect_resume_memory_messages(list(records or []))
