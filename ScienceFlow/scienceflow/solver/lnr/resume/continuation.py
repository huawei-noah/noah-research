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

"""Execute safe LNR resume continuations from persisted agent memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from scienceflow.solver.lnr.resume.memory_state import (
    ResumeMemoryState,
    inspect_agent_resume_state,
)


def _pending_bash_command(pending: Any) -> str:
    if str(getattr(pending, "tool_name", "") or "") != "bash":
        return ""
    try:
        args = json.loads(str(getattr(pending, "arguments_json", "") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or "")


def _emit_resume_monitor_event(agent: Any, event: str, pending: Any, *, status: str = "", payload: dict[str, Any] | None = None) -> None:
    observer = getattr(agent, "_resource_observer", None)
    hook = getattr(observer, "resume_monitor_event", None)
    if not callable(hook):
        return
    try:
        hook(
            event,
            tool_name=str(getattr(pending, "tool_name", "") or ""),
            tool_call_id=str(getattr(pending, "tool_call_id", "") or ""),
            command=_pending_bash_command(pending),
            status=status or event,
            payload=dict(payload or {}),
        )
    except Exception:
        pass


@dataclass(frozen=True)
class ResumeContinuationResult:
    state: ResumeMemoryState
    executed_tool: bool = False
    early_out: str | None = None
    effective_max: int | None = None

    def to_event(self) -> dict[str, Any]:
        payload = self.state.to_event()
        payload.update(
            {
                "executed_tool": self.executed_tool,
                "early_out": bool(self.early_out),
                "effective_max": self.effective_max,
            }
        )
        return payload


async def resume_loaded_agent_from_memory(
    agent: Any,
    *,
    effective_max: int | None,
    initial_max: int | None,
) -> ResumeContinuationResult:
    state = inspect_agent_resume_state(agent)
    if state.action != "execute_pending_tool" or len(state.pending_tool_calls) != 1:
        return ResumeContinuationResult(state=state, effective_max=effective_max)

    pending = state.pending_tool_calls[0]
    runner = getattr(agent, "_run_one_tool_after_assistant_logged", None)
    if not callable(runner):
        return ResumeContinuationResult(
            state=ResumeMemoryState(
                action="resume_executor_unavailable",
                reason="agent_has_no_single_tool_executor",
                message_count=state.message_count,
                last_role=state.last_role,
                pending_tool_calls=state.pending_tool_calls,
            ),
            effective_max=effective_max,
        )

    _emit_resume_monitor_event(
        agent,
        "pending_tool_resume_started",
        pending,
        status="started",
        payload={"effective_max": effective_max, "initial_max": initial_max},
    )
    try:
        early_out, new_effective_max = await runner(
            pending.tool_call,
            pending.assistant_message,
            effective_max,
            initial_max,
            log_assistant_thinking=False,
        )
    except BaseException as exc:
        _emit_resume_monitor_event(
            agent,
            "pending_tool_resume_failed",
            pending,
            status="failed",
            payload={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    _emit_resume_monitor_event(
        agent,
        "pending_tool_resume_finished",
        pending,
        status="finished",
        payload={"early_out": bool(early_out), "effective_max": new_effective_max},
    )
    on_round_complete = getattr(agent, "_lnr_on_round_complete", None)
    if callable(on_round_complete):
        try:
            on_round_complete([pending.tool_name] if pending.tool_name else [])
        except Exception:
            pass
    return ResumeContinuationResult(
        state=state,
        executed_tool=True,
        early_out=str(early_out) if early_out is not None else None,
        effective_max=new_effective_max if new_effective_max is not None else effective_max,
    )
