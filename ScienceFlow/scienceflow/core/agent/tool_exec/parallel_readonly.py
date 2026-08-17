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

"""Parallel read/grep/glob/ls execution and tool-call mode classification."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any

from deepcraft_core import Message
from deepcraft_core.tool import ToolResult
from rich.markup import escape

from scienceflow.core.agent.tools.bash_utils import _bash_command_parallel_safe
from scienceflow.core.agent.shared.constants import PARALLEL_READONLY_TOOLS
from scienceflow.core.agent.io.interaction_log import (
    format_tool_call_lines_for_interaction_log,
    tool_result_text_for_interaction_log,
    truncate_for_interaction_log,
)
from scienceflow.core.agent.memory.memory_utils import (
    _tool_call_args_from_tc,
    _tool_call_names_from_list,
    maybe_collapse_stale_snapshots,
)

FILE_MUTATION_TOOLS = {"write", "edit"}
BLOCKED_WRITE_EDIT_BUNDLE_MODE = "blocked_write_edit_bundle"


class ParallelReadonlyMixin:
    """Classify multi-tool responses and run read-only file tools in parallel."""

    def _normalize_tool_calls_for_execution(
        self, tool_calls: list[Any],
    ) -> tuple[str, list[Any]]:
        """Classify multi-tool responses.

        Returns ``(mode, calls)``: ``single`` (0–1 calls), ``parallel_readonly`` (only
        ``read``/``grep``), ``parallel_bash`` (only safe concurrent ``bash``),
        ``blocked_write_edit_bundle`` (multi-tool bundle containing ``write``/``edit``),
        or ``sequential`` (mixed non-file-mutating tools).
        """
        if len(tool_calls) <= 1:
            return ("single", tool_calls)
        names = _tool_call_names_from_list(tool_calls)
        force_result_md = getattr(self, "_lnr_should_force_result_md_now", None)
        if callable(force_result_md) and force_result_md():
            self._log_info(
                "[result-md-code] multi-tool response while result.md pending; "
                "executing sequentially so non-result.md tools can be blocked: %s",
                ",".join(n or "?" for n in names),
            )
            return ("sequential", tool_calls)
        if any(n in FILE_MUTATION_TOOLS for n in names):
            self._log_info(
                "[tool-calls] blocked multi-tool bundle containing write/edit: %s",
                ",".join(n or "?" for n in names),
            )
            return (BLOCKED_WRITE_EDIT_BUNDLE_MODE, tool_calls)
        if all(n in PARALLEL_READONLY_TOOLS for n in names):
            return ("parallel_readonly", tool_calls)
        if (
            self._parallel_bash_enabled
            and len(tool_calls) > 1
            and all(n == "bash" for n in names)
        ):
            all_safe = True
            for tc in tool_calls:
                args = _tool_call_args_from_tc(tc)
                cmd = (args.get("command") or "").strip()
                if not _bash_command_parallel_safe(cmd):
                    all_safe = False
                    break
            if all_safe:
                self._log_info(
                    "[tool-calls] multi-bash parallel bundle (%d calls)",
                    len(tool_calls),
                )
                return ("parallel_bash", tool_calls)
        self._log_info(
            "[tool-calls] multi-tool response has %d calls; executing sequentially in one LLM round",
            len(tool_calls),
        )
        return ("sequential", tool_calls)

    def _record_blocked_write_edit_bundle(self, tool_calls: list[Any]) -> None:
        """Record a retry instruction when write/edit appeared in a multi-tool bundle."""
        names = _tool_call_names_from_list(tool_calls)
        rendered = ", ".join(n or "?" for n in names)
        msg = (
            "[Guard] Multiple tool calls were returned in one turn and at least one "
            "was `write` or `edit`. No tools were executed. Retry with exactly one "
            "`write` or exactly one `edit` as the sole tool call, then wait for its "
            "result before the next file operation."
        )
        self.memory.add_message(Message.user_message(msg))
        self._log_info(
            "[tool-calls] write/edit bundle blocked; no tools executed; names=%s",
            rendered,
        )

    async def _execute_parallel_readonly_bundle(
        self,
        assistant_msg: Any,
        tool_calls: list[Any],
        *,
        effective_max: int | None,
        initial_max: int | None,
        recovery: bool,
    ) -> tuple[str | None, int | None]:
        """Execute several read-only tools in parallel; append assistant + tool messages.

        Returns ``(early_exit_message, new_effective_max)``. *effective_max* is unchanged when
        *recovery* is true (no step-budget expansion).
        """
        narrow_assistant = SimpleNamespace(
            content=getattr(assistant_msg, "content", None) or "",
            reasoning_content=getattr(assistant_msg, "reasoning_content", None),
            tool_calls=tool_calls,
        )
        self._add_assistant_api_message(narrow_assistant)

        _llm_content = getattr(assistant_msg, "content", None) or ""
        if _llm_content.strip():
            self._log_info(
                "[assistant-thinking] %s",
                truncate_for_interaction_log(_llm_content),
            )

        for tc in tool_calls:
            fn = tc.function
            name = fn.name
            raw_args = fn.arguments if isinstance(fn.arguments, str) else str(fn.arguments)
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                args = {}
            args.pop("config", None)
            self._pop_and_log_thought(args)
            for _tcl in format_tool_call_lines_for_interaction_log(
                name, args, policy=self._interaction_log_policy,
            ):
                self._log_info("%s", _tcl)

        async def _run_one(tc: Any) -> tuple[Any, str, dict[str, Any], ToolResult]:
            fn = tc.function
            name = fn.name
            raw_args = fn.arguments if isinstance(fn.arguments, str) else str(fn.arguments)
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                args = {}
            args.pop("config", None)
            self._pop_and_log_thought(args)
            tool_result = await self._execute_tool_maybe_edit_guard(
                name,
                args,
            )
            if not isinstance(tool_result, ToolResult):
                tool_result = ToolResult(output=str(tool_result))
            tool_result = self._maybe_rewrite_tool_result_paths(tool_result)
            if not tool_result.error and name == "read":
                self._record_read_for_edit_guard(str(args.get("path") or ""))
            return tc, name, args, tool_result

        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])

        em = effective_max
        for tc, name, args, tool_result in results:
            _tool_out_logged = tool_result_text_for_interaction_log(
                name, str(tool_result), self._interaction_log_policy,
            )
            self._log_info(
                "[tool-result] %s exit=%s %s",
                name,
                "err" if tool_result.error else "ok",
                _tool_out_logged,
            )
            guard_mgr = getattr(self, "_guard_manager", None)
            guard_coaching = ""
            if guard_mgr is not None:
                guard_coaching = guard_mgr.process_tool_result(
                    name,
                    args,
                    tool_result,
                )

            if (
                not recovery
                and em is not None
                and initial_max is not None
                and tool_result.error
            ):
                block_budget = bool(
                    guard_mgr is not None
                    and guard_mgr.any_blocks_budget_expansion(
                        name,
                        args,
                        tool_result,
                    )
                )
                if block_budget:
                    self._log_info(
                        "[budget] expansion blocked by tool guard (%s)",
                        name,
                    )
                else:
                    old_cap = em
                    _budget_type = self.classify_tool_error_for_budget(
                        name, args, tool_result,
                    )
                    em = self._run_policy.expand_round_budget_after_tool_error(
                        tool_name=name,
                        tool_error=True,
                        tool_error_type=_budget_type,
                        effective_max=em,
                        initial_max=initial_max,
                        debug_dynamic_steps_enabled=self._debug_dynamic_steps_enabled,
                        debug_step_boost=self._debug_step_boost,
                        debug_max_steps_cap=self._debug_max_steps_cap,
                    )
                    if em > old_cap:
                        self._log_info(
                            "[budget] expanded max rounds %d -> %d after tool error (%s)",
                            old_cap,
                            em,
                            name,
                        )
                    self._effective_max_steps = em

            feedback = self._prepare_tool_feedback_for_memory(
                name,
                args,
                tool_result,
                guard_coaching=guard_coaching,
            )

            maybe_collapse_stale_snapshots(
                self.memory,
                name,
                args,
                tool_result,
                enabled=getattr(self, "_file_snapshot_latest_only", True),
            )
            self.memory.add_message(
                Message.tool_message(
                    feedback,
                    name,
                    tc.id,
                ),
            )

            if self._ui:
                code_preview = json.dumps(args, ensure_ascii=False)[:500]
                self._ui.render_exec_step(
                    step=self._next_step(),
                    lang=name,
                    code=escape(code_preview),
                    output=escape(str(tool_result)),
                    returncode=0 if not tool_result.error else 1,
                    exec_time=0.0,
                )
            sys.stdout.write("\n")
            sys.stdout.flush()

        return None, em
