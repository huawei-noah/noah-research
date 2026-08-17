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

"""Minimal sequential tool execution for result.md recovery (no auto QT / embedded)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

from deepcraft_core import Message
from deepcraft_core.tool import ToolResult
from rich.markup import escape

from scienceflow.core.agent.io.interaction_log import (
    format_tool_call_lines_for_interaction_log,
    tool_result_text_for_interaction_log,
    truncate_for_interaction_log,
)
from scienceflow.core.agent.memory.memory_utils import (
    maybe_collapse_stale_snapshots,
)
from scienceflow.utils.workspace_interaction_log import (
    BashStreamFilter,
    write_raw_to_interaction_log,
)


class SequentialRecoveryMixin:
    """Sequential bundle without budget expansion hooks (recovery path)."""

    async def _execute_sequential_bundle_recovery(
        self,
        assistant_msg: Any,
        tool_calls: list[Any],
    ) -> tuple[None, None]:
        """Minimal sequential execution for result.md recovery (no auto quick-test / embedded)."""
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
        previous_pending = getattr(self, "_tool_bundle_deferred_messages", None)
        had_previous_pending = isinstance(previous_pending, list)
        pending_messages: list[Message] = []
        self._tool_bundle_deferred_messages = pending_messages
        try:
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
                if _llm_content.strip():
                    self._log_info(
                        "[assistant-thinking] %s",
                        truncate_for_interaction_log(_llm_content),
                    )
                for _tcl in format_tool_call_lines_for_interaction_log(
                    name, args, policy=self._interaction_log_policy,
                ):
                    self._log_info("%s", _tcl)
                force_result_md = getattr(self, "_lnr_should_force_result_md_now", None)
                if callable(force_result_md) and force_result_md():
                    target_result_md = getattr(self, "_tool_targets_result_md", None)
                    if callable(target_result_md) and not target_result_md(name, args):
                        block_tool = getattr(self, "_lnr_block_non_result_md_tool", None)
                        if callable(block_tool):
                            block_tool(
                                tool_name=name,
                                tool_call_id=tc.id,
                                effective_max=None,
                            )
                        continue
                _bash_stream_summary: str | None = None
                _bash_stream_emitted = False
                if name == "bash":
                    _tty = sys.stdout.isatty()
                    _cap = 0 if _tty else self._scienceflow_stdout_max_chars
                    _bash_shown = 0
                    _bash_omitted = 0
                    _bash_omit_chunks = 0
                    _bash_lines_printed = 0
                    _stream_filter: BashStreamFilter | None = None
                    _cmd = str(args.get("command") or "")
                    if (
                        self._bash_stream_to_interaction_log
                        and self._ws_interaction_log is not None
                        and ("python" in _cmd.lower() or "solution.py" in _cmd.lower())
                    ):
                        lg = self._ws_interaction_log

                        def _emit_stream_line(msg: str) -> None:
                            write_raw_to_interaction_log(
                                lg,
                                msg if msg.endswith("\n") else msg + "\n",
                            )

                        _stream_filter = BashStreamFilter(
                            _emit_stream_line,
                            prefix="[bash-stream] ",
                            dedup_enabled=self._bash_output_dedup_enabled,
                            dedup_min_repeat=self._bash_output_dedup_min_repeat,
                            dedup_summary_prefix=self._bash_output_dedup_summary_prefix,
                        )

                    def _live_bash_output(chunk: str) -> None:
                        nonlocal _bash_shown, _bash_omitted, _bash_omit_chunks, _bash_lines_printed
                        if _tty:
                            sys.stdout.write(f"\033[2m{chunk}\033[0m")
                            _bash_lines_printed += chunk.count("\n")
                        elif _cap <= 0:
                            sys.stdout.write(chunk)
                            _bash_lines_printed += chunk.count("\n")
                        else:
                            remain = _cap - _bash_shown
                            if remain <= 0:
                                _bash_omitted += len(chunk)
                                _bash_omit_chunks += 1
                            elif len(chunk) <= remain:
                                sys.stdout.write(chunk)
                                _bash_shown += len(chunk)
                                _bash_lines_printed += chunk.count("\n")
                            else:
                                sys.stdout.write(chunk[:remain])
                                _bash_omitted += len(chunk) - remain
                                _bash_shown = _cap
                                _bash_lines_printed += chunk[:remain].count("\n")
                        sys.stdout.flush()
                        if _stream_filter is not None:
                            _stream_filter.feed_chunk(chunk)

                    args = self._maybe_normalize_tool_input_paths(name, args)
                    tool_result = await self.availableTools.execute(
                        name=name,
                        tool_input=args,
                        on_output=_live_bash_output,
                    )
                    if _stream_filter is not None:
                        _stream_filter.flush()
                        _bash_stream_summary = _stream_filter.summary()
                        _bash_stream_emitted = _stream_filter.lines_emitted > 0
                    if not _tty and _cap > 0 and _bash_omitted:
                        sys.stdout.write(
                            "\n[ScienceAgent tool stdout] source=bash_live "
                            f"printed_chars={_bash_shown} omitted_chars={_bash_omitted} "
                            f"omitted_chunks={_bash_omit_chunks} "
                            f"printed_newlines={_bash_lines_printed} cap={_cap}\n",
                        )
                        sys.stdout.flush()
                else:
                    tool_result = await self._execute_tool_maybe_edit_guard(
                        name,
                        args,
                    )
                if not isinstance(tool_result, ToolResult):
                    tool_result = ToolResult(output=str(tool_result))
                tool_result = self._maybe_rewrite_tool_result_paths(tool_result)
                if name == "bash" and _bash_stream_emitted:
                    _tool_out_logged = (
                        f"[stream-mode] {_bash_stream_summary or 'streamed filtered lines'}"
                    )
                else:
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
                    ui_output = str(tool_result)
                    if name == "bash" and _bash_stream_emitted:
                        ui_output = f"[stream-mode] {_bash_stream_summary or 'streamed filtered lines'}"
                    self._ui.render_exec_step(
                        step=self._next_step(),
                        lang=name,
                        code=escape(code_preview),
                        output=escape(ui_output),
                        returncode=0 if not tool_result.error else 1,
                        exec_time=0.0,
                    )
                sys.stdout.write("\n")
                sys.stdout.flush()
        finally:
            if had_previous_pending:
                self._tool_bundle_deferred_messages = previous_pending
                previous_pending.extend(pending_messages)
            else:
                try:
                    delattr(self, "_tool_bundle_deferred_messages")
                except AttributeError:
                    pass
                for msg in pending_messages:
                    self.memory.add_message(msg)
        return None, None
