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

"""result.md recovery: extra LLM rounds after main loop exhaustion."""

from __future__ import annotations

import json
import logging
import sys
import time
from types import SimpleNamespace

from deepcraft_core import Message
from deepcraft_core.tool import ToolResult
from rich.markup import escape

from scienceflow.core.agent.tools.bash_command_classifier import classify_bash_command
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

logger = logging.getLogger("scienceflow")


class RecoveryMixin:
    """One-off LLM+tool round for missing result.md."""

    async def _run_one_recovery_llm_round(self) -> None:
        """Single LLM + optional first tool call (same semantics as main ``run`` loop)."""
        chat_messages = self._memory_ctx.build_messages_for_llm()
        if not hasattr(self.llm, "ask_tool_stream"):
            raise RuntimeError(
                "LLM backend has no ask_tool_stream(); use OnlineLLM / PooledLLM.",
            )

        assistant_msg, timeout_exit, t_llm = await self._ask_tool_stream_with_same_round_retries(
            chat_messages=chat_messages,
            round_idx=None,
            request=None,
            tool_choice="auto",
            timeout_assistant_message=(
                "[LLM stream timed out during result.md recovery; retrying if rounds remain.]"
            ),
            hook_recovery=True,
            timeout_log_label="[result-md-recovery]",
        )
        if timeout_exit:
            return

        dur_ok = time.time() - t_llm
        try:
            ti = getattr(self.llm, "_last_call_input_tokens", None)
            to = getattr(self.llm, "_last_call_output_tokens", None)
            tc = getattr(self.llm, "_last_call_input_cached_tokens", None)
            if ti is not None:
                self._run_tokens_in += int(ti or 0)
            if to is not None:
                self._run_tokens_out += int(to or 0)
            if tc is not None:
                self._run_tokens_cached += int(tc or 0)
            self._run_llm_calls += 1
        except (TypeError, ValueError):
            pass
        tool_calls_raw = getattr(assistant_msg, "tool_calls", None) or []
        tool_calls_raw = self._normalize_repl_file_change_tool_calls(tool_calls_raw)
        _first_rec: str | None = None
        _first_bash_kind: str | None = None
        if tool_calls_raw:
            try:
                _fn0 = tool_calls_raw[0].function
                _first_rec = getattr(_fn0, "name", None) or None
                if _first_rec == "bash":
                    _raw_args0 = (
                        _fn0.arguments
                        if isinstance(_fn0.arguments, str)
                        else str(_fn0.arguments)
                    )
                    _args0 = json.loads(_raw_args0 or "{}")
                    _first_bash_kind = classify_bash_command(
                        str(_args0.get("command") or ""),
                    )
            except Exception:
                _first_rec = None
                _first_bash_kind = None
        _unavailable_repl_tools = self._unavailable_repl_tool_names(tool_calls_raw)
        if _unavailable_repl_tools:
            _first_rec = "invalid_tool"
            _first_bash_kind = None
        if self._parallel_bash_enabled and tool_calls_raw:
            self._log_info(
                "[tool-calls-count] recovery n=%d first=%s",
                len(tool_calls_raw),
                _first_rec or "?",
            )
        self._record_llm_call(
            "ask_tool_stream",
            dur_ok,
            None,
            "ok",
            recovery=True,
            turn_kind="tool" if tool_calls_raw else "qa",
            first_tool_name=_first_rec,
            first_bash_kind=_first_bash_kind,
        )

        if not tool_calls_raw:
            text = (getattr(assistant_msg, "content", None) or "").strip()
            if not text:
                text = "(empty assistant message)"
            self.memory.add_message(Message.assistant_message(text))
            self._log_info(
                "[result-md-recovery] assistant (no tool): %s",
                truncate_for_interaction_log(text),
            )
            force_result_md = getattr(self, "_lnr_should_force_result_md_now", None)
            if callable(force_result_md) and force_result_md():
                repeat_prompt = getattr(
                    self,
                    "_lnr_repeat_result_md_after_success_prompt",
                    None,
                )
                if callable(repeat_prompt):
                    repeat_prompt(reason="recovery text-only response")
            return

        if _unavailable_repl_tools:
            self._block_unavailable_repl_tool_calls(
                tool_calls_raw,
                _unavailable_repl_tools,
            )
            return

        mode, tool_calls = self._normalize_tool_calls_for_execution(tool_calls_raw)
        if mode == "blocked_write_edit_bundle":
            self._record_blocked_write_edit_bundle(tool_calls)
            return

        if mode == "parallel_readonly" and len(tool_calls) > 1:
            await self._execute_parallel_readonly_bundle(
                assistant_msg,
                tool_calls,
                effective_max=None,
                initial_max=None,
                recovery=True,
            )
            return

        if mode == "parallel_bash" and len(tool_calls) > 1:
            await self._execute_parallel_bash_bundle(
                assistant_msg,
                tool_calls,
                effective_max=None,
                initial_max=None,
                recovery=True,
            )
            return

        if mode == "sequential" and len(tool_calls) > 1:
            await self._execute_sequential_bundle_recovery(
                assistant_msg,
                tool_calls,
            )
            return

        tc = tool_calls[0]
        narrow_assistant = SimpleNamespace(
            content=getattr(assistant_msg, "content", None) or "",
            reasoning_content=getattr(assistant_msg, "reasoning_content", None),
            tool_calls=[tc],
        )
        self._add_assistant_api_message(narrow_assistant)
        fn = tc.function
        name = fn.name
        raw_args = fn.arguments if isinstance(fn.arguments, str) else str(fn.arguments)
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = {}
        args.pop("config", None)
        self._pop_and_log_thought(args)

        _llm_content = getattr(assistant_msg, "content", None) or ""
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
                return

        if name == "bash":
            _tty = sys.stdout.isatty()
            _cap = 0 if _tty else self._scienceflow_stdout_max_chars
            _bash_shown = 0
            _bash_omitted = 0
            _bash_omit_chunks = 0
            _bash_lines_printed = 0
            _bash_stream_summary: str | None = None
            _bash_stream_emitted = False
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
        if not tool_result.error and name == "read":
            self._record_read_for_edit_guard(str(args.get("path") or ""))

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
