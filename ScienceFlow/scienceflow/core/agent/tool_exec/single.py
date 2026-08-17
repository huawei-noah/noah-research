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

"""Single-tool execution and ordered multi-tool bundles (main loop)."""

from __future__ import annotations

import inspect
import json
import re
import sys
from types import SimpleNamespace
from typing import Any

from deepcraft_core import Message
from deepcraft_core.tool import ToolResult
from rich.markup import escape

from scienceflow.core.agent.run_policy import (
    RESULT_MD_AFTER_SUCCESS_RETRY_PROMPT,
)
from scienceflow.core.agent.tools.bash_utils import (
    _looks_like_bare_solution_run,
)
from scienceflow.core.agent.io.interaction_log import (
    format_tool_call_lines_for_interaction_log,
    tool_result_text_for_interaction_log,
    truncate_for_interaction_log,
)
from scienceflow.core.agent.memory.memory_utils import (
    maybe_collapse_stale_snapshots,
    scrub_last_assistant_write_tool_call_content,
)
from scienceflow.core.tools.write_placeholder import looks_like_write_placeholder_mimicry
from scienceflow.utils.workspace_interaction_log import (
    BashStreamFilter,
    write_raw_to_interaction_log,
)


class SingleToolExecMixin:
    """Run one tool after assistant message; sequential multi-tool in one LLM round."""

    def _add_message_after_current_tool_bundle(self, msg: Message) -> None:
        pending = getattr(self, "_tool_bundle_deferred_messages", None)
        if isinstance(pending, list):
            pending.append(msg)
        else:
            self.memory.add_message(msg)

    @staticmethod
    def _tool_targets_result_md(tool_name: str, args: dict[str, Any]) -> bool:
        if tool_name in {"write", "edit"}:
            raw_path = str(args.get("path") or "").replace("\\", "/").strip()
            raw_path = raw_path.lstrip("./")
            return raw_path == "result.md" or raw_path.endswith("/result.md")
        if tool_name != "bash":
            return False
        cmd = str(args.get("command") or "")
        lines = cmd.splitlines()
        probe = "\n".join(lines[:8]) if lines else cmd[:1000]
        if "result.md" not in probe:
            return False
        rel = r"(?:['\"]?\./)?result\.md(?:['\"])?\b"
        return bool(
            re.search(rf"(?:>>|>)\s*{rel}", probe, re.MULTILINE)
            or re.search(rf"\btee\s+(?:-a\s+)?{rel}", probe, re.MULTILINE)
            or re.search(rf"\b(?:cp|mv)\b.*\s{rel}(?:\s|$)", probe, re.MULTILINE)
            or re.search(rf"\bsed\s+-i\b.*\s{rel}(?:\s|$)", probe, re.MULTILINE)
        )

    @staticmethod
    def _tool_targets_solution_py(tool_name: str, args: dict[str, Any]) -> bool:
        if tool_name not in {"write", "edit"}:
            return False
        raw_path = str(args.get("path") or "").replace("\\", "/").strip()
        raw_path = raw_path.lstrip("./")
        return raw_path == "solution.py" or raw_path.endswith("/solution.py")

    def _long_horizon_ledger_filename(self) -> str:
        name = str(
            getattr(self, "_lnr_ledger_filename", "run_results.md")
            or "run_results.md"
        ).replace("\\", "/").strip()
        if not name or "/" in name or name.startswith("."):
            return "run_results.md"
        return name

    def _tool_targets_long_horizon_ledger(self, tool_name: str, args: dict[str, Any]) -> bool:
        if not bool(getattr(self, "_lnr_stage_commit_enabled", False)):
            return False
        ledger = self._long_horizon_ledger_filename()
        if tool_name in {"write", "edit"}:
            raw_path = str(args.get("path") or "").replace("\\", "/").strip().lstrip("./")
            return raw_path == ledger or raw_path.endswith("/" + ledger)
        if tool_name != "bash":
            return False
        cmd = str(args.get("command") or "")
        lines = cmd.splitlines()
        probe = "\n".join(lines[:8]) if lines else cmd[:1000]
        if ledger not in probe:
            return False
        escaped = re.escape(ledger)
        rel = rf"(?:['\"]?\./)?{escaped}(?:['\"])?\b"
        if re.search(rf"(?:>>|>)\s*{rel}", probe, re.MULTILINE):
            return True
        if re.search(rf"\btee\s+(?:-a\s+)?{rel}", probe, re.MULTILINE):
            return True
        if re.search(rf"\b(?:cp|mv)\b.*\s{rel}(?:\s|$)", probe, re.MULTILINE):
            return True
        if re.search(rf"\bsed\s+-i\b.*\s{rel}(?:\s|$)", probe, re.MULTILINE):
            return True
        return False

    def _tool_reads_long_horizon_ledger(self, tool_name: str, args: dict[str, Any]) -> bool:
        if not bool(getattr(self, "_lnr_stage_commit_enabled", False)):
            return False
        ledger = self._long_horizon_ledger_filename()
        if tool_name == "read":
            raw_path = str(args.get("path") or "").replace("\\", "/").strip().lstrip("./")
            return raw_path == ledger or raw_path.endswith("/" + ledger)
        if tool_name != "bash":
            return False
        cmd = str(args.get("command") or "")
        lines = cmd.splitlines()
        probe = "\n".join(lines[:8]) if lines else cmd[:1000]
        if ledger not in probe or re.search(r"(?:>>|>)", probe):
            return False
        return bool(re.search(rf"\b(?:cat|head|tail|grep|wc|sed)\b.*\b{re.escape(ledger)}\b", probe, re.MULTILINE))

    def _long_horizon_ledger_entries(self) -> list[dict[str, str]]:
        if not bool(getattr(self, "_lnr_stage_commit_enabled", False)):
            return []
        path = self._workspace_dir / self._long_horizon_ledger_filename()
        if not path.is_file():
            return []
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        matches = list(re.finditer(r"(?im)^\s*###\s+(S?\d+)\b.*$", text))
        entries: list[dict[str, str]] = []
        for idx, match in enumerate(matches):
            raw = match.group(1).upper()
            digits = raw[1:] if raw.startswith("S") else raw
            if not digits.isdigit():
                continue
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            entries.append({"step": f"S{int(digits):02d}", "body": text[start:end].strip()})
        return entries

    def _long_horizon_ledger_step_count(self) -> int:
        return len(self._long_horizon_ledger_entries())

    def _maybe_record_long_horizon_node_ledger_commit(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        tool_result: ToolResult,
        effective_max: int | None,
        previous_step_count: int | None = None,
    ) -> tuple[str | None, int | None] | None:
        if tool_result.error or not self._tool_targets_long_horizon_ledger(tool_name, args):
            return None
        ledger_entries = self._long_horizon_ledger_entries()
        step_count = len(ledger_entries)
        baseline_raw = getattr(
            self,
            "_lnr_committed_ledger_step_count",
            None,
        )
        if isinstance(baseline_raw, int):
            baseline_count = max(0, baseline_raw)
        else:
            baseline_count = max(0, int(previous_step_count or 0))
            self._lnr_committed_ledger_step_count = baseline_count
        new_entries = step_count - baseline_count
        if step_count <= 0:
            self._lnr_stage_commit_guard_failed = True
            self._log_info(
                "[long-horizon-flow] run_results.md write ignored: no parseable Sxx stage",
            )
            self._add_message_after_current_tool_bundle(
                Message.user_message(
                    "[Guard] `run_results.md` was updated but no parseable `### Sxx` "
                    "stage entry was found. Append one compact stage entry with "
                    "`metric:`, `lower_is_better:`, `BRIEF:`, and `WHY:` to submit "
                    "the current long-horizon stage."
                ),
            )
            return None
        if new_entries <= 0:
            self._lnr_stage_commit_guard_failed = True
            self._log_info(
                "[long-horizon-flow] run_results.md write ignored: no new Sxx stage baseline=%d current=%d",
                baseline_count,
                step_count,
            )
            self._add_message_after_current_tool_bundle(
                Message.user_message(
                    "[Guard] `run_results.md` was updated, but no new `### Sxx` "
                    "entry was added after the committed baseline. Preserve prior "
                    "entries and add one compact entry for the metric-backed effective "
                    "experiment you are submitting, with `metric:`, `lower_is_better:`, `BRIEF:`, and `WHY:`."
                ),
            )
            return None
        steps = [str(entry.get("step") or "") for entry in ledger_entries]
        duplicate_steps = sorted({step for step in steps if steps.count(step) > 1})
        if duplicate_steps:
            self._lnr_stage_commit_guard_failed = True
            self._log_info(
                "[long-horizon-flow] run_results.md write ignored: duplicate Sxx labels=%s",
                ",".join(duplicate_steps),
            )
            self._add_message_after_current_tool_bundle(
                Message.user_message(
                    "[Guard] `run_results.md` has duplicate `### Sxx` labels. "
                    "Use monotonically increasing unique Sxx labels. Preserve prior "
                    "entries and add a unique Sxx entry with `metric:`, `lower_is_better:`, `BRIEF:`, and `WHY:`."
                ),
            )
            return None
        new_ledger_entries = ledger_entries[baseline_count:]
        required_fields = ("metric", "lower_is_better", "BRIEF", "WHY")
        missing_field_steps: list[str] = []
        for entry in new_ledger_entries:
            body = str(entry.get("body") or "")
            missing = [
                field
                for field in required_fields
                if not re.search(rf"(?im)^\s*{re.escape(field)}\s*:\s*\S+", body)
            ]
            if missing:
                step_name = str(entry.get("step") or "")
                missing_text = ",".join(missing)
                missing_field_steps.append(f"{step_name} missing {missing_text}")
        if missing_field_steps:
            self._lnr_stage_commit_guard_failed = True
            self._log_info(
                "[long-horizon-flow] run_results.md write ignored: missing fields steps=%s",
                ";".join(missing_field_steps),
            )
            self._add_message_after_current_tool_bundle(
                Message.user_message(
                    "[Guard] Every new long-horizon ledger entry must correspond "
                    "to a metric-backed effective experiment. Each new Sxx entry needs "
                    "`metric:`, `lower_is_better:`, `BRIEF:`, and `WHY:` before submitting the current stage."
                ),
            )
            return None
        has_valid = getattr(self, "_lnr_has_current_valid_bare_run", None)
        if callable(has_valid) and not has_valid():
            self._log_info(
                "[long-horizon-flow] stage commit waits: no current run-control-ready solution run",
            )
            self._add_message_after_current_tool_bundle(
                Message.user_message(
                    "[Guard] The long-horizon stage journal was updated, but the "
                    "current workspace has no matching run-control-ready metric-backed "
                    "experiment in this node. Produce and validate a metric-backed "
                    "experiment before submitting the current stage."
                ),
            )
            return None
        msg = (
            "[ScienceAgent] Long-horizon flow: `run_results.md` node ledger "
            "updated. Stopping this node session so the controller can record "
            "the submitted current stage."
        )
        self._lnr_committed_ledger_step_count = step_count
        self._lnr_ledger_committed = True
        self._lnr_ledger_step_count = step_count
        self._lnr_ledger_new_entries = new_entries
        self._lnr_stage_commit_guard_failed = False
        self._lnr_stage_journal_pending = False
        self._log_info(
            "[long-horizon-flow] node ledger commit accepted ledger=%s steps=%d new_entries=%d; stopping node",
            self._long_horizon_ledger_filename(),
            step_count,
            new_entries,
        )
        self._sync_last_run_token_totals()
        return (msg, effective_max)

    def _lnr_long_horizon_stage_journal_pending(self) -> bool:
        return (
            bool(getattr(self, "_lnr_force_stage_journal_after_run", False))
            and bool(getattr(self, "_lnr_stage_commit_enabled", False))
            and bool(getattr(self, "_lnr_stage_journal_pending", False))
        )

    def _lnr_block_non_long_horizon_stage_journal_tool(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        effective_max: int | None,
    ) -> tuple[None, int | None]:
        feedback = (
            "[Guard] Tool call blocked: the current solution already has a "
            "run-control-ready `python3 solution.py` run. Before more edits or "
            "training, commit exactly one new `### Sxx` entry in "
            "`run_results.md` with `metric:`, `lower_is_better:`, and `BRIEF:`. No command was run "
            "and no file was changed."
        )
        self.memory.add_message(Message.tool_message(feedback, tool_name, tool_call_id))
        self._log_info(
            "[long-horizon-flow] blocked %s while stage journal commit is pending",
            tool_name,
        )
        self._add_message_after_current_tool_bundle(Message.user_message(feedback))
        return (None, effective_max)

    def _lnr_block_long_horizon_result_md_tool(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        effective_max: int | None,
    ) -> tuple[None, int | None]:
        feedback = (
            "[Guard] In long-horizon flow, do not write `result.md`. "
            "The system will synthesize a machine-readable `result.md` after "
            "an accepted `run_results.md` commit. Commit the node by writing "
            "one new `### Sxx` entry in `run_results.md` with `metric:`, "
            "`lower_is_better:`, and `BRIEF:`. No command was run and no "
            "file was changed."
        )
        self._lnr_stage_commit_guard_failed = True
        self.memory.add_message(Message.tool_message(feedback, tool_name, tool_call_id))
        self._log_info(
            "[long-horizon-flow] blocked %s result.md write; result.md is system-generated",
            tool_name,
        )
        self._add_message_after_current_tool_bundle(Message.user_message(feedback))
        return (None, effective_max)

    def _lnr_should_force_result_md_now(self) -> bool:
        if not bool(getattr(self, "_lnr_result_md_after_success_pending", False)):
            return False
        if not bool(getattr(self, "_lnr_stop_after_bare_solution_success", False)):
            return False
        has_valid = getattr(self, "_lnr_has_current_valid_bare_run", None)
        if callable(has_valid) and not has_valid():
            return False
        return True

    def _lnr_repeat_result_md_after_success_prompt(self, *, reason: str = "") -> None:
        if reason:
            self._log_info("[result-md-code] repeat prompt: %s", reason)
        self._add_message_after_current_tool_bundle(
            Message.user_message(RESULT_MD_AFTER_SUCCESS_RETRY_PROMPT),
        )

    def _lnr_block_non_result_md_tool(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        effective_max: int | None,
    ) -> tuple[None, int | None]:
        feedback = (
            "[Guard] Tool call blocked: the last bare `python3 solution.py` run "
            "passed run-control checks, so the only allowed next action is exactly one "
            "`write` tool call to path `result.md`. No command was run and no "
            "file was changed."
        )
        self.memory.add_message(Message.tool_message(feedback, tool_name, tool_call_id))
        self._log_info(
            "[result-md-code] blocked %s while result.md write is pending",
            tool_name,
        )
        self._lnr_repeat_result_md_after_success_prompt(reason=f"blocked {tool_name}")
        return (None, effective_max)

    async def _run_one_tool_after_assistant_logged(
        self,
        tc: Any,
        assistant_msg: Any,
        effective_max: int | None,
        initial_max: int | None,
        *,
        log_assistant_thinking: bool = True,
    ) -> tuple[str | None, int | None]:
        """Run one tool after assistant message(s) already stored in memory.

        Caller must have appended the assistant ``tool_calls`` message for this round.
        """
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
        if log_assistant_thinking and _llm_content.strip():
            self._log_info(
                "[assistant-thinking] %s",
                truncate_for_interaction_log(_llm_content),
            )
        for _tcl in format_tool_call_lines_for_interaction_log(
            name, args, policy=self._interaction_log_policy,
        ):
            self._log_info("%s", _tcl)

        if (
            bool(getattr(self, "_lnr_stage_commit_enabled", False))
            and self._tool_targets_result_md(name, args)
        ):
            return self._lnr_block_long_horizon_result_md_tool(
                tool_name=name,
                tool_call_id=tc.id,
                effective_max=effective_max,
            )

        if (
            self._lnr_long_horizon_stage_journal_pending()
            and not self._tool_targets_long_horizon_ledger(name, args)
            and not self._tool_reads_long_horizon_ledger(name, args)
        ):
            return self._lnr_block_non_long_horizon_stage_journal_tool(
                tool_name=name,
                tool_call_id=tc.id,
                effective_max=effective_max,
            )

        if self._lnr_should_force_result_md_now() and not self._tool_targets_result_md(
            name,
            args,
        ):
            return self._lnr_block_non_result_md_tool(
                tool_name=name,
                tool_call_id=tc.id,
                effective_max=effective_max,
            )

        _bash_stream_summary: str | None = None
        _bash_stream_emitted = False
        if name == "bash":
            _cmd0 = (args.get("command") or "").strip()
            _tty = sys.stdout.isatty()
            _cap = 0 if _tty else self._scienceflow_stdout_max_chars
            _bash_shown = 0
            _bash_omitted = 0
            _bash_omit_chunks = 0
            _bash_lines_printed = 0
            _stream_filter: BashStreamFilter | None = None
            if (
                self._bash_stream_to_interaction_log
                and self._ws_interaction_log is not None
                and ("python" in _cmd0.lower() or "solution.py" in _cmd0.lower())
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
            _lhf_ledger_step_count_before = (
                self._long_horizon_ledger_step_count()
                if self._tool_targets_long_horizon_ledger(name, args)
                else None
            )
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
            _lhf_ledger_step_count_before = (
                self._long_horizon_ledger_step_count()
                if self._tool_targets_long_horizon_ledger(name, args)
                else None
            )
            tool_result = await self._execute_tool_maybe_edit_guard(
                name,
                args,
            )
        if not isinstance(tool_result, ToolResult):
            tool_result = ToolResult(output=str(tool_result))

        # Teleport S4: rewrite current-workspace absolute paths in the result so
        # the LLM only sees ``./`` (node directory rotation stays invisible).
        tool_result = self._maybe_rewrite_tool_result_paths(tool_result)

        if (
            name == "write"
            and tool_result.error
            and isinstance(args.get("content"), str)
            and looks_like_write_placeholder_mimicry(args["content"])
        ):
            scrub_last_assistant_write_tool_call_content(self.memory, tool_call_id=tc.id)

        if not tool_result.error and name == "read":
            self._record_read_for_edit_guard(str(args.get("path") or ""))

        if name == "write" and tool_result.error:
            rel_w = str(args.get("path") or "").replace("\\", "/").lstrip("/")
            if rel_w.endswith("solution.py") and self._solution_write_syntax_or_short_failure(
                args, tool_result,
            ):
                self._consecutive_write_syntax_fails += 1
                nf = self._consecutive_write_syntax_fails
                if nf >= 1:
                    recovered = await self._try_write_solution_from_assistant_text(
                        args, _llm_content,
                    )
                    if recovered is not None and not recovered.error:
                        tool_result = recovered
                        self._consecutive_write_syntax_fails = 0

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
        checkpoint_hook = getattr(self, "_maybe_workspace_git_auto_checkpoint", None)
        if callable(checkpoint_hook):
            checkpoint_hook(name, tool_result)
        guard_mgr = getattr(self, "_guard_manager", None)
        guard_coaching = ""
        if guard_mgr is not None:
            guard_coaching = guard_mgr.process_tool_result(
                name,
                args,
                tool_result,
            )

        if tool_result.error:
            # Detect infrastructure failures (e.g. taskset affinity mis-configuration).
            # These are signalled by [infra-error] in the tool output.  For such errors:
            #   1. Do NOT expand the round budget (spending steps on infra issues is wasteful).
            #   2. Keep a consecutive counter for diagnostics, but keep LNR alive until
            #      the wall-clock budget expires.
            _is_infra_error = (
                name == "bash"
                and "[infra-error]" in (tool_result.output or "")
            )
            if _is_infra_error:
                _consecutive = getattr(self, "_consecutive_infra_errors", 0) + 1
                self._consecutive_infra_errors = _consecutive
                _infra_threshold = 5
                self._log_info(
                    "[budget] infra-error detected for bash; NOT expanding budget "
                    "(consecutive=%d/%d)",
                    _consecutive,
                    _infra_threshold,
                )
                if _consecutive >= _infra_threshold:
                    self._log_info(
                        "[budget] infra-error threshold reached; keeping worker alive "
                        "until wall-clock budget expires (consecutive=%d)",
                        _consecutive,
                    )
            else:
                self._consecutive_infra_errors = 0

            block_budget = _is_infra_error or bool(
                guard_mgr is not None
                and guard_mgr.any_blocks_budget_expansion(
                    name,
                    args,
                    tool_result,
                )
            )
            if block_budget:
                if not _is_infra_error:
                    self._log_info(
                        "[budget] expansion blocked by tool guard (%s)",
                        name,
                    )
            else:
                old_cap = effective_max
                _budget_type = self.classify_tool_error_for_budget(
                    name, args, tool_result,
                )
                effective_max = self._run_policy.expand_round_budget_after_tool_error(
                    tool_name=name,
                    tool_error=True,
                    tool_error_type=_budget_type,
                    effective_max=effective_max,
                    initial_max=initial_max,
                    debug_dynamic_steps_enabled=self._debug_dynamic_steps_enabled,
                    debug_step_boost=self._debug_step_boost,
                    debug_max_steps_cap=self._debug_max_steps_cap,
                )
                if effective_max > old_cap:
                    self._log_info(
                        "[budget] expanded max rounds %d -> %d after tool error (%s)",
                        old_cap,
                        effective_max,
                        name,
                    )
                self._effective_max_steps = effective_max
        else:
            # Reset infra-error counter on any successful tool call.
            if name == "bash":
                self._consecutive_infra_errors = 0

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

        candidate_archive_cb = getattr(self, "_lnr_candidate_archive_callback", None)
        if callable(candidate_archive_cb):
            try:
                archive_out = candidate_archive_cb(
                    agent=self,
                    tool_name=name,
                    args=args,
                    tool_result=tool_result,
                )
                if inspect.isawaitable(archive_out):
                    await archive_out
            except Exception as exc:
                self._log_info(
                    "[long-horizon-flow] candidate artifact archive callback failed: %s",
                    exc,
                )

        stage_commit_out = self._maybe_record_long_horizon_node_ledger_commit(
            tool_name=name,
            args=args,
            tool_result=tool_result,
            effective_max=effective_max,
            previous_step_count=_lhf_ledger_step_count_before,
        )
        if stage_commit_out is not None:
            return stage_commit_out

        if (
            not tool_result.error
            and self._tool_targets_solution_py(name, args)
            and bool(getattr(self, "_lnr_stop_after_bare_solution_success", False))
        ):
            invalidate = getattr(self, "_lnr_invalidate_current_valid_run", None)
            if callable(invalidate):
                invalidate("solution_changed_after_valid_run")

        if (
            name == "write"
            and tool_result.error
            and self._consecutive_write_syntax_fails >= 3
        ):
            rel_abort = str(args.get("path") or "").replace("\\", "/").lstrip("/")
            if rel_abort.endswith("solution.py") and self._solution_write_syntax_or_short_failure(
                args, tool_result,
            ):
                abort_msg = (
                    "[ScienceAgent] Stopping: repeated failed writes to solution.py "
                    "(syntax / truncated `content`). Put the full file in the `write` "
                    "tool `content` argument."
                )
                self._log_info("%s", abort_msg)
                self._add_message_after_current_tool_bundle(
                    Message.assistant_message(abort_msg),
                )
                self._sync_last_run_token_totals()
                return (abort_msg, effective_max)

        if name == "bash":
            early_out = await self._maybe_embedded_full_run_after_quick_test(
                args, tool_result,
            )
            if early_out is not None:
                self._sync_last_run_token_totals()
                return (early_out, effective_max)
            # Metric-first stage capture when safety/embedded full-run is off. A true
            # A run-control-ready candidate is distinguished by _lnr_snapshot_reason below.
            await self._maybe_write_bare_run_tail_snapshot(args, tool_result)
            lhr_stage_cb = getattr(self, "_lnr_stage_capture_callback", None)
            lhr_candidate_artifact = str(
                getattr(self, "_lnr_candidate_artifact_rel", "") or "",
            ).strip()
            lhr_candidate_exists = bool(
                lhr_candidate_artifact
                and (self._workspace_dir / lhr_candidate_artifact).exists()
            )
            lhr_evaluator_capture = bool(
                getattr(self, "_lnr_stage_capture_on_candidate_artifact", False)
                and lhr_candidate_exists
            )
            if (
                callable(lhr_stage_cb)
                and not self._embedded_full_run_enabled
                and not tool_result.error
                and (
                    bool(getattr(self, "_lnr_snapshot_ok", False))
                    or lhr_evaluator_capture
                )
            ):
                lhr_out = await lhr_stage_cb(agent=self, args=args, tool_result=tool_result)
                if lhr_out:
                    self._sync_last_run_token_totals()
                    return (str(lhr_out), effective_max)
            if (
                bool(getattr(self, "_lnr_force_stage_journal_after_run", False))
                and bool(getattr(self, "_lnr_stage_commit_enabled", False))
                and not self._embedded_full_run_enabled
                and not tool_result.error
                and _looks_like_bare_solution_run((args.get("command") or "").strip())
                and bool(getattr(self, "_lnr_snapshot_ok", False))
                and str(getattr(self, "_lnr_snapshot_reason", "") or "") == "gate_ok_tail_snapshot"
            ):
                self._lnr_stage_journal_pending = True
            if (
                bool(getattr(self, "_lnr_stop_after_bare_solution_success", False))
                and not self._embedded_full_run_enabled
                and not tool_result.error
                and _looks_like_bare_solution_run((args.get("command") or "").strip())
                and bool(getattr(self, "_lnr_snapshot_ok", False))
                and str(getattr(self, "_lnr_snapshot_reason", "") or "") == "gate_ok_tail_snapshot"
            ):
                result_md_path = self._workspace_dir / "result.md"
                if not result_md_path.exists():
                    msg = (
                        "[ScienceAgent] LNR: bare `solution.py` passed run-control checks. "
                        "Stopping this node session; feedback LLM will write result.md from "
                        "the clean run-control-ready snapshot."
                    )
                    self._lnr_result_md_after_success_pending = False
                    self._log_info(
                        "[result-md-feedback] bare solution succeeded; stop for feedback LLM result.md"
                    )
                    self._sync_last_run_token_totals()
                    return (msg, effective_max)
                msg = (
                    "[ScienceAgent] LNR: bare `solution.py` passed run-control checks. "
                    "Stopping this node session; result.md already exists."
                )
                self._log_info("%s", msg)
                # Do not persist this control message into chat memory: clone-continue inherits
                # .agent_memory and an assistant "session stopped" line makes the child LLM
                # treat the task as finished.
                self._sync_last_run_token_totals()
                return (msg, effective_max)
        if (
            self._tool_targets_result_md(name, args)
            and not tool_result.error
        ):
            requires_valid = bool(getattr(self, "_lnr_stop_after_bare_solution_success", False))
            has_valid = getattr(self, "_lnr_has_current_valid_bare_run", None)
            if requires_valid and callable(has_valid) and not has_valid():
                result_md = self._workspace_dir / "result.md"
                if result_md.exists():
                    try:
                        result_md.unlink()
                    except OSError:
                        self._log_warning(
                            "[result-md-code] failed to remove invalid result.md",
                            exc_info=True,
                        )
                self._lnr_result_md_after_success_pending = False
                self._lnr_snapshot_ok = False
                self._lnr_snapshot_reason = "result_md_without_current_valid_run"
                self._add_message_after_current_tool_bundle(
                    Message.user_message(
                        "[Guard] result.md was not accepted because the current "
                        "`solution.py` has no matching run-control-ready `python3 solution.py` "
                        "run in this node. Fix/run `solution.py` until run-control checks pass, "
                        "then write result.md."
                    ),
                )
                return (None, effective_max)
            if not bool(getattr(self, "_lnr_result_md_after_success_pending", False)):
                return (None, effective_max)
            result_md = self._workspace_dir / "result.md"
            if not result_md.is_file():
                self._lnr_repeat_result_md_after_success_prompt(
                    reason="result.md tool returned ok but root result.md is missing",
                )
                return (None, effective_max)
            self._lnr_result_md_after_success_pending = False
            msg = (
                "[ScienceAgent] LNR: result.md written after successful bare solution run. "
                "Stopping this node session."
            )
            self._log_info("%s", msg)
            self._sync_last_run_token_totals()
            return (msg, effective_max)
        return (None, effective_max)

    async def _execute_sequential_bundle(
        self,
        assistant_msg: Any,
        tool_calls: list[Any],
        *,
        effective_max: int | None,
        initial_max: int | None,
    ) -> tuple[str | None, int | None]:
        """Execute several tools in order; one assistant turn, multiple tool results."""
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
        em = effective_max
        early_out_first: str | None = None
        previous_pending = getattr(self, "_tool_bundle_deferred_messages", None)
        had_previous_pending = isinstance(previous_pending, list)
        pending_messages: list[Message] = []
        self._tool_bundle_deferred_messages = pending_messages
        try:
            for tc in tool_calls:
                early_out, em = await self._run_one_tool_after_assistant_logged(
                    tc,
                    assistant_msg,
                    em,
                    initial_max,
                    log_assistant_thinking=False,
                )
                if early_out is not None and early_out_first is None:
                    early_out_first = early_out
                if early_out is not None:
                    break
        except BaseException:
            if had_previous_pending:
                self._tool_bundle_deferred_messages = previous_pending
            else:
                try:
                    delattr(self, "_tool_bundle_deferred_messages")
                except AttributeError:
                    pass
            raise
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
        return early_out_first, em
