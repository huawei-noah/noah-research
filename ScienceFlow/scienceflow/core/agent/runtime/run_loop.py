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

"""Main ``run()`` loop and REPL small-talk handling."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from deepcraft_agent.base.state import AgentState
from deepcraft_core import Message
from deepcraft_core.llm import StreamHandle

from scienceflow.core.agent.tools.bash_command_classifier import classify_bash_command
from scienceflow.core.agent.io.interaction_log import truncate_for_interaction_log
from scienceflow.core.agent.memory.memory_utils import (
    _tool_call_names_from_list,
)
from scienceflow.core.agent.memory.reasoning_replay import (
    messages_with_synthetic_reasoning_replay,
)
from scienceflow.core.agent.runtime.repl_helpers import (
    looks_like_repl_small_talk,
    tool_choice_for_main_loop as _tool_choice_for_main_loop_fn,
)
from scienceflow.core.agent.run_policy import (
    AutoContinuePolicy,
    RESULT_MD_AFTER_SUCCESS_RETRY_PROMPT,
    RoundContext,
)
from scienceflow.safety.execution_policy import clear_embedded_full_run_result
from scienceflow.core.agent.memory.resource_feedback_memory import RESOURCE_STATE_SUMMARY_MARKER
from scienceflow.core.llm_reasoning_compat import is_missing_reasoning_replay_error

logger = logging.getLogger("scienceflow")


_LNR_FIRST_TASK_PROMPT_PREFIXES = (
    "You are solving one ML task in a continuous REPL workspace.",
    "You are solving one optimization task in a continuous REPL workspace.",
)


def _is_lnr_first_task_prompt_text(text: str | None) -> bool:
    clean = str(text or "").lstrip()
    return any(clean.startswith(prefix) for prefix in _LNR_FIRST_TASK_PROMPT_PREFIXES)


def _message_text(message: Message) -> str:
    return message.content if isinstance(message.content, str) else str(message.content or "")


def _clip_stage_commit_context_text(text: str, *, max_chars: int) -> str:
    clean = str(text or "").strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    marker = "\n...[stage context clipped]...\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return clean[:head].rstrip() + marker + clean[-tail:].lstrip()


def _is_stage_commit_memory_text(text: str) -> bool:
    clean = str(text or "")
    return (
        "[stage append-only write]" in clean
        or "[LNR_STAGE_COMMIT_REQUEST]" in clean
        or ("STAGE_COMMIT_BEGIN" in clean and "STAGE_COMMIT_END" in clean)
    )


def _memory_has_lnr_first_task_prompt(memory: Any) -> bool:
    try:
        records = memory.chat_history_memory.retrieve(window_size=None)
    except Exception:
        return False
    for record in records or []:
        try:
            message = record.memory_record.message
        except Exception:
            continue
        if getattr(message, "role", None) == "user" and _is_lnr_first_task_prompt_text(_message_text(message)):
            return True
    return False


def _should_append_run_request(memory: Any, request: str | None) -> bool:
    if not request:
        return False
    if _is_lnr_first_task_prompt_text(request) and _memory_has_lnr_first_task_prompt(memory):
        return False
    return True


def _is_retryable_empty_tool_stream_error(exc: BaseException) -> bool:
    """True when a same-round stream retry is safe and likely useful."""
    if isinstance(exc, ValueError):
        msg = str(exc)
        return (
            "Empty response from streaming tool LLM" in msg
            or "Incomplete streaming tool call" in msg
        )
    return isinstance(exc, httpx.TransportError) and not isinstance(exc, httpx.TimeoutException)


class RunLoopMixin:
    """Drive multi-round LLM + tool execution until completion or limit."""

    def _agentic_route_log_dir(self) -> Path:
        override = getattr(self, "_agentic_route_log_dir_override", None)
        if override:
            return Path(override)
        return self._workspace_dir / ".logs"

    def _write_agentic_route_response(self, text: str) -> None:
        body = str(text or "").strip()
        if not body:
            return
        try:
            path = self._agentic_route_log_dir() / "agentic_route_response.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body.rstrip() + "\n", encoding="utf-8")
        except OSError:
            logger.debug("[agentic-route] route response capture skipped", exc_info=True)

    def _build_lnr_stage_commit_compact_messages(
        self,
        *,
        base_messages: list[Message],
        transient_user_prompt: str,
    ) -> list[Message]:
        task_context = ""
        recent_parts: list[str] = []
        recent_budget = int(getattr(self, "_lnr_stage_commit_recent_context_chars", 7000) or 7000)
        per_message_budget = int(getattr(self, "_lnr_stage_commit_message_chars", 1200) or 1200)

        for msg in base_messages:
            text = _message_text(msg).strip()
            if not text:
                continue
            if getattr(msg, "role", None) == "user" and _is_lnr_first_task_prompt_text(text):
                task_context = _clip_stage_commit_context_text(text, max_chars=3000)
                break

        used = 0
        for msg in reversed(base_messages):
            role = str(getattr(msg, "role", "") or "")
            if role == "system":
                continue
            text = _message_text(msg).strip()
            if not text or _is_lnr_first_task_prompt_text(text) or _is_stage_commit_memory_text(text):
                continue
            clipped = _clip_stage_commit_context_text(text, max_chars=per_message_budget)
            if not clipped:
                continue
            part = f"{role or 'message'}:\n{clipped}"
            if used + len(part) > recent_budget and recent_parts:
                break
            recent_parts.append(part)
            used += len(part)

        messages: list[Message] = []
        if task_context:
            messages.append(Message.user_message("[LNR_STAGE_COMMIT_TASK_CONTEXT]\n" + task_context))
        if recent_parts:
            recent_parts.reverse()
            messages.append(
                Message.user_message(
                    "[LNR_STAGE_COMMIT_RECENT_CONTEXT]\n"
                    "This is a clipped recent context window for stage bookkeeping only.\n\n"
                    + "\n\n".join(recent_parts),
                ),
            )
        messages.append(Message.user_message(transient_user_prompt))
        return messages

    def _sync_resource_state_summary_slot(self) -> None:
        """Remove legacy resource summary slots from main-agent chat memory."""
        try:
            records = self.memory.chat_history_memory.retrieve(window_size=None)
        except Exception:
            return
        kept: list[Message] = []
        removed = 0
        for record in records or []:
            try:
                message = record.memory_record.message
            except Exception:
                continue
            content = message.content if isinstance(message.content, str) else str(message.content or "")
            if content.startswith(RESOURCE_STATE_SUMMARY_MARKER):
                removed += 1
                continue
            kept.append(message)
        self._resource_state_summary_last_text = ""
        if removed <= 0:
            return
        try:
            rewrite = getattr(getattr(self, "_memory_ctx", None), "rewrite_messages", None)
            if callable(rewrite):
                rewrite(kept)
            else:
                self.memory.chat_history_memory.storage.clear()
                for message in kept:
                    self.memory.add_message(message)
            logger.info("[resource-state] removed %d legacy memory summary slot(s)", removed)
        except Exception:
            logger.debug("[resource-state] legacy summary slot cleanup skipped", exc_info=True)

    def _append_agentic_route_decision_log(
        self,
        *,
        prompt: str,
        response: str,
        trigger: str,
        base_messages_count: int,
    ) -> None:
        try:
            path = self._agentic_route_log_dir() / "agentic_route_decisions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            prompt_text = str(prompt or "")
            response_text = str(response or "")
            record = {
                "event": "agentic_route_decision",
                "timestamp": time.time(),
                "trigger": str(trigger or ""),
                "workspace_node": self._workspace_dir.name,
                "input_compact_sha": hashlib.sha256(
                    prompt_text.encode("utf-8", errors="ignore"),
                ).hexdigest(),
                "input_compact_chars": len(prompt_text),
                "input_compact": prompt_text,
                "raw_output": response_text,
                "raw_output_chars": len(response_text),
                "base_messages_count": int(base_messages_count),
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError:
            logger.debug("[agentic-route] route decision audit log skipped", exc_info=True)

    def _accumulate_last_llm_call_tokens_into_run(self) -> None:
        try:
            ti = getattr(self.llm, "_last_call_input_tokens", None)
            to = getattr(self.llm, "_last_call_output_tokens", None)
            tc = getattr(self.llm, "_last_call_input_cached_tokens", None)
            if ti is not None:
                self._run_route_tokens_in += int(ti or 0)
            if to is not None:
                self._run_route_tokens_out += int(to or 0)
            if tc is not None:
                self._run_route_tokens_cached += int(tc or 0)
            self._run_route_llm_calls += 1
        except (TypeError, ValueError):
            pass

    async def run_ephemeral_agentic_route_prompt(
        self,
        prompt: str,
        *,
        workspace: Any | None = None,
        trigger: str = "route",
        base_messages: list[Any] | None = None,
    ) -> str:
        """Ask for an agentic route decision without appending prompt/output to memory."""
        _ = workspace
        route_prompt = str(prompt or "").strip()
        if not route_prompt:
            return ""
        if not hasattr(self.llm, "ask_tool_stream"):
            raise RuntimeError("LLM backend has no ask_tool_stream(); use OnlineLLM / PooledLLM.")

        messages = list(base_messages) if base_messages is not None else self._memory_ctx.build_messages_for_llm()
        messages.append(Message.user_message(route_prompt))
        t0 = time.time()
        handle = StreamHandle()
        try:
            assistant_msg = await self.llm.ask_tool_stream(
                messages=messages,
                system_msgs=self._build_system_messages(),
                timeout=self._llm_stream_timeout_sec,
                tools=self._tools_with_thought,
                tool_choice="none",
                parallel_tool_calls=self._parallel_llm_tool_calls,
                handle=handle,
                collect_all_tool_calls=True,
            )
        except BaseException:
            self._record_llm_call(
                "agentic_route",
                time.time() - t0,
                None,
                "error",
                recovery=True,
                turn_kind="route",
            )
            raise

        dur_ok = time.time() - t0
        self._accumulate_last_llm_call_tokens_into_run()
        self._record_llm_call(
            "agentic_route",
            dur_ok,
            None,
            "ok",
            recovery=True,
            turn_kind="route",
        )
        text = (getattr(assistant_msg, "content", None) or "").strip()
        if not text:
            text = (getattr(assistant_msg, "reasoning_content", None) or "").strip()
        self._write_agentic_route_response(text)
        self._append_agentic_route_decision_log(
            prompt=route_prompt,
            response=text,
            trigger=trigger,
            base_messages_count=len(messages) - 1,
        )
        return text

    def _looks_like_repl_small_talk(self, text: str) -> bool:
        return looks_like_repl_small_talk(text)

    def _tool_choice_for_main_loop(self, round_idx: int, run_request: str | None) -> str:
        return _tool_choice_for_main_loop_fn(self._run_policy, round_idx, run_request)

    def _unavailable_repl_tool_names(self, tool_calls: list[Any]) -> list[str]:
        """Unavailable tool names in REPL bash-file-change mode."""
        if bool(getattr(self, "_include_write_edit_tools", True)):
            return []
        tool_map = getattr(getattr(self, "availableTools", None), "tool_map", {}) or {}
        available = set(tool_map)
        unavailable: list[str] = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None)
            if name and name not in available:
                unavailable.append(str(name))
        return unavailable

    @staticmethod
    def _tc_function_name(tc: Any) -> str | None:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None)
        return str(name) if name else None

    @staticmethod
    def _tc_arguments_dict(tc: Any) -> dict[str, Any]:
        fn = getattr(tc, "function", None)
        raw = getattr(fn, "arguments", None)
        if not isinstance(raw, str):
            raw = str(raw or "{}")
        try:
            obj = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        return obj if isinstance(obj, dict) else {}

    @staticmethod
    def _repl_file_change_command_from_tool(name: str, args: dict[str, Any]) -> str | None:
        path = str(args.get("path") or "").strip()
        if not path:
            return None
        if name == "edit":
            old = args.get("old_str")
            new = args.get("new_str")
            if not isinstance(old, str) or not isinstance(new, str):
                return None
            return (
                "python3 - <<'PY'\n"
                "from pathlib import Path\n"
                f"path = Path({path!r})\n"
                f"old = {old!r}\n"
                f"new = {new!r}\n"
                "text = path.read_text()\n"
                "count = text.count(old)\n"
                "if count != 1:\n"
                "    raise SystemExit(f'exact anchor matched {count} times in {path}')\n"
                "path.write_text(text.replace(old, new, 1))\n"
                "PY"
            )
        if name == "write":
            content = args.get("content")
            if not isinstance(content, str):
                return None
            return (
                "python3 - <<'PY'\n"
                "from pathlib import Path\n"
                f"Path({path!r}).write_text({content!r})\n"
                "PY"
            )
        return None

    def _normalize_repl_file_change_tool_calls(self, tool_calls: list[Any]) -> list[Any]:
        """Convert file-change shaped calls to bash in REPL bash-file-change mode."""
        if bool(getattr(self, "_include_write_edit_tools", True)) or not tool_calls:
            return tool_calls
        tool_map = getattr(getattr(self, "availableTools", None), "tool_map", {}) or {}
        if "bash" not in tool_map:
            return tool_calls
        normalized: list[Any] = []
        changed = 0
        for tc in tool_calls:
            name = self._tc_function_name(tc)
            if name not in {"write", "edit"}:
                normalized.append(tc)
                continue
            command = self._repl_file_change_command_from_tool(
                name,
                self._tc_arguments_dict(tc),
            )
            if not command:
                normalized.append(tc)
                continue
            normalized.append(
                SimpleNamespace(
                    id=getattr(tc, "id", ""),
                    type=getattr(tc, "type", "function"),
                    function=SimpleNamespace(
                        name="bash",
                        arguments=json.dumps(
                            {
                                "command": command,
                                "thought": "Apply the requested file change through bash.",
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ),
            )
            changed += 1
        if changed:
            self._log_info(
                "[tool-calls] normalized %d REPL file-change call(s) to bash",
                changed,
            )
        return normalized

    def _block_unavailable_repl_tool_calls(
        self,
        tool_calls: list[Any],
        unavailable_names: list[str],
    ) -> None:
        """Ask the model to retry with available REPL tools without storing bad args."""
        tool_map = getattr(getattr(self, "availableTools", None), "tool_map", {}) or {}
        available = ", ".join(f"`{name}`" for name in sorted(tool_map))
        self.memory.add_message(
            Message.user_message(
                "[Guard] Unavailable tool call blocked. "
                f"Available tools: {available}. "
                "Use `bash` for file changes; put the complete content or exact "
                "rewrite operation in the bash command. Retry now with one "
                "available tool call.",
            ),
        )
        self._log_info(
            "[tool-calls] unavailable REPL tool blocked; args omitted; count=%d",
            len(unavailable_names or _tool_call_names_from_list(tool_calls)),
        )

    async def _ask_tool_stream_with_same_round_retries(
        self,
        *,
        chat_messages: list,
        round_idx: int | None,
        request: str | None,
        tool_choice: str,
        timeout_assistant_message: str,
        hook_recovery: bool,
        timeout_log_label: str,
    ) -> tuple[Any, bool, float]:
        """Run ``ask_tool_stream`` with retries on empty-stream failures.

        Returns
        -------
        (assistant_msg, timed_out, t_llm_start_success)
            *timed_out* True ⇒ caller should not use ``assistant_msg`` (HTTP timeout path).
        *t_llm_start_success* is the start time of the successful streaming attempt (for duration).
        """
        max_a = self._llm_tool_stream_max_attempts
        base_d = self._llm_tool_stream_retry_base_delay_sec
        max_d = self._llm_tool_stream_retry_max_delay_sec
        stream_retry_cap = int(getattr(self, "_stream_repetition_retry_max", 0) or 0)
        stream_retry_used = 0
        reasoning_replay_used = False
        retry_messages = list(chat_messages)
        if bool(getattr(self, "_llm_reasoning_replay_required", False)):
            replay_messages, replay_count = messages_with_synthetic_reasoning_replay(
                retry_messages,
                purpose="thinking-mode resume compatibility",
            )
            if replay_count > 0:
                retry_messages = replay_messages
                reasoning_replay_used = True
        assistant_msg: Any = None
        t_success_start = 0.0
        for attempt in range(max_a):
            t_llm = time.time()
            self._last_stream_guard_reason = None
            self._last_stream_guard_detail = ""
            self._last_stream_guard_chars = 0
            handle = StreamHandle()
            _stream_lg = (
                self._ws_interaction_log
                if self._interaction_log_policy.llm_stream_to_file
                else None
            )
            consumer = asyncio.create_task(
                self._consume_stream(handle, _stream_lg),
            )
            try:
                assistant_msg = await self.llm.ask_tool_stream(
                    messages=retry_messages,
                    system_msgs=self._build_system_messages(),
                    timeout=self._llm_stream_timeout_sec,
                    tools=self._tools_with_thought,
                    tool_choice=tool_choice,
                    parallel_tool_calls=self._parallel_llm_tool_calls,
                    handle=handle,
                    collect_all_tool_calls=True,
                )
            except (httpx.ReadTimeout, TimeoutError) as exc:
                self._log_warning(
                    "%s round=%s %s: %r",
                    timeout_log_label,
                    round_idx,
                    type(exc).__name__,
                    exc,
                )
                print(
                    f"[ScienceAgent] Timeout ({timeout_log_label.strip()}): "
                    f"{type(exc).__name__}: {exc!r}",
                    file=sys.stderr,
                )
                if not handle.interrupted:
                    handle.finish()
                await consumer
                self.memory.add_message(Message.assistant_message(timeout_assistant_message))
                self._record_llm_call(
                    "ask_tool_stream",
                    time.time() - t_llm,
                    round_idx,
                    "error",
                    recovery=hook_recovery,
                )
                return None, True, t_success_start
            except asyncio.CancelledError:
                logger.warning(
                    "[llm-cancelled] round=%s ask_tool_stream cancelled",
                    round_idx,
                )
                if not handle.interrupted:
                    handle.finish()
                await consumer
                self._record_llm_call(
                    "ask_tool_stream",
                    time.time() - t_llm,
                    round_idx,
                    "cancelled",
                    recovery=hook_recovery,
                )
                raise
            except BaseException as exc:
                if (
                    not reasoning_replay_used
                    and attempt + 1 < max_a
                    and is_missing_reasoning_replay_error(exc)
                ):
                    replay_messages, replay_count = messages_with_synthetic_reasoning_replay(
                        retry_messages,
                        purpose="thinking-mode resume compatibility",
                    )
                    if replay_count > 0:
                        reasoning_replay_used = True
                        self._llm_reasoning_replay_required = True
                        logger.warning(
                            "[llm-retry] round=%s attempt=%d/%d %s: %r "
                            "(same round; added synthetic reasoning_content to %d "
                            "assistant history message(s))",
                            round_idx,
                            attempt + 1,
                            max_a,
                            type(exc).__name__,
                            exc,
                            replay_count,
                        )
                        print(
                            f"[ScienceAgent] LLM requires reasoning_content replay; "
                            f"retrying same round with synthetic metadata "
                            f"({replay_count} assistant message(s)).",
                            file=sys.stderr,
                        )
                        if not handle.interrupted:
                            handle.finish()
                        await consumer
                        self._record_llm_call(
                            "ask_tool_stream",
                            time.time() - t_llm,
                            round_idx,
                            "error",
                            recovery=False,
                        )
                        retry_messages = replay_messages
                        continue
                if (
                    attempt + 1 < max_a
                    and _is_retryable_empty_tool_stream_error(exc)
                ):
                    logger.warning(
                        "[llm-retry] round=%s attempt=%d/%d %s: %r "
                        "(same round; backoff before retry)",
                        round_idx,
                        attempt + 1,
                        max_a,
                        type(exc).__name__,
                        exc,
                    )
                    print(
                        f"[ScienceAgent] LLM stream failed, retrying same round "
                        f"({attempt + 1}/{max_a}): {type(exc).__name__}: {exc!r}",
                        file=sys.stderr,
                    )
                    if not handle.interrupted:
                        handle.finish()
                    await consumer
                    self._record_llm_call(
                        "ask_tool_stream",
                        time.time() - t_llm,
                        round_idx,
                        "error",
                        recovery=False,
                    )
                    delay = min(base_d * (2**attempt), max_d)
                    await asyncio.sleep(delay)
                    continue
                logger.exception(
                    "[llm-error] round=%s %s: %r",
                    round_idx,
                    type(exc).__name__,
                    exc,
                )
                print(
                    f"[ScienceAgent] LLM error: {type(exc).__name__}: {exc!r}",
                    file=sys.stderr,
                )
                if not handle.interrupted:
                    handle.finish()
                await consumer
                self._record_llm_call(
                    "ask_tool_stream",
                    time.time() - t_llm,
                    round_idx,
                    "error",
                    recovery=hook_recovery,
                )
                raise
            await consumer
            guard_reason = str(getattr(self, "_last_stream_guard_reason", "") or "").strip()
            if guard_reason:
                self._record_llm_call(
                    "ask_tool_stream",
                    time.time() - t_llm,
                    round_idx,
                    "error",
                    recovery=hook_recovery,
                )
                can_retry_stream = (
                    stream_retry_used < stream_retry_cap
                    and attempt + 1 < max_a
                )
                if not can_retry_stream:
                    self._log_warning(
                        "[llm-guard] round=%s guard=%s detail=%s retries_exhausted=%d/%d",
                        round_idx,
                        guard_reason,
                        getattr(self, "_last_stream_guard_detail", ""),
                        stream_retry_used,
                        stream_retry_cap,
                    )
                    self.memory.add_message(
                        Message.assistant_message(
                            "[LLM stream interrupted for repetition/length guard; continuing next round.]",
                        ),
                    )
                    return None, True, t_success_start
                stream_retry_used += 1
                logger.warning(
                    "[llm-guard-retry] round=%s attempt=%d/%d guard=%s detail=%s "
                    "(same round; retry with anti-repetition nudge)",
                    round_idx,
                    attempt + 1,
                    max_a,
                    guard_reason,
                    getattr(self, "_last_stream_guard_detail", ""),
                )
                retry_messages = list(chat_messages)
                retry_messages.append(
                    Message.user_message(
                        "Your previous response was interrupted due to excessive repetition "
                        "or verbosity. Respond concisely, avoid repeating prior analysis, and "
                        "proceed directly to the next action/tool call.",
                    ),
                )
                delay = min(base_d * (2**attempt), max_d)
                await asyncio.sleep(delay)
                continue
            t_success_start = t_llm
            break

        if assistant_msg is None:
            raise RuntimeError("ask_tool_stream returned without result (internal error)")
        return assistant_msg, False, t_success_start

    async def run(
        self,
        request: str | None = None,
        *,
        first_round_tool_choice: str | None = None,
    ) -> str:  # type: ignore[override]
        """Append user turn and run tool/LLM steps until a text reply or limit.

        *first_round_tool_choice* overrides :meth:`_tool_choice_for_main_loop` for ``round_idx==0``
        only (e.g. ``\"required\"`` for clone minimal handoff with no new user message).
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"ScienceAgent cannot run from state: {self.state}")

        if _should_append_run_request(self.memory, request):
            self.memory.add_message(Message.user_message(str(request)))
            self._log_info("[user] %s", truncate_for_interaction_log(str(request)))

        try:
            self.state = AgentState.RUNNING
            self._run_tokens_in = 0
            self._run_tokens_out = 0
            self._run_tokens_cached = 0
            self._run_llm_calls = 0
            self._run_route_tokens_in = 0
            self._run_route_tokens_out = 0
            self._run_route_tokens_cached = 0
            self._run_route_llm_calls = 0
            self._run_compact_tokens_in = 0
            self._run_compact_tokens_out = 0
            self._run_compact_tokens_cached = 0
            self._run_compact_llm_calls = 0
            self._call_seq = 0
            self._run_policy.reset_for_new_run()
            self._embedded_full_run_done = False
            clear_embedded_full_run_result(self._workspace_dir)
            self._run_control_user_injections = 0
            self._lnr_snapshot_ok = False
            self._lnr_snapshot_reason = ""
            self._lnr_last_valid_solution_sha = ""
            self._lnr_last_valid_submission_sha = ""
            self._lnr_last_valid_bash_cmd = ""
            self._lnr_result_md_after_success_pending = False
            self._lnr_ledger_committed = False
            self._lnr_ledger_step_count = 0
            self._lnr_ledger_new_entries = 0
            if str(getattr(self, "_lnr_agentic_text_only_route_prompt", "") or "").strip():
                self._lnr_agentic_text_only_route_prompt_injected = False
                self._lnr_agentic_text_only_route_without_result_md = False
                try:
                    (self._agentic_route_log_dir() / "agentic_route_response.md").unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.debug("[agentic-route] stale route response cleanup skipped", exc_info=True)
            self._consecutive_write_syntax_fails = 0
            self._initial_solution_sha = self._sha256_of_solution()
            self._mid_run_compacted = False
            guard_mgr = getattr(self, "_guard_manager", None)
            if guard_mgr is not None:
                guard_mgr.reset_all()
            if hasattr(self, "_reset_edit_read_guard_state"):
                self._reset_edit_read_guard_state()
            aborted_by_policy = False
            abort_reason = ""
            initial_max = int(self.max_steps)
            effective_max = initial_max
            self._effective_max_steps = effective_max
            _round = 0
            while _round < effective_max:
                self._current_round = _round
                self._effective_max_steps = effective_max
                if guard_mgr is not None and guard_mgr.should_terminate_run():
                    self._log_info(
                        "[policy] early abort at round %d: no-progress hardstop",
                        _round,
                    )
                    aborted_by_policy = True
                    abort_reason = "no-progress hardstop"
                    break
                rctx_start = RoundContext(
                    _round,
                    effective_max,
                    "",
                    self._workspace_dir,
                )
                if not self._run_policy.on_round_start(rctx_start):
                    self._log_info("[policy] early abort at round %d", _round)
                    aborted_by_policy = True
                    abort_reason = "run-policy"
                    break
                self._log_iteration_header(_round + 1)
                self._lnr_maybe_fresh_workspace_hint_first_round(_round)
                self._lnr_maybe_periodic_inject_at_round_start(_round)
                self._sync_resource_state_summary_slot()
                chat_messages, omitted = self._memory_ctx.build_messages_for_llm_with_stats()
                compact_on_context_threshold = bool(
                    getattr(self, "_lnr_compact_on_context_threshold", False),
                )
                auto_compact_on_threshold = omitted > 0 and (
                    compact_on_context_threshold
                    or not bool(getattr(self, "_lnr_superloop_enabled", False))
                )
                legacy_compact_on_threshold = (
                    self._mid_run_compact_enabled
                    and not self._mid_run_compacted
                    and omitted > 0
                )
                if auto_compact_on_threshold or legacy_compact_on_threshold:
                    compact_event_cb = getattr(self, "_context_compact_event_callback", None)
                    compact_mode = "inband" if auto_compact_on_threshold else "durable"
                    compact_recent_keep = 0 if compact_on_context_threshold else 8
                    compact_tokens_before = {
                        "tokens_in": int(getattr(self, "_run_compact_tokens_in", 0) or 0),
                        "tokens_out": int(getattr(self, "_run_compact_tokens_out", 0) or 0),
                        "tokens_cached": int(getattr(self, "_run_compact_tokens_cached", 0) or 0),
                    }

                    def _emit_compact_event(phase: str, **payload: Any) -> bool:
                        if not callable(compact_event_cb):
                            return False
                        try:
                            compact_event_cb(phase=phase, round=_round + 1, **payload)
                            return True
                        except Exception:
                            logger.debug("[context-compact] event callback failed", exc_info=True)
                            return False

                    context_limit_estra_cb = getattr(self, "_context_limit_estra_callback", None)
                    if auto_compact_on_threshold and callable(context_limit_estra_cb):
                        _emit_compact_event(
                            "estra_check",
                            mode=compact_mode,
                            reason="context_limit",
                            omitted_before=omitted,
                            recent_keep=compact_recent_keep,
                        )
                        try:
                            estra_out = context_limit_estra_cb(
                                agent=self,
                                round_idx=_round,
                                max_steps=effective_max,
                                omitted=omitted,
                            )
                            if asyncio.iscoroutine(estra_out):
                                estra_out = await estra_out
                        except Exception:
                            logger.debug(
                                "[context-compact] context-limit estra callback failed",
                                exc_info=True,
                            )
                            estra_out = None
                        if estra_out:
                            _emit_compact_event(
                                "deferred",
                                mode=compact_mode,
                                reason="estra_restore",
                                omitted_before=omitted,
                                result=str(estra_out)[:240],
                            )
                            self._sync_last_run_token_totals()
                            return str(estra_out)

                    emitted_start = _emit_compact_event(
                        "started",
                        mode=compact_mode,
                        reason="context_limit" if auto_compact_on_threshold else "mid_run",
                        omitted_before=omitted,
                        recent_keep=compact_recent_keep,
                    )
                    if not emitted_start:
                        self._log_info(
                            "[context-compact] context threshold exceeded; %d message(s) would be omitted. "
                            "running %s compact before the next LLM request",
                            omitted,
                            compact_mode,
                        )
                    compact_out = (
                        await self.compact_inband(mid_run=True)
                        if auto_compact_on_threshold
                        else await self.compact(mid_run=True)
                    )
                    co = (compact_out or "").strip()
                    if co.startswith("Compact failed") and compact_on_context_threshold:
                        _emit_compact_event(
                            "failed",
                            mode="inband",
                            status="failed",
                            omitted_before=omitted,
                            reason_detail=co[:240],
                            fallback="durable",
                        )
                        self._log_warning(
                            "[context-compact] in-band compact failed in compact-only mode; "
                            "falling back to durable compact: %s",
                            co[:200],
                        )
                        _emit_compact_event(
                            "started",
                            mode="durable",
                            reason="fallback",
                            omitted_before=omitted,
                            recent_keep=8,
                        )
                        compact_out = await self.compact(mid_run=True)
                        co = (compact_out or "").strip()
                        compact_mode = "durable"
                    if co.startswith("Compact failed"):
                        self._log_warning(
                            "[context-compact] compact did not replace history: %s",
                            co[:200],
                        )
                    elif co.startswith("Compacted "):
                        self._mid_run_compacted = True
                        if request:
                            try:
                                stored = self.memory.chat_history_memory.retrieve(
                                    window_size=None,
                                )
                                has_current_request = any(
                                    r.memory_record.message.role == "user"
                                    and r.memory_record.message.content == request
                                    for r in stored
                                )
                            except Exception:
                                has_current_request = False
                            if not has_current_request and _should_append_run_request(self.memory, request):
                                self.memory.add_message(Message.user_message(str(request)))
                    chat_messages, omitted_after_compact = (
                        self._memory_ctx.build_messages_for_llm_with_stats()
                    )
                    compact_tokens_after = {
                        "tokens_in": int(getattr(self, "_run_compact_tokens_in", 0) or 0),
                        "tokens_out": int(getattr(self, "_run_compact_tokens_out", 0) or 0),
                        "tokens_cached": int(getattr(self, "_run_compact_tokens_cached", 0) or 0),
                    }
                    summary_chars: int | None = None
                    m_summary = re.search(r"\((\d+)\s+chars\)", co)
                    if m_summary:
                        try:
                            summary_chars = int(m_summary.group(1))
                        except ValueError:
                            summary_chars = None
                    status = "failed" if co.startswith("Compact failed") else (
                        "skipped" if co == "Nothing to compact." else "ok"
                    )
                    _emit_compact_event(
                        "finished",
                        mode=compact_mode,
                        status=status,
                        omitted_before=omitted,
                        omitted_after=omitted_after_compact,
                        summary_chars=summary_chars,
                        tokens_in=compact_tokens_after["tokens_in"] - compact_tokens_before["tokens_in"],
                        tokens_out=compact_tokens_after["tokens_out"] - compact_tokens_before["tokens_out"],
                        tokens_cached=compact_tokens_after["tokens_cached"] - compact_tokens_before["tokens_cached"],
                        result=co[:240],
                    )
                    if (
                        compact_on_context_threshold
                        and omitted_after_compact > 0
                        and compact_mode != "durable"
                    ):
                        self._log_warning(
                            "[context-compact] %s compact still omitted %d message(s); "
                            "retrying durable compact",
                            compact_mode,
                            omitted_after_compact,
                        )
                        _emit_compact_event(
                            "started",
                            mode="durable",
                            reason="post_compact_omitted_fallback",
                            omitted_before=omitted_after_compact,
                            recent_keep=0,
                        )
                        fallback_tokens_before = {
                            "tokens_in": int(getattr(self, "_run_compact_tokens_in", 0) or 0),
                            "tokens_out": int(getattr(self, "_run_compact_tokens_out", 0) or 0),
                            "tokens_cached": int(getattr(self, "_run_compact_tokens_cached", 0) or 0),
                        }
                        compact_out = await self.compact(mid_run=True)
                        co = (compact_out or "").strip()
                        compact_mode = "durable"
                        chat_messages, omitted_after_compact = (
                            self._memory_ctx.build_messages_for_llm_with_stats()
                        )
                        compact_tokens_after = {
                            "tokens_in": int(getattr(self, "_run_compact_tokens_in", 0) or 0),
                            "tokens_out": int(getattr(self, "_run_compact_tokens_out", 0) or 0),
                            "tokens_cached": int(getattr(self, "_run_compact_tokens_cached", 0) or 0),
                        }
                        status = "failed" if co.startswith("Compact failed") else (
                            "skipped" if co == "Nothing to compact." else "ok"
                        )
                        _emit_compact_event(
                            "finished",
                            mode="durable",
                            status=status,
                            omitted_before=omitted,
                            omitted_after=omitted_after_compact,
                            summary_chars=summary_chars,
                            tokens_in=compact_tokens_after["tokens_in"] - fallback_tokens_before["tokens_in"],
                            tokens_out=compact_tokens_after["tokens_out"] - fallback_tokens_before["tokens_out"],
                            tokens_cached=compact_tokens_after["tokens_cached"] - fallback_tokens_before["tokens_cached"],
                            result=co[:240],
                        )
                    if compact_on_context_threshold and omitted_after_compact > 0:
                        raise RuntimeError(
                            "lnr compact did not fit context; refusing "
                            "to send an omitted-history main-agent request"
                        )
                self._maybe_log_lnr_llm_turn(_round)
                if not hasattr(self.llm, "ask_tool_stream"):
                    raise RuntimeError(
                        "LLM backend has no ask_tool_stream(); use OnlineLLM / PooledLLM.",
                    )

                try:
                    self._resource_advisory_base_messages = list(chat_messages)
                    self._resource_advisory_base_system_msgs = self._build_system_messages()
                except Exception:
                    logger.debug("[resource-advisory] main-prefix snapshot skipped", exc_info=True)

                transient_user_prompt = str(getattr(self, "_lnr_transient_user_prompt", "") or "").strip()
                if transient_user_prompt:
                    if str(getattr(self, "_lnr_transient_context_mode", "") or "") == "stage_commit_compact":
                        chat_messages = self._build_lnr_stage_commit_compact_messages(
                            base_messages=list(chat_messages),
                            transient_user_prompt=transient_user_prompt,
                        )
                    else:
                        chat_messages = list(chat_messages)
                        chat_messages.append(Message.user_message(transient_user_prompt))
                    self._lnr_transient_user_prompt_active = True
                else:
                    self._lnr_transient_user_prompt_active = False

                main_tool_choice = (
                    first_round_tool_choice
                    if _round == 0 and first_round_tool_choice is not None
                    else self._tool_choice_for_main_loop(_round, request)
                )
                if bool(getattr(self, "_lnr_transient_tool_choice_none", False)):
                    main_tool_choice = "none"

                (
                    assistant_msg,
                    timeout_exit,
                    t_llm_success,
                ) = await self._ask_tool_stream_with_same_round_retries(
                    chat_messages=chat_messages,
                    round_idx=_round,
                    request=request,
                    tool_choice=main_tool_choice,
                    timeout_assistant_message=(
                        "[LLM stream timed out; continuing — please retry or shorten the request.]"
                    ),
                    hook_recovery=False,
                    timeout_log_label="[timeout]",
                )
                if timeout_exit:
                    self._lnr_on_round_complete(None)
                    _round += 1
                    continue

                self._maybe_log_sft_turn(
                    round_idx=_round,
                    input_messages=chat_messages,
                    assistant_msg=assistant_msg,
                    system_prompt=self._build_system_prompt(),
                )

                dur_ok = time.time() - t_llm_success
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
                tool_calls_raw = self._normalize_repl_file_change_tool_calls(
                    tool_calls_raw,
                )
                _first_main: str | None = None
                _first_bash_kind: str | None = None
                if tool_calls_raw:
                    try:
                        _fnm = tool_calls_raw[0].function
                        _first_main = getattr(_fnm, "name", None) or None
                        if _first_main == "bash":
                            _raw_args0 = (
                                _fnm.arguments
                                if isinstance(_fnm.arguments, str)
                                else str(_fnm.arguments)
                            )
                            _args0 = json.loads(_raw_args0 or "{}")
                            _first_bash_kind = classify_bash_command(
                                str(_args0.get("command") or ""),
                            )
                    except Exception:
                        _first_main = None
                        _first_bash_kind = None
                _unavailable_repl_tools = self._unavailable_repl_tool_names(tool_calls_raw)
                if _unavailable_repl_tools:
                    _first_main = "invalid_tool"
                    _first_bash_kind = None
                _stage_commit_text_pending = bool(getattr(self, "_lnr_stage_commit_text_pending", False))
                _resource_advisory_text_pending = bool(getattr(self, "_lnr_resource_advisory_text_pending", False))
                _transient_turn_kind = str(getattr(self, "_lnr_transient_turn_kind", "") or "").strip()
                _turn_kind = "tool" if tool_calls_raw else "qa"
                if _stage_commit_text_pending:
                    _turn_kind = "stage_commit_tool_blocked" if tool_calls_raw else "stage_commit_text"
                elif _resource_advisory_text_pending:
                    _turn_kind = "inline_resource_advisory_tool_blocked" if tool_calls_raw else "inline_resource_advisory"
                elif not tool_calls_raw and bool(getattr(self, "_lnr_transient_user_prompt_active", False)):
                    _turn_kind = _transient_turn_kind or "stage_commit_text"
                self._record_llm_call(
                    "ask_tool_stream",
                    dur_ok,
                    _round,
                    "ok",
                    recovery=False,
                    turn_kind=_turn_kind,
                    first_tool_name=_first_main,
                    first_bash_kind=_first_bash_kind,
                )
                if self._parallel_bash_enabled and tool_calls_raw:
                    # Teleport F5: print the search-wide cumulative round when offset>0.
                    _round_disp = _round + int(
                        getattr(self, "_search_round_offset", 0) or 0,
                    )
                    self._log_info(
                        "[tool-calls-count] round=%d n=%d first=%s",
                        _round_disp,
                        len(tool_calls_raw),
                        _first_main or "?",
                    )

                if tool_calls_raw and (_stage_commit_text_pending or _resource_advisory_text_pending):
                    blocked_names = [self._tc_function_name(tc) or "unknown" for tc in tool_calls_raw]
                    block_label = "resource-advisory" if _resource_advisory_text_pending else "stage-commit"
                    self._log_info(
                        "[%s] blocked tool call(s) during text-mode turn: %s",
                        block_label,
                        ",".join(blocked_names),
                    )
                    if _resource_advisory_text_pending:
                        text = (getattr(assistant_msg, "content", None) or "").strip()
                        if not text:
                            text = (getattr(assistant_msg, "reasoning_content", None) or "").strip()
                        setattr(self, "_lnr_resource_advisory_tool_call_rejected", True)
                        text = text or "RESOURCE_ADVISORY_TOOL_CALL_REJECTED: " + ",".join(blocked_names)
                    else:
                        text = "STAGE_COMMIT_TOOL_CALL_BLOCKED: " + ",".join(blocked_names)
                    lhr_text_only_cb = getattr(self, "_lnr_text_only_callback", None)
                    if callable(lhr_text_only_cb):
                        lhr_out = lhr_text_only_cb(
                            agent=self,
                            assistant_text=text,
                            round_idx=_round,
                            max_steps=effective_max,
                        )
                        if asyncio.iscoroutine(lhr_out):
                            lhr_out = await lhr_out
                        if lhr_out:
                            self._sync_last_run_token_totals()
                            return str(lhr_out)
                    if bool(getattr(self, "_lnr_stage_commit_text_handled", False)):
                        self._lnr_stage_commit_text_handled = False
                    if bool(getattr(self, "_lnr_resource_advisory_text_handled", False)):
                        self._lnr_resource_advisory_text_handled = False
                    self._sync_last_run_token_totals()
                    self._lnr_on_round_complete([])
                    _round += 1
                    continue

                if _unavailable_repl_tools:
                    self._block_unavailable_repl_tool_calls(
                        tool_calls_raw,
                        _unavailable_repl_tools,
                    )
                    self._lnr_on_round_complete([])
                    _round += 1
                    continue

                if tool_calls_raw:
                    self._run_policy.on_tool_calls(
                        RoundContext(
                            _round,
                            effective_max,
                            "",
                            self._workspace_dir,
                        ),
                    )

                if not tool_calls_raw:
                    text = (getattr(assistant_msg, "content", None) or "").strip()
                    if not text:
                        text = "(empty assistant message)"
                    _ar = getattr(assistant_msg, "reasoning_content", None)
                    if (
                        isinstance(self._run_policy, AutoContinuePolicy)
                        and _round == 0
                        and request
                        and self._looks_like_repl_small_talk(request)
                    ):
                        self.memory.add_message(
                            Message.assistant_message(text, reasoning_content=_ar),
                        )
                        self._log_info(
                            "[assistant] %s (repl small-talk, skip tool nudge)",
                            truncate_for_interaction_log(text),
                        )
                        self._sync_last_run_token_totals()
                        return text
                    if hasattr(self, "_lnr_suppress_current_text_only_memory"):
                        self._lnr_suppress_current_text_only_memory = ""
                    lhr_text_only_cb = getattr(
                        self,
                        "_lnr_text_only_callback",
                        None,
                    )
                    if callable(lhr_text_only_cb):
                        lhr_out = lhr_text_only_cb(
                            agent=self,
                            assistant_text=text,
                            round_idx=_round,
                            max_steps=effective_max,
                        )
                        if asyncio.iscoroutine(lhr_out):
                            lhr_out = await lhr_out
                        if lhr_out:
                            self._sync_last_run_token_totals()
                            return str(lhr_out)
                    if bool(getattr(self, "_lnr_stage_commit_text_handled", False)):
                        self._lnr_stage_commit_text_handled = False
                        self._sync_last_run_token_totals()
                        self._lnr_on_round_complete([])
                        _round += 1
                        continue
                    suppress_text_only_memory_reason = str(
                        getattr(self, "_lnr_suppress_current_text_only_memory", "") or "",
                    ).strip()
                    route_prompt = str(
                        getattr(self, "_lnr_agentic_text_only_route_prompt", "") or "",
                    ).strip()
                    if route_prompt:
                        route_allowed = True
                        if bool(getattr(self, "_lnr_stage_commit_enabled", False)):
                            route_allowed = (
                                bool(getattr(self, "_lnr_last_valid_bare_run_is_local", False))
                                and not bool(getattr(self, "_lnr_stage_commit_guard_failed", False))
                            )
                        if route_allowed:
                            self._log_info(
                                "[assistant-text-only] %s (agentic route prompt; ephemeral)",
                                truncate_for_interaction_log(text),
                            )
                            route_messages = list(chat_messages)
                            route_messages.append(
                                Message.assistant_message(text, reasoning_content=_ar),
                            )
                            route_text = await self.run_ephemeral_agentic_route_prompt(
                                route_prompt,
                                trigger="text_only",
                                base_messages=route_messages,
                            )
                            self._lnr_agentic_text_only_route_prompt_injected = True
                            self._lnr_agentic_text_only_route_without_result_md = not (
                                self._workspace_dir / "result.md"
                            ).exists()
                            self._log_info(
                                "[agentic-route] %s",
                                truncate_for_interaction_log(route_prompt),
                            )
                            self._sync_last_run_token_totals()
                            return (text + ("\n" + route_text if route_text else "")).strip()
                        self._log_info(
                            "[assistant-text-only] %s (long-horizon route deferred: no local run-control-ready run)",
                            truncate_for_interaction_log(text),
                        )
                        text = (
                            "[Long-horizon route deferred: no local run-control-ready "
                            "solution run in this node. Continue with a concrete "
                            "tool action before routing.]"
                        )
                        _ar = None
                        suppress_text_only_memory_reason = ""
                    force_result_md = getattr(
                        self,
                        "_lnr_should_force_result_md_now",
                        None,
                    )
                    if callable(force_result_md) and force_result_md():
                        if suppress_text_only_memory_reason:
                            self._log_info(
                                "[assistant-text-only] %s (memory suppressed: %s; result.md pending)",
                                truncate_for_interaction_log(text),
                                suppress_text_only_memory_reason,
                            )
                        else:
                            self.memory.add_message(
                                Message.assistant_message(text, reasoning_content=_ar),
                            )
                            self._log_info(
                                "[assistant-text-only] %s (result.md pending: continue)",
                                truncate_for_interaction_log(text),
                            )
                        repeat_prompt = getattr(
                            self,
                            "_lnr_repeat_result_md_after_success_prompt",
                            None,
                        )
                        if callable(repeat_prompt):
                            repeat_prompt(reason="text-only response")
                        self._lnr_on_round_complete([])
                        _round += 1
                        continue
                    rctx_text = RoundContext(
                        _round,
                        effective_max,
                        text,
                        self._workspace_dir,
                    )
                    cont, inject_msg = self._run_policy.on_text_only(rctx_text)
                    if not cont:
                        if suppress_text_only_memory_reason:
                            self._log_info(
                                "[assistant] %s (memory suppressed: %s)",
                                truncate_for_interaction_log(text),
                                suppress_text_only_memory_reason,
                            )
                        else:
                            self.memory.add_message(
                                Message.assistant_message(text, reasoning_content=_ar),
                            )
                            self._log_info(
                                "[assistant] %s",
                                truncate_for_interaction_log(text),
                            )
                        self._sync_last_run_token_totals()
                        return text
                    if suppress_text_only_memory_reason:
                        self._log_info(
                            "[assistant-text-only] %s (memory suppressed: %s; policy: continue)",
                            truncate_for_interaction_log(text),
                            suppress_text_only_memory_reason,
                        )
                    else:
                        self.memory.add_message(
                            Message.assistant_message(text, reasoning_content=_ar),
                        )
                        self._log_info(
                            "[assistant-text-only] %s (policy: continue)",
                            truncate_for_interaction_log(text),
                        )
                    if inject_msg:
                        adjust_injection = getattr(
                            self,
                            "_lnr_adjust_policy_injection",
                            None,
                        )
                        if callable(adjust_injection):
                            inject_msg = adjust_injection(inject_msg)
                        self.memory.add_message(Message.user_message(inject_msg))
                        self._log_info(
                            "[policy-inject] %s",
                            truncate_for_interaction_log(inject_msg),
                        )
                    self._lnr_on_round_complete([])
                    _round += 1
                    continue

                mode, tool_calls_to_run = self._normalize_tool_calls_for_execution(
                    tool_calls_raw,
                )
                if mode == "blocked_write_edit_bundle":
                    self._record_blocked_write_edit_bundle(tool_calls_to_run)
                    self._lnr_on_round_complete([])
                    _round += 1
                    continue

                if mode == "parallel_readonly" and len(tool_calls_to_run) > 1:
                    early_out, new_em = await self._execute_parallel_readonly_bundle(
                        assistant_msg,
                        tool_calls_to_run,
                        effective_max=effective_max,
                        initial_max=initial_max,
                        recovery=False,
                    )
                    if new_em is not None:
                        effective_max = new_em
                    if early_out is not None:
                        self._sync_last_run_token_totals()
                        return early_out
                    names_done = _tool_call_names_from_list(tool_calls_to_run)
                    self._lnr_on_round_complete(names_done)
                    _round += 1
                    continue

                if mode == "parallel_bash" and len(tool_calls_to_run) > 1:
                    early_out, new_em = await self._execute_parallel_bash_bundle(
                        assistant_msg,
                        tool_calls_to_run,
                        effective_max=effective_max,
                        initial_max=initial_max,
                        recovery=False,
                    )
                    if new_em is not None:
                        effective_max = new_em
                    if early_out is not None:
                        self._sync_last_run_token_totals()
                        return early_out
                    names_done = _tool_call_names_from_list(tool_calls_to_run)
                    self._lnr_on_round_complete(names_done)
                    _round += 1
                    continue

                if mode == "sequential" and len(tool_calls_to_run) > 1:
                    early_out, new_em = await self._execute_sequential_bundle(
                        assistant_msg,
                        tool_calls_to_run,
                        effective_max=effective_max,
                        initial_max=initial_max,
                    )
                    if new_em is not None:
                        effective_max = new_em
                    if early_out is not None:
                        self._sync_last_run_token_totals()
                        return early_out
                    names_done = _tool_call_names_from_list(tool_calls_to_run)
                    self._lnr_on_round_complete(names_done)
                    _round += 1
                    continue

                tc = tool_calls_to_run[0]
                narrow_assistant = SimpleNamespace(
                    content=getattr(assistant_msg, "content", None) or "",
                    reasoning_content=getattr(assistant_msg, "reasoning_content", None),
                    tool_calls=[tc],
                )
                self._add_assistant_api_message(narrow_assistant)
                early_out, effective_max = await self._run_one_tool_after_assistant_logged(
                    tc,
                    assistant_msg,
                    effective_max,
                    initial_max,
                    log_assistant_thinking=True,
                )
                if early_out is not None:
                    self._sync_last_run_token_totals()
                    return early_out
                names_done = _tool_call_names_from_list(tool_calls_to_run)
                self._lnr_on_round_complete(names_done)
                _round += 1

            if aborted_by_policy:
                if abort_reason == "no-progress hardstop":
                    abort_msg = "[ScienceAgent] Run stopped early by no-progress hardstop."
                else:
                    abort_msg = "[ScienceAgent] Run aborted by policy."
                self._log_info("[policy] %s", abort_msg)
                self.memory.add_message(Message.assistant_message(abort_msg))
                if self._ui:
                    self._ui.render_agent_reply(abort_msg)
                else:
                    print(abort_msg, file=sys.stdout)
                self._sync_last_run_token_totals()
                return abort_msg

            result_md_path = self._workspace_dir / "result.md"
            rctx_exhaust = RoundContext(
                effective_max - 1,
                effective_max,
                "",
                self._workspace_dir,
            )
            n_recovery, recovery_prompt = self._run_policy.on_loop_exhausted(rctx_exhaust)
            has_current_valid = getattr(self, "_lnr_has_current_valid_bare_run", None)
            force_result_md = getattr(self, "_lnr_should_force_result_md_now", None)
            if (
                n_recovery > 0
                and callable(force_result_md)
                and force_result_md()
                and not result_md_path.exists()
            ):
                self._log_info(
                    "[result-md-recovery] current run-control-ready run pending; "
                    "using after-success result.md prompt",
                )
                recovery_prompt = RESULT_MD_AFTER_SUCCESS_RETRY_PROMPT
            if (
                n_recovery > 0
                and recovery_prompt
                and bool(getattr(self, "_lnr_stop_after_bare_solution_success", False))
                and callable(has_current_valid)
                and not has_current_valid()
            ):
                self._log_warning(
                    "[result-md-recovery] skip: no current run-control-ready bare run "
                    "for current solution.py (snapshot_reason=%s)",
                    str(getattr(self, "_lnr_snapshot_reason", "") or ""),
                )
                n_recovery = 0
                recovery_prompt = None
            if n_recovery > 0 and recovery_prompt:
                self._log_info(
                    "[result-md-recovery] result.md missing after %d rounds, "
                    "attempting up to %d extra rounds",
                    effective_max,
                    n_recovery,
                )
                self.memory.add_message(Message.user_message(recovery_prompt))
                for _recovery_round in range(n_recovery):
                    self._log_info(
                        "====== Agent Iteration recovery %d/%d =======",
                        _recovery_round + 1,
                        n_recovery,
                    )
                    try:
                        await self._run_one_recovery_llm_round()
                    except Exception:
                        self._log_warning(
                            "[result-md-recovery] round %d failed",
                            _recovery_round + 1,
                            exc_info=True,
                        )
                    if result_md_path.exists():
                        self._log_info(
                            "[result-md-recovery] result.md created on round %d",
                            _recovery_round + 1,
                        )
                        break
                else:
                    self._log_warning(
                        "[result-md-recovery] all %d attempts failed",
                        n_recovery,
                    )

            limit_msg = f"[ScienceAgent] Stopped after {effective_max} LLM rounds."
            self._log_info("[limit] Stopped after %d rounds", effective_max)
            self.memory.add_message(Message.assistant_message(limit_msg))
            if self._ui:
                self._ui.render_agent_reply(limit_msg)
            else:
                print(limit_msg, file=sys.stdout)
            self._sync_last_run_token_totals()
            return limit_msg
        finally:
            writer = getattr(self, "_write_productivity_snapshot", None)
            if callable(writer):
                try:
                    writer()
                except Exception as exc:
                    logger.warning("[productivity] failed to write productivity.json: %s", exc)
            self.state = AgentState.IDLE
