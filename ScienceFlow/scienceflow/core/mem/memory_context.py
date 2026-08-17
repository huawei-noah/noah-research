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

"""Sliding-window context, file snapshots, and optional /compact for ScienceAgent."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from deepcraft_core import Memory, Message
from deepcraft_core.memory.record import MemoryRecord
from deepcraft_core.tool import ToolCall, ToolResult
from deepcraft_core.tool.base import Function

from scienceflow.core.bash_solution_cmd import looks_like_bare_solution_run
from scienceflow.core.mem.source_snapshot import (
    _AUTO_SNAPSHOT_PREFIX,
    _CHANGED_RANGE_PREFIX,
    _CODE_MAP_PREFIX,
    _SYMBOL_SUMMARY_PREFIX,
    SymbolReadCoverage,
    _build_redundant_read_coverage_summary,
    _build_redundant_symbol_read_coverage_summary,
    _build_source_code_map,
    _build_write_auto_snapshot_block,
    _merge_read_intervals,
    _numbered_source_ranges_in_text,
    _parse_python_source_symbols,
    _range_fully_covered_by_intervals,
    _requested_read_range_for_file,
    _sha256_short_bytes,
    _symbol_sha,
    _tool_output_raw_id_from_content,
    extract_write_auto_snapshot_block,
)
from scienceflow.core.tools.file_utils import PathGuard
from scienceflow.core.tools.write_placeholder import looks_like_write_placeholder_mimicry
from scienceflow.utils.workspace_interaction_log import collapse_consecutive_repeated_lines_in_text

logger = logging.getLogger("scienceflow")

# After ``write`` to selected paths, append a line-numbered snapshot to tool memory (not a fake ``read``).
_TOOL_FEEDBACK_TRIMMED_PREFIX = "[Tool feedback trimmed for LLM context:"

# Tier 2 (silent read interception, 2026-04-28):
# When a later ``read`` request falls entirely within lines already read at the same file SHA,
# we used to append a prescriptive nudge telling the LLM to "Do not re-read for confidence".
# That phrasing leaks an out-of-band instruction and matches the forbidden-phrase list maintained
# by the teleport health check (do not call read / treat as the only source of truth).
# The replacement is a neutral note/coverage summary that states the on-disk file is unchanged
# without duplicating already-covered source ranges.
_READ_OVERLAP_COACHING = (
    "\n\n[file unchanged since last read; sha matches]"
)

# When the same rel path is written again without a bare ``python3 solution.py`` in between.
_WRITE_REPEAT_TO_SAME_FILE_COACHING = (
    "\n\n[Guard] This is write #{n} to `{rel}` in this session with no `bash "
    "python3 solution.py` between writes. Previous write(s) already landed on disk "
    "(not a placeholder). **Next turn must be bash**: run `python3 solution.py` "
    "(or equivalent validation). Do NOT write this file again until you have run it."
)

# Must match the prefix used in :meth:`MemoryContextManager.compact` when injecting the summary.
COMPACTED_CONVERSATION_SUMMARY_MARKER = "[Compacted conversation summary]"

_PINNED_TRUNC_SUFFIX = "[... pinned content truncated for budget ...]"


def _should_omit_write_body_from_llm_projection(content: str) -> bool:
    return looks_like_write_placeholder_mimicry(content)


def _message_text_for_chars(message: Message) -> str:
    c = message.content
    if isinstance(c, str):
        return c
    return str(c or "")


def _message_with_text(message: Message, text: str) -> Message:
    return message.model_copy(update={"content": text})


def _apply_pinned_budget(
    pinned: list[Message],
    *,
    budget_chars: int,
    pinned_budget_ratio: float,
) -> list[Message]:
    """Trim pinned list so total char length stays within ``budget_chars * pinned_budget_ratio``.

    The first pinned message is never truncated (task + data preview); overflow is taken from
    subsequent pinned messages, each truncated in order with :data:`_PINNED_TRUNC_SUFFIX`.

    When ``pinned_budget_ratio`` is ``<= 0``, pinned content is not subject to this cap.
    """
    if pinned_budget_ratio <= 0 or not pinned:
        return pinned
    cap = max(0, int(budget_chars * float(pinned_budget_ratio)))
    total = sum(_message_char_len(m) for m in pinned)
    if total <= cap:
        return pinned

    first = pinned[0]
    first_len = _message_char_len(first)
    out: list[Message] = [first]
    rest = pinned[1:]
    if not rest:
        return out

    remainder_cap = max(0, cap - first_len)
    suffix = "\n\n" + _PINNED_TRUNC_SUFFIX
    for m in rest:
        lc = _message_char_len(m)
        if lc <= remainder_cap:
            out.append(m)
            remainder_cap -= lc
            continue
        text = _message_text_for_chars(m)
        if remainder_cap <= len(suffix) + 1:
            out.append(_message_with_text(m, _PINNED_TRUNC_SUFFIX))
            remainder_cap = 0
            continue
        max_body = remainder_cap - len(suffix)
        new_text = text[:max_body].rstrip() + suffix
        out.append(_message_with_text(m, new_text))
        remainder_cap = 0
    return out


def _split_leading_user_prefix(
    messages: list[Message],
    *,
    max_count: int = 4,
) -> tuple[list[Message], list[Message]]:
    """Return leading user turns as a stable prefix and the remaining dynamic tail.

    LNR's cache-first path relies on keeping early task/user contract bytes fixed.
    REPL uses the same shape: task_description may be pinned separately, while the
    first auto user query lives in chat history. Treating a small number of
    consecutive leading user messages as an L1.5 prefix preserves the visible order
    without changing prompt text, and prevents sliding-window suffix selection from
    dropping or moving them. The cap avoids pathological user-only histories turning
    the whole conversation into pinned context.
    """
    if not messages or max_count <= 0:
        return [], messages
    prefix: list[Message] = []
    limit = max(0, int(max_count))
    for m in messages:
        if len(prefix) >= limit:
            break
        if getattr(m, "role", None) != "user":
            break
        prefix.append(m)
        text = _message_text_for_chars(m)
        if text.startswith(COMPACTED_CONVERSATION_SUMMARY_MARKER):
            break
    if not prefix:
        return [], messages
    return prefix, messages[len(prefix):]


def _message_priority_score(message: Message) -> int:
    """Higher = more important to keep when trimming the sliding window (errors/EDA > write/edit OK)."""
    role = getattr(message, "role", "") or ""
    content = message.content if isinstance(message.content, str) else str(message.content or "")
    if role == "tool":
        tcid = str(getattr(message, "tool_call_id", "") or "")
        if tcid.startswith("inherited_fullrun_"):
            return 88
        first = content.splitlines()[0] if content else ""
        lower = content.lower()
        if "traceback" in lower or "error:" in lower[:2000]:
            return 95
        if first.startswith("[exit="):
            head = first[:120]
            if "exit=0" in head or " exit=0," in head:
                # Successful bash — lower priority than failures / reads
                return 22
            return 92
        if first.startswith("[") and ("lines total" in first or "showing" in first):
            return 72
        if first.startswith("Edited ") and "1 replacement OK." in first:
            return 28
        if "wrote " in lower and "bytes" in lower[:200]:
            if "solution.py" in lower[:300]:
                return 75
            return 28
        return 48
    return 50


def _message_char_len(message: Message) -> int:
    """Approximate serialized size for sliding-window budget (content + tool_calls JSON)."""
    n = 0
    c = message.content
    if isinstance(c, str):
        n += len(c)
    elif c is not None:
        n += len(str(c))
    tcs = getattr(message, "tool_calls", None)
    if tcs:
        try:
            n += len(json.dumps(tcs, ensure_ascii=False))
        except (TypeError, ValueError):
            n += len(str(tcs))
    return n


def _tool_call_ids_from_assistant_message(message: Message) -> list[str]:
    """Extract tool call ids from an assistant message (dict or model objects)."""
    tcs = getattr(message, "tool_calls", None) or []
    out: list[str] = []
    for tc in tcs:
        if isinstance(tc, dict):
            out.append(str(tc.get("id") or ""))
        else:
            out.append(str(getattr(tc, "id", "") or ""))
    return out


def _ensure_complete_tool_turn_prefix(messages: list[Message]) -> list[Message]:
    """Drop leading messages until the list starts with a valid chat prefix.

    Sliding-window truncation can leave an ``assistant`` message with ``tool_calls`` but
    fewer following ``role=tool`` responses than required; OpenAI rejects such requests.
    Also drops orphaned leading ``tool`` messages (handled here for robustness).
    """
    msgs = list(messages)
    while msgs:
        if msgs[0].role == "tool":
            msgs.pop(0)
            continue
        m0 = msgs[0]
        tcs = getattr(m0, "tool_calls", None) or []
        if m0.role != "assistant" or not tcs:
            break
        n = len(tcs)
        ids = _tool_call_ids_from_assistant_message(m0)
        if len(msgs) < 1 + n:
            msgs.pop(0)
            continue
        valid = True
        for j in range(n):
            tm = msgs[1 + j]
            if tm.role != "tool":
                valid = False
                break
            tcid = getattr(tm, "tool_call_id", None) or ""
            if ids[j] and tcid and tcid != ids[j]:
                valid = False
                break
        if valid:
            break
        msgs.pop(0)
    return msgs


def _assistant_without_tool_calls_for_broken_turn(message: Message) -> Message:
    content = message.content if isinstance(message.content, str) else str(message.content or "")
    note = (
        "[Tool-call history note: an earlier assistant tool-call turn was converted "
        "to text because its tool responses were not contiguous in memory.]"
    )
    text = (content.strip() + "\n\n" + note).strip() if content.strip() else note
    return Message.assistant_message(
        text,
        reasoning_content=getattr(message, "reasoning_content", None),
    )


def _sanitize_incomplete_tool_call_turns(messages: list[Message]) -> list[Message]:
    """Convert non-contiguous assistant tool-call turns to text-only history.

    OpenAI-compatible APIs require an assistant message with N ``tool_calls`` to be
    followed immediately by N matching ``role=tool`` messages. Legacy guard/user
    injections can leave the matching tool output later in the transcript; preserve
    the information as text, but remove executable ``tool_calls`` from that turn.
    """
    out: list[Message] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        tcs = getattr(m, "tool_calls", None) or []
        if getattr(m, "role", None) != "assistant" or not tcs:
            out.append(m)
            i += 1
            continue

        expected_ids = _tool_call_ids_from_assistant_message(m)
        complete = len(messages) >= i + 1 + len(tcs)
        if complete:
            for j, expected_id in enumerate(expected_ids):
                tm = messages[i + 1 + j]
                if getattr(tm, "role", None) != "tool":
                    complete = False
                    break
                actual_id = str(getattr(tm, "tool_call_id", "") or "")
                if expected_id and actual_id and actual_id != expected_id:
                    complete = False
                    break
        if complete:
            out.append(m)
        else:
            out.append(_assistant_without_tool_calls_for_broken_turn(m))
        i += 1
    return out


def _sanitize_orphan_tool_messages(messages: list[Message]) -> list[Message]:
    """Downgrade ``role=tool`` messages that do not match pending assistant ``tool_calls``.

    Strict OpenAI-compatible APIs (e.g. DashScope) reject requests where a ``tool`` output
    has no matching ``tool_call`` id on the preceding assistant turn (in order). This can
    happen if legacy code injected a synthetic ``tool`` without a corresponding call.
    """
    out: list[Message] = []
    pending: list[str] = []

    for m in messages:
        if m.role == "assistant":
            out.append(m)
            tcs = getattr(m, "tool_calls", None) or []
            pending = _tool_call_ids_from_assistant_message(m) if tcs else []
        elif m.role == "tool":
            tcid = getattr(m, "tool_call_id", None) or ""
            if pending and tcid == pending[0]:
                pending.pop(0)
                out.append(m)
            else:
                content = m.content if isinstance(m.content, str) else str(m.content or "")
                out.append(
                    Message.user_message(
                        "[Tool output downgraded to user message "
                        "(no matching assistant tool_call; strict API compatibility)]\n"
                        + content,
                    ),
                )
        else:
            out.append(m)
            pending = []

    return out


def _tool_call_name_args_for_projection(tc: Any) -> tuple[str, dict[str, Any], str, str]:
    """Return ``(name, args, id, type)`` from a raw stored tool call."""
    tcid = str(tc.get("id") or "") if isinstance(tc, dict) else str(getattr(tc, "id", "") or "")
    tctype = str(tc.get("type") or "function") if isinstance(tc, dict) else str(getattr(tc, "type", "function") or "function")
    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
    if fn is None:
        return "", {}, tcid, tctype
    name = str(fn.get("name") or "") if isinstance(fn, dict) else str(getattr(fn, "name", "") or "")
    raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
    try:
        args = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args, tcid, tctype


def _tool_call_from_projection(
    tc: Any,
    *,
    name: str,
    args: dict[str, Any],
    tcid: str,
    tctype: str,
) -> ToolCall:
    """Build a validated tool call for the LLM view without mutating raw storage."""
    return ToolCall(
        id=tcid or _tool_call_id_from_raw(tc),
        type=tctype or "function",
        function=Function(
            name=name,
            arguments=json.dumps(args, ensure_ascii=False),
        ),
    )


def _coerce_tool_call_model_for_llm_projection(tc: Any) -> ToolCall:
    """Return a ``ToolCall`` model so downstream serializers never see raw dicts."""
    if isinstance(tc, ToolCall):
        return tc
    name, args, tcid, tctype = _tool_call_name_args_for_projection(tc)
    return _tool_call_from_projection(tc, name=name, args=args, tcid=tcid, tctype=tctype)


def _assistant_message_with_tool_calls_for_llm_projection(
    message: Message,
    tool_calls: list[Any],
) -> Message:
    """Copy an assistant message while validating projected tool calls."""
    return Message(
        role="assistant",
        content=message.content,
        tool_calls=[
            _coerce_tool_call_model_for_llm_projection(tc)
            for tc in tool_calls
        ],
        name=message.name,
        tool_call_id=message.tool_call_id,
        reasoning_content=message.reasoning_content,
    )


def _project_regular_tool_call_for_llm(tc: Any) -> Any:
    """Strip non-essential args from ordinary tool calls in the LLM-visible view."""
    name, args, tcid, tctype = _tool_call_name_args_for_projection(tc)
    if not name:
        return tc
    changed = False
    if args.pop("config", None) is not None:
        changed = True
    if args.pop("thought", None) is not None:
        changed = True
    if not changed:
        return tc
    return _tool_call_from_projection(tc, name=name, args=args, tcid=tcid, tctype=tctype)


def _write_tool_call_needs_llm_projection(tc: Any) -> bool:
    name, args, _tcid, _tctype = _tool_call_name_args_for_projection(tc)
    return (
        name == "write"
        and isinstance(args.get("content"), str)
        and _should_omit_write_body_from_llm_projection(args["content"])
    )


def _edit_tool_call_needs_llm_projection(tc: Any) -> bool:
    name, args, _tcid, _tctype = _tool_call_name_args_for_projection(tc)
    if name != "edit":
        return False
    for key in ("old_str", "new_str", "old_string", "new_string"):
        val = args.get(key)
        if not isinstance(val, str):
            continue
        if "<<<MEMORY_COMPRESSED" in val or len(val) > 240:
            return True
    return False


def _tool_call_needs_llm_projection(tc: Any) -> bool:
    return _write_tool_call_needs_llm_projection(tc) or _edit_tool_call_needs_llm_projection(tc)


def _tool_result_first_line(tool_msg: Message | None) -> str:
    if tool_msg is None:
        return "(no paired tool result in visible window)"
    content = tool_msg.content if isinstance(tool_msg.content, str) else str(tool_msg.content or "")
    first = content.splitlines()[0].strip() if content else ""
    return first or "(empty tool result)"


def _extract_code_map_block_from_tool_feedback(content: str, *, max_chars: int = 1800) -> str:
    if not content or _CODE_MAP_PREFIX not in content:
        return ""
    lines = content.splitlines()
    start = -1
    for i, line in enumerate(lines):
        if line.startswith(_CODE_MAP_PREFIX):
            start = i
            break
    if start < 0:
        return ""
    picked: list[str] = []
    for line in lines[start:]:
        if picked and line.startswith("[source-excerpt"):
            break
        if picked and line.startswith(_CHANGED_RANGE_PREFIX):
            break
        if picked and line.startswith(_SYMBOL_SUMMARY_PREFIX):
            break
        if picked and line.startswith(_AUTO_SNAPSHOT_PREFIX):
            break
        picked.append(line)
    block = "\n".join(picked).strip()
    if len(block) > max_chars:
        block = block[: max_chars - 32].rstrip() + "\n...[code-map truncated]"
    return block


def _historical_write_completion_summary(tool_msg: Message | None) -> str:
    """Stable LLM-visible completion summary for hidden historical write args."""
    if tool_msg is None:
        return ""
    content = tool_msg.content if isinstance(tool_msg.content, str) else str(tool_msg.content or "")
    if not content:
        return ""
    first = _tool_result_first_line(tool_msg)
    code_map = _extract_code_map_block_from_tool_feedback(content)
    if code_map:
        return first + "\n" + code_map
    return first


def _extract_changed_range_line_from_tool_feedback(content: str) -> str:
    if not content:
        return ""
    for line in content.splitlines():
        if line.startswith(_CHANGED_RANGE_PREFIX):
            return line.strip()
    return ""


def _project_historical_tool_result_for_llm(tc: Any, tool_msg: Message | None) -> Message:
    """Convert a historical tool turn into ordinary text when a write payload is hidden."""
    name, args, _tcid, _tctype = _tool_call_name_args_for_projection(tc)
    path = str(args.get("path") or "").replace("\\", "/").lstrip("/")
    first = _tool_result_first_line(tool_msg)
    if name == "write" and isinstance(args.get("content"), str):
        completion = _historical_write_completion_summary(tool_msg)
        lines = [
            "[Historical write tool call omitted from executable LLM context]",
            "Tool: write",
            f"Path: {path or '(unknown)'}",
            f"Result: {first}",
            (
                "This historical write payload looked like a placeholder or memory "
                "compression marker, so it is not shown as callable tool JSON here."
            ),
        ]
        if completion and completion != first:
            lines.extend(["Tool completion summary:", completion])
        lines.append(
            "For exact edit anchors, use a targeted read around the listed line range."
        )
        return Message.user_message("\n".join(lines))

    if name == "edit":
        content = tool_msg.content if tool_msg and isinstance(tool_msg.content, str) else ""
        changed = _extract_changed_range_line_from_tool_feedback(content)
        code_map = _extract_code_map_block_from_tool_feedback(content, max_chars=900)
        lines = [
            "[Historical edit tool call omitted from executable LLM context]",
            "Tool: edit",
            f"Path: {path or '(unknown)'}",
            f"Result: {first}",
            (
                "The exact old_str/new_str payload was large and is not shown as "
                "callable JSON to avoid copying memory-compression markers."
            ),
        ]
        if changed:
            lines.append(f"Changed range: {changed}")
        if code_map:
            lines.extend(["Tool completion summary:", code_map])
        lines.append("For exact edit anchors, use a targeted read around the current symbol range.")
        return Message.user_message("\n".join(lines))

    safe_args = dict(args)
    safe_args.pop("config", None)
    safe_args.pop("thought", None)
    for key in ("content", "old_str", "new_str"):
        if isinstance(safe_args.get(key), str) and len(safe_args[key]) > 240:
            safe_args[key] = safe_args[key][:200] + "...[truncated]"
    try:
        args_text = json.dumps(safe_args, ensure_ascii=False)
    except (TypeError, ValueError):
        args_text = "{}"
    return Message.user_message(
        "[Historical tool result from a turn whose large write payload was hidden]\n"
        f"Tool: {name or '(unknown)'}\n"
        f"Args: {args_text}\n"
        f"Result: {first}"
    )


def _project_messages_for_llm_view(messages: list[Message]) -> list[Message]:
    """Return an LLM-only projection, leaving the stored event log untouched.

    Assistant-authored write/edit/bash arguments are preserved for cacheable history.
    Only unsafe placeholder-like writes or legacy compressed edit payloads are converted
    to ordinary text so the model does not copy memory-compression markers as code.
    """
    out: list[Message] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if getattr(m, "role", None) != "assistant" or not getattr(m, "tool_calls", None):
            out.append(m)
            i += 1
            continue

        tcs = list(getattr(m, "tool_calls", None) or [])
        needs_projection = any(_tool_call_needs_llm_projection(tc) for tc in tcs)
        if not needs_projection:
            projected_tcs = [_project_regular_tool_call_for_llm(tc) for tc in tcs]
            out.append(_assistant_message_with_tool_calls_for_llm_projection(m, projected_tcs))
            i += 1
            continue

        tool_msgs: list[Message] = []
        j = i + 1
        expected_ids = [_tool_call_id_from_raw(tc) for tc in tcs]
        while j < len(messages) and getattr(messages[j], "role", None) == "tool":
            tool_msgs.append(messages[j])
            j += 1
            if len(tool_msgs) >= len(tcs):
                break

        content = m.content if isinstance(m.content, str) else str(m.content or "")
        summary = content.strip()
        note = (
            "[LLM memory view: one or more unsafe historical write payloads or "
            "compressed edit payloads from this assistant turn are summarized "
            "below as text, not exposed as tool-call JSON.]"
        )
        if summary:
            summary = summary + "\n\n" + note
        else:
            summary = note
        out.append(Message.assistant_message(summary, reasoning_content=getattr(m, "reasoning_content", None)))

        by_id = {
            str(getattr(tm, "tool_call_id", "") or ""): tm
            for tm in tool_msgs
        }
        for idx, tc in enumerate(tcs):
            tm = by_id.get(expected_ids[idx]) if idx < len(expected_ids) else None
            if tm is None and idx < len(tool_msgs):
                tm = tool_msgs[idx]
            out.append(_project_historical_tool_result_for_llm(tc, tm))
        i = j

    return out


def _finalize_messages_for_llm(messages: list[Message]) -> list[Message]:
    """Prefix fix + strict tool-call adjacency sanitization for LLM APIs."""
    return _sanitize_orphan_tool_messages(
        _sanitize_incomplete_tool_call_turns(
            _ensure_complete_tool_turn_prefix(messages),
        ),
    )


def _best_suffix_for_budget_priority(
    rest: list[Message],
    budget_rest: int,
    *,
    min_k: int = 0,
) -> tuple[list[Message], int]:
    """Pick ``rest[k:]`` that maximizes sum of :func:`_message_priority_score` within ``budget_rest`` chars.

    *min_k* enforces a monotone lower bound on the suffix start so the prefix sent to the LLM
    never grows backwards — improving prefix-cache hit rates across consecutive calls.

    Tie-break: prefer smaller *k* (longer suffix, more recent context). If no non-empty suffix fits,
    fall back to newest-first greedy (same as legacy sliding window).

    Returns ``(chosen_messages, actual_k)``.
    """
    n = len(rest)
    # Safety: if previous min_k is beyond current history length, reset to 0.
    if min_k > n:
        min_k = 0
    best_k = n
    best_score = -1
    for k in range(min_k, n + 1):
        suffix = rest[k:]
        total_len = sum(_message_char_len(m) for m in suffix)
        if total_len > budget_rest:
            continue
        score = sum(_message_priority_score(m) for m in suffix)
        if score > best_score or (score == best_score and k < best_k):
            best_score = score
            best_k = k
    if best_score >= 0:
        return rest[best_k:], best_k
    # Fallback: newest-first greedy, still constrained to >= min_k.
    chosen_rev: list[Message] = []
    tot = 0
    fallback_k = n
    for i, m in enumerate(reversed(rest)):
        lc = _message_char_len(m)
        if chosen_rev and tot + lc > budget_rest:
            break
        chosen_rev.append(m)
        tot += lc
        fallback_k = n - 1 - i
    actual_k = max(min_k, fallback_k) if chosen_rev else n
    return list(reversed(chosen_rev)), actual_k


_EDIT_SNAPSHOT_MARKER = "--- Current file snapshot (after edit) ---"


def compress_edit_success_output_for_memory(obs: str) -> str:
    """Drop multi-line context preview from successful edit tool output (keep header + hash line).

    Never raises: failures fall back to *obs* unchanged.
    """
    try:
        body = obs
        had_snapshot = _EDIT_SNAPSHOT_MARKER in obs
        if had_snapshot:
            body = obs.split(_EDIT_SNAPSHOT_MARKER, 1)[0].rstrip()
        lines = body.splitlines()
        if len(lines) < 2:
            return body if had_snapshot else obs
        first, last = lines[0], lines[-1]
        if not first.startswith("Edited ") or "1 replacement OK." not in first:
            return body if had_snapshot else obs
        # Allow variable hash length and optional trailing metadata on the summary line.
        if not re.match(r"^\(\d+ lines, sha256~[a-f0-9]+", last):
            return body if had_snapshot else obs
        return f"{first}\n{last}"
    except Exception:
        logger.debug("compress_edit_success_output_for_memory failed", exc_info=True)
        return obs


def _parse_write_success_metadata_from_output(text: str) -> tuple[int | None, str | None]:
    """Parse (lines, sha256_short) from WriteTool / no-op success text."""
    if not (text and isinstance(text, str)):
        return None, None
    m = re.search(
        r"\((\d+)\s+lines,.*?sha256~([a-f0-9]+)",
        text[:4000],
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def write_success_feedback_for_memory(
    rel_path: str,
    *,
    lines: int | None = None,
    sha256_short: str | None = None,
) -> str:
    """Short success line for write tool results stored in memory.

    Includes line count + content fingerprint when available so the model can
    trust the write; avoids suggesting unnecessary reads.
    """
    p = (rel_path or "").replace("\\", "/").lstrip("/").strip() or "file"
    if lines is not None and sha256_short:
        return (
            f"File `{p}` written successfully ({lines} lines, sha256~{sha256_short}). "
            "Syntax check passed; file is complete on disk. Use bash to test when ready. "
            "Do not rewrite the same content without changes."
        )
    return (
        f"File `{p}` written successfully; syntax check passed. "
        "File is on disk. Use bash to test when ready."
    )












































































def _strip_tool_feedback_trim_header(content: str) -> str:
    """Drop the generic LLM-context trim header before inherited compaction."""
    if not content.startswith(_TOOL_FEEDBACK_TRIMMED_PREFIX):
        return content
    lines = content.splitlines()
    if not lines:
        return content
    return "\n".join(lines[1:]).lstrip()


def _snapshot_inherit_limits(
    *,
    read_max_lines: int,
    write_snapshot_inherit_max_lines: int,
    write_snapshot_inherit_max_chars: int,
) -> tuple[int, int]:
    wlim = (
        int(write_snapshot_inherit_max_lines)
        if int(write_snapshot_inherit_max_lines) > 0
        else max(int(read_max_lines or 0), 120)
    )
    wch = (
        int(write_snapshot_inherit_max_chars)
        if int(write_snapshot_inherit_max_chars) > 0
        else 24000
    )
    return wlim, wch


def _edit_success_feedback_for_inherit(content: str, rel: str) -> str:
    body = _strip_tool_feedback_trim_header(content).strip()
    prefix = body.split(_AUTO_SNAPSHOT_PREFIX, 1)[0].rstrip()
    compact = compress_edit_success_output_for_memory(prefix)
    if compact.strip():
        return compact.strip()
    ln, sha = _parse_write_success_metadata_from_output(body)
    p = (rel or "file").replace("\\", "/").lstrip("/")
    if ln is not None and sha:
        return f"Edited `{p}` successfully ({ln} lines, sha256~{sha})."
    return f"Edited `{p}` successfully; file is complete on disk."


def _compact_snapshot_tool_feedback_for_inherit(
    content: str,
    *,
    tool_name: str,
    tool_args: dict[str, Any] | None,
    read_max_lines: int,
    write_snapshot_inherit_max_lines: int,
    write_snapshot_inherit_max_chars: int,
    keep_auto_snapshot: bool = True,
) -> str | None:
    """Compact write/edit feedback that carries a canonical auto-snapshot."""
    snap = extract_write_auto_snapshot_block(content)
    if not snap:
        return None
    stripped = _strip_tool_feedback_trim_header(content)
    rel = _rel_path_for_write_compact(tool_args, stripped)
    if tool_name == "edit":
        base = _edit_success_feedback_for_inherit(stripped, rel)
    else:
        ln, sha = _parse_write_success_metadata_from_output(stripped)
        base = write_success_feedback_for_memory(rel, lines=ln, sha256_short=sha)
    if not keep_auto_snapshot:
        return (
            base
            + "\n[inherited] auto-snapshot omitted; superseded by a later "
            "solution.py snapshot or the current child fork instruction."
        )
    wlim, wch = _snapshot_inherit_limits(
        read_max_lines=read_max_lines,
        write_snapshot_inherit_max_lines=write_snapshot_inherit_max_lines,
        write_snapshot_inherit_max_chars=write_snapshot_inherit_max_chars,
    )
    return base + "\n\n" + truncate_inherited_snapshot_block(
        snap,
        max_lines=wlim,
        max_chars=wch,
    )


def trim_tool_feedback_for_llm_context(feedback: str, *, max_chars: int) -> str:
    """Trim tool feedback before storing it in chat memory.

    Ordinary tool output uses the legacy head+tail trim. Write auto-snapshots are different:
    they are deliberately injected as line-numbered source context so the next turn does not
    need to re-read ``solution.py``. Preserve the snapshot block intact and trim only the
    prefix around it; the snapshot itself is already bounded by ``write_auto_snapshot_max_chars``.
    """
    if max_chars <= 0 or not feedback or len(feedback) <= max_chars:
        return feedback

    orig_len = len(feedback)
    orig_lines = len(feedback.splitlines())

    if _AUTO_SNAPSHOT_PREFIX in feedback:
        idx = feedback.find(_AUTO_SNAPSHOT_PREFIX)
        prefix = feedback[:idx].rstrip()
        snapshot = feedback[idx:].lstrip()
        trimmed_prefix = prefix
        if len(trimmed_prefix) > max_chars:
            half = max(1, max_chars // 2)
            trimmed_prefix = (
                trimmed_prefix[:half].rstrip()
                + "\n...[prefix truncated; auto-snapshot preserved]...\n"
                + trimmed_prefix[-half:].lstrip()
            )
        header = (
            f"{_TOOL_FEEDBACK_TRIMMED_PREFIX} {orig_len} chars, "
            f"{orig_lines} lines; auto-snapshot preserved]"
        )
        body = (trimmed_prefix + "\n\n" + snapshot).strip()
        return header + "\n" + body

    half = max(1, max_chars // 2)
    return (
        f"{_TOOL_FEEDBACK_TRIMMED_PREFIX} {orig_len} chars, "
        f"{orig_lines} lines -> head+tail ~{max_chars} chars]\n"
        + feedback[:half]
        + "\n...[truncated]...\n"
        + feedback[-half:]
    )


def truncate_inherited_snapshot_block(
    snap: str,
    *,
    max_lines: int,
    max_chars: int,
) -> str:
    """Trim a snapshot block for clone-inherited memory."""
    if max_lines <= 0 or max_chars <= 0:
        return snap
    lines = snap.splitlines()
    if len(lines) <= max_lines + 6 and len(snap) <= max_chars:
        return snap
    # Keep intro + meta line(s) heuristically: first block until a line starting with "     1|"
    body_start = 0
    for i, ln in enumerate(lines):
        if re.match(r"^\s*1\|", ln):
            body_start = i
            break
    head = "\n".join(lines[:body_start])
    body_lines = lines[body_start:]
    if len(body_lines) <= max_lines:
        out = head + "\n" + "\n".join(body_lines)
    else:
        keep = body_lines[:max_lines]
        out = head + "\n" + "\n".join(keep) + "\n...[inherited snapshot truncated; use read tool]..."
    if len(out) > max_chars:
        out = out[: max_chars // 2] + "\n...[truncated]...\n" + out[-(max_chars // 2) :]
    return out


def compress_read_output_for_memory(obs: str, *, max_lines: int = 80) -> str:
    """Truncate long read output keeping header + first *max_lines* content lines.

    Never raises: failures fall back to *obs* unchanged.
    """
    try:
        if max_lines <= 0:
            return obs
        lines = obs.splitlines()
        if not lines:
            return obs
        # Header line looks like: [path: N lines total, showing 1-200]
        first = lines[0]
        if not first.startswith("["):
            return obs
        body_lines = lines[1:]
        n = len(body_lines)
        if n <= max_lines:
            return obs
        kept = body_lines[:max_lines]
        return (
            first + "\n"
            + "\n".join(kept) + "\n"
            + f"...[read output truncated for LLM context: showing first {max_lines} of {n} content lines. "
            "Use offset parameter to read further.]"
        )
    except Exception:
        logger.debug("compress_read_output_for_memory failed", exc_info=True)
        return obs


def _read_args_has_explicit_range(args: dict[str, Any]) -> bool:
    return "offset" in args or "limit" in args


def _read_args_limit(args: dict[str, Any]) -> int:
    try:
        return int(args.get("limit") or 200)
    except (TypeError, ValueError):
        return 200


def _read_header_line(obs: str) -> str:
    return obs.splitlines()[0] if obs else ""


def _build_read_code_map_summary_for_memory(
    obs: str,
    *,
    rel: str,
    text: str,
    raw_id: str = "",
    max_chars: int = 2200,
) -> str:
    first = _read_header_line(obs)
    code_map = _build_source_code_map(rel, text, max_chars=max(800, max_chars - 500))
    lines = text.splitlines()
    parts = [
        first or f"[{rel}: {len(lines)} lines total]",
        (
            f"[read compressed: reducer=read_code_map_v2 "
            f"raw_chars={len(obs)} raw_lines={len(obs.splitlines())}]"
        ),
        code_map,
        (
            "[full read body omitted from memory; use read with explicit offset/limit "
            "for the target function or line range.]"
        ),
    ]
    if raw_id:
        parts.append(f"[exact raw output: {raw_id}]")
    out = "\n".join(p for p in parts if p)
    if len(out) > max_chars:
        out = out[: max(1, max_chars - 28)].rstrip() + "\n...[read code-map truncated]"
    return out


def _build_snapshot_ref_output_for_memory(
    obs: str,
    *,
    rel: str,
    sha: str,
    raw_id: str = "",
    reason: str = "source body omitted to avoid duplicate context",
) -> str:
    first = _read_header_line(obs)
    lines = [
        first or f"[{rel}: current file snapshot available]",
        (
            f"[tool-output compressed: reducer=snapshot_ref_v1 "
            f"raw_chars={len(obs)} raw_lines={len(obs.splitlines())}]"
        ),
        f"[see current snapshot: {rel} sha~{sha}; {reason}.]",
    ]
    if raw_id:
        lines.append(f"[exact raw output: {raw_id}]")
    return "\n".join(lines)


def _strip_bash_stderr_section_from_block(obs: str) -> str:
    """Drop trailing ``[stderr]`` section from a bash tool block (stdout-only for memory)."""
    lines = obs.splitlines()
    if len(lines) < 2 or not lines[0].startswith("[exit="):
        return obs
    body = "\n".join(lines[1:])
    marker = "\n[stderr]\n"
    idx = body.find(marker)
    if idx == -1:
        return obs
    new_body = body[:idx].rstrip()
    return f"{lines[0]}\n{new_body}" if new_body else lines[0]


def _bash_tool_feedback_header_is_success(feedback: str) -> bool:
    first = feedback.splitlines()[0] if feedback else ""
    if not first.startswith("[exit="):
        return False
    return first.startswith("[exit=0,") or first.startswith("[exit=0]")


def _bash_tool_feedback_is_failed_wrapped(feedback: str) -> bool:
    return bool(feedback and feedback.lstrip().startswith("Error:"))


def find_last_real_bare_solution_bash_tool_feedback(messages: Sequence[Message]) -> str | None:
    """Scan *messages* (chronological) for the last real ``bash`` tool result for a bare
    ``python3 solution.py``-class command.

    Skips synthetic :data:`inherited_fullrun_*` tool_call ids. Returns the tool **content**
    string as stored in memory (already stdout-only / verbatim for successful parent runs).
    """
    msgs = list(messages)
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if getattr(m, "role", None) != "tool":
            continue
        if (getattr(m, "name", None) or "") != "bash":
            continue
        tcid = str(getattr(m, "tool_call_id", "") or "")
        if tcid.startswith("inherited_fullrun_"):
            continue
        cmd = _bash_command_for_tool_message(msgs, i)
        if not cmd or not looks_like_bare_solution_run(cmd):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content or "")
        if not content.strip():
            continue
        return content
    return None


def clone_memory_has_successful_bare_solution_bash(messages: Sequence[Message]) -> bool:
    """True if the last bare ``solution.py`` bash in *messages* is a successful run (for skipping
    synthetic full-run tail inject).
    """
    fb = find_last_real_bare_solution_bash_tool_feedback(messages)
    if not fb:
        return False
    if _bash_tool_feedback_is_failed_wrapped(fb):
        return False
    return _bash_tool_feedback_header_is_success(fb)


def memory_has_inherited_fullrun_synthetic_round(messages: Sequence[Message]) -> bool:
    """True if a fork-ready synthetic full-run recap exists (``tool_call_id`` ``inherited_fullrun_*``).

    Used for idempotent parent persistence and to avoid duplicate child fallback injects.
    """
    for m in reversed(list(messages)):
        if getattr(m, "role", None) != "tool":
            continue
        tid = str(getattr(m, "tool_call_id", "") or "")
        if tid.startswith("inherited_fullrun_"):
            return True
    return False


def _tool_call_id_from_raw(tc: Any) -> str:
    if isinstance(tc, dict):
        return str(tc.get("id") or "")
    return str(getattr(tc, "id", "") or "")


def _bash_command_from_tool_call(tc: Any) -> str | None:
    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
    if fn is None:
        return None
    name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
    if name != "bash":
        return None
    raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
    try:
        args = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(args, dict):
        return None
    return str(args.get("command") or "")


def _bash_command_for_tool_message(messages: list[Message], tool_idx: int) -> str | None:
    tcid = str(getattr(messages[tool_idx], "tool_call_id", "") or "")
    j = tool_idx - 1
    while j >= 0 and getattr(messages[j], "role", None) == "tool":
        j -= 1
    if j < 0:
        return None
    assistant = messages[j]
    if getattr(assistant, "role", None) != "assistant":
        return None
    for tc in getattr(assistant, "tool_calls", None) or []:
        if _tool_call_id_from_raw(tc) == tcid:
            return _bash_command_from_tool_call(tc)
    return None


def _tool_args_dict_for_tool_message(messages: list[Message], tool_idx: int) -> dict[str, Any] | None:
    tcid = str(getattr(messages[tool_idx], "tool_call_id", "") or "")
    j = tool_idx - 1
    while j >= 0 and getattr(messages[j], "role", None) == "tool":
        j -= 1
    if j < 0:
        return None
    assistant = messages[j]
    if getattr(assistant, "role", None) != "assistant":
        return None
    for tc in getattr(assistant, "tool_calls", None) or []:
        if _tool_call_id_from_raw(tc) != tcid:
            continue
        fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
        if fn is None:
            return None
        raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
        try:
            args = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            return None
        if isinstance(args, dict):
            return args
        return None
    return None


# Default regex strings for lines that must survive aggressive truncation on clone inherit.
_DEFAULT_CLONE_INHERIT_SIGNAL_PATTERNS: tuple[str, ...] = (
    r"Final Validation Score",
    r"Validation MCC",
    r"Best threshold",
    r"(?i)submission is valid",
    r"(?i)is_valid\s*=\s*True",
    r"Traceback",
    r"(?i)Error:",
    r"(?i)Killed",
    r"(?i)\bOOM\b",
    r"(?i)out of memory",
    r"(?i)cuda out of memory",
)


def _compile_clone_inherit_signal_patterns(raw: list[str] | None) -> list[re.Pattern[str]]:
    src = list(raw) if raw else list(_DEFAULT_CLONE_INHERIT_SIGNAL_PATTERNS)
    out: list[re.Pattern[str]] = []
    for s in src:
        try:
            out.append(re.compile(str(s)))
        except re.error:
            logger.warning("Invalid clone_inherit_signal_patterns entry %r; skipped", s)
    return out


def slice_clone_minimal_inherited_messages(msgs: list[Message]) -> list[Message]:
    """Keep leading user block + tail from parent node's last assistant (inclusive).

    Drops middle transcript for token savings while preserving the original task user
    turn(s) and the latest model reasoning plus any following tool results.
    """
    if len(msgs) < 2:
        return msgs
    j = 0
    while j < len(msgs) and getattr(msgs[j], "role", "") == "user":
        j += 1
    leading = msgs[:j]
    last_a: int | None = None
    for i in range(len(msgs) - 1, -1, -1):
        if getattr(msgs[i], "role", "") == "assistant":
            last_a = i
            break
    if last_a is None:
        return msgs
    if last_a < j:
        return msgs
    tail = msgs[last_a:]
    if not leading:
        return tail
    return leading + tail


def _is_inherited_write_path_solution_py(path: str) -> bool:
    """True when a write tool targets ``solution.py`` (any relative path ending with it)."""
    p = str(path or "").replace("\\", "/").strip().lstrip("/")
    if not p:
        return False
    return p.endswith("solution.py") or p.split("/")[-1] == "solution.py"


def _norm_code_extensions_for_inherit(raw: Any = None) -> tuple[str, ...]:
    vals = raw if isinstance(raw, (list, tuple)) else (".py",)
    out = tuple(
        e if e.startswith(".") else f".{e}"
        for e in (str(x).strip().lower() for x in vals)
        if e
    )
    return out or (".py",)


def _is_inherited_code_file_path(
    path: str,
    code_extensions: tuple[str, ...] = (".py",),
) -> bool:
    p = str(path or "").replace("\\", "/").strip().lstrip("/")
    return bool(p and PurePosixPath(p).suffix.lower() in code_extensions)


def _snapshot_path_from_auto_snapshot_content(content: str) -> str:
    if not content or _AUTO_SNAPSHOT_PREFIX not in content:
        return ""
    tail = content.split(_AUTO_SNAPSHOT_PREFIX, 1)[1]
    first = tail.split("\n", 1)[0].strip()
    if first.endswith("]"):
        first = first[:-1].strip()
    return first.replace("\\", "/").lstrip("/")


def _read_path_from_content_or_args(content: str, tool_args: dict[str, Any] | None) -> str:
    if tool_args:
        rel = str(tool_args.get("path") or "").replace("\\", "/").strip().lstrip("/")
        if rel:
            return rel
    first = content.splitlines()[0] if content else ""
    m = re.match(r"^\[([^:\]]+):\s+\d+\s+lines total", first)
    if m:
        return m.group(1).replace("\\", "/").strip().lstrip("/")
    return ""


def _is_read_of_solution_py(content: str, tool_args: dict[str, Any] | None) -> bool:
    rel = _read_path_from_content_or_args(content, tool_args)
    base = rel.split("/")[-1] if rel else ""
    if base == "solution.py" or rel.endswith("/solution.py"):
        return True
    if not content:
        return False
    first = content.splitlines()[0] if content else ""
    if "solution.py" in first and ("lines total" in first or "showing" in first):
        return True
    head = content[:2000]
    return "solution.py" in head and ("lines total" in head or "showing" in head)


def _is_read_of_code_file(
    content: str,
    tool_args: dict[str, Any] | None,
    *,
    code_extensions: tuple[str, ...],
) -> bool:
    rel = _read_path_from_content_or_args(content, tool_args)
    if rel:
        return _is_inherited_code_file_path(rel, code_extensions)
    return _is_read_of_solution_py(content, tool_args)


def _lines_matching_signal_patterns(
    lines: list[str],
    patterns: list[re.Pattern[str]],
) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        if ln in seen:
            continue
        if any(p.search(ln) for p in patterns):
            picked.append(ln)
            seen.add(ln)
    return picked


def _truncate_middle_on_inherit(s: str, max_total: int) -> str:
    if max_total <= 0 or len(s) <= max_total:
        return s
    half = max_total // 2
    omitted = len(s) - max_total
    return (
        f"{s[:half]}\n…[args truncated on inherit: {omitted} chars]…\n{s[len(s) - half :]}"
    )


def _bash_exit_header_success(first: str) -> bool:
    return first.startswith("[exit=0]") or first.startswith("[exit=0,")


def _compress_inherited_bash_output(
    obs: str,
    *,
    tail_lines: int,
    patterns: list[re.Pattern[str]],
    repl_mode: bool = False,
) -> str:
    """Compress a bash tool output for inherited memory.

    Default (``repl_mode=False``): errors are kept verbatim (< 32 kB) so the *same* node can
    debug them; only successful long outputs are tail-truncated with signal-line rescue.

    When ``repl_mode=True`` (cross-node REPL-style inheritance): both errors *and* successes are
    tail-truncated with signal-line rescue. Old verbose tracebacks from a previous node's approach
    are not useful to a child node that will use a different method — only the key error type and
    metric lines matter (signal-to-noise improvement).
    """
    if not obs or not isinstance(obs, str):
        return obs
    if not repl_mode:
        low = obs.lower()
        if "traceback" in low or "error:" in obs[:4000]:
            if len(obs) > 32000:
                return obs[:16000] + "\n...[truncated]...\n" + obs[-16000:]
            return obs
        if obs.lstrip().startswith("Error:"):
            if len(obs) > 32000:
                return obs[:16000] + "\n...[truncated]...\n" + obs[-16000:]
            return obs
    stripped = _strip_bash_stderr_section_from_block(obs)
    lines = stripped.splitlines()
    if not lines:
        return obs
    first = lines[0]
    if not first.startswith("[exit="):
        if len(stripped) > 12000:
            return stripped[:6000] + "\n...[truncated]...\n" + stripped[-6000:]
        return stripped
    if not repl_mode and not _bash_exit_header_success(first):
        if len(stripped) > 32000:
            return stripped[:16000] + "\n...[truncated]...\n" + stripped[-16000:]
        return stripped
    if "[bash:" in first and "showing last" in first:
        return stripped
    body = lines[1:]
    if tail_lines <= 0 or len(body) <= tail_lines:
        return stripped
    tail = body[-tail_lines:]
    dropped = body[:-tail_lines]
    extra = _lines_matching_signal_patterns(dropped, patterns)
    tail_set = set(tail)
    extra = [ln for ln in extra if ln not in tail_set]
    parts: list[str] = [first]
    if extra:
        parts.append("---")
        parts.append("[preserved signal lines from truncated bash stdout]")
        parts.extend(extra)
    parts.extend(tail)
    return "\n".join(parts)


def _compress_inherited_read_output(
    content: str,
    *,
    max_lines: int,
    patterns: list[re.Pattern[str]],
) -> str:
    if not content or max_lines <= 0:
        return content
    lines = content.splitlines()
    if not lines:
        return content
    first = lines[0]
    if not first.startswith("["):
        return content
    body_lines = lines[1:]
    if len(body_lines) <= max_lines:
        return content
    dropped = body_lines[max_lines:]
    extra = _lines_matching_signal_patterns(dropped, patterns)
    base = compress_read_output_for_memory(content, max_lines=max_lines)
    if not extra:
        return base
    tail_set = set(base.splitlines())
    extra = [ln for ln in extra if ln not in tail_set]
    if not extra:
        return base
    return base + "\n---\n[preserved signal lines from truncated read]\n" + "\n".join(extra)


def _write_tool_feedback_is_success(content: str) -> bool:
    if not content or not isinstance(content, str):
        return False
    if content.startswith("Error:") or content.lstrip().startswith("Error:"):
        return False
    first = content.split("\n", 1)[0]
    if first.startswith("File `") and (
        "updated." in first or "written successfully" in first
    ):
        return True
    if first.startswith("Written ") and "lines" in first:
        return True
    return False


def _rel_path_for_write_compact(tool_args: dict[str, Any] | None, content: str) -> str:
    if tool_args and tool_args.get("path"):
        return str(tool_args["path"]).replace("\\", "/").lstrip("/").strip() or "file"
    first = content.split("\n", 1)[0] if content else ""
    m = re.match(r"^Written\s+(\S+)", first.strip())
    if m:
        return m.group(1).strip().replace("\\", "/").lstrip("/") or "file"
    return "file"


def compress_inherited_tool_message_content(
    content: str,
    *,
    tool_name: str,
    tool_args: dict[str, Any] | None,
    read_max_lines: int,
    bash_tail_lines: int,
    omit_solution_py_reads: bool,
    write_compact: bool,
    edit_compact: bool,
    patterns: list[re.Pattern[str]],
    write_snapshot_inherit_max_lines: int = 0,
    write_snapshot_inherit_max_chars: int = 0,
    keep_auto_snapshot: bool = True,
    code_extensions: tuple[str, ...] = (".py",),
) -> str:
    """Shorten one tool message body when loading clone-inherited memory."""
    if not content or not isinstance(content, str):
        return content
    if tool_name == "read":
        if omit_solution_py_reads and _is_read_of_code_file(
            content,
            tool_args,
            code_extensions=code_extensions,
        ):
            rel = _read_path_from_content_or_args(content, tool_args) or "code file"
            return (
                f"[inherited] {rel} read omitted — file is unchanged on disk; "
                "use the `read` tool with `offset`/`limit` to fetch a fresh range only "
                "if you need to inspect specific lines."
            )
        return _compress_inherited_read_output(
            content, max_lines=read_max_lines, patterns=patterns,
        )
    if tool_name == "bash":
        return _compress_inherited_bash_output(
            content, tail_lines=bash_tail_lines, patterns=patterns,
        )
    if tool_name in ("write", "edit") and (
        (tool_name == "write" and write_compact)
        or (tool_name == "edit" and edit_compact)
    ):
        compacted = _compact_snapshot_tool_feedback_for_inherit(
            content,
            tool_name=tool_name,
            tool_args=tool_args,
            read_max_lines=read_max_lines,
            write_snapshot_inherit_max_lines=write_snapshot_inherit_max_lines,
            write_snapshot_inherit_max_chars=write_snapshot_inherit_max_chars,
            keep_auto_snapshot=keep_auto_snapshot,
        )
        if compacted is not None:
            return compacted
    if tool_name == "write" and write_compact and _write_tool_feedback_is_success(content):
        rel = _rel_path_for_write_compact(tool_args, content)
        ln, sha = _parse_write_success_metadata_from_output(content)
        base = write_success_feedback_for_memory(rel, lines=ln, sha256_short=sha)
        snap = extract_write_auto_snapshot_block(content)
        if snap:
            wlim, wch = _snapshot_inherit_limits(
                read_max_lines=read_max_lines,
                write_snapshot_inherit_max_lines=write_snapshot_inherit_max_lines,
                write_snapshot_inherit_max_chars=write_snapshot_inherit_max_chars,
            )
            return base + "\n\n" + truncate_inherited_snapshot_block(
                snap, max_lines=wlim, max_chars=wch,
            )
        return base
    if tool_name == "edit" and edit_compact:
        if content.startswith("Edited ") and "1 replacement OK." in content.split("\n", 1)[0]:
            return compress_edit_success_output_for_memory(content)
        return content
    return content


def _estimate_transcript_chars(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        c = m.content
        if isinstance(c, str):
            total += len(c)
        elif c is not None:
            total += len(str(c))
        for tc in getattr(m, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            if fn is not None:
                raw = getattr(fn, "arguments", "") or ""
                total += len(raw)
    return total


def _compress_stale_fork_user_message_for_inherit(content: str) -> str:
    """Replace old fork-control prompts after EDA with a short marker."""
    if not content or not isinstance(content, str):
        return content
    stripped = content.strip()
    if stripped.startswith("## Trajectory branch history"):
        return (
            "[inherited fork-control prompt omitted: prior C_BACKTRACK trajectory "
            "selection. The current child receives a fresh fork instruction.]"
        )
    if stripped.startswith("## Reference peer trajectories"):
        return (
            "[inherited peer trajectory prompt omitted: prior B_ENSEMBLE peer list. "
            "The current child receives a bounded trajectory rollup / peer artifact summary.]"
        )
    if stripped.startswith("Switch to **deep** mode:"):
        return (
            "[inherited fork-control prompt omitted: prior B_DEEP full-data "
            "instruction. The current child receives a fresh fork instruction.]"
        )
    if stripped.startswith("**Phase override (ensemble):**"):
        return (
            "[inherited fork-control prompt omitted: prior B_ENSEMBLE fusion "
            "instruction. The current child receives a fresh fork instruction.]"
        )
    if stripped.startswith("[Guard] Detected repeated single-`read` rounds."):
        return (
            "[inherited guard note omitted: prior repeated-read warning. Current "
            "workspace state and tool guards remain active.]"
        )
    if stripped.startswith("[LNR] Detected repeated single-`read` rounds."):
        return (
            "[inherited guard note omitted: prior repeated-read warning. Current "
            "workspace state and tool guards remain active.]"
        )
    if stripped.startswith("[Guard] No meaningful progress detected for multiple rounds"):
        return (
            "[inherited guard note omitted: prior no-progress warning. Current "
            "child should use the latest workspace state.]"
        )
    if stripped.startswith("[System] No meaningful progress detected for multiple rounds"):
        return (
            "[inherited guard note omitted: prior no-progress warning. Current "
            "child should use the latest workspace state.]"
        )
    return content


def _compress_protected_user_message_for_inherit(content: str) -> str:
    """Compress inherited non-EDA user prose while preserving EDA tool outputs."""
    if not content or not isinstance(content, str):
        return content
    stripped = content.strip()
    if stripped.startswith("[fresh-workspace]") or stripped.startswith("[Guard] Fresh workspace note:"):
        return (
            "[inherited setup hint omitted: the parent started from a fresh workspace, "
            "but this child inherits the current workspace and solution state.]"
        )
    if stripped.startswith("Design a **gold** strong single pipeline for this node."):
        task_start = stripped.find("# Nomad2018")
        if task_start < 0:
            task_start = stripped.find("## Task objective")
        if task_start < 0:
            task_start = stripped.find("# ")
        task_brief = stripped[task_start:].strip() if task_start >= 0 else ""
        if task_brief:
            return (
                "## Inherited task brief (compressed)\n\n"
                "The original draft single-solution policy prose is omitted here; "
                "the current branch first-user prompt carries the active fork policy.\n\n"
                + task_brief
            )
    return content


def _is_inherited_auto_snapshot_tool_message(msg: Message) -> bool:
    if (getattr(msg, "role", "") or "") != "tool":
        return False
    if str(getattr(msg, "name", "") or "") not in ("write", "edit"):
        return False
    content = msg.content
    return isinstance(content, str) and _AUTO_SNAPSHOT_PREFIX in content


def _maybe_compress_assistant_tool_calls_for_inherit(
    msg: Message,
    *,
    max_total_chars: int,
    exempt_write_call_index: int | None = None,
) -> Message:
    if max_total_chars <= 0 or not msg.tool_calls:
        return msg
    new_tcs: list[ToolCall] = []
    changed = False
    for j, tc in enumerate(msg.tool_calls):
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except (TypeError, json.JSONDecodeError):
            new_tcs.append(tc)
            continue
        if not isinstance(args, dict):
            new_tcs.append(tc)
            continue
        args_changed = False
        if name == "write" and j == exempt_write_call_index:
            new_tcs.append(tc)
            continue
        if name == "write" and isinstance(args.get("content"), str):
            new_c = _truncate_middle_on_inherit(args["content"], max_total_chars)
            if new_c != args["content"]:
                args["content"] = new_c
                args_changed = True
        elif name == "edit":
            for key in ("new_string", "old_string", "new_str", "old_str"):
                if isinstance(args.get(key), str):
                    new_s = _truncate_middle_on_inherit(args[key], max_total_chars)
                    if new_s != args[key]:
                        args[key] = new_s
                        args_changed = True
        if args_changed:
            changed = True
            try:
                raw = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                new_tcs.append(tc)
                continue
            new_tcs.append(
                tc.model_copy(
                    update={"function": Function(name=name, arguments=raw)},
                ),
            )
        else:
            new_tcs.append(tc)
    if not changed:
        return msg
    return msg.model_copy(update={"tool_calls": new_tcs})


_READ_ONLY_BASH_CMDS: frozenset[str] = frozenset(
    # ``sed`` is common in read-only ``cat … | sed -n 'a,bp'`` pipelines; ``-i`` is rejected below.
    {"cat", "head", "tail", "less", "more", "grep", "egrep", "rg", "sed"},
)

_BASH_WRITE_REDIRECT_RE = re.compile(r"(?:^|\s)(?:\d?>|&>|>>)")


def _bash_leading_token(segment: str) -> str:
    """First command token in a segment, after stripping ``VAR=value`` prefixes."""
    parts = segment.strip().split()
    i = 0
    while i < len(parts):
        tok = parts[i]
        if "=" in tok:
            key, _, _ = tok.partition("=")
            if key and key.replace("_", "").isupper():
                i += 1
                continue
        return tok
    return ""


def _bash_first_command_words(cmd: str) -> list[str]:
    """Return first shell segment words after stripping leading env assignments."""
    s = (cmd or "").strip()
    if not s:
        return []
    first_segment = re.split(r"\s*(?:\|\||&&|\||;|&)\s*", s, maxsplit=1)[0].strip()
    if not first_segment:
        return []
    try:
        words = shlex.split(first_segment)
    except ValueError:
        words = first_segment.split()
    out = list(words)
    while out:
        tok = out[0]
        if tok == "env":
            out.pop(0)
            continue
        if "=" in tok:
            key, _, _ = tok.partition("=")
            if key and key.replace("_", "").isupper():
                out.pop(0)
                continue
        break
    return out


def _bash_command_looks_like_test(cmd: str) -> bool:
    words = _bash_first_command_words(cmd)
    if not words:
        return False
    first = words[0]
    if first == "pytest":
        return True
    if first in {"python", "python3"} and len(words) >= 3:
        return words[1] == "-m" and words[2] == "pytest"
    if first == "uv" and len(words) >= 3 and words[1] == "run":
        rest = words[2:]
        if not rest:
            return False
        if rest[0] == "pytest":
            return True
        return len(rest) >= 3 and rest[0] in {"python", "python3"} and rest[1] == "-m" and rest[2] == "pytest"
    return False


def _bash_command_looks_like_install(cmd: str) -> bool:
    words = _bash_first_command_words(cmd)
    if len(words) < 2:
        return False
    first = words[0]
    if first in {"pip", "pip3"}:
        return words[1] == "install"
    if first == "uv":
        return words[1] in {"sync", "lock"} or (
            len(words) >= 3 and words[1] == "pip" and words[2] == "install"
        )
    if first == "npm":
        return words[1] == "install"
    if first == "yarn":
        return words[1] in {"install", "add"}
    if first == "apt":
        return words[1] == "install"
    return False


def _bash_is_pure_read_only(cmd: str) -> bool:
    """True if every ``||``, ``&&``, ``|``, ``;``, ``&`` segment starts with a whitelisted command."""
    if not cmd.strip():
        return False
    segments = re.split(r"\s*(?:\|\||&&|\||;|&)\s*", cmd.strip())
    for seg in segments:
        if not seg.strip():
            continue
        seg_for_redirect = (
            seg.replace("2>&1", "")
            .replace("1>&2", "")
            .replace("2> /dev/null", "")
            .replace("2>/dev/null", "")
        )
        if "<<" in seg_for_redirect or _BASH_WRITE_REDIRECT_RE.search(seg_for_redirect):
            return False
        first = _bash_leading_token(seg)
        if not first or first not in _READ_ONLY_BASH_CMDS:
            return False
        if first == "sed" and re.search(r"(?<![A-Za-z0-9_])-i", seg):
            # in-place / backup suffix forms are not read-only
            return False
    return True


def _bash_success_tail_lines_for_command(
    cmd: str,
    *,
    fallback: int,
    solution: int,
    test: int,
    readonly: int,
    install: int,
) -> int:
    if looks_like_bare_solution_run(cmd):
        return min(int(fallback), int(solution))
    if _bash_command_looks_like_test(cmd):
        return int(test)
    if _bash_is_pure_read_only(cmd):
        return int(readonly)
    if _bash_command_looks_like_install(cmd):
        return int(install)
    return int(fallback)


# Commands often used to dump whole-file / line-range source in bash (vs. grep/rg search).
_BASH_SOURCE_DUMP_CMDS: frozenset[str] = frozenset({"cat", "head", "tail", "sed"})

_PYPATH_IN_BASH_CMD = re.compile(
    r"(?:^|[\s/])([A-Za-z0-9_.-]+\.py)\b",
)


def bash_command_dumps_python_source(cmd: str) -> bool:
    """True if *cmd* is read-only and uses cat/head/tail/sed to show ``*.py`` / known entry files.

    Used to discourage repeated raw source dumps; targeted bash snippets remain acceptable in
    bash-first mode, and ``read`` is only a fallback for exact line-numbered anchors.
    """
    if not _bash_is_pure_read_only(cmd):
        return False
    # Any ``*.py`` path anywhere in the command (pipelines may only name the file in the first segment).
    if not _PYPATH_IN_BASH_CMD.search(cmd):
        return False
    segments = re.split(r"\s*(?:\|\||&&|\||;|&)\s*", cmd.strip())
    for seg in segments:
        s = seg.strip()
        if not s:
            continue
        first = _bash_leading_token(s)
        if first in _BASH_SOURCE_DUMP_CMDS:
            return True
    return False


_BASH_READ_PY_SOURCE_COACHING = (
    "\n\n[Guard] Bash source dump detected. Keep source inspection compact: use one "
    "bounded snippet/search command, or targeted `read` tool only when exact line-numbered "
    "anchors are needed. Do not spend rounds dumping code."
)


def _tool_call_signature(tool_name: str, args: dict[str, Any]) -> tuple[str, str] | None:
    """Stable signature for repeat-detection (read-only ``bash`` / ``read`` only)."""
    if tool_name == "bash":
        cmd = " ".join(str(args.get("command") or "").split())
        if not cmd or not _bash_is_pure_read_only(cmd):
            return None
        return ("bash", cmd)
    if tool_name == "read":
        path = str(args.get("path") or "").replace("\\", "/").strip().lstrip("/")
        if not path:
            return None
        offset = args.get("offset")
        limit = args.get("limit")
        return ("read", f"{path}|{offset}|{limit}")
    return None


def _clear_memory_and_long_term_log(memory: Memory) -> None:
    """Clear chat storage and truncate ``long_term.jsonl`` before bulk re-import.

    ``JsonKeyValueStorage.clear()`` / ``BaseChatHistoryMemory.clear()`` clears
    ``short_term.json`` but leaves ``long_term.jsonl`` append-only; subsequent
    :meth:`Memory.add_message` would duplicate every record in the JSONL.
    """
    memory.chat_history_memory.clear()
    lt = getattr(memory, "long_term_log", None)
    if lt:
        p = Path(str(lt))
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")
        except OSError:
            logger.debug("truncate long_term.jsonl failed: %s", lt, exc_info=True)
    else:
        storage = getattr(memory.chat_history_memory, "storage", None)
        jp = getattr(storage, "json_path", None) if storage is not None else None
        if jp is not None:
            alt = Path(jp).parent / "long_term.jsonl"
            try:
                if alt.is_file():
                    alt.write_text("", encoding="utf-8")
            except OSError:
                logger.debug("truncate fallback long_term failed: %s", alt, exc_info=True)


def _tool_call_targets_result_md(tc: Any) -> bool:
    name, args, _tcid, _tctype = _tool_call_name_args_for_projection(tc)
    if name not in {"write", "edit"}:
        return False
    raw_path = str(args.get("path") or "").replace("\\", "/").strip()
    raw_path = raw_path.lstrip("./")
    return raw_path == "result.md" or raw_path.endswith("/result.md")


def _assistant_writes_result_md(msg: Message) -> bool:
    if (getattr(msg, "role", "") or "") != "assistant":
        return False
    return any(
        _tool_call_targets_result_md(tc)
        for tc in (getattr(msg, "tool_calls", None) or [])
    )


def _is_result_md_recovery_user_message(msg: Message) -> bool:
    if (getattr(msg, "role", "") or "") != "user":
        return False
    content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
    lower = content.lower()
    if "result.md" not in lower:
        return False
    if "write" not in lower and "must run a full training pass" not in lower:
        return False
    return (
        "result.md is missing" in lower
        or "result.md does not exist yet" in lower
        or "write result.md" in lower
        or "write it now" in lower
    )


def _is_science_agent_stop_summary(msg: Message) -> bool:
    if (getattr(msg, "role", "") or "") != "assistant":
        return False
    content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
    return content.strip().startswith("[ScienceAgent] Stopped after")


def _is_result_md_interstitial_guard_user_message(msg: Message) -> bool:
    """Return true for guard notes injected while the result.md writer is pending."""
    if (getattr(msg, "role", "") or "") != "user":
        return False
    content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
    stripped = content.strip()
    return (
        stripped.startswith("[Guard] No meaningful progress detected for multiple rounds")
        or stripped.startswith("[System] No meaningful progress detected for multiple rounds")
        or stripped.startswith("[Guard] No successful `python3 solution.py`")
        or stripped.startswith("[System] No successful `python3 solution.py`")
        or stripped.startswith("[Guard] Detected repeated single-`read` rounds.")
        or stripped.startswith("[LNR] Detected repeated single-`read` rounds.")
    )


def _is_text_only_assistant(msg: Message) -> bool:
    return (
        (getattr(msg, "role", "") or "") == "assistant"
        and not (getattr(msg, "tool_calls", None) or [])
    )


def _skip_assistant_tool_round(messages: list[Message], assistant_idx: int) -> int:
    """Return the index after one assistant tool-call turn.

    Strict LLM APIs require every assistant ``tool_calls`` message to be followed by
    matching ``role=tool`` responses. When dropping a report write, drop the
    assistant message and its matching tool responses together.
    """
    msg = messages[assistant_idx]
    tcs = list(getattr(msg, "tool_calls", None) or [])
    expected = [_tool_call_id_from_raw(tc) for tc in tcs]
    expected_set = {x for x in expected if x}
    j = assistant_idx + 1
    seen = 0
    while j < len(messages) and getattr(messages[j], "role", None) == "tool":
        tcid = str(getattr(messages[j], "tool_call_id", "") or "")
        if expected_set and tcid and tcid not in expected_set:
            break
        seen += 1
        j += 1
        if tcs and seen >= len(tcs):
            break
    return j


def _skip_result_md_recovery_segment(messages: list[Message], start_idx: int) -> int:
    """Return the index after a result.md recovery segment.

    The recovery prompt is bookkeeping for closing the current node. It should
    not be inherited by the next node, and neither should the immediate tool
    attempts made only to satisfy that prompt. Guard notes can be injected as
    user messages while the report writer is pending; keep treating those as
    part of the bookkeeping segment so the child inherits the preceding
    successful ``python3 solution.py`` tool result as the natural transcript tail.
    """
    j = start_idx + 1
    while j < len(messages):
        if _assistant_writes_result_md(messages[j]):
            j = _skip_assistant_tool_round(messages, j)
            while j < len(messages) and (
                _is_science_agent_stop_summary(messages[j])
                or _is_text_only_assistant(messages[j])
            ):
                j += 1
            return j
        if (
            getattr(messages[j], "role", None) == "user"
            and not _is_result_md_interstitial_guard_user_message(messages[j])
        ):
            break
        j += 1
    while j < len(messages) and _is_science_agent_stop_summary(messages[j]):
        j += 1
    return j


def drop_result_md_summary_rounds_for_inherit(memory: Memory) -> dict[str, Any]:
    """Remove result-report summary turns from clone-inherited chat memory.

    ``result.md`` is a node artifact for tracking/journal context, not part of
    the next node's natural tool trajectory. This removes recovery prompts and
    assistant ``write/edit result.md`` tool turns as whole assistant+tool groups,
    preserving valid tool-call pairing in the remaining transcript. The preceding
    successful ``python3 solution.py`` bash turn is intentionally kept so clone
    children inherit the parent's final score and run-control evidence naturally.
    """
    all_records = memory.chat_history_memory.retrieve(window_size=None)
    if not all_records:
        return {"changed": False, "messages_dropped": 0, "rounds_dropped": 0}
    messages = [cr.memory_record.message for cr in all_records]
    new_messages: list[Message] = []
    dropped = 0
    rounds = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if _is_result_md_recovery_user_message(msg):
            end = _skip_result_md_recovery_segment(messages, i)
            dropped += end - i
            rounds += 1
            i = end
            continue
        if _assistant_writes_result_md(msg):
            end = _skip_assistant_tool_round(messages, i)
            while end < len(messages) and _is_science_agent_stop_summary(messages[end]):
                end += 1
            dropped += end - i
            rounds += 1
            i = end
            continue
        new_messages.append(msg)
        i += 1

    if not dropped:
        return {"changed": False, "messages_dropped": 0, "rounds_dropped": 0}
    _clear_memory_and_long_term_log(memory)
    for msg in new_messages:
        memory.add_message(msg)
    return {
        "changed": True,
        "messages_dropped": dropped,
        "rounds_dropped": rounds,
    }


def _supersede_stale_compressed_records(
    *,
    memory: Memory,
    released_sigs: set[tuple[str, str]],
    seen_sigs: dict[tuple[str, str], int],
) -> None:
    """Replace first-call compressed tool rows with a one-line note; drop tracking state."""
    if not released_sigs:
        return
    storage = getattr(getattr(memory, "chat_history_memory", None), "storage", None)
    if storage is None:
        released_sigs.clear()
        return
    try:
        records = storage.load()
    except Exception:
        logger.debug("supersede: storage.load failed", exc_info=True)
        released_sigs.clear()
        return
    if not isinstance(records, list) or not records:
        released_sigs.clear()
        return

    changed = False
    for sig in list(released_sigs):
        idx = seen_sigs.get(sig)
        if idx is None or idx < 0 or idx >= len(records):
            seen_sigs.pop(sig, None)
            continue
        rec = records[idx]
        if not isinstance(rec, dict):
            seen_sigs.pop(sig, None)
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            seen_sigs.pop(sig, None)
            continue
        msg["content"] = (
            f"[superseded by later {sig[0]} call with identical args — compressed output collapsed]"
        )
        changed = True
        seen_sigs.pop(sig, None)

    released_sigs.clear()

    if not changed:
        return

    chm = memory.chat_history_memory
    chm.clear()
    for rec in records:
        chm.add_record(MemoryRecord.from_dict(rec))


def apply_clone_inherit_compression(
    memory: Memory,
    cfg: Any,
    *,
    repl_mode: bool = False,
    preserve_initial_eda: bool = False,
) -> dict[str, Any]:
    """Rewrite inherited chat messages in-place for token savings (clone-continue only).

    When *repl_mode* is True, the parent's tool results are partially preserved
    for prefix-safe continuation: ``read``/``write``/``edit`` outputs pass
    through verbatim, ``bash`` outputs are tail-truncated with signal-line rescue (improving
    cross-node signal-to-noise while keeping key error types and metric lines). The ``assistant``
    ``tool_call`` argument middle-truncation is still applied (gated by
    ``clone_inherit_tool_args_head_tail_chars``) so that historic 30k+ ``write.content``
    arguments do not balloon the next session's context.
    Historical assistant write arguments are compressed; canonical write/edit snapshots
    are the source of truth for inherited code state.
    When *preserve_initial_eda* is True, this legacy clone helper keeps the
    transcript prefix before the first code-file ``write`` verbatim. Active LNR
    EDA protection is stage-index based: S01 completion records a fixed memory
    boundary via ``MemoryContextManager`` and does not infer from file names.
    """
    if not bool(getattr(cfg, "clone_inherit_compress", True)):
        return {
            "changed": False,
            "user_msgs": 0,
            "tool_msgs": 0,
            "assistant_tc_args": 0,
            "snapshots_dropped": 0,
            "chars_before": 0,
            "chars_after": 0,
        }
    raw_patterns = getattr(cfg, "clone_inherit_signal_patterns", None)
    if raw_patterns is not None and not isinstance(raw_patterns, list):
        raw_patterns = None
    str_patterns: list[str] | None = (
        [str(x) for x in raw_patterns] if raw_patterns else None
    )
    patterns = _compile_clone_inherit_signal_patterns(str_patterns)
    code_extensions = _norm_code_extensions_for_inherit(
        getattr(cfg, "write_auto_snapshot_code_extensions", None),
    )
    read_max = int(getattr(cfg, "clone_inherit_read_max_lines", 80) or 80)
    bash_tail = int(getattr(cfg, "clone_inherit_bash_tail_lines", 20) or 20)
    omit_sol = bool(getattr(cfg, "clone_inherit_omit_solution_py_reads", True))
    write_c = bool(getattr(cfg, "clone_inherit_write_compact", True))
    edit_c = bool(getattr(cfg, "clone_inherit_edit_compact", True))
    tc_budget = int(getattr(cfg, "clone_inherit_tool_args_head_tail_chars", 2000) or 0)
    wsnap_l = int(getattr(cfg, "clone_inherit_write_snapshot_max_lines", 0) or 0)
    wsnap_c = int(getattr(cfg, "clone_inherit_write_snapshot_max_chars", 0) or 0)
    exempt_last_state = bool(
        getattr(cfg, "clone_inherit_last_state_write_exempt", True),
    ) and repl_mode and not preserve_initial_eda

    all_records = memory.chat_history_memory.retrieve(window_size=None)
    if not all_records:
        return {
            "changed": False,
            "user_msgs": 0,
            "tool_msgs": 0,
            "assistant_tc_args": 0,
            "snapshots_dropped": 0,
            "chars_before": 0,
            "chars_after": 0,
        }
    messages = [cr.memory_record.message for cr in all_records]
    before = _estimate_transcript_chars(messages)

    protected_prefix_end = 0
    if preserve_initial_eda:
        protected_prefix_end = len(messages)
        for i, m in enumerate(messages):
            if (getattr(m, "role", "") or "") != "assistant" or not m.tool_calls:
                continue
            found_code_write = False
            for tc in m.tool_calls:
                name = str(getattr(getattr(tc, "function", None), "name", "") or "")
                if name != "write":
                    continue
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(a, dict):
                    continue
                if _is_inherited_code_file_path(str(a.get("path") or ""), code_extensions):
                    found_code_write = True
                    break
            if found_code_write:
                protected_prefix_end = i
                break

    last_state_write_call: tuple[int, int] | None = None
    if exempt_last_state:
        for ai, am in enumerate(messages):
            if (getattr(am, "role", "") or "") != "assistant" or not am.tool_calls:
                continue
            for tj, tc in enumerate(am.tool_calls):
                name = str(getattr(getattr(tc, "function", None), "name", "") or "")
                if name != "write":
                    continue
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(a, dict) and _is_inherited_code_file_path(str(a.get("path") or ""), code_extensions):
                    last_state_write_call = (ai, tj)

    latest_snapshot_tool_idx_by_path: dict[str, int] = {}
    if preserve_initial_eda:
        for i, m in enumerate(messages):
            if i < protected_prefix_end:
                continue
            if _is_inherited_auto_snapshot_tool_message(m):
                args_d = _tool_args_dict_for_tool_message(messages, i)
                content = m.content if isinstance(m.content, str) else str(m.content or "")
                rel = str((args_d or {}).get("path") or "").replace("\\", "/").lstrip("/")
                if not rel:
                    rel = _snapshot_path_from_auto_snapshot_content(content)
                if rel and _is_inherited_code_file_path(rel, code_extensions):
                    latest_snapshot_tool_idx_by_path[rel] = i

    new_messages: list[Message] = []
    user_changed = 0
    tool_changed = 0
    asst_changed = 0
    snapshots_dropped = 0
    for i, m in enumerate(messages):
        mm = m
        role = getattr(m, "role", "") or ""
        if (
            preserve_initial_eda
            and i < protected_prefix_end
            and role == "user"
            and isinstance(m.content, str)
        ):
            new_c = _compress_protected_user_message_for_inherit(m.content)
            if new_c != m.content:
                user_changed += 1
                mm = m.model_copy(update={"content": new_c})
            new_messages.append(mm)
            continue
        if preserve_initial_eda and i < protected_prefix_end:
            new_messages.append(mm)
            continue
        if role == "user" and preserve_initial_eda and isinstance(m.content, str):
            new_c = _compress_stale_fork_user_message_for_inherit(m.content)
            if new_c != m.content:
                user_changed += 1
                mm = m.model_copy(update={"content": new_c})
        elif role == "assistant" and m.tool_calls and tc_budget > 0:
            exempt_idx = (
                last_state_write_call[1]
                if last_state_write_call is not None and last_state_write_call[0] == i
                else None
            )
            mm = _maybe_compress_assistant_tool_calls_for_inherit(
                m,
                max_total_chars=tc_budget,
                exempt_write_call_index=exempt_idx,
            )
            if mm is not m:
                asst_changed += 1
        elif role == "tool" and not repl_mode:
            tcid = str(getattr(m, "tool_call_id", "") or "")
            if tcid.startswith("inherited_fullrun_"):
                new_messages.append(mm)
                continue
            name = str(getattr(m, "name", "") or "")
            if not isinstance(m.content, str):
                new_messages.append(mm)
                continue
            args_d = _tool_args_dict_for_tool_message(messages, i)
            rel = str((args_d or {}).get("path") or "").replace("\\", "/").lstrip("/")
            if not rel:
                rel = _snapshot_path_from_auto_snapshot_content(m.content)
            keep_auto_snapshot = (
                not preserve_initial_eda
                or not rel
                or latest_snapshot_tool_idx_by_path.get(rel) == i
            )
            new_c = compress_inherited_tool_message_content(
                m.content,
                tool_name=name,
                tool_args=args_d,
                read_max_lines=read_max,
                bash_tail_lines=bash_tail,
                omit_solution_py_reads=omit_sol,
                write_compact=write_c,
                edit_compact=edit_c,
                patterns=patterns,
                write_snapshot_inherit_max_lines=wsnap_l,
                write_snapshot_inherit_max_chars=wsnap_c,
                keep_auto_snapshot=keep_auto_snapshot,
                code_extensions=code_extensions,
            )
            if new_c != m.content:
                tool_changed += 1
                if (
                    not keep_auto_snapshot
                    and _is_inherited_auto_snapshot_tool_message(m)
                ):
                    snapshots_dropped += 1
                mm = m.model_copy(update={"content": new_c})
        elif role == "tool" and repl_mode:
            # REPL-style inherit: read/write/edit results pass through verbatim (preserves file
            # content so LLM does not re-read). bash outputs (both success and error) are
            # tail-truncated with signal-line rescue to improve signal-to-noise: verbose old
            # tracebacks from a previous node's approach are trimmed while key lines (Error type,
            # OOM, score) survive and remain visible to the child node.
            tcid = str(getattr(m, "tool_call_id", "") or "")
            if tcid.startswith("inherited_fullrun_"):
                new_messages.append(mm)
                continue
            name = str(getattr(m, "name", "") or "")
            if name == "bash" and isinstance(m.content, str):
                new_c = _compress_inherited_bash_output(
                    m.content, tail_lines=bash_tail, patterns=patterns, repl_mode=True,
                )
                if new_c != m.content:
                    tool_changed += 1
                    mm = m.model_copy(update={"content": new_c})
        new_messages.append(mm)
    after = _estimate_transcript_chars(new_messages)
    changed = tool_changed > 0 or asst_changed > 0 or user_changed > 0
    if changed:
        _clear_memory_and_long_term_log(memory)
        for m in new_messages:
            memory.add_message(m)
    return {
        "changed": bool(changed),
        "user_msgs": user_changed,
        "tool_msgs": tool_changed,
        "assistant_tc_args": asst_changed,
        "snapshots_dropped": snapshots_dropped,
        "chars_before": before,
        "chars_after": after,
    }


def _assistant_tool_call_writes_solution(msg: Message) -> bool:
    if (getattr(msg, "role", "") or "") != "assistant":
        return False
    for tc in getattr(msg, "tool_calls", None) or []:
        name = ""
        raw_args = ""
        if isinstance(tc, dict):
            tc_fn = tc.get("function") or {}
            if isinstance(tc_fn, dict):
                name = str(tc_fn.get("name") or "")
                raw_args = str(tc_fn.get("arguments") or "")
        else:
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", "") or "") if fn is not None else ""
            raw_args = str(getattr(fn, "arguments", "") or "") if fn is not None else ""
        if name not in {"write", "edit"}:
            continue
        try:
            args = json.loads(raw_args or "{}")
        except (TypeError, json.JSONDecodeError):
            args = {}
        if isinstance(args, dict) and _is_inherited_code_file_path(str(args.get("path") or "")):
            return True
    return False


def _first_solution_write_assistant_index(messages: list[Message]) -> int:
    for idx, msg in enumerate(messages):
        if _assistant_tool_call_writes_solution(msg):
            return idx
    return len(messages)


def _assistant_index_for_tool_index(messages: list[Message], tool_idx: int) -> int | None:
    tcid = str(getattr(messages[tool_idx], "tool_call_id", "") or "")
    j = tool_idx - 1
    while j >= 0 and getattr(messages[j], "role", None) == "tool":
        j -= 1
    if j < 0 or getattr(messages[j], "role", None) != "assistant":
        return None
    if not tcid:
        return j
    for tc in getattr(messages[j], "tool_calls", None) or []:
        if _tool_call_id_from_raw(tc) == tcid:
            return j
    return j


def _assistant_round_indices(messages: list[Message], assistant_idx: int | None) -> set[int]:
    if assistant_idx is None or assistant_idx < 0 or assistant_idx >= len(messages):
        return set()
    out = {assistant_idx}
    expected = _tool_call_id_set_for_message(messages[assistant_idx])
    j = assistant_idx + 1
    while j < len(messages) and getattr(messages[j], "role", None) == "tool":
        tcid = str(getattr(messages[j], "tool_call_id", "") or "")
        if not expected or not tcid or tcid in expected:
            out.add(j)
        j += 1
    return out


def _tool_call_id_set_for_message(msg: Message) -> set[str]:
    ids: set[str] = set()
    for tc in getattr(msg, "tool_calls", None) or []:
        tcid = _tool_call_id_from_raw(tc)
        if tcid:
            ids.add(tcid)
    return ids


def _last_solution_snapshot_tool_index(messages: list[Message]) -> int | None:
    by_path = _last_code_snapshot_tool_indices_by_path(messages)
    return max(by_path.values(), default=None)


def _last_code_snapshot_tool_indices_by_path(messages: list[Message]) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if (getattr(msg, "role", "") or "") != "tool":
            continue
        name = str(getattr(msg, "name", "") or "")
        if name not in {"write", "edit"}:
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        args = _tool_args_dict_for_tool_message(messages, idx) or {}
        rel = str(args.get("path") or "").replace("\\", "/").lstrip("/")
        if not rel:
            rel = _snapshot_path_from_auto_snapshot_content(content)
        if not _is_inherited_code_file_path(rel):
            continue
        if _AUTO_SNAPSHOT_PREFIX in content or _write_tool_feedback_is_success(content) or content.startswith("Edited "):
            out.setdefault(rel, idx)
    return out


def _last_signal_bash_tool_index(messages: list[Message]) -> int | None:
    fallback: int | None = None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if (getattr(msg, "role", "") or "") != "tool":
            continue
        if str(getattr(msg, "name", "") or "") != "bash":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if not content:
            continue
        if fallback is None:
            fallback = idx
        if "Final Validation Score" in content or "METRIC:" in content:
            return idx
        first = content.splitlines()[0] if content else ""
        if first.startswith("[exit=0"):
            return idx
    return fallback


def _message_chars(msg: Message) -> int:
    return _estimate_transcript_chars([msg])


def _selected_chars(messages: list[Message], indices: set[int]) -> int:
    return sum(_message_chars(messages[i]) for i in sorted(indices))


def _truncate_for_capsule(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    keep = max(0, max_chars - 32)
    head = max(0, keep // 2)
    tail = max(0, keep - head)
    return s[:head].rstrip() + "\n...[truncated]...\n" + s[-tail:].lstrip()


def build_shallow_continuity_capsule(
    memory: Memory,
    *,
    max_chars: int = 12000,
) -> str:
    """Bounded B_DEEP handoff from the most recent shallow memory.

    The capsule is deterministic: it preserves recent intent, current code state,
    and last validation signal without copying the whole branch transcript.
    """
    messages = _all_messages(memory)
    if not messages or max_chars <= 0:
        return ""
    first_write = _first_solution_write_assistant_index(messages)
    latest_snapshot_indices = _last_code_snapshot_tool_indices_by_path(messages)
    latest_bash_idx = _last_signal_bash_tool_index(messages)

    parts: list[str] = ["## Recent shallow continuity capsule"]
    if first_write < len(messages):
        parts.append("- scope: latest shallow state after initial EDA and solution creation")
    else:
        parts.append("- scope: recent shallow state; no solution write found in inherited memory")

    last_intent = ""
    for msg in reversed(messages[first_write:]):
        if (getattr(msg, "role", "") or "") != "assistant":
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if content.strip():
            last_intent = _truncate_for_capsule(content, 1000)
            break
    if last_intent:
        parts.append("\n### Recent assistant intent\n" + last_intent)

    if latest_snapshot_indices:
        snap_parts: list[str] = []
        for rel, idx in sorted(latest_snapshot_indices.items()):
            msg = messages[idx]
            content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            snap = extract_write_auto_snapshot_block(content) or content
            snap_parts.append(f"#### {rel}\n" + _truncate_for_capsule(snap, 1200))
        parts.append("\n### Current code state\n" + _truncate_for_capsule("\n\n".join(snap_parts), 2600))

    if latest_bash_idx is not None:
        msg = messages[latest_bash_idx]
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        parts.append("\n### Latest shallow run signal\n" + _truncate_for_capsule(content, 1800))

    last_gate = ""
    for msg in reversed(messages[first_write:]):
        if (getattr(msg, "role", "") or "") != "user":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if "NODE-GATE" in content or "Phase budget" in content or "result.md" in content:
            last_gate = _truncate_for_capsule(content, 900)
            break
    if last_gate:
        parts.append("\n### Recent gate / budget signal\n" + last_gate)
    out = "\n".join(parts).strip()
    return _truncate_for_capsule(out, max_chars)


def apply_clone_non_a_memory_budget(
    memory: Memory,
    cfg: Any,
    *,
    fork_class: str,
) -> dict[str, Any]:
    """Hard-budget B/C inherited memory while preserving draft EDA and current state.

    This runs after clone inheritance compression. It is intentionally not used for
    A_TELEPORT, whose cache behavior depends on prefix-safe transcript continuity.
    """
    budget = int(getattr(cfg, "clone_non_a_inherit_budget_chars", 0) or 0)
    if budget <= 0:
        return {
            "changed": False,
            "chars_before": 0,
            "chars_after": 0,
            "budget": budget,
            "messages_before": 0,
            "messages_after": 0,
        }
    messages = _all_messages(memory)
    if not messages:
        return {
            "changed": False,
            "chars_before": 0,
            "chars_after": 0,
            "budget": budget,
            "messages_before": 0,
            "messages_after": 0,
        }
    before = _estimate_transcript_chars(messages)
    if before <= budget:
        return {
            "changed": False,
            "chars_before": before,
            "chars_after": before,
            "budget": budget,
            "messages_before": len(messages),
            "messages_after": len(messages),
        }

    fork = str(fork_class or "").upper()
    first_write = _first_solution_write_assistant_index(messages)
    selected: set[int] = set(range(0, first_write))

    latest_code_indices = _last_code_snapshot_tool_indices_by_path(messages)
    latest_bash_idx = _last_signal_bash_tool_index(messages)

    essential_groups: list[set[int]] = []
    for latest_solution_idx in sorted(set(latest_code_indices.values())):
        essential_groups.append(
            _assistant_round_indices(messages, _assistant_index_for_tool_index(messages, latest_solution_idx)),
        )
    if latest_bash_idx is not None:
        essential_groups.append(
            _assistant_round_indices(messages, _assistant_index_for_tool_index(messages, latest_bash_idx)),
        )

    for group in essential_groups:
        selected.update(group)

    optional_groups: list[set[int]] = []
    if fork == "B_DEEP":
        # Preserve recent shallow continuity as rounds, newest first. This is a
        # bounded tail, not the whole branch transcript.
        idx = len(messages) - 1
        while idx >= first_write:
            role = getattr(messages[idx], "role", "") or ""
            if role == "tool":
                aidx = _assistant_index_for_tool_index(messages, idx)
                group = _assistant_round_indices(messages, aidx)
                if group:
                    optional_groups.append(group)
                    idx = min(group) - 1
                    continue
            optional_groups.append({idx})
            idx -= 1
            if len(optional_groups) >= 12:
                break
    else:
        kept_users = 0
        for idx in range(len(messages) - 1, first_write - 1, -1):
            msg = messages[idx]
            if (getattr(msg, "role", "") or "") == "user":
                optional_groups.append({idx})
                kept_users += 1
                if kept_users >= 2:
                    break

    for group in optional_groups:
        if not group or group.issubset(selected):
            continue
        trial = selected | group
        if _selected_chars(messages, trial) <= budget:
            selected = trial

    new_messages = [messages[i] for i in sorted(selected)]
    after = _estimate_transcript_chars(new_messages)
    changed = len(new_messages) != len(messages) or after != before
    if changed:
        _clear_memory_and_long_term_log(memory)
        for msg in new_messages:
            memory.add_message(msg)
    return {
        "changed": bool(changed),
        "chars_before": before,
        "chars_after": after,
        "budget": budget,
        "messages_before": len(messages),
        "messages_after": len(new_messages),
        "budget_overflow": after > budget,
        "protected_prefix_messages": first_write,
    }


def apply_clone_minimal_slice_to_memory(memory: Memory) -> bool:
    """Drop middle transcript; keep leading users + suffix from last assistant."""
    msgs = _all_messages(memory)
    sliced = slice_clone_minimal_inherited_messages(msgs)
    if sliced is msgs:
        return False
    if len(sliced) == len(msgs) and all(a is b for a, b in zip(sliced, msgs, strict=False)):
        return False
    _clear_memory_and_long_term_log(memory)
    for m in sliced:
        memory.add_message(m)
    return True


_BASH_TRAINING_SIGNAL_RE = re.compile(
    r"(score|metric|rmse|rmsle|mae|mse|accuracy|acc\b|auc|loss|epoch|fold|best|"
    r"valid|validation|final|saved|written|submission|artifact|model|csv|pkl)",
    re.IGNORECASE,
)
_BASH_PYTEST_SUMMARY_RE = re.compile(
    r"(=+\s*.*(?:passed|failed|errors?|warnings?|skipped|xfailed|xpassed).*=+|"
    r"\b\d+\s+(?:passed|failed|errors?|warnings?|skipped)\b)",
    re.IGNORECASE,
)
_BASH_PYTEST_FAILURE_RE = re.compile(
    r"(^FAILED\s+|^ERROR\s+|FAILURES|ERRORS|short test summary info|"
    r"\bFAILED\b|\bERROR\b|assert\s|^E\s+|^>\s+|Traceback|File \")",
)
_BASH_INSTALL_SIGNAL_RE = re.compile(
    r"(successfully installed|installed|resolved|prepared|done|success|error|failed|failure)",
    re.IGNORECASE,
)


def _dedupe_recent_nonempty_lines(lines: list[str], *, limit: int) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    for line in reversed(lines):
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        picked.append(line)
        if len(picked) >= limit:
            break
    picked.reverse()
    return picked


def _dedupe_preserve_order(lines: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= limit:
            break
    return out


_PY_TRACEBACK_START = "Traceback (most recent call last):"
_PY_EXCEPTION_LINE_RE = re.compile(
    r"^\s*(?:[A-Za-z_][\w.]*\.)*[A-Za-z_]\w*(?:Error|Exception|Warning):\s+.+"
)


def _extract_recent_python_error_block(lines: list[str], *, max_lines: int = 12) -> list[str]:
    """Return the most recent Python traceback/exception block from bash output."""
    if not lines:
        return []
    start: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        if _PY_TRACEBACK_START in lines[i]:
            start = i
            break
    if start is not None:
        end = min(len(lines), start + max_lines)
        return lines[start:end]

    for i in range(len(lines) - 1, -1, -1):
        if _PY_EXCEPTION_LINE_RE.search(lines[i]):
            lo = max(0, i - 3)
            hi = min(len(lines), i + 2)
            return lines[lo:hi]
    return []


def _bash_memory_marker(first: str, reducer: str, obs: str) -> list[str]:
    return [
        first,
        (
            f"[tool-output compressed: reducer={reducer} "
            f"raw_chars={len(obs)} raw_lines={len(obs.splitlines())}]"
        ),
    ]


def _compress_training_bash_output_for_memory(obs: str, *, tail_lines: int = 5) -> str:
    lines = obs.splitlines()
    if not lines:
        return obs
    first, body = lines[0], lines[1:]
    if len(body) <= 20:
        return obs
    tail_n = max(1, int(tail_lines or 5))
    tail = body[-tail_n:]
    tail_keys = {ln.strip() for ln in tail}
    error_block = _extract_recent_python_error_block(body)
    error_keys = {ln.strip() for ln in error_block}
    signals = _dedupe_recent_nonempty_lines(
        [ln for ln in body if _BASH_TRAINING_SIGNAL_RE.search(ln)],
        limit=15,
    )
    signals = [
        ln for ln in signals
        if ln.strip() not in tail_keys and ln.strip() not in error_keys
    ]
    parts = _bash_memory_marker(first, "bash_training_signal_v2", obs)
    if error_block:
        parts.append("--- preserved error/traceback lines ---")
        parts.extend(error_block)
    if signals:
        parts.append("--- preserved training signal lines ---")
        parts.extend(signals)
    parts.append(f"--- tail last {tail_n} lines ---")
    parts.extend(tail)
    return "\n".join(parts)


def _compress_pytest_bash_output_for_memory(obs: str) -> str:
    lines = obs.splitlines()
    if not lines:
        return obs
    if len(obs) <= 3600 and len(lines) <= 80:
        return obs
    first, body = lines[0], lines[1:]
    picked: list[str] = []
    for ln in body:
        if " PASSED " in f" {ln} " and not _BASH_PYTEST_SUMMARY_RE.search(ln):
            continue
        if _BASH_PYTEST_SUMMARY_RE.search(ln) or _BASH_PYTEST_FAILURE_RE.search(ln):
            picked.append(ln)
    picked = _dedupe_preserve_order([ln for ln in picked if ln.strip()], limit=80)
    if not picked:
        picked = body[-8:]
    parts = _bash_memory_marker(first, "bash_pytest_summary_v2", obs)
    parts.extend(picked[:80])
    return "\n".join(parts)


def _compress_install_bash_output_for_memory(obs: str) -> str:
    lines = obs.splitlines()
    if not lines:
        return obs
    first, body = lines[0], lines[1:]
    if len(body) <= 5:
        return obs
    signals = _dedupe_recent_nonempty_lines(
        [ln for ln in body if _BASH_INSTALL_SIGNAL_RE.search(ln)],
        limit=4,
    )
    tail = body[-1:] if body else []
    kept = _dedupe_preserve_order([*signals, *tail], limit=5)
    parts = _bash_memory_marker(first, "bash_install_summary_v2", obs)
    parts.extend(kept)
    return "\n".join(parts)


def compress_bash_tool_output_for_memory(
    obs: str,
    *,
    tail_lines: int = 20,
    command: str = "",
    dedup_enabled: bool = True,
    dedup_min_repeat: int = 3,
    dedup_summary_prefix: str = "[log-dedup]",
) -> str:
    """Keep bash header + last *tail_lines* lines of stdout/stderr (success paths only).

    When *dedup_enabled*, consecutive duplicate lines are collapsed before tail truncation.

    Never raises: failures fall back to *obs* unchanged.
    """
    try:
        if tail_lines <= 0:
            return obs
        obs_work = obs
        if dedup_enabled:
            obs_work = collapse_consecutive_repeated_lines_in_text(
                obs_work,
                min_repeat=int(dedup_min_repeat),
                summary_prefix=dedup_summary_prefix,
            )
        lines = obs_work.splitlines()
        if not lines:
            return obs
        first = lines[0]
        if not first.startswith("[exit="):
            return obs_work
        if looks_like_bare_solution_run(command):
            return _compress_training_bash_output_for_memory(
                obs_work,
                tail_lines=tail_lines,
            )
        if _bash_command_looks_like_test(command):
            return _compress_pytest_bash_output_for_memory(obs_work)
        if _bash_command_looks_like_install(command):
            return _compress_install_bash_output_for_memory(obs_work)
        body_lines = lines[1:]
        n = len(body_lines)
        if n <= tail_lines:
            return obs_work
        tail = body_lines[-tail_lines:]
        # Do not use str.strip("[]") — that strips any leading/trailing [ or ] chars, not a pair.
        meta = (
            first[1:-1]
            if len(first) >= 2 and first.startswith("[") and first.endswith("]")
            else first
        )
        return (
            f"[{meta}] [bash: {n} lines total, showing last {tail_lines}]\n"
            + "\n".join(tail)
        )
    except Exception:
        logger.debug("compress_bash_tool_output_for_memory failed", exc_info=True)
        return obs


def _split_recent_turn_suffix(
    messages: list[Message],
    *,
    keep_recent_turns: int,
) -> tuple[list[Message], list[Message]]:
    """Split messages into (older, recent suffix) by assistant-turn count."""
    if keep_recent_turns <= 0 or not messages:
        return list(messages), []
    idx = len(messages)
    turns = 0
    while idx > 0 and turns < keep_recent_turns:
        idx -= 1
        if messages[idx].role == "assistant":
            turns += 1
    return messages[:idx], messages[idx:]


# Anchor that splits the first user message (msg[0]) into:
#   head — verbatim policy / ensemble-fusion format / lite-task title (must be preserved)
#   body — the description_lite.md content (may be summarized)
# Matches the literal "## Task description" line emitted by orchestrator.pin_message
# (see ``scienceflow/core/orchestrator.py``: ``"## Task description\n\n" + td_stripped``)
# and present at line 3 of every registered ``tasks/**/<exp_id>/description_lite.md``.
_TASK_DESC_BODY_RE = re.compile(r"(?m)^##\s+Task description\b")

MSG0_BODY_TRUNCATED_PLACEHOLDER = "[task description body truncated for context]"


def _split_msg0_head_body(text: str) -> tuple[str, str]:
    """Split the leading user message into (head, body) at ``## Task description``.

    Head is everything strictly before the first line that starts with
    ``## Task description``; body is that line plus everything after it.
    When the anchor is absent (older tasks, custom callers, or messages that
    are not the lite-task pin), the whole text is treated as ``head`` and
    body is empty — preserving the existing behavior of leaving such
    messages untouched.
    """
    if not text:
        return "", ""
    m = _TASK_DESC_BODY_RE.search(text)
    if not m:
        return text, ""
    head = text[: m.start()].rstrip("\n")
    body = text[m.start():]
    return head, body


def _compress_tool_output_for_mechanical_compact(content: str) -> str:
    """Deterministic compression for old tool outputs (mid-run, no LLM)."""
    try:
        if not content:
            return content
        low = content.lower()
        first = content.splitlines()[0] if content else ""

        # Keep failures/errors intact for debugging.
        if "traceback" in low or "error:" in low[:2000] or "non-zero exit code" in low[:2000]:
            return content

        if first.startswith("[exit="):
            return compress_bash_tool_output_for_memory(content, tail_lines=3)

        if first.startswith("[") and ("lines total" in first or "showing" in first):
            return compress_read_output_for_memory(content, max_lines=5)

        if content.startswith("Edited ") and "1 replacement OK." in content:
            return compress_edit_success_output_for_memory(content)

        if "wrote " in low and "bytes" in low[:200]:
            lines = content.splitlines()
            if not lines:
                return content
            keep = [lines[0]]
            for ln in lines[1:]:
                if "sha256~" in ln:
                    keep.append(ln)
                    break
            return "\n".join(keep)

        if first.startswith("[skill:") or "[skill:" in low[:120]:
            if len(content) <= 200:
                return content
            return content[:200].rstrip() + "\n...[skill output truncated for context]"

        if len(content) > 800:
            return content[:800].rstrip() + "\n...[tool output truncated for context]"
        return content
    except Exception:
        logger.debug("_compress_tool_output_for_mechanical_compact failed", exc_info=True)
        return content


def _all_messages(memory: Memory) -> list[Message]:
    records = memory.chat_history_memory.retrieve(window_size=None)
    return [cr.memory_record.message for cr in records]


def _format_message_excerpt_for_compact(message: Message) -> str:
    """Build a longer excerpt per role for compact input (was fixed 500 chars)."""
    text = message.content if isinstance(message.content, str) else str(message.content or "")
    role = getattr(message, "role", "") or ""

    if role == "tool":
        # Prefer command + head of output for bash-like tool results
        lines = text.splitlines()
        head_n = 40
        if len(lines) <= head_n:
            body = text
        else:
            body = "\n".join(lines[:head_n]) + f"\n... ({len(lines)} lines total) ..."
        return body[:6000]

    # user / assistant / system: more room for code and reasoning
    cap = 2000
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n...[truncated, total {len(text)} chars]"


def extract_compacted_summary_body(memory: Memory) -> str | None:
    """Return text after :data:`COMPACTED_CONVERSATION_SUMMARY_MARKER` in a post-compact message.

    New compacts store this as ``role=user`` (Gemini-compatible handoff); legacy workspaces
    may still have ``role=system``.
    """
    marker = COMPACTED_CONVERSATION_SUMMARY_MARKER
    for m in _all_messages(memory):
        if getattr(m, "role", None) not in ("system", "user"):
            continue
        c = m.content if isinstance(m.content, str) else str(m.content or "")
        if marker not in c:
            continue
        idx = c.find(marker)
        rest = c[idx + len(marker) :].lstrip()
        if rest.startswith("\n"):
            rest = rest[1:]
        rest = rest.strip()
        return rest or None
    return None


@dataclass
class FileSnapshotInfo:
    lines: int
    sha256_short: str


class MemoryContextManager:
    """Sliding-window chat + file snapshots + optional compact summary.

    **Pinned messages (L1):** registered via :meth:`pin_message`; prepended before the sliding-window
    slice. The total size of pinned content may be capped by :attr:`pinned_budget_ratio` relative to
    :attr:`budget_chars` (the first pinned block is never truncated).
    """

    def __init__(
        self,
        memory: Memory,
        workspace_dir: Path,
        *,
        budget_chars: int = 60_000,
        file_snapshot_max: int = 8,
        path_guard_extra_roots: Sequence[str | Path] | None = None,
        tool_memory_compression: bool = True,
        bash_success_tail_lines: int = 20,
        bash_success_tail_lines_solution: int = 8,
        bash_success_tail_lines_test: int = 120,
        bash_success_tail_lines_readonly: int = 30,
        bash_success_tail_lines_install: int = 5,
        read_success_max_lines: int = 200,
        sliding_window_priority_enabled: bool = True,
        pinned_budget_ratio: float = 0.45,
        bash_output_dedup_apply_to_memory: bool = True,
        bash_output_dedup_min_repeat: int = 3,
        bash_output_dedup_summary_prefix: str = "[log-dedup]",
        write_auto_snapshot_enabled: bool = True,
        write_auto_snapshot_paths: Sequence[str] | None = None,
        write_auto_snapshot_code_extensions: Sequence[str] | None = None,
        write_auto_snapshot_max_lines: int = 400,
        write_auto_snapshot_max_chars: int = 8_000,
        write_auto_snapshot_changed_context_lines: int = 10,
        write_auto_snapshot_symbol_body_lines: int = 3,
        read_overlap_guard_enabled: bool = True,
        msg0_compress_body: bool = False,
    ) -> None:
        self._memory = memory
        self._workspace = Path(workspace_dir).resolve()
        self._guard = PathGuard(self._workspace, extra_roots=path_guard_extra_roots)
        self._budget_chars = int(budget_chars)
        self._pinned_budget_ratio = float(pinned_budget_ratio)
        self._file_snapshot_max = int(file_snapshot_max)
        self._file_snapshots: dict[str, FileSnapshotInfo] = {}
        self._edit_fail_counter: dict[str, int] = {}
        self._edit_fail_total_counter: dict[str, int] = {}
        self._edit_fail_escalation_level: dict[str, int] = {}
        self._write_counter_by_path: dict[str, int] = {}
        self._tool_memory_compression = bool(tool_memory_compression)
        self._bash_success_tail_lines = int(bash_success_tail_lines)
        self._bash_success_tail_lines_solution = int(bash_success_tail_lines_solution)
        self._bash_success_tail_lines_test = int(bash_success_tail_lines_test)
        self._bash_success_tail_lines_readonly = int(bash_success_tail_lines_readonly)
        self._bash_success_tail_lines_install = int(bash_success_tail_lines_install)
        self._read_success_max_lines = int(read_success_max_lines)
        self._sliding_window_priority_enabled = bool(sliding_window_priority_enabled)
        self._bash_output_dedup_apply_to_memory = bool(bash_output_dedup_apply_to_memory)
        self._bash_output_dedup_min_repeat = int(bash_output_dedup_min_repeat)
        self._bash_output_dedup_summary_prefix = str(bash_output_dedup_summary_prefix)
        self._write_auto_snapshot_enabled = bool(write_auto_snapshot_enabled)
        _wpaths = list(write_auto_snapshot_paths) if write_auto_snapshot_paths else []
        self._write_auto_snapshot_paths: tuple[str, ...] = tuple(
            (p or "").replace("\\", "/").lstrip("/").strip() for p in _wpaths if (p or "").strip()
        )
        _code_exts = (
            list(write_auto_snapshot_code_extensions)
            if write_auto_snapshot_code_extensions is not None
            else [".py"]
        )
        self._write_auto_snapshot_code_extensions: tuple[str, ...] = tuple(
            e if e.startswith(".") else f".{e}"
            for e in (str(x).strip().lower() for x in _code_exts)
            if e
        ) or (".py",)
        self._write_auto_snapshot_max_lines = max(1, int(write_auto_snapshot_max_lines))
        self._write_auto_snapshot_max_chars = max(256, int(write_auto_snapshot_max_chars))
        self._write_auto_snapshot_changed_context_lines = max(
            1,
            int(write_auto_snapshot_changed_context_lines),
        )
        self._write_auto_snapshot_symbol_body_lines = max(
            0,
            int(write_auto_snapshot_symbol_body_lines),
        )
        self._read_overlap_guard_enabled = bool(read_overlap_guard_enabled)
        # When False (default), msg[0] is preserved verbatim by both the in-session
        # mechanical compactor and the LLM-fallback ``compact()`` — the entire user
        # pin (head + ``## Task description`` body) survives every compress pass.
        # When True, only the head is preserved verbatim and the body is replaced
        # with ``MSG0_BODY_TRUNCATED_PLACEHOLDER`` to save tokens.
        self._msg0_compress_body = bool(msg0_compress_body)
        # Optional LHR hook: preserve early EDA/tool evidence verbatim across compact.
        # This is an absolute message index in the current compacted memory layout;
        # messages after the leading user/task pin and before this index are never summarized.
        self._protected_raw_prefix_end_index: int = 0
        self._protected_raw_prefix_warn_chars: int = 0
        self._protected_raw_prefix_label: str = "protected raw prefix"
        # rel path -> (sha256~short, sorted merged (lo, hi) intervals) for read-overlap nudges
        self._read_coverage: dict[str, tuple[str, list[tuple[int, int]]]] = {}
        # rel path -> symbol key -> last complete read coverage for that Python symbol.
        self._read_symbol_coverage: dict[str, dict[str, SymbolReadCoverage]] = {}
        # rel path -> last sha256 short for which an auto-snapshot was injected.
        # Used to dedup repeated ``[auto-snapshot after successful write]`` blocks
        # when the on-disk content does not change between successive writes.
        self._last_auto_snapshot_sha_by_path: dict[str, str] = {}
        # rel path -> last code text used for semantic changed-range snapshot diffing.
        self._last_auto_snapshot_text_by_path: dict[str, str] = {}
        # Tier 2 (silent read interception): rel path -> count of redundant reads (range fully
        # covered by prior reads at the same on-disk sha). Used by teleport health checks and
        # tests to verify the silent-intercept path actually fired.
        self._redundant_read_count_by_path: dict[str, int] = {}
        self._pinned_messages: list[Message] = []
        # Monotone lower bound on the suffix start index for prefix-cache stability.
        self._last_window_k: int = 0
        self._seen_tool_signatures: dict[tuple[str, str], int] = {}
        self._released_full_signatures: set[tuple[str, str]] = set()

    def _norm_rel(self, r: str) -> str:
        return (r or "").replace("\\", "/").lstrip("/").strip()

    def _path_matches_write_snapshot(self, rel: str) -> bool:
        """True if *rel* is configured for code auto-snapshots."""
        nr = self._norm_rel(rel)
        if not nr:
            return False
        name = PurePosixPath(nr).name
        for pat in self._write_auto_snapshot_paths:
            p = self._norm_rel(pat)
            if not p:
                continue
            if nr == p or name == PurePosixPath(p).name:
                return True
        suffix = PurePosixPath(nr).suffix.lower()
        return bool(suffix and suffix in self._write_auto_snapshot_code_extensions)

    def _current_snapshot_sha_for_rel(self, rel: str) -> str:
        nr = self._norm_rel(rel)
        if not nr or not self._path_matches_write_snapshot(nr):
            return ""
        expected = self._last_auto_snapshot_sha_by_path.get(nr, "")
        if not expected:
            return ""
        try:
            resolved = self._guard.resolve(nr)
            if not resolved.is_file():
                return ""
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            return ""
        cur = _sha256_short_bytes(text.encode("utf-8", errors="replace"))
        return cur if cur == expected else ""

    def _compress_grep_snapshot_refs_for_memory(self, obs: str, *, raw_id: str = "") -> str:
        if not obs:
            return obs
        groups: dict[str, list[tuple[int, str]]] = {}
        order: list[str] = []
        passthrough: list[str] = []
        for line in obs.splitlines():
            m = re.match(r"^(.+?):(\d+):(.*)$", line)
            if not m:
                passthrough.append(line)
                continue
            rel = self._norm_rel(m.group(1))
            if not rel:
                passthrough.append(line)
                continue
            if rel not in groups:
                groups[rel] = []
                order.append(rel)
            groups[rel].append((int(m.group(2)), m.group(3)))
        if not groups:
            return obs
        replaced_any = False
        out: list[str] = [
            (
                f"[tool-output compressed: reducer=snapshot_ref_v1 "
                f"raw_chars={len(obs)} raw_lines={len(obs.splitlines())}]"
            )
        ]
        for rel in order:
            matches = groups[rel]
            sha = self._current_snapshot_sha_for_rel(rel)
            if sha and len(matches) > 5:
                nums = ",".join(str(n) for n, _ in matches[:10])
                if len(matches) > 10:
                    nums += ",..."
                out.append(
                    f"{rel}: {len(matches)} matches (lines {nums}); "
                    f"[see current snapshot: {rel} sha~{sha}]"
                )
                replaced_any = True
                continue
            for lineno, text in matches:
                out.append(f"{rel}:{lineno}:{text}")
        if passthrough:
            out.extend(passthrough[-3:])
        if raw_id:
            out.append(f"[exact raw output: {raw_id}]")
        return "\n".join(out) if replaced_any else obs

    def _update_read_coverage_and_is_redundant(
        self,
        rel: str,
        args: dict[str, Any],
        *,
        raw_id: str = "",
    ) -> tuple[bool, str | None]:
        """Update read coverage; return redundant flag and optional replacement summary."""
        if not self._read_overlap_guard_enabled or not rel:
            return False, None
        try:
            resolved = self._guard.resolve(rel)
        except (ValueError, OSError):
            return False, None
        if not resolved.is_file():
            return False, None
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False, None
        lines = text.splitlines(keepends=False)
        n = len(lines)
        if n == 0:
            return False, None
        start0 = max(0, int(args.get("offset") or 1) - 1)
        lim = int(args.get("limit") or 200)
        if lim <= 0:
            lo, hi = 1, n
        else:
            lo = start0 + 1
            hi = min(n, start0 + lim)
        if lo > hi:
            return False, None
        raw = text.encode("utf-8", errors="replace")
        short = _sha256_short_bytes(raw)

        # Python code gets a finer-grained symbol coverage index. Once a complete
        # function/class has been read at a specific symbol sha, repeated reads inside
        # that same symbol can be summarized even if unrelated parts of the file changed.
        symbols = (
            _parse_python_source_symbols(text)
            if PurePosixPath(rel).suffix.lower() == ".py"
            else []
        )
        symbol_redundant_summary: str | None = None
        if symbols:
            cov_by_key = self._read_symbol_coverage.setdefault(rel, {})
            for sym in symbols:
                cov = cov_by_key.get(sym.key)
                if (
                    cov is not None
                    and cov.symbol_sha == _symbol_sha(text, sym)
                    and lo >= cov.start
                    and hi <= cov.end
                ):
                    symbol_redundant_summary = _build_redundant_symbol_read_coverage_summary(
                        rel,
                        requested=(lo, hi),
                        coverage=cov,
                    )
                    break
            for sym in symbols:
                if lo <= sym.start and hi >= sym.end:
                    previous_cov = cov_by_key.get(sym.key)
                    cov_by_key[sym.key] = SymbolReadCoverage(
                        symbol_name=sym.name,
                        symbol_kind=sym.kind,
                        start=sym.start,
                        end=sym.end,
                        symbol_sha=_symbol_sha(text, sym),
                        raw_id=raw_id or (previous_cov.raw_id if previous_cov else ""),
                    )
            if symbol_redundant_summary:
                return True, symbol_redundant_summary

        ent = self._read_coverage.get(rel)
        merged_prior = _merge_read_intervals(list(ent[1])) if ent and ent[0] == short else []
        redundant = bool(
            ent is not None
            and ent[0] == short
            and merged_prior
            and _range_fully_covered_by_intervals(lo, hi, merged_prior),
        )
        if ent is None or ent[0] != short:
            self._read_coverage[rel] = (short, _merge_read_intervals([(lo, hi)]))
        else:
            self._read_coverage[rel] = (short, _merge_read_intervals([*ent[1], (lo, hi)]))
        return redundant, None

    def _prune_symbol_read_coverage_after_write(self, rel: str, text: str) -> None:
        cov_by_key = self._read_symbol_coverage.get(rel)
        if not cov_by_key:
            return
        symbols = {
            sym.key: (sym, _symbol_sha(text, sym))
            for sym in _parse_python_source_symbols(text)
        }
        if not symbols:
            self._read_symbol_coverage.pop(rel, None)
            return
        kept: dict[str, SymbolReadCoverage] = {}
        for key, cov in cov_by_key.items():
            current = symbols.get(key)
            if current is None:
                continue
            sym, sym_sha = current
            if cov.symbol_sha != sym_sha:
                continue
            kept[key] = SymbolReadCoverage(
                symbol_name=sym.name,
                symbol_kind=sym.kind,
                start=sym.start,
                end=sym.end,
                symbol_sha=sym_sha,
                raw_id=cov.raw_id,
            )
        if kept:
            self._read_symbol_coverage[rel] = kept
        else:
            self._read_symbol_coverage.pop(rel, None)

    def _trim_snapshots(self) -> None:
        if len(self._file_snapshots) <= self._file_snapshot_max:
            return
        for key in sorted(self._file_snapshots.keys())[
            : -self._file_snapshot_max
        ]:
            self._file_snapshots.pop(key, None)

    def _update_snapshot_from_path(self, rel_path: str) -> None:
        try:
            p = self._guard.resolve(rel_path)
        except ValueError:
            return
        if not p.is_file():
            return
        text = p.read_text(encoding="utf-8", errors="replace")
        raw = text.encode("utf-8")
        short = hashlib.sha256(raw).hexdigest()[:16]
        self._file_snapshots[rel_path] = FileSnapshotInfo(
            lines=len(text.splitlines()),
            sha256_short=short,
        )
        self._trim_snapshots()

    @property
    def file_state_summary(self) -> str:
        if not self._file_snapshots:
            return ""
        lines = ["[Workspace file state]"]
        for path, info in sorted(self._file_snapshots.items()):
            lines.append(
                f"  {path}: {info.lines} lines, hash~{info.sha256_short}",
            )
        return "\n".join(lines)

    def pin_message(self, msg: Message) -> None:
        """Append a pinned user message (L1). Pinned messages are always sent in full
        before the sliding-window slice of chat history."""
        self._pinned_messages.append(msg)

    @property
    def pinned_messages(self) -> list[Message]:
        """Copy of pinned messages for auditing."""
        return list(self._pinned_messages)

    def build_messages_for_llm_with_stats(self) -> tuple[list[Message], int]:
        """Return chat messages for the next LLM call and how many raw messages were dropped.

        *omitted* counts messages from the in-memory conversation that did not fit the sliding
        window (before the synthetic ``[Context budget: … omitted]`` user message is added).
        """
        raw_messages = _all_messages(self._memory)
        messages = (
            _project_messages_for_llm_view(raw_messages)
            if self._tool_memory_compression
            else list(raw_messages)
        )
        stable_leading_prefix, messages = _split_leading_user_prefix(messages)
        pinned_base = _apply_pinned_budget(
            list(self._pinned_messages),
            budget_chars=self._budget_chars,
            pinned_budget_ratio=self._pinned_budget_ratio,
        )
        pinned = [*pinned_base, *stable_leading_prefix]

        if not messages and not pinned:
            return [], 0

        budget = self._budget_chars
        if budget <= 0:
            out = pinned + messages
            return _finalize_messages_for_llm(out), 0

        leading, tail_messages = _split_leading_user_prefix(messages)
        pinned_len = sum(_message_char_len(m) for m in pinned)
        leading_len = sum(_message_char_len(m) for m in leading)
        budget_rest = max(0, budget - pinned_len - leading_len)

        if not tail_messages:
            return _finalize_messages_for_llm([*pinned, *leading]), 0

        if self._sliding_window_priority_enabled:
            chosen, new_k = _best_suffix_for_budget_priority(
                tail_messages, budget_rest, min_k=self._last_window_k,
            )
            self._last_window_k = new_k
        else:
            total = 0
            chosen_rev: list[Message] = []
            fallback_k = len(tail_messages)
            for i, m in enumerate(reversed(tail_messages)):
                lc = _message_char_len(m)
                if chosen_rev and total + lc > budget_rest:
                    break
                chosen_rev.append(m)
                total += lc
                fallback_k = len(tail_messages) - 1 - i
            chosen = list(reversed(chosen_rev))
            self._last_window_k = max(
                self._last_window_k,
                fallback_k if chosen else len(tail_messages),
            )

        selected_count_before_format_cleanup = len(chosen)
        chosen = _finalize_messages_for_llm(chosen)

        omitted = len(tail_messages) - selected_count_before_format_cleanup
        # OpenAI-style chat payloads cannot start a visible suffix with a tool
        # result. Dropping those leading tool messages is format cleanup, not a
        # context-budget omission; otherwise a successful compact can be
        # misreported as still omitting history.
        while chosen and chosen[0].role == "tool":
            chosen.pop(0)

        result: list[Message] = [*pinned, *leading]
        if omitted > 0:
            # Fixed placeholder text (no variable count) keeps the prefix byte-identical
            # across consecutive calls, maximising LLM prefix-cache hit rates.
            result.append(
                Message.user_message(
                    "[Context budget: earlier messages omitted from this request. "
                    "Use tools to re-read files if needed.]",
                ),
            )
        result.extend(chosen)
        return _finalize_messages_for_llm(result), omitted

    def build_messages_for_llm(self) -> list[Message]:
        """Return chat messages for the next LLM call: pinned (L1) + conversation (L2/L3).

        File snapshot metadata is no longer a synthetic leading user message; it is merged
        into the system prompt by :class:`~scienceflow.core.agent.prompts.system_prompt.SystemPromptMixin`.
        """
        return self.build_messages_for_llm_with_stats()[0]

    def rewrite_messages(self, messages: list[Message]) -> None:
        """Replace chat memory and truncate the append-only long-term log."""
        _clear_memory_and_long_term_log(self._memory)
        for message in messages:
            self._memory.add_message(message)
        self._last_window_k = 0

    def replay_state_from_inherited_memory(self) -> dict[str, int]:
        """Rebuild session-local guard state from already-loaded inherited memory.

        Clone-continue starts a fresh :class:`MemoryContextManager`, so in-session state such as
        read coverage and repeated read/bash signatures would otherwise be empty even though the
        inherited transcript contains the same tool calls. Replay only deterministic metadata:
        no messages are added or rewritten here.
        """
        messages = _all_messages(self._memory)
        if not messages:
            return {
                "read_coverage": 0,
                "file_snapshots": 0,
                "tool_signatures": 0,
            }

        self._read_coverage.clear()
        self._read_symbol_coverage.clear()
        self._file_snapshots.clear()
        self._seen_tool_signatures.clear()
        self._released_full_signatures.clear()

        read_coverage = 0
        file_snapshots = 0
        tool_signatures = 0
        for i, m in enumerate(messages):
            role = getattr(m, "role", "") or ""
            if role != "tool":
                continue
            tcid = str(getattr(m, "tool_call_id", "") or "")
            if tcid.startswith("inherited_fullrun_"):
                continue
            name = str(getattr(m, "name", "") or "")
            args_d = _tool_args_dict_for_tool_message(messages, i)
            if not isinstance(args_d, dict):
                continue

            sig = _tool_call_signature(name, args_d)
            if sig is not None:
                self._seen_tool_signatures[sig] = i
                tool_signatures += 1

            rel = str(args_d.get("path") or "").replace("\\", "/").lstrip("/")
            if not rel:
                continue

            content = m.content if isinstance(m.content, str) else str(m.content or "")
            if name == "read":
                before = dict(self._read_coverage)
                raw_id = _tool_output_raw_id_from_content(content)
                self._update_read_coverage_and_is_redundant(
                    rel,
                    args_d,
                    raw_id=raw_id,
                )
                if self._read_coverage != before or rel in self._read_symbol_coverage:
                    read_coverage += 1
                continue

            if name in ("write", "edit"):
                ok = (
                    name == "write" and _write_tool_feedback_is_success(content)
                ) or (
                    name == "edit"
                    and content.startswith("Edited ")
                    and "1 replacement OK." in content.split("\n", 1)[0]
                )
                if ok:
                    self._read_coverage.pop(rel, None)
                    try:
                        resolved = self._guard.resolve(rel)
                        if resolved.is_file():
                            self._prune_symbol_read_coverage_after_write(
                                rel,
                                resolved.read_text(encoding="utf-8", errors="replace"),
                            )
                    except (ValueError, OSError):
                        self._read_symbol_coverage.pop(rel, None)
                    self._update_snapshot_from_path(rel)
                    if rel in self._file_snapshots:
                        file_snapshots += 1

        return {
            "read_coverage": read_coverage,
            "file_snapshots": file_snapshots,
            "tool_signatures": tool_signatures,
        }

    def mechanical_compress_old_messages(self, *, keep_recent_turns: int = 3) -> dict[str, int]:
        """Deterministically compress older chat messages (no LLM call).

        Keeps the latest ``keep_recent_turns`` assistant turns untouched and compresses only
        older content, mainly successful tool outputs.
        """
        messages = _all_messages(self._memory)
        if not messages:
            return {"total": 0, "kept": 0, "changed": 0, "dropped": 0}

        old, recent = _split_recent_turn_suffix(
            messages,
            keep_recent_turns=max(0, int(keep_recent_turns)),
        )

        # Leading-user pin: indices of consecutive ``user`` messages at the head of ``old``.
        # Mirrors the teleport cross-fork compress (see
        # ``scienceflow/solver/lnr/_internal/teleport_inherit_compress.py``)
        # so that the task pin (policy + ensemble-fusion format + lite title) is never
        # truncated to 400 chars and thus the model never loses constraints like
        # "Do NOT ensemble / stack / blend".
        leading_pin: set[int] = set()
        for j, mm in enumerate(old):
            if getattr(mm, "role", None) == "user":
                leading_pin.add(j)
            else:
                break
        msg0_idx = 0 if (old and getattr(old[0], "role", None) == "user") else None

        changed = 0
        dropped = 0
        compressed_old: list[Message] = []
        for i, m in enumerate(old):
            role = getattr(m, "role", "") or ""
            text = _message_text_for_chars(m)
            if role == "tool":
                new_text = _compress_tool_output_for_mechanical_compact(text)
                if new_text != text:
                    changed += 1
                compressed_old.append(_message_with_text(m, new_text))
                continue

            if role == "assistant":
                if len(text) > 500:
                    changed += 1
                    compressed_old.append(
                        _message_with_text(
                            m,
                            text[:500].rstrip() + "\n...[assistant message truncated for context]",
                        ),
                    )
                else:
                    compressed_old.append(m)
                continue

            if role == "user":
                # msg[0]: by default kept verbatim (whole user pin). When the
                # ``msg0_compress_body`` switch is on, the body after
                # ``## Task description`` is replaced with a placeholder while
                # the head (policy + ensemble fusion + lite title) stays intact.
                if i == msg0_idx:
                    if self._msg0_compress_body:
                        head, body = _split_msg0_head_body(text)
                        if body:
                            new_text = (
                                head.rstrip("\n")
                                + "\n\n## Task description\n"
                                + MSG0_BODY_TRUNCATED_PLACEHOLDER
                            )
                            if new_text != text:
                                changed += 1
                                compressed_old.append(_message_with_text(m, new_text))
                                continue
                    compressed_old.append(m)
                    continue
                # Other leading user messages: preserve verbatim (cross-branch overview,
                # additional pinned turns), aligned with teleport cross-fork compress.
                if i in leading_pin:
                    compressed_old.append(m)
                    continue
                if (
                    text.startswith("[Context budget:")
                    or "## Time Budget" in text[:1200]
                    or "## Trajectory branch history" in text[:1200]
                    or "## Cross-branch overview" in text[:1200]
                ):
                    dropped += 1
                    changed += 1
                    continue
                if len(text) > 400:
                    changed += 1
                    compressed_old.append(
                        _message_with_text(
                            m,
                            text[:400].rstrip() + "\n...[user message truncated for context]",
                        ),
                    )
                else:
                    compressed_old.append(m)
                continue

            compressed_old.append(m)

        merged = compressed_old + recent
        _clear_memory_and_long_term_log(self._memory)
        for m in merged:
            self._memory.add_message(m)

        return {
            "total": len(messages),
            "kept": len(merged),
            "changed": changed,
            "dropped": dropped,
        }

    def _leading_user_messages_for_compact(self, messages: list[Message]) -> list[Message]:
        """Leading user/task messages to preserve verbatim across a compact rewrite."""
        pinned_leading: list[Message] = []
        for j, mm in enumerate(messages):
            if getattr(mm, "role", None) != "user":
                break
            text = _message_text_for_chars(mm)
            if text.startswith(COMPACTED_CONVERSATION_SUMMARY_MARKER):
                break
            if j == 0 and self._msg0_compress_body:
                head, body = _split_msg0_head_body(text)
                if body:
                    pinned_leading.append(
                        _message_with_text(
                            mm,
                            head.rstrip("\n")
                            + "\n\n## Task description\n"
                            + MSG0_BODY_TRUNCATED_PLACEHOLDER,
                        ),
                    )
                    continue
            pinned_leading.append(mm)
        return pinned_leading

    def set_protected_raw_prefix(
        self,
        end_index: int,
        *,
        warn_chars: int = 50_000,
        label: str = "protected raw prefix",
    ) -> dict[str, int | str]:
        """Preserve messages before *end_index* verbatim across future compact calls.

        Used by long-horizon REPL to keep early EDA evidence raw while compacting
        later exploration.  The first user/task pin is already protected separately;
        this marker protects the dynamic records immediately after that pin.
        If the fixed prefix exceeds ``warn_chars`` we warn but still keep it verbatim.
        """
        messages = _all_messages(self._memory)
        end = max(0, min(int(end_index), len(messages)))
        self._protected_raw_prefix_end_index = end
        self._protected_raw_prefix_warn_chars = max(0, int(warn_chars or 0))
        self._protected_raw_prefix_label = str(label or "protected raw prefix")
        pinned = self._leading_user_messages_for_compact(messages)
        protected, _rest, chars = self._protected_raw_prefix_parts(messages, len(pinned))
        return {
            "end_index": end,
            "message_count": len(protected),
            "chars": chars,
            "warn_chars": self._protected_raw_prefix_warn_chars,
            "label": self._protected_raw_prefix_label,
        }

    def _protected_raw_prefix_parts(
        self,
        messages: list[Message],
        pinned_len: int,
    ) -> tuple[list[Message], list[Message], int]:
        end = max(0, int(getattr(self, "_protected_raw_prefix_end_index", 0) or 0))
        end = min(end, len(messages))
        start = max(0, min(int(pinned_len), len(messages)))
        if end <= start:
            return [], list(messages[start:]), 0
        protected = list(messages[start:end])
        rest = list(messages[end:])
        chars = 0
        for m in protected:
            role = str(getattr(m, "role", "") or "")
            chars += len(role) + 3 + _message_char_len(m)
        warn_chars = max(0, int(getattr(self, "_protected_raw_prefix_warn_chars", 0) or 0))
        if warn_chars and chars > warn_chars:
            label = str(getattr(self, "_protected_raw_prefix_label", "protected raw prefix") or "protected raw prefix")
            logger.warning(
                "%s is %d chars, exceeding warning threshold %d; keeping it verbatim as fixed prefix",
                label,
                chars,
                warn_chars,
            )
        return protected, rest, chars

    def replace_protected_raw_prefix_with_summary(
        self,
        end_index: int,
        summary: str,
        *,
        warn_chars: int = 50_000,
        label: str = "protected summary prefix",
    ) -> dict[str, int | str]:
        """Replace a raw protected prefix with one compact protected summary card.

        Long-horizon REPL uses this after the first metric-backed stage: early EDA
        facts stay as a fixed prefix, but scratch Python, full heredocs, and probe
        tool-call arguments are removed from the stable LLM context. Raw audit logs
        remain on disk under workspace ``.logs`` and stage snapshots.
        """
        messages = _all_messages(self._memory)
        if not messages:
            return {
                "end_index": 0,
                "message_count": 0,
                "chars": 0,
                "warn_chars": max(0, int(warn_chars or 0)),
                "label": str(label or "protected summary prefix"),
                "original_message_count": 0,
                "original_chars": 0,
                "mode": "summary",
            }
        end = max(0, min(int(end_index), len(messages)))
        pinned = self._leading_user_messages_for_compact(messages)
        start = max(0, min(len(pinned), len(messages)))
        protected_original = list(messages[start:end]) if end > start else []
        rest = list(messages[end:])
        original_chars = 0
        for m in protected_original:
            role = str(getattr(m, "role", "") or "")
            original_chars += len(role) + 3 + _message_char_len(m)

        body = (summary or "").strip()
        if not body:
            body = "No compact EDA facts were extracted; use dataset files and workspace logs for details."
        card_label = str(label or "protected summary prefix")
        summary_msg = Message.assistant_message(f"[{card_label}]\n{body}")

        _clear_memory_and_long_term_log(self._memory)
        for pm in pinned:
            self._memory.add_message(pm)
        self._memory.add_message(summary_msg)
        for m in rest:
            self._memory.add_message(m)

        self._protected_raw_prefix_end_index = len(pinned) + 1
        self._protected_raw_prefix_warn_chars = max(0, int(warn_chars or 0))
        self._protected_raw_prefix_label = card_label
        chars = len("assistant") + 3 + _message_char_len(summary_msg)
        if self._protected_raw_prefix_warn_chars and chars > self._protected_raw_prefix_warn_chars:
            logger.warning(
                "%s is %d chars, exceeding warning threshold %d; keeping it verbatim as fixed prefix",
                card_label,
                chars,
                self._protected_raw_prefix_warn_chars,
            )
        return {
            "end_index": self._protected_raw_prefix_end_index,
            "message_count": 1,
            "chars": chars,
            "warn_chars": self._protected_raw_prefix_warn_chars,
            "label": card_label,
            "original_message_count": len(protected_original),
            "original_chars": original_chars,
            "mode": "summary",
        }

    def build_inband_compact_messages(
        self,
        *,
        mid_run: bool = True,
        max_history_chars: int | None = None,
    ) -> tuple[list[Message], int]:
        """Build a same-session compact request using the normal agent system prompt.

        The returned messages are meant to be sent with the main system prompt and
        ``tool_choice=none``. This avoids the old separate summarizer system prompt
        while still letting the model produce a concise continuation state.
        """
        messages = _all_messages(self._memory)
        if not messages:
            return [], 0

        pinned_leading = self._leading_user_messages_for_compact(messages)
        protected_raw, dynamic_messages, protected_chars = self._protected_raw_prefix_parts(
            messages,
            len(pinned_leading),
        )
        history_parts: list[str] = []
        for m in dynamic_messages:
            role = getattr(m, "role", "") or ""
            text = _message_text_for_chars(m)
            if text.startswith("[Context budget:"):
                continue
            excerpt = _format_message_excerpt_for_compact(m)
            history_parts.append(f"[{role}]: {excerpt}")

        if max_history_chars is None:
            history_cap = max(8000, min(45000, int(self._budget_chars * 0.70)))
        else:
            history_cap = max(2000, int(max_history_chars))

        kept_rev: list[str] = []
        used = 0
        for part in reversed(history_parts):
            part_len = len(part) + 1
            if kept_rev and used + part_len > history_cap:
                break
            if not kept_rev and part_len > history_cap:
                part = part[-history_cap:]
                part_len = len(part) + 1
            kept_rev.append(part)
            used += part_len
        kept = list(reversed(kept_rev))
        omitted = len(history_parts) - len(kept)
        if omitted > 0:
            kept.insert(
                0,
                "[Earlier dynamic history omitted from this compact request: "
                f"{omitted} message(s). Leading user/task messages are preserved verbatim.]",
            )
        history_text = (
            "\n".join(kept).strip()
            or "(No dynamic history beyond the preserved leading user/task messages.)"
        )

        workspace_state: list[str] = []
        sol_path = self._workspace / "solution.py"
        if sol_path.is_file():
            try:
                sol_code = sol_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                sol_code = ""
            if sol_code.strip():
                if len(sol_code) > 6000:
                    sol_code = sol_code[:6000] + "\n... [truncated]"
                workspace_state.append(
                    "[CURRENT solution.py]\n```python\n" + sol_code + "\n```",
                )
        result_path = self._workspace / "result.md"
        if result_path.is_file():
            try:
                rt = result_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                rt = ""
            if rt.strip():
                workspace_state.append("[CURRENT result.md]\n" + rt[:3000])

        compact_kind = "same REPL session" if mid_run else "next continuation session"
        protected_note = ""
        if protected_raw:
            protected_note = (
                f"The runtime also preserves {len(protected_raw)} early EDA/tool message(s) "
                f"verbatim ({protected_chars} chars); do not repeat or paraphrase them. "
            )
        prompt = (
            "Internal context maintenance request: summarize the conversation state so the "
            f"{compact_kind} can continue without losing task intent. Do not call tools. "
            "Return only the compact state summary.\n\n"
            "The leading user/task messages immediately before this request are preserved "
            "verbatim by the runtime; do not repeat or paraphrase them. "
            f"{protected_note}"
            "Preserve concrete "
            "ML research state: EDA/data insights, current approach, validation/submission "
            "status, metrics, files changed, recent failures, leakage risks, and the next "
            "3-6 actions. "
            "Quote still-active explicit user prohibitions only if they are needed to avoid "
            "a mistake. Max ~700 words.\n\n"
            "[Dynamic conversation to compact]\n"
            f"{history_text}"
        )
        if workspace_state:
            prompt += "\n\n[Ground-truth workspace state]\n" + "\n\n".join(workspace_state)
        return [*pinned_leading, Message.user_message(prompt)], len(messages)

    def replace_history_with_compacted_summary(
        self,
        summary: str,
        *,
        recent_messages: int = 8,
    ) -> str:
        """Rewrite chat history as stable leading prefix + compact summary + recent tail."""
        messages = _all_messages(self._memory)
        body = (summary or "").strip()
        if not messages:
            return "Nothing to compact."
        if not body:
            return "Compact failed: empty summary."

        pinned_leading = self._leading_user_messages_for_compact(messages)
        protected_raw, dynamic_messages, protected_chars = self._protected_raw_prefix_parts(
            messages,
            len(pinned_leading),
        )
        n_recent = max(0, int(recent_messages))
        recent = list(dynamic_messages[-n_recent:]) if n_recent else []
        recent = [
            m
            for m in recent
            if COMPACTED_CONVERSATION_SUMMARY_MARKER not in _message_text_for_chars(m)
            and not _message_text_for_chars(m).startswith("[Context budget:")
        ]
        while recent and getattr(recent[0], "role", None) == "tool":
            recent.pop(0)

        _clear_memory_and_long_term_log(self._memory)
        for pm in pinned_leading:
            self._memory.add_message(pm)
        for pm in protected_raw:
            self._memory.add_message(pm)
        self._memory.add_message(
            Message.user_message(
                f"{COMPACTED_CONVERSATION_SUMMARY_MARKER}\n{body}",
            ),
        )
        for m in recent:
            self._memory.add_message(m)
        if protected_raw:
            self._protected_raw_prefix_end_index = len(pinned_leading) + len(protected_raw)
        self._last_window_k = 0
        if protected_raw:
            return (
                f"Compacted {len(messages)} messages into in-band summary "
                f"({len(body)} chars); kept {len(pinned_leading)} leading + "
                f"{len(protected_raw)} protected raw + {len(recent)} recent."
            )
        return (
            f"Compacted {len(messages)} messages into in-band summary "
            f"({len(body)} chars); kept {len(pinned_leading)} leading + {len(recent)} recent."
        )

    def record_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
        *,
        raw_id: str = "",
    ) -> str:
        """Format tool feedback for memory; update snapshots and edit-fail counters."""
        path = args.get("path") or ""
        rel = str(path).replace("\\", "/").lstrip("/")
        read_overlap_redundant = False
        read_overlap_summary: str | None = None

        if tool_name == "edit" and result.error:
            self._edit_fail_counter[rel] = self._edit_fail_counter.get(rel, 0) + 1
            self._edit_fail_total_counter[rel] = self._edit_fail_total_counter.get(rel, 0) + 1
        elif tool_name == "edit" and not result.error:
            self._edit_fail_counter.pop(rel, None)
            self._edit_fail_total_counter.pop(rel, None)
            self._edit_fail_escalation_level.pop(rel, None)

        if tool_name in ("write", "edit") and not result.error and rel:
            self._read_coverage.pop(rel, None)
            try:
                resolved = self._guard.resolve(rel)
                if resolved.is_file():
                    self._prune_symbol_read_coverage_after_write(
                        rel,
                        resolved.read_text(encoding="utf-8", errors="replace"),
                    )
            except (ValueError, OSError):
                self._read_symbol_coverage.pop(rel, None)
            self._update_snapshot_from_path(rel)

        obs = str(result)
        if (
            tool_name == "read"
            and not result.error
            and rel
            and _read_args_has_explicit_range(args)
            and _read_args_limit(args) > 0
        ):
            read_overlap_redundant, read_overlap_summary = (
                self._update_read_coverage_and_is_redundant(
                    rel,
                    args,
                    raw_id=raw_id,
                )
            )
        cmd_for_bash = str(args.get("command") or "") if tool_name == "bash" else ""
        is_bare_solution_bash = bool(
            tool_name == "bash"
            and not result.error
            and looks_like_bare_solution_run(cmd_for_bash)
        )

        if tool_name == "edit" and result.error and self._edit_fail_counter.get(rel, 0) >= 2:
            try:
                auto = self.inject_file_content(rel, reason="edit failed twice")
                obs = obs + "\n\n" + auto
                self._edit_fail_counter[rel] = 0
            except OSError as e:
                logger.debug("auto-read after edit fail: %s", e)

        if tool_name == "edit" and result.error:
            total_fails = self._edit_fail_total_counter.get(rel, 0)
            prev_level = self._edit_fail_escalation_level.get(rel, 0)
            level = 0
            coaching = ""
            if total_fails >= 6:
                level = 2
                coaching = (
                    "[Guard] edit has failed repeatedly on this file (6+ times). "
                    "Stop retrying `edit` with guessed snippets. You MUST `read` the latest file "
                    "content and then switch to a full `write` (entire file body) as fallback."
                )
            elif total_fails >= 4:
                level = 1
                coaching = (
                    "[Guard] edit has failed multiple times on this file (4+ times). "
                    "Re-sync context before the next attempt: `read` the file and either use a "
                    "larger exact `old_str` span or fallback to `write` full-file replacement."
                )
            if level > prev_level and coaching:
                obs = obs + "\n\n" + coaching
                self._edit_fail_escalation_level[rel] = level

        if is_bare_solution_bash:
            obs = _strip_bash_stderr_section_from_block(obs)
            self._write_counter_by_path.clear()

        sig = (
            _tool_call_signature(tool_name, args)
            if self._tool_memory_compression
            and not result.error
            and tool_name in ("bash", "read")
            else None
        )
        is_repeat = bool(sig and sig in self._seen_tool_signatures)

        if self._released_full_signatures and not is_repeat:
            _supersede_stale_compressed_records(
                memory=self._memory,
                released_sigs=self._released_full_signatures,
                seen_sigs=self._seen_tool_signatures,
            )

        if self._tool_memory_compression:
            if tool_name == "edit" and not result.error:
                obs = compress_edit_success_output_for_memory(obs)
            elif tool_name == "write" and not result.error:
                ln, sha = _parse_write_success_metadata_from_output(obs)
                obs = write_success_feedback_for_memory(
                    rel,
                    lines=ln,
                    sha256_short=sha,
                )
            elif tool_name == "bash" and not result.error:
                if is_bare_solution_bash:
                    obs = compress_bash_tool_output_for_memory(
                        obs,
                        tail_lines=min(
                            self._bash_success_tail_lines,
                            self._bash_success_tail_lines_solution,
                        ),
                        command=cmd_for_bash,
                        dedup_enabled=self._bash_output_dedup_apply_to_memory,
                        dedup_min_repeat=self._bash_output_dedup_min_repeat,
                        dedup_summary_prefix=self._bash_output_dedup_summary_prefix,
                    )
                elif is_repeat and sig is not None:
                    self._released_full_signatures.add(sig)
                else:
                    bash_tail_lines = _bash_success_tail_lines_for_command(
                        cmd_for_bash,
                        fallback=self._bash_success_tail_lines,
                        solution=self._bash_success_tail_lines_solution,
                        test=self._bash_success_tail_lines_test,
                        readonly=self._bash_success_tail_lines_readonly,
                        install=self._bash_success_tail_lines_install,
                    )
                    obs = compress_bash_tool_output_for_memory(
                        obs,
                        tail_lines=bash_tail_lines,
                        command=cmd_for_bash,
                        dedup_enabled=self._bash_output_dedup_apply_to_memory,
                        dedup_min_repeat=self._bash_output_dedup_min_repeat,
                        dedup_summary_prefix=self._bash_output_dedup_summary_prefix,
                    )
                    if bash_command_dumps_python_source(cmd_for_bash):
                        obs = obs + _BASH_READ_PY_SOURCE_COACHING
            elif tool_name == "read" and not result.error:
                if is_repeat and sig is not None:
                    self._released_full_signatures.add(sig)
                elif _read_args_has_explicit_range(args) and _read_args_limit(args) > 0:
                    # Explicit offset/limit reads are usually edit-oriented. Keep the
                    # requested body intact so the next turn has exact anchors.
                    obs = obs
                else:
                    replaced = False
                    if rel and self._path_matches_write_snapshot(rel):
                        try:
                            resolved = self._guard.resolve(rel)
                            if resolved.is_file():
                                text = resolved.read_text(encoding="utf-8", errors="replace")
                                cur_sha = _sha256_short_bytes(
                                    text.encode("utf-8", errors="replace"),
                                )
                                if (
                                    cur_sha
                                    and cur_sha == self._last_auto_snapshot_sha_by_path.get(rel, "")
                                ):
                                    obs = _build_snapshot_ref_output_for_memory(
                                        obs,
                                        rel=rel,
                                        sha=cur_sha,
                                        raw_id=raw_id,
                                    )
                                    replaced = True
                                elif (
                                    PurePosixPath(rel).suffix.lower()
                                    in self._write_auto_snapshot_code_extensions
                                    and len(text.splitlines()) > self._read_success_max_lines
                                ):
                                    obs = _build_read_code_map_summary_for_memory(
                                        obs,
                                        rel=rel,
                                        text=text,
                                        raw_id=raw_id,
                                    )
                                    replaced = True
                        except (ValueError, OSError) as e:
                            logger.debug("read code-map/snapshot compression skipped: %s", e)
                    if not replaced:
                        obs = compress_read_output_for_memory(
                            obs,
                            max_lines=self._read_success_max_lines,
                        )
            elif tool_name == "grep" and not result.error:
                obs = self._compress_grep_snapshot_refs_for_memory(obs, raw_id=raw_id)

        if read_overlap_redundant:
            # Tier 2 (silent read interception): if a read range was already present in
            # memory at the same file SHA, do not inject another source copy. A compact,
            # deterministic coverage summary preserves natural tool history while avoiding
            # context growth from repeated exploration.
            self._redundant_read_count_by_path[rel] = (
                self._redundant_read_count_by_path.get(rel, 0) + 1
            )
            replaced_with_coverage = False
            if read_overlap_summary:
                first_nl = obs.find("\n")
                header = obs if first_nl < 0 else obs[:first_nl]
                obs = (header.rstrip() + "\n\n" + read_overlap_summary).strip()
                replaced_with_coverage = True
            if (
                not replaced_with_coverage
                and self._write_auto_snapshot_enabled
                and rel
                and self._path_matches_write_snapshot(rel)
            ):
                try:
                    resolved = self._guard.resolve(rel)
                    if resolved.is_file():
                        text = resolved.read_text(encoding="utf-8", errors="replace")
                        sha = _sha256_short_bytes(text.encode("utf-8", errors="replace"))
                        covered = list((self._read_coverage.get(rel) or ("", []))[1])
                        requested = _requested_read_range_for_file(
                            args,
                            len(text.splitlines()),
                        )
                        first_nl = obs.find("\n")
                        header = obs if first_nl < 0 else obs[:first_nl]
                        obs = (
                            header.rstrip()
                            + "\n\n"
                            + _build_redundant_read_coverage_summary(
                                rel,
                                sha=sha,
                                requested=requested,
                                covered_ranges=covered,
                            )
                        ).strip()
                        replaced_with_coverage = True
                except (ValueError, OSError) as e:
                    logger.debug(
                        "silent read intercept: coverage summary failed (%s)", e,
                        exc_info=True,
                    )
            if not replaced_with_coverage:
                obs = obs + _READ_OVERLAP_COACHING

        if (
            self._tool_memory_compression
            and tool_name in ("write", "edit")
            and not result.error
            and rel
            and self._write_auto_snapshot_enabled
            and self._path_matches_write_snapshot(rel)
        ):
            try:
                resolved = self._guard.resolve(rel)
                if resolved.is_file():
                    # Compute current sha first; if it matches the last-injected
                    # one for this path, skip the snapshot to avoid bloating chat
                    # with redundant copies (S4 dedup).
                    try:
                        current_text = resolved.read_text(encoding="utf-8", errors="replace")
                        _cur_sha = _sha256_short_bytes(
                            current_text.encode("utf-8", errors="replace"),
                        )
                    except OSError:
                        current_text = ""
                        _cur_sha = ""
                    _prev_sha = self._last_auto_snapshot_sha_by_path.get(rel, "")
                    if _cur_sha and _cur_sha == _prev_sha:
                        logger.debug(
                            "%s auto-snapshot: skipped (sha %s unchanged for %s)",
                            tool_name,
                            _cur_sha, rel,
                        )
                    else:
                        snap = _build_write_auto_snapshot_block(
                            rel,
                            resolved,
                            max_lines=self._write_auto_snapshot_max_lines,
                            max_chars=self._write_auto_snapshot_max_chars,
                            previous_text=self._last_auto_snapshot_text_by_path.get(rel),
                            tool_name=tool_name,
                            args=args,
                            changed_context_lines=self._write_auto_snapshot_changed_context_lines,
                            symbol_body_lines=self._write_auto_snapshot_symbol_body_lines,
                        )
                        obs = (obs or "").rstrip() + "\n\n" + snap
                        if _cur_sha:
                            self._last_auto_snapshot_sha_by_path[rel] = _cur_sha
                            self._last_auto_snapshot_text_by_path[rel] = current_text
                            ranges = _numbered_source_ranges_in_text(snap)
                            if ranges:
                                self._read_coverage[rel] = (
                                    _cur_sha,
                                    _merge_read_intervals(ranges),
                                )
            except (ValueError, OSError) as e:
                logger.debug("%s auto-snapshot: skipped (%s)", tool_name, e, exc_info=True)

        if tool_name == "write" and not result.error and rel:
            n = self._write_counter_by_path.get(rel, 0) + 1
            self._write_counter_by_path[rel] = n
            if n >= 2:
                obs = obs + _WRITE_REPEAT_TO_SAME_FILE_COACHING.format(n=n, rel=rel)

        if sig is not None and not is_repeat:
            self._seen_tool_signatures[sig] = len(_all_messages(self._memory))

        return obs

    def inject_file_content(self, path: str, *, reason: str = "") -> str:
        resolved = self._guard.resolve(path)
        if not resolved.is_file():
            raise OSError(f"not a file: {path}")
        text = resolved.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        header = f"[auto-read: {reason}]\n" if reason else "[auto-read]\n"
        if len(lines) > 300:
            preview = (
                "\n".join(lines[:150])
                + f"\n... ({len(lines)} lines total) ...\n"
                + "\n".join(lines[-50:])
            )
        else:
            preview = text
        return header + preview

    async def compact(self, llm: Any, *, mid_run: bool = False) -> str:
        """Summarize chat with LLM, clear history, inject summary as a system message.

        When *mid_run* is True, the prompt targets the **same** session continuing after a
        sliding-window overflow (debugging focus; shorter handoff).

        Note: the leading user-pin (msg[0]'s head + any subsequent leading user
        messages) is **preserved verbatim** across compact — the LLM summary is
        appended *after* those pinned messages, not in place of them. This guards
        explicit user constraints (e.g. "do NOT ensemble / stack / K-fold") from
        being silently paraphrased away.
        """
        messages = _all_messages(self._memory)
        if not messages:
            return "Nothing to compact."

        # Capture leading user messages so we can re-attach them verbatim after clear().
        # By default, every leading user message — including msg[0] in full — is
        # preserved as-is. When ``msg0_compress_body`` is True, msg[0] is split at
        # the ``## Task description`` anchor: the head (policy + ensemble fusion
        # format + lite title) stays verbatim while the body is replaced with a
        # placeholder so the LLM summary can fold it in.
        pinned_leading: list[Message] = []
        for j, mm in enumerate(messages):
            if getattr(mm, "role", None) != "user":
                break
            text = _message_text_for_chars(mm)
            if j == 0 and self._msg0_compress_body:
                head, body = _split_msg0_head_body(text)
                if body:
                    pinned_leading.append(
                        _message_with_text(
                            mm,
                            head.rstrip("\n")
                            + "\n\n## Task description\n"
                            + MSG0_BODY_TRUNCATED_PLACEHOLDER,
                        ),
                    )
                else:
                    pinned_leading.append(mm)
            else:
                pinned_leading.append(mm)

        protected_raw, compact_source_messages, protected_chars = self._protected_raw_prefix_parts(
            messages,
            len(pinned_leading),
        )
        history_source = [*pinned_leading, *compact_source_messages] if protected_raw else messages
        history_parts: list[str] = []
        for m in history_source:
            role = m.role
            excerpt = _format_message_excerpt_for_compact(m)
            history_parts.append(f"[{role}]: {excerpt}")
        history_text = "\n".join(history_parts)

        # Ground-truth workspace files: chat excerpts are often truncated; this keeps compact summaries
        # aligned with the actual solution and reported metrics.
        sol_path = self._workspace / "solution.py"
        if sol_path.is_file():
            try:
                sol_code = sol_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                sol_code = ""
            if sol_code.strip():
                max_sol = 6000
                if len(sol_code) > max_sol:
                    sol_code = sol_code[:max_sol] + "\n... [truncated]"
                history_text += (
                    "\n\n[CURRENT solution.py — MUST be reflected in the summary]\n"
                    "```python\n"
                    + sol_code
                    + "\n```"
                )
        result_path = self._workspace / "result.md"
        if result_path.is_file():
            try:
                rt = result_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                rt = ""
            if rt.strip():
                history_text += "\n\n[CURRENT result.md]\n" + rt[:3000]

        # The user-pin head (everything in msg[0] before ``## Task description``) is
        # preserved verbatim by the system *outside* this summary; the LLM should not
        # repeat it, paraphrase it, or drop the constraints it contains.
        protected_guard = ""
        if protected_raw:
            protected_guard = (
                f"- {len(protected_raw)} early EDA/tool message(s) are also re-attached verbatim "
                f"after compact ({protected_chars} chars); do NOT repeat or paraphrase them.\n"
            )
        pin_guard_block = (
            "USER PIN (preserved verbatim by the system — do NOT repeat or paraphrase here):\n"
            "- The first user message's head (everything before the line `## Task description`) "
            "is re-attached verbatim after this summary; you do NOT need to include it.\n"
            + protected_guard
            + "- Treat prohibitions in that pin as **phase-scoped** unless the recent chat "
            "explicitly says otherwise: e.g. \"do **NOT** ensemble across models\" applies to "
            "**single-model training** (explore / exploit_explore), not after a later message "
            "switches the workflow to **ensemble** peer fusion.\n"
            "- Do NOT drop, soften, or rewrite explicit user constraints / prohibitions written "
            "there when they still apply to the **current** phase. If the recent chat shows the "
            "agent violating an applicable constraint, call it out in a section titled "
            "`User constraints (verbatim)` and quote each violated line word-for-word.\n\n"
        )

        if mid_run:
            compress_prompt = (
                pin_guard_block
                + "The chat history below no longer fits in the model context window. Summarize it so the "
                "**same** agent can continue debugging and improving in this session.\n\n"
                "IMPORTANT: Reflect the **current** `solution.py` (pipeline, model, key hyperparameters) "
                "and any metrics in ``[CURRENT result.md]``. Preserve recent **errors, tracebacks, and "
                "failed bash runs** verbatim enough to fix them.\n\n"
                "Use markdown headings:\n"
                "1. **Current approach** — data loading, features, model, validation.\n"
                "2. **Recent failures** — last errors, exit codes, stderr highlights.\n"
                "3. **What to do next** — concrete fixes or checks (files, commands).\n"
                "4. **User constraints (verbatim)** — quote only prohibitions from the user pin that "
                "**still apply** to the current phase (see phase-scoping note above); omit this "
                "section if none apply.\n\n"
                "Max ~600 words. Be specific (paths, function names, numbers).\n\n"
                f"{history_text}"
            )
            system_compact = (
                "You compress prior chat into a short continuation summary for the same debugging session. "
                "The user pin (msg[0] head) is re-attached verbatim by the system, so do not repeat it; "
                "do not soften or drop explicit user prohibitions."
            )
        else:
            compress_prompt = (
                pin_guard_block
                + "Summarize the following conversation into a structured handoff document for the NEXT agent "
                "that will continue this workspace. Include these sections (use markdown headings):\n\n"
                "IMPORTANT: Your summary MUST accurately reflect the **current** `solution.py` (model family, "
                "key hyperparameters, main training logic) and any metric values in ``[CURRENT result.md]`` "
                "or the chat history. Do not claim \"no modeling\" or \"not yet implemented\" if the files "
                "below show otherwise.\n\n"
                "1. **Data insights** — column types, missing patterns, key correlations, train/val split notes.\n"
                "2. **Feature engineering pipeline** — encoding, imputation, scaling, derived features; "
                "name key functions.\n"
                "3. **Model architecture** — algorithms, main hyperparameters, training setup.\n"
                "4. **What was tried and results** — approaches attempted, metric values (quick vs full if known), "
                "what failed.\n"
                "5. **Pitfalls** — bugs, leakage risks, NaNs, timeouts.\n"
                "6. **Recommended next steps** — concrete improvements.\n"
                "7. **User constraints (verbatim)** — quote only prohibitions from the user pin that "
                "**still apply** to the current phase (see phase-scoping note above); omit this "
                "section if none apply.\n\n"
                "Max ~1000 words. Be concise but specific: prefer function names, column lists, and numbers over "
                "vague prose.\n\n"
                f"{history_text}"
            )
            system_compact = (
                "You compress prior chat into a durable summary. The user pin (msg[0] head) is "
                "re-attached verbatim by the system, so do not repeat it; do not soften or drop "
                "explicit user prohibitions."
            )

        summary = await llm.ask(
            messages=[Message.user_message(compress_prompt)],
            system_msgs=[Message.system_message(system_compact)],
            stream=False,
        )
        if not (summary or "").strip():
            return "Compact failed: empty summary."

        _clear_memory_and_long_term_log(self._memory)
        # Re-attach leading user-pin verbatim (msg[0] with head preserved + body
        # placeholder, plus any subsequent leading user messages).
        for pm in pinned_leading:
            self._memory.add_message(pm)
        for pm in protected_raw:
            self._memory.add_message(pm)
        body = summary.strip()
        self._memory.add_message(
            Message.user_message(
                f"{COMPACTED_CONVERSATION_SUMMARY_MARKER}\n{body}",
            ),
        )
        if protected_raw:
            self._protected_raw_prefix_end_index = len(pinned_leading) + len(protected_raw)
        return f"Compacted {len(messages)} messages into summary ({len(summary)} chars)."
