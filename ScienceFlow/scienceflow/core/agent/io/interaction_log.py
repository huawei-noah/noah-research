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

"""Formatting and caps for workspace interaction.log."""

from __future__ import annotations

import json
from typing import Any

from scienceflow.core.agent.shared.constants import (
    _WS_LOG_BODY_CAP,
    _WS_LOG_TOOL_CALL_SINGLE_CAP,
    _WS_LOG_TOOL_RESULT_CAP,
)
from scienceflow.core.agent.tools.bash_command_classifier import classify_bash_command
from scienceflow.core.agent.io.interaction_log_policy import InteractionLogPolicy
from scienceflow.utils.workspace_interaction_log import collapse_consecutive_repeated_lines_in_text

# Max lines kept per [user] / [assistant] / [assistant-thinking] line in interaction.log.
_WS_LOG_MAX_LINES = 30

# BashTool appends stderr after this marker (see ``bash_tool.py``). Interaction log mirrors stdout only.
_BASH_STDERR_SECTION_SEP = "\n[stderr]\n"


def _strip_bash_stderr_section_for_interaction_log(text: str) -> str:
    """Drop BashTool stderr tail so ``interaction.log`` shows stdout only (LLM still sees full output)."""
    if _BASH_STDERR_SECTION_SEP in text:
        return text.split(_BASH_STDERR_SECTION_SEP, 1)[0]
    return text
def truncate_for_interaction_log(text: str, max_lines: int = _WS_LOG_MAX_LINES) -> str:
    """Keep first *max_lines* lines for interaction.log; append stats for the rest."""
    if not text:
        return text
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return text
    kept = "".join(lines[:max_lines])
    omitted_lines = len(lines) - max_lines
    omitted_chars = sum(len(ln) for ln in lines[max_lines:])
    return (
        f"{kept}"
        f"...[truncated: {omitted_lines} more lines, {omitted_chars} chars omitted; "
        f"total {len(lines)} lines, {len(text)} chars]"
    )


def _interaction_log_write_body_placeholder(n: int, sha16: str = "") -> str:
    """One-line stand-in for write ``content`` in workspace ``interaction.log`` only (not valid Python).

    Starts with ``WRITE_OK`` (positive signal) and embeds sha256 short-hash so the LLM can
    visually distinguish consecutive writes. Marked NOT FILE CONTENT to prevent mimicry.
    """
    sha_part = f" -> sha256:{sha16}" if sha16 else ""
    return f"[WRITE_OK memory-compression: {n} chars{sha_part}; full body applied to disk; NOT file content]"


def _memory_edit_old_str_placeholder(n: int) -> str:
    """Placeholder for edit ``old_str`` in memory + interaction.log."""

    return f"[old text omitted - {n} chars]"


def _bash_command_meta_placeholder(n: int) -> str:
    """Short meta for bash ``command`` in interaction.log (avoid ``<N chars>`` pattern)."""

    return f"[bash command omitted - {n} chars]"


def _preview_bash_command(cmd: str, max_chars: int | None) -> str:
    """Inline bash command preview when not using ``tool_call_full`` bodies."""
    if max_chars is None:
        return cmd
    if len(cmd) <= max_chars:
        return cmd
    return cmd[:max_chars] + f"...[truncated, total {len(cmd)} chars]"


def _truncate_interaction_body(s: str, full: bool, cap: int) -> str:
    if full or len(s) <= cap:
        return s
    return s[:cap] + f"\n...[truncated, total {len(s)} chars]"


def format_tool_call_lines_for_interaction_log(
    name: str,
    args: dict[str, Any],
    full: bool | None = None,
    *,
    policy: InteractionLogPolicy | None = None,
) -> list[str]:
    """Lines to append to workspace interaction.log for one tool call (may contain newlines)."""
    if policy is None:
        policy = InteractionLogPolicy.from_legacy_full(bool(full))
    tc_full = policy.tool_call_full
    lines: list[str] = []
    a = dict(args)

    if name == "write" and isinstance(a.get("content"), str):
        import hashlib
        content = a.pop("content")
        n = len(content)
        sha16 = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        meta = dict(a)
        meta["content"] = _interaction_log_write_body_placeholder(n, sha16)
        try:
            meta_line = f"[tool-call] {name} {json.dumps(meta, ensure_ascii=False)}"
        except (TypeError, ValueError):
            meta_line = f"[tool-call] {name} {meta!r}"
        if not tc_full and len(meta_line) > _WS_LOG_TOOL_CALL_SINGLE_CAP:
            meta_line = meta_line[:_WS_LOG_TOOL_CALL_SINGLE_CAP] + "..."
        lines.append(meta_line)
        return lines

    if name == "edit":
        has_old = isinstance(a.get("old_str"), str)
        has_new = isinstance(a.get("new_str"), str)
        if has_old or has_new:
            old_s = a.pop("old_str", "") if has_old else ""
            new_s = a.pop("new_str", "") if has_new else ""
            meta = dict(a)
            if has_old:
                meta["old_str"] = _memory_edit_old_str_placeholder(len(old_s))
            if has_new:
                meta["new_str"] = f"<{len(new_s)} chars>"
            try:
                meta_line = f"[tool-call] {name} {json.dumps(meta, ensure_ascii=False)}"
            except (TypeError, ValueError):
                meta_line = f"[tool-call] {name} {meta!r}"
            if not tc_full and len(meta_line) > _WS_LOG_TOOL_CALL_SINGLE_CAP:
                meta_line = meta_line[:_WS_LOG_TOOL_CALL_SINGLE_CAP] + "..."
            lines.append(meta_line)
            if tc_full:
                if has_old:
                    body = _truncate_interaction_body(old_s, True, _WS_LOG_BODY_CAP)
                    lines.append(
                        f"[tool-call-body] {name}.old_str ({len(old_s)} chars)\n"
                        f"---- begin ----\n{body}\n---- end ----",
                    )
                if has_new:
                    body = _truncate_interaction_body(new_s, True, _WS_LOG_BODY_CAP)
                    lines.append(
                        f"[tool-call-body] {name}.new_str ({len(new_s)} chars)\n"
                        f"---- begin ----\n{body}\n---- end ----",
                    )
            return lines

    if name == "bash" and isinstance(a.get("command"), str):
        cmd = a.pop("command")
        if cmd:
            n = len(cmd)
            meta = dict(a)
            meta["bash_kind"] = classify_bash_command(cmd)
            if tc_full:
                meta["command"] = _bash_command_meta_placeholder(n)
            else:
                meta["command"] = _preview_bash_command(cmd, policy.bash_preview_max_chars)
            try:
                meta_line = f"[tool-call] {name} {json.dumps(meta, ensure_ascii=False)}"
            except (TypeError, ValueError):
                meta_line = f"[tool-call] {name} {meta!r}"
            if not tc_full and len(meta_line) > _WS_LOG_TOOL_CALL_SINGLE_CAP:
                meta_line = meta_line[:_WS_LOG_TOOL_CALL_SINGLE_CAP] + "..."
            lines.append(meta_line)
            if tc_full:
                body = _truncate_interaction_body(cmd, True, _WS_LOG_BODY_CAP)
                lines.append(
                    f"[tool-call-body] {name}.command ({n} chars)\n"
                    f"---- begin ----\n{body}\n---- end ----",
                )
            return lines

    try:
        one = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        one = str(args)
    line = f"[tool-call] {name} {one}"
    if not tc_full and len(line) > _WS_LOG_TOOL_CALL_SINGLE_CAP:
        line = line[:_WS_LOG_TOOL_CALL_SINGLE_CAP] + "..."
    return [line]


def format_tool_result_for_interaction_log(
    tool_out: str,
    full: bool | None = None,
    *,
    policy: InteractionLogPolicy | None = None,
) -> str:
    if policy is None:
        policy = InteractionLogPolicy.from_legacy_full(bool(full))
    if policy.tool_result_unlimited:
        return tool_out
    if policy.tool_result_max_lines is None:
        if len(tool_out) <= _WS_LOG_TOOL_RESULT_CAP:
            return tool_out
        return (
            tool_out[:_WS_LOG_TOOL_RESULT_CAP]
            + f"...[truncated, total {len(tool_out)} chars]"
        )
    # Line cap first, then character cap (does not affect bash-stream mirror path).
    out = truncate_for_interaction_log(
        tool_out,
        max_lines=policy.tool_result_max_lines,
    )
    if len(out) <= _WS_LOG_TOOL_RESULT_CAP:
        return out
    return (
        out[:_WS_LOG_TOOL_RESULT_CAP]
        + f"...[truncated, total {len(tool_out)} chars]"
    )


def tool_result_text_for_interaction_log(
    tool_name: str,
    tool_out: str,
    policy: InteractionLogPolicy,
) -> str:
    """Format tool stdout for ``[tool-result]``; collapses write/edit when policy is non-verbose."""
    if tool_name == "bash":
        tool_out = _strip_bash_stderr_section_for_interaction_log(tool_out)
        if policy.bash_output_dedup_enabled:
            tool_out = collapse_consecutive_repeated_lines_in_text(
                tool_out,
                min_repeat=int(policy.bash_output_dedup_min_repeat),
                summary_prefix=policy.bash_output_dedup_summary_prefix,
            )
    if tool_name in ("write", "edit") and not policy.write_edit_tool_result_verbose:
        line = tool_out.splitlines()[0] if tool_out else ""
        if len(line) > 400:
            line = line[:400] + "..."
        return format_tool_result_for_interaction_log(line, policy=policy)
    return format_tool_result_for_interaction_log(tool_out, policy=policy)
