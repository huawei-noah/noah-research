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

"""Tool-call compression and assistant message conversion for memory."""

from __future__ import annotations

import copy
import json
import logging
import re
from types import SimpleNamespace
from typing import Any

from deepcraft_core import Message

logger = logging.getLogger("scienceflow")


def _transcript_has_reasoning_content(messages: list[Any]) -> bool:
    """Return True when the transcript already uses assistant reasoning_content."""
    for msg in messages:
        rc = getattr(msg, "reasoning_content", None)
        if isinstance(rc, str) and rc.strip():
            return True
    return False


def _synthetic_reasoning_content_for_transcript(
    messages: list[Any],
    purpose: str,
) -> str | None:
    """Provide minimal synthetic reasoning only for transcripts that already require it."""
    if not _transcript_has_reasoning_content(messages):
        return None
    clean = re.sub(r"\s+", " ", str(purpose or "synthetic replay")).strip()
    return f"Synthetic replay for {clean}." if clean else "Synthetic replay."


def scrub_last_assistant_write_tool_call_content(
    memory: Any,
    *,
    tool_call_id: str,
    note: str | None = None,
) -> None:
    """Compatibility hook: raw memory is no longer rewritten after bad write calls.

    Tool-call storage is now the truthful event log. Risky historical ``write`` payloads are
    hidden only by the LLM-visible projection in ``MemoryContextManager``.
    """
    return

_THOUGHT_TOOL_PARAM: dict[str, Any] = {
    "type": "string",
    "description": (
        "One to two concise sentences stating what this tool call will do and why. "
        "Do not write multi-paragraph plans."
    ),
}


def inject_thought_into_tool_params(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deep-copy *tools* and add required ``thought`` to each function's JSON Schema."""

    out = copy.deepcopy(tools)
    for item in out:
        if item.get("type") != "function":
            continue
        fn = item.get("function") or {}
        params = fn.get("parameters")
        if not isinstance(params, dict):
            continue
        props = params.setdefault("properties", {})
        if not isinstance(props, dict):
            continue
        props["thought"] = dict(_THOUGHT_TOOL_PARAM)
        req = params.setdefault("required", [])
        if not isinstance(req, list):
            req = []
            params["required"] = req
        if "thought" not in req:
            req.append("thought")
    return out


def _compress_edit_payload_for_memory(value: str, *, key: str) -> str:
    head_tail = 60
    if len(value) <= head_tail * 2 + 80:
        return value
    if key == "old_str":
        marker = (
            f"# <<<MEMORY_COMPRESSED: edit.old_str={len(value.encode('utf-8'))}bytes "
            "— NOT CODE — real content on disk>>>"
        )
    else:
        omitted = max(0, len(value) - head_tail * 2)
        marker = f"# <<<MEMORY_COMPRESSED: omitted {omitted} chars from {key} — NOT CODE>>>"
    return value[:head_tail] + "\n" + marker + "\n" + value[-head_tail:]


def _compress_tool_call_for_memory(tc: Any, *, enabled: bool) -> Any:
    """Return a shallow copy of *tc* with non-essential JSON args stripped.

    Never raises: failures fall back to *tc* unchanged.

    Code-bearing write/bash arguments are assistant-authored output. Once they
    become input history on the next LLM call, they are eligible for provider
    KV-cache reuse. Very large edit anchors are summarized because the post-edit
    snapshot is the source of truth for historical context.
    """
    if not enabled:
        return tc
    try:
        fn = getattr(tc, "function", None)
        if fn is None:
            return tc
        name = getattr(fn, "name", "") or ""
        raw_args = fn.arguments if isinstance(fn.arguments, str) else str(fn.arguments)
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return tc
        changed = args.pop("config", None) is not None
        if args.pop("thought", None) is not None:
            changed = True
        if name == "edit":
            for key in ("old_str", "new_str"):
                val = args.get(key)
                if isinstance(val, str):
                    new_val = _compress_edit_payload_for_memory(val, key=key)
                    if new_val != val:
                        args[key] = new_val
                        changed = True
        if not changed:
            return tc
        try:
            new_args = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            return tc
        new_fn = SimpleNamespace(
            name=name,
            arguments=new_args,
        )
        return SimpleNamespace(
            id=getattr(tc, "id", ""),
            type=getattr(tc, "type", "function"),
            function=new_fn,
        )
    except Exception:
        logger.debug("_compress_tool_call_for_memory failed", exc_info=True)
        return tc


def _assistant_message_from_api(assistant_msg: Any) -> Message:
    """Build a deepcraft ``Message`` from an OpenAI-style completion message."""
    tool_calls = getattr(assistant_msg, "tool_calls", None) or []
    content = getattr(assistant_msg, "content", None) or ""
    rc = getattr(assistant_msg, "reasoning_content", None)
    if not tool_calls:
        return Message.assistant_message(
            content.strip() or "(empty assistant message)",
            reasoning_content=rc,
        )
    formatted: list[dict[str, Any]] = []
    for tc in tool_calls:
        tc = _compress_tool_call_for_memory(tc, enabled=True)
        fn = tc.function
        args = fn.arguments if isinstance(fn.arguments, str) else json.dumps(fn.arguments)
        model_dump = getattr(fn, "model_dump", None)
        if callable(model_dump):
            fd = model_dump()
        else:
            fd = {"name": fn.name, "arguments": args}
        formatted.append({"id": tc.id, "type": "function", "function": fd})
    return Message(
        role="assistant",
        content=content,
        tool_calls=formatted,
        reasoning_content=rc,
    )


def _tool_call_names_from_list(tool_calls: list[Any]) -> list[str]:
    names: list[str] = []
    for tc in tool_calls:
        try:
            fn = tc.function
            names.append(getattr(fn, "name", None) or "")
        except Exception:
            names.append("")
    return names


def _tool_call_args_from_tc(tc: Any) -> dict[str, Any]:
    """Parse JSON tool arguments from an OpenAI-style tool_call object."""
    fn = tc.function
    raw_args = fn.arguments if isinstance(fn.arguments, str) else str(fn.arguments)
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        args = {}
    args.pop("config", None)
    return args


_SUPERSEDED_MARKER = (
    "[superseded by a later write/edit; use read tool for current file contents]"
)


def _norm_rel_path_snapshot(p: str) -> str:
    return (p or "").strip().replace("\\", "/").lstrip("./")


def extract_path_from_tool_feedback_text(content: str, tool_name: str) -> str | None:
    """Parse relative path from first line of write/edit tool feedback stored in memory."""
    if not content or not isinstance(content, str):
        return None
    first = content.split("\n", 1)[0].strip()
    if _SUPERSEDED_MARKER in first:
        return None
    if tool_name == "write":
        m = re.match(r"^Written\s+(.+?)\s+\(\d+\s+lines,", first)
        if m:
            return m.group(1).strip()
        m2 = re.match(r"^File `(.+?)` written successfully", first)
        if m2:
            return m2.group(1).strip()
    elif tool_name == "edit":
        m = re.match(r"^Edited\s+([^:]+):\s*", first)
        if m:
            return m.group(1).strip()
    return None


def collapse_stale_file_snapshots_before_add(
    memory: Any,
    *,
    tool_name: str,
    rel_path: str,
    enabled: bool,
) -> None:
    """Before appending a new write/edit tool message, collapse older full snapshots for the same file."""
    if not enabled or tool_name not in ("write", "edit"):
        return
    storage = getattr(getattr(memory, "chat_history_memory", None), "storage", None)
    memory_list = getattr(storage, "memory_list", None)
    if not isinstance(memory_list, list):
        return
    target = _norm_rel_path_snapshot(rel_path)
    if not target:
        return
    for rec in memory_list:
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "tool":
            continue
        name = msg.get("name") or ""
        if name not in ("write", "edit"):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if _SUPERSEDED_MARKER in content:
            continue
        p = extract_path_from_tool_feedback_text(content, name)
        if not p:
            continue
        if _norm_rel_path_snapshot(p) != target:
            continue
        first = content.split("\n", 1)[0].strip()
        if "\n" not in content.rstrip():
            continue
        msg["content"] = f"{first} {_SUPERSEDED_MARKER}"


def maybe_collapse_stale_snapshots(
    memory: Any,
    tool_name: str,
    args: dict[str, Any],
    tool_result: Any,
    *,
    enabled: bool,
) -> None:
    """Call :func:`collapse_stale_file_snapshots_before_add` when a write/edit succeeded."""
    err = getattr(tool_result, "error", None)
    if err:
        return
    if tool_name not in ("write", "edit"):
        return
    rel = str(args.get("path") or "").replace("\\", "/").lstrip("/")
    collapse_stale_file_snapshots_before_add(
        memory,
        tool_name=tool_name,
        rel_path=rel,
        enabled=enabled,
    )
