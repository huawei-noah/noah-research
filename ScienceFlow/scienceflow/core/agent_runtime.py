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

from __future__ import annotations

import logging
import json
import re
from pathlib import Path
from typing import Any

from deepcraft_core import Memory, Message, OnlineLLM
from deepcraft_core.memory import BaseChatHistoryMemory, BaseContextCreator
from deepcraft_core.storage.kv_storage import JsonKeyValueStorage

from scienceflow.config.settings import StageConfig
from scienceflow.core.key_pool import PooledLLM, parse_key_env
from scienceflow.core.llm_http import llm_extra_client_kwargs

logger = logging.getLogger("scienceflow")


def create_agent_memory(memory_dir: Path, agent_name: str, max_messages: int) -> Memory:
    agent_dir = memory_dir / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)
    return Memory(
        max_messages=max_messages,
        chat_history_memory=BaseChatHistoryMemory(
            storage=JsonKeyValueStorage(path=str(agent_dir / "short_term.json"), mode="w")
        ),
        context_creator=BaseContextCreator(),
        long_term_log=str(agent_dir / "long_term.jsonl"),
    )


def load_agent_memory(
    memory_dir: Path,
    agent_name: str,
    max_messages: int,
    *,
    recent_rounds: int = 0,
) -> Memory:
    """Load existing agent memory from disk (clone-continue mode).

    Parameters
    ----------
    recent_rounds:
        If > 0, keep only the last *recent_rounds* tool-call rounds
        (each round ≈ 1 assistant + 1 tool message). 0 = keep all.
    """
    agent_dir = memory_dir / agent_name
    if not agent_dir.is_dir():
        return create_agent_memory(memory_dir, agent_name, max_messages)
    _repair_agent_memory_short_term(agent_dir, max_messages=max_messages)
    memory = Memory(
        max_messages=max_messages,
        chat_history_memory=BaseChatHistoryMemory(
            storage=JsonKeyValueStorage(path=str(agent_dir / "short_term.json"), mode="a")
        ),
        context_creator=BaseContextCreator(),
        long_term_log=str(agent_dir / "long_term.jsonl"),
    )
    if recent_rounds > 0:
        _prune_memory_to_recent_rounds(memory, recent_rounds)
    return memory



def select_prefix_safe_agent_memory_records(
    agent_dir: Path,
    *,
    max_messages: int,
) -> list[dict[str, Any]]:
    """Return the best prefix-safe snapshot from an agent memory directory.

    ``short_term.json`` is bounded by ``max_messages`` and can lose the leading
    task/user prefix. ``long_term.jsonl`` is usually a better recovery source,
    but it is not guaranteed to exist on every path. This helper prefers the
    source that still has a leading system/user prefix and then trims it while
    preserving that prefix.
    """
    agent_dir = Path(agent_dir)
    short_records = _read_memory_jsonl(agent_dir / "short_term.json")
    long_records = _read_memory_jsonl(agent_dir / "long_term.jsonl")
    short_repaired = _filter_agentic_route_memory_records(
        _repair_memory_record_window(short_records, max_messages=max_messages),
    )
    long_repaired = _filter_agentic_route_memory_records(
        _repair_memory_record_window(long_records, max_messages=max_messages),
    )
    source, records = max(
        [("short_term", short_repaired), ("long_term", long_repaired)],
        key=lambda item: _memory_record_score(item[1]),
    )
    if source == "long_term":
        logger.info(
            "[memory] using long_term.jsonl as prefix-safe recovery snapshot: %s records=%d",
            agent_dir,
            len(records),
        )
    return list(records)


def write_agent_memory_record_files(
    agent_dir: Path,
    records: list[dict[str, Any]],
    *,
    write_long_term: bool = False,
) -> None:
    """Persist raw chat-history records for an agent memory directory."""
    agent_dir = Path(agent_dir)
    _write_memory_jsonl(agent_dir / "short_term.json", records)
    if write_long_term:
        _write_memory_jsonl(agent_dir / "long_term.jsonl", records)


def _repair_agent_memory_short_term(agent_dir: Path, *, max_messages: int) -> None:
    short_path = Path(agent_dir) / "short_term.json"
    before = _read_memory_jsonl(short_path)
    after = select_prefix_safe_agent_memory_records(agent_dir, max_messages=max_messages)
    if before == after:
        return
    _write_memory_jsonl(short_path, after)
    logger.info(
        "[memory] repaired short_term prefix window: %s before=%d after=%d first_role=%s",
        short_path,
        len(before),
        len(after),
        _record_role(after[0]) if after else "",
    )


def _read_memory_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("[memory] failed reading %s", path, exc_info=True)
        return []
    if not text.strip():
        return []
    records: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("[memory] bad jsonl row skipped in %s", path)
            continue
        if isinstance(obj, dict):
            records.append(obj)
        elif isinstance(obj, list):
            records.extend(x for x in obj if isinstance(x, dict))
    if records:
        return _dedupe_memory_records(records)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, list):
        return _dedupe_memory_records([x for x in obj if isinstance(x, dict)])
    if isinstance(obj, dict):
        return [obj]
    return []


def _write_memory_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(
        json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
        for rec in records
    )
    path.write_text(data, encoding="utf-8")


def _dedupe_memory_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in records:
        key = str(rec.get("uuid") or "")
        if not key:
            try:
                key = json.dumps(rec.get("message", rec), sort_keys=True, default=str)
            except TypeError:
                key = str(rec)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _record_message_dict(record: dict[str, Any]) -> dict[str, Any]:
    msg = record.get("message")
    if isinstance(msg, dict):
        return msg
    return record


def _record_role(record: dict[str, Any]) -> str:
    msg = _record_message_dict(record)
    return str(msg.get("role") or record.get("role") or "").strip()


def _record_content(record: dict[str, Any]) -> str:
    msg = _record_message_dict(record)
    content = msg.get("content")
    return content if isinstance(content, str) else str(content or "")


def _record_has_tool_calls(record: dict[str, Any]) -> bool:
    msg = _record_message_dict(record)
    calls = msg.get("tool_calls") or record.get("tool_calls")
    return bool(calls)


def _is_agentic_route_prompt_record(record: dict[str, Any]) -> bool:
    if _record_role(record) != "user":
        return False
    text = _record_content(record)
    markers = (
        "You returned pure text without a tool call",
        "Make the second-stage route decision now",
        "missing a parseable `## Next Search Decision`",
        "previous loop-local route reply is missing parseable JSON",
        "says `exit_current_branch`",
    )
    return any(marker in text for marker in markers)


def _is_agentic_route_response_record(record: dict[str, Any]) -> bool:
    if _record_role(record) != "assistant":
        return False
    if _record_has_tool_calls(record):
        return False
    text = _record_content(record)
    if not text.strip():
        return False
    route_tokens = (
        '"action"',
        "'action'",
        "rewind_to_step",
        "rewind_to_node",
        "continue_current",
        "exit_current_branch",
        "new_branch",
        "## Next Search Decision",
    )
    return any(token in text for token in route_tokens)


def _filter_agentic_route_memory_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    out: list[dict[str, Any]] = []
    skip_route_response = False
    for rec in records:
        if _is_agentic_route_prompt_record(rec):
            skip_route_response = True
            continue
        if skip_route_response:
            if _is_agentic_route_response_record(rec):
                skip_route_response = False
                continue
            skip_route_response = False
        out.append(rec)
    return out


def _record_contains_l0_anchor(record: dict[str, Any]) -> bool:
    try:
        return "l0_anchor_" in json.dumps(record, ensure_ascii=False, default=str)
    except TypeError:
        return "l0_anchor_" in str(record)


def _leading_task_prefix_len(records: list[dict[str, Any]]) -> int:
    end = 0
    for rec in records:
        if _record_role(rec) in {"system", "user"}:
            end += 1
            continue
        break
    return end


def _drop_to_first_task_prefix(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    if _record_role(records[0]) in {"system", "user"}:
        return records
    for idx, rec in enumerate(records):
        if _record_role(rec) in {"system", "user"}:
            return records[idx:]
    out = list(records)
    while out and _record_role(out[0]) == "tool":
        out.pop(0)
    return out


def _repair_memory_record_window(
    records: list[dict[str, Any]],
    *,
    max_messages: int,
) -> list[dict[str, Any]]:
    records = _drop_to_first_task_prefix(_dedupe_memory_records(list(records)))
    if not records:
        return []
    if max_messages <= 0 or len(records) <= max_messages:
        return records
    prefix_end = _leading_task_prefix_len(records)
    if prefix_end <= 0:
        kept = records[-max_messages:]
        while kept and _record_role(kept[0]) == "tool":
            kept = kept[1:]
        return kept
    tail_budget = max(0, max_messages - prefix_end)
    if tail_budget <= 0:
        return records[:max_messages]
    suffix = records[prefix_end:]
    start = max(0, len(suffix) - tail_budget)
    kept = suffix[start:]
    while start > 0 and kept and _record_role(kept[0]) == "tool":
        start -= 1
        kept = suffix[start:]
    while kept and _record_role(kept[0]) == "tool":
        kept = kept[1:]
    return records[:prefix_end] + kept


def _memory_record_score(records: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    if not records:
        return (0, 0, 0, 0)
    first_role = _record_role(records[0])
    starts_with_task = 1 if first_role in {"system", "user"} else 0
    has_user = 1 if any(_record_role(r) == "user" for r in records) else 0
    has_l0 = 1 if any(_record_contains_l0_anchor(r) for r in records) else 0
    not_tool_start = 1 if first_role != "tool" else 0
    return (starts_with_task, has_l0, has_user + not_tool_start, len(records))


_WSP_ABS_PREFIX_RE = re.compile(r"/\S+/wsp/[0-9a-f]{32}/")


def _sanitize_wsp_paths_in_text(text: str) -> str:
    """Replace ``/.../wsp/<32hex>/`` with ``./`` (clone-continue path hygiene)."""
    if not text:
        return text
    return _WSP_ABS_PREFIX_RE.sub("./", text)


def rewrite_workspace_abs_to_relative(text: str, workspace_abs: str | None) -> str:
    """Replace runtime-absolute *workspace* prefix with ``./`` (teleport path hygiene).

    Used at tool-result emission to keep LLM-visible paths stable as the
    underlying node directory rotates. Combined with the historical
    ``_sanitize_wsp_paths_in_text`` (which still mops up ``wsp/<32hex>``
    leftovers in inherited memory).

    Empty / falsy ``workspace_abs`` is a no-op so the helper is safe to call
    unconditionally from teleport-aware tool execution paths.
    """
    if not text or not workspace_abs:
        return text
    abs_str = str(workspace_abs).rstrip("/")
    if not abs_str or abs_str == "/" or abs_str not in text:
        return text
    text = text.replace(abs_str + "/", "./")
    text = text.replace(abs_str, ".")
    return text


def _sanitize_message_wsp_paths(m: Message) -> Message:
    """Rewrite absolute workspace paths inside one message (content + tool call JSON)."""
    from deepcraft_core.tool.base import Function as ToolFunction

    updates: dict[str, Any] = {}
    if isinstance(m.content, str):
        c = _sanitize_wsp_paths_in_text(m.content)
        if c != m.content:
            updates["content"] = c
    if m.tool_calls:
        new_tcs = []
        tc_changed = False
        for tc in m.tool_calls:
            args = _sanitize_wsp_paths_in_text(tc.function.arguments)
            if args != tc.function.arguments:
                tc_changed = True
                new_tc = tc.model_copy(
                    update={
                        "function": ToolFunction(
                            name=tc.function.name,
                            arguments=args,
                        )
                    }
                )
                new_tcs.append(new_tc)
            else:
                new_tcs.append(tc)
        if tc_changed:
            updates["tool_calls"] = new_tcs

    if not updates:
        return m
    return m.model_copy(update=updates)


def sanitize_inherited_memory_workspace_paths(memory: Memory) -> None:
    """Replace absolute ``.../wsp/<uuid>/...`` paths with ``./...`` in all chat messages.

    Run after loading clone-inherited memory so the model does not copy parent node paths
    from stored history (assistant tool_calls and tool outputs).
    """
    all_records = memory.chat_history_memory.retrieve(window_size=None)
    if not all_records:
        return
    messages = [cr.memory_record.message for cr in all_records]
    out: list[Message] = []
    changed = False
    for m in messages:
        new_m = _sanitize_message_wsp_paths(m)
        if new_m is not m:
            changed = True
        out.append(new_m)
    if not changed:
        return
    memory.chat_history_memory.storage.clear()
    for m in out:
        memory.add_message(m)


def _prune_memory_to_recent_rounds(memory: Memory, rounds: int) -> None:
    """Keep only the last *rounds* tool-call rounds in memory.

    A "round" ≈ 1 assistant message (with tool_calls) + 1 tool message.
    Also preserves leading system/user messages (task instructions).
    """
    all_records = memory.chat_history_memory.retrieve(window_size=None)
    if not all_records:
        return
    messages = [cr.memory_record.message for cr in all_records]

    # Find prefix: leading system + user messages (task setup)
    prefix_end = 0
    for i, m in enumerate(messages):
        if m.role in ("system", "user"):
            prefix_end = i + 1
        else:
            break

    suffix = messages[prefix_end:]
    keep_count = rounds * 2  # assistant + tool per round
    if len(suffix) <= keep_count:
        return  # nothing to prune

    start = len(suffix) - keep_count
    kept = suffix[start:]

    # Slice boundaries can land on a ``tool`` message (e.g. odd number of trailing
    # assistant messages after the last tool). OpenAI-compatible APIs reject
    # ``role=tool`` without a preceding assistant message that issued ``tool_calls``.
    while start > 0 and kept and getattr(kept[0], "role", None) == "tool":
        start -= 1
        kept = suffix[start:]

    # If the suffix itself begins with orphan tool rows (corrupt / truncated history),
    # drop them so the first message after prefix is never ``tool``.
    while kept and getattr(kept[0], "role", None) == "tool":
        kept = kept[1:]

    # Rebuild memory storage
    memory.chat_history_memory.storage.clear()
    for m in messages[:prefix_end] + kept:
        memory.add_message(m)


def _build_llm(stage: StageConfig) -> PooledLLM | OnlineLLM:
    """Create an LLM from a StageConfig (env vars already injected by _apply_env)."""
    shared = dict(
        model=stage.model,
        max_tokens=stage.max_tokens,
        frequency_penalty=stage.frequency_penalty,
        stream=True,
        tracker=True,
        **llm_extra_client_kwargs(stage),
    )

    if stage.api_keys:
        endpoints = parse_key_env(
            api_keys_csv=",".join(stage.api_keys),
            base_urls_csv=",".join(stage.base_urls) if stage.base_urls else None,
            fallback_url=stage.base_url,
        )
        endpoint_models = [str(x).strip() for x in (stage.models or []) if str(x).strip()]
        if endpoint_models:
            if len(endpoint_models) == 1 and len(endpoints) > 1:
                endpoint_models = endpoint_models * len(endpoints)
            elif len(endpoint_models) != len(endpoints):
                endpoint_models = [endpoint_models[i % len(endpoint_models)] for i in range(len(endpoints))]
        logger.info(
            f"[LLM] {stage.model}: PooledLLM with {len(endpoints)} keys"
            + (f", {len(set(endpoint_models))} model(s)" if endpoint_models else "")
        )
        return PooledLLM.from_endpoints(
            endpoints=endpoints,
            endpoint_models=endpoint_models or None,
            routing_mode=stage.api_routing_mode,
            sticky_id=stage.api_sticky_id,
            sticky_primary_index=stage.api_sticky_primary_index,
            rate_limit_cooldown_sec=stage.api_rate_limit_cooldown_sec,
            connection_cooldown_sec=stage.api_connection_cooldown_sec,
            **shared,
        )

    return OnlineLLM(base_url=stage.base_url, api_key=stage.api_key, **shared)
