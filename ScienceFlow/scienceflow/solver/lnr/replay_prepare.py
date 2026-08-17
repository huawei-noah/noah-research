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

"""Prepare isolated LNR tool-call replay directories.

This module intentionally does not execute the replay. It copies a saved run or
bad-case directory, prefers the copied long-term memory as the historical source,
materializes the truncated replay state into short-term memory, and records
enough metadata for a later resume.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scienceflow.solver.lnr.resume.memory_state import inspect_resume_memory_messages


class ReplayPrepareError(RuntimeError):
    """Raised when a replay directory cannot be prepared safely."""


@dataclass(frozen=True)
class ReplayPrepareResult:
    source: str
    output: str
    memory_path: str
    memory_source_path: str
    memory_source_kind: str
    short_term_path: str
    long_term_path: str
    selected_message_index: int
    tool_call_id: str
    tool_name: str
    arguments_json: str
    command: str
    records_before: int
    records_after: int
    manifest_path: str | None = None
    patched_workspace_base: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayMemoryFiles:
    source_path: Path
    source_kind: str
    short_term_path: Path
    long_term_path: Path


def prepare_lnr_replay(
    source: str | Path,
    output: str | Path,
    *,
    message_index: int | None = None,
    tool_call_id: str | None = None,
    memory_path: str | Path | None = None,
    manifest_name: str = "replay.yaml",
    patch_manifest: bool = True,
) -> ReplayPrepareResult:
    """Copy ``source`` to ``output`` and truncate copied memory to a tool call.

    Exactly one selector is required:

    - ``message_index``: zero-based memory record index
    - ``tool_call_id``: id of the target tool call

    By default, the historical source is ``long_term.jsonl`` when available.
    The replay entry is always materialized into ``short_term.json`` because
    that is what runtime resume reads. ``long_term.jsonl`` is synchronized to
    the same truncation for audit consistency.

    The selected memory record must be an assistant message with exactly one
    tool call. This matches the currently supported LNR resume shape.
    """

    source_path = Path(source).expanduser().resolve(strict=False)
    output_path = Path(output).expanduser().resolve(strict=False)
    _validate_selector(message_index=message_index, tool_call_id=tool_call_id)
    if not source_path.is_dir():
        raise ReplayPrepareError(f"source is not a directory: {source_path}")
    if output_path.exists():
        raise ReplayPrepareError(f"output already exists: {output_path}")

    shutil.copytree(source_path, output_path)

    memory_files = _resolve_memory_files(
        output_path,
        memory_path=memory_path,
        source_path=source_path,
    )
    records = _read_jsonl(memory_files.source_path)
    selected_index = _select_record_index(
        records,
        message_index=message_index,
        tool_call_id=tool_call_id,
    )
    selected_message = _record_message(records[selected_index])
    calls = _message_tool_calls(selected_message)
    if len(calls) != 1:
        raise ReplayPrepareError(
            f"selected assistant message must contain exactly one tool call; found {len(calls)}",
        )
    call = calls[0]
    if tool_call_id and _tool_call_id(call) != tool_call_id:
        raise ReplayPrepareError(
            f"selected tool_call_id mismatch: expected {tool_call_id!r}, got {_tool_call_id(call)!r}",
        )

    truncated = records[: selected_index + 1]
    resume_state = inspect_resume_memory_messages(truncated)
    if resume_state.action != "execute_pending_tool":
        raise ReplayPrepareError(
            "truncated memory is not a single pending tool-call state: "
            f"{resume_state.action} ({resume_state.reason})",
        )

    _write_jsonl_atomic(memory_files.short_term_path, truncated)
    _write_jsonl_atomic(memory_files.long_term_path, truncated)

    manifest_path: Path | None = None
    patched_workspace_base: str | None = None
    if patch_manifest:
        manifest_path = output_path / manifest_name
        if manifest_path.exists():
            patched_workspace_base = str(output_path / "run")
            _patch_workspace_base(manifest_path, patched_workspace_base)
        else:
            manifest_path = None

    result = ReplayPrepareResult(
        source=str(source_path),
        output=str(output_path),
        memory_path=str(memory_files.short_term_path),
        memory_source_path=str(memory_files.source_path),
        memory_source_kind=memory_files.source_kind,
        short_term_path=str(memory_files.short_term_path),
        long_term_path=str(memory_files.long_term_path),
        selected_message_index=selected_index,
        tool_call_id=_tool_call_id(call),
        tool_name=_tool_call_name(call),
        arguments_json=_tool_call_arguments(call),
        command=_command_from_arguments(_tool_call_arguments(call)),
        records_before=len(records),
        records_after=len(truncated),
        manifest_path=str(manifest_path) if manifest_path else None,
        patched_workspace_base=patched_workspace_base,
    )
    metadata_path = output_path / "replay_prepare_manifest.json"
    _write_json_atomic(metadata_path, result.to_dict())
    return result


def _validate_selector(*, message_index: int | None, tool_call_id: str | None) -> None:
    if message_index is None and not tool_call_id:
        raise ReplayPrepareError("provide exactly one of message_index or tool_call_id")
    if message_index is not None and tool_call_id:
        raise ReplayPrepareError("provide exactly one of message_index or tool_call_id")
    if message_index is not None and message_index < 0:
        raise ReplayPrepareError("message_index must be >= 0")


def _resolve_memory_files(
    copied_root: Path,
    *,
    memory_path: str | Path | None,
    source_path: Path,
) -> ReplayMemoryFiles:
    if memory_path:
        candidate = _map_memory_path(copied_root, source_path=source_path, memory_path=memory_path)
        if not candidate.is_file():
            raise ReplayPrepareError(f"memory file not found: {candidate}")
        if candidate.name not in {"short_term.json", "long_term.jsonl"}:
            raise ReplayPrepareError(
                f"--memory must point to short_term.json or long_term.jsonl, got {candidate.name}",
            )
        return _memory_files_from_agent_dir(candidate.parent, explicit_source=candidate)

    agent_dirs = _find_agent_memory_dirs(copied_root)
    if not agent_dirs:
        raise ReplayPrepareError(
            f"no task_logs/**/memory/**/(long_term.jsonl|short_term.json) found under {copied_root}",
        )
    if len(agent_dirs) > 1:
        rels = "\n".join(str(p.relative_to(copied_root)) for p in agent_dirs[:20])
        raise ReplayPrepareError(
            "multiple agent memory directories found; pass --memory to disambiguate:\n" + rels,
        )
    return _memory_files_from_agent_dir(agent_dirs[0], explicit_source=None)


def _map_memory_path(copied_root: Path, *, source_path: Path, memory_path: str | Path) -> Path:
    raw = Path(memory_path).expanduser()
    if raw.is_absolute():
        try:
            rel = raw.resolve(strict=False).relative_to(source_path)
            return copied_root / rel
        except ValueError:
            return raw
    return copied_root / raw


def _find_agent_memory_dirs(copied_root: Path) -> list[Path]:
    dirs: set[Path] = set()
    for name in ("long_term.jsonl", "short_term.json"):
        for p in copied_root.rglob(name):
            if "task_logs" in p.parts and "memory" in p.parts:
                dirs.add(p.parent)
    return sorted(dirs)


def _memory_files_from_agent_dir(agent_dir: Path, *, explicit_source: Path | None) -> ReplayMemoryFiles:
    short_path = agent_dir / "short_term.json"
    long_path = agent_dir / "long_term.jsonl"
    if explicit_source is not None:
        source_path = explicit_source
        source_kind = "long_term" if explicit_source.name == "long_term.jsonl" else "short_term"
    elif _nonempty_file(long_path):
        source_path = long_path
        source_kind = "long_term"
    elif _nonempty_file(short_path):
        source_path = short_path
        source_kind = "short_term"
    else:
        raise ReplayPrepareError(f"no non-empty memory source in {agent_dir}")
    return ReplayMemoryFiles(
        source_path=source_path,
        source_kind=source_kind,
        short_term_path=short_path,
        long_term_path=long_path,
    )


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _read_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ReplayPrepareError(f"{path}:{lineno}: invalid JSONL record") from exc
    if not records:
        raise ReplayPrepareError(f"memory file has no records: {path}")
    return records


def _write_jsonl_atomic(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _select_record_index(
    records: list[Any],
    *,
    message_index: int | None,
    tool_call_id: str | None,
) -> int:
    if message_index is not None:
        if message_index >= len(records):
            raise ReplayPrepareError(
                f"message_index {message_index} out of range for {len(records)} records",
            )
        _validate_record_is_tool_call(records[message_index], message_index)
        return message_index

    assert tool_call_id
    for idx, record in enumerate(records):
        message = _record_message(record)
        if _message_role(message) != "assistant":
            continue
        if any(_tool_call_id(call) == tool_call_id for call in _message_tool_calls(message)):
            _validate_record_is_tool_call(record, idx)
            return idx
    raise ReplayPrepareError(f"tool_call_id not found in memory: {tool_call_id}")


def _validate_record_is_tool_call(record: Any, idx: int) -> None:
    message = _record_message(record)
    if _message_role(message) != "assistant":
        raise ReplayPrepareError(f"message {idx} is not an assistant message")
    if not _message_tool_calls(message):
        raise ReplayPrepareError(f"message {idx} has no tool_calls")


def _record_message(record: Any) -> Any:
    if isinstance(record, dict):
        msg = record.get("message")
        return msg if isinstance(msg, dict) else record
    return record


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "").strip()
    return str(getattr(message, "role", "") or "").strip()


def _message_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, dict):
        calls = message.get("tool_calls") or []
    else:
        calls = getattr(message, "tool_calls", None) or []
    return list(calls) if isinstance(calls, (list, tuple)) else []


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or "").strip()
    return str(getattr(tool_call, "id", "") or "").strip()


def _tool_call_function(tool_call: Any) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get("function") or {}
    return getattr(tool_call, "function", None)


def _tool_call_name(tool_call: Any) -> str:
    fn = _tool_call_function(tool_call)
    if isinstance(fn, dict):
        return str(fn.get("name") or "").strip()
    return str(getattr(fn, "name", "") or "").strip()


def _tool_call_arguments(tool_call: Any) -> str:
    fn = _tool_call_function(tool_call)
    if isinstance(fn, dict):
        args = fn.get("arguments")
    else:
        args = getattr(fn, "arguments", "")
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False, sort_keys=True)


def _command_from_arguments(arguments_json: str) -> str:
    try:
        data = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    command = data.get("cmd") or data.get("command") or ""
    return str(command)


_WORKSPACE_BASE_RE = re.compile(r"^(\s*workspace_base\s*:\s*).*$")


def _patch_workspace_base(path: Path, workspace_base: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    patched = False
    out: list[str] = []
    for line in lines:
        match = _WORKSPACE_BASE_RE.match(line)
        if match and not patched:
            out.append(f"{match.group(1)}{workspace_base}")
            patched = True
        else:
            out.append(line)
    if not patched:
        raise ReplayPrepareError(f"manifest has no workspace_base field: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.replace(tmp, path)
