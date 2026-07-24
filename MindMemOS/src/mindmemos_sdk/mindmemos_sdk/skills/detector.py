"""Best-effort skill-context detection from memory add messages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .bundle import compute_content_hash
from .models import SkillContext, SkillUsage
from .registry import SkillRegistry

_TOOL_CALL_RE = re.compile(r"^\s*\[tool_call\]\s*([A-Za-z0-9_.-]+)\((.*)\)\s*$", re.DOTALL)
_SKILL_MD_RE = re.compile(r"(?:^|[/\\])SKILL\.md$")
_STRONG_USAGE = {SkillUsage.MODIFIED: 2, SkillUsage.INJECTED: 1}


@dataclass
class _Candidate:
    path: str
    content: str
    usage: SkillUsage


def detect_skill_context(
    messages: list[BaseModel | dict[str, Any]],
    *,
    registry: SkillRegistry | None = None,
) -> list[SkillContext]:
    """Detect skill references from normalized add messages.

    The detector consumes only the same ``messages`` payload sent to memory add.
    It recognizes OpenClaw-style text tool calls for ``read`` / ``write`` /
    ``edit`` operations on ``SKILL.md``.
    """

    serialized = [_message_dict(message) for message in messages]
    candidates: dict[str, _Candidate] = {}
    for index, message in enumerate(serialized):
        if message.get("role") != "assistant":
            continue
        call = _parse_tool_call(str(message.get("content") or ""))
        if call is None:
            continue
        tool, args = call
        path = _arg_path(args)
        if not path or not _SKILL_MD_RE.search(path):
            continue
        key = _skill_key(path)
        if tool == "read":
            content = _next_tool_content(serialized, index)
            usage = SkillUsage.INJECTED
        elif tool == "write":
            content = _arg_text(args, "content")
            usage = SkillUsage.MODIFIED
        elif tool == "edit":
            content = _edit_content(args)
            usage = SkillUsage.MODIFIED
        else:
            continue
        if not content:
            continue

        existing = candidates.get(key)
        if existing is None or _STRONG_USAGE[usage] >= _STRONG_USAGE[existing.usage]:
            candidates[key] = _Candidate(path=path, content=content, usage=usage)

    contexts: list[SkillContext] = []
    for candidate in candidates.values():
        name, version_label = _parse_skill_metadata(candidate.content)
        record = _lookup_record(registry, candidate.path, name)
        contexts.append(
            SkillContext(
                name=name or _skill_dir_name(candidate.path),
                content_hash=compute_content_hash({"SKILL.md": candidate.content}),
                base_version_id=record.base_version_id if record else "",
                version_label=version_label,
                usage=candidate.usage,
            )
        )
    return contexts


def _message_dict(message: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, BaseModel):
        return message.model_dump()
    return message


def _parse_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    match = _TOOL_CALL_RE.match(content)
    if not match:
        return None
    tool = match.group(1).strip().lower()
    raw_args = match.group(2).strip()
    if not raw_args:
        return tool, {}
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError:
        return None
    return (tool, parsed) if isinstance(parsed, dict) else None


def _arg_path(args: dict[str, Any]) -> str | None:
    for key in ("path", "file_path", "filepath"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _arg_text(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    return value if isinstance(value, str) else ""


def _edit_content(args: dict[str, Any]) -> str:
    for key in ("content", "new_content", "replacement", "replace"):
        value = _arg_text(args, key)
        if value:
            return value
    return ""


def _next_tool_content(messages: list[dict[str, Any]], index: int) -> str:
    if index + 1 >= len(messages):
        return ""
    next_message = messages[index + 1]
    if next_message.get("role") != "tool":
        return ""
    content = next_message.get("content")
    return content if isinstance(content, str) else ""


def _skill_key(path: str) -> str:
    normalized = path.replace("\\", "/")
    return normalized.rsplit("/", 1)[0]


def _skill_dir_name(path: str) -> str:
    parent = Path(path.replace("\\", "/")).parent.name
    return parent or "skill"


def _lookup_record(registry: SkillRegistry | None, path: str, name: str | None):
    if registry is None:
        return None
    skill_dir = str(Path(path).expanduser().resolve().parent)
    record = registry.get_by_path(skill_dir)
    if record is not None:
        return record
    if name:
        return next((item for item in registry.list() if item.skill_name == name), None)
    return None


def _parse_skill_metadata(content: str) -> tuple[str | None, str | None]:
    return _find_simple_field(content, "name"), _find_simple_field(content, "version")


def _find_simple_field(content: str, field: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(field)}\s*:\s*[\"']?([^\"'\n#]+)", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return None
    return match.group(1).strip() or None
