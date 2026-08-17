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

"""Persist raw tool outputs and build compact memory-facing feedback."""

from __future__ import annotations

import hashlib
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from deepcraft_core.tool import ToolResult

from scienceflow.core.bash_solution_cmd import looks_like_python_script_run
from scienceflow.utils.node_paths import node_log_path
from scienceflow.utils.stage_tool_index import append_tool_index

logger = logging.getLogger("scienceflow")

_AUTO_SNAPSHOT_PREFIX = "[auto-snapshot after successful write:"
_TOOL_OUTPUT_REF_PREFIX = "[tool-output raw_id="


@dataclass(frozen=True)
class ToolOutputArtifactRef:
    seq: int
    raw_id: str
    tool_name: str
    raw_chars: int
    sha256_short: str


class ToolOutputArtifactStore:
    """Append-only per-workspace raw tool output store.

    The store intentionally uses deterministic per-node sequence numbers and relative
    ``raw_id`` names so memory text stays stable across workspace rotations.
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        *,
        output_parts: tuple[str, ...] | None = None,
        mirror_parts: tuple[str, ...] | None = None,
        mirror_raw_id_prefix: str = "",
        log_dir_override: str | Path | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.output_parts = tuple(output_parts or ("tool_outputs",))
        self.mirror_parts = tuple(mirror_parts or ())
        self.mirror_raw_id_prefix = _safe_prefix(mirror_raw_id_prefix)
        direct_log_dir = Path(log_dir_override) if log_dir_override is not None else None
        self.output_dir = (
            direct_log_dir.joinpath(*self.output_parts)
            if direct_log_dir is not None
            else node_log_path(self.workspace_dir, *self.output_parts)
        )
        self.mirror_output_dir = (
            (
                direct_log_dir.joinpath(*self.mirror_parts)
                if direct_log_dir is not None
                else node_log_path(self.workspace_dir, *self.mirror_parts)
            )
            if self.mirror_parts
            else None
        )
        self.index_path = self.output_dir / "index.txt"
        self.stage_log_dir: Path | None = None
        self._next_seq = self._discover_next_seq()

    def set_mirror_raw_id_prefix(self, prefix: str) -> None:
        self.mirror_raw_id_prefix = _safe_prefix(prefix)

    def set_stage_log_dir(self, path: str | Path | None) -> None:
        self.stage_log_dir = Path(path) if path is not None and str(path).strip() else None

    def _mirror_raw_id(self, raw_id: str) -> str:
        prefix = self.mirror_raw_id_prefix
        return f"{prefix}{raw_id}" if prefix else raw_id

    def _discover_next_seq(self) -> int:
        try:
            if not self.output_dir.is_dir():
                return 1
            max_seq = 0
            for path in self.output_dir.glob("tool_*.txt"):
                m = re.match(r"^tool_(\d{6})_", path.name)
                if m:
                    max_seq = max(max_seq, int(m.group(1)))
            return max_seq + 1
        except Exception:
            logger.debug("tool output artifact seq discovery failed", exc_info=True)
            return 1

    def reserve(self, tool_name: str, raw_text: str) -> ToolOutputArtifactRef | None:
        try:
            safe_tool = _safe_tool_name(tool_name)
            seq = self._next_seq
            self._next_seq += 1
            raw_id = f"tool_{seq:06d}_{safe_tool}.txt"
            sha = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()[:16]
            return ToolOutputArtifactRef(
                seq=seq,
                raw_id=raw_id,
                tool_name=safe_tool,
                raw_chars=len(raw_text),
                sha256_short=sha,
            )
        except Exception:
            logger.debug("tool output artifact reserve failed", exc_info=True)
            return None

    def write_raw(self, ref: ToolOutputArtifactRef, raw_text: str) -> bool:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / ref.raw_id
            path.write_text(raw_text, encoding="utf-8", errors="replace")
            if self.mirror_output_dir is not None:
                self.mirror_output_dir.mkdir(parents=True, exist_ok=True)
                mirror_path = self.mirror_output_dir / self._mirror_raw_id(ref.raw_id)
                mirror_path.write_text(raw_text, encoding="utf-8", errors="replace")
            return True
        except Exception:
            logger.debug("tool output artifact write failed", exc_info=True)
            return False

    def append_index(
        self,
        ref: ToolOutputArtifactRef,
        *,
        reducer_name: str,
        compressed_chars: int,
    ) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            line = (
                f"seq={ref.seq} raw_id={ref.raw_id} tool={ref.tool_name} "
                f"raw_chars={ref.raw_chars} compressed_chars={int(compressed_chars)} "
                f"sha256={ref.sha256_short} reducer={_safe_value(reducer_name)}"
            )
            with self.index_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if self.mirror_output_dir is not None:
                self.mirror_output_dir.mkdir(parents=True, exist_ok=True)
                mirror_raw_id = self._mirror_raw_id(ref.raw_id)
                mirror_line = (
                    f"seq={ref.seq} raw_id={mirror_raw_id} source_raw_id={ref.raw_id} "
                    f"tool={ref.tool_name} raw_chars={ref.raw_chars} "
                    f"compressed_chars={int(compressed_chars)} sha256={ref.sha256_short} "
                    f"reducer={_safe_value(reducer_name)}"
                )
                with (self.mirror_output_dir / "index.txt").open("a", encoding="utf-8") as f:
                    f.write(mirror_line + "\n")
        except Exception:
            logger.debug("tool output artifact index append failed", exc_info=True)

    def append_stage_index(
        self,
        ref: ToolOutputArtifactRef,
        *,
        args: dict[str, Any],
        reducer_name: str,
        compressed_chars: int,
        tool_error: bool,
    ) -> None:
        if self.stage_log_dir is None:
            return
        try:
            append_tool_index(
                self.stage_log_dir,
                raw_id=ref.raw_id,
                tool_name=ref.tool_name,
                raw_chars=ref.raw_chars,
                compressed_chars=int(compressed_chars),
                reducer_name=reducer_name,
                full_output_path=self.output_dir / ref.raw_id,
                tool_error=bool(tool_error),
                args=args,
            )
        except Exception:
            logger.debug("tool output stage index append failed", exc_info=True)


def tool_result_text(tool_result: ToolResult) -> str:
    try:
        return str(tool_result)
    except Exception:
        return repr(tool_result)


def raw_tool_result_text(tool_result: ToolResult) -> str:
    """Return exact raw output when a tool carried one out-of-band for artifacts."""
    try:
        raw = getattr(tool_result, "system", None)
        if isinstance(raw, str) and raw:
            return raw
    except Exception:
        pass
    return tool_result_text(tool_result)


def reduce_tool_feedback_for_memory(
    *,
    tool_name: str,
    args: dict[str, Any],
    feedback: str,
    raw_text: str,
    tool_error: bool,
) -> tuple[str, str]:
    """Return ``(feedback, reducer_name)`` for memory-facing tool output."""
    del raw_text
    name = (tool_name or "").strip()
    if not feedback:
        return feedback, "empty"
    existing = _existing_reducer_name(feedback)
    if existing:
        return feedback, existing
    if name == "bash":
        return _reduce_bash_feedback(
            feedback,
            command=str((args or {}).get("command") or ""),
            tool_error=tool_error,
        )
    if name == "grep":
        return _reduce_grep_feedback(feedback)
    if name == "read":
        return _reduce_text_lookup_feedback(feedback, tool_name=name)
    if name in {"glob", "ls"}:
        return _reduce_glob_ls_feedback(feedback)
    if name in {"write", "edit"}:
        return _reduce_write_edit_feedback(feedback, tool_name=name)
    return feedback, "none"


def attach_tool_output_reference(
    feedback: str,
    *,
    raw_id: str,
    raw_chars: int,
    reducer: str,
) -> str:
    line = (
        f"{_TOOL_OUTPUT_REF_PREFIX}{raw_id} raw_chars={int(raw_chars)} "
        f"reducer={_safe_value(reducer)}]"
    )
    if not feedback:
        return line
    idx = feedback.find(_AUTO_SNAPSHOT_PREFIX)
    if idx >= 0:
        prefix = feedback[:idx].rstrip()
        snap = feedback[idx:].lstrip()
        if prefix:
            return prefix + "\n" + line + "\n\n" + snap
        return line + "\n\n" + snap
    return feedback.rstrip() + "\n" + line


def _safe_prefix(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return f"{s}_" if s and not s.endswith("_") else s


def _safe_tool_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "tool")).strip("._-")
    return s or "tool"


def _safe_value(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "none")).strip("_")
    return s or "none"


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _existing_reducer_name(feedback: str) -> str:
    head = (feedback or "")[:1600]
    if "[see current snapshot:" in head:
        return "snapshot_ref_v1"
    m = re.search(r"\[tool-output compressed:\s+reducer=([A-Za-z0-9_.:-]+)", head)
    if m:
        return _safe_value(m.group(1))
    if "[read compressed: reducer=read_code_map_v2" in head:
        return "read_code_map_v2"
    if "[edit-anchor suggestion:" in head:
        return "edit_anchor_suggestion_v1"
    return ""


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


_BASH_SIGNAL_RE = re.compile(
    r"(traceback|error|exception|failed|failure|warning|metric|score|auc|rmse|mae|mse|"
    r"accuracy|acc\b|loss|best|iteration|epoch|cv|fold|valid|validation|train|"
    r"shape|columns?|missing|target|submission|saved|written|rows?|cols?)",
    re.IGNORECASE,
)


_BASH_TRAINING_SIGNAL_RE = re.compile(
    r"(score|metric|rmse|rmsle|mae|mse|accuracy|acc\b|auc|loss|epoch|fold|best|"
    r"valid|validation|final|saved|written|submission|artifact|model|csv|pkl)",
    re.IGNORECASE,
)
_BASH_INSTALL_SIGNAL_RE = re.compile(
    r"(successfully installed|installed|resolved|prepared|done|success|error|failed|failure)",
    re.IGNORECASE,
)


def _bash_first_command_words(cmd: str) -> list[str]:
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
        return (
            len(rest) >= 3
            and rest[0] in {"python", "python3"}
            and rest[1] == "-m"
            and rest[2] == "pytest"
        )
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


def _dedupe_recent_lines(lines: list[str], *, limit: int) -> list[str]:
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


_PY_TRACEBACK_START = "Traceback (most recent call last):"
_PY_EXCEPTION_LINE_RE = re.compile(
    r"^\s*(?:[A-Za-z_][\w.]*\.)*[A-Za-z_]\w*(?:Error|Exception|Warning):\s+.+"
)


def _extract_recent_python_error_block(lines: list[str], *, max_lines: int = 12) -> list[str]:
    if not lines:
        return []
    start: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        if _PY_TRACEBACK_START in lines[i]:
            start = i
            break
    if start is not None:
        return lines[start:min(len(lines), start + max_lines)]
    for i in range(len(lines) - 1, -1, -1):
        if _PY_EXCEPTION_LINE_RE.search(lines[i]):
            return lines[max(0, i - 3):min(len(lines), i + 2)]
    return []


def _bash_marker(first: str, reducer: str, feedback: str) -> list[str]:
    return [
        first,
        (
            f"[tool-output compressed: reducer={reducer} "
            f"raw_chars={len(feedback)} raw_lines={_line_count(feedback)}]"
        ),
    ]


def _reduce_bash_training_feedback(feedback: str) -> tuple[str, str]:
    lines = feedback.splitlines()
    if not lines:
        return feedback, "none"
    first, body = lines[0], lines[1:]
    if len(body) <= 20:
        return feedback, "none"
    tail = body[-5:]
    tail_keys = {ln.strip() for ln in tail}
    error_block = _extract_recent_python_error_block(body)
    error_keys = {ln.strip() for ln in error_block}
    signals = _dedupe_recent_lines(
        [ln for ln in body if _BASH_TRAINING_SIGNAL_RE.search(ln)],
        limit=15,
    )
    signals = [
        ln for ln in signals
        if ln.strip() not in tail_keys and ln.strip() not in error_keys
    ]
    parts = _bash_marker(first, "bash_training_signal_v2", feedback)
    if error_block:
        parts.append("--- preserved error/traceback lines ---")
        parts.extend(error_block)
    if signals:
        parts.append("--- preserved training signal lines ---")
        parts.extend(signals)
    parts.append("--- tail last 5 lines ---")
    parts.extend(tail)
    return "\n".join(parts), "bash_training_signal_v2"


def _reduce_bash_pytest_feedback(feedback: str, *, tool_error: bool) -> tuple[str, str]:
    lines = feedback.splitlines()
    if len(lines) <= 80 and len(feedback) <= 3600:
        return feedback, "none"
    if not lines:
        return feedback, "none"
    first, body = lines[0], lines[1:]
    summary_re = re.compile(
        r"(=+\s*.*(?:passed|failed|errors?|warnings?|skipped|xfailed|xpassed).*=+|"
        r"\b\d+\s+(?:passed|failed|errors?|warnings?|skipped)\b)",
        re.IGNORECASE,
    )
    failure_re = re.compile(
        r"(^FAILED\s+|^ERROR\s+|FAILURES|ERRORS|short test summary info|"
        r"\bFAILED\b|\bERROR\b|assert\s|^E\s+|^>\s+|Traceback|File \")",
    )
    picked: list[str] = []
    for ln in body:
        if " PASSED " in f" {ln} " and not summary_re.search(ln):
            continue
        if summary_re.search(ln) or failure_re.search(ln):
            picked.append(ln)
    if tool_error and len(picked) < 20:
        picked.extend(body[-20:])
    picked = _dedupe_preserve_order(picked, limit=80 if tool_error or picked else 20)
    if not picked:
        picked = body[-8:]
    parts = _bash_marker(first, "bash_pytest_summary_v2", feedback)
    parts.extend(picked[:80])
    return "\n".join(parts), "bash_pytest_summary_v2"


def _reduce_bash_install_feedback(feedback: str, *, tool_error: bool) -> tuple[str, str]:
    lines = feedback.splitlines()
    if not lines:
        return feedback, "none"
    first, body = lines[0], lines[1:]
    if not tool_error and len(body) <= 5:
        return feedback, "none"
    if tool_error:
        signals = _dedupe_recent_lines(
            [ln for ln in body if _BASH_INSTALL_SIGNAL_RE.search(ln)],
            limit=20,
        )
        tail = body[-20:]
        parts = _bash_marker(first, "bash_install_summary_v2", feedback)
        parts.extend(_dedupe_preserve_order([*signals, *tail], limit=40))
        return "\n".join(parts), "bash_install_summary_v2"
    signals = _dedupe_recent_lines(
        [ln for ln in body if _BASH_INSTALL_SIGNAL_RE.search(ln)],
        limit=4,
    )
    tail = body[-1:] if body else []
    kept = _dedupe_preserve_order([*signals, *tail], limit=5)
    parts = _bash_marker(first, "bash_install_summary_v2", feedback)
    parts.extend(kept)
    return "\n".join(parts), "bash_install_summary_v2"


def _reduce_bash_tail_feedback(feedback: str, *, tool_error: bool) -> tuple[str, str]:
    lines = feedback.splitlines()
    if len(feedback) <= 3600 and len(lines) <= 80:
        return feedback, "none"
    if not lines:
        return feedback, "none"
    first = lines[0]
    body = lines[1:]
    signals = _dedupe_preserve_order(
        [ln for ln in body if _BASH_SIGNAL_RE.search(ln)],
        limit=36 if tool_error else 28,
    )
    tail_n = 40 if tool_error else 24
    tail = body[-tail_n:] if len(body) > tail_n else body
    parts = [
        first,
        (
            "[tool-output compressed: reducer=bash_tail_v1 "
            f"raw_chars={len(feedback)} raw_lines={_line_count(feedback)}]"
        ),
    ]
    if signals:
        parts.append("--- preserved signal lines ---")
        parts.extend(signals)
    parts.append(f"--- tail last {len(tail)} lines ---")
    parts.extend(tail)
    reduced = "\n".join(parts)
    return reduced, "bash_tail_v1"


def _reduce_bash_feedback(
    feedback: str,
    *,
    command: str,
    tool_error: bool,
) -> tuple[str, str]:
    if looks_like_python_script_run(command):
        return _reduce_bash_training_feedback(feedback)
    if _bash_command_looks_like_test(command):
        return _reduce_bash_pytest_feedback(feedback, tool_error=tool_error)
    if _bash_command_looks_like_install(command):
        return _reduce_bash_install_feedback(feedback, tool_error=tool_error)
    return _reduce_bash_tail_feedback(feedback, tool_error=tool_error)


def _reduce_text_lookup_feedback(feedback: str, *, tool_name: str) -> tuple[str, str]:
    if len(feedback) <= 4200 and _line_count(feedback) <= 120:
        return feedback, "none"
    lines = feedback.splitlines()
    if not lines:
        return feedback, "none"
    signal_re = re.compile(
        r"(shape|columns?|dtype|missing|null|nan|target|label|train|test|valid|"
        r"submission|rows?|cols?|matches?|total)",
        re.IGNORECASE,
    )
    head = lines[:40]
    signals = _dedupe_preserve_order(
        [ln for ln in lines[40:-30] if signal_re.search(ln)],
        limit=30,
    )
    tail = lines[-30:] if len(lines) > 70 else []
    parts = [
        (
            f"[tool-output compressed: reducer={tool_name}_eda_v1 "
            f"raw_chars={len(feedback)} raw_lines={len(lines)}]"
        ),
        "--- head ---",
        *head,
    ]
    if signals:
        parts.append("--- preserved EDA signal lines ---")
        parts.extend(signals)
    if tail:
        parts.append("--- tail ---")
        parts.extend(tail)
    return "\n".join(parts), f"{tool_name}_eda_v1"


def _parse_grep_line(line: str) -> tuple[str, int, str] | None:
    m = re.match(r"^(.+?):(\d+):(.*)$", line)
    if not m:
        return None
    try:
        return m.group(1), int(m.group(2)), m.group(3)
    except ValueError:
        return None


def _reduce_grep_feedback(feedback: str) -> tuple[str, str]:
    if not feedback or feedback.startswith("No matches for pattern:"):
        return feedback, "none"
    groups: dict[str, list[tuple[int, str]]] = {}
    order: list[str] = []
    footer: list[str] = []
    for line in feedback.splitlines():
        parsed = _parse_grep_line(line)
        if parsed is None:
            footer.append(line)
            continue
        path, lineno, text = parsed
        if path not in groups:
            order.append(path)
            groups[path] = []
        groups[path].append((lineno, text))
    total_matches = sum(len(v) for v in groups.values())
    if total_matches < 8 and all(len(v) <= 5 for v in groups.values()):
        return feedback, "none"
    parts = [
        (
            "[tool-output compressed: reducer=grep_grouped_v2 "
            f"raw_chars={len(feedback)} raw_lines={_line_count(feedback)}]"
        )
    ]
    rendered = 1
    omitted_files = 0
    for path in order:
        matches = groups[path]
        if rendered >= 50:
            omitted_files += 1
            continue
        if len(matches) > 5:
            shown_lines = ",".join(str(n) for n, _ in matches[:10])
            if len(matches) > 10:
                shown_lines += ",..."
            parts.append(f"{path}: {len(matches)} matches (lines {shown_lines})")
            rendered += 1
            continue
        parts.append(f"{path}:")
        rendered += 1
        for lineno, text in matches:
            if rendered >= 50:
                omitted_files += 1
                break
            trimmed = text if len(text) <= 180 else text[:177].rstrip() + "..."
            parts.append(f"  {lineno}| {trimmed}")
            rendered += 1
    if omitted_files:
        parts.append(f"... [{omitted_files} files omitted by grep grouped reducer]")
    for line in footer[-3:]:
        if line and not line.startswith("... ["):
            parts.append(line)
    return "\n".join(parts), "grep_grouped_v2"


_CONFIG_EXTS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}
_DATA_EXTS = {".csv", ".parquet", ".feather", ".jsonl", ".pkl", ".joblib"}


def _path_from_ls_or_glob_line(line: str) -> tuple[str, bool] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("[") or stripped.startswith("... ["):
        return None
    m = re.match(r"^(?:DIRLINK|DIR|FILELINK|FILE|LINK|OTHER|BLOCKED)\s+(.+?)(?:\s+\(\d+ bytes\)|\s+\(size unknown\))?$", stripped)
    if m:
        path = m.group(1).strip()
        return path.rstrip("/"), path.endswith("/") or stripped.startswith(("DIR ", "DIRLINK "))
    return stripped.rstrip("/"), stripped.endswith("/")


def _append_limited_group(parts: list[str], title: str, values: list[str], *, limit: int) -> int:
    if not values:
        return 0
    parts.append(f"--- {title} ({len(values)}) ---")
    for val in values[:limit]:
        parts.append(val)
    if len(values) > limit:
        parts.append(f"... [{len(values) - limit} more {title.lower()} omitted]")
    return min(len(values), limit)


def _reduce_glob_ls_feedback(feedback: str) -> tuple[str, str]:
    lines = feedback.splitlines()
    if len(feedback) <= 4200 and len(lines) <= 40:
        return feedback, "none"
    if not lines:
        return feedback, "none"
    header = lines[0]
    entries: list[tuple[str, bool]] = []
    footer: list[str] = []
    for line in lines[1:]:
        parsed = _path_from_ls_or_glob_line(line)
        if parsed is None:
            footer.append(line)
        else:
            entries.append(parsed)
    if len(entries) <= 40:
        return feedback, "none"
    py_files: list[str] = []
    config_files: list[str] = []
    data_files: list[str] = []
    dirs: list[str] = []
    other: list[str] = []
    for path, is_dir in entries:
        if is_dir:
            dirs.append(path.rstrip("/") + "/")
            continue
        suffix = PurePath(path).suffix.lower()
        if suffix == ".py":
            py_files.append(path)
        elif suffix in _CONFIG_EXTS:
            config_files.append(path)
        elif suffix in _DATA_EXTS:
            data_files.append(path)
        else:
            other.append(path)
    parts = [
        header,
        (
            "[tool-output compressed: reducer=glob_ls_typed_v2 "
            f"raw_chars={len(feedback)} raw_lines={len(lines)}]"
        ),
    ]
    shown = 0
    shown += _append_limited_group(parts, "Python files", py_files, limit=12)
    shown += _append_limited_group(parts, "Config files", config_files, limit=10)
    shown += _append_limited_group(parts, "Data files", data_files, limit=8)
    shown += _append_limited_group(parts, "Directories", dirs, limit=8)
    if other:
        parts.append(f"--- Other files ({len(other)}) ---")
        for val in other[:2]:
            parts.append(val)
        if len(other) > 2:
            parts.append(f"... [{len(other) - 2} other files omitted]")
    if footer:
        parts.extend(footer[-2:])
    if shown == 0 and not other:
        return feedback, "none"
    return "\n".join(parts), "glob_ls_typed_v2"


def _reduce_write_edit_feedback(feedback: str, *, tool_name: str) -> tuple[str, str]:
    if _AUTO_SNAPSHOT_PREFIX in feedback:
        return feedback, "write_snapshot_existing" if tool_name == "write" else "edit_snapshot_existing"
    if len(feedback) <= 4200 and _line_count(feedback) <= 120:
        return feedback, "none"
    lines = feedback.splitlines()
    head = lines[:30]
    tail = lines[-30:] if len(lines) > 60 else []
    parts = [
        (
            f"[tool-output compressed: reducer={tool_name}_head_tail_v1 "
            f"raw_chars={len(feedback)} raw_lines={len(lines)}]"
        ),
        *head,
    ]
    if tail:
        parts.append("--- tail ---")
        parts.extend(tail)
    return "\n".join(parts), f"{tool_name}_head_tail_v1"
