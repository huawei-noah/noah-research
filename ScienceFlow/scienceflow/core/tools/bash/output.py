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

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

_OBSERVATION_EXEC_RE = re.compile(
    r"(^|[;\n&|()]\s*)(?:timeout\s+\S+\s+)?(?:env\s+\S+=\S+\s+)*"
    r"(?:ls|find|cat|head|tail|sed|wc|du|stat|ps)\b",
)


_CAT_WRITE_OR_HEREDOC_RE = re.compile(r"(^|[;\n&|()]\s*)cat\s*(?:>|>>|<<)")


_COMPUTE_EXEC_RE = re.compile(
    r"(^|[;\n&|()]\s*)(?:timeout\s+\S+\s+)?(?:env\s+\S+=\S+\s+)*"
    r"(?:python(?:3)?|uv|pip(?:3)?|pytest|mlebench|kaggle|git)\b",
)


def _sanitize_workspace_prefix_in_output(text: str, workspace: Path) -> str:
    """Replace absolute workspace path with ``.`` so the LLM does not see host/node paths.

    Uses a boundary-aware replacement so ``/a/b`` does not corrupt ``/a/b_extra``.
    """
    ws = str(workspace.resolve())
    if not ws:
        return text
    return re.sub(re.escape(ws) + r"(?=/|$)", ".", text)


def _sanitize_root_prefix_in_output(text: str, root: Path, marker: str) -> str:
    root_s = str(root.resolve()).rstrip("/")
    if not text or not root_s or root_s == "/":
        return text
    return re.sub(re.escape(root_s) + r"(?=/|$)", marker.rstrip("/"), text)


def _sanitize_python_env_paths(text: str) -> str:
    """Replace virtualenv/site-packages absolute paths with stable labels."""
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        path = match.group(0)
        for marker in ("/site-packages/", "/dist-packages/"):
            idx = path.find(marker)
            if idx >= 0:
                return "<python-env>" + path[idx:]
        return "<python-env>/" + Path(path).name

    return re.sub(
        r"/(?:[^\s\"'<>:]+/)+(?:site-packages|dist-packages)/[^\s\"'<>),]+",
        repl,
        text,
    )


def _sanitize_host_absolute_paths(text: str) -> str:
    """Hide remaining host absolute paths in model-visible bash output."""
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        path = match.group(0)
        name = Path(path).name
        return f"<abs-path>/{name}" if name else "<abs-path>"

    return re.sub(r"(?<![\w.])/(?:home|work|mnt|kaggle|tmp)/[^\s\"'<>),]+", repl, text)


def _strip_symlink_targets(text: str) -> str:
    """Remove symlink arrow targets (`` -> /real/path``) from bash output lines.

    Only strips absolute-path targets (starting with ``/``) so that regular
    text containing `` -> `` (e.g. log messages, code output) is not affected.
    """
    return re.sub(r" -> /\S+", "", text)


def _sanitize_model_visible_output_paths(
    text: str,
    workspace: Path,
    extra_roots: Sequence[Path | str] = (),
    *,
    strip_symlink_targets: bool = True,
) -> str:
    """Apply the same path hygiene to final bash output and live stream mirrors."""
    if not text:
        return text
    out = _sanitize_workspace_prefix_in_output(text, workspace)
    for idx, root in enumerate(extra_roots or (), start=1):
        out = _sanitize_root_prefix_in_output(out, Path(root), f"<extra-root-{idx}>")
    if strip_symlink_targets:
        out = _strip_symlink_targets(out)
    out = _sanitize_python_env_paths(out)
    return _sanitize_host_absolute_paths(out)


def _trim_output(text: str, max_chars: int) -> str:
    """Head+tail truncation: keep start and end of long output."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    k = max(max_chars // 2 - 40, 800)
    if 2 * k + 80 > max_chars:
        k = max((max_chars - 80) // 2, 400)
    omitted = len(text) - 2 * k
    nlines = len(text.splitlines())
    return (
        f"{text[:k]}\n"
        f"... [truncated {omitted} of {len(text)} chars, {nlines} lines total; "
        f"showing head+tail within max_output_chars={max_chars}] ...\n"
        f"{text[-k:]}"
    )


def _looks_like_lossless_observation_command(command: str) -> bool:
    """Return true for read-only observation commands worth summarizing losslessly.

    This is intentionally conservative: it never rewrites or blocks execution,
    and it avoids Python/training/install/git commands where stdout carries
    progress or error semantics better handled by existing reducers.
    """
    cmd = str(command or "").strip()
    if not cmd:
        return False
    if _CAT_WRITE_OR_HEREDOC_RE.search(cmd):
        return False
    if _COMPUTE_EXEC_RE.search(cmd):
        return False
    return bool(_OBSERVATION_EXEC_RE.search(cmd))


def _edge_lines_for_lossless_summary(
    lines: list[str],
    *,
    max_chars: int,
) -> tuple[list[str], list[str]]:
    if not lines:
        return [], []
    # Keep the visible summary comfortably below exec_feedback_max_chars while
    # preserving exact prefix/suffix evidence. The full raw output is carried
    # separately in ToolResult.system for artifact storage.
    budget = max(1200, min(max_chars if max_chars > 0 else 6000, 6000))
    edge_lines = max(8, min(40, budget // 180))
    if len(lines) <= edge_lines * 2:
        return lines, []
    return lines[:edge_lines], lines[-edge_lines:]


def _lossless_observation_summary(
    text: str,
    *,
    command: str,
    max_chars: int,
) -> str:
    raw_bytes = len(text.encode("utf-8", errors="replace"))
    raw_chars = len(text)
    raw_lines = len(text.splitlines())
    sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    lines = text.splitlines()
    head, tail = _edge_lines_for_lossless_summary(lines, max_chars=max_chars)
    omitted = max(0, raw_lines - len(head) - len(tail))
    command_one_line = re.sub(r"\s+", " ", str(command or "")).strip()
    if len(command_one_line) > 240:
        command_one_line = command_one_line[:237].rstrip() + "..."
    parts = [
        (
            "[tool-output compressed: reducer=bash_observation_head_tail_v1 "
            f"raw_chars={raw_chars} raw_bytes={raw_bytes} raw_lines={raw_lines} "
            f"sha256={sha[:16]}]"
        ),
        "Visible output is an exact head/tail projection, not the complete output.",
        f"command: {command_one_line}",
    ]
    if head:
        parts.append(f"--- head first {len(head)} lines ---")
        parts.extend(head)
    if tail:
        parts.append(f"--- omitted middle: {omitted} lines ---")
        parts.append(f"--- tail last {len(tail)} lines ---")
        parts.extend(tail)
    return "\n".join(parts)


def _maybe_lossless_observation_summary(
    *,
    command: str,
    output: str,
    enabled: bool,
    max_chars: int,
    returncode: int,
) -> tuple[str, str | None]:
    """Return ``(visible_output, raw_for_artifact)`` for large observation output."""
    if not enabled or returncode != 0:
        return output, None
    if max_chars <= 0 or len(output) <= max_chars:
        return output, None
    if not _looks_like_lossless_observation_command(command):
        return output, None
    return (
        _lossless_observation_summary(output, command=command, max_chars=max_chars),
        output,
    )


def _dedup_repeated_blocks(
    text: str,
    *,
    min_repeat: int = 3,
    max_block_size: int = 4,
) -> str:
    """Collapse consecutive identical line-blocks when repeated at least *min_repeat* times.

    Tries block sizes from *max_block_size* down to 1 at each position so that
    multi-line patterns (e.g. warning + code line) are folded as a unit when possible.
    If *min_repeat* is 0, returns *text* unchanged.
    """
    if min_repeat <= 0 or not text:
        return text
    lines = text.splitlines()
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        best_end: int | None = None
        best_block_size = 1
        best_repeat = 1
        max_bs = min(max_block_size, n - i)
        for block_size in range(max_bs, 0, -1):
            block = lines[i : i + block_size]
            pos = i + block_size
            repeat_count = 1
            while pos + block_size <= n and lines[pos : pos + block_size] == block:
                repeat_count += 1
                pos += block_size
            if repeat_count >= min_repeat:
                best_end = pos
                best_block_size = block_size
                best_repeat = repeat_count
                break
        if best_end is not None:
            out.extend(lines[i : i + best_block_size])
            out.append(
                f"[... above {best_block_size}-line block repeated {best_repeat} times total, collapsed]"
            )
            i = best_end
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


_TB_HEADER_RE = re.compile(r"^Traceback \(most recent call last\):\s*$")


_PYTHON_EXEC_TOKEN_RE = re.compile(r"\b(?:python3?|ipython)\b")


_FILE_LINE_RE = re.compile(
    r'^  File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$'
)


def _has_masked_python_traceback(command: str, output: str, rc: int) -> bool:
    """Detect Python failures hidden by shell pipelines such as ``python3 x.py | cat``."""
    if rc != 0 or not output:
        return False
    if "Traceback (most recent call last):" not in output:
        return False
    return "|" in (command or "") and bool(_PYTHON_EXEC_TOKEN_RE.search(command or ""))


def _is_user_workspace_path(path_str: str, workspace: Path) -> bool:
    """True if *path_str* refers to a file under *workspace* (resolved)."""
    try:
        ws = workspace.resolve()
        p = Path(path_str)
        if not p.is_absolute():
            p = (ws / p).resolve()
        else:
            p = p.resolve()
        p.relative_to(ws)
        return True
    except (ValueError, OSError):
        return False


def _path_relative_to_workspace_str(path_str: str, workspace: Path) -> str | None:
    """Return *path_str* as a path relative to *workspace* (posix), or None if not under it."""
    try:
        ws = workspace.resolve()
        p = Path(path_str)
        if not p.is_absolute():
            p = (ws / p).resolve()
        else:
            p = p.resolve()
        rel = p.relative_to(ws)
        return rel.as_posix()
    except (ValueError, OSError):
        return None


def _rewrite_tb_file_line_to_relative(line: str, workspace: Path) -> str:
    """Rewrite ``  File \"...\", line N, in ...`` to use workspace-relative path when applicable."""
    m = _FILE_LINE_RE.match(line)
    if not m:
        return line
    path = m.group("path")
    rel = _path_relative_to_workspace_str(path, workspace)
    if rel is None:
        return line
    return (
        f'  File "{rel}", line {m.group("line")}, in {m.group("func")}'
    )


def _tb_collapse_label(first_path: str, last_path: str) -> str:
    """Short labels for collapsed library frames (e.g. sklearn.base -> sklearn.utils.validation)."""

    def short(p: str) -> str:
        parts = Path(p).parts
        if "site-packages" in parts:
            idx = parts.index("site-packages")
            rest = parts[idx + 1 :]
            if not rest:
                return Path(p).name
            return ".".join(x.replace(".py", "") for x in rest)
        return Path(p).name

    return f"{short(first_path)} -> {short(last_path)}"


def _is_tb_frame_continuation(line: str) -> bool:
    if line.startswith("    "):
        return True
    stripped = line.lstrip()
    if stripped.startswith("^"):
        return True
    return False


def _is_tb_exception_line(line: str) -> bool:
    if not line.strip():
        return False
    if line.startswith("  "):
        return False
    if line.startswith("During handling"):
        return False
    if line.startswith("Traceback"):
        return False
    if line.strip() == "KeyboardInterrupt":
        return True
    if re.match(
        r"^[\w.]+(?:Error|Exception|Exit|Interrupt|Warning|DeprecationWarning)\s*:",
        line,
    ):
        return True
    return False


def _format_distilled_tb(frames: list[dict], exc_line: str) -> str:
    out = ["Traceback (most recent call last):"]
    i = 0
    n = len(frames)
    while i < n:
        if frames[i]["user"]:
            out.extend(frames[i]["block"])
            i += 1
            continue
        j = i
        while j < n and not frames[j]["user"]:
            j += 1
        count = j - i
        first = frames[i]["path"]
        last = frames[j - 1]["path"]
        out.append(
            f"  [... {count} library frames collapsed ({_tb_collapse_label(first, last)})]"
        )
        i = j
    out.append(exc_line)
    return "\n".join(out)


def _consume_and_distill_one_tb(
    lines: list[str],
    start: int,
    workspace: Path,
) -> tuple[int, str]:
    """Parse one traceback starting at *start* (header line index). Returns (next_index, distilled_text)."""
    assert _TB_HEADER_RE.match(lines[start])
    i = start + 1
    frames: list[dict] = []
    while i < len(lines):
        m = _FILE_LINE_RE.match(lines[i])
        if m:
            path = m.group("path")
            block = [lines[i]]
            i += 1
            while i < len(lines) and _is_tb_frame_continuation(lines[i]):
                block.append(lines[i])
                i += 1
            user = _is_user_workspace_path(path, workspace)
            if user:
                block[0] = _rewrite_tb_file_line_to_relative(block[0], workspace)
            frames.append(
                {
                    "user": user,
                    "block": block,
                    "path": path,
                }
            )
            continue
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if _is_tb_exception_line(line):
            exc = line
            i += 1
            distilled = _format_distilled_tb(frames, exc)
            parts: list[str] = [distilled]
            while i < len(lines):
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i >= len(lines):
                    break
                if (
                    "During handling" in lines[i]
                    and "another exception" in lines[i]
                ):
                    parts.append(lines[i])
                    i += 1
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    if i < len(lines) and _TB_HEADER_RE.match(lines[i]):
                        j, nested = _consume_and_distill_one_tb(lines, i, workspace)
                        parts.append(nested)
                        i = j
                        continue
                    break
                break
            return i, "\n\n".join(parts)
        # Unrecognized line while parsing frames (malformed traceback): keep remainder verbatim.
        return len(lines), "\n".join(lines[start:])
    return i, "\n".join(lines[start:i])


def _distill_tracebacks(
    text: str,
    workspace_dir: Path,
    *,
    enabled: bool = True,
) -> str:
    """Collapse library-internal frames in Python tracebacks; keep user workspace frames + exception."""
    if not enabled or not text:
        return text
    lines = text.splitlines()
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if _TB_HEADER_RE.match(lines[i]):
            j, distilled = _consume_and_distill_one_tb(lines, i, workspace_dir)
            out.extend(distilled.splitlines())
            i = j
        else:
            out.append(lines[i])
            i += 1
    result = "\n".join(out)
    if text.endswith("\n") and result and not result.endswith("\n"):
        result += "\n"
    return result
