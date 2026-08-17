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

"""Parse execution output: traceback extraction, metric detection, error line mapping."""

from __future__ import annotations

import re

_FILE_PATTERN = re.compile(r'^\s*File "(.*?)", line (\d+), in (.*)$')

_METRIC_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:score|metric|accuracy|f1|auc|rmsle|rmse|mae|mse|r2|val(?:idation)?[\s_]?(?:score|acc|loss))"
    r"\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

_METRIC_VALUE_LINE = re.compile(
    r"^\s*METRIC_VALUE\s*=",
    re.IGNORECASE,
)

_ID_EXCEPTION_TAIL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def stderr_looks_like_python_exception(stderr: str) -> bool:
    """True if stderr looks like a traceback or a non-warning exception line."""
    if not stderr or not stderr.strip():
        return False
    if "Traceback (most recent call last)" in stderr:
        return True
    for line in stderr.splitlines():
        if _FILE_PATTERN.match(line.strip()):
            return True
    last = stderr.strip().split("\n")[-1].strip()
    if ":" not in last:
        return False
    head = last.split(":", 1)[0].strip()
    parts = head.split()
    name = parts[-1] if parts else head
    if not _ID_EXCEPTION_TAIL.match(name):
        return False
    if name.endswith("Warning"):
        return False
    if name.endswith("Error") or name.endswith("Exception"):
        return True
    if name in ("KeyboardInterrupt", "SystemExit", "GeneratorExit"):
        return True
    return False


def parse_traceback(
    stderr: str,
    returncode: int = 0,
) -> tuple[str | None, dict | None, list[tuple] | None]:
    """Extract structured exception info from subprocess stderr.

    When ``returncode == 0``, only non-empty stderr that looks like a Python
    traceback or exception is parsed; benign warnings/logs do not yield
    ``exc_type``. When ``returncode != 0`` and stderr is not traceback-like,
    returns a generic ``SubprocessError`` with the stderr body as message.

    Returns:
        (exc_type, exc_info, exc_stack) where:
        - exc_type: exception class name or None
        - exc_info: dict with 'msg' key or None
        - exc_stack: list of (lineno_str, name_str, source_line) tuples or None
    """
    if not stderr or not stderr.strip():
        return None, None, None

    looks = stderr_looks_like_python_exception(stderr)
    if returncode == 0 and not looks:
        return None, None, None

    if returncode != 0 and not looks:
        msg = stderr.strip()
        if len(msg) > 4000:
            msg = msg[:3997] + "..."
        return "SubprocessError", {"msg": msg}, None

    lines = stderr.strip().split("\n")
    exc_type: str = "SubprocessError"
    exc_info: dict = {}

    if lines:
        last_line = lines[-1].strip()
        if ":" in last_line:
            parts = last_line.split(":", 1)
            if len(parts) == 2 and len(parts[0].split()) < 3:
                exc_type = parts[0].strip()
                exc_info["msg"] = parts[1].strip()
        else:
            exc_info["msg"] = last_line

    exc_stack: list[tuple] = []
    raw_lines = stderr.split("\n")
    i = 0
    while i < len(raw_lines):
        match = _FILE_PATTERN.match(raw_lines[i].strip())
        if match:
            lineno = f"line: {match.group(2)}"
            name = f"type: {match.group(3)}"
            source_line = "code: "
            if (
                i + 1 < len(raw_lines)
                and raw_lines[i + 1].strip()
                and not raw_lines[i + 1].strip().startswith("Traceback")
            ):
                source_line += raw_lines[i + 1].strip()
                i += 1
            exc_stack.append((lineno, name, source_line))
        i += 1

    return (
        exc_type,
        exc_info if exc_info else None,
        exc_stack if exc_stack else None,
    )


def extract_metric_lines_from_stdout(stdout: str) -> list[str]:
    """Collect METRIC_VALUE= lines from stdout; fall back to per-line compat patterns."""
    if not stdout:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in stdout.splitlines():
        raw = line.rstrip()
        if not raw:
            continue
        if _METRIC_VALUE_LINE.match(line):
            if raw not in seen:
                seen.add(raw)
                out.append(raw)
    if out:
        return out
    for line in stdout.splitlines():
        raw = line.rstrip()
        if not raw:
            continue
        if _METRIC_PATTERNS.search(raw):
            if raw not in seen:
                seen.add(raw)
                out.append(raw)
    return out


def extract_metric_from_output(stdout: str) -> float | None:
    """Search stdout for common metric patterns and return the last numeric value found."""
    if not stdout:
        return None
    matches = _METRIC_PATTERNS.findall(stdout)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (ValueError, IndexError):
        return None


def extract_error_line(
    code: str, exc_stack: list[tuple] | None,
) -> tuple[int, str] | None:
    """Given source code and an exc_stack from parse_traceback, find the error line
    number and its content in the original code.

    Returns (line_number, line_content) or None if not determinable.
    """
    if not exc_stack or not code:
        return None

    code_lines = code.splitlines()
    last_frame = exc_stack[-1]
    lineno_str = last_frame[0]  # "line: 42"

    match = re.search(r"(\d+)", lineno_str)
    if not match:
        return None

    lineno = int(match.group(1))
    if 1 <= lineno <= len(code_lines):
        return lineno, code_lines[lineno - 1]
    return None


def strip_working_dir_from_content(content: str, working_dir: str) -> str:
    """Replace full working_dir paths in File "..." references with just the filename,
    avoiding leaking full paths in terminal output shown to the LLM."""
    if not content or not (working_dir or "").strip():
        return content
    base = working_dir.rstrip("/\\")
    if not base:
        return content

    import os

    def _repl(m: re.Match) -> str:
        path = m.group(1)
        if path.startswith(base):
            suffix = path[len(base):].lstrip("/\\")
            filename = os.path.basename(suffix) if suffix else path
            return f'File "{filename}"'
        return m.group(0)

    return re.sub(r'File "([^"]*)"', _repl, content)


_RE_ENV_TOKEN_NOTE = re.compile(
    r"Note:\s*Environment variable `[^`]+` is set and is the current active token "
    r"independently from the token you've just configured\.?\s*",
    re.IGNORECASE,
)


def strip_env_token_notes(content: str) -> str:
    """Remove HuggingFace token environment variable notices from output."""
    if not content:
        return content
    return _RE_ENV_TOKEN_NOTE.sub("", content).strip("\n")


def format_compact_exc_stack(
    exc_stack: list[tuple] | None,
    max_frames: int = 6,
    code_max_len: int = 80,
) -> str:
    """Format exc_stack into compact debug info for filtered term_out."""
    if not exc_stack:
        return ""
    lines = []
    for lineno, name, source_line in exc_stack[:max_frames]:
        code = (
            source_line.replace("code:", "").strip()
            if "code:" in source_line.lower()
            else source_line.strip()
        )
        if len(code) > code_max_len:
            code = code[: code_max_len - 3] + "..."
        lines.append(f"  {lineno} {name}: {code}")
    return "\n".join(lines)
