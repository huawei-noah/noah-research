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

"""Bash command classification and quick-test detection."""

from __future__ import annotations

import re

from scienceflow.core.bash_solution_cmd import (
    command_sets_subsample_row_env,
    looks_like_bare_solution_run as _looks_like_bare_solution_run,
    looks_like_python_script_run as _looks_like_python_script_run,
    python_script_run_rel_path as _python_script_run_rel_path,
)
from scienceflow.core.agent.shared.constants import _EMBEDDED_FAIL_STDERR_CAP
from scienceflow.core.tools.write_placeholder import looks_like_write_placeholder_mimicry

__all__ = [
    "_bash_command_parallel_safe",
    "_embedded_failure_user_message",
    "_looks_like_bare_solution_run",
    "_looks_like_python_script_run",
    "_looks_like_quick_test_solution_run",
    "_looks_like_write_placeholder_mimicry",
    "_python_script_run_rel_path",
    "_quick_test_output_has_traceback",
    "classify_runtime_error",
    "command_sets_subsample_row_env",
]


def _looks_like_write_placeholder_mimicry(content: str) -> bool:
    """True if ``content`` is a log/memory placeholder echoed as ``write`` payload (model mimicry)."""

    return looks_like_write_placeholder_mimicry(content)


def _embedded_failure_user_message(stderr: str, exit_code: int | None) -> str:
    tail = stderr
    if len(tail) > _EMBEDDED_FAIL_STDERR_CAP:
        tail = tail[:_EMBEDDED_FAIL_STDERR_CAP] + "\n...[truncated]...\n"
    return (
        f"[Guard] Full `python3 solution.py` failed. exit_code={exit_code!r}. "
        "Fix `solution.py` so the full run succeeds. "
        "For faster iteration during the fix, use argparse knobs your solution exposes, "
        "e.g. `python3 solution.py --epochs 1` or `python3 solution.py --num_samples 20`.\n\n"
        "=== stderr (excerpt) ===\n"
        f"{tail}"
        "\n\nBefore patching, inspect `solution.py` broadly enough to understand "
        "the data flow end-to-end, not just the crash site."
    )


def _looks_like_quick_test_solution_run(cmd: str) -> bool:
    """True if bash runs ``solution.py`` with QUICK_TEST_ROWS (quick-test pass)."""
    s = (cmd or "").strip()
    if not s or "solution.py" not in s:
        return False
    if "QUICK_TEST_ROWS" not in s and "QUICK_TEST" not in s:
        return False
    low = s.lower()
    return "python3" in low or "python " in low


_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\)|"
    r"(?:^|\n)\s*(?:ValueError|RuntimeError|AssertionError|KeyError|"
    r"AttributeError|TypeError|IndexError|FileNotFoundError|OSError)"
    r"\s*:",
    re.MULTILINE,
)

# Exception class names for :func:`classify_runtime_error` (must match ``Name:`` at line start).
_KNOWN_RUNTIME_EXCEPTION_TYPES = (
    "ValueError",
    "RuntimeError",
    "AssertionError",
    "KeyError",
    "AttributeError",
    "TypeError",
    "IndexError",
    "FileNotFoundError",
    "OSError",
    "ZeroDivisionError",
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "StopIteration",
    "NotImplementedError",
    "MemoryError",
    "RecursionError",
    "IntCastingNaNError",
)

_RUNTIME_EXC_LINE_RE = re.compile(
    r"^\s*(" + "|".join(_KNOWN_RUNTIME_EXCEPTION_TYPES) + r")\s*:",
    re.MULTILINE,
)


def classify_runtime_error(output: str) -> str | None:
    """Extract a Python exception class name from bash stdout/stderr (traceback or one-liner).

    Returns the **last** matching ``ExceptionName:`` in the string (typically the leaf error in a
    traceback). Returns ``None`` if no known exception line is found.
    """
    if not isinstance(output, str) or not output.strip():
        return None
    # pandas / numpy errors often appear as ``pandas.errors.IntCastingNaNError:`` (dotted name).
    if "IntCastingNaNError" in output:
        return "IntCastingNaNError"
    matches = list(_RUNTIME_EXC_LINE_RE.finditer(output))
    if matches:
        return matches[-1].group(1)
    return None


def _quick_test_output_has_traceback(output: str) -> bool:
    """Return True if quick-test stdout/stderr contains a Python exception traceback."""
    return bool(_TRACEBACK_RE.search(output or ""))


def _bash_command_parallel_safe(command: str) -> bool:
    """True if *command* may run concurrently with other bash tools in the same LLM round.

    Conservative: python/pip/git/install/network/rm/sudo patterns force sequential execution.
    Splits on ``&&`` / ``||`` / ``;`` so ``grep foo pip`` is not treated as pip.
    """
    import re

    s = (command or "").strip()
    if not s:
        return False
    low = s.lower()
    if _looks_like_quick_test_solution_run(s):
        return False
    for seg in re.split(r"\s*(?:&&|\|\||;)\s*", low):
        seg = seg.strip()
        if not seg:
            continue
        toks = seg.split()
        head = toks[0]
        if head in ("python", "python3", "pip", "pip3"):
            return False
        if head == "uv":
            return False
    if re.search(r"\b(?:conda|mamba|npm|npx|yarn|pnpm)\b", low):
        return False
    if re.search(r"\b(?:git|svn)\b", low):
        return False
    if re.search(r"\b(?:sudo|su)\b", low):
        return False
    if re.search(r"\b(?:docker|kubectl|helm)\b", low):
        return False
    if re.search(r"\b(?:wget|curl)\b", low):
        return False
    if re.search(r"\brm\b", low):
        return False
    if re.search(r"\b(?:apt|apt-get|yum|dnf|apk)\b", low):
        return False
    return True
