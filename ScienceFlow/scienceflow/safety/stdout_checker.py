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

"""Detect stdout-hostile patterns in agent-produced solution code.

Catches patterns that break ``asyncio.StreamReader.readline()`` in the
execution safety policy - most notably **tqdm** which uses ``\\r`` (carriage
return) instead of ``\\n``, producing "single lines" that can exceed the
stream buffer limit and crash the agent loop.

Design mirrors ``leakage_detector``: returns a list of human-readable
issue strings (empty = clean).
"""

from __future__ import annotations

import ast
import re

# =====================================================================
# Pattern: tqdm usage
# =====================================================================

_RE_IMPORT_TQDM = re.compile(
    r"^\s*(?:from\s+tqdm\b.*import|import\s+tqdm\b)", re.MULTILINE
)

_RE_TQDM_CALL = re.compile(r"\btqdm(?:\.tqdm)?\s*\(")

# =====================================================================
# Pattern: keras/sklearn verbose=1 (high-frequency per-batch logging)
# =====================================================================

_RE_VERBOSE_ONE = re.compile(r"\bverbose\s*=\s*1\b")

# =====================================================================
# Public API
# =====================================================================


def check_stdout_hostile(code: str) -> list[str]:
    """Return issue strings for stdout-hostile patterns found in *code*.

    Currently checks:
    * ``tqdm`` import or call — ``\\r``-based progress breaks stream reading.
    * ``verbose=1`` — high-frequency per-batch logging floods output.

    Returns an empty list when the code is clean.
    """
    if not (code or "").strip():
        return []

    issues: list[str] = []

    if _RE_IMPORT_TQDM.search(code) or _RE_TQDM_CALL.search(code):
        issues.append(
            "tqdm usage detected — its \\r-based progress output can crash "
            "the execution safety policy (use print() for progress instead)"
        )

    live_lines = [
        ln for ln in code.splitlines() if not ln.lstrip().startswith("#")
    ]
    code_nc = "\n".join(live_lines)
    if _RE_VERBOSE_ONE.search(code_nc):
        issues.append(
            "verbose=1 detected — high-frequency per-batch logging floods "
            "stdout and may degrade safety monitoring performance"
        )

    return issues


def strip_tqdm(code: str) -> str:
    """Best-effort removal of tqdm from *code*, returning patched source.

    Transforms:
    * ``from tqdm import tqdm``          → removed
    * ``from tqdm.auto import tqdm``     → removed
    * ``import tqdm``                    → removed
    * ``tqdm(iterable, ...)``            → ``iterable``  (keeps first arg)
    * ``tqdm.tqdm(iterable, ...)``       → ``iterable``

    The result is syntactically valid in common cases.  Falls back to the
    original *code* if AST parsing of the result fails.
    """
    if not (code or "").strip():
        return code

    # Phase 1: remove import lines
    cleaned = re.sub(
        r"^\s*from\s+tqdm(?:\.\w+)*\s+import\s+[^\n]+\n?",
        "",
        code,
        flags=re.MULTILINE,
    )
    cleaned = re.sub(
        r"^\s*import\s+tqdm\b[^\n]*\n?",
        "",
        cleaned,
        flags=re.MULTILINE,
    )

    # Phase 2: replace tqdm(...) / tqdm.tqdm(...) calls with the first
    # positional argument.  Process matches in reverse order so that
    # earlier offsets remain valid after each replacement.
    _PAT_TQDM_CALL = re.compile(r"\btqdm(?:\.tqdm)?\s*\(")
    for m in reversed(list(_PAT_TQDM_CALL.finditer(cleaned))):
        open_paren = m.end() - 1
        depth = 0
        close_paren = open_paren
        for close_paren in range(open_paren, len(cleaned)):
            if cleaned[close_paren] == "(":
                depth += 1
            elif cleaned[close_paren] == ")":
                depth -= 1
                if depth == 0:
                    break
        if depth != 0:
            continue
        inner = cleaned[open_paren + 1 : close_paren].strip()
        first_arg = _extract_first_arg(inner)
        cleaned = cleaned[: m.start()] + first_arg + cleaned[close_paren + 1 :]

    try:
        ast.parse(cleaned)
    except SyntaxError:
        return code

    return cleaned


def _extract_first_arg(args_str: str) -> str:
    """Extract the first positional argument from a comma-separated arg string,
    respecting nested brackets and string literals."""
    depth = 0
    in_str: str | None = None
    for i, ch in enumerate(args_str):
        if in_str:
            if ch == in_str and (i == 0 or args_str[i - 1] != "\\"):
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch in ("(", "[", "{"):
            depth += 1
        elif ch in (")", "]", "}"):
            depth -= 1
        elif ch == "," and depth == 0:
            return args_str[:i].strip()
    return args_str.strip()
