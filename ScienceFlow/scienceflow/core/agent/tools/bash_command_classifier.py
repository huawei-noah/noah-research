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

"""Lightweight semantic labels for bash commands in logs and traces."""

from __future__ import annotations

import re

_HEREDOC_WRITE_RE = re.compile(r"(^|\n)\s*cat\s*>\s*[^\n]+<<")
_HEREDOC_APPEND_RE = re.compile(r"(^|\n)\s*cat\s*>>\s*[^\n]+<<")
_CP_RE = re.compile(r"(^|\n)\s*cp\s+")
_MV_RE = re.compile(r"(^|\n)\s*mv\s+")
_INSTALL_RE = re.compile(r"\b(uv\s+pip\s+install|pip3?\s+install|apt-get\s+install|apt\s+install)\b")
_GIT_RE = re.compile(r"(^|\n)\s*git\s+")
_PY_SOLUTION_RE = re.compile(r"\bpython(?:3)?\b[^\n;|&]*\bsolution\.py\b")
_PY_HEREDOC_RE = re.compile(r"(^|\n)\s*python(?:3)?\s*-\s*<<")
_READ_TEXT_RE = re.compile(r"\.read_text\s*\(")
_WRITE_TEXT_RE = re.compile(r"\.write_text\s*\(")
_OPEN_WRITE_RE = re.compile(r"\bopen\s*\([^)]*,\s*['\"](?:w|a|x)")
_PANDAS_WRITE_RE = re.compile(r"\.to_(?:csv|parquet|json|pickle|feather|xlsx)\s*\(")


def classify_bash_command(command: str) -> str:
    """Return a stable, coarse action label for a bash command.

    The label is for downstream analysis only. It does not affect tool execution
    or the model-facing tool schema.
    """
    cmd = (command or "").strip()
    if not cmd:
        return "empty"
    low = cmd.lower()

    if _HEREDOC_APPEND_RE.search(cmd):
        return "append"
    if _HEREDOC_WRITE_RE.search(cmd):
        return "write"

    if _PY_HEREDOC_RE.search(cmd):
        reads = bool(_READ_TEXT_RE.search(cmd))
        writes = bool(_WRITE_TEXT_RE.search(cmd) or _OPEN_WRITE_RE.search(cmd))
        if reads and writes:
            return "edit"
        if writes or _PANDAS_WRITE_RE.search(cmd):
            return "write"
        return "python_inline"

    if _CP_RE.search(cmd):
        return "copy"
    if _MV_RE.search(cmd):
        return "move"
    if _INSTALL_RE.search(low):
        return "install"
    if _GIT_RE.search(cmd):
        return "git"
    if _PY_SOLUTION_RE.search(cmd):
        return "run_solution"
    if "mlebench" in low or "kaggle" in low or "submission" in low:
        return "evaluate"
    if re.search(r"(^|\n)\s*(ls|find|du|wc|stat)\b", cmd):
        return "inspect"
    if re.search(r"(^|\n)\s*(python|python3|uv\s+run\s+python)\b", cmd):
        return "python"
    return "other"
