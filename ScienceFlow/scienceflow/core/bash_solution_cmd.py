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

"""Classification for Python training script runs (shared, cycle-free imports)."""

from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath

# Env assignments that subsample train/val/test rows — must not count as a "bare" full-data run
# for progressive embedded-full-run (see embedded_fullrun progressive_bare_agent_run).
_SUBSAMPLE_ROW_ENV_ASSIGN_RE = re.compile(
    r"(?i)(?:^|[\s;])(?:NROWS|QUICK_TEST_ROWS|QUICK_TEST|SAMPLE_N|LIMIT_ROWS|"
    r"MAX_ROWS|DEBUG_ROWS|DEV_ROWS)\s*=",
)
# argparse / CLI help — not a training run; must not trigger LNR bare-run early exit.
_HELP_FLAG_RE = re.compile(r"(?:^|\s)--?h(?:elp)?\b")


def command_sets_subsample_row_env(cmd: str) -> bool:
    """True if *cmd* assigns an env var that typically subsamples dataset rows."""
    s = (cmd or "").strip()
    if not s:
        return False
    return bool(_SUBSAMPLE_ROW_ENV_ASSIGN_RE.search(s))


def python_script_run_rel_path(cmd: str) -> str | None:
    """Return the relative ``*.py`` script run by ``python`` if it is a full-data run.

    The result is intentionally limited to workspace-relative paths. Absolute
    paths and parent traversal are rejected so LHR can keep agent-visible and
    controller metadata workspace-local.
    """
    s = (cmd or "").strip()
    if not s or ".py" not in s:
        return None
    if command_sets_subsample_row_env(s):
        return None
    if _HELP_FLAG_RE.search(s):
        return None
    try:
        tokens = shlex.split(s, posix=True)
    except ValueError:
        return None
    python_no_arg_flags = {"-B", "-E", "-I", "-O", "-OO", "-P", "-q", "-s", "-S", "-u"}
    python_one_arg_flags = {"-W", "-X"}
    for i, tok in enumerate(tokens):
        exe = PurePosixPath(tok).name.lower()
        if not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", exe):
            continue
        j = i + 1
        while j < len(tokens):
            arg = tokens[j]
            if arg in python_no_arg_flags:
                j += 1
                continue
            if arg in python_one_arg_flags:
                j += 2
                continue
            if arg in {"-c", "-m"}:
                break
            if arg == "--":
                j += 1
                continue
            if arg.startswith("-"):
                break
            script = PurePosixPath(arg)
            if script.suffix == ".py" and not script.is_absolute() and ".." not in script.parts:
                return str(script)
            break
    return None


def looks_like_python_script_run(cmd: str) -> bool:
    """True if bash runs a workspace-relative ``*.py`` script as a full-data run."""

    return python_script_run_rel_path(cmd) is not None


def looks_like_bare_solution_run(cmd: str) -> bool:
    """True if bash runs ``python3 solution.py`` without quick-test / row-subsample env."""

    return python_script_run_rel_path(cmd) == "solution.py"
