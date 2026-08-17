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

"""Unified project Python / venv resolution for bash subprocesses and CodeRunner."""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("scienceflow")


@lru_cache(maxsize=1)
def resolve_project_python() -> tuple[str, str]:
    """Return (python_executable, bin_directory) for subprocess use.

    Strategy:
    1. Walk up from the ScienceFlow package to find .venv/bin/python next to pyproject.toml
    2. Fallback: use sys.executable and its parent directory
    """
    try:
        import scienceflow as _nf
        pkg_dir = Path(_nf.__file__).resolve().parent
    except Exception:
        exe = sys.executable
        return exe, str(Path(exe).parent)

    for parent in (pkg_dir, *pkg_dir.parents):
        venv_python = parent / ".venv" / "bin" / "python"
        pyproject = parent / "pyproject.toml"
        if venv_python.exists() and pyproject.exists():
            bin_dir = str(parent / ".venv" / "bin")
            logger.debug("Resolved project venv: python=%s bin=%s", venv_python, bin_dir)
            return str(venv_python), bin_dir

    exe = sys.executable
    # Do not .resolve() the exe path: venv ``python`` often symlinks to /usr/bin/python3.x;
    # resolving would point PATH at /usr/bin instead of .venv/bin.
    bin_dir = str(Path(exe).parent)
    logger.debug("No project .venv found; using sys.executable bin: %s", bin_dir)
    return exe, bin_dir


def make_path_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build env dict with the **running** interpreter's bin dir first, then project venv.

    Always prepending ``sys.executable``'s directory ensures ``python`` / ``python3`` in bash
    subprocesses match the ScienceFlow CLI (e.g. ``uv run``), even when ``scienceflow`` is not
    installed editable and :func:`resolve_project_python` cannot find a repo ``.venv``.

    *extra* (e.g. CUDA_VISIBLE_DEVICES) is merged on top.
    """
    # Parent dir without .resolve(): ``.venv/bin/python`` may symlink to system interpreter.
    exe_bin = str(Path(sys.executable).parent)
    _, proj_bin = resolve_project_python()
    bins: list[str] = []
    for b in (exe_bin, proj_bin):
        if b and b not in bins:
            bins.append(b)
    path_prefix = ":".join(bins)
    env: dict[str, str] = {"PATH": path_prefix + ":" + os.environ.get("PATH", "")}
    if hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix:
        env["VIRTUAL_ENV"] = sys.prefix
    if extra:
        env.update(extra)
    return env
