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

"""Environment selection for command evaluator subprocesses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def task_command_env(ctx: Any, *, python: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    selected = python or task_python_executable(ctx)
    if not selected:
        return env
    bin_dir = selected.parent
    env_root = bin_dir.parent
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["CONDA_PREFIX"] = str(env_root)
    if (env_root / "pyvenv.cfg").exists():
        env["VIRTUAL_ENV"] = str(env_root)
    else:
        env.pop("VIRTUAL_ENV", None)
    return env


def task_python_executable(ctx: Any) -> Path | None:
    raw = _str_cfg(ctx, ("command", "python_executable"), "").strip()
    if not raw:
        raw = str(getattr(ctx, "task_python_executable", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve(strict=False)
    env_path = _str_cfg(ctx, ("command", "environment_path"), "").strip()
    if not env_path:
        return None
    root = Path(env_path).expanduser().resolve(strict=False)
    for rel in ("bin/python", "Scripts/python.exe", "python"):
        candidate = root / rel
        if candidate.exists():
            return candidate
    return root / "bin" / "python"


def _str_cfg(ctx: Any, path: tuple[str, str], default: str) -> str:
    value = _nested_cfg(ctx, path, default)
    return str(value if value is not None else default)


def _nested_cfg(ctx: Any, path: tuple[str, str], default: Any) -> Any:
    evaluator = _get_cfg(ctx.cfg, "evaluator", None)
    block = _get_cfg(evaluator, path[0], None)
    value = _get_cfg(block, path[1], None)
    if value is not None:
        return value
    return _get_cfg(ctx.cfg, path[1], default)


def _get_cfg(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)
