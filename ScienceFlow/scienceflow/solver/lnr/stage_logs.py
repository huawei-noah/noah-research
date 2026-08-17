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

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from scienceflow.utils.workspace_interaction_log import (
    InteractionColorFormatter,
    InteractionPlainFormatter,
)

STAGE_LOG_DIRNAME = ".logs"


def stage_log_dir(workspace_dir: str | Path) -> Path:
    return Path(workspace_dir) / STAGE_LOG_DIRNAME


def ensure_stage_log_dir(workspace_dir: str | Path) -> Path:
    root = stage_log_dir(workspace_dir)
    for rel in (
        ("interaction",),
        ("traj_interaction",),
    ):
        root.joinpath(*rel).mkdir(parents=True, exist_ok=True)
    for rel in (
        ("interaction", "interaction.log"),
        ("traj_interaction", "traj_interaction.log"),
        ("tool.log",),
        ("stage.log",),
    ):
        root.joinpath(*rel).touch(exist_ok=True)
    return root


def reset_stage_log_dir(workspace_dir: str | Path) -> Path:
    root = stage_log_dir(workspace_dir)
    try:
        if root.is_dir() and not root.is_symlink():
            shutil.rmtree(root)
        elif root.exists() or root.is_symlink():
            root.unlink()
    except OSError:
        pass
    root = ensure_stage_log_dir(workspace_dir)
    append_stage_event(root, "reset")
    return root


def append_stage_event(log_dir: str | Path, event: str, **fields: Any) -> None:
    root = Path(log_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
        parts = [f"ts={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", f"event={event}"]
        for key in sorted(fields):
            val = _one_line(fields[key])
            parts.append(f"{key}={val}")
        with (root / "stage.log").open("a", encoding="utf-8") as f:
            f.write(" ".join(parts) + "\n")
    except OSError:
        return


def attach_stage_interaction_handlers(
    logger: logging.Logger | None,
    log_dir: str | Path,
    *,
    color: bool,
) -> None:
    if logger is None:
        return
    root = ensure_stage_log_dir(log_dir)
    fmt: logging.Formatter = InteractionColorFormatter() if color else InteractionPlainFormatter()
    targets = (
        root / "interaction" / "interaction.log",
        root / "traj_interaction" / "traj_interaction.log",
    )
    existing = {
        str(Path(getattr(handler, "baseFilename", "")).resolve(strict=False))
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    }
    for target in targets:
        key = str(target.resolve(strict=False))
        if key in existing:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(target, mode="a", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def copy_stage_logs_to_snapshot(
    *,
    workspace_dir: str | Path,
    snapshot_dir: str | Path,
    warnings: list[str] | None = None,
) -> int:
    src = stage_log_dir(workspace_dir)
    dst = Path(snapshot_dir) / STAGE_LOG_DIRNAME
    if not src.is_dir():
        return 0
    try:
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.copytree(src, dst, symlinks=True, ignore=_ignore_stage_log_copy)
        return sum(1 for p in dst.rglob("*") if p.is_file())
    except OSError as exc:
        if warnings is not None:
            warnings.append(f"stage_log_copy_failed:{type(exc).__name__}")
        return 0


def _ignore_stage_log_copy(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in {"tool_outputs", "__pycache__"}}


def _one_line(value: Any, *, max_chars: int = 500) -> str:
    text = str(value or "").replace("\n", "\\n").replace("\r", "\\r").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 15].rstrip() + "...[truncated]"
    return text.replace(" ", "%20")
