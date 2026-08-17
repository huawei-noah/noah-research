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

from pathlib import Path


RESTORE_PRESERVED_DIR_NAMES = frozenset({
    ".git",
    ".agent_memory",
    ".logs",
    ".restore_trash",
    "logs",
    ".memory",
    ".scienceflow_runs",
    ".scienceflow_checkpoints",
    "stage_memory",
    "submission_snapshots",
    "submission_history",
    "submissions",
    "tmp",
})

RESTORE_PRESERVED_REL_PATHS: tuple[Path, ...] = ()


def restore_preserved_paths(workspace_dir: Path) -> tuple[Path, ...]:
    """Return live runtime/control paths that restore must never delete."""
    workspace = Path(workspace_dir)
    dir_paths = tuple(workspace / name for name in sorted(RESTORE_PRESERVED_DIR_NAMES))
    rel_paths = tuple(workspace / rel for rel in RESTORE_PRESERVED_REL_PATHS)
    return dir_paths + rel_paths


def is_restore_preserved_rel(rel: str | Path) -> bool:
    """Return whether a snapshot relpath is reserved by the live runtime."""
    raw = str(rel or "").replace("\\", "/").strip()
    if not raw:
        return False
    path = Path(raw)
    parts = path.parts
    if parts and parts[0] in RESTORE_PRESERVED_DIR_NAMES:
        return True
    for preserved in RESTORE_PRESERVED_REL_PATHS:
        if path == preserved or _is_relative_to(path, preserved) or _is_relative_to(preserved, path):
            return True
    return False


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
