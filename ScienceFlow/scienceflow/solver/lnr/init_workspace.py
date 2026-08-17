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

import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXCLUDED_DIR_NAMES = frozenset({
    ".git",
    ".agent_memory",
    ".memory",
    ".logs",
    "logs",
    ".scienceflow_runs",
    ".scienceflow_checkpoints",
    ".snapshots",
    "snapshots",
    ".estra_archives",
    "stage_memory",
    "submission_snapshots",
    "submission_history",
    "submissions",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "dataset",
})

INITIAL_WORKSPACE_STATE_REL = Path(".logs") / "initial_workspace_state.md"
DEFAULT_CONTEXT_MAX_CHARS = 8 * 1024


@dataclass(frozen=True)
class InitWorkspaceResult:
    applied: bool
    source_workspace: str = ""
    copied_file_count: int = 0
    skipped_entry_count: int = 0
    initial_workspace_state: str = ""
    initial_workspace_state_path: str = ""


def load_initial_workspace_state(source_workspace: str | Path, *, max_chars: int = DEFAULT_CONTEXT_MAX_CHARS) -> tuple[str, str]:
    source = Path(source_workspace).expanduser().resolve(strict=False)
    path = source / INITIAL_WORKSPACE_STATE_REL
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", ""
    text = text.strip()
    if not text:
        return "", str(path)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n[truncated: initial workspace state exceeded limit]"
    return text, str(path)


def initialize_workspace_from_path(
    *,
    source_workspace: str | Path,
    target_workspace: str | Path,
    context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    excluded_dir_names: frozenset[str] = DEFAULT_EXCLUDED_DIR_NAMES,
) -> InitWorkspaceResult:
    source = Path(source_workspace).expanduser().resolve(strict=False)
    target = Path(target_workspace).expanduser().resolve(strict=False)
    if not source.is_dir():
        raise FileNotFoundError(f"init_workspace source does not exist or is not a directory: {source}")
    if source == target:
        state, state_path = load_initial_workspace_state(source, max_chars=context_max_chars)
        return InitWorkspaceResult(
            applied=False,
            source_workspace=str(source),
            initial_workspace_state=state,
            initial_workspace_state_path=state_path,
        )

    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for child in sorted(source.iterdir(), key=lambda p: p.name):
        if child.name in excluded_dir_names:
            skipped += 1
            continue
        child_copied, child_skipped = _copy_entry_tree(child, target / child.name, source, excluded_dir_names)
        copied += child_copied
        skipped += child_skipped

    state, state_path = load_initial_workspace_state(source, max_chars=context_max_chars)
    return InitWorkspaceResult(
        applied=True,
        source_workspace=str(source),
        copied_file_count=copied,
        skipped_entry_count=skipped,
        initial_workspace_state=state,
        initial_workspace_state_path=state_path,
    )


def _copy_entry_tree(src: Path, dst: Path, source_root: Path, excluded_dir_names: frozenset[str]) -> tuple[int, int]:
    if src.name in excluded_dir_names:
        return 0, 1
    if src.is_symlink():
        return _copy_symlink(src, dst, source_root)
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.copy2(src, dst)
        return 1, 0
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        skipped = 0
        for child in sorted(src.iterdir(), key=lambda p: p.name):
            child_copied, child_skipped = _copy_entry_tree(child, dst / child.name, source_root, excluded_dir_names)
            copied += child_copied
            skipped += child_skipped
        return copied, skipped
    return 0, 1


def _copy_symlink(src: Path, dst: Path, source_root: Path) -> tuple[int, int]:
    try:
        resolved = src.resolve(strict=True)
        resolved.relative_to(source_root)
    except (OSError, ValueError):
        return 0, 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    target = Path(readlink_text(src))
    if target.is_absolute():
        if not resolved.is_file():
            return 0, 1
        shutil.copy2(resolved, dst)
    else:
        dst.symlink_to(target)
    return 1, 0


def readlink_text(path: Path) -> str:
    return path.readlink().as_posix()
