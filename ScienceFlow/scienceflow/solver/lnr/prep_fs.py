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

"""Filesystem helpers for the REPL-native long-horizon solver."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

LEGACY_PREP_SPLIT_NAMES = ("Deep", "Shallow")

_TASK_INSTRUCTION_SUFFIX = (
    "\n\n## Data Structure Note\n"
    "Training and inference data live under the workspace-relative `dataset/` directory. "
    "Treat `dataset/` as the single data root.\n"
)


def has_legacy_prep_split_dirs(root: Path) -> bool:
    """True when *root* has immediate legacy Deep/ and Shallow/ children."""
    return root.is_dir() and all((root / name).is_dir() for name in LEGACY_PREP_SPLIT_NAMES)


def _iter_child_symlinks(root: Path) -> list[tuple[Path, str, Path]]:
    if not root.is_dir():
        return []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    links: list[tuple[Path, str, Path]] = []
    for child in children:
        if not child.is_symlink():
            continue
        try:
            raw_target = os.readlink(child)
        except OSError:
            continue
        target = Path(raw_target)
        target_abs = target if target.is_absolute() else child.parent / target
        links.append((child, raw_target, target_abs))
    return links


def _has_self_referential_child_symlink(root: Path) -> bool:
    for child, _raw_target, target_abs in _iter_child_symlinks(root):
        if os.path.abspath(str(target_abs)) == os.path.abspath(str(child)):
            return True
    return False


def _public_sibling_for_legacy_source(source: Path) -> Path | None:
    if source.name in LEGACY_PREP_SPLIT_NAMES and source.parent.name == "dataset_split":
        public = source.parent.parent / "public"
    elif source.name == "dataset_split":
        public = source.parent / "public"
    else:
        return None
    return public if public.is_dir() else None


def resolve_workspace_dataset_source(source: Path) -> tuple[Path, str]:
    """Return the source directory to expose as flat ``workspace/dataset``.

    Current LNR runs use a single dataset root. Older prepared caches may still
    point at ``prepared/dataset_split`` with ``Deep/`` and ``Shallow/`` children;
    for those, expose ``Deep`` as the canonical full-data source and keep
    ``Shallow`` hidden from the agent-facing workspace.
    Some historical caches contain self-referential symlinks inside ``Deep``;
    in that case fall back to the sibling ``prepared/public`` directory.
    Broken media-directory symlinks inside ``Deep`` are repaired later while
    exposing the workspace dataset so split metadata still comes from ``Deep``.
    """
    if source.name in LEGACY_PREP_SPLIT_NAMES and _has_self_referential_child_symlink(source):
        public = _public_sibling_for_legacy_source(source)
        if public is not None:
            return public, "legacy_split_public_fallback"
    if has_legacy_prep_split_dirs(source):
        deep = source / "Deep"
        if _has_self_referential_child_symlink(deep):
            public = _public_sibling_for_legacy_source(source)
            if public is not None:
                return public, "legacy_split_public_fallback"
        return deep, "legacy_split_deep"
    return source, "flat"


def _workspace_dataset_entry_target(path: Path, source: Path) -> Path:
    if path.is_symlink():
        try:
            raw_target = os.readlink(path)
        except OSError:
            raw_target = ""
        if raw_target:
            target = Path(raw_target)
            target_abs = target if target.is_absolute() else path.parent / target
            if not target_abs.exists():
                public = _public_sibling_for_legacy_source(source)
                candidate = public / path.name if public is not None else None
                if candidate is not None and candidate.is_dir():
                    return candidate.resolve()
    return path.resolve()


def prepare_workspace_dataset_flat(source: Path, workspace_dataset: Path) -> None:
    """Symlink every top-level input entry into workspace/dataset/."""
    if workspace_dataset.exists() or workspace_dataset.is_symlink():
        if workspace_dataset.is_dir() and not workspace_dataset.is_symlink():
            shutil.rmtree(workspace_dataset)
        else:
            workspace_dataset.unlink()
    workspace_dataset.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        target = _workspace_dataset_entry_target(path, source)
        os.symlink(str(target), str(workspace_dataset / path.name))
