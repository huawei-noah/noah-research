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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scienceflow.solver.lnr.restore_cleanup import remove_path_for_restore
from scienceflow.solver.lnr.runtime_paths import is_restore_preserved_rel
from scienceflow.solver.lnr.workspace_snapshot import (
    WorkspaceSnapshotStore,
    _copy_file_atomic,
    _object_path,
    _sha256_file,
)


@dataclass(frozen=True)
class WorkspaceSnapshotSubsetRestoreResult:
    manifest_path: Path
    target_dir: Path
    restored_file_count: int
    restored_symlink_count: int
    restored_size_bytes: int
    missing_entry_count: int


def restore_manifest_entries(
    store: WorkspaceSnapshotStore,
    manifest_path: str | Path,
    *,
    target_dir: str | Path,
    include_rels: Iterable[str],
) -> WorkspaceSnapshotSubsetRestoreResult:
    """Overlay selected manifest entries without clearing the target workspace."""

    manifest_file = Path(manifest_path).expanduser().resolve(strict=False)
    target = Path(target_dir).expanduser().resolve(strict=False)
    target.mkdir(parents=True, exist_ok=True)
    manifest = store.load_manifest(manifest_file)
    entries = manifest.get("entries") if isinstance(manifest.get("entries"), dict) else {}
    restored_files = 0
    restored_symlinks = 0
    restored_size = 0
    missing = 0

    for rel in sorted({_safe_rel(rel) for rel in include_rels}):
        if not rel:
            continue
        row = entries.get(rel)
        if not isinstance(row, dict):
            missing += 1
            continue
        if is_restore_preserved_rel(rel):
            continue
        dst = target / rel
        kind = str(row.get("kind") or "")
        if kind == "file":
            _restore_file(store, row, rel=rel, dst=dst)
            restored_files += 1
            restored_size += int(row.get("size") or 0)
        elif kind == "symlink":
            _restore_symlink(store, row, dst=dst)
            restored_symlinks += 1
        elif kind == "deleted":
            _remove_path(dst)

    return WorkspaceSnapshotSubsetRestoreResult(
        manifest_path=manifest_file,
        target_dir=target,
        restored_file_count=restored_files,
        restored_symlink_count=restored_symlinks,
        restored_size_bytes=restored_size,
        missing_entry_count=missing,
    )


def _safe_rel(value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe workspace snapshot relpath: {value}")
    return path.as_posix()


def _restore_file(store: WorkspaceSnapshotStore, row: dict, *, rel: str, dst: Path) -> None:
    digest = str(row.get("digest") or "")
    obj = _object_path(store.object_store_root, digest)
    if not obj.is_file():
        raise FileNotFoundError(f"workspace_snapshot_missing_object:{digest}:{rel}")
    if store.verify_objects:
        actual = "sha256:" + _sha256_file(obj)
        if actual != digest:
            raise ValueError(f"workspace_snapshot_object_digest_mismatch:{digest}:{rel}")
    _copy_file_atomic(obj, dst)
    try:
        os.chmod(dst, int(row.get("mode") or 0o644))
    except OSError:
        pass
    mtime_ns = int(row.get("mtime_ns") or 0)
    if mtime_ns > 0:
        try:
            os.utime(dst, ns=(mtime_ns, mtime_ns))
        except OSError:
            pass


def _restore_symlink(store: WorkspaceSnapshotStore, row: dict, *, dst: Path) -> None:
    target_text = str(row.get("target") or "")
    store._validate_symlink_target(target_text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(dst)
    os.symlink(target_text, dst)


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    remove_path_for_restore(path, trash_dir=path.parent / ".restore_trash")
