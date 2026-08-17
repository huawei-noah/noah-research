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

import hashlib
import json
import os
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scienceflow.solver.lnr.restore_cleanup import clear_directory_contents, remove_path_for_restore
from scienceflow.solver.lnr.runtime_paths import (
    RESTORE_PRESERVED_DIR_NAMES,
    is_restore_preserved_rel,
    restore_preserved_paths,
)
from scienceflow.solver.lnr.workspace_snapshot_io import (
    _atomic_json_write,
    _copy_file_atomic,
    _fsync_dir,
    _is_relative_to,
    _object_path,
    _read_json,
    _safe_rel,
    _sha256_file,
)

_EXCLUDED_DIR_NAMES = frozenset({
    ".git",
    ".snapshots",
    ".estra_archives",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "task_logs",
    "snapshots",
    ".snapshots",
    ".estra_archives",
    ".agent_memory",
    "stage_memory",
    ".logs",
    "logs",
    "tmp",
}) | RESTORE_PRESERVED_DIR_NAMES
_DATASET_DIR_NAMES = frozenset({"dataset"})
_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class WorkspaceSnapshotCaptureResult:
    snapshot_id: str
    manifest_path: Path
    object_store_root: Path
    captured_file_count: int
    symlink_count: int
    logical_size_bytes: int
    new_physical_bytes: int
    new_object_count: int
    reused_object_count: int
    skipped_hash_count: int
    skipped_symlink_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "manifest_path": str(self.manifest_path),
            "object_store_root": str(self.object_store_root),
            "captured_file_count": self.captured_file_count,
            "symlink_count": self.symlink_count,
            "logical_size_bytes": self.logical_size_bytes,
            "new_physical_bytes": self.new_physical_bytes,
            "new_object_count": self.new_object_count,
            "reused_object_count": self.reused_object_count,
            "skipped_hash_count": self.skipped_hash_count,
            "skipped_symlink_count": self.skipped_symlink_count,
        }


@dataclass(frozen=True)
class WorkspaceSnapshotRestoreResult:
    manifest_path: Path
    target_dir: Path
    restored_file_count: int
    restored_symlink_count: int
    restored_size_bytes: int

    def to_json(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "target_dir": str(self.target_dir),
            "restored_file_count": self.restored_file_count,
            "restored_symlink_count": self.restored_symlink_count,
            "restored_size_bytes": self.restored_size_bytes,
        }


class WorkspaceSnapshotStore:
    """Content-addressed workspace snapshot helper.

    The store captures workspace files into immutable SHA-256 objects and
    records a manifest mapping live workspace paths to those objects. Repeated
    captures reuse prior digests when the path fingerprint proves unchanged,
    avoiding repeated reads of very large model artifacts.
    """

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        object_store_dir: str | Path,
        manifest_dir: str | Path,
        data_roots: Iterable[str | Path] = (),
        verify_objects: bool = True,
        extra_excluded_dir_names: Iterable[str] = (),
    ) -> None:
        self.workspace_dir = Path(workspace_dir).expanduser().resolve(strict=False)
        self.object_store_root = Path(object_store_dir).expanduser().resolve(strict=False)
        self.manifest_dir = Path(manifest_dir).expanduser().resolve(strict=False)
        self.data_roots = tuple(Path(p).expanduser().resolve(strict=False) for p in data_roots)
        self.verify_objects = bool(verify_objects)
        extra_excluded = {str(name).strip() for name in extra_excluded_dir_names if str(name).strip()}
        self.excluded_dir_names = frozenset(set(_EXCLUDED_DIR_NAMES) | extra_excluded)
        self.latest_manifest_path = self.manifest_dir / "latest_manifest.json"

    def capture(self, *, snapshot_id: str | None = None) -> WorkspaceSnapshotCaptureResult:
        if not self.workspace_dir.is_dir():
            raise FileNotFoundError(f"workspace does not exist: {self.workspace_dir}")
        sid = self._safe_snapshot_id(snapshot_id or f"snapshot-{int(time.time())}-{uuid.uuid4().hex[:8]}")
        previous_entries = self._load_latest_manifest_entries()
        entries: dict[str, dict[str, Any]] = {}
        logical_size = 0
        new_physical = 0
        new_objects = 0
        reused_objects = 0
        skipped_hashes = 0
        skipped_symlinks = 0
        symlinks = 0

        for path in self._iter_capture_paths():
            rel = _safe_rel(path, self.workspace_dir)
            if path.is_symlink():
                target = os.readlink(path)
                try:
                    self._validate_symlink_target(target)
                except ValueError:
                    if self._is_relative_symlink_escape(target):
                        skipped_symlinks += 1
                        continue
                    raise
                try:
                    st = path.lstat()
                    mode = stat.S_IMODE(st.st_mode)
                    mtime_ns = int(st.st_mtime_ns)
                except OSError:
                    mode = 0
                    mtime_ns = 0
                entries[rel] = {
                    "kind": "symlink",
                    "target": target,
                    "mode": mode,
                    "mtime_ns": mtime_ns,
                }
                symlinks += 1
                continue
            if not path.is_file():
                continue
            st = path.stat()
            size = int(st.st_size)
            fingerprint = self._file_fingerprint(st)
            previous = previous_entries.get(rel) if isinstance(previous_entries.get(rel), dict) else {}
            digest = ""
            created = False
            if (
                previous.get("kind") == "file"
                and previous.get("fingerprint") == fingerprint
                and int(previous.get("size") or -1) == size
            ):
                prior_digest = str(previous.get("digest") or "")
                if prior_digest and _object_path(self.object_store_root, prior_digest).is_file():
                    digest = prior_digest
                    skipped_hashes += 1
            if not digest:
                digest, created = self._store_file(path)
            logical_size += size
            if created:
                new_objects += 1
                new_physical += size
            else:
                reused_objects += 1
            entries[rel] = {
                "kind": "file",
                "digest": digest,
                "size": size,
                "mode": stat.S_IMODE(st.st_mode),
                "mtime_ns": int(st.st_mtime_ns),
                "fingerprint": fingerprint,
            }

        manifest_path = self.manifest_dir / f"{sid}.json"
        manifest = {
            "version": _MANIFEST_VERSION,
            "snapshot_id": sid,
            "created_at": time.time(),
            "workspace_dir": str(self.workspace_dir),
            "object_store_root": str(self.object_store_root),
            "entries": entries,
            "stats": {
                "captured_file_count": sum(1 for row in entries.values() if row.get("kind") == "file"),
                "symlink_count": symlinks,
                "logical_size_bytes": logical_size,
                "new_physical_bytes": new_physical,
                "new_object_count": new_objects,
                "reused_object_count": reused_objects,
                "skipped_hash_count": skipped_hashes,
                "skipped_symlink_count": skipped_symlinks,
            },
        }
        _atomic_json_write(manifest_path, manifest)
        _atomic_json_write(self.latest_manifest_path, {"snapshot_id": sid, "manifest_path": str(manifest_path)})
        return WorkspaceSnapshotCaptureResult(
            snapshot_id=sid,
            manifest_path=manifest_path,
            object_store_root=self.object_store_root,
            captured_file_count=int(manifest["stats"]["captured_file_count"]),
            symlink_count=symlinks,
            logical_size_bytes=logical_size,
            new_physical_bytes=new_physical,
            new_object_count=new_objects,
            reused_object_count=reused_objects,
            skipped_hash_count=skipped_hashes,
            skipped_symlink_count=skipped_symlinks,
        )

    def restore(self, manifest_path: str | Path, *, target_dir: str | Path | None = None) -> WorkspaceSnapshotRestoreResult:
        manifest_file = Path(manifest_path).expanduser().resolve(strict=False)
        manifest = _read_json(manifest_file)
        if int(manifest.get("version") or 0) != _MANIFEST_VERSION:
            raise ValueError(f"unsupported workspace-snapshot manifest version: {manifest.get('version')}")
        entries = manifest.get("entries") if isinstance(manifest.get("entries"), dict) else {}
        self._preflight_restore_entries(entries)
        target = Path(target_dir).expanduser().resolve(strict=False) if target_dir is not None else self.workspace_dir
        target.mkdir(parents=True, exist_ok=True)
        self._clear_target_preserving_git(target)
        restored_files = 0
        restored_symlinks = 0
        restored_size = 0
        for rel in sorted(entries):
            row = entries[rel]
            if not isinstance(row, dict):
                continue
            if is_restore_preserved_rel(rel):
                continue
            dst = target / rel
            kind = str(row.get("kind") or "")
            if kind == "file":
                digest = str(row.get("digest") or "")
                obj = _object_path(self.object_store_root, digest)
                _copy_file_atomic(obj, dst)
                mode = int(row.get("mode") or 0o644)
                try:
                    os.chmod(dst, mode)
                except OSError:
                    pass
                mtime_ns = int(row.get("mtime_ns") or 0)
                if mtime_ns > 0:
                    try:
                        os.utime(dst, ns=(mtime_ns, mtime_ns))
                    except OSError:
                        pass
                restored_files += 1
                restored_size += int(row.get("size") or 0)
            elif kind == "symlink":
                target_text = str(row.get("target") or "")
                self._validate_symlink_target(target_text)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() or dst.is_symlink():
                    remove_path_for_restore(dst, trash_dir=target / ".restore_trash")
                os.symlink(target_text, dst)
                restored_symlinks += 1
            elif kind == "deleted":
                if dst.exists() or dst.is_symlink():
                    remove_path_for_restore(dst, trash_dir=target / ".restore_trash")
        _atomic_json_write(self.latest_manifest_path, {"snapshot_id": manifest.get("snapshot_id"), "manifest_path": str(manifest_file)})
        return WorkspaceSnapshotRestoreResult(
            manifest_path=manifest_file,
            target_dir=target,
            restored_file_count=restored_files,
            restored_symlink_count=restored_symlinks,
            restored_size_bytes=restored_size,
        )

    def _preflight_restore_entries(self, entries: dict[str, Any]) -> None:
        """Validate every referenced object before mutating the restore target."""
        for rel in sorted(entries):
            row = entries[rel]
            if not isinstance(row, dict) or is_restore_preserved_rel(rel):
                continue
            kind = str(row.get("kind") or "")
            if kind == "file":
                digest = str(row.get("digest") or "")
                obj = _object_path(self.object_store_root, digest)
                if not obj.is_file():
                    raise FileNotFoundError(f"workspace_snapshot_missing_object:{digest}:{rel}")
                if self.verify_objects:
                    actual = "sha256:" + _sha256_file(obj)
                    if actual != digest:
                        raise ValueError(f"workspace_snapshot_object_digest_mismatch:{digest}:{rel}")
            elif kind == "symlink":
                self._validate_symlink_target(str(row.get("target") or ""))

    def load_manifest(self, manifest_path: str | Path) -> dict[str, Any]:
        return _read_json(Path(manifest_path))

    def _load_latest_manifest_entries(self) -> dict[str, dict[str, Any]]:
        try:
            pointer = _read_json(self.latest_manifest_path)
            manifest_path = Path(str(pointer.get("manifest_path") or ""))
            if not manifest_path.is_absolute():
                manifest_path = self.manifest_dir / manifest_path
            manifest = _read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        entries = manifest.get("entries")
        return entries if isinstance(entries, dict) else {}

    def _iter_capture_paths(self) -> list[Path]:
        out: list[Path] = []

        def visit(directory: Path) -> None:
            try:
                children = sorted(directory.iterdir(), key=lambda p: p.name)
            except OSError:
                return
            for child in children:
                if self._is_excluded(child):
                    continue
                if child.is_symlink():
                    out.append(child)
                    continue
                if child.is_file():
                    out.append(child)
                    continue
                if child.is_dir():
                    visit(child)

        visit(self.workspace_dir)
        return out

    def _is_excluded(self, path: Path) -> bool:
        try:
            rel = path.relative_to(self.workspace_dir)
        except ValueError:
            return True
        if not rel.parts:
            return False
        first = rel.parts[0]
        if first in self.excluded_dir_names:
            return True
        if first in _DATASET_DIR_NAMES and len(rel.parts) == 1 and path.is_dir():
            return False
        if first in _DATASET_DIR_NAMES and not path.is_symlink():
            return True
        if _is_relative_to(path.resolve(strict=False), self.object_store_root):
            return True
        if _is_relative_to(path.resolve(strict=False), self.manifest_dir):
            return True
        return False

    def _store_file(self, path: Path) -> tuple[str, bool]:
        tmp_root = self.object_store_root / ".tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp = tmp_root / f"obj-{uuid.uuid4().hex}.tmp"
        h = hashlib.sha256()
        try:
            with path.open("rb") as src, tmp.open("wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    h.update(chunk)
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            digest = "sha256:" + h.hexdigest()
            final = _object_path(self.object_store_root, digest)
            final.parent.mkdir(parents=True, exist_ok=True)
            created = False
            if not final.exists():
                try:
                    os.link(tmp, final)
                    os.chmod(final, 0o444)
                    _fsync_dir(final.parent)
                    created = True
                except FileExistsError:
                    created = False
            return digest, created
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def _validate_symlink_target(self, target: str) -> None:
        if not target:
            raise ValueError("empty symlink target is not allowed")
        target_path = Path(target)
        if target_path.is_absolute():
            resolved = target_path.resolve(strict=False)
            if self.data_roots and not any(resolved == root or _is_relative_to(resolved, root) for root in self.data_roots):
                raise ValueError(f"symlink target outside configured data roots: {target}")
            return
        normalized = Path(os.path.normpath(target))
        if normalized.parts and normalized.parts[0] == "..":
            raise ValueError(f"relative symlink escaping workspace is not allowed: {target}")

    @staticmethod
    def _is_relative_symlink_escape(target: str) -> bool:
        if not target:
            return False
        target_path = Path(target)
        if target_path.is_absolute():
            return False
        normalized = Path(os.path.normpath(target))
        return bool(normalized.parts and normalized.parts[0] == "..")

    @staticmethod
    def _safe_snapshot_id(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value or ""))
        return safe.strip("._-")[:120] or f"snapshot-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _file_fingerprint(st: os.stat_result) -> dict[str, Any]:
        return {
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "ctime_ns": int(st.st_ctime_ns),
            "inode": int(st.st_ino),
            "device": int(st.st_dev),
        }

    def _clear_target_preserving_git(self, target: Path) -> None:
        preserve_roots = list(restore_preserved_paths(target))
        for root in (self.object_store_root, self.manifest_dir):
            resolved = root.resolve(strict=False)
            if resolved == target.resolve(strict=False) or _is_relative_to(resolved, target.resolve(strict=False)):
                preserve_roots.append(resolved)
        preserved = tuple(path.resolve(strict=False) for path in preserve_roots)

        clear_directory_contents(target, preserve_paths=preserved, trash_dir=target / ".restore_trash")
