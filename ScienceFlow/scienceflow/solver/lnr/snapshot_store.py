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
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scienceflow.solver.lnr.restore_cleanup import clear_directory_contents
from scienceflow.solver.lnr.runtime_paths import restore_preserved_paths
from scienceflow.solver.lnr.stage_logs import copy_stage_logs_to_snapshot
from scienceflow.solver.lnr.workspace_snapshot import WorkspaceSnapshotStore
from scienceflow.solver.lnr.workspace_snapshot_io import workspace_snapshot_object_path
from scienceflow.solver.lnr.workspace_snapshot_subset import restore_manifest_entries

_INFRASTRUCTURE_DIR_NAMES = frozenset({
    ".git",
    ".snapshots",
    ".estra_archives",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "task_logs",
    "snapshots",
    ".agent_memory",
    "stage_memory",
})


_MINIMAL_FILE_NAMES = frozenset({
    "submission.csv",
    "run_results.md",
    "result.md",
    "requirements.txt",
})

_MINIMAL_FILE_SUFFIXES = frozenset({
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".joblib",
})

_EXCLUDED_DIR_NAMES = {
    ".git",
    ".agent_memory",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "snapshots",
    "stage_memory",
    "tmp",
}
_EXCLUDED_FILE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".dcm",
    ".tfrecord",
    ".tfrec",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
}
_WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".joblib"}
_WORKSPACE_SNAPSHOT_COMPAT_FILE_NAMES = frozenset({
    "submission.csv",
    "run_results.md",
    ".run_results.md",
    "result.md",
    "requirements.txt",
})
_WORKSPACE_SNAPSHOT_COMPAT_SUFFIXES = frozenset({
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
})
_WORKSPACE_SNAPSHOT_COMPAT_MAX_BYTES = 32 * 1024 * 1024
_FINGERPRINT_HASH_MAX_BYTES = 32 * 1024 * 1024
_FINGERPRINT_HASH_SUFFIXES = frozenset({
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
})


@dataclass(frozen=True)
class StageSnapshot:
    stage_id: str
    snapshot_id: str
    snapshot_path: Path
    metric_value: float | None
    metric_name: str
    lower_is_better: bool | None
    memory_cut: int
    source_event: dict[str, Any]
    node_uid: str = ""
    lineage_id: str = ""
    snapshot_mode: str = "changed_files"
    snapshot_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TerminalArchive:
    archive_path: Path
    manifest_path: Path | None
    object_store_root: Path | None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_minimal_workspace(src: Path, dst: Path, *, warnings: list[str]) -> None:
    """Best-effort source/artifact snapshot used only after changed-file copy failure."""

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if name not in _INFRASTRUCTURE_DIR_NAMES]
        rel_root = root_path.relative_to(src)
        in_memory = rel_root.parts[:1] == (".memory",)
        in_logs = rel_root.parts[:1] == (".logs",)
        for name in files:
            source = root_path / name
            rel = source.relative_to(src)
            if any(part in _INFRASTRUCTURE_DIR_NAMES for part in rel.parts):
                continue
            if not (
                in_memory
                or in_logs
                or name in _MINIMAL_FILE_NAMES
                or source.suffix.lower() in _MINIMAL_FILE_SUFFIXES
            ):
                continue
            target = dst / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied += 1
            except OSError as exc:
                warnings.append(f"minimal_copy_file_failed:{rel}:{type(exc).__name__}")
                continue
    warnings.append(f"minimal_snapshot_file_count:{copied}")


def _copy_entry(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_symlink():
        os.symlink(os.readlink(src), dst)
    else:
        shutil.copy2(src, dst)


def _entry_fingerprint(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink():
            stat = path.lstat()
            return {
                "type": "symlink",
                "target": os.readlink(path),
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        if path.is_file():
            stat = path.stat()
            fp = {
                "type": "file",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
            if (
                stat.st_size <= _FINGERPRINT_HASH_MAX_BYTES
                and (path.name in _MINIMAL_FILE_NAMES or path.suffix.lower() in _FINGERPRINT_HASH_SUFFIXES)
            ):
                try:
                    fp["sha256"] = _sha256_file(path)
                except OSError:
                    pass
            return fp
    except OSError:
        return None
    return None


def _copy_tree_overlay(src: Path, dst: Path, *, exclude_paths: tuple[Path, ...] = ()) -> None:
    excluded = tuple(p.resolve(strict=False) for p in exclude_paths)

    def excluded_path(candidate: Path) -> bool:
        resolved = candidate.resolve(strict=False)
        return any(resolved == root or _is_relative_to(resolved, root) for root in excluded)

    def visit(directory: Path) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for child in children:
            if excluded_path(child):
                continue
            rel = child.relative_to(src)
            target = dst / rel
            if child.is_symlink() or child.is_file():
                _copy_entry(child, target)
            elif child.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                visit(child)

    visit(src)


class SnapshotStore:
    """Changed-file workspace snapshot store for lnr stages."""

    def __init__(
        self,
        *,
        root_dir: Path,
        workspace_dir: Path,
        snapshot_dirname: str,
        archive_dirname: str,
        fallback_root_dir: Path | None = None,
        control_log_dir: Path | None = None,
        metadata_dirname: str = ".logs",
        strict_layout: bool = False,
        memory_dir: Path | None = None,
        workspace_snapshot_enabled: bool = True,
        workspace_snapshot_verify_objects: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.workspace_dir = Path(workspace_dir)
        self.snapshot_dirname = snapshot_dirname or ".snapshots"
        self.archive_dirname = archive_dirname or f"{self.snapshot_dirname}/archives"
        self.snapshot_root = self.root_dir / self.snapshot_dirname
        self.archive_root = self.root_dir / self.archive_dirname
        self.fallback_root = Path(fallback_root_dir) if fallback_root_dir is not None else None
        self.control_log_dir = Path(control_log_dir) if control_log_dir is not None else self.root_dir / metadata_dirname
        self.metadata_dirname = str(metadata_dirname or ".logs").strip() or ".logs"
        self.strict_layout = bool(strict_layout)
        self.memory_dir = Path(memory_dir) if memory_dir is not None else None
        self.workspace_snapshot_enabled = bool(workspace_snapshot_enabled)
        self.workspace_snapshot_verify_objects = bool(workspace_snapshot_verify_objects)
        self._excluded_dir_names = set(_EXCLUDED_DIR_NAMES)
        if self.strict_layout:
            self._excluded_dir_names.update({
                ".logs",
                "logs",
                ".agent_memory",
                ".memory",
                ".scienceflow_checkpoints",
                "snapshots",
                "stage_memory",
                "submission_snapshots",
                "submission_history",
                "submissions",
            })
        self.baseline_manifest_path = self.control_log_dir / "lnr_snapshot_baseline.json"

    @staticmethod
    def _ensure_writable_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write-test-{uuid.uuid4().hex}.tmp"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return path

    def initialize_baseline(self) -> bool:
        """Persist the workspace baseline used to snapshot agent-created changes."""
        try:
            entries = self._scan_workspace(for_baseline=True)
            payload = {
                "version": 1,
                "created_at": time.time(),
                "workspace_dir": str(self.workspace_dir),
                "entries": entries,
            }
            self.baseline_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.baseline_manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return True
        except OSError:
            return False

    def _load_baseline_entries(self) -> dict[str, dict[str, Any]] | None:
        try:
            payload = json.loads(self.baseline_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        entries = payload.get("entries") if isinstance(payload, dict) else None
        return entries if isinstance(entries, dict) else None

    def _snapshot_root_key(self) -> str:
        raw = str(self.root_dir.resolve(strict=False)).encode("utf-8", errors="replace")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _snapshot_root_candidates(self) -> list[Path]:
        candidates: list[Path] = [self.root_dir / self.snapshot_dirname]
        if self.fallback_root is not None:
            candidates.append(self.fallback_root)
        if not self.strict_layout:
            candidates.extend([
                self.workspace_dir / self.snapshot_dirname,
                self.root_dir / ".logs" / "stage_snapshots",
                Path(tempfile.gettempdir()) / "scienceflow_stage_snapshots" / self._snapshot_root_key(),
            ])
        out: list[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            if any(existing.resolve(strict=False) == resolved for existing in out):
                continue
            out.append(candidate)
        return out

    def _ensure_snapshot_root(self, *, skip: tuple[Path, ...] = ()) -> Path:
        first_exc: OSError | None = None
        skip_resolved = tuple(p.resolve(strict=False) for p in skip)
        for candidate in self._snapshot_root_candidates():
            resolved = candidate.resolve(strict=False)
            if any(resolved == blocked for blocked in skip_resolved):
                continue
            try:
                self._ensure_writable_dir(candidate)
                self.snapshot_root = candidate
                return candidate
            except OSError as exc:
                if first_exc is None:
                    first_exc = exc
                continue
        if first_exc is not None:
            raise first_exc
        raise PermissionError(f"no usable snapshot root for {self.root_dir}")

    def _fallback_snapshot_root(self, primary_exc: OSError, *, skip: tuple[Path, ...] = ()) -> Path:
        try:
            return self._ensure_snapshot_root(skip=skip)
        except OSError:
            raise primary_exc

    def _is_excluded_snapshot_path(self, path: Path, *, for_baseline: bool) -> bool:
        try:
            rel = path.relative_to(self.workspace_dir)
        except ValueError:
            return True
        parts = rel.parts
        if not parts:
            return False
        first = parts[0]
        if first in {self.snapshot_dirname, self.archive_dirname}:
            return True
        if first in self._excluded_dir_names:
            return True
        if first == "dataset":
            if not for_baseline:
                return True
            if len(parts) == 1:
                return False
            return not path.is_symlink()
        if path.is_file() and path.suffix.lower() in _EXCLUDED_FILE_SUFFIXES:
            return True
        return False

    def _scan_workspace(self, *, for_baseline: bool) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}

        def visit(directory: Path) -> None:
            try:
                children = sorted(directory.iterdir(), key=lambda p: p.name)
            except OSError:
                return
            for child in children:
                if self._is_excluded_snapshot_path(child, for_baseline=for_baseline):
                    continue
                rel = _safe_rel(child, self.workspace_dir)
                fp = _entry_fingerprint(child)
                if fp is not None:
                    entries[rel] = fp
                    continue
                if child.is_dir():
                    visit(child)

        if self.workspace_dir.is_dir():
            visit(self.workspace_dir)
        return entries

    def _selected_snapshot_entries(self) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, dict[str, Any]] | None]:
        baseline = self._load_baseline_entries()
        current_snapshot_scope = self._scan_workspace(for_baseline=False)
        current_baseline_scope = self._scan_workspace(for_baseline=True)
        if baseline is None:
            return current_snapshot_scope, [], None
        selected = {rel: fp for rel, fp in current_snapshot_scope.items() if baseline.get(rel) != fp}
        deleted = sorted(rel for rel in baseline if rel not in current_baseline_scope)
        return selected, deleted, baseline

    def _copy_snapshot_payload(self, dst: Path, entries: dict[str, dict[str, Any]]) -> None:
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        for rel in sorted(entries):
            src = self.workspace_dir / rel
            if not src.exists() and not src.is_symlink():
                continue
            _copy_entry(src, dst / rel)

    def _materialize_snapshot_dir(
        self,
        *,
        snapshot_root: Path,
        snapshot_id: str,
        selected_entries: dict[str, dict[str, Any]],
    ) -> tuple[Path, bool, str]:
        final_dir = snapshot_root / snapshot_id
        tmp_dir = snapshot_root / f".{snapshot_id}.tmp"
        try:
            self._copy_snapshot_payload(tmp_dir, selected_entries)
            tmp_dir.rename(final_dir)
            return final_dir, True, ""
        except OSError as exc:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            try:
                final_dir.mkdir(parents=True, exist_ok=True)
                return final_dir, False, f"{type(exc).__name__}: {exc}"
            except OSError:
                raise exc

    def _materialize_minimal_snapshot(
        self,
        *,
        snapshot_root: Path,
        snapshot_id: str,
        warnings: list[str],
    ) -> Path:
        final_dir = snapshot_root / snapshot_id
        tmp_dir = snapshot_root / f".{snapshot_id}.tmp"
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        _copy_minimal_workspace(self.workspace_dir, tmp_dir, warnings=warnings)
        tmp_dir.rename(final_dir)
        return final_dir

    def _workspace_has_git(self) -> bool:
        git_path = self.workspace_dir / ".git"
        return git_path.exists() or git_path.is_symlink()

    def _clear_workspace_for_restore_preserving_git(self, *, preserve_paths: tuple[Path, ...] = ()) -> None:
        """Clear workspace contents for restore without touching live control dirs."""
        if not self.workspace_dir.exists():
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            return
        runtime_paths = list(restore_preserved_paths(self.workspace_dir))
        preserved = [self.workspace_dir / ".git", *runtime_paths, *preserve_paths]
        preserved_resolved = tuple(path.resolve(strict=False) for path in preserved)

        clear_directory_contents(
            self.workspace_dir,
            preserve_paths=preserved_resolved,
            trash_dir=self.root_dir / ".restore_trash",
        )

    def _remove_strict_workspace_control_dirs(self) -> None:
        """Legacy hook kept for compatibility; restore now preserves live control dirs."""
        return

    def _workspace_snapshot_store(self, snapshot_root: Path) -> WorkspaceSnapshotStore:
        return WorkspaceSnapshotStore(
            workspace_dir=self.workspace_dir,
            object_store_dir=snapshot_root / "objects",
            manifest_dir=snapshot_root / "manifests",
            verify_objects=self.workspace_snapshot_verify_objects,
            extra_excluded_dir_names=(self._excluded_dir_names if self.strict_layout else ()),
        )

    def _archive_snapshot_root(self, snapshot_path: Path) -> Path:
        try:
            root = self.snapshot_root.resolve(strict=False)
            candidate = snapshot_path.resolve(strict=False)
            if candidate == root or _is_relative_to(candidate, root):
                return self.snapshot_root
        except OSError:
            pass
        return snapshot_path.parent

    def _capture_terminal_archive(self, *, stage_id: str, snapshot_path: Path) -> TerminalArchive:
        archive_id = f"terminal-before-{stage_id}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        archive_path = self.archive_root / archive_id
        if archive_path.exists():
            shutil.rmtree(archive_path, ignore_errors=True)
        archive_path.mkdir(parents=True, exist_ok=True)
        if not self.workspace_dir.is_dir():
            return TerminalArchive(archive_path=archive_path, manifest_path=None, object_store_root=None)

        snapshot_root = self._archive_snapshot_root(snapshot_path)
        store = WorkspaceSnapshotStore(
            workspace_dir=self.workspace_dir,
            object_store_dir=snapshot_root / "objects",
            manifest_dir=archive_path / "manifests",
            verify_objects=self.workspace_snapshot_verify_objects,
            extra_excluded_dir_names=(self._excluded_dir_names if self.strict_layout else ()),
        )
        result = store.capture(snapshot_id=archive_id)
        warnings: list[str] = []
        stage_log_file_count = copy_stage_logs_to_snapshot(
            workspace_dir=self.workspace_dir,
            snapshot_dir=archive_path,
            warnings=warnings,
        )
        meta = {
            "archive_id": archive_id,
            "archive_path": str(archive_path),
            "created_at": time.time(),
            "source_stage_id": str(stage_id or ""),
            "snapshot_mode": "estra_terminal_archive",
            "workspace_snapshot_manifest_path": str(result.manifest_path),
            "workspace_snapshot_object_store_root": str(result.object_store_root),
            "workspace_snapshot_stats": result.to_json(),
            "copied_stage_logs": stage_log_file_count > 0,
            "stage_log_file_count": stage_log_file_count,
            "archive_warnings": warnings,
        }
        (archive_path / "terminal_archive_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return TerminalArchive(
            archive_path=archive_path,
            manifest_path=result.manifest_path,
            object_store_root=result.object_store_root,
        )

    def _restore_terminal_archive(self, archive: TerminalArchive) -> None:
        if archive.manifest_path is None or archive.object_store_root is None:
            return
        store = WorkspaceSnapshotStore(
            workspace_dir=self.workspace_dir,
            object_store_dir=archive.object_store_root,
            manifest_dir=archive.manifest_path.parent,
            verify_objects=self.workspace_snapshot_verify_objects,
            extra_excluded_dir_names=(self._excluded_dir_names if self.strict_layout else ()),
        )
        store.restore(archive.manifest_path, target_dir=self.workspace_dir)

    def restore_terminal_archive(self, archive_path: str | Path) -> None:
        """Restore a terminal archive returned by :meth:`restore`."""
        path = Path(archive_path).expanduser().resolve(strict=False)
        meta_path = path / "terminal_archive_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        manifest_path = Path(str(meta.get("workspace_snapshot_manifest_path") or ""))
        object_store_root = Path(str(meta.get("workspace_snapshot_object_store_root") or ""))
        if not manifest_path.is_absolute():
            manifest_path = path / manifest_path
        if not object_store_root.is_absolute():
            object_store_root = path / object_store_root
        self._restore_terminal_archive(
            TerminalArchive(
                archive_path=path,
                manifest_path=manifest_path,
                object_store_root=object_store_root,
            )
        )

    def _copy_workspace_snapshot_compat_view(self, dst: Path) -> dict[str, int]:
        """Copy only small human/selection files into the compatibility directory."""
        copied = 0
        skipped_large = 0
        dst.mkdir(parents=True, exist_ok=True)

        def visit(directory: Path) -> None:
            nonlocal copied, skipped_large
            try:
                children = sorted(directory.iterdir(), key=lambda p: p.name)
            except OSError:
                return
            for child in children:
                if self._is_excluded_snapshot_path(child, for_baseline=False):
                    continue
                rel = child.relative_to(self.workspace_dir)
                if child.is_symlink():
                    continue
                if child.is_dir():
                    visit(child)
                    continue
                if not child.is_file():
                    continue
                name = child.name
                suffix = child.suffix.lower()
                if name not in _WORKSPACE_SNAPSHOT_COMPAT_FILE_NAMES and suffix not in _WORKSPACE_SNAPSHOT_COMPAT_SUFFIXES:
                    continue
                try:
                    size = child.stat().st_size
                except OSError:
                    continue
                if size > _WORKSPACE_SNAPSHOT_COMPAT_MAX_BYTES:
                    skipped_large += 1
                    continue
                _copy_entry(child, dst / rel)
                copied += 1

        visit(self.workspace_dir)
        return {"compat_view_file_count": copied, "compat_view_skipped_large_count": skipped_large}

    @staticmethod
    def _link_workspace_snapshot_merge_payload(
        *,
        dst: Path,
        entries: dict[str, Any],
        object_store_root: Path,
    ) -> int:
        """Expose merge inputs as links to immutable CAS objects."""
        linked = 0
        for rel, row in sorted(entries.items()):
            if not isinstance(row, dict) or row.get("kind") != "file":
                continue
            rel_path = Path(str(rel))
            if (
                rel_path.is_absolute()
                or not rel_path.parts
                or rel_path.parts[0] != "merge_payload"
                or ".." in rel_path.parts
            ):
                continue
            digest = str(row.get("digest") or "")
            if not digest:
                continue
            source = workspace_snapshot_object_path(object_store_root, digest)
            if not source.is_file():
                continue
            target = dst / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists():
                continue
            target.symlink_to(os.path.relpath(source, start=target.parent))
            linked += 1
        return linked

    def _capture_workspace_snapshot(
        self,
        *,
        snapshot_root: Path,
        snapshot_id: str,
        stage_id: str,
        metric_value: float | None,
        metric_name: str,
        lower_is_better: bool | None,
        memory_cut: int,
        source_event: dict[str, Any],
        node_uid: str,
        lineage_id: str,
        warnings: list[str] | None = None,
    ) -> StageSnapshot:
        warnings = list(warnings or [])
        final_dir = snapshot_root / snapshot_id
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        store = self._workspace_snapshot_store(snapshot_root)
        result = store.capture(snapshot_id=snapshot_id)
        compat_stats = self._copy_workspace_snapshot_compat_view(final_dir)
        manifest = store.load_manifest(result.manifest_path)
        entries = manifest.get("entries") if isinstance(manifest.get("entries"), dict) else {}
        merge_payload_link_count = self._link_workspace_snapshot_merge_payload(
            dst=final_dir,
            entries=entries,
            object_store_root=result.object_store_root,
        )
        snap = StageSnapshot(
            stage_id=stage_id,
            snapshot_id=snapshot_id,
            snapshot_path=final_dir,
            metric_value=metric_value,
            metric_name=metric_name,
            lower_is_better=lower_is_better,
            memory_cut=memory_cut,
            source_event=dict(source_event),
            node_uid=str(node_uid or ""),
            lineage_id=str(lineage_id or ""),
            snapshot_mode="workspace_snapshot",
            snapshot_warnings=tuple(warnings),
        )
        meta = {
            "stage_id": stage_id,
            "snapshot_id": snapshot_id,
            "snapshot_path": str(final_dir),
            "metric_value": metric_value,
            "metric_name": metric_name,
            "lower_is_better": lower_is_better,
            "memory_cut": memory_cut,
            "source_event": source_event,
            "node_uid": str(node_uid or ""),
            "lineage_id": str(lineage_id or ""),
            "captured_at": time.time(),
            "snapshot_mode": "workspace_snapshot",
            "snapshot_warnings": warnings,
            "workspace_snapshot_manifest_path": str(result.manifest_path),
            "workspace_snapshot_object_store_root": str(result.object_store_root),
            "workspace_snapshot_stats": result.to_json(),
            "captured_file_count": result.captured_file_count,
            "logical_size_bytes": result.logical_size_bytes,
            "new_physical_bytes": result.new_physical_bytes,
            "new_object_count": result.new_object_count,
            "reused_object_count": result.reused_object_count,
            "skipped_hash_count": result.skipped_hash_count,
            "copied_memory": False,
            "copied_weights": any(Path(rel).suffix.lower() in _WEIGHT_SUFFIXES for rel in entries),
            "compat_view_file_count": compat_stats["compat_view_file_count"],
            "compat_view_skipped_large_count": compat_stats["compat_view_skipped_large_count"],
            "merge_payload_link_count": merge_payload_link_count,
            "excluded_dirs": sorted(self._excluded_dir_names),
        }
        metadata_dir = final_dir / self.metadata_dirname
        metadata_dir.mkdir(parents=True, exist_ok=True)
        stage_log_file_count = copy_stage_logs_to_snapshot(
            workspace_dir=self.workspace_dir,
            snapshot_dir=final_dir,
            warnings=warnings,
        )
        meta["copied_stage_logs"] = stage_log_file_count > 0
        meta["stage_log_file_count"] = stage_log_file_count
        meta["snapshot_warnings"] = warnings
        if self.memory_dir is not None and self.memory_dir.is_dir():
            memory_target = metadata_dir / "memory"
            try:
                if memory_target.exists():
                    shutil.rmtree(memory_target)
                shutil.copytree(self.memory_dir, memory_target, symlinks=True)
                meta["copied_memory"] = True
            except OSError:
                warnings.append("memory_sidecar_copy_failed")
                meta["snapshot_warnings"] = warnings
        (metadata_dir / "lhr_snapshot_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return snap

    def capture(
        self,
        *,
        stage_id: str,
        metric_value: float | None,
        metric_name: str,
        lower_is_better: bool | None,
        memory_cut: int,
        source_event: dict[str, Any],
        node_uid: str = "",
        lineage_id: str = "",
    ) -> StageSnapshot:
        snapshot_root = self._ensure_snapshot_root()
        node_label = str(node_uid or stage_id).replace(":", "-")
        snapshot_id = f"{node_label}-{uuid.uuid4().hex[:10]}"
        if self.workspace_snapshot_enabled:
            return self._capture_workspace_snapshot(
                snapshot_root=snapshot_root,
                snapshot_id=snapshot_id,
                stage_id=stage_id,
                metric_value=metric_value,
                metric_name=metric_name,
                lower_is_better=lower_is_better,
                memory_cut=memory_cut,
                source_event=source_event,
                node_uid=node_uid,
                lineage_id=lineage_id,
            )
        selected_entries, deleted_files, baseline_entries = self._selected_snapshot_entries()
        warnings: list[str] = []
        snapshot_mode = "changed_files"
        final_dir, copy_complete, copy_error = self._materialize_snapshot_dir(
            snapshot_root=snapshot_root,
            snapshot_id=snapshot_id,
            selected_entries=selected_entries,
        )
        if not copy_complete:
            warnings.append(f"primary_snapshot_copy_failed:{copy_error}")
            try:
                fallback = self._fallback_snapshot_root(OSError(copy_error), skip=(snapshot_root,))
                if fallback.resolve(strict=False) != snapshot_root.resolve(strict=False):
                    shutil.rmtree(final_dir, ignore_errors=True)
                    final_dir, copy_complete, copy_error = self._materialize_snapshot_dir(
                        snapshot_root=fallback,
                        snapshot_id=snapshot_id,
                        selected_entries=selected_entries,
                    )
                    if copy_complete:
                        snapshot_mode = "fallback_changed_files"
                    else:
                        warnings.append(f"fallback_snapshot_copy_failed:{copy_error}")
            except OSError as fallback_exc:
                warnings.append(f"fallback_snapshot_root_failed:{type(fallback_exc).__name__}")
        if not copy_complete:
            try:
                final_dir = self._materialize_minimal_snapshot(
                    snapshot_root=self.snapshot_root,
                    snapshot_id=snapshot_id,
                    warnings=warnings,
                )
                snapshot_mode = "minimal_fallback"
                copy_complete = True
                copy_error = ""
            except OSError as minimal_exc:
                warnings.append(f"minimal_snapshot_copy_failed:{type(minimal_exc).__name__}")
                final_dir.mkdir(parents=True, exist_ok=True)
                snapshot_mode = "metadata_only"
        snap = StageSnapshot(
            stage_id=stage_id,
            snapshot_id=snapshot_id,
            snapshot_path=final_dir,
            metric_value=metric_value,
            metric_name=metric_name,
            lower_is_better=lower_is_better,
            memory_cut=memory_cut,
            source_event=dict(source_event),
            node_uid=str(node_uid or ""),
            lineage_id=str(lineage_id or ""),
            snapshot_mode=snapshot_mode,
            snapshot_warnings=tuple(warnings),
        )
        meta = {
            "stage_id": stage_id,
            "snapshot_id": snapshot_id,
            "snapshot_path": str(final_dir),
            "metric_value": metric_value,
            "metric_name": metric_name,
            "lower_is_better": lower_is_better,
            "memory_cut": memory_cut,
            "source_event": source_event,
            "node_uid": str(node_uid or ""),
            "lineage_id": str(lineage_id or ""),
            "captured_at": time.time(),
            "snapshot_mode": snapshot_mode,
            "snapshot_warnings": warnings,
            "snapshot_copy_complete": copy_complete,
            "snapshot_copy_error": copy_error,
            "baseline_available": baseline_entries is not None,
            "captured_file_count": len(selected_entries),
            "deleted_file_count": len(deleted_files),
            "preserved_dirs": (
                [f"{self.metadata_dirname}/memory"]
                if self.memory_dir is not None
                else [".logs", ".memory"]
            ),
            "copied_memory": (
                bool(self.memory_dir and self.memory_dir.is_dir())
                if self.memory_dir is not None
                else any(rel == ".memory" or rel.startswith(".memory/") for rel in selected_entries)
            ),
            "copied_weights": any(Path(rel).suffix.lower() in _WEIGHT_SUFFIXES for rel in selected_entries),
            "baseline_entries": baseline_entries or {},
            "snapshot_entries": selected_entries if snapshot_mode in {"changed_files", "fallback_changed_files"} else {},
            "deleted_files": deleted_files,
            "excluded_dirs": sorted(self._excluded_dir_names),
        }
        metadata_dir = final_dir / self.metadata_dirname
        metadata_dir.mkdir(parents=True, exist_ok=True)
        stage_log_file_count = copy_stage_logs_to_snapshot(
            workspace_dir=self.workspace_dir,
            snapshot_dir=final_dir,
            warnings=warnings,
        )
        meta["copied_stage_logs"] = stage_log_file_count > 0
        meta["stage_log_file_count"] = stage_log_file_count
        meta["snapshot_warnings"] = warnings
        if self.memory_dir is not None and self.memory_dir.is_dir():
            memory_target = metadata_dir / "memory"
            try:
                if memory_target.exists():
                    shutil.rmtree(memory_target)
                shutil.copytree(self.memory_dir, memory_target, symlinks=True)
            except OSError:
                warnings.append("memory_sidecar_copy_failed")
                meta["snapshot_warnings"] = warnings
                meta["copied_memory"] = False
        (metadata_dir / "lhr_snapshot_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return snap

    def load(self, snapshot_path: str | Path) -> StageSnapshot | None:
        """Load one durable stage snapshot from its metadata sidecar."""

        path = Path(snapshot_path)
        metadata_file = path / self.metadata_dirname / "lhr_snapshot_meta.json"
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        stage_id = str(payload.get("stage_id") or "").strip().upper()
        snapshot_id = str(payload.get("snapshot_id") or path.name).strip()
        if not stage_id or not snapshot_id:
            return None
        metric_value = payload.get("metric_value")
        try:
            metric_value = float(metric_value) if metric_value is not None else None
        except (TypeError, ValueError):
            metric_value = None
        lower_is_better = payload.get("lower_is_better")
        if not isinstance(lower_is_better, bool):
            lower_is_better = None
        source_event = payload.get("source_event")
        if not isinstance(source_event, dict):
            source_event = {}
        warnings = payload.get("snapshot_warnings")
        if not isinstance(warnings, list):
            warnings = []
        try:
            memory_cut = int(payload.get("memory_cut") or 0)
        except (TypeError, ValueError):
            memory_cut = 0
        return StageSnapshot(
            stage_id=stage_id,
            snapshot_id=snapshot_id,
            snapshot_path=path,
            metric_value=metric_value,
            metric_name=str(payload.get("metric_name") or ""),
            lower_is_better=lower_is_better,
            memory_cut=memory_cut,
            source_event=dict(source_event),
            node_uid=str(payload.get("node_uid") or ""),
            lineage_id=str(payload.get("lineage_id") or ""),
            snapshot_mode=str(payload.get("snapshot_mode") or "changed_files"),
            snapshot_warnings=tuple(str(item) for item in warnings),
        )

    def _discover_with_timestamps(self) -> list[tuple[float, StageSnapshot]]:
        discovered: list[tuple[float, StageSnapshot]] = []
        for root in self._snapshot_root_candidates():
            if not root.is_dir():
                continue
            try:
                children = sorted(root.iterdir())
            except OSError:
                continue
            for path in children:
                if not path.is_dir():
                    continue
                snapshot = self.load(path)
                if snapshot is None:
                    continue
                metadata_file = path / self.metadata_dirname / "lhr_snapshot_meta.json"
                try:
                    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
                    captured_at = float(payload.get("captured_at") or 0.0)
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    captured_at = 0.0
                discovered.append((captured_at, snapshot))
        return discovered

    def discover_all(self) -> dict[str, StageSnapshot]:
        """Discover the full archive, keyed by stable node identity.

        Legacy snapshots without a node UID remain individually addressable by
        snapshot ID instead of being collapsed by their visible Stage ID.
        """

        discovered: dict[str, tuple[float, StageSnapshot]] = {}
        for captured_at, snapshot in self._discover_with_timestamps():
            key = str(snapshot.node_uid or "").strip() or f"legacy:{snapshot.snapshot_id}"
            previous = discovered.get(key)
            if previous is None or captured_at >= previous[0]:
                discovered[key] = (captured_at, snapshot)
        return {node_uid: item[1] for node_uid, item in discovered.items()}

    def discover(self) -> dict[str, StageSnapshot]:
        """Discover the latest durable snapshot for each visible stage id."""

        discovered: dict[str, tuple[float, StageSnapshot]] = {}
        for captured_at, snapshot in self._discover_with_timestamps():
            previous = discovered.get(snapshot.stage_id)
            if previous is None or captured_at >= previous[0]:
                discovered[snapshot.stage_id] = (captured_at, snapshot)
        return {stage_id: item[1] for stage_id, item in discovered.items()}

    def restore(self, snapshot: StageSnapshot | dict[str, Any]) -> Path:
        if isinstance(snapshot, StageSnapshot):
            src = snapshot.snapshot_path
            stage_id = snapshot.stage_id
        else:
            src = Path(str(snapshot.get("snapshot_path") or ""))
            stage_id = str(snapshot.get("stage_id") or "stage")
        if not src.is_dir():
            raise FileNotFoundError(f"snapshot not found: {src}")
        meta: dict[str, Any] = {}
        metadata_dir = src / self.metadata_dirname
        metadata_file = metadata_dir / "lhr_snapshot_meta.json"
        try:
            loaded = json.loads(metadata_file.read_text(encoding="utf-8"))
            meta = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            meta = {}
        workspace_has_git = self._workspace_has_git()
        terminal_archive = self._capture_terminal_archive(stage_id=stage_id, snapshot_path=src)
        archive_path = terminal_archive.archive_path
        try:
            snapshot_mode = str(meta.get("snapshot_mode") or "")
            if snapshot_mode == "workspace_snapshot":
                manifest_path = Path(str(meta.get("workspace_snapshot_manifest_path") or ""))
                if not manifest_path.is_absolute():
                    manifest_path = src / manifest_path
                raw_object_store_root = str(meta.get("workspace_snapshot_object_store_root") or "")
                object_store_root = Path(raw_object_store_root) if raw_object_store_root else self.snapshot_root / "objects"
                store = WorkspaceSnapshotStore(
                    workspace_dir=self.workspace_dir,
                    object_store_dir=object_store_root,
                    manifest_dir=manifest_path.parent,
                    verify_objects=self.workspace_snapshot_verify_objects,
                    extra_excluded_dir_names=(self._excluded_dir_names if self.strict_layout else ()),
                )
                store.restore(manifest_path, target_dir=self.workspace_dir)
            else:
                if self.workspace_dir.exists():
                    self._clear_workspace_for_restore_preserving_git(
                        preserve_paths=(src, self.snapshot_root, self.archive_root),
                    )
                if snapshot_mode in {"changed_files", "fallback_changed_files"}:
                    self.workspace_dir.mkdir(parents=True, exist_ok=True)
                    baseline_entries = meta.get("baseline_entries") if isinstance(meta.get("baseline_entries"), dict) else {}
                    deleted_files = set(meta.get("deleted_files") or [])
                    restore_rels = [rel for rel in sorted(baseline_entries) if rel not in deleted_files]
                    if terminal_archive.manifest_path is not None and terminal_archive.object_store_root is not None:
                        archive_store = WorkspaceSnapshotStore(
                            workspace_dir=self.workspace_dir,
                            object_store_dir=terminal_archive.object_store_root,
                            manifest_dir=terminal_archive.manifest_path.parent,
                            verify_objects=self.workspace_snapshot_verify_objects,
                            extra_excluded_dir_names=(self._excluded_dir_names if self.strict_layout else ()),
                        )
                        restore_manifest_entries(
                            archive_store,
                            terminal_archive.manifest_path,
                            target_dir=self.workspace_dir,
                            include_rels=restore_rels,
                        )
                    exclude = ((metadata_dir,) if self.strict_layout else (metadata_file,)) + restore_preserved_paths(src)
                    _copy_tree_overlay(src, self.workspace_dir, exclude_paths=exclude)
                elif workspace_has_git:
                    self.workspace_dir.mkdir(parents=True, exist_ok=True)
                    exclude = ((metadata_dir,) if self.strict_layout else (metadata_file,)) + restore_preserved_paths(src)
                    _copy_tree_overlay(src, self.workspace_dir, exclude_paths=exclude)
                else:
                    _copy_tree_overlay(src, self.workspace_dir, exclude_paths=restore_preserved_paths(src))
            if self.strict_layout:
                self._remove_strict_workspace_control_dirs()
            else:
                self.workspace_dir.mkdir(parents=True, exist_ok=True)
                (self.workspace_dir / ".logs").mkdir(parents=True, exist_ok=True)
        except Exception as restore_exc:
            try:
                self._restore_terminal_archive(terminal_archive)
            except Exception as rollback_exc:
                raise RuntimeError(
                    "snapshot_restore_failed_and_terminal_rollback_failed:"
                    f"{type(rollback_exc).__name__}:{rollback_exc}"
                ) from restore_exc
            raise
        return archive_path
