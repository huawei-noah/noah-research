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

import errno
import json
import os
import stat
from pathlib import Path

import pytest

from scienceflow.solver.lnr import restore_cleanup
from scienceflow.solver.lnr.restore_cleanup import clear_directory_contents
from scienceflow.solver.lnr.workspace_snapshot import (
    WorkspaceSnapshotStore,
)
from scienceflow.solver.lnr.workspace_snapshot_io import workspace_snapshot_object_path


def _store(tmp_path: Path, workspace: Path, data_root: Path | None = None) -> WorkspaceSnapshotStore:
    return WorkspaceSnapshotStore(
        workspace_dir=workspace,
        object_store_dir=tmp_path / "objects",
        manifest_dir=tmp_path / "manifests",
        data_roots=[data_root] if data_root is not None else [],
    )


def _manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_workspace_snapshot_reuses_unchanged_large_weight(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    weight = workspace / "model_best.pth"
    weight.write_bytes(b"a" * (2 * 1024 * 1024))
    store = _store(tmp_path, workspace)

    first = store.capture(snapshot_id="s01")
    second = store.capture(snapshot_id="s02")

    assert first.new_object_count == 1
    assert second.new_object_count == 0
    assert second.reused_object_count == 1
    objects = [p for p in (tmp_path / "objects" / "sha256").rglob("*") if p.is_file()]
    assert len(objects) == 1




def test_workspace_snapshot_unchanged_50g_sparse_weight_skips_hash(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    weight = workspace / "huge_model.pth"
    logical_size = 50 * 1024**3
    with weight.open("wb") as f:
        f.truncate(logical_size)
    store = _store(tmp_path, workspace)
    digest = "sha256:" + "0" * 64
    object_path = tmp_path / "objects" / "sha256" / "00" / "00" / ("0" * 64)
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(b"already-stored-placeholder")
    fingerprint = WorkspaceSnapshotStore._file_fingerprint(weight.stat())
    previous_manifest = {
        "version": 1,
        "snapshot_id": "s00",
        "entries": {
            "huge_model.pth": {
                "kind": "file",
                "digest": digest,
                "size": logical_size,
                "mode": 0o644,
                "mtime_ns": weight.stat().st_mtime_ns,
                "fingerprint": fingerprint,
            }
        },
    }
    previous_path = tmp_path / "manifests" / "s00.json"
    previous_path.parent.mkdir(parents=True)
    previous_path.write_text(json.dumps(previous_manifest), encoding="utf-8")
    (tmp_path / "manifests" / "latest_manifest.json").write_text(
        json.dumps({"snapshot_id": "s00", "manifest_path": str(previous_path)}),
        encoding="utf-8",
    )

    def fail_store_file(_path: Path):
        raise AssertionError("unchanged 50G artifact should not be rehashed")

    monkeypatch.setattr(store, "_store_file", fail_store_file)
    result = store.capture(snapshot_id="s01")

    assert result.logical_size_bytes == logical_size
    assert result.new_object_count == 0
    assert result.new_physical_bytes == 0
    assert result.skipped_hash_count == 1
    manifest = _manifest(result.manifest_path)
    assert manifest["entries"]["huge_model.pth"]["digest"] == digest


def test_workspace_snapshot_changed_artifact_same_size_and_mtime_gets_new_object(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    weight = workspace / "model_best.pth"
    weight.write_bytes(b"a" * 4096)
    fixed_ns = 1_700_000_000_000_000_000
    os.utime(weight, ns=(fixed_ns, fixed_ns))
    store = _store(tmp_path, workspace)

    first = store.capture(snapshot_id="s01")
    weight.write_bytes(b"b" * 4096)
    os.utime(weight, ns=(fixed_ns, fixed_ns))
    second = store.capture(snapshot_id="s02")

    first_manifest = _manifest(first.manifest_path)
    second_manifest = _manifest(second.manifest_path)
    assert first_manifest["entries"]["model_best.pth"]["digest"] != second_manifest["entries"]["model_best.pth"]["digest"]
    objects = [p for p in (tmp_path / "objects" / "sha256").rglob("*") if p.is_file()]
    assert len(objects) == 2


def test_workspace_snapshot_records_dataset_symlink_without_copying_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "image.bin").write_bytes(b"dataset-bytes")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "dataset").symlink_to(data_root)
    (workspace / "train.py").write_text("print('ok')\n", encoding="utf-8")
    store = _store(tmp_path, workspace, data_root=data_root)

    result = store.capture(snapshot_id="s01")
    manifest = _manifest(result.manifest_path)

    assert manifest["entries"]["dataset"]["kind"] == "symlink"
    assert "dataset/image.bin" not in manifest["entries"]
    assert result.captured_file_count == 1


def test_workspace_snapshot_excludes_git_and_preserves_runtime_dirs_and_mode(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("secret", encoding="utf-8")
    memory_dir = workspace / ".memory"
    memory_dir.mkdir()
    (memory_dir / "notes.md").write_text("route evidence", encoding="utf-8")
    script = workspace / "run.sh"
    script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    script.chmod(0o755)
    store = _store(tmp_path, workspace)

    result = store.capture(snapshot_id="s01")
    manifest = _manifest(result.manifest_path)
    assert not any(path == ".git" or path.startswith(".git/") for path in manifest["entries"])
    assert not any(path == ".memory" or path.startswith(".memory/") for path in manifest["entries"])

    (memory_dir / "notes.md").write_text("changed", encoding="utf-8")
    script.unlink()
    store.restore(result.manifest_path)

    assert (workspace / ".git" / "config").read_text(encoding="utf-8") == "secret"
    assert (workspace / ".memory" / "notes.md").read_text(encoding="utf-8") == "changed"
    restored_mode = stat.S_IMODE((workspace / "run.sh").stat().st_mode)
    assert restored_mode & 0o111
    (workspace / "run.sh").write_text("#!/bin/sh\necho replaced\n", encoding="utf-8")


def test_workspace_snapshot_excludes_tmp_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    tmp_dir = workspace / "tmp"
    tmp_dir.mkdir()
    (tmp_dir / "cache.bin").write_bytes(b"cache")
    store = _store(tmp_path, workspace)

    result = store.capture(snapshot_id="s01")
    manifest = _manifest(result.manifest_path)

    assert "solution.py" in manifest["entries"]
    assert not any(path == "tmp" or path.startswith("tmp/") for path in manifest["entries"])


def test_workspace_snapshot_restore_preserves_tmp_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text("print('stage')\n", encoding="utf-8")
    deps = workspace / "tmp" / "deps" / "tokenizers"
    deps.mkdir(parents=True)
    (deps / "__init__.py").write_text("# cached dependency\n", encoding="utf-8")
    (workspace / "tmp" / "scratch.log").write_text("old scratch\n", encoding="utf-8")
    store = _store(tmp_path, workspace)

    result = store.capture(snapshot_id="s01")
    (workspace / "solution.py").write_text("print('later')\n", encoding="utf-8")
    (deps / "__init__.py").write_text("# still cached\n", encoding="utf-8")
    (workspace / "tmp" / "scratch.log").write_text("new scratch\n", encoding="utf-8")

    store.restore(result.manifest_path)

    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print('stage')\n"
    assert (deps / "__init__.py").read_text(encoding="utf-8") == "# still cached\n"
    assert (workspace / "tmp" / "scratch.log").read_text(encoding="utf-8") == "new scratch\n"


def test_workspace_snapshot_restore_preflights_objects_before_clearing_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    solution = workspace / "solution.py"
    solution.write_text("print('stage')\n", encoding="utf-8")
    store = _store(tmp_path, workspace)

    result = store.capture(snapshot_id="s01")
    manifest = _manifest(result.manifest_path)
    digest = manifest["entries"]["solution.py"]["digest"]
    object_path = workspace_snapshot_object_path(store.object_store_root, digest)
    object_path.unlink()
    solution.write_text("print('terminal')\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="workspace_snapshot_missing_object"):
        store.restore(result.manifest_path)

    assert solution.read_text(encoding="utf-8") == "print('terminal')\n"


def test_clear_directory_contents_moves_transient_rmtree_failure_to_trash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    scratch = root / "scratch"
    scratch.mkdir()
    (scratch / "artifact.tmp").write_text("transient\n", encoding="utf-8")
    trash = tmp_path / "restore_trash"
    original_rmtree = restore_cleanup.shutil.rmtree
    calls = 0

    def flaky_rmtree(path: Path, *args, **kwargs):
        nonlocal calls
        if Path(path) == scratch and not kwargs.get("ignore_errors"):
            calls += 1
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(restore_cleanup.shutil, "rmtree", flaky_rmtree)

    clear_directory_contents(root, trash_dir=trash)

    assert calls == 3
    assert not scratch.exists()
    assert trash.is_dir()


def test_workspace_snapshot_restore_preserves_in_workspace_object_store(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "model_best.pth").write_bytes(b"checkpoint")
    store = WorkspaceSnapshotStore(
        workspace_dir=workspace,
        object_store_dir=workspace / ".snapshots" / "objects",
        manifest_dir=workspace / ".snapshots" / "manifests",
    )

    result = store.capture(snapshot_id="s01")
    object_paths = [p for p in (workspace / ".snapshots" / "objects" / "sha256").rglob("*") if p.is_file()]
    assert object_paths

    (workspace / "model_best.pth").write_bytes(b"changed")
    store.restore(result.manifest_path)

    assert (workspace / "model_best.pth").read_bytes() == b"checkpoint"
    assert all(path.is_file() for path in object_paths)



def test_workspace_snapshot_skips_relative_symlink_escaping_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "cached_blob").symlink_to("../../../blobs/deadbeef")
    store = _store(tmp_path, workspace)

    result = store.capture(snapshot_id="s01")
    manifest = _manifest(result.manifest_path)

    assert result.skipped_symlink_count == 1
    assert manifest["stats"]["skipped_symlink_count"] == 1
    assert "cached_blob" not in manifest["entries"]
    assert manifest["entries"]["solution.py"]["kind"] == "file"

def test_workspace_snapshot_rejects_absolute_symlink_outside_data_roots(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "bad_link").symlink_to(outside)
    store = _store(tmp_path, workspace, data_root=data_root)

    with pytest.raises(ValueError, match="outside configured data roots"):
        store.capture(snapshot_id="s01")


def test_workspace_snapshot_restore_preserves_busy_logs_dir(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text("print('snapshot')\n", encoding="utf-8")
    logs = workspace / ".logs"
    logs.mkdir()
    (logs / "live.log").write_text("live\n", encoding="utf-8")
    run_dir = workspace / ".scienceflow_runs"
    run_dir.mkdir()
    (run_dir / "run_state.json").write_text("{}\n", encoding="utf-8")
    store = _store(tmp_path, workspace)

    result = store.capture(snapshot_id="s01")
    (workspace / "solution.py").write_text("print('current')\n", encoding="utf-8")
    (logs / "live.log").write_text("still-live\n", encoding="utf-8")

    original_rmtree = __import__("shutil").rmtree

    def fail_on_logs(path, *args, **kwargs):
        if Path(path).name == ".logs":
            raise AssertionError("restore must not delete live .logs")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(restore_cleanup.shutil, "rmtree", fail_on_logs)
    store.restore(result.manifest_path)

    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print('snapshot')\n"
    assert (logs / "live.log").read_text(encoding="utf-8") == "still-live\n"
    assert (run_dir / "run_state.json").read_text(encoding="utf-8") == "{}\n"
