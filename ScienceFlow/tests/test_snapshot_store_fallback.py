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

import json
import shutil
from pathlib import Path

import pytest

from scienceflow.solver.lnr.snapshot_store import SnapshotStore
from scienceflow.solver.lnr.workspace_snapshot import WorkspaceSnapshotStore


def test_snapshot_store_falls_back_when_primary_copy_fails(tmp_path: Path, monkeypatch) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    primary = worker_root / "snaps"
    primary.mkdir()
    (workspace / "solution.py").write_text("print(1)\n", encoding="utf-8")

    original_copy = SnapshotStore._copy_snapshot_payload

    def fail_primary_copy(self, dst, entries):
        if primary == dst or primary in dst.parents:
            raise PermissionError("primary snapshot root blocked")
        return original_copy(self, dst, entries)

    monkeypatch.setattr(SnapshotStore, "_copy_snapshot_payload", fail_primary_copy)
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )

    snap = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
        node_uid="W00-L01-S01",
    )

    assert snap.snapshot_path.parent == workspace / "snaps"
    assert snap.snapshot_mode == "fallback_changed_files"
    assert (snap.snapshot_path / "solution.py").read_text(encoding="utf-8") == "print(1)\n"
    assert not (snap.snapshot_path / "snaps").exists()
    assert (snap.snapshot_path / ".logs" / "lhr_snapshot_meta.json").is_file()


def test_snapshot_store_copies_only_changed_files_memory_and_weights(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    data_src = tmp_path / "data_src"
    workspace.mkdir(parents=True)
    data_src.mkdir()
    (data_src / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (workspace / "dataset").mkdir()
    (workspace / "dataset" / "train.csv").symlink_to(data_src / "train.csv")
    (workspace / "description.md").write_text("task\n", encoding="utf-8")
    (workspace / "solution.py").write_text("print('base')\n", encoding="utf-8")

    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname=".snapshots",
        archive_dirname=".estra_archives",
        workspace_snapshot_enabled=False,
    )
    assert store.initialize_baseline() is True

    (workspace / "solution.py").write_text("print('changed')\n", encoding="utf-8")
    (workspace / ".memory" / "ScienceAgent").mkdir(parents=True)
    (workspace / ".memory" / "ScienceAgent" / "short_term.json").write_text("[]\n", encoding="utf-8")
    (workspace / "artifacts").mkdir()
    (workspace / "artifacts" / "model.pth").write_bytes(b"weights")
    (workspace / "dataset" / "agent_output.csv").write_text("do not snapshot\n", encoding="utf-8")
    (workspace / "preview.jpg").write_bytes(b"raw image")

    snap = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
    )

    assert (snap.snapshot_path / "solution.py").read_text(encoding="utf-8") == "print('changed')\n"
    assert (snap.snapshot_path / ".memory" / "ScienceAgent" / "short_term.json").is_file()
    assert (snap.snapshot_path / "artifacts" / "model.pth").read_bytes() == b"weights"
    assert not (snap.snapshot_path / "dataset").exists()
    assert not (snap.snapshot_path / "description.md").exists()
    assert not (snap.snapshot_path / "preview.jpg").exists()
    meta = json.loads((snap.snapshot_path / ".logs" / "lhr_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["snapshot_mode"] == "changed_files"
    assert meta["baseline_available"] is True
    assert meta["copied_memory"] is True
    assert meta["copied_weights"] is True
    loaded = store.load(snap.snapshot_path)
    assert loaded is not None
    assert loaded.stage_id == "S01"
    assert loaded.snapshot_id == snap.snapshot_id
    assert loaded.metric_value == 0.5
    assert store.discover()["S01"].snapshot_path == snap.snapshot_path


def test_snapshot_store_discovers_same_visible_stage_across_lineages_by_node_uid(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("print('lineage-one')\n", encoding="utf-8")
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snapshots",
        archive_dirname="snapshots/archives",
        workspace_snapshot_enabled=False,
    )

    first = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="score",
        lower_is_better=False,
        memory_cut=1,
        source_event={"lineage_id": "L01"},
        node_uid="W00:L01:S01",
        lineage_id="L01",
    )
    (workspace / "solution.py").write_text("print('lineage-two')\n", encoding="utf-8")
    second = store.capture(
        stage_id="S01",
        metric_value=0.6,
        metric_name="score",
        lower_is_better=False,
        memory_cut=2,
        source_event={"lineage_id": "L02"},
        node_uid="W00:L02:S01",
        lineage_id="L02",
    )

    archived = store.discover_all()

    assert set(archived) == {"W00:L01:S01", "W00:L02:S01"}
    assert archived["W00:L01:S01"].snapshot_path == first.snapshot_path
    assert archived["W00:L02:S01"].snapshot_path == second.snapshot_path


def test_snapshot_store_uses_explicit_fallback_root_before_workspace(tmp_path: Path, monkeypatch) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    primary = worker_root / "snaps"
    primary.mkdir()
    task_fallback = tmp_path / "task_logs" / "snapshots" / "W00"
    (workspace / "solution.py").write_text("print(2)\n", encoding="utf-8")

    original_copy = SnapshotStore._copy_snapshot_payload

    def fail_primary_copy(self, dst, entries):
        if primary == dst or primary in dst.parents:
            raise PermissionError("primary snapshot root blocked")
        return original_copy(self, dst, entries)

    monkeypatch.setattr(SnapshotStore, "_copy_snapshot_payload", fail_primary_copy)
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        fallback_root_dir=task_fallback,
        workspace_snapshot_enabled=False,
    )

    snap = store.capture(
        stage_id="S01",
        metric_value=0.6,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
        node_uid="W00-L01-S01",
    )

    assert snap.snapshot_path.parent == task_fallback
    assert snap.snapshot_mode == "fallback_changed_files"
    assert (snap.snapshot_path / "solution.py").read_text(encoding="utf-8") == "print(2)\n"
    assert not (workspace / "snaps").exists()
    assert (snap.snapshot_path / ".logs" / "lhr_snapshot_meta.json").is_file()


def test_snapshot_store_uses_log_fallback_when_dot_snapshot_roots_blocked(tmp_path: Path, monkeypatch) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("print(1)\n", encoding="utf-8")
    blocked = {worker_root / ".snapshots", workspace / ".snapshots"}
    original_mkdir = Path.mkdir

    def guarded_mkdir(self: Path, *args, **kwargs):
        if self in blocked:
            raise PermissionError("dot snapshot root blocked")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname=".snapshots",
        archive_dirname=".estra_archives",
        workspace_snapshot_enabled=False,
    )

    snap = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
    )

    assert snap.snapshot_path.parent == worker_root / ".logs" / "stage_snapshots"
    assert (snap.snapshot_path / "solution.py").is_file()


def test_snapshot_store_minimal_fallback_when_payload_copy_fails(tmp_path: Path, monkeypatch) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("print(4)\n", encoding="utf-8")
    (workspace / "submission.csv").write_text("id,target\n1,0.5\n", encoding="utf-8")
    memory_file = workspace / ".memory" / "ScienceAgent" / "long_term.jsonl"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text('{"note":"keep"}\n', encoding="utf-8")
    git_object = workspace / ".git" / "objects" / "aa" / "bad"
    git_object.parent.mkdir(parents=True)
    git_object.write_text("should not be copied", encoding="utf-8")

    def fail_copy(self, dst, entries):
        raise PermissionError("copy blocked")

    monkeypatch.setattr(SnapshotStore, "_copy_snapshot_payload", fail_copy)
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )

    snap = store.capture(
        stage_id="S01",
        metric_value=0.8,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
        node_uid="W00-L01-S01",
    )

    assert snap.snapshot_mode == "minimal_fallback"
    assert any(w.startswith("primary_snapshot_copy_failed") for w in snap.snapshot_warnings)
    assert any(w.startswith("minimal_snapshot_file_count") for w in snap.snapshot_warnings)
    assert (snap.snapshot_path / "solution.py").read_text(encoding="utf-8") == "print(4)\n"
    assert (snap.snapshot_path / "submission.csv").is_file()
    assert (snap.snapshot_path / ".memory" / "ScienceAgent" / "long_term.jsonl").is_file()
    assert not (snap.snapshot_path / ".git").exists()
    meta = json.loads((snap.snapshot_path / ".logs" / "lhr_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["snapshot_mode"] == "minimal_fallback"
    assert any(w.startswith("minimal_snapshot_file_count") for w in meta["snapshot_warnings"])


def test_snapshot_store_excludes_git_directory(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("print(3)\n", encoding="utf-8")
    git_object = workspace / ".git" / "objects" / "aa" / "bad"
    git_object.parent.mkdir(parents=True)
    git_object.write_text("repo internals are not a research artifact", encoding="utf-8")

    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )

    snap = store.capture(
        stage_id="S01",
        metric_value=0.7,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
        node_uid="W00-L01-S01",
    )

    assert (snap.snapshot_path / "solution.py").is_file()
    assert not (snap.snapshot_path / ".git").exists()
    meta = json.loads((snap.snapshot_path / ".logs" / "lhr_snapshot_meta.json").read_text(encoding="utf-8"))
    assert ".git" in meta["excluded_dirs"]
    assert meta["snapshot_mode"] == "changed_files"


def test_snapshot_restore_preserves_live_logs_with_workspace_git(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / ".git").mkdir()
    (workspace / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (workspace / ".logs").mkdir()
    (workspace / ".logs" / "interaction.log").write_text("live log\n", encoding="utf-8")
    (workspace / ".logs" / ".efc_leftover_interaction.log").write_text("", encoding="utf-8")
    (workspace / "solution.py").write_text("print('stage1')\n", encoding="utf-8")

    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )
    assert store.initialize_baseline() is True
    (workspace / "solution.py").write_text("print('stage2')\n", encoding="utf-8")
    snap = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
    )

    (workspace / "solution.py").write_text("print('terminal')\n", encoding="utf-8")
    (workspace / ".logs" / ".efc_restore_race_interaction.log").write_text("", encoding="utf-8")
    archive = store.restore(snap)

    assert archive.is_dir()
    assert (workspace / ".git" / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert (workspace / ".logs" / "interaction.log").read_text(encoding="utf-8") == "live log\n"
    assert (workspace / ".logs" / ".efc_restore_race_interaction.log").is_file()
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print('stage2')\n"


def test_snapshot_restore_preserves_live_logs_without_workspace_git(tmp_path: Path, monkeypatch) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    logs_dir = workspace / ".logs"
    logs_dir.mkdir()
    (logs_dir / "interaction.log").write_text("live log\n", encoding="utf-8")
    (workspace / "solution.py").write_text("print('stage1')\n", encoding="utf-8")

    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )
    assert store.initialize_baseline() is True
    (workspace / "solution.py").write_text("print('stage2')\n", encoding="utf-8")
    snap = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
    )

    (workspace / "solution.py").write_text("print('terminal')\n", encoding="utf-8")
    (logs_dir / "interaction.log").write_text("still-live\n", encoding="utf-8")

    original_rmtree = shutil.rmtree

    def fail_on_logs(path, *args, **kwargs):
        if Path(path).name == ".logs":
            raise AssertionError("restore must not delete live .logs")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("scienceflow.solver.lnr.snapshot_store.shutil.rmtree", fail_on_logs)
    archive = store.restore(snap)

    assert archive.is_dir()
    archive_meta = json.loads((archive / "terminal_archive_meta.json").read_text(encoding="utf-8"))
    archive_manifest = json.loads(Path(archive_meta["workspace_snapshot_manifest_path"]).read_text(encoding="utf-8"))
    assert archive_meta["snapshot_mode"] == "estra_terminal_archive"
    assert archive_manifest["entries"]["solution.py"]["kind"] == "file"
    assert not (archive / "solution.py").exists()
    assert (workspace / ".logs" / "interaction.log").read_text(encoding="utf-8") == "still-live\n"
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print('stage2')\n"


def test_snapshot_restore_preserves_current_workspace_git(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "description.md").write_text("task\n", encoding="utf-8")
    (workspace / "solution.py").write_text("print('stage1')\n", encoding="utf-8")
    git_head = workspace / ".git" / "HEAD"
    git_head.parent.mkdir(parents=True)
    git_head.write_text("ref: refs/heads/main\n", encoding="utf-8")

    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )
    assert store.initialize_baseline() is True
    (workspace / "solution.py").write_text("print('changed')\n", encoding="utf-8")
    snap = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
    )

    (workspace / "solution.py").write_text("print('terminal')\n", encoding="utf-8")
    archive = store.restore(snap)

    assert (workspace / ".git" / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert not (archive / ".git").exists()
    assert not (worker_root / ".logs" / "restore_preserved_git").exists()
    archive_meta = json.loads((archive / "terminal_archive_meta.json").read_text(encoding="utf-8"))
    archive_manifest = json.loads(Path(archive_meta["workspace_snapshot_manifest_path"]).read_text(encoding="utf-8"))
    assert archive_manifest["entries"]["solution.py"]["kind"] == "file"
    assert not (archive / "solution.py").exists()
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print('changed')\n"


def test_snapshot_store_partial_restore_preserves_baseline_and_removes_later_files(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    data_src = tmp_path / "data_src"
    workspace.mkdir(parents=True)
    data_src.mkdir()
    (data_src / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (workspace / "dataset").mkdir()
    (workspace / "dataset" / "train.csv").symlink_to(data_src / "train.csv")
    (workspace / "description.md").write_text("task\n", encoding="utf-8")

    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )
    assert store.initialize_baseline() is True
    (workspace / "solution.py").write_text("print('stage1')\n", encoding="utf-8")
    (workspace / ".memory" / "ScienceAgent").mkdir(parents=True)
    (workspace / ".memory" / "ScienceAgent" / "short_term.json").write_text("stage1\n", encoding="utf-8")
    snap = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
    )

    (workspace / "solution.py").write_text("print('stage2')\n", encoding="utf-8")
    (workspace / "extra.py").write_text("print('remove')\n", encoding="utf-8")
    archive = store.restore(snap)

    assert archive.is_dir()
    assert (workspace / "description.md").read_text(encoding="utf-8") == "task\n"
    assert (workspace / "dataset" / "train.csv").is_symlink()
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print('stage1')\n"
    assert (workspace / ".memory" / "ScienceAgent" / "short_term.json").read_text(encoding="utf-8") == "stage1\n"
    assert not (workspace / "extra.py").exists()


def test_snapshot_store_workspace_snapshot_backend_restores_without_copying_weight_view(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    memory_dir = worker_root / "logs" / "memory" / "ScienceAgent"
    memory_dir.mkdir(parents=True)
    (memory_dir / "short_term.json").write_text("[]\n", encoding="utf-8")
    (workspace / "solution.py").write_text("print('stage1')\n", encoding="utf-8")
    (workspace / "submission.csv").write_text("id,target\n1,0.5\n", encoding="utf-8")
    (workspace / "model_best.pth").write_bytes(b"weights-v1")
    (workspace / "merge_payload").mkdir()
    (workspace / "merge_payload" / "probabilities.npy").write_bytes(b"soft-probabilities")
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        metadata_dirname="logs",
        strict_layout=True,
        memory_dir=worker_root / "logs" / "memory",
        workspace_snapshot_enabled=True,
    )

    first = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
        node_uid="W00-L01-S01",
    )
    second = store.capture(
        stage_id="S02",
        metric_value=0.6,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=4,
        source_event={"event_type": "stage_completed"},
        node_uid="W00-L01-S02",
    )

    assert first.snapshot_mode == "workspace_snapshot"
    assert (first.snapshot_path / "solution.py").is_file()
    assert (first.snapshot_path / "submission.csv").is_file()
    assert not (first.snapshot_path / "model_best.pth").exists()
    payload_link = first.snapshot_path / "merge_payload" / "probabilities.npy"
    assert payload_link.is_symlink()
    assert payload_link.read_bytes() == b"soft-probabilities"
    assert (first.snapshot_path / "logs" / "memory" / "ScienceAgent" / "short_term.json").is_file()
    meta = json.loads((first.snapshot_path / "logs" / "lhr_snapshot_meta.json").read_text(encoding="utf-8"))
    manifest_path = Path(meta["workspace_snapshot_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["entries"]["model_best.pth"]["kind"] == "file"
    assert manifest["entries"]["merge_payload/probabilities.npy"]["kind"] == "file"
    assert meta["merge_payload_link_count"] == 1
    assert Path(meta["workspace_snapshot_object_store_root"]) in payload_link.resolve().parents
    assert meta["copied_weights"] is True
    meta2 = json.loads((second.snapshot_path / "logs" / "lhr_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta2["skipped_hash_count"] >= 3
    assert meta2["new_physical_bytes"] == 0

    (workspace / "solution.py").write_text("print('terminal')\n", encoding="utf-8")
    (workspace / "model_best.pth").write_bytes(b"weights-v2")
    (workspace / "extra.py").write_text("print('remove')\n", encoding="utf-8")
    archive = store.restore(first)

    assert archive.is_dir()
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print('stage1')\n"
    assert (workspace / "submission.csv").read_text(encoding="utf-8") == "id,target\n1,0.5\n"
    assert (workspace / "model_best.pth").read_bytes() == b"weights-v1"
    assert not (workspace / "extra.py").exists()


def test_snapshot_restore_terminal_archive_uses_object_store(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("print('stage1')\n", encoding="utf-8")
    (workspace / "artifacts").mkdir()
    (workspace / "artifacts" / "model.pkl").write_bytes(b"stage1-weights")
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snapshots",
        archive_dirname="snapshots/archives",
        metadata_dirname="logs",
        strict_layout=True,
        workspace_snapshot_enabled=True,
    )

    snap = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
        node_uid="W00-L01-S01",
    )
    (workspace / "solution.py").write_text("print('terminal')\n", encoding="utf-8")
    (workspace / "artifacts" / "model.pkl").write_bytes(b"terminal-weights")

    archive = store.restore(snap)

    assert archive.parent == worker_root / "snapshots" / "archives"
    assert not (worker_root / "logs" / "estra_archives").exists()
    assert not (archive / "solution.py").exists()
    assert not (archive / "artifacts" / "model.pkl").exists()
    archive_meta = json.loads((archive / "terminal_archive_meta.json").read_text(encoding="utf-8"))
    manifest_path = Path(archive_meta["workspace_snapshot_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["entries"]["solution.py"]["kind"] == "file"
    assert manifest["entries"]["artifacts/model.pkl"]["kind"] == "file"
    assert Path(archive_meta["workspace_snapshot_object_store_root"]) == worker_root / "snapshots" / "objects"
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print('stage1')\n"
    assert (workspace / "artifacts" / "model.pkl").read_bytes() == b"stage1-weights"


def test_snapshot_restore_rolls_back_terminal_workspace_after_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    solution = workspace / "solution.py"
    solution.write_text("print('stage')\n", encoding="utf-8")
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snapshots",
        archive_dirname="snapshots/archives",
        metadata_dirname="logs",
        strict_layout=True,
        workspace_snapshot_enabled=True,
    )
    snap = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
        node_uid="W00:L01:S01",
        lineage_id="L01",
    )
    solution.write_text("print('terminal')\n", encoding="utf-8")
    (workspace / "terminal_only.py").write_text("terminal\n", encoding="utf-8")

    original_restore = WorkspaceSnapshotStore.restore
    calls = 0

    def fail_target_once(self, manifest_path, *, target_dir=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            target = Path(target_dir) if target_dir is not None else self.workspace_dir
            self._clear_target_preserving_git(target)
            (target / "partial.py").write_text("partial\n", encoding="utf-8")
            raise OSError("injected target restore failure")
        return original_restore(self, manifest_path, target_dir=target_dir)

    monkeypatch.setattr(WorkspaceSnapshotStore, "restore", fail_target_once)

    with pytest.raises(OSError, match="injected target restore failure"):
        store.restore(snap)

    assert calls == 2
    assert solution.read_text(encoding="utf-8") == "print('terminal')\n"
    assert (workspace / "terminal_only.py").read_text(encoding="utf-8") == "terminal\n"
    assert not (workspace / "partial.py").exists()


def test_snapshot_store_workspace_snapshot_failure_does_not_fallback(tmp_path: Path, monkeypatch) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("print('stage1')\n", encoding="utf-8")
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=True,
    )

    def fail_workspace_snapshot(*args, **kwargs):
        raise OSError("object store unavailable")

    monkeypatch.setattr(store, "_capture_workspace_snapshot", fail_workspace_snapshot)

    with pytest.raises(OSError, match="object store unavailable"):
        store.capture(
            stage_id="S01",
            metric_value=0.5,
            metric_name="auc",
            lower_is_better=False,
            memory_cut=3,
            source_event={"event_type": "stage_completed"},
            node_uid="W00-L01-S01",
        )

    assert not any((worker_root / "snaps").glob("W00-L01-S01-*"))
