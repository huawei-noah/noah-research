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

from scienceflow.solver.lnr.snapshot_store import SnapshotStore


def test_snapshot_store_restore_preserves_tmp_dir(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("print('stage')\n", encoding="utf-8")
    deps = workspace / "tmp" / "deps" / "tokenizers"
    deps.mkdir(parents=True)
    (deps / "__init__.py").write_text("# dependency cache\n", encoding="utf-8")
    (workspace / "tmp" / "scratch.log").write_text("old scratch\n", encoding="utf-8")
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snapshots",
        archive_dirname="archives",
        workspace_snapshot_enabled=True,
    )

    snap = store.capture(
        stage_id="S01",
        metric_value=0.8,
        metric_name="score",
        lower_is_better=False,
        memory_cut=3,
        source_event={"event_type": "stage_completed"},
    )
    (workspace / "solution.py").write_text("print('later')\n", encoding="utf-8")
    (deps / "__init__.py").write_text("# reused dependency cache\n", encoding="utf-8")
    (workspace / "tmp" / "scratch.log").write_text("new scratch\n", encoding="utf-8")

    store.restore(snap)

    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print('stage')\n"
    assert (deps / "__init__.py").read_text(encoding="utf-8") == "# reused dependency cache\n"
    assert (workspace / "tmp" / "scratch.log").read_text(encoding="utf-8") == "new scratch\n"
