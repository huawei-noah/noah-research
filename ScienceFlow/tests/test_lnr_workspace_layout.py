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
from pathlib import Path
from types import SimpleNamespace

import pytest

from scienceflow.core.agent.tools.tool_output_artifacts import ToolOutputArtifactStore
from scienceflow.config.settings import Config, prep_cfg
from scienceflow.core.tools.bash_tool import BashTool
from scienceflow.core.tools.ls_tool import LsTool
from scienceflow.core.tools.read_tool import ReadTool
from scienceflow.solver.lnr.snapshot_store import SnapshotStore
from scienceflow.solver.lnr.stage_logs import (
    attach_stage_interaction_handlers,
    ensure_stage_log_dir,
    stage_log_dir,
)
from scienceflow.solver.lnr.worker_layout import ensure_lnr_worker_layout, lnr_worker_layout
from scienceflow.utils.workspace_git import (
    auto_checkpoint_workspace_source,
    ensure_workspace_source_git,
)
from scienceflow.utils.workspace_interaction_log import (
    attach_workspace_interaction_logger,
    close_workspace_interaction_logger,
)


def test_lnr_worker_layout_uses_workspace_logs_snapshots_siblings(tmp_path: Path) -> None:
    layout = lnr_worker_layout(tmp_path, SimpleNamespace(worker_dirname="workers"), 0)

    assert layout.root == tmp_path / "workers" / "w00"
    assert layout.workspace == layout.root / "workspace"
    assert layout.logs == layout.root / "logs"
    assert layout.snapshots == layout.root / "snapshots"

    ensured = ensure_lnr_worker_layout(tmp_path, SimpleNamespace(worker_dirname="workers"), 1)
    assert ensured.root == tmp_path / "workers" / "w01"
    assert ensured.workspace.is_dir()
    assert ensured.logs.is_dir()
    assert ensured.snapshots.is_dir()

    cfg = Config()
    cfg.task_workspace_root_dir = layout.root
    setattr(cfg, "_workspace_dir_override", layout.workspace)
    setattr(cfg, "_log_dir_override", layout.logs)
    prepped = prep_cfg(cfg)

    assert prepped.task_workspace_root_dir == layout.root
    assert prepped.workspace_dir == layout.workspace.resolve()
    assert prepped.log_dir == layout.logs.resolve()
    assert prepped.submission_dir == (layout.workspace / "submissions").resolve()


def test_lnr_worker_layout_keeps_control_artifacts_outside_workspace(tmp_path: Path) -> None:
    worker_root = tmp_path / "workers" / "w00"
    workspace = worker_root / "workspace"
    logs = worker_root / "logs"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("print(1)\n", encoding="utf-8")

    lg = attach_workspace_interaction_logger(
        worker_root,
        layout="split",
        log_dir_override=logs,
    )
    assert lg is not None
    attach_stage_interaction_handlers(lg, workspace, color=False)
    lg.info("hello lnr")
    close_workspace_interaction_logger(worker_root, layout="split", log_dir_override=logs)
    assert (logs / "interaction" / "interaction.log").is_file()
    assert (logs / "traj_interaction" / "traj_interaction.log").is_file()
    assert "hello lnr" in (
        workspace / ".logs" / "interaction" / "interaction.log"
    ).read_text(encoding="utf-8")
    assert not (worker_root / ".logs").exists()
    assert not (workspace / "logs").exists()

    stage_logs = ensure_stage_log_dir(workspace)
    store = ToolOutputArtifactStore(
        workspace,
        output_parts=("interaction", "tool_outputs"),
        mirror_parts=("traj_interaction", "tool_outputs"),
        log_dir_override=logs,
    )
    store.set_stage_log_dir(stage_logs)
    ref = store.reserve("bash", "raw output")
    assert ref is not None
    assert store.write_raw(ref, "raw output") is True
    store.append_index(ref, reducer_name="none", compressed_chars=10)
    store.append_stage_index(
        ref,
        args={"command": "echo raw output"},
        reducer_name="none",
        compressed_chars=10,
        tool_error=False,
    )
    assert (logs / "interaction" / "tool_outputs" / ref.raw_id).is_file()
    assert (logs / "traj_interaction" / "tool_outputs" / ref.raw_id).is_file()
    tool_log = (workspace / ".logs" / "tool.log").read_text(encoding="utf-8")
    assert f"raw_id={ref.raw_id}" in tool_log
    assert "full_output=" in tool_log
    assert not (workspace / ".logs" / "interaction" / "tool_outputs" / ref.raw_id).exists()
    assert not (workspace / "logs" / "interaction" / "tool_outputs" / ref.raw_id).exists()

    init = ensure_workspace_source_git(workspace, track_globs=["*.py"], initial_commit=True)
    assert init.ready is True
    (workspace / "solution.py").write_text("print(2)\n", encoding="utf-8")
    (workspace / "submission.csv").write_text("id,target\n1,0.1\n", encoding="utf-8")
    checkpoint = auto_checkpoint_workspace_source(
        workspace,
        track_globs=["*.py"],
        tool_name="lhr_stage_s01",
        metric_value_override=0.5,
        stage_id="S01",
        submission_snapshot_dir=logs / "submission_snapshots",
        checkpoint_dir=logs / "checkpoints",
    )
    assert checkpoint.ready is True
    assert checkpoint.ledger_path == "../logs/checkpoints/ledger.jsonl"
    assert (logs / "checkpoints" / "ledger.jsonl").is_file()
    assert (logs / "submission_snapshots").is_dir()
    assert not (workspace / ".scienceflow_checkpoints").exists()
    assert not (workspace / "submission_snapshots").exists()

    memory = workspace / ".agent_memory" / "ScienceAgent"
    memory.mkdir(parents=True)
    (memory / "short_term.json").write_text("[]\n", encoding="utf-8")
    snapshot_store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snapshots",
        archive_dirname="snapshots/archives",
        control_log_dir=logs,
        metadata_dirname="logs",
        strict_layout=True,
        memory_dir=workspace / ".agent_memory",
    )
    assert snapshot_store.initialize_baseline() is True
    (workspace / "solution.py").write_text("print(3)\n", encoding="utf-8")
    (workspace / "logs").mkdir()
    (workspace / "logs" / "agent.log").write_text("runtime\n", encoding="utf-8")
    (workspace / ".memory").mkdir()
    (workspace / ".memory" / "agent.txt").write_text("hidden\n", encoding="utf-8")
    (stage_log_dir(workspace) / "stage.log").write_text("stage-local\n", encoding="utf-8")

    snap = snapshot_store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="auc",
        lower_is_better=False,
        memory_cut=1,
        source_event={"event_type": "stage_completed"},
        node_uid="W00-L01-S01",
    )
    assert snap.snapshot_path.parent == worker_root / "snapshots"
    assert (snap.snapshot_path / "solution.py").read_text(encoding="utf-8") == "print(3)\n"
    assert (snap.snapshot_path / ".logs" / "stage.log").read_text(encoding="utf-8") == "stage-local\n"
    assert not (snap.snapshot_path / ".agent_memory").exists()
    assert not (snap.snapshot_path / ".memory").exists()
    assert not (snap.snapshot_path / "logs" / "agent.log").exists()
    assert (snap.snapshot_path / "logs" / "lhr_snapshot_meta.json").is_file()
    assert (snap.snapshot_path / "logs" / "memory" / "ScienceAgent" / "short_term.json").is_file()
    meta = json.loads((snap.snapshot_path / "logs" / "lhr_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["copied_memory"] is True
    assert meta["copied_stage_logs"] is True

    snapshot_store.restore(snap)
    assert (workspace / ".git").is_dir()
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print(3)\n"
    assert (workspace / ".logs" / "stage.log").is_file()
    assert (workspace / ".agent_memory" / "ScienceAgent" / "short_term.json").is_file()
    assert (workspace / ".memory" / "agent.txt").read_text(encoding="utf-8") == "hidden\n"
    assert (workspace / "logs" / "agent.log").read_text(encoding="utf-8") == "runtime\n"
    assert (worker_root / "snapshots" / "archives").is_dir()
    assert not (logs / "estra_archives").exists()


@pytest.mark.asyncio
async def test_lnr_hidden_stage_logs_are_blocked_from_agent_tools(tmp_path: Path) -> None:
    hidden = ensure_stage_log_dir(tmp_path)
    (hidden / "stage.log").write_text("secret current stage\n", encoding="utf-8")
    prefixes = [".logs"]

    read = ReadTool(workspace_dir=tmp_path, path_guard_denied_prefixes=prefixes)
    read_result = await read.execute(path=".logs/stage.log")
    assert read_result.error
    assert "hidden from agent" in read_result.error

    ls = LsTool(workspace_dir=tmp_path, path_guard_denied_prefixes=prefixes)
    ls_result = await ls.execute(path=".", include_hidden=True)
    assert not ls_result.error
    assert ".logs" not in (ls_result.output or "")

    bash = BashTool(
        workspace_dir=tmp_path,
        path_guard_denied_prefixes=prefixes,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
    )
    bash_result = await bash.execute("cat .logs/stage.log")
    assert bash_result.error
    assert "hidden from the agent" in bash_result.error
