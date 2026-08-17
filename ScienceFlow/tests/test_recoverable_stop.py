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

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from scienceflow.core.subprocess_utils import spawn_exec, terminate_tree_recoverable
from scienceflow.core.tools.bash.signals import _latest_artifact_snapshot


def test_run_scoped_artifact_must_be_stable_before_recoverable(tmp_path: Path) -> None:
    started_at = time.time()
    run_artifacts = tmp_path / ".scienceflow_runs" / "W00_bash_00001" / "artifacts"
    run_artifacts.mkdir(parents=True)
    artifact = run_artifacts / "best.pt"
    artifact.write_bytes(b"checkpoint")
    cache: dict[str, object] = {}

    first = _latest_artifact_snapshot(
        tmp_path,
        started_at=started_at,
        run_artifact_dir=run_artifacts,
        stability_cache=cache,
        settle_sec=0.0,
        stable_poll_count=1,
    )
    second = _latest_artifact_snapshot(
        tmp_path,
        started_at=started_at,
        run_artifact_dir=run_artifacts,
        stability_cache=cache,
        settle_sec=0.0,
        stable_poll_count=1,
    )

    assert first["stability"] == "candidate"
    assert first["recoverable_artifact_on_disk"] is False
    assert second["stability"] == "stable"
    assert second["artifact_scope"] == "current_run"
    assert second["recoverable_artifact_on_disk"] is True


def test_workspace_recent_artifact_needs_marker_or_run_state(tmp_path: Path) -> None:
    started_at = time.time()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    artifact = artifacts / "best.pt"
    artifact.write_bytes(b"checkpoint")
    cache: dict[str, object] = {}

    _latest_artifact_snapshot(
        tmp_path,
        started_at=started_at,
        stability_cache=cache,
        settle_sec=0.0,
        stable_poll_count=1,
    )
    snapshot = _latest_artifact_snapshot(
        tmp_path,
        started_at=started_at,
        stability_cache=cache,
        settle_sec=0.0,
        stable_poll_count=1,
    )

    assert snapshot["artifact_scope"] == "workspace_recent"
    assert snapshot["stability"] == "stable"
    assert snapshot["recoverable_artifact_on_disk"] is False

    (artifact.with_name(artifact.name + ".done")).write_text("done\n", encoding="utf-8")
    done_snapshot = _latest_artifact_snapshot(tmp_path, started_at=started_at)

    assert done_snapshot["stability"] == "done_marker"
    assert done_snapshot["recoverable_artifact_on_disk"] is True


def test_run_state_can_confirm_current_workspace_artifact(tmp_path: Path) -> None:
    started_at = time.time()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    artifact = artifacts / "best.pt"
    artifact.write_bytes(b"checkpoint")
    run_dir = tmp_path / ".scienceflow_runs" / "W00_bash_00001"
    run_dir.mkdir(parents=True)
    run_state = run_dir / "run_state.json"
    run_state.write_text(
        json.dumps({
            "status": "running",
            "checkpoint_path": "artifacts/best.pt",
            "safe_to_resume": True,
        }),
        encoding="utf-8",
    )

    snapshot = _latest_artifact_snapshot(
        tmp_path,
        started_at=started_at,
        run_state_path=run_state,
    )

    assert snapshot["artifact_scope"] == "workspace_recent"
    assert snapshot["stability"] == "run_state_confirmed"
    assert snapshot["run_state_confirmed"] is True
    assert snapshot["recoverable_artifact_on_disk"] is True


def test_stale_workspace_artifact_is_ignored(tmp_path: Path) -> None:
    started_at = time.time()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    artifact = artifacts / "best.pt"
    artifact.write_bytes(b"old")
    old_time = started_at - 30.0
    os.utime(artifact, (old_time, old_time))

    snapshot = _latest_artifact_snapshot(tmp_path, started_at=started_at)

    assert snapshot == {}


@pytest.mark.asyncio
async def test_terminate_tree_recoverable_uses_sigusr1_marker(tmp_path: Path) -> None:
    run_state = tmp_path / "run_state.json"
    code = r'''
import json
import os
import signal
import time
from pathlib import Path

state = Path(os.environ["RUN_STATE"])

def handler(signum, frame):
    tmp = state.with_suffix(".tmp")
    tmp.write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")
    os.replace(tmp, state)
    raise SystemExit(0)

signal.signal(signal.SIGUSR1, handler)
while True:
    time.sleep(0.1)
'''
    env = {**os.environ, "RUN_STATE": str(run_state)}
    proc = await spawn_exec(sys.executable, "-c", code, env=env)
    await asyncio.sleep(0.2)

    await terminate_tree_recoverable(
        proc,
        marker_check=lambda: run_state.exists(),
        sigusr1_grace=2.0,
        marker_exit_grace=1.0,
        sigterm_grace=0.5,
    )

    assert json.loads(run_state.read_text(encoding="utf-8"))["status"] == "interrupted"
    assert proc.returncode == 0
