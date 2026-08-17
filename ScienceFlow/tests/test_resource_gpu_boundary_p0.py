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

import pytest

from scienceflow.core.tools.bash_tool import BashTool
from scienceflow.solver.lnr.resource_runtime.gpu_feedback import build_gpu_boundary_feedback
from scienceflow.solver.lnr.resource_runtime.unified_store import (
    UnifiedResourceStore,
    format_resource_run_summary,
    summarize_resource_run,
)
from scienceflow.solver.lnr.resource_runtime.utilization import (
    normalize_cuda_visible_devices_for_task_pool,
    process_tree_gpu_placement_snapshot,
)
from scienceflow.solver.lnr.resource_runtime.workspace_gpu_guard import (
    cleanup_workspace_gpu_processes,
    scan_workspace_gpu_processes,
)
from tests.lnr_resource_test_utils import event_types, make_observer, payloads


def test_cuda_visible_mapping_handles_multi_digit_physical_ids() -> None:
    mapped, ids, remapped, reason = normalize_cuda_visible_devices_for_task_pool("0,1", ["10", "11"])

    assert mapped == "10,11"
    assert ids == ["10", "11"]
    assert remapped is True
    assert reason == "logical_ordinal_mapped_to_task_physical_gpu"


def test_cuda_visible_mapping_rejects_outside_ordinal_for_task_pool() -> None:
    mapped, ids, remapped, reason = normalize_cuda_visible_devices_for_task_pool("2", ["10", "11"])

    assert mapped == "2"
    assert ids == ["2"]
    assert remapped is False
    assert reason == "outside_task_pool"


def test_process_tree_gpu_placement_snapshot_reports_violation(monkeypatch) -> None:
    monkeypatch.setattr(
        "scienceflow.solver.lnr.resource_runtime.utilization.process_tree_pids",
        lambda root_pid: {123, 456},
    )
    monkeypatch.setattr(
        "scienceflow.solver.lnr.resource_runtime.utilization.sample_nvidia_compute_apps",
        lambda timeout_sec=1.5: {
            "available": True,
            "apps": [
                {"pid": 456, "gpu_id": "0", "gpu_uuid": "GPU-0", "used_memory_mb": 2048},
                {"pid": 789, "gpu_id": "4", "gpu_uuid": "GPU-4", "used_memory_mb": 1024},
            ],
        },
    )

    snapshot = process_tree_gpu_placement_snapshot(123, ["4", "5"])

    assert snapshot["available"] is True
    assert snapshot["violations"] == [{"pid": 456, "gpu_id": "0", "used_memory_mb": 2048}]


def test_workspace_gpu_scan_matches_workspace_and_skips_unsafe(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "task"
    other = tmp_path / "other"
    workspace.mkdir()
    other.mkdir()
    task_started_at = 1000.0
    uid = os.getuid() if hasattr(os, "getuid") else 1000

    def fake_proc(pid: int) -> dict:
        rows = {
            11: {
                "pid": 11,
                "exists": True,
                "cwd": str(workspace),
                "cwd_resolved": str(workspace.resolve()),
                "cmdline": "python train.py",
                "uid": uid,
                "start_time_epoch": 1001.0,
            },
            12: {
                "pid": 12,
                "exists": True,
                "cwd": str(other),
                "cwd_resolved": str(other.resolve()),
                "cmdline": "python other.py",
                "uid": uid,
                "start_time_epoch": 1001.0,
            },
            13: {
                "pid": 13,
                "exists": True,
                "cwd": str(workspace),
                "cwd_resolved": str(workspace.resolve()),
                "cmdline": "old train.py",
                "uid": uid,
                "start_time_epoch": 900.0,
            },
        }
        return rows[pid]

    monkeypatch.setattr(
        "scienceflow.solver.lnr.resource_runtime.workspace_gpu_guard.read_proc_info",
        fake_proc,
    )

    scan = scan_workspace_gpu_processes(
        workspace_dir=workspace,
        allowed_gpu_ids=["4", "5"],
        task_started_at=task_started_at,
        current_uid=uid,
        sample_compute_apps=lambda timeout_sec=1.5: {
            "available": True,
            "apps": [
                {"pid": 11, "gpu_id": "0", "gpu_uuid": "GPU-0", "used_memory_mb": 2048},
                {"pid": 12, "gpu_id": "4", "gpu_uuid": "GPU-4", "used_memory_mb": 1024},
                {"pid": 13, "gpu_id": "5", "gpu_uuid": "GPU-5", "used_memory_mb": 512},
            ],
        },
    )

    assert [row["pid"] for row in scan["matched_processes"]] == [11]
    assert [row["pid"] for row in scan["violations"]] == [11]
    assert {row["skip_reason"] for row in scan["skipped"]} == {
        "workspace_mismatch",
        "process_started_before_task",
    }


def test_workspace_gpu_cleanup_dry_run_reports_would_kill(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "task"
    workspace.mkdir()
    uid = os.getuid() if hasattr(os, "getuid") else 1000
    monkeypatch.setattr(
        "scienceflow.solver.lnr.resource_runtime.workspace_gpu_guard.read_proc_info",
        lambda pid: {
            "pid": pid,
            "exists": True,
            "cwd": str(workspace),
            "cwd_resolved": str(workspace.resolve()),
            "cmdline": "python train.py",
            "uid": uid,
            "start_time_epoch": 1001.0,
        },
    )

    cleanup = cleanup_workspace_gpu_processes(
        workspace_dir=workspace,
        allowed_gpu_ids=["4"],
        task_started_at=1000.0,
        reason="test_cleanup",
        dry_run=True,
        sample_compute_apps=lambda timeout_sec=1.5: {
            "available": True,
            "apps": [{"pid": 22, "gpu_id": "4", "gpu_uuid": "GPU-4", "used_memory_mb": 1024}],
        },
    )

    assert cleanup["would_kill_count"] == 1
    assert cleanup["killed"] == []


def test_gpu_boundary_feedback_is_separate_from_nf_heartbeat() -> None:
    feedback = build_gpu_boundary_feedback(actual_gpu_ids=["0"], allowed_gpu_ids=["4", "5"])

    assert feedback.startswith("RESOURCE_FEEDBACK: STOP_BOUNDARY_VIOLATION")
    assert "SCIENCEFLOW_HB" not in feedback
    assert "CUDA_VISIBLE_DEVICES" in feedback
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "3"; torch.device("cuda:0")' in feedback
    assert 'torch.device("cuda:0") is valid because it means logical device 0' in feedback
    assert 'when it assumes host GPU 0' not in feedback


@pytest.mark.asyncio
async def test_bash_tool_stops_boundary_violation_and_records_cleanup(monkeypatch, tmp_path) -> None:
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["4", "5"],
        min_register_sec=0.0,
        stalled_stdout_sec=100.0,
        kill_mode="auto",
    )
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "4", "SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL": "4,5"},
        resource_observer=observer,
    )

    monkeypatch.setattr(
        "scienceflow.core.tools.bash_tool._process_tree_gpu_placement_snapshot",
        lambda root_pid, allowed: {
            "available": True,
            "root_pid": root_pid,
            "allowed_gpu_ids": allowed,
            "used_gpu_processes": [{"pid": root_pid, "gpu_id": "0", "used_memory_mb": 2048}],
            "violations": [{"pid": root_pid, "gpu_id": "0", "used_memory_mb": 2048}],
        },
    )
    monkeypatch.setattr(
        "scienceflow.core.tools.bash_tool._cleanup_workspace_gpu_processes",
        lambda **kwargs: {
            "available": True,
            "matched_processes": [],
            "violations": [],
            "skipped": [],
            "killed": [],
            "killed_count": 0,
            "skipped_count": 0,
            "cleanup_reason": kwargs.get("reason"),
        },
    )

    result = await tool.execute("python3 -c 'import time; time.sleep(3)' --epochs 1")

    assert result.error == "Resource guard stopped boundary violation"
    assert "RESOURCE_FEEDBACK: STOP_BOUNDARY_VIOLATION" in (result.output or "")
    assert "SCIENCEFLOW_HB" not in (result.output or "")
    actions = payloads(sm, "resource_guard_action")
    assert any(action.get("raw_action") == "stop_boundary_violation" for action in actions)
    assert "resource_job_finished" in event_types(sm)


def test_resource_summary_counts_boundary_cleanup_and_skipped(tmp_path) -> None:
    resource_dir = tmp_path / "task-a" / "logs" / "resource"
    store = UnifiedResourceStore(resource_dir)
    store.append_event(
        "resource_guard_action",
        worker_id="W00",
        command_id="W00:bash:00001",
        payload={"action": "stop_boundary_violation", "reason": "task_gpu_boundary_violation"},
    )
    store.append_event(
        "resource_guard_action",
        worker_id="W00",
        command_id="W00:bash:00001",
        payload={"action": "gpu_orphan_cleanup", "reason": "command_finally", "placement": {"killed_count": 1}},
    )
    store.append_event(
        "resource_cleanup_heartbeat",
        worker_id="W00",
        command_id="W00:bash:00001",
        payload={"action": "gpu_cleanup_skipped", "reason": "cwd_unreadable", "placement": {"skipped_count": 1}},
    )

    summary = summarize_resource_run(tmp_path)
    task = summary["tasks"][0]
    rendered = format_resource_run_summary(summary)

    assert task["boundary_violations"] == 1
    assert task["gpu_orphan_cleanups"] == 1
    assert task["gpu_cleanup_skipped"] == 1
    assert "orphan=1" in rendered
    assert "cleanup_skipped=1" in rendered


def test_cleanup_heartbeat_does_not_install_feedback_guard(tmp_path) -> None:
    observer, sm = make_observer(tmp_path, min_register_sec=0.0)
    job_id = observer.job_created(
        command="python3 solution.py",
        inferred_class="heavy_cpu_candidate",
        gpu_ids=[],
        timeout_sec=60,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

    observer.resource_cleanup_heartbeat(
        job_id,
        action="gpu_cleanup_skipped",
        reason="workspace_cleanup_after_command_finally",
        elapsed_sec=0.2,
        placement={"skipped_count": 1, "skipped": [{"skip_reason": "cwd_unreadable"}]},
    )
    observer.job_created(
        command="python3 inspect.py",
        inferred_class="heavy_cpu_candidate",
        gpu_ids=[],
        timeout_sec=60,
        workspace_dir=tmp_path,
    )

    assert payloads(sm, "resource_cleanup_heartbeat")
    assert not payloads(sm, "resource_guard_action")
    assert not payloads(sm, "post_feedback_action")
