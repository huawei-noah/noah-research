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
import signal
import time
from pathlib import Path
from typing import Any, Callable

from scienceflow.solver.lnr.resource_runtime.proc_inspect import int_or_none, read_proc_info
from scienceflow.solver.lnr.resource_runtime.utilization import sample_nvidia_compute_apps


SCHEMA_VERSION = 1


def scan_workspace_gpu_processes(
    *,
    workspace_dir: str | Path,
    allowed_gpu_ids: list[str],
    task_started_at: float,
    sample_compute_apps: Callable[..., dict[str, Any]] = sample_nvidia_compute_apps,
    current_uid: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    workspace = _resolve_workspace(workspace_dir)
    allowed = set(_clean_ids(allowed_gpu_ids))
    if workspace is None:
        return _base_result(
            workspace_dir=workspace_dir,
            allowed_gpu_ids=allowed_gpu_ids,
            task_started_at=task_started_at,
            available=False,
            reason="workspace_missing",
        )
    if current_uid is None and hasattr(os, "getuid"):
        current_uid = os.getuid()
    try:
        sample = sample_compute_apps(timeout_sec=1.5)
    except Exception as exc:
        return _base_result(
            workspace_dir=workspace,
            allowed_gpu_ids=allowed_gpu_ids,
            task_started_at=task_started_at,
            available=False,
            reason=type(exc).__name__,
        )
    if not isinstance(sample, dict) or sample.get("available") is False:
        return _base_result(
            workspace_dir=workspace,
            allowed_gpu_ids=allowed_gpu_ids,
            task_started_at=task_started_at,
            available=False,
            reason=str((sample or {}).get("reason") or "nvidia_smi_unavailable"),
        )

    result = _base_result(
        workspace_dir=workspace,
        allowed_gpu_ids=allowed_gpu_ids,
        task_started_at=task_started_at,
        available=True,
        reason="",
    )
    for app in sample.get("apps") or []:
        if not isinstance(app, dict):
            continue
        pid = int_or_none(app.get("pid"))
        if pid is None or pid <= 0:
            continue
        proc = read_proc_info(pid)
        row = _process_row(app, proc)
        skip = _safe_skip_reason(
            proc,
            workspace=workspace,
            task_started_at=float(task_started_at or 0.0),
            current_uid=current_uid,
        )
        if skip:
            row["skip_reason"] = skip
            result["skipped"].append(row)
            continue
        result["matched_processes"].append(row)
        gpu_id = str(row.get("gpu_id") or "").strip()
        if allowed and gpu_id and gpu_id not in allowed:
            result["violations"].append(row)
    result["matched_count"] = len(result["matched_processes"])
    result["violation_count"] = len(result["violations"])
    result["skipped_count"] = len(result["skipped"])
    result["observed_at"] = float(now if now is not None else time.time())
    return result


def cleanup_workspace_gpu_processes(
    *,
    workspace_dir: str | Path,
    allowed_gpu_ids: list[str],
    task_started_at: float,
    reason: str,
    dry_run: bool = False,
    max_kill: int = 32,
    sigterm_grace_sec: float = 1.0,
    sample_compute_apps: Callable[..., dict[str, Any]] = sample_nvidia_compute_apps,
) -> dict[str, Any]:
    result = scan_workspace_gpu_processes(
        workspace_dir=workspace_dir,
        allowed_gpu_ids=allowed_gpu_ids,
        task_started_at=task_started_at,
        sample_compute_apps=sample_compute_apps,
    )
    result["cleanup_reason"] = str(reason or "workspace_gpu_cleanup")
    result["dry_run"] = bool(dry_run)
    result["killed"] = []
    result["would_kill"] = []
    if not result.get("available"):
        return result

    candidates = list(result.get("matched_processes") or [])
    for idx, row in enumerate(candidates):
        if idx >= max(0, int(max_kill or 0)):
            skipped = dict(row)
            skipped["skip_reason"] = "max_kill_limit"
            result["skipped"].append(skipped)
            continue
        pid = int_or_none(row.get("pid"))
        if pid is None or pid <= 0:
            continue
        if dry_run:
            result["would_kill"].append(dict(row))
            continue
        killed = _terminate_pid(pid, grace_sec=max(0.0, float(sigterm_grace_sec or 0.0)))
        out = dict(row)
        out.update(killed)
        result["killed"].append(out)
    result["killed_count"] = len(result["killed"])
    result["would_kill_count"] = len(result["would_kill"])
    result["skipped_count"] = len(result["skipped"])
    return result


def _base_result(
    *,
    workspace_dir: str | Path,
    allowed_gpu_ids: list[str],
    task_started_at: float,
    available: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": bool(available),
        "reason": str(reason or ""),
        "workspace_dir": str(workspace_dir),
        "allowed_gpu_ids": _clean_ids(allowed_gpu_ids),
        "task_started_at": float(task_started_at or 0.0),
        "matched_processes": [],
        "violations": [],
        "skipped": [],
        "matched_count": 0,
        "violation_count": 0,
        "skipped_count": 0,
    }


def _clean_ids(values: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return [str(x).strip() for x in values if str(x).strip()]


def _resolve_workspace(value: str | Path) -> Path | None:
    try:
        path = Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    return path if path.exists() else None


def _process_row(app: dict[str, Any], proc: dict[str, Any]) -> dict[str, Any]:
    return {
        "pid": int_or_none(app.get("pid")) or 0,
        "gpu_id": str(app.get("gpu_id") or "").strip(),
        "gpu_uuid": str(app.get("gpu_uuid") or "").strip(),
        "used_memory_mb": app.get("used_memory_mb"),
        "cwd": str(proc.get("cwd") or ""),
        "cmdline": str(proc.get("cmdline") or "")[:500],
        "uid": proc.get("uid"),
        "start_time_epoch": proc.get("start_time_epoch"),
    }


def _safe_skip_reason(
    proc: dict[str, Any],
    *,
    workspace: Path,
    task_started_at: float,
    current_uid: int | None,
) -> str:
    if proc.get("exists") is False:
        return str(proc.get("missing_reason") or "pid_disappeared")
    if current_uid is not None and proc.get("uid") != current_uid:
        return "owner_mismatch"
    start_epoch = proc.get("start_time_epoch")
    if task_started_at > 0:
        if not isinstance(start_epoch, (int, float)):
            return "start_time_unavailable"
        if float(start_epoch) + 5.0 < task_started_at:
            return "process_started_before_task"
    cwd_resolved = str(proc.get("cwd_resolved") or "")
    if not cwd_resolved:
        return "cwd_unreadable"
    try:
        cwd_path = Path(cwd_resolved)
        cwd_path.relative_to(workspace)
    except ValueError:
        return "workspace_mismatch"
    return ""


def _terminate_pid(pid: int, *, grace_sec: float) -> dict[str, Any]:
    result: dict[str, Any] = {"terminated": False, "sigkilled": False}
    try:
        os.kill(pid, signal.SIGTERM)
        result["terminated"] = True
    except ProcessLookupError:
        result["already_exited"] = True
        return result
    except PermissionError:
        result["kill_error"] = "permission_denied"
        return result
    if grace_sec > 0:
        time.sleep(min(grace_sec, 5.0))
    if not (Path("/proc") / str(pid)).exists():
        return result
    try:
        os.kill(pid, signal.SIGKILL)
        result["sigkilled"] = True
    except ProcessLookupError:
        result["already_exited"] = True
    except PermissionError:
        result["kill_error"] = "permission_denied"
    return result
