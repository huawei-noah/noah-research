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

import re
import shlex
import subprocess
from typing import Any


def sample_nvidia_smi(gpu_ids: list[str] | None = None, *, timeout_sec: float = 3.0) -> dict[str, Any]:
    """Return a compact, path-free GPU utilization sample.

    Missing nvidia-smi is not an error for CPU-only hosts; callers receive
    available=False and can still keep resource state consistent.
    """

    wanted = {str(x).strip() for x in (gpu_ids or []) if str(x).strip()}
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_sec or 3.0)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "reason": type(exc).__name__,
            "gpus": [],
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": "nvidia_smi_failed",
            "gpus": [],
        }
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_id = parts[0]
        if wanted and gpu_id not in wanted:
            continue
        rows.append(
            {
                "gpu_id": gpu_id,
                "utilization_gpu_pct": _float_or_none(parts[1]),
                "memory_used_mb": _float_or_none(parts[2]),
                "memory_total_mb": _float_or_none(parts[3]),
            }
        )
    return {
        "available": True,
        "reason": "",
        "gpus": rows,
    }




def sample_nvidia_compute_apps(*, timeout_sec: float = 2.0) -> dict[str, Any]:
    """Return compute app PID to physical GPU index mapping from nvidia-smi."""

    try:
        gpu_info = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_sec or 2.0)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__, "apps": []}
    if gpu_info.returncode != 0:
        return {"available": False, "reason": "nvidia_smi_gpu_query_failed", "apps": []}
    uuid_to_index: dict[str, str] = {}
    for line in gpu_info.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            uuid_to_index[parts[1]] = parts[0]

    try:
        apps_info = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_sec or 2.0)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__, "apps": []}
    if apps_info.returncode != 0:
        return {"available": False, "reason": "nvidia_smi_compute_query_failed", "apps": []}

    apps: list[dict[str, Any]] = []
    for line in apps_info.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        pid = _int_or_none(parts[0])
        if pid is None:
            continue
        gpu_uuid = parts[1]
        apps.append(
            {
                "pid": pid,
                "gpu_uuid": gpu_uuid,
                "gpu_id": uuid_to_index.get(gpu_uuid, ""),
                "used_memory_mb": _float_or_none(parts[2]) if len(parts) >= 3 else None,
            }
        )
    return {"available": True, "reason": "", "apps": apps}


def parse_visible_gpu_ids(value: str | None) -> list[str]:
    if not value:
        return []
    visible = str(value).strip().lower()
    if visible in {"-1", "none", "cpu", "nodevfile"}:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def process_tree_cpu_snapshot(root_pid: int | None) -> dict[str, Any]:
    try:
        root = int(root_pid or 0)
    except (TypeError, ValueError):
        return {"available": False, "reason": "invalid_pid"}
    if root <= 0:
        return {"available": False, "reason": "invalid_pid"}
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,pcpu=,comm="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__}
    if completed.returncode != 0:
        return {"available": False, "reason": "ps_failed"}

    rows: dict[int, dict[str, Any]] = {}
    children: dict[int, list[int]] = {}
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            pcpu = float(parts[2])
        except (TypeError, ValueError):
            continue
        rows[pid] = {"pid": pid, "ppid": ppid, "pcpu": pcpu, "comm": parts[3].strip()}
        children.setdefault(ppid, []).append(pid)

    stack = [root]
    seen: set[int] = set()
    descendants: list[dict[str, Any]] = []
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        row = rows.get(pid)
        if row:
            descendants.append(row)
        stack.extend(children.get(pid, []))
    if not descendants:
        return {"available": False, "reason": "process_missing", "root_pid": root}

    child_rows = [row for row in descendants if int(row.get("pid") or 0) != root]
    total_cpu = sum(float(row.get("pcpu") or 0.0) for row in descendants)
    child_cpu = sum(float(row.get("pcpu") or 0.0) for row in child_rows)
    busy = [row for row in child_rows if float(row.get("pcpu") or 0.0) >= 50.0]
    top = sorted(child_rows, key=lambda row: float(row.get("pcpu") or 0.0), reverse=True)[:5]
    return {
        "available": True,
        "root_pid": root,
        "descendant_count": max(0, len(descendants) - 1),
        "total_cpu_pct": round(total_cpu, 3),
        "child_cpu_pct": round(child_cpu, 3),
        "busy_child_count": len(busy),
        "python_child_count": sum(1 for row in child_rows if str(row.get("comm") or "").startswith("python")),
        "top_children": [
            {"pid": int(row.get("pid") or 0), "pcpu": round(float(row.get("pcpu") or 0.0), 3), "comm": str(row.get("comm") or "")[:32]}
            for row in top
        ],
    }


def process_tree_pids(root_pid: int | None) -> set[int]:
    try:
        root = int(root_pid or 0)
    except (TypeError, ValueError):
        return set()
    if root <= 0:
        return set()
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {root}
    if completed.returncode != 0:
        return {root}

    children: dict[int, list[int]] = {}
    rows: set[int] = set()
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except (TypeError, ValueError):
            continue
        rows.add(pid)
        children.setdefault(ppid, []).append(pid)

    stack = [root]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return seen or ({root} if root in rows else set())


def process_tree_gpu_placement_snapshot(root_pid: int | None, allowed_gpu_ids: list[str]) -> dict[str, Any]:
    allowed = {str(x).strip() for x in (allowed_gpu_ids or []) if str(x).strip()}
    if not allowed:
        return {"available": False, "reason": "no_allowed_gpu_ids", "violations": []}
    pids = process_tree_pids(root_pid)
    if not pids:
        return {"available": False, "reason": "process_missing", "violations": []}
    try:
        sample = sample_nvidia_compute_apps(timeout_sec=1.5)
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__, "violations": []}
    if not isinstance(sample, dict) or sample.get("available") is False:
        return {
            "available": False,
            "reason": str((sample or {}).get("reason") or "nvidia_smi_unavailable"),
            "violations": [],
        }

    violations: list[dict[str, Any]] = []
    used: list[dict[str, Any]] = []
    for app in sample.get("apps") or []:
        if not isinstance(app, dict):
            continue
        try:
            pid = int(app.get("pid") or 0)
        except (TypeError, ValueError):
            continue
        if pid not in pids:
            continue
        gpu_id = str(app.get("gpu_id") or "").strip()
        row = {"pid": pid, "gpu_id": gpu_id, "used_memory_mb": app.get("used_memory_mb")}
        used.append(row)
        if gpu_id and gpu_id not in allowed:
            violations.append(row)
    return {
        "available": True,
        "reason": "",
        "root_pid": int(root_pid or 0),
        "allowed_gpu_ids": sorted(allowed),
        "process_count": len(pids),
        "used_gpu_processes": used[:20],
        "violations": violations[:20],
    }


def normalize_cuda_visible_devices_for_task_pool(value: str, task_physical_pool: list[str]) -> tuple[str, list[str], bool, str]:
    ids = parse_visible_gpu_ids(value)
    pool = [str(x).strip() for x in (task_physical_pool or []) if str(x).strip()]
    if not ids or not pool:
        return str(value or ""), ids, False, "no_task_pool"
    if set(ids).issubset(set(pool)):
        return ",".join(ids), ids, False, "physical_subset"
    mapped: list[str] = []
    for raw in ids:
        if not raw.isdigit():
            return str(value or ""), ids, False, "outside_task_pool"
        ordinal = int(raw)
        if ordinal < 0 or ordinal >= len(pool):
            return str(value or ""), ids, False, "outside_task_pool"
        mapped.append(pool[ordinal])
    return ",".join(mapped), mapped, True, "logical_ordinal_mapped_to_task_physical_gpu"


def logical_cuda_ordinals_for_assignment(assigned_gpu_ids: list[str]) -> str:
    ids = [str(x).strip() for x in (assigned_gpu_ids or []) if str(x).strip()]
    return ",".join(str(i) for i, _ in enumerate(ids))


def _scan_shell_word(command: str, start: int) -> tuple[int, int] | None:
    n = len(command)
    i = start
    while i < n and command[i].isspace():
        i += 1
    if i >= n or command[i] in ";&|()<>":
        return None
    word_start = i
    quote = ""
    while i < n:
        ch = command[i]
        if quote:
            if ch == quote:
                quote = ""
                i += 1
                continue
            if quote == '"' and ch == "\\":
                i += 2
                continue
            i += 1
            continue
        if ch.isspace() or ch in ";&|()<>":
            break
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch == "\\":
            i += 2
            continue
        i += 1
    return word_start, i


def replace_leading_cuda_visible_devices(command: str, value: str) -> str:
    """Replace shell env assignments that would override a GPU lease."""

    cmd = str(command or "")
    replacement = f"CUDA_VISIBLE_DEVICES={shlex.quote(str(value))}"
    spans: list[tuple[int, int]] = []
    pos = 0
    while pos < len(cmd):
        word = _scan_shell_word(cmd, pos)
        if word is None:
            pos += 1
            continue
        start, end = word
        token = cmd[start:end]
        if re.match(r"CUDA_VISIBLE_DEVICES=", token):
            spans.append((start, end))
        pos = max(end, pos + 1)
    if not spans:
        return cmd
    out = cmd
    for start, end in reversed(spans):
        out = out[:start] + replacement + out[end:]
    return out

def _float_or_none(raw: str) -> float | None:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _int_or_none(raw: str) -> int | None:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None
