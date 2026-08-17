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
from pathlib import Path
from typing import Any


def build_process_liveness(
    *,
    root_pid: int | None,
    root_pgid: int | None = None,
    expected_root_start_time_epoch: float | None = None,
    exit_code_seen: bool = False,
    terminal_signal_seen: bool = False,
    process_tree_cpu: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a conservative process-liveness fact card for resource arbitration.

    Completion is proven only when the root process is gone, the tracked process
    group is empty, descendants are absent, and the caller observed an exit or a
    terminal signal. Otherwise the arbiter sees alive/unknown/inconsistent.
    """

    pid = _int_or_zero(root_pid)
    if pid <= 0:
        return {
            "status": "unknown",
            "root_pid": 0,
            "root_pid_alive": False,
            "reason": "missing_root_pid",
            "exit_code_seen": bool(exit_code_seen),
            "terminal_signal_seen": bool(terminal_signal_seen),
        }

    rows = _proc_rows()
    root = rows.get(pid)
    root_alive = root is not None and str(root.get("state") or "") != "Z"
    pgid = _int_or_zero(root_pgid) or _int_or_zero((root or {}).get("pgid")) or pid
    session_id = _int_or_zero((root or {}).get("session"))
    descendants = _descendant_pids(rows, pid)
    live_descendants = sorted(
        child for child in descendants if str((rows.get(child) or {}).get("state") or "") != "Z"
    )
    zombie_descendants = sorted(
        child for child in descendants if str((rows.get(child) or {}).get("state") or "") == "Z"
    )
    group_members = sorted(
        row_pid
        for row_pid, row in rows.items()
        if _int_or_zero(row.get("pgid")) == pgid and str(row.get("state") or "") != "Z"
    )
    child_like = sorted(set(descendants) | ({x for x in group_members if x != pid}))
    cpu = process_tree_cpu if isinstance(process_tree_cpu, dict) else {}
    busy_child_count = _int_or_zero(cpu.get("busy_child_count"))
    python_child_count = _int_or_zero(cpu.get("python_child_count"))
    if python_child_count <= 0:
        python_child_count = sum(1 for child in child_like if _looks_python(rows.get(child, {})))
    start_match = True
    current_start = None
    if root is not None:
        current_start = _float_or_none(root.get("start_time_epoch"))
    expected_start = _float_or_none(expected_root_start_time_epoch)
    if expected_start is not None and current_start is not None:
        start_match = abs(expected_start - current_start) < 0.01

    live_evidence = bool(root_alive or live_descendants or group_members)
    terminal_seen = bool(exit_code_seen or terminal_signal_seen)
    if live_evidence and terminal_seen:
        status = "inconsistent"
        reason = "terminal_signal_but_process_alive"
    elif root_alive and not start_match:
        status = "inconsistent"
        reason = "root_pid_start_time_mismatch"
    elif live_evidence:
        status = "alive"
        reason = "process_or_group_alive"
    elif terminal_seen:
        status = "exited"
        reason = "terminal_signal_seen_and_no_processes"
    else:
        status = "unknown"
        reason = "process_missing_without_terminal_signal"

    return {
        "status": status,
        "reason": reason,
        "root_pid": pid,
        "root_pid_alive": bool(root_alive),
        "root_pgid": pgid,
        "root_session_id": session_id,
        "descendant_count": len(descendants),
        "live_descendant_count": len(live_descendants),
        "zombie_descendant_count": len(zombie_descendants),
        "process_group_member_count": len(group_members),
        "busy_child_count": busy_child_count,
        "python_child_count": python_child_count,
        "process_group_alive": bool(group_members),
        "owner_pid_start_time_match": bool(start_match),
        "root_start_time_epoch": current_start,
        "expected_root_start_time_epoch": expected_start,
        "exit_code_seen": bool(exit_code_seen),
        "terminal_signal_seen": bool(terminal_signal_seen),
        "last_known_pid_seen_at": None,
        "child_pids_sample": child_like[:10],
    }


def _proc_rows() -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    proc = Path("/proc")
    for path in proc.iterdir() if proc.exists() else []:
        if not path.name.isdigit():
            continue
        pid = _int_or_zero(path.name)
        stat = _read_stat(path)
        if stat:
            rows[pid] = stat
    return rows


def _read_stat(proc_path: Path) -> dict[str, Any]:
    try:
        raw = (proc_path / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    right = raw.rfind(")")
    if right < 0:
        return {}
    try:
        pid = int(raw[: raw.find(" ")])
    except (TypeError, ValueError):
        pid = _int_or_zero(proc_path.name)
    comm = raw[raw.find("(") + 1 : right]
    tail = raw[right + 2 :].split()
    if len(tail) < 20:
        return {}
    boot = _boot_time_epoch()
    hz = _clock_ticks_per_second()
    start_ticks = _float_or_none(tail[19])
    start_epoch = None
    if boot is not None and start_ticks is not None and hz > 0:
        start_epoch = float(boot) + float(start_ticks) / float(hz)
    return {
        "pid": pid,
        "comm": comm,
        "state": tail[0],
        "ppid": _int_or_zero(tail[1]),
        "pgid": _int_or_zero(tail[2]),
        "session": _int_or_zero(tail[3]),
        "start_time_epoch": start_epoch,
    }


def _descendant_pids(rows: dict[int, dict[str, Any]], root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for pid, row in rows.items():
        children.setdefault(_int_or_zero(row.get("ppid")), []).append(pid)
    stack = list(children.get(root_pid, []))
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return sorted(seen)


def _looks_python(row: dict[str, Any]) -> bool:
    return str(row.get("comm") or "").lower().startswith("python")


def _boot_time_epoch() -> float | None:
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _clock_ticks_per_second() -> int:
    try:
        return int(os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK")))
    except (OSError, TypeError, ValueError):
        return 100


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None
