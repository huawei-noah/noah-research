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


_DELETED_SUFFIX = " (deleted)"


def read_proc_info(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return {"pid": pid, "exists": False, "missing_reason": "pid_disappeared"}
    info: dict[str, Any] = {"pid": pid, "exists": True}
    try:
        cwd = os.readlink(proc / "cwd")
        if cwd.endswith(_DELETED_SUFFIX):
            cwd = cwd[: -len(_DELETED_SUFFIX)]
        info["cwd"] = cwd
        info["cwd_resolved"] = str(Path(cwd).resolve(strict=False))
    except OSError as exc:
        info["cwd_error"] = type(exc).__name__
    try:
        raw = (proc / "cmdline").read_bytes()
        info["cmdline"] = raw.replace(b"\0", b" ").decode(errors="replace").strip()[:500]
    except OSError as exc:
        info["cmdline_error"] = type(exc).__name__
    uid = read_proc_uid(proc)
    if uid is not None:
        info["uid"] = uid
    start_epoch = read_proc_start_epoch(proc)
    if start_epoch is not None:
        info["start_time_epoch"] = start_epoch
    return info


def read_proc_uid(proc: Path) -> int | None:
    try:
        for line in (proc / "status").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Uid:"):
                parts = line.split()
                return int(parts[1]) if len(parts) > 1 else None
    except (OSError, ValueError):
        return None
    return None


def read_proc_start_epoch(proc: Path) -> float | None:
    boot = boot_time_epoch()
    if boot is None:
        return None
    try:
        raw = (proc / "stat").read_text(encoding="utf-8", errors="replace")
        tail = raw.rsplit(") ", 1)[1].split()
        ticks = int(tail[19])
        hz = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK"))
        return float(boot) + (float(ticks) / float(hz or 100))
    except (OSError, IndexError, ValueError, TypeError):
        return None


def boot_time_epoch() -> float | None:
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
