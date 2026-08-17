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

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProcessLifecycle:
    status: str
    reason: str
    active_work_detected: bool
    safe_to_mark_finished: bool
    safe_to_terminate_pid: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "active_work_detected": self.active_work_detected,
            "safe_to_mark_finished": self.safe_to_mark_finished,
            "safe_to_terminate_pid": self.safe_to_terminate_pid,
        }


def classify_process_lifecycle(
    *,
    process_liveness: dict[str, Any] | None,
    process_tree_cpu: dict[str, Any] | None = None,
    tool_waiting: bool = True,
) -> ProcessLifecycle:
    """Normalize low-level process facts into lifecycle facts for arbitration."""

    live = process_liveness if isinstance(process_liveness, dict) else {}
    cpu = process_tree_cpu if isinstance(process_tree_cpu, dict) else {}
    root_alive = bool(live.get("root_pid_alive"))
    group_alive = bool(live.get("process_group_alive"))
    live_descendants = _int_or_zero(live.get("live_descendant_count"))
    descendants = _int_or_zero(live.get("descendant_count"))
    zombies = _int_or_zero(live.get("zombie_descendant_count"))
    busy_children = max(_int_or_zero(live.get("busy_child_count")), _int_or_zero(cpu.get("busy_child_count")))
    total_cpu = _float_or_zero(cpu.get("total_cpu_pct"))
    terminal_seen = bool(live.get("exit_code_seen") or live.get("terminal_signal_seen"))
    status = str(live.get("status") or "unknown")
    reason = str(live.get("reason") or "")

    if live.get("owner_pid_start_time_match") is False:
        return ProcessLifecycle(
            status="stale_pid_binding",
            reason="root_pid_start_time_mismatch",
            active_work_detected=False,
            safe_to_mark_finished=True,
            safe_to_terminate_pid=False,
        )

    active_work = bool(root_alive or group_alive or live_descendants > 0 or busy_children > 0 or total_cpu >= 5.0)
    if active_work:
        if busy_children > 0 or total_cpu >= 5.0:
            return ProcessLifecycle(
                status="alive_with_busy_children",
                reason="process_group_has_active_work",
                active_work_detected=True,
                safe_to_mark_finished=False,
                safe_to_terminate_pid=False,
            )
        return ProcessLifecycle(
            status="alive",
            reason=reason or "process_or_group_alive",
            active_work_detected=True,
            safe_to_mark_finished=False,
            safe_to_terminate_pid=False,
        )

    if zombies > 0 and descendants <= zombies:
        return ProcessLifecycle(
            status="zombie_only",
            reason="only_zombie_descendants_remain",
            active_work_detected=False,
            safe_to_mark_finished=True,
            safe_to_terminate_pid=False,
        )

    if terminal_seen or status == "exited":
        return ProcessLifecycle(
            status="completed_but_waiting" if tool_waiting else "completed",
            reason="terminal_signal_seen_and_no_active_processes",
            active_work_detected=False,
            safe_to_mark_finished=True,
            safe_to_terminate_pid=False,
        )

    return ProcessLifecycle(
        status="unknown",
        reason=reason or "missing_terminal_signal_and_no_active_processes",
        active_work_detected=False,
        safe_to_mark_finished=False,
        safe_to_terminate_pid=False,
    )


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_zero(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out else 0.0
