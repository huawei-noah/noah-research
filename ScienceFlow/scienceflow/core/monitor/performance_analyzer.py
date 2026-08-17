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

import time
from typing import TYPE_CHECKING, Any

from scienceflow.core.monitor.cpu_monitor import CPUMonitor
from scienceflow.core.monitor.gpu_monitor import GPUMonitor

if TYPE_CHECKING:
    pass

_LOW_GPU_UTIL_THRESHOLD = 30.0
_HIGH_MEM_THRESHOLD = 90.0
_SUSTAINED_WINDOW = 3


class PerformanceAnalyzer:
    def __init__(
        self,
        monitor_agent: Any,
        gpu_monitor: GPUMonitor,
        cpu_monitor: CPUMonitor,
        task_name: str = "",
        skill_inject_fn: Any = None,
    ):
        self.monitor_agent = monitor_agent
        self.gpu_monitor = gpu_monitor
        self.cpu_monitor = cpu_monitor
        self.task_name = task_name
        self._skill_inject_fn = skill_inject_fn
        self._history: list[dict] = []

    async def analyze(self) -> dict:
        snapshot = self._build_snapshot()
        self._history.append(snapshot)

        simple = self.check_simple_rules()
        if simple is not None:
            return simple

        prompt = self._format_prompt(snapshot)
        system_prompt = ""
        if self._skill_inject_fn:
            system_prompt = self._skill_inject_fn("{skill_context}", "monitor")
        decision = await self.monitor_agent.run(system_prompt, prompt)
        return decision

    def _build_snapshot(self) -> dict:
        snapshot: dict[str, Any] = {"timestamp": time.time()}

        gpu_metrics = self.gpu_monitor.latest
        if gpu_metrics:
            snapshot["gpus"] = [
                {
                    "id": g.gpu_id,
                    "util": g.utilization,
                    "mem_used": g.memory_used,
                    "mem_total": g.memory_total,
                    "temp": g.temperature,
                    "power": g.power_draw,
                }
                for g in gpu_metrics
            ]

        cpu_metrics = self.cpu_monitor.latest
        if cpu_metrics:
            snapshot["cpu"] = {
                "cpu_percent": cpu_metrics.cpu_percent,
                "mem_percent": cpu_metrics.memory_percent,
                "mem_used_gb": cpu_metrics.memory_used_gb,
                "io_read": cpu_metrics.io_read_bytes,
                "io_write": cpu_metrics.io_write_bytes,
            }
            if cpu_metrics.per_core:
                snapshot["cpu"]["per_core"] = cpu_metrics.per_core

        return snapshot

    def _format_prompt(self, snapshot: dict) -> str:
        if self.task_name:
            lines = [
                f"Resource metrics for task '{self.task_name}' "
                f"(isolated resources, NOT global system metrics):"
            ]
        else:
            lines = ["Current system metrics:"]

        if "gpus" in snapshot:
            for g in snapshot["gpus"]:
                lines.append(
                    f"  GPU {g['id']}: util={g['util']:.1f}%, "
                    f"mem={g['mem_used']}/{g['mem_total']} MiB, "
                    f"temp={g['temp']}\u00b0C, power={g['power']:.1f}W"
                )

        if "cpu" in snapshot:
            c = snapshot["cpu"]
            lines.append(
                f"  CPU: {c['cpu_percent']:.1f}% (assigned cores), "
                f"RAM: {c['mem_percent']:.1f}% ({c['mem_used_gb']:.2f} GB used)"
            )
            if "per_core" in c:
                core_strs = [f"core{k}={v:.0f}%" for k, v in sorted(c["per_core"].items())]
                lines.append(f"  Per-core: {', '.join(core_strs)}")

        lines.append("")
        lines.append(
            "Based on these metrics, should we adjust resource allocation? "
            "Respond with a JSON object containing 'action' (one of: 'none', "
            "'increase_batch', 'decrease_batch', 'scale_workers', 'alert') "
            "and 'reason'."
        )
        return "\n".join(lines)

    def check_simple_rules(self) -> dict | None:
        if len(self._history) < _SUSTAINED_WINDOW:
            return None

        recent = self._history[-_SUSTAINED_WINDOW:]

        gpu_underutilized = all(
            all(g["util"] < _LOW_GPU_UTIL_THRESHOLD for g in snap.get("gpus", []))
            for snap in recent
            if snap.get("gpus")
        )
        if gpu_underutilized and any(snap.get("gpus") for snap in recent):
            return {
                "action": "increase_batch",
                "reason": f"GPU utilization below {_LOW_GPU_UTIL_THRESHOLD}% for {_SUSTAINED_WINDOW} consecutive polls",
                "source": "rule",
            }

        mem_critical = all(
            any(
                g["mem_used"] / max(g["mem_total"], 1) * 100 > _HIGH_MEM_THRESHOLD
                for g in snap.get("gpus", [])
            )
            for snap in recent
            if snap.get("gpus")
        )
        if mem_critical and any(snap.get("gpus") for snap in recent):
            return {
                "action": "decrease_batch",
                "reason": f"GPU memory usage above {_HIGH_MEM_THRESHOLD}% for {_SUSTAINED_WINDOW} consecutive polls",
                "source": "rule",
            }

        return None
