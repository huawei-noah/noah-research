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
import logging
from dataclasses import dataclass

logger = logging.getLogger("scienceflow")


@dataclass
class GPUMetrics:
    gpu_id: int = 0
    utilization: float = 0.0
    memory_used: int = 0
    memory_total: int = 0
    temperature: int = 0
    power_draw: float = 0.0


class GPUMonitor:
    """Monitor GPU metrics, optionally scoped to specific GPU IDs."""

    QUERY_FIELDS = "index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"

    def __init__(self, interval: float = 5.0, gpu_ids: list[int] | None = None):
        self.interval = interval
        self.gpu_ids = set(gpu_ids) if gpu_ids else None
        self._running = False
        self._latest: list[GPUMetrics] = []
        self._callbacks: list = []

    def on_update(self, callback):
        self._callbacks.append(callback)

    @property
    def latest(self) -> list[GPUMetrics]:
        return self._latest

    async def poll_once(self) -> list[GPUMetrics]:
        cmd = ["nvidia-smi"]
        if self.gpu_ids is not None:
            cmd.extend(["--id=" + ",".join(str(g) for g in sorted(self.gpu_ids))])
        cmd.extend([
            "--query-gpu=" + self.QUERY_FIELDS,
            "--format=csv,noheader,nounits",
        ])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return []

        metrics: list[GPUMetrics] = []
        for line in stdout.decode().strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gid = int(parts[0])
            if self.gpu_ids is not None and gid not in self.gpu_ids:
                continue
            metrics.append(
                GPUMetrics(
                    gpu_id=gid,
                    utilization=float(parts[1]),
                    memory_used=int(parts[2]),
                    memory_total=int(parts[3]),
                    temperature=int(parts[4]),
                    power_draw=float(parts[5]),
                )
            )
        return metrics

    async def poll_loop(self):
        self._running = True
        while self._running:
            try:
                self._latest = await self.poll_once()
                for cb in self._callbacks:
                    await cb(self._latest)
            except Exception as e:
                logger.debug(f"GPU poll error: {e}")
            await asyncio.sleep(self.interval)

    def stop(self):
        self._running = False
