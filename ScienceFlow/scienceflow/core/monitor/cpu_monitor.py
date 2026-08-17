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
import os
from dataclasses import dataclass

import psutil

logger = logging.getLogger("scienceflow")


@dataclass
class CPUMetrics:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    per_core: dict[int, float] | None = None


class CPUMonitor:
    """Monitor CPU metrics, optionally scoped to specific CPU core IDs."""

    def __init__(
        self,
        interval: float = 5.0,
        cpu_ids: list[int] | None = None,
        pid: int | None = None,
    ):
        self.interval = interval
        self.cpu_ids = set(cpu_ids) if cpu_ids else None
        self.pid = pid or os.getpid()
        self._running = False
        self._latest: CPUMetrics | None = None
        self._callbacks: list = []

    def on_update(self, callback):
        self._callbacks.append(callback)

    @property
    def latest(self) -> CPUMetrics | None:
        return self._latest

    async def poll_once(self) -> CPUMetrics:
        if self.cpu_ids is not None:
            return await self._poll_scoped()
        return await self._poll_global()

    async def _poll_global(self) -> CPUMetrics:
        cpu_pct = await asyncio.to_thread(psutil.cpu_percent, interval=0.1)
        vmem = await asyncio.to_thread(psutil.virtual_memory)
        disk_io = await asyncio.to_thread(psutil.disk_io_counters)

        return CPUMetrics(
            cpu_percent=cpu_pct,
            memory_percent=vmem.percent,
            memory_used_gb=round(vmem.used / (1024**3), 2),
            io_read_bytes=disk_io.read_bytes if disk_io else 0,
            io_write_bytes=disk_io.write_bytes if disk_io else 0,
        )

    async def _poll_scoped(self) -> CPUMetrics:
        per_cpu = await asyncio.to_thread(psutil.cpu_percent, interval=0.1, percpu=True)

        per_core: dict[int, float] = {}
        scoped_pcts: list[float] = []
        for core_id in sorted(self.cpu_ids):
            if core_id < len(per_cpu):
                per_core[core_id] = per_cpu[core_id]
                scoped_pcts.append(per_cpu[core_id])

        avg_pct = sum(scoped_pcts) / len(scoped_pcts) if scoped_pcts else 0.0

        mem_used_gb = 0.0
        mem_pct = 0.0
        try:
            proc = psutil.Process(self.pid)
            mem_info = proc.memory_info()
            mem_used_gb = round(mem_info.rss / (1024**3), 2)
            vmem = psutil.virtual_memory()
            mem_pct = round(mem_info.rss / vmem.total * 100, 1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            vmem = await asyncio.to_thread(psutil.virtual_memory)
            mem_pct = vmem.percent
            mem_used_gb = round(vmem.used / (1024**3), 2)

        disk_io = await asyncio.to_thread(psutil.disk_io_counters)

        return CPUMetrics(
            cpu_percent=round(avg_pct, 1),
            memory_percent=mem_pct,
            memory_used_gb=mem_used_gb,
            io_read_bytes=disk_io.read_bytes if disk_io else 0,
            io_write_bytes=disk_io.write_bytes if disk_io else 0,
            per_core=per_core,
        )

    async def poll_loop(self):
        self._running = True
        while self._running:
            try:
                self._latest = await self.poll_once()
                for cb in self._callbacks:
                    await cb(self._latest)
            except Exception as e:
                logger.debug(f"CPU poll error: {e}")
            await asyncio.sleep(self.interval)

    def stop(self):
        self._running = False
