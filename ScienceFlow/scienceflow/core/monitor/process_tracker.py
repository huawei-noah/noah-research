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
from dataclasses import dataclass, field
from enum import Enum


class ProcessStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    KILLED = "killed"
    TIMEOUT = "timeout"


@dataclass
class ProcessInfo:
    pid: int
    node_id: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    exit_code: int | None = None
    status: ProcessStatus = ProcessStatus.RUNNING


class ProcessTracker:
    def __init__(self):
        self._processes: dict[int, ProcessInfo] = {}

    def register(self, pid: int, node_id: str) -> ProcessInfo:
        info = ProcessInfo(pid=pid, node_id=node_id)
        self._processes[pid] = info
        return info

    def mark_complete(self, pid: int, exit_code: int) -> ProcessInfo | None:
        info = self._processes.get(pid)
        if info is None:
            return None
        info.status = ProcessStatus.COMPLETED
        info.exit_code = exit_code
        info.end_time = time.time()
        return info

    def mark_killed(self, pid: int) -> ProcessInfo | None:
        info = self._processes.get(pid)
        if info is None:
            return None
        info.status = ProcessStatus.KILLED
        info.end_time = time.time()
        return info

    def mark_timeout(self, pid: int) -> ProcessInfo | None:
        info = self._processes.get(pid)
        if info is None:
            return None
        info.status = ProcessStatus.TIMEOUT
        info.end_time = time.time()
        return info

    def get_active(self) -> list[ProcessInfo]:
        return [p for p in self._processes.values() if p.status == ProcessStatus.RUNNING]

    def get_by_node(self, node_id: str) -> list[ProcessInfo]:
        return [p for p in self._processes.values() if p.node_id == node_id]

    def get(self, pid: int) -> ProcessInfo | None:
        return self._processes.get(pid)

    @property
    def all_processes(self) -> list[ProcessInfo]:
        return list(self._processes.values())
