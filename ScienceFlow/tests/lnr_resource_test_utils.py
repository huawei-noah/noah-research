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

from pathlib import Path
from typing import Any

from scienceflow.solver.lnr.resource_observer import LHRResourceObserver


class FakeStateMachine:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    @property
    def event_count(self) -> int:
        return len(self.events)

    def append_event(self, event_type: str, **kwargs: Any) -> None:
        self.events.append((event_type, kwargs))


def make_observer(
    tmp_path: Path,
    *,
    worker_id: str = "W00",
    state_machine: FakeStateMachine | None = None,
    gpu_pool: list[str] | None = None,
    assignment: str = "env_only",
    max_wait_sec: float = 0.2,
    min_register_sec: float = 0.0,
    stalled_stdout_sec: float = 0.0,
    low_progress_enabled: bool = False,
    **kwargs: Any,
) -> tuple[LHRResourceObserver, FakeStateMachine]:
    sm = state_machine or FakeStateMachine()
    params = {
        "state_machine": sm,
        "worker_id": worker_id,
        "min_register_sec": min_register_sec,
        "check_interval_sec": 1.0,
        "stalled_stdout_sec": stalled_stdout_sec,
        "low_progress_enabled": low_progress_enabled,
        "task_resource_dir": tmp_path / "resource",
        "resource_runtime_enabled": True,
        "gpu_queue_enabled": True,
        "gpu_pool": gpu_pool or [],
        "gpu_assignment": assignment,
        "gpu_queue_max_wait_sec": max_wait_sec,
        "gpu_queue_heartbeat_sec": 0.1,
        "gpu_max_heavy_per_gpu": 1,
        "gpu_capacity_slots": 1.0,
        "gpu_pressure_min_free_mem_gb": 0.0,
        "gpu_pressure_yellow_free_mem_buffer_gb": 0.0,
        "gpu_util_sample_interval_sec": 1.0,
    }
    params.update(kwargs)
    observer = LHRResourceObserver(**params)
    return observer, sm


def event_types(sm: FakeStateMachine) -> list[str]:
    return [event for event, _ in sm.events]


def payloads(sm: FakeStateMachine, event_type: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event, kwargs in sm.events:
        if event == event_type:
            payload = kwargs.get("payload") if isinstance(kwargs.get("payload"), dict) else {}
            out.append(payload)
    return out
