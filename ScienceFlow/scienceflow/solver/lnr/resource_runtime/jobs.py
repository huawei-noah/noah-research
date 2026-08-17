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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_LIGHT_CPU,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_READONLY_CPU,
    RESOURCE_UNKNOWN_EXEC,
    RESOURCE_UNKNOWN_GPU_EXEC,
)
from scienceflow.solver.lnr.resource_runtime.source_hints import ResourceSourceHint


TRACKABLE_CLASSES = {
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_UNKNOWN_GPU_EXEC,
    RESOURCE_UNKNOWN_EXEC,
}
TRAIN_CLASSES = {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN}
TT_ALLOWED_AFTER_TIMEOUT = {
    RESOURCE_PURE_TT_CPU,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_LIGHT_CPU,
    RESOURCE_READONLY_CPU,
}


@dataclass
class ResourceJob:
    job_id: str
    command: str
    command_digest: str
    resource_class: str
    gpu_ids: list[str]
    cpu_set: str
    timeout_sec: float
    created_at: float
    workspace_dir: Path | None = None
    candidate_artifact: str = ""
    run_dir: Path | None = None
    run_artifact_dir: Path | None = None
    run_state_path: Path | None = None
    gpu_queue_relevant: bool = False
    gpu_request_count: int | None = None
    source_hint: ResourceSourceHint | None = None
    visible: bool = False
    pid: int | None = None
    pgid: int | None = None
    pid_start_time_epoch: float | None = None
    exit_code_seen: bool = False
    terminal_signal_seen: bool = False
    last_signal: dict[str, Any] = field(default_factory=dict)
    last_progress: dict[str, Any] = field(default_factory=dict)
    metric_history_text: str = ""
    metric_history_line_count: int = 0
    live_metric_state: dict[str, Any] = field(default_factory=dict)
    last_artifact_progress: dict[str, Any] = field(default_factory=dict)
    last_recoverability: dict[str, Any] = field(default_factory=dict)
    last_gpu_util_sample: dict[str, Any] = field(default_factory=dict)
    idle_gpu_lease_samples: int = 0
    dataloader_bottleneck_samples: int = 0
    deliverable_complete_samples: int = 0
    last_deliverable_completion_check: dict[str, Any] = field(default_factory=dict)
    value_hint: dict[str, Any] = field(default_factory=dict)
    post_feedback_gate: dict[str, Any] = field(default_factory=dict)
    progress_signal: str = "unknown"
    progress_signal_windows: int = 0
    progress_signal_state: dict[str, Any] = field(default_factory=dict)
    resource_efficiency_state: dict[str, Any] = field(default_factory=dict)
    stalled_mark_count: int = 0
    gpu_mem_peak_gb: float = 0.0
    deliverable_validity: str = "none"
    route_viability_proven: bool = False
    route_viability_state: dict[str, Any] = field(default_factory=dict)
    active_lease_suspect_state: dict[str, Any] = field(default_factory=dict)
    quick_probe: dict[str, Any] = field(default_factory=dict)
    observe_first_state: dict[str, Any] = field(default_factory=dict)
    lease_mode: str = ""
    shared_primary_job_id: str = ""
    monitor_heartbeat_count: int = 0
    last_monitor_heartbeat_elapsed_sec: float = 0.0
