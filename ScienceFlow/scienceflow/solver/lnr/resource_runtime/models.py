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
from typing import Any


@dataclass(frozen=True)
class GPUQueueConfig:
    enabled: bool = False
    gpu_pool: list[str] = field(default_factory=list)
    default_request: int = 1
    max_request: int = 1
    assignment: str = "env_only"
    max_wait_sec: float = 1800.0
    heartbeat_sec: float = 15.0
    max_heavy_per_gpu: int = 1
    capacity_slots: float = 1.0
    gpu_tt_max_per_gpu: int = 3
    gpu_feature_max_per_gpu: int = 2
    share_tt_with_train: bool = False
    lease_ttl_sec: float = 7200.0
    duplicate_digest_cooldown_sec: float = 600.0
    duplicate_digest_threshold: int = 2
    admission_queue_enabled: bool = True
    admission_waiter_ttl_sec: float = 900.0
    pressure_yellow_hold_sec: float = 120.0
    pressure_red_to_yellow_sec: float = 120.0
    pressure_yellow_util_pct: float = 85.0
    pressure_min_free_mem_gb: float = 8.0
    pressure_yellow_free_mem_buffer_gb: float = 8.0
    idle_release_admission_mode: str = "strict_exclusive"


@dataclass
class ResourceLease:
    job_id: str
    worker_id: str
    gpu_ids: list[str]
    acquired_at: float
    heartbeat_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "gpu_ids": list(self.gpu_ids),
            "acquired_at": float(self.acquired_at),
            "heartbeat_at": float(self.heartbeat_at),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ResourceLease":
        return cls(
            job_id=str(data.get("job_id") or ""),
            worker_id=str(data.get("worker_id") or ""),
            gpu_ids=[str(x) for x in (data.get("gpu_ids") or []) if str(x).strip()],
            acquired_at=float(data.get("acquired_at") or 0.0),
            heartbeat_at=float(data.get("heartbeat_at") or 0.0),
            metadata=dict(data.get("metadata") or {}),
        )
