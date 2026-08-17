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

from typing import Any


MAGENT_TRAIN_OBSERVED = "magent_train_observed"
MAGENT_FORK_CONSIDERED = "magent_fork_considered"
MAGENT_SIDECAR_STARTED = "magent_sidecar_started"
MAGENT_JOIN_PACKET_READY = "magent_join_packet_ready"


def train_observed_payload(
    *,
    worker_id: str,
    parent_job_id: str,
    parent_state: str,
    elapsed_sec: float,
    resource_class: str,
    resource_mode: str,
) -> dict[str, Any]:
    return {
        "worker_id": str(worker_id or ""),
        "parent_job_id": str(parent_job_id or ""),
        "parent_state": str(parent_state or ""),
        "elapsed_sec": float(elapsed_sec or 0.0),
        "resource_class": str(resource_class or ""),
        "resource_mode": str(resource_mode or "UNKNOWN"),
    }


def fork_considered_payload(
    *,
    worker_id: str,
    parent_job_id: str,
    decision: str,
    reason: str,
    task_type: str,
    sidecar_mode: str,
) -> dict[str, Any]:
    return {
        "worker_id": str(worker_id or ""),
        "parent_job_id": str(parent_job_id or ""),
        "decision": str(decision or "skip"),
        "reason": str(reason or ""),
        "task_type": str(task_type or ""),
        "sidecar_mode": str(sidecar_mode or "cpu_only"),
    }


def sidecar_started_payload(
    *,
    sidecar_id: str,
    parent_worker_id: str,
    parent_job_id: str,
    budget_sec: float,
    task_type: str,
    workspace: str,
) -> dict[str, Any]:
    return {
        "sidecar_id": str(sidecar_id or ""),
        "parent_worker_id": str(parent_worker_id or ""),
        "parent_job_id": str(parent_job_id or ""),
        "budget_sec": float(budget_sec or 0.0),
        "task_type": str(task_type or ""),
        "workspace": str(workspace or ""),
    }


def join_packet_ready_payload(
    *,
    sidecar_id: str,
    parent_worker_id: str,
    quality_gate: str,
    inject_targets: list[str],
    artifact_count: int,
) -> dict[str, Any]:
    return {
        "sidecar_id": str(sidecar_id or ""),
        "parent_worker_id": str(parent_worker_id or ""),
        "quality_gate": str(quality_gate or ""),
        "inject_targets": [str(x) for x in inject_targets if str(x).strip()],
        "artifact_count": max(0, int(artifact_count or 0)),
    }
