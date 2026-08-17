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

from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
)


SOFT_GPU_GATE_CLASSES = {
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_UNKNOWN_GPU_EXEC,
}

RESOURCE_UNLOCK_CONDITIONS = {
    "",
    "holder_released",
    "resource_context_changed",
    "resource_pressure_cleared",
    "resource_slot_available",
}

RECONCILE_BLOCKED_REASONS = {
    "active_gpu_reservation_present",
    "gpu_not_observed_free",
    "gpu_sample_unavailable",
    "gpu_sample_missing_ids",
}


def normalize_gpu_ids(gpu_ids: list[str] | None) -> list[str]:
    return [str(x) for x in (gpu_ids or []) if str(x).strip()]


def feedback_guard_is_hard_constraint(guard: dict[str, Any] | None) -> bool:
    if not isinstance(guard, dict):
        return False
    reason = str(guard.get("reason") or guard.get("source_reason") or "").strip().lower()
    if any(token in reason for token in {"boundary", "schema", "deliverable"}):
        return True
    unlock = str(guard.get("unlock_condition") or "").strip().lower()
    if bool(guard.get("blocked_until_unlock")) and unlock not in RESOURCE_UNLOCK_CONDITIONS:
        return True
    return False


def resource_class_can_use_soft_gpu_gate(resource_class: str) -> bool:
    return str(resource_class or "") in SOFT_GPU_GATE_CLASSES


def reconcile_free_gpu_pressure_blocked(reconcile: dict[str, Any]) -> bool:
    return str((reconcile or {}).get("reason") or "") in RECONCILE_BLOCKED_REASONS


def gpu_store_snapshot_has_lease(snapshot: dict[str, Any], gpu_ids: list[str] | None) -> bool:
    wanted = set(normalize_gpu_ids(gpu_ids))
    if not wanted:
        return True
    for row in (snapshot.get("leases") or {}).values():
        if not isinstance(row, dict):
            continue
        leased = set(normalize_gpu_ids(row.get("gpu_ids") or []))
        if wanted & leased:
            return True
    return False


def gpu_store_snapshot_has_pressure(
    snapshot: dict[str, Any],
    gpu_ids: list[str] | None,
    *,
    include_waiters: bool = True,
) -> bool:
    if gpu_store_snapshot_has_lease(snapshot, gpu_ids):
        return True
    if not include_waiters:
        return False
    wanted = set(normalize_gpu_ids(gpu_ids))
    if not wanted:
        return True
    for row in (snapshot.get("waiters") or {}).values():
        if not isinstance(row, dict):
            continue
        waiting = set(normalize_gpu_ids(row.get("gpu_ids") or []))
        if wanted & waiting:
            return True
    return False
