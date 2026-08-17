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

from scienceflow.solver.lnr.resource_feedback_contract import resource_feedback_text

from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
)


GPU_RESOURCE_CLASSES = {
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_UNKNOWN_GPU_EXEC,
}


def _clean_gpu_ids(gpu_ids: list[str] | tuple[str, ...] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in gpu_ids or []:
        gpu_id = str(raw).strip()
        if gpu_id and gpu_id not in seen:
            seen.add(gpu_id)
            out.append(gpu_id)
    return out


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _row_gpu_id(row: dict[str, Any]) -> str:
    return str(row.get("gpu_id") or row.get("index") or "").strip()


def _row_free_mem_gb(row: dict[str, Any]) -> float | None:
    used_mb = _float_or_none(row.get("memory_used_mb"))
    total_mb = _float_or_none(row.get("memory_total_mb"))
    if used_mb is None or total_mb is None or total_mb <= 0:
        return None
    return max(0.0, (total_mb - used_mb) / 1024.0)


def _sample_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def evaluate_yellow_safety_hold(
    *,
    sample: dict[str, Any] | None,
    gpu_ids: list[str],
    util_exit_threshold_pct: float = 50.0,
    min_free_mem_gb: float = 0.0,
) -> dict[str, Any]:
    ids = _clean_gpu_ids(gpu_ids)
    if not ids:
        return {"blocked": False, "gpu_ids": []}
    if not isinstance(sample, dict) or sample.get("available") is False:
        return {"blocked": False, "gpu_ids": ids, "sample": sample if isinstance(sample, dict) else {}}

    wanted = set(ids)
    util_limit = max(0.0, float(util_exit_threshold_pct or 0.0))
    reserve = max(0.0, float(min_free_mem_gb or 0.0))
    blocked: dict[str, dict[str, Any]] = {}
    for row in _sample_rows(sample):
        gpu_id = _row_gpu_id(row)
        if not gpu_id or gpu_id not in wanted:
            continue
        util = _float_or_none(row.get("utilization_gpu_pct")) or 0.0
        free_gb = _row_free_mem_gb(row)
        reasons: list[str] = []
        if util_limit > 0 and util >= util_limit:
            reasons.append("util_p50_exit_guard")
        if reserve > 0 and free_gb is not None and free_gb < reserve:
            reasons.append("free_mem_below_reserve")
        if reasons:
            blocked[gpu_id] = {
                "gpu_id": gpu_id,
                "utilization_gpu_pct": util,
                "free_mem_gb": free_gb,
                "reasons": reasons,
            }
    return {
        "blocked": bool(blocked),
        "gpu_ids": ids,
        "blocked_gpu_ids": sorted(blocked),
        "blocked_gpus": blocked,
        "sample": sample,
        "util_exit_threshold_pct": util_limit,
        "min_free_mem_gb": reserve,
    }


def evaluate_admission_safety(
    *,
    sample: dict[str, Any] | None,
    gpu_ids: list[str],
    resource_class: str,
    request_count: int = 1,
    assignment: str = "env_only",
    min_free_mem_gb: float = 0.0,
    free_mem_buffer_gb: float = 0.0,
) -> dict[str, Any]:
    ids = _clean_gpu_ids(gpu_ids)
    cls = str(resource_class or "").strip()
    mode = str(assignment or "env_only").strip().lower()
    requested = max(1, int(request_count or 1))
    reserve = max(0.0, float(min_free_mem_gb or 0.0))
    buffer = max(0.0, float(free_mem_buffer_gb or 0.0))
    threshold = reserve + buffer if reserve > 0 else 0.0
    if not ids or cls not in GPU_RESOURCE_CLASSES or threshold <= 0:
        return {
            "blocked": False,
            "gpu_ids": ids,
            "safe_gpu_ids": ids,
            "min_free_mem_gb": reserve,
            "free_mem_buffer_gb": buffer,
        }
    if not isinstance(sample, dict) or sample.get("available") is False:
        return {
            "blocked": False,
            "gpu_ids": ids,
            "safe_gpu_ids": ids,
            "sample": sample if isinstance(sample, dict) else {},
            "min_free_mem_gb": reserve,
            "free_mem_buffer_gb": buffer,
        }

    wanted = set(ids)
    blocked: dict[str, dict[str, Any]] = {}
    seen_rows: set[str] = set()
    for row in _sample_rows(sample):
        gpu_id = _row_gpu_id(row)
        if not gpu_id or gpu_id not in wanted:
            continue
        seen_rows.add(gpu_id)
        free_gb = _row_free_mem_gb(row)
        if free_gb is None or free_gb >= threshold:
            continue
        blocked[gpu_id] = {
            "gpu_id": gpu_id,
            "free_mem_gb": free_gb,
            "utilization_gpu_pct": _float_or_none(row.get("utilization_gpu_pct")),
            "reasons": ["free_mem_below_admission_reserve"],
        }
    safe_ids = [gpu_id for gpu_id in ids if gpu_id not in blocked]
    blocked_request = len(safe_ids) < requested if mode == "lease" else bool(blocked)
    return {
        "blocked": blocked_request,
        "status": "PENDING" if blocked_request else "GRANTED",
        "reason": "gpu_memory_below_admission_reserve" if blocked_request else "",
        "gpu_ids": ids,
        "safe_gpu_ids": safe_ids,
        "blocked_gpu_ids": sorted(blocked),
        "blocked_gpus": blocked,
        "observed_gpu_ids": sorted(seen_rows),
        "sample": sample,
        "min_free_mem_gb": reserve,
        "free_mem_buffer_gb": buffer,
        "free_mem_threshold_gb": threshold,
        "request_count": requested,
        "assignment": mode,
    }


def build_admission_safety_deferred_result(
    *,
    resource_class: str,
    policy_resource_class: str,
    gpu_ids: list[str],
    safety: dict[str, Any],
    max_wait_sec: float,
    heartbeat_sec: float,
    slot_weight: float,
    capacity_slots: float,
    details: dict[str, Any],
    value_hint: dict[str, Any],
    priority: dict[str, float],
    queue_started_first: bool,
    pressure: dict[str, Any],
    eta_next_train_sec: float,
) -> dict[str, Any]:
    ids = _clean_gpu_ids(gpu_ids)
    blocked_ids = _clean_gpu_ids(safety.get("blocked_gpu_ids") or ids)
    eta = max(0.0, float(eta_next_train_sec or 0.0))
    reason = str(safety.get("reason") or "gpu_memory_below_admission_reserve")
    allowed_classes = ["pure_tt_cpu", "heavy_cpu_candidate", "readonly_cpu", "light_cpu"]
    priority_score = float(priority.get("admission_priority_score") or 0.0)
    return {
        "enabled": True,
        "acquired": False,
        "reason": reason,
        "status": "PENDING",
        "admission_action": "PENDING",
        "resource_mode": "YELLOW",
        "max_wait_sec": float(max_wait_sec),
        "heartbeat_sec": float(heartbeat_sec),
        "gpu_ids": ids,
        "resource_class": str(resource_class or ""),
        "policy_resource_class": str(policy_resource_class or ""),
        "slot_weight": float(slot_weight),
        "capacity_slots": float(capacity_slots),
        "details": details,
        "queue_started_first": bool(queue_started_first),
        "queue_position": 1,
        "queue_len": 0,
        "top_waiter_job_id": "",
        "value_hint": value_hint,
        "allowed_classes": allowed_classes,
        "pressure": pressure,
        "safety": safety,
        "eta_next_train_sec": eta,
        "eta_confidence": "low",
        "eta_source": "admission_safety",
        **priority,
        "feedback": resource_feedback_text(
            status="PENDING",
            reason=reason,
            scope="per_gpu",
            resource_mode="YELLOW",
            blocked_class=str(policy_resource_class or ""),
            gpu_ids=blocked_ids or ids,
            allowed_classes=allowed_classes,
            eta_next_train_sec=eta,
            eta_confidence="low",
            unlock_condition="resource_memory_available",
            blocked_until_unlock=True,
            extra_facts={
                "eta_source": "admission_safety",
                "admission_priority_score": f"{priority_score:.4f}",
            },
        ),
    }
