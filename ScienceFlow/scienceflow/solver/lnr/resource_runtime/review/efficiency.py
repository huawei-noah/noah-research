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

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResourceEfficiencyConfig:
    warmup_sec: float = 900.0
    min_assigned_cpu_count: int = 4
    max_cpu_capacity_utilization: float = 0.25
    min_active_cpu_pct: float = 20.0
    min_long_eta_sec: float = 900.0
    long_eta_budget_fraction: float = 0.20
    min_mismatch_windows: int = 2
    max_review_llm_calls: int = 2
    max_completion_grace_sec: float = 300.0
    min_single_sample_eta_runtime_sec: float = 1800.0
    unknown_eta_review_runtime_sec: float = 1800.0


def cpu_set_count(cpu_set: str | None) -> int:
    """Count unique CPU IDs from taskset-compatible comma/range syntax."""

    values: set[int] = set()
    for token in str(cpu_set or "").split(","):
        part = token.strip()
        if not part:
            continue
        if "-" not in part:
            value = _int_or_none(part)
            if value is not None and value >= 0:
                values.add(value)
            continue
        start_raw, end_raw = part.split("-", 1)
        start = _int_or_none(start_raw)
        end = _int_or_none(end_raw)
        if start is None or end is None or start < 0 or end < start:
            continue
        values.update(range(start, end + 1))
    return len(values)


def assess_resource_efficiency(
    *,
    previous_state: dict[str, Any] | None,
    cpu_set: str,
    process_tree_cpu: dict[str, Any] | None,
    assigned_gpu_count: int,
    gpu_expected: bool,
    gpu_active: bool,
    gpu_sample_available: bool,
    runtime_sec: float,
    phase: str,
    eta_to_deliverable_sec: float | None,
    eta_confidence: str,
    remaining_useful_budget_sec: float | None,
    phase_completion_protected: bool,
    metric_useful: bool | None,
    deliverable_validity: str,
    progress_source: str,
    progress_evidence_trust: str,
    progress_interval_samples: int,
    llm_call_count: int,
    config: ResourceEfficiencyConfig | None = None,
) -> dict[str, Any]:
    """Build a bounded review-only signal for inefficient long-running work."""

    cfg = config or ResourceEfficiencyConfig()
    previous = dict(previous_state or {})
    normalized_phase = _normalize_phase(phase)
    previous_scope = str(previous.get("scope_key") or "")
    same_scope = bool(previous_scope and previous_scope == normalized_phase)
    assigned_cpus = cpu_set_count(cpu_set)
    process_cpu_pct = _process_cpu_pct(process_tree_cpu)
    cpu_capacity = (
        process_cpu_pct / (100.0 * assigned_cpus)
        if assigned_cpus > 0 and process_cpu_pct is not None
        else None
    )
    cpu_underutilized = bool(
        assigned_cpus >= cfg.min_assigned_cpu_count
        and process_cpu_pct is not None
        and process_cpu_pct >= cfg.min_active_cpu_pct
        and cpu_capacity is not None
        and cpu_capacity <= cfg.max_cpu_capacity_utilization
    )
    gpu_idle_mismatch = bool(
        assigned_gpu_count > 0
        and gpu_expected
        and gpu_sample_available
        and not gpu_active
        and process_cpu_pct is not None
        and process_cpu_pct >= cfg.min_active_cpu_pct
    )
    cpu_underutilized_mismatch = bool(
        cpu_underutilized
        and not (
            gpu_expected
            and assigned_gpu_count > 0
            and (gpu_active or not gpu_sample_available)
        )
    )
    resource_mismatch = bool(cpu_underutilized_mismatch or gpu_idle_mismatch)

    eta = _positive_float(eta_to_deliverable_sec)
    remaining = _positive_float(remaining_useful_budget_sec)
    eta_threshold = max(
        cfg.min_long_eta_sec,
        (remaining or 0.0) * cfg.long_eta_budget_fraction,
    )
    progress_samples = max(0, int(progress_interval_samples or 0))
    eta_confidence_normalized = str(eta_confidence or "").strip().lower()
    eta_reliable = bool(
        eta is not None
        and eta_confidence_normalized in {"medium", "high"}
        and (
            progress_samples >= 1
            or float(runtime_sec or 0.0) >= cfg.min_single_sample_eta_runtime_sec
        )
    )
    long_eta = bool(eta_reliable and eta is not None and eta >= eta_threshold)
    unknown_eta_persistent = bool(
        not eta_reliable
        and float(runtime_sec or 0.0) >= cfg.unknown_eta_review_runtime_sec
    )
    valid_deliverable = str(deliverable_validity or "").strip().lower() == "produced_valid"
    insufficient_value_progress = bool(metric_useful is not True and not valid_deliverable)
    warmup_complete = float(runtime_sec or 0.0) >= cfg.warmup_sec
    previous_grace_start = (
        _float_or_none(previous.get("completion_grace_started_runtime_sec"))
        if same_scope
        else None
    )
    completion_grace_start = previous_grace_start
    if phase_completion_protected and completion_grace_start is None:
        completion_grace_start = max(0.0, float(runtime_sec or 0.0))
    completion_grace_elapsed = (
        max(0.0, float(runtime_sec or 0.0) - completion_grace_start)
        if completion_grace_start is not None
        else 0.0
    )
    completion_grace_expired = bool(
        phase_completion_protected
        and completion_grace_elapsed >= cfg.max_completion_grace_sec
    )
    completion_protection_effective = bool(
        phase_completion_protected and not completion_grace_expired
    )
    candidate = bool(
        warmup_complete
        and resource_mismatch
        and (long_eta or unknown_eta_persistent)
        and insufficient_value_progress
        and not completion_protection_effective
    )

    previous_windows = int(previous.get("mismatch_windows") or 0) if same_scope else 0
    mismatch_windows = previous_windows + 1 if candidate else 0
    review_started = bool(previous.get("efficiency_review_started"))
    if (
        candidate
        and mismatch_windows >= cfg.min_mismatch_windows
        and not review_started
    ):
        review_started = True
        baseline_calls = max(0, int(llm_call_count or 0))
    elif review_started:
        baseline_calls = int(previous.get("baseline_llm_call_count") or 0)
    else:
        baseline_calls = max(0, int(llm_call_count or 0))
    review_calls = (
        max(0, int(llm_call_count or 0) - baseline_calls)
        if review_started
        else 0
    )
    review_limit_reached = bool(
        review_started and review_calls >= cfg.max_review_llm_calls
    )
    review_required = bool(
        candidate
        and mismatch_windows >= cfg.min_mismatch_windows
        and not review_limit_reached
    )

    reasons: list[str] = []
    if cpu_underutilized_mismatch:
        reasons.append("cpu_capacity_underutilized")
    if gpu_idle_mismatch:
        reasons.append("assigned_gpu_idle_for_gpu_intent")
    if long_eta:
        reasons.append("long_eta")
    if unknown_eta_persistent:
        reasons.append("eta_unavailable_after_long_runtime")
    if insufficient_value_progress:
        reasons.append("no_verified_value_progress")
    if completion_protection_effective:
        reasons.append("phase_completion_protected")
    if completion_grace_expired:
        reasons.append("phase_completion_grace_expired")
    if review_limit_reached:
        reasons.append("efficiency_review_call_limit_reached")

    if not warmup_complete:
        status = "warmup"
    elif candidate and unknown_eta_persistent:
        status = "inefficient_unknown_eta"
    elif not eta_reliable:
        status = "unknown_eta"
    elif candidate:
        status = "inefficient"
    else:
        status = "healthy"
    return {
        "status": status,
        "scope_key": normalized_phase,
        "assigned_cpu_count": assigned_cpus,
        "process_cpu_pct": process_cpu_pct,
        "cpu_capacity_utilization": _rounded(cpu_capacity),
        "cpu_underutilized": cpu_underutilized,
        "cpu_underutilized_mismatch": cpu_underutilized_mismatch,
        "assigned_gpu_count": max(0, int(assigned_gpu_count or 0)),
        "gpu_expected": bool(gpu_expected),
        "gpu_active": bool(gpu_active),
        "gpu_sample_available": bool(gpu_sample_available),
        "gpu_idle_mismatch": gpu_idle_mismatch,
        "resource_mismatch": resource_mismatch,
        "eta_to_deliverable_sec": eta,
        "eta_threshold_sec": eta_threshold,
        "eta_confidence": str(eta_confidence or "low"),
        "eta_reliable": eta_reliable,
        "long_eta": long_eta,
        "unknown_eta_persistent": unknown_eta_persistent,
        "metric_useful": metric_useful if isinstance(metric_useful, bool) else None,
        "deliverable_validity": str(deliverable_validity or "none"),
        "insufficient_value_progress": insufficient_value_progress,
        "phase_completion_protected": completion_protection_effective,
        "phase_completion_protection_reported": bool(phase_completion_protected),
        "completion_grace_started_runtime_sec": completion_grace_start,
        "completion_grace_elapsed_sec": completion_grace_elapsed,
        "completion_grace_expired": completion_grace_expired,
        "max_completion_grace_sec": cfg.max_completion_grace_sec,
        "mismatch_windows": mismatch_windows,
        "minimum_mismatch_windows": cfg.min_mismatch_windows,
        "efficiency_review_started": review_started,
        "baseline_llm_call_count": baseline_calls,
        "efficiency_review_llm_calls": review_calls,
        "max_efficiency_review_llm_calls": cfg.max_review_llm_calls,
        "review_limit_reached": review_limit_reached,
        "review_required": review_required,
        "review_generation": (
            min(review_calls + 1, cfg.max_review_llm_calls)
            if review_started
            else 0
        ),
        "kill_authority": "review_only",
        "progress_source": str(progress_source or "unknown"),
        "progress_evidence_trust": str(progress_evidence_trust or "unknown"),
        "progress_interval_samples": progress_samples,
        "reason_codes": reasons,
    }


def _process_cpu_pct(snapshot: dict[str, Any] | None) -> float | None:
    data = snapshot if isinstance(snapshot, dict) else {}
    if data.get("available") is False:
        return None
    for key in ("total_cpu_pct", "child_cpu_pct"):
        value = _float_or_none(data.get(key))
        if value is not None and value >= 0.0:
            return value
    return None


def _normalize_phase(value: str) -> str:
    text = str(value or "unknown").strip().lower()
    return text[:120] or "unknown"


def _positive_float(value: Any) -> float | None:
    parsed = _float_or_none(value)
    return parsed if parsed is not None and parsed > 0.0 else None


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result == result and result not in {float("inf"), float("-inf")} else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
