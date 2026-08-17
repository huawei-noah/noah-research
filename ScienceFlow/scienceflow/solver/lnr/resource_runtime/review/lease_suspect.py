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
class LeaseSuspectThresholds:
    warmup_sec: float = 300.0
    min_progress_windows: int = 2
    low_gpu_util_pct: float = 1.0
    min_gpu_mem_gb: float = 1.0
    route_viability_window_sec: float = 1800.0
    unknown_progress_grace_sec: float = 900.0
    unknown_progress_stale_output_sec: float = 600.0
    active_work_cpu_pct: float = 20.0


def classify_deliverable_validity(completion_state: dict[str, Any]) -> str:
    """Return none, produced_invalid, produced_valid, or produced_unknown."""

    state = completion_state if isinstance(completion_state, dict) else {}
    submission_state = state.get("submission_state") if isinstance(state.get("submission_state"), dict) else state
    validity = str(submission_state.get("deliverable_validity") or state.get("deliverable_validity") or "").strip().lower()
    if validity in {"none", "produced_invalid", "produced_valid", "produced_unknown"}:
        return validity
    if state.get("complete"):
        return "produced_valid" if str(state.get("mode") or "") == "submission" else "produced_unknown"
    if submission_state.get("submission_path") and not submission_state.get("complete"):
        return "produced_invalid" if str(submission_state.get("reason") or "").startswith("invalid_") else "produced_unknown"
    return "none"


def classify_route_viability(
    *,
    elapsed_sec: float,
    resource_class: str,
    progress_snapshot: dict[str, Any],
    deliverable_validity: str,
    thresholds: LeaseSuspectThresholds,
) -> dict[str, Any]:
    progress_signal = str(progress_snapshot.get("progress_signal") or "unknown").strip().lower()
    near_submission = bool(progress_snapshot.get("near_submission"))
    useful_artifact = _has_useful_artifact(progress_snapshot)
    metric_age = _float(progress_snapshot.get("metric_last_update_age_sec"), 1e9)
    stdout_lines = int(_float(progress_snapshot.get("stdout_lines"), 0.0))
    heavy_route = _looks_heavy(resource_class)
    positive_evidence = bool(near_submission or useful_artifact or metric_age < 600.0)
    if positive_evidence:
        state = "proven"
        reason = "progress_or_artifact_evidence"
    elif heavy_route and elapsed_sec >= thresholds.route_viability_window_sec:
        state = "unproven"
        reason = "heavy_route_without_viability_evidence"
    elif progress_signal in {"stalled", "degraded"} and stdout_lines <= 0:
        state = "unproven"
        reason = f"progress_signal={progress_signal}"
    else:
        state = "pending"
        reason = "within_viability_window_or_missing_evidence"
    return {
        "state": state,
        "route_viability_proven": state == "proven",
        "route_viability_unproven": state == "unproven",
        "reason": reason,
        "heavy_route": heavy_route,
        "positive_evidence": positive_evidence,
        "useful_artifact_evidence": useful_artifact,
        "route_viability_window_sec": thresholds.route_viability_window_sec,
    }


def classify_active_lease_suspect(
    *,
    elapsed_sec: float,
    has_waiter: bool,
    has_active_lease: bool,
    resource_class: str,
    progress_snapshot: dict[str, Any],
    gpu_memory: dict[str, Any],
    deliverable_validity: str,
    route_viability: dict[str, Any],
    thresholds: LeaseSuspectThresholds,
) -> dict[str, Any]:
    progress_signal = str(progress_snapshot.get("progress_signal") or "unknown").strip().lower()
    progress_windows = int(_float(progress_snapshot.get("progress_signal_windows"), 0.0))
    progress_confidence = str(progress_snapshot.get("progress_confidence") or "unknown").strip().lower()
    gpu_util_p90 = _float(gpu_memory.get("gpu_util_p90_pct"), 0.0)
    mem_current = _float(gpu_memory.get("gpu_mem_current_gb"), 0.0)
    mem_peak = _float(gpu_memory.get("gpu_mem_peak_gb"), mem_current)
    sample_available = bool(gpu_memory.get("sample_available"))
    progress_evidence_ok = bool(
        progress_signal in {"stalled", "degraded"}
        and (progress_windows >= thresholds.min_progress_windows or bool(progress_snapshot.get("multi_window_low_progress")))
        and progress_confidence in {"medium", "high"}
    )
    active_work = active_work_counterevidence(progress_snapshot, thresholds=thresholds)
    recent_useful_output = _has_recent_useful_output(
        progress_snapshot,
        deliverable_validity=deliverable_validity,
        thresholds=thresholds,
    )
    unknown_windows_ok = bool(
        progress_windows >= thresholds.min_progress_windows
        or bool(progress_snapshot.get("multi_window_low_progress"))
        or elapsed_sec >= thresholds.unknown_progress_grace_sec + thresholds.unknown_progress_stale_output_sec
    )
    resource_suspect = bool(
        has_waiter
        and has_active_lease
        and elapsed_sec >= thresholds.warmup_sec
        and sample_available
        and max(mem_current, mem_peak) >= thresholds.min_gpu_mem_gb
        and gpu_util_p90 <= thresholds.low_gpu_util_pct
        and progress_evidence_ok
    )
    unknown_progress_suspect = bool(
        has_waiter
        and has_active_lease
        and elapsed_sec >= thresholds.unknown_progress_grace_sec
        and sample_available
        and max(mem_current, mem_peak) >= thresholds.min_gpu_mem_gb
        and gpu_util_p90 <= thresholds.low_gpu_util_pct
        and progress_signal == "unknown"
        and unknown_windows_ok
        and not recent_useful_output
        and not bool(active_work.get("active"))
    )
    value_suspect = bool(
        has_waiter
        and has_active_lease
        and elapsed_sec >= thresholds.warmup_sec
        and bool(route_viability.get("route_viability_unproven"))
    )
    if resource_suspect:
        stop_review_priority = "high"
        action_bias = "review_stop_or_release"
    elif unknown_progress_suspect:
        stop_review_priority = "medium"
        action_bias = "observe_collect_active_work_evidence"
    elif value_suspect:
        stop_review_priority = "low"
        action_bias = "lower_priority_or_extend_proof_window"
    else:
        stop_review_priority = "none"
        action_bias = "continue_observe"
    return {
        "active_lease_suspect": bool(resource_suspect or unknown_progress_suspect or value_suspect),
        "resource_suspect": resource_suspect,
        "unknown_progress_suspect": unknown_progress_suspect,
        "value_suspect": value_suspect,
        "stop_review_priority": stop_review_priority,
        "action_bias": action_bias,
        "progress_evidence_ok": progress_evidence_ok,
        "recent_useful_output": recent_useful_output,
        "active_work_counterevidence": dict(active_work),
        "unknown_windows_ok": unknown_windows_ok,
        "evidence": {
            "has_waiter": bool(has_waiter),
            "has_active_lease": bool(has_active_lease),
            "elapsed_sec": float(elapsed_sec or 0.0),
            "warmup_sec": float(thresholds.warmup_sec or 0.0),
            "resource_class": str(resource_class or ""),
            "progress_signal": progress_signal,
            "progress_signal_windows": progress_windows,
            "progress_confidence": progress_confidence,
            "gpu_util_p90_pct": gpu_util_p90,
            "gpu_mem_current_gb": mem_current,
            "gpu_mem_peak_gb": mem_peak,
            "route_viability": dict(route_viability or {}),
            "unknown_progress_grace_sec": float(thresholds.unknown_progress_grace_sec or 0.0),
            "unknown_progress_stale_output_sec": float(thresholds.unknown_progress_stale_output_sec or 0.0),
            "active_work_counterevidence": dict(active_work),
            "recent_useful_output": recent_useful_output,
        },
    }


def active_work_counterevidence(progress_snapshot: dict[str, Any], *, thresholds: LeaseSuspectThresholds) -> dict[str, Any]:
    progress = progress_snapshot if isinstance(progress_snapshot, dict) else {}
    reasons: list[str] = []
    cpu = progress.get("process_tree_cpu") if isinstance(progress.get("process_tree_cpu"), dict) else {}
    if cpu and cpu.get("available", True):
        total_cpu = _float(cpu.get("total_cpu_pct"), 0.0)
        child_cpu = _float(cpu.get("child_cpu_pct"), 0.0)
        busy_children = int(_float(cpu.get("busy_child_count"), 0.0))
        if max(total_cpu, child_cpu) >= thresholds.active_work_cpu_pct:
            reasons.append("process_tree_cpu_active")
        if busy_children > 0:
            reasons.append("busy_child_processes")
        for key in ("io_write_bytes_delta", "io_read_bytes_delta", "write_bytes_delta", "read_bytes_delta"):
            if _float(cpu.get(key), 0.0) > 0.0:
                reasons.append(f"{key}_active")
                break
    stage = str(progress.get("known_stage") or progress.get("current_phase") or "").strip().lower()
    if any(token in stage for token in ("preprocess", "cache", "checkpoint", "write", "saving", "eval", "inference", "predict")):
        reasons.append("known_active_phase")
    for artifact in progress.get("artifact_updates") or []:
        if not isinstance(artifact, dict):
            continue
        if bool(artifact.get("active_write") or artifact.get("writing")):
            reasons.append("artifact_active_write")
            break
        for key in ("size_delta_bytes", "size_bytes_delta", "delta_bytes"):
            if _float(artifact.get(key), 0.0) > 0.0:
                reasons.append("artifact_size_growth")
                break
        if reasons and reasons[-1] == "artifact_size_growth":
            break
    return {"active": bool(reasons), "reasons": sorted(set(reasons))}


def _has_recent_useful_output(
    progress_snapshot: dict[str, Any],
    *,
    deliverable_validity: str,
    thresholds: LeaseSuspectThresholds,
) -> bool:
    progress = progress_snapshot if isinstance(progress_snapshot, dict) else {}
    if bool(progress.get("near_submission")):
        return True
    stale_sec = max(1.0, float(thresholds.unknown_progress_stale_output_sec or 1.0))
    if _float(progress.get("metric_last_update_age_sec"), 1e9) < stale_sec:
        return True
    if bool(progress.get("meaningful_stdout")) and _float(progress.get("stdout_last_line_age_sec"), 1e9) < stale_sec:
        return True
    if _has_useful_artifact(progress) and _float(progress.get("artifact_last_update_age_sec"), 1e9) < stale_sec:
        return True
    return False


def _has_useful_artifact(progress_snapshot: dict[str, Any]) -> bool:
    progress = progress_snapshot if isinstance(progress_snapshot, dict) else {}
    for artifact in progress.get("artifact_updates") or []:
        if not isinstance(artifact, dict):
            continue
        if bool(artifact.get("useful") or artifact.get("route_viability")):
            return True
        text = " ".join(str(artifact.get(key) or "") for key in ("kind", "type", "path", "name")).lower()
        if any(token in text for token in ("checkpoint", "ckpt", "model", "logit", "feature", "embedding", "blend", "submission")):
            return True
        if any(text.endswith(suffix) for suffix in (".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl")):
            return True
    return False


def _looks_heavy(resource_class: str) -> bool:
    return "heavy" in str(resource_class or "").lower() or "train" in str(resource_class or "").lower()


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default
