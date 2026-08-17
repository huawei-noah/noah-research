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

import re
from typing import Any

from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_LIGHT_CPU,
    RESOURCE_LIGHT_GPU_PROBE,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_READONLY_CPU,
    RESOURCE_UNKNOWN_EXEC,
    RESOURCE_UNKNOWN_GPU_EXEC,
)


GPU_CLASSES = {
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_LIGHT_GPU_PROBE,
    RESOURCE_UNKNOWN_GPU_EXEC,
}
CPU_CLASSES = {
    RESOURCE_READONLY_CPU,
    RESOURCE_LIGHT_CPU,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_UNKNOWN_EXEC,
}
_INTENT_RE = re.compile(
    r"(?:^|\s)(?:SCIENCEFLOW_RESOURCE_INTENT|_SCIENCEFLOW_RESOURCE_INTENT)="
    r"([A-Za-z0-9_.:-]+)"
)


def build_execution_facts(
    *,
    resource_snapshot: dict[str, Any] | None,
    progress_snapshot: dict[str, Any] | None,
    command: str = "",
    source_hint: Any = None,
    assigned_gpu_ids: list[str] | tuple[str, ...] | None = None,
    gpu_expected: bool | None = None,
) -> dict[str, Any]:
    """Build task-agnostic execution facts for arbiter/advisory decisions."""

    resource = resource_snapshot if isinstance(resource_snapshot, dict) else {}
    progress = progress_snapshot if isinstance(progress_snapshot, dict) else {}
    declared_class = str(
        resource.get("resource_class_declared")
        or resource.get("resource_class_observed")
        or ""
    )
    raw_intent = _declared_intent(command)
    declared_device = _declared_device(
        declared_class=declared_class,
        raw_intent=raw_intent,
        source_hint=source_hint,
        assigned_gpu_ids=assigned_gpu_ids,
    )
    gpu = _gpu_observation(resource, assigned_gpu_ids=assigned_gpu_ids)
    gpu_expected_value = bool(gpu["assigned_gpu_ids"]) if gpu_expected is None else bool(gpu_expected)
    cpu_pct = _process_cpu_pct(progress, resource)
    observed_device = _observed_device(gpu_active=gpu["active"], gpu_seen=gpu["sample_available"], cpu_pct=cpu_pct)

    runtime_sec = _float(progress.get("runtime_sec"), 0.0)
    metric_count = int(_float(progress.get("metric_history_line_count"), 0.0))
    near_submission = bool(progress.get("near_submission"))
    meaningful_stdout = bool(progress.get("meaningful_stdout"))
    stdout_age = _float(progress.get("stdout_last_line_age_sec", progress.get("stdout_age_sec")), runtime_sec)
    artifact_age = _float(progress.get("artifact_last_update_age_sec"), runtime_sec)
    artifact_updates = progress.get("artifact_updates") if isinstance(progress.get("artifact_updates"), list) else []
    has_recent_artifact = bool(artifact_updates) and artifact_age <= 300.0
    metric_age = _float(progress.get("metric_last_update_age_sec"), runtime_sec)
    has_recent_metric = bool(metric_count > 0 and metric_age <= 300.0)
    has_recent_metric_or_submission = bool(near_submission or has_recent_metric)
    metric_submission_or_stdout_signal = bool(has_recent_metric_or_submission or meaningful_stdout or near_submission)
    artifact_update_without_metric_or_stdout = bool(has_recent_artifact and not metric_submission_or_stdout_signal)
    artifact_updates_log_only = bool(artifact_updates) and all(_artifact_update_is_log_only(item) for item in artifact_updates)
    deliverable_validity = str(progress.get("deliverable_validity") or "none").strip().lower()
    has_recent_useful_artifact = bool(
        has_recent_artifact and any(_artifact_update_is_useful(item, deliverable_validity) for item in artifact_updates)
    )
    cpu_busy = cpu_pct >= 100.0
    intent_device_mismatch = bool(
        gpu_expected_value
        and declared_device == "gpu"
        and observed_device == "cpu"
        and gpu["assigned_idle"]
    )
    activity_without_metric_submission_or_stdout = bool(
        (cpu_busy or gpu["active"] or has_recent_artifact)
        and not metric_submission_or_stdout_signal
        and runtime_sec >= 300.0
    )
    source_payload = source_hint.to_event_payload() if hasattr(source_hint, "to_event_payload") else {}

    return {
        "declared_resource_class": declared_class,
        "declared_intent": raw_intent or declared_class or "unknown",
        "declared_device": declared_device,
        "observed_device": observed_device,
        "task_device_mode": "gpu_assigned" if gpu_expected_value else "cpu_only",
        "gpu_expected": gpu_expected_value,
        "intent_device_mismatch": intent_device_mismatch,
        "assigned_resource_idle": bool(gpu_expected_value and gpu["assigned_idle"]),
        "assigned_gpu_ids": list(gpu["assigned_gpu_ids"]),
        "assigned_gpu_max_util_pct": gpu["max_util_pct"],
        "assigned_gpu_max_mem_mb": gpu["max_mem_mb"],
        "process_cpu_pct": cpu_pct,
        "cpu_busy": cpu_busy,
        "runtime_sec": runtime_sec,
        "runtime_bucket": _runtime_bucket(runtime_sec),
        "stdout_age_sec": stdout_age,
        "has_recent_artifact": has_recent_artifact,
        "has_recent_useful_artifact": has_recent_useful_artifact,
        "deliverable_validity": deliverable_validity,
        "artifact_updates_log_only": artifact_updates_log_only,
        "has_recent_metric_or_submission": has_recent_metric_or_submission,
        "meaningful_stdout": meaningful_stdout,
        "artifact_update_without_metric_or_stdout": artifact_update_without_metric_or_stdout,
        "activity_without_metric_submission_or_stdout": activity_without_metric_submission_or_stdout,
        "near_submission": near_submission,
        "source_hint": source_payload,
    }


def _artifact_update_is_useful(item: Any, deliverable_validity: str) -> bool:
    if _artifact_update_is_log_only(item):
        return False
    if isinstance(item, dict):
        if bool(item.get("recoverable_artifact_on_disk") or item.get("run_state_confirmed") or item.get("done_marker")):
            return True
        path = str(item.get("path") or item.get("name") or "").strip().lower()
    else:
        path = str(item or "").strip().lower()
    name = path.rsplit("/", 1)[-1]
    if name in {"submission.csv", "predictions.csv"}:
        return deliverable_validity == "produced_valid"
    return bool(path)


def _artifact_update_is_log_only(item: Any) -> bool:
    if isinstance(item, dict):
        path = str(item.get("path") or item.get("name") or "").strip().lower()
        if bool(item.get("recoverable_artifact_on_disk") or item.get("run_state_confirmed") or item.get("done_marker")):
            return False
    else:
        path = str(item or "").strip().lower()
    if not path:
        return False
    name = path.rsplit("/", 1)[-1]
    if name in {"submission.csv", "predictions.csv"}:
        return False
    if name.endswith((".pt", ".pth", ".ckpt", ".pkl", ".joblib", ".npy", ".npz", ".parquet", ".feather")):
        return False
    if name.endswith(".log"):
        return True
    if name.endswith(".txt") and any(token in name for token in ("log", "stdout", "stderr", "trace")):
        return True
    return False


def _declared_intent(command: str) -> str:
    match = _INTENT_RE.search(str(command or ""))
    return str(match.group(1)).strip().lower() if match else ""


def _declared_device(
    *,
    declared_class: str,
    raw_intent: str,
    source_hint: Any,
    assigned_gpu_ids: list[str] | tuple[str, ...] | None,
) -> str:
    intent = str(raw_intent or "").lower()
    if intent.startswith("gpu") or intent in {"heavy_gpu", "light_train", "unknown_gpu"}:
        return "gpu"
    if intent.startswith("cpu") or intent in {"readonly", "light", "pure_tt_cpu"}:
        return "cpu"
    if declared_class in GPU_CLASSES:
        return "gpu"
    if declared_class in CPU_CLASSES:
        return "cpu"
    if bool(getattr(source_hint, "has_gpu_evidence", False)):
        return "gpu"
    if assigned_gpu_ids:
        return "gpu"
    return "unknown"


def _gpu_observation(resource: dict[str, Any], *, assigned_gpu_ids: list[str] | tuple[str, ...] | None) -> dict[str, Any]:
    rows = resource.get("resources") if isinstance(resource.get("resources"), list) else []
    assigned = {str(x) for x in (assigned_gpu_ids or resource.get("gpu_ids") or []) if str(x).strip()}
    lease = resource.get("lease") if isinstance(resource.get("lease"), dict) else {}
    for value in lease.get("resource_ids") or []:
        token = str(value).strip()
        if token:
            assigned.add(token)
    gpu_rows = []
    for row in rows:
        if not isinstance(row, dict) or str(row.get("resource_type") or "") != "gpu":
            continue
        gpu_id = str(row.get("id") or row.get("gpu_id") or row.get("index") or "").strip()
        if assigned and gpu_id and gpu_id not in assigned:
            continue
        gpu_rows.append(row)
    max_util = max((_float(row.get("utilization_gpu_pct"), 0.0) for row in gpu_rows), default=0.0)
    max_mem = max((_float(row.get("memory_used_mb"), 0.0) for row in gpu_rows), default=0.0)
    return {
        "sample_available": bool(gpu_rows),
        "assigned_gpu_ids": sorted(assigned),
        "max_util_pct": max_util,
        "max_mem_mb": max_mem,
        "assigned_idle": bool(gpu_rows) and max_util <= 1.0 and max_mem <= 1536.0,
        "active": bool(gpu_rows) and (max_util > 1.0 or max_mem > 1536.0),
    }


def _process_cpu_pct(progress: dict[str, Any], resource: dict[str, Any]) -> float:
    for parent in (progress, resource):
        cpu = parent.get("process_tree_cpu") if isinstance(parent.get("process_tree_cpu"), dict) else {}
        value = _float(cpu.get("total_cpu_pct"), -1.0)
        if value >= 0:
            return value
        value = _float(cpu.get("child_cpu_pct"), -1.0)
        if value >= 0:
            return value
    for row in resource.get("resources") if isinstance(resource.get("resources"), list) else []:
        if isinstance(row, dict) and str(row.get("resource_type") or "") == "cpu":
            cpu = row.get("process_tree_cpu") if isinstance(row.get("process_tree_cpu"), dict) else {}
            value = _float(cpu.get("total_cpu_pct"), -1.0)
            if value >= 0:
                return value
    return 0.0


def _observed_device(*, gpu_active: bool, gpu_seen: bool, cpu_pct: float) -> str:
    if gpu_active and cpu_pct >= 20.0:
        return "mixed"
    if gpu_active:
        return "gpu"
    if cpu_pct >= 20.0:
        return "cpu"
    if gpu_seen:
        return "idle_or_unknown"
    return "unknown"


def _runtime_bucket(runtime_sec: float) -> str:
    if runtime_sec < 900:
        return "lt_15m"
    if runtime_sec < 1800:
        return "15m_30m"
    if runtime_sec < 3600:
        return "30m_1h"
    if runtime_sec < 7200:
        return "1h_2h"
    return "gte_2h"


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default
