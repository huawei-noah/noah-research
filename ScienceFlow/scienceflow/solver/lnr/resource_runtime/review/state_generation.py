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

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResourceStateGeneration:
    fingerprint: str
    control_generation_key: str
    feedback_generation_key: str
    control_fields: dict[str, Any]
    feedback_fields: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "control_generation_key": self.control_generation_key,
            "feedback_generation_key": self.feedback_generation_key,
            "control_fields": dict(self.control_fields),
            "feedback_fields": dict(self.feedback_fields),
        }


def build_resource_state_generation(snapshot: dict[str, Any] | None) -> ResourceStateGeneration:
    data = snapshot if isinstance(snapshot, dict) else {}
    progress = data.get("progress_snapshot") if isinstance(data.get("progress_snapshot"), dict) else {}
    resource = data.get("resource_snapshot") if isinstance(data.get("resource_snapshot"), dict) else {}
    execution = data.get("execution_facts") if isinstance(data.get("execution_facts"), dict) else {}
    blocker = data.get("blocker") if isinstance(data.get("blocker"), dict) else {}
    liveness = _first_dict(
        progress.get("process_liveness"),
        resource.get("process_liveness"),
        blocker.get("process_liveness"),
    )
    lease = resource.get("lease") if isinstance(resource.get("lease"), dict) else {}
    runtime_sec = _float(progress.get("runtime_sec"), _float(blocker.get("runtime_sec"), 0.0))
    metric_history_text = str(progress.get("metric_history_text") or "").strip()
    efficiency = progress.get("resource_efficiency") if isinstance(progress.get("resource_efficiency"), dict) else {}
    control_fields = {
        "proposal_type": str(data.get("proposal_type") or ""),
        "reason_code": str(data.get("reason_code") or ""),
        "process_liveness": str(liveness.get("status") or "unknown"),
        "gpu_lease_state": _lease_state(lease),
        "gpu_access_state": _gpu_access_state(resource),
        "blocked_until_unlock": bool(data.get("blocked_until_unlock") or progress.get("blocked_until_unlock")),
        "unlock_condition": str(data.get("unlock_condition") or progress.get("unlock_condition") or ""),
        "admission_action": str(data.get("admission_action") or ""),
        "resource_class_blocked": str(data.get("resource_class_blocked") or data.get("blocked_class") or ""),
        "assigned_or_blocked_gpu_set": _gpu_set(data, resource, lease),
        "progress_signal": str(progress.get("progress_signal") or blocker.get("progress_signal") or "unknown"),
        "progress_confidence": str(progress.get("progress_confidence") or blocker.get("progress_confidence") or "unknown"),
        "stdout_state": _age_state(progress.get("stdout_last_line_age_sec", progress.get("stdout_age_sec")), stale_sec=900.0),
        "artifact_state": _age_state(progress.get("artifact_last_update_age_sec"), stale_sec=900.0),
        "metric_state": _age_state(progress.get("metric_last_update_age_sec"), stale_sec=900.0),
        "metric_history_digest": _text_digest(metric_history_text),
        "runtime_bucket": _runtime_bucket(runtime_sec),
        "runtime_tick": int(runtime_sec // 1800.0) if runtime_sec > 0 else 0,
        "gpu_util_bucket": _gpu_util_bucket(resource),
        "declared_device": str(execution.get("declared_device") or "unknown"),
        "observed_device": str(execution.get("observed_device") or "unknown"),
        "intent_device_mismatch": bool(execution.get("intent_device_mismatch")),
        "assigned_resource_idle": bool(execution.get("assigned_resource_idle")),
        "artifact_update_without_metric_or_stdout": bool(execution.get("artifact_update_without_metric_or_stdout")),
        "activity_without_metric_submission_or_stdout": bool(execution.get("activity_without_metric_submission_or_stdout")),
        "resource_efficiency_status": str(efficiency.get("status") or "unknown"),
        "resource_efficiency_review_required": bool(efficiency.get("review_required")),
        "resource_efficiency_review_generation": int(efficiency.get("review_generation") or 0),
        "resource_efficiency_review_limit_reached": bool(efficiency.get("review_limit_reached")),
    }
    feedback_fields = {
        key: control_fields[key]
        for key in (
            "process_liveness",
            "gpu_lease_state",
            "gpu_access_state",
            "blocked_until_unlock",
            "unlock_condition",
            "admission_action",
            "resource_class_blocked",
            "assigned_or_blocked_gpu_set",
            "declared_device",
            "observed_device",
            "intent_device_mismatch",
            "assigned_resource_idle",
        )
    }
    control_key = _digest(control_fields)
    feedback_key = _digest(feedback_fields)
    fingerprint = _digest({"control": control_key, "feedback": feedback_key})
    return ResourceStateGeneration(
        fingerprint=fingerprint,
        control_generation_key=control_key,
        feedback_generation_key=feedback_key,
        control_fields=control_fields,
        feedback_fields=feedback_fields,
    )


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _lease_state(lease: dict[str, Any]) -> str:
    if not lease:
        return "unknown"
    if lease.get("active"):
        return "active"
    if lease.get("released_idle"):
        return "released_idle"
    return "inactive"


def _gpu_access_state(resource: dict[str, Any]) -> str:
    rows = resource.get("resources") if isinstance(resource.get("resources"), list) else []
    gpu_rows = [row for row in rows if isinstance(row, dict) and str(row.get("resource_type") or "") == "gpu"]
    if not gpu_rows:
        return "no_gpu_sample"
    util = max((_float(row.get("utilization_gpu_pct"), 0.0) for row in gpu_rows), default=0.0)
    mem = max((_float(row.get("memory_used_mb"), 0.0) for row in gpu_rows), default=0.0)
    if util <= 1.0 and mem <= 1536.0:
        return "idle_small_mem"
    if util <= 1.0:
        return "idle_mem_resident"
    return "active_gpu"


def _gpu_set(data: dict[str, Any], resource: dict[str, Any], lease: dict[str, Any]) -> str:
    values = data.get("gpu_ids") or resource.get("gpu_ids") or lease.get("resource_ids") or []
    ids = sorted({str(x) for x in values if str(x).strip()})
    return ",".join(ids)


def _runtime_bucket(runtime_sec: float) -> str:
    if runtime_sec < 900:
        return "lt_15m"
    if runtime_sec < 1800:
        return "15m_30m"
    if runtime_sec < 3600:
        return "30m_1h"
    if runtime_sec < 7200:
        return "1h_2h"
    if runtime_sec < 14400:
        return "2h_4h"
    if runtime_sec < 28800:
        return "4h_8h"
    if runtime_sec < 43200:
        return "8h_12h"
    return "gte_12h"


def _age_state(value: Any, *, stale_sec: float) -> str:
    age = _float(value, -1.0)
    if age < 0:
        return "unknown"
    return "stale" if age >= stale_sec else "recent"


def _gpu_util_bucket(resource: dict[str, Any]) -> str:
    rows = resource.get("resources") if isinstance(resource.get("resources"), list) else []
    util = max(
        (_float(row.get("utilization_gpu_pct"), 0.0) for row in rows if isinstance(row, dict) and str(row.get("resource_type") or "") == "gpu"),
        default=0.0,
    )
    if util <= 1.0:
        return "idle"
    if util <= 35.0:
        return "low"
    return "active"


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def _digest(value: dict[str, Any]) -> str:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _text_digest(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]
