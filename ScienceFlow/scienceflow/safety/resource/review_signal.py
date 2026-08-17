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
from dataclasses import dataclass, field
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bucket_cpu(cpu_pct: float, busy_children: int = 0) -> str:
    if cpu_pct >= 200.0 or busy_children >= 2:
        return "heavy"
    if cpu_pct >= 50.0 or busy_children >= 1:
        return "active"
    if cpu_pct >= 5.0:
        return "light"
    return "idle"


def _bucket_gpu(active: bool, unknown: bool = False) -> str:
    if unknown:
        return "unknown"
    return "active" if active else "idle"


def _bucket_artifact(recent: bool, grew: bool = False) -> str:
    if grew:
        return "growing"
    if recent:
        return "fresh"
    return "none"


def _bucket_phase(phase: str) -> str:
    text = str(phase or "").strip().lower()
    if any(token in text for token in ("predict", "infer", "submission")):
        return "prediction"
    if any(token in text for token in ("eval", "valid", "score", "test")):
        return "evaluation"
    if any(token in text for token in ("train", "epoch", "fit")):
        return "training"
    if any(token in text for token in ("setup", "load", "cache", "prep", "feature")):
        return "setup"
    return "unknown"


def stable_signature(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass(frozen=True)
class ResourceReviewSignal:
    elapsed_sec: float = 0.0
    active_work: bool = False
    useful_progress: bool = False
    stdout_changed: bool = False
    metric_changed: bool = False
    raw_metric_changed: bool = False
    metric_value_status: str = ""
    metric_value_delta_to_best: float | None = None
    metric_scope_key: str = ""
    structured_progress_advanced: bool = False
    stdout_bytes: int = 0
    metric_history_text: str = ""
    active_work_signature: str = ""
    active_channels: tuple[str, ...] = field(default_factory=tuple)
    cpu_bucket: str = "idle"
    gpu_bucket: str = "idle"
    artifact_bucket: str = "none"
    phase_bucket: str = "unknown"
    active_waiter_pressure: bool = False
    blocked_worker_count: int = 0
    terminal_signal_events: int = 0
    terminal_signal_advanced: bool = False
    proof_reset_progress: bool = False
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "elapsed_sec": float(self.elapsed_sec),
            "active_work": bool(self.active_work),
            "useful_progress": bool(self.useful_progress),
            "stdout_changed": bool(self.stdout_changed),
            "metric_changed": bool(self.metric_changed),
            "raw_metric_changed": bool(self.raw_metric_changed),
            "metric_value_status": self.metric_value_status,
            "metric_value_delta_to_best": self.metric_value_delta_to_best,
            "metric_scope_key": self.metric_scope_key,
            "structured_progress_advanced": bool(self.structured_progress_advanced),
            "stdout_bytes": int(self.stdout_bytes),
            "metric_history_text": self.metric_history_text,
            "active_work_signature": self.active_work_signature,
            "active_channels": list(self.active_channels),
            "cpu_bucket": self.cpu_bucket,
            "gpu_bucket": self.gpu_bucket,
            "artifact_bucket": self.artifact_bucket,
            "phase_bucket": self.phase_bucket,
            "active_waiter_pressure": bool(self.active_waiter_pressure),
            "blocked_worker_count": int(self.blocked_worker_count),
            "terminal_signal_events": int(self.terminal_signal_events),
            "terminal_signal_advanced": bool(self.terminal_signal_advanced),
            "proof_reset_progress": bool(self.proof_reset_progress),
            "reason": self.reason,
        }


def build_review_signal(
    *,
    elapsed_sec: float = 0.0,
    stdout_lines: int = 0,
    stdout_bytes: int = 0,
    previous_stdout_bytes: int = 0,
    metric_history_text: str = "",
    previous_metric_history_text: str = "",
    saw_training_progress: bool = False,
    saw_final_score: bool = False,
    terminal_signal_events: int = 0,
    previous_terminal_signal_events: int = 0,
    current_phase: str = "",
    process_tree_cpu: dict[str, Any] | None = None,
    gpu_active: bool = False,
    gpu_unknown: bool = False,
    artifact_recent: bool = False,
    artifact_grew: bool = False,
    structured_progress_advanced: bool = False,
    metric_value_useful: bool | None = None,
    metric_value_status: str = "",
    metric_value_delta_to_best: float | None = None,
    metric_scope_key: str = "",
    active_waiter_pressure: bool = False,
    blocked_worker_count: int = 0,
) -> ResourceReviewSignal:
    cpu = process_tree_cpu if isinstance(process_tree_cpu, dict) else {}
    cpu_pct = _float(cpu.get("total_cpu_pct"), 0.0)
    busy_children = _int(cpu.get("busy_child_count"), 0)
    stdout_byte_count = max(0, _int(stdout_bytes, 0))
    previous_stdout = max(0, _int(previous_stdout_bytes, 0))
    stdout_changed = stdout_byte_count > previous_stdout
    metric_text = str(metric_history_text or "").strip()
    previous_metric = str(previous_metric_history_text or "").strip()
    raw_metric_changed = bool(metric_text and metric_text != previous_metric)
    metric_changed = raw_metric_changed if metric_value_useful is not False else False
    terminal_events = max(0, _int(terminal_signal_events, 0))
    previous_terminal_events = max(0, _int(previous_terminal_signal_events, 0))
    terminal_signal_advanced = terminal_events > previous_terminal_events

    cpu_bucket = _bucket_cpu(cpu_pct, busy_children)
    gpu_bucket = _bucket_gpu(bool(gpu_active), bool(gpu_unknown))
    artifact_bucket = _bucket_artifact(bool(artifact_recent), bool(artifact_grew))
    phase_bucket = _bucket_phase(current_phase)

    channels: list[str] = []
    if stdout_changed or int(stdout_lines or 0) > 0 and stdout_byte_count > 0:
        channels.append("stdout")
    if cpu_bucket in {"light", "active", "heavy"}:
        channels.append("cpu")
    if gpu_bucket == "active":
        channels.append("gpu")
    if artifact_bucket in {"fresh", "growing"}:
        channels.append("artifact")
    if saw_training_progress or saw_final_score or terminal_signal_advanced:
        channels.append("process")
    if structured_progress_advanced:
        channels.append("structured_progress")

    active_channels = tuple(sorted(set(channels)))
    useful_structured_progress = bool(structured_progress_advanced and metric_value_useful is True)
    useful_progress = bool(
        metric_changed
        or saw_final_score
        or terminal_signal_advanced
        or artifact_grew
        or useful_structured_progress
    )
    active_work = bool(active_channels or useful_progress)
    signature_payload = {
        "active_channels": list(active_channels),
        "cpu_bucket": cpu_bucket,
        "gpu_bucket": gpu_bucket,
        "artifact_bucket": artifact_bucket,
        "phase_bucket": phase_bucket,
    }
    reason = "+".join(active_channels) if active_channels else "inactive"
    if useful_progress and "progress" not in reason:
        reason = f"{reason}+progress" if reason else "progress"
    return ResourceReviewSignal(
        elapsed_sec=max(0.0, _float(elapsed_sec, 0.0)),
        active_work=active_work,
        useful_progress=useful_progress,
        stdout_changed=stdout_changed,
        metric_changed=metric_changed,
        raw_metric_changed=raw_metric_changed,
        metric_value_status=str(metric_value_status or ""),
        metric_value_delta_to_best=metric_value_delta_to_best,
        metric_scope_key=str(metric_scope_key or ""),
        structured_progress_advanced=bool(structured_progress_advanced),
        stdout_bytes=stdout_byte_count,
        metric_history_text=metric_text,
        active_work_signature=stable_signature(signature_payload),
        active_channels=active_channels,
        cpu_bucket=cpu_bucket,
        gpu_bucket=gpu_bucket,
        artifact_bucket=artifact_bucket,
        phase_bucket=phase_bucket,
        active_waiter_pressure=bool(active_waiter_pressure),
        blocked_worker_count=max(0, _int(blocked_worker_count, 0)),
        terminal_signal_events=terminal_events,
        terminal_signal_advanced=terminal_signal_advanced,
        proof_reset_progress=bool(saw_final_score or terminal_signal_advanced),
        reason=reason,
    )
