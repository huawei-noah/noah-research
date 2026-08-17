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

LOW_PROGRESS_SIGNALS = {"stalled", "degraded"}


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


def _recent(age_sec: float, limit_sec: float) -> bool:
    return limit_sec <= 0.0 or age_sec < limit_sec


def _evidence_windows(age_sec: float, limit_sec: float, *, fallback_window_sec: float = 300.0) -> int:
    if age_sec <= 0.0:
        return 0
    window = limit_sec if limit_sec > 0.0 else fallback_window_sec
    if window <= 0.0:
        window = fallback_window_sec
    return max(1, int(age_sec // max(1.0, window)))


def classify_progress_signal(
    signal: dict[str, Any],
    *,
    progress_age_sec: float,
    artifact_age_sec: float,
    low_progress_warmup_sec: float,
    stalled_stdout_sec: float,
    no_progress_sec: float,
    no_artifact_sec: float,
    idle_samples: int = 0,
    dataloader_samples: int = 0,
    previous_signal: str = "",
    previous_windows: int = 0,
    min_confidence_windows: int = 2,
) -> dict[str, Any]:
    """Summarize external command progress into a deterministic arbiter anchor.

    The monitor owns this classification. It should not decide whether a route is
    scientifically worth continuing; it only states whether recent stdout,
    progress heartbeats, artifact updates, and resource samples show external
    progress.
    """

    elapsed = max(0.0, _float(signal.get("elapsed_sec")))
    stdout_age = max(0.0, _float(signal.get("stdout_age_sec"), elapsed))
    stdout_lines = max(0, _int(signal.get("stdout_lines")))
    near_submission = bool(signal.get("saw_final_score")) or _int(signal.get("terminal_signal_events")) > 0

    structured_progress = signal.get("structured_progress") if isinstance(signal.get("structured_progress"), dict) else {}
    structured_progress_present = bool(structured_progress)
    structured_progress_advanced = bool(signal.get("structured_progress_advanced") or structured_progress.get("advanced"))

    stdout_recent = stdout_lines > 0 and _recent(stdout_age, stalled_stdout_sec)
    progress_recent = bool(signal.get("saw_training_progress")) and _recent(progress_age_sec, no_progress_sec)
    if structured_progress_present and not structured_progress_advanced:
        progress_recent = False
    artifact_recent = _recent(artifact_age_sec, no_artifact_sec) and artifact_age_sec < max(elapsed, 1.0)
    warm = elapsed >= max(0.0, low_progress_warmup_sec)

    reasons: list[str] = []
    if near_submission:
        progress_signal = "active"
        reasons.append("near_submission_or_terminal_signal")
    elif progress_recent or artifact_recent:
        progress_signal = "active"
        if progress_recent:
            reasons.append("recent_progress_heartbeat")
        if artifact_recent:
            reasons.append("recent_artifact_update")
    else:
        stdout_stalled = bool(stalled_stdout_sec > 0.0 and stdout_age >= stalled_stdout_sec)
        progress_stalled = bool(no_progress_sec > 0.0 and progress_age_sec >= no_progress_sec)
        artifact_stalled = bool(no_artifact_sec <= 0.0 or artifact_age_sec >= no_artifact_sec)
        if warm and (stdout_stalled or (progress_stalled and artifact_stalled)):
            progress_signal = "stalled"
            if stdout_stalled:
                reasons.append("stdout_stalled")
            if progress_stalled:
                reasons.append("progress_heartbeat_stalled")
            if artifact_stalled:
                reasons.append("artifact_stalled")
        elif warm and (idle_samples > 0 or dataloader_samples > 0 or (stdout_recent and progress_stalled)):
            progress_signal = "degraded"
            if idle_samples > 0:
                reasons.append("idle_gpu_samples")
            if dataloader_samples > 0:
                reasons.append("low_gpu_high_cpu_samples")
            if stdout_recent and progress_stalled:
                reasons.append("stdout_without_structured_progress")
        else:
            progress_signal = "unknown"
            reasons.append("insufficient_observation_window")

    evidence_windows = max(
        _evidence_windows(stdout_age, stalled_stdout_sec),
        _evidence_windows(progress_age_sec, no_progress_sec),
        _evidence_windows(artifact_age_sec, no_artifact_sec),
    )
    if progress_signal == str(previous_signal or ""):
        consecutive = max(int(previous_windows or 0) + 1, evidence_windows)
    else:
        consecutive = max(1 if progress_signal != "unknown" else 0, evidence_windows)

    if progress_signal == "active":
        recent_count = int(stdout_recent) + int(progress_recent) + int(artifact_recent) + int(near_submission)
        confidence = "high" if recent_count >= 2 else "medium"
    elif progress_signal in LOW_PROGRESS_SIGNALS:
        confidence = "high" if consecutive >= max(1, int(min_confidence_windows or 1)) else "medium"
    else:
        confidence = "low"

    return {
        "progress_signal": progress_signal,
        "progress_confidence": confidence,
        "progress_signal_windows": int(consecutive),
        "progress_signal_reason": "+".join(reasons) if reasons else progress_signal,
        "multi_window_low_progress": bool(
            progress_signal in LOW_PROGRESS_SIGNALS
            and consecutive >= max(1, int(min_confidence_windows or 1))
            and confidence == "high"
        ),
        "progress_age_sec": max(0.0, float(progress_age_sec or 0.0)),
        "artifact_age_sec": max(0.0, float(artifact_age_sec or 0.0)),
        "stdout_age_sec": stdout_age,
    }
