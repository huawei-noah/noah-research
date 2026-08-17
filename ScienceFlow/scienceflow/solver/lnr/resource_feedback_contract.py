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

"""Observation-only RESOURCE_FEEDBACK formatting."""

from __future__ import annotations

from typing import Any


def eta_bucket(eta_next_train_sec: float | int | None, *, confidence: str = "") -> str:
    if eta_next_train_sec is None:
        return "unknown"
    try:
        eta = max(0.0, float(eta_next_train_sec))
    except (TypeError, ValueError):
        return "unknown"
    if eta <= 0:
        return "unknown" if str(confidence or "").lower() == "low" else "short"
    if eta <= 120:
        return "short"
    if eta <= 600:
        return "medium"
    return "long"


def normalize_eta_confidence(value: str | None) -> str:
    confidence = str(value or "").strip().lower()
    return confidence if confidence in {"high", "medium", "low"} else "unknown"


def resource_feedback_text(
    *,
    status: str,
    reason: str,
    scope: str,
    resource_mode: str,
    blocked_class: str,
    gpu_ids: list[str] | tuple[str, ...] | None = None,
    allowed_classes: list[str] | tuple[str, ...] | None = None,
    holder_job_id: str = "",
    queue_position: int | None = None,
    queue_len: int | None = None,
    cooldown_sec: float | None = None,
    eta_next_train_sec: float | None = None,
    eta_confidence: str = "",
    pressure_generation: int | None = None,
    duplicate_digest_count: int | None = None,
    unlock_condition: str = "",
    blocked_until_unlock: bool | None = None,
    schema_state: str = "",
    artifact_state: str = "",
    progress_state: str = "",
    extra_facts: dict[str, Any] | None = None,
) -> str:
    """Format concise observable resource facts.

    The feedback may include constraint unlock conditions, but it must not
    recommend research routes or tell the main agent what to try next.
    """
    status_text = _token(status or "DENIED_REPLAN").upper()
    reason_text = _value(reason or "resource_policy_gate")
    details = [
        f"mode={_token(resource_mode or 'UNKNOWN').upper()}",
        f"scope={_token(scope or 'task')}",
        f"blocked={_token(blocked_class or 'unknown')}",
    ]
    _append_list(details, "gpu", gpu_ids)
    if holder_job_id:
        details.append(f"holder={_token(holder_job_id)}")
    if queue_position is not None:
        details.append(f"queue_position={max(0, int(float(queue_position or 0)))}")
    if queue_len is not None:
        details.append(f"queue_len={max(0, int(float(queue_len or 0)))}")
    _append_list(details, "allowed", allowed_classes)
    if cooldown_sec is not None:
        details.append(f"cooldown_sec={max(0.0, float(cooldown_sec or 0.0)):.0f}")
    if eta_next_train_sec is not None:
        eta = max(0.0, float(eta_next_train_sec or 0.0))
        confidence = normalize_eta_confidence(eta_confidence)
        details.append(f"eta_next_train_sec={eta:.0f}")
        details.append(f"eta_bucket={eta_bucket(eta, confidence=confidence)}")
        details.append(f"eta_confidence={confidence}")
    if unlock_condition:
        details.append(f"unlock_condition={_token(unlock_condition)}")
    if blocked_until_unlock is not None:
        details.append(f"blocked_until_unlock={str(bool(blocked_until_unlock)).lower()}")
    for key, value in (
        ("schema_state", schema_state),
        ("artifact_state", artifact_state),
        ("progress_state", progress_state),
    ):
        if value:
            details.append(f"{key}={_token(value)}")
    if pressure_generation is not None:
        details.append(f"pressure_generation={int(pressure_generation or 0)}")
    if duplicate_digest_count is not None:
        details.append(f"duplicate_digest_count={int(duplicate_digest_count or 0)}")
    for key, raw in sorted((extra_facts or {}).items()):
        token_key = _token(str(key)).lower()
        if token_key and raw is not None and str(raw).strip():
            details.append(f"{token_key}={_token(str(raw))}")
    return f"RESOURCE_FEEDBACK: {status_text} because {reason_text}; {'; '.join(details)}.\n"


def _append_list(details: list[str], key: str, values: list[str] | tuple[str, ...] | None) -> None:
    clean = [_token(str(x)) for x in (values or []) if str(x).strip()]
    if clean:
        details.append(f"{key}=" + ",".join(clean))


def _token(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "_.:-," else "_" for ch in str(value or "").strip())
    return out.strip("_") or "unknown"


def _value(value: str) -> str:
    return " ".join(str(value or "").strip().rstrip(".").split()) or "resource_policy_gate"
