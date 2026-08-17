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

import time
from typing import Any


_BACKOFF_GPU_RETRY_ACTIONS = {
    "gpu_train": "sleep_then_gpu_train_retry",
    "unknown_gpu": "sleep_then_unknown_gpu_retry",
    "gpu_feature_extract": "sleep_then_gpu_feature_retry",
}


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def decorate_post_feedback_action_after_backoff(
    *,
    action: str,
    pending_feedback: dict[str, Any],
    last_backoff: dict[str, Any] | None,
    now: float | None = None,
    max_age_sec: float = 30.0,
) -> dict[str, Any]:
    base = str(action or "cpu_support")
    if base not in _BACKOFF_GPU_RETRY_ACTIONS or not isinstance(last_backoff, dict):
        return {"action": base, "base_action": base, "preceded_by_agent_backoff": False}

    ts = time.time() if now is None else float(now)
    feedback_at = _float_or_none(pending_feedback.get("created_at")) or 0.0
    finished_at = _float_or_none(last_backoff.get("finished_at"))
    if finished_at is None or finished_at + max(0.0, float(max_age_sec or 0.0)) < ts:
        return {"action": base, "base_action": base, "preceded_by_agent_backoff": False}
    if feedback_at > 0 and finished_at + 1.0 < feedback_at:
        return {"action": base, "base_action": base, "preceded_by_agent_backoff": False}

    return {
        "action": _BACKOFF_GPU_RETRY_ACTIONS[base],
        "base_action": base,
        "preceded_by_agent_backoff": True,
        "agent_backoff_wait_id": str(last_backoff.get("wait_id") or ""),
        "agent_backoff_elapsed_sec": max(0.0, float(last_backoff.get("elapsed_sec") or 0.0)),
        "agent_backoff_planned_sleep_sec": max(0.0, float(last_backoff.get("planned_sleep_sec") or 0.0)),
        "agent_backoff_wake_reason": str(last_backoff.get("wake_reason") or ""),
        "agent_backoff_status": str(last_backoff.get("status") or ""),
    }
