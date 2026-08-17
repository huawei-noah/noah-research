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
class ResourceControlProfile:
    """Task-local resource control preset.

    The profile tunes observation cadence and stale gate cleanup only. It does
    not encode task-specific routing or model choices.
    """

    name: str
    min_register_sec: float | None = None
    kill_proposal_cooldown_sec: float | None = None
    arbiter_min_progress_windows: int | None = None
    arbiter_kill_requires_high_confidence: bool | None = None
    arbiter_proposal_coalesce_window_sec: float | None = None
    arbiter_periodic_min_runtime_sec: float | None = None
    arbiter_periodic_min_interval_sec: float | None = None
    arbiter_contention_min_interval_sec: float | None = None
    stale_pressure_observe_window_sec: float | None = None
    stale_pressure_observe_max_sec: float | None = None
    stalled_stdout_sec: float | None = None
    low_progress_warmup_sec: float | None = None
    low_progress_no_heartbeat_sec: float | None = None
    low_progress_no_artifact_sec: float | None = None
    review_warmup_windows: int | None = None
    review_value_windows: int | None = None
    main_agent_advisory_min_interval_sec: float | None = None
    gpu_idle_lease_warmup_sec: float | None = None
    gpu_idle_lease_min_samples: int | None = None
    gpu_dataloader_bottleneck_warmup_sec: float | None = None
    gpu_dataloader_bottleneck_min_samples: int | None = None
    quick_probe_expected_runtime_sec: float | None = None
    quick_probe_hard_review_sec: float | None = None
    observe_more_sec_cap: float | None = None
    deny_kill_ttl_cap: float | None = None
    clear_stale_soft_gpu_gate_on_free_gpu: bool = False
    ignore_waiters_for_soft_gpu_gate: bool = False


_NORMAL = ResourceControlProfile(name="normal")

_HIGH = ResourceControlProfile(
    name="high",
    min_register_sec=120.0,
    kill_proposal_cooldown_sec=180.0,
    arbiter_min_progress_windows=1,
    arbiter_kill_requires_high_confidence=False,
    arbiter_proposal_coalesce_window_sec=30.0,
    arbiter_periodic_min_runtime_sec=300.0,
    arbiter_periodic_min_interval_sec=60.0,
    arbiter_contention_min_interval_sec=60.0,
    stale_pressure_observe_window_sec=60.0,
    stale_pressure_observe_max_sec=180.0,
    stalled_stdout_sec=300.0,
    low_progress_warmup_sec=600.0,
    low_progress_no_heartbeat_sec=600.0,
    low_progress_no_artifact_sec=600.0,
    review_warmup_windows=1,
    review_value_windows=2,
    main_agent_advisory_min_interval_sec=120.0,
    gpu_idle_lease_warmup_sec=120.0,
    gpu_idle_lease_min_samples=2,
    gpu_dataloader_bottleneck_warmup_sec=360.0,
    gpu_dataloader_bottleneck_min_samples=2,
    quick_probe_expected_runtime_sec=180.0,
    quick_probe_hard_review_sec=420.0,
    observe_more_sec_cap=120.0,
    deny_kill_ttl_cap=180.0,
    clear_stale_soft_gpu_gate_on_free_gpu=True,
    ignore_waiters_for_soft_gpu_gate=True,
)


def build_resource_control_profile(name: str | None) -> ResourceControlProfile:
    value = str(name or "normal").strip().lower()
    if value == "high":
        return _HIGH
    return _NORMAL


def cap_float(current: float, cap: float | None, *, minimum: float) -> float:
    """Apply an optional upper cap and required lower bound to a float config."""

    if cap is None:
        return max(float(minimum), float(current))
    return max(float(minimum), min(float(current), float(cap)))


def cap_int(current: int, cap: int | None, *, minimum: int) -> int:
    """Apply an optional upper cap and required lower bound to an int config."""

    if cap is None:
        return max(int(minimum), int(current))
    return max(int(minimum), min(int(current), int(cap)))


def apply_control_profile_to_decision_delay(
    decision: dict[str, Any],
    *,
    profile: ResourceControlProfile,
) -> dict[str, Any]:
    """Cap arbiter delay/TTL fields according to the selected control profile."""

    out = dict(decision or {})
    action = str(out.get("action") or "").upper()
    if action in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL", "CONTINUE_SHARED_OBSERVE"}:
        out["observe_more_sec"] = cap_float(
            float(out.get("observe_more_sec") or 300.0),
            profile.observe_more_sec_cap,
            minimum=1.0,
        )
    if action in {"DENY_KILL", "CONTINUE"}:
        out["ttl_sec"] = cap_float(
            float(out.get("ttl_sec") or 600.0),
            profile.deny_kill_ttl_cap,
            minimum=1.0,
        )
    if profile.name != "normal":
        out["resource_control_profile"] = profile.name
    return out
