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


DEFAULT_TRIAL_WINDOW_SEC = 300.0
DEFAULT_TRIAL_HARD_REVIEW_SEC = 900.0
HIGH_PROFILE_TRIAL_WINDOW_SEC = 180.0
HIGH_PROFILE_TRIAL_HARD_REVIEW_SEC = 480.0


HARD_BOUNDARY_REASONS = frozenset(
    {
        "assigned_gpu_outside_task_pool",
        "invalid_cuda_visible_devices",
        "gpu_boundary_violation",
        "resource_boundary_violation",
        "workspace_safety_violation",
        "destructive_command_outside_workspace",
        "explicit_stop_requested",
        "invalid_deliverable_schema_preflight",
    }
)

SOFT_TRIAL_REASONS = frozenset(
    {
        "stale_resource_context",
        "block_train_after_gpu_pressure",
        "yellow_pressure_exit_guard",
        "gpu_pressure_yellow_or_red",
        "gpu_memory_below_admission_reserve",
        "post_feedback_blocked_class",
        "active_resource_plan_guard",
        "block_train_after_queue_timeout",
        "tt_only_after_queue_timeouts",
    }
)

PENDING_ONLY_REASONS = frozenset(
    {
        "admission_waiter_ahead",
        "gpu_slot_unavailable",
        "slot_capacity_exceeded",
        "incompatible_active_class",
        "class_limit_exceeded",
    }
)


@dataclass(frozen=True)
class StartupBlockClassification:
    block_class: str
    reason: str
    source_reason: str = ""


def classify_preflight_block(
    *,
    reason: str,
    source_reason: str = "",
    resource_class: str = "",
) -> StartupBlockClassification:
    """Classify startup blocks for monitor-first admission.

    Startup should only hard-deny deterministic boundaries. Resource uncertainty
    should usually become a monitored trial if a task-local lease can be assigned.
    """

    clean_reason = _clean(reason)
    clean_source = _clean(source_reason)
    if clean_reason in HARD_BOUNDARY_REASONS or clean_source in HARD_BOUNDARY_REASONS:
        return StartupBlockClassification("hard_boundary", clean_reason, clean_source)
    if clean_reason == "duplicate_digest_cooldown" or clean_source == "duplicate_digest_cooldown":
        return StartupBlockClassification("pending_only", clean_reason, clean_source)
    if clean_reason in SOFT_TRIAL_REASONS or clean_source in SOFT_TRIAL_REASONS:
        return StartupBlockClassification("soft_trial", clean_reason, clean_source)
    if clean_reason in PENDING_ONLY_REASONS or clean_source in PENDING_ONLY_REASONS:
        return StartupBlockClassification("pending_only", clean_reason, clean_source)
    if _is_gpu_training_like(resource_class) and clean_reason in {"replan", "denied_replan"}:
        return StartupBlockClassification("soft_trial", clean_reason, clean_source)
    return StartupBlockClassification("pending_only", clean_reason, clean_source)


def startup_policy_trial_first(policy: str) -> bool:
    return _clean(policy) in {"trial_first", "monitor_first"}


def trial_windows_for_profile(
    *,
    profile: str,
    window_sec: float | None = None,
    hard_review_sec: float | None = None,
) -> tuple[float, float]:
    """Return monitor-first trial windows from a small public config surface."""

    base_window = max(1.0, float(window_sec or DEFAULT_TRIAL_WINDOW_SEC))
    base_hard = max(base_window, float(hard_review_sec or DEFAULT_TRIAL_HARD_REVIEW_SEC))
    if _clean(profile) == "high":
        high_window = min(base_window, HIGH_PROFILE_TRIAL_WINDOW_SEC)
        high_hard = min(base_hard, HIGH_PROFILE_TRIAL_HARD_REVIEW_SEC)
        return high_window, max(high_window, high_hard)
    return base_window, base_hard


def _clean(value: str) -> str:
    return str(value or "").strip().lower()


def _is_gpu_training_like(resource_class: str) -> bool:
    value = _clean(resource_class)
    return bool(value and ("gpu" in value or "train" in value or "cuda" in value))
