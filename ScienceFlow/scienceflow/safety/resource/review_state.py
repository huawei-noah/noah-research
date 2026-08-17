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

import math
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from .review_boundary import PROGRESS_WINDOW, RESOURCE_PRESSURE, ROUTE_VALUE, STALL, TIMEBOX_EXPIRED, ReviewBoundary
from .review_signal import ResourceReviewSignal

WARMUP = "WARMUP"
RUNNING_HEALTHY = "RUNNING_HEALTHY"
RUNNING_NO_PROGRESS = "RUNNING_NO_PROGRESS"
STALL_SUSPECT = "STALL_SUSPECT"
TIMEBOX_ACTIVE = "TIMEBOX_ACTIVE"
ACTIONED = "ACTIONED"
MAX_METRIC_SCOPE_MEMORY = 32

CLEAR_ON_METRIC_UPDATE = "metric_update"
CLEAR_ON_ARTIFACT_GROWTH = "artifact_growth"
CLEAR_ON_PROGRESS_ADVANCE = "progress_advance"
CLEAR_ON_ACTIVE_WORK_RECOVERED = "active_work_recovered"
CLEAR_ON_OPPORTUNITY_COST_CLEARED = "opportunity_cost_cleared"
CLEAR_ON_USEFUL_PROGRESS = "useful_progress"
VALID_CLEAR_ON = {
    CLEAR_ON_METRIC_UPDATE,
    CLEAR_ON_ARTIFACT_GROWTH,
    CLEAR_ON_PROGRESS_ADVANCE,
    CLEAR_ON_ACTIVE_WORK_RECOVERED,
    CLEAR_ON_OPPORTUNITY_COST_CLEARED,
}
_LEGACY_CLEAR_ON = {CLEAR_ON_USEFUL_PROGRESS}


@dataclass(frozen=True)
class ResourceReviewConfig:
    warmup_windows: int = 1
    inactive_windows: int = 1
    value_windows: int = 5
    progress_event_min_windows: int = 5
    timebox_windows: int = 10
    max_proof_windows: int = 2
    min_timebox_sec: float = 60.0
    max_timebox_sec: float = 1800.0
    timebox_budget_fraction: float = 0.10

    def normalized(self) -> "ResourceReviewConfig":
        return ResourceReviewConfig(
            warmup_windows=max(0, int(self.warmup_windows or 0)),
            inactive_windows=max(1, int(self.inactive_windows or 1)),
            value_windows=max(1, int(self.value_windows or 1)),
            progress_event_min_windows=max(1, int(self.progress_event_min_windows or 1)),
            timebox_windows=max(1, int(self.timebox_windows or 1)),
            max_proof_windows=max(1, int(self.max_proof_windows or 1)),
            min_timebox_sec=max(1.0, float(self.min_timebox_sec or 60.0)),
            max_timebox_sec=max(1.0, float(self.max_timebox_sec or 1800.0)),
            timebox_budget_fraction=min(1.0, max(0.01, float(self.timebox_budget_fraction or 0.10))),
        )


@dataclass(frozen=True)
class ResourceReviewState:
    job_id: str = ""
    job_state_bucket: str = WARMUP
    heartbeat_index: int = 0
    last_advance_elapsed_sec: float = 0.0
    inactive_windows: int = 0
    no_useful_progress_windows: int = 0
    progress_event_count_since_review: int = 0
    windows_since_progress_review: int = 0
    timebox_id: str = ""
    timebox_windows: int = 0
    timebox_deadline_windows: int = 0
    timebox_success_condition: str = ""
    timebox_started_with_waiter_pressure: bool = False
    timebox_counts_as_proof_failure: bool = True
    timebox_failure_recorded: bool = False
    proof_window_index: int = 0
    failed_proof_window_count: int = 0
    proof_window_source: str = ""
    active_waiter_pressure: bool = False
    blocked_worker_count: int = 0
    last_metric_update_elapsed_sec: float = 0.0
    last_artifact_growth_elapsed_sec: float = 0.0
    last_active_work_elapsed_sec: float = 0.0
    metric_update_interval_sec: float = 0.0
    artifact_growth_interval_sec: float = 0.0
    active_work_interval_sec: float = 0.0
    metric_scope_key: str = ""
    metric_scope_change_count: int = 0
    metric_scope_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_work_signature: str = ""
    last_review_signature: str = ""
    last_review_boundary: str = ""
    last_outcome: str = ""
    last_stdout_bytes: int = 0
    last_metric_history_text: str = ""
    last_terminal_signal_events: int = 0

    @property
    def in_timebox(self) -> bool:
        return bool(self.timebox_id) and self.job_state_bucket == TIMEBOX_ACTIVE

    def to_json(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_state_bucket": self.job_state_bucket,
            "heartbeat_index": self.heartbeat_index,
            "last_advance_elapsed_sec": float(self.last_advance_elapsed_sec),
            "inactive_windows": self.inactive_windows,
            "no_useful_progress_windows": self.no_useful_progress_windows,
            "progress_event_count_since_review": self.progress_event_count_since_review,
            "windows_since_progress_review": self.windows_since_progress_review,
            "timebox_id": self.timebox_id,
            "timebox_windows": self.timebox_windows,
            "timebox_deadline_windows": self.timebox_deadline_windows,
            "timebox_success_condition": self.timebox_success_condition,
            "timebox_started_with_waiter_pressure": bool(self.timebox_started_with_waiter_pressure),
            "timebox_counts_as_proof_failure": bool(self.timebox_counts_as_proof_failure),
            "timebox_failure_recorded": bool(self.timebox_failure_recorded),
            "proof_window_index": int(self.proof_window_index),
            "failed_proof_window_count": int(self.failed_proof_window_count),
            "proof_window_source": self.proof_window_source,
            "active_waiter_pressure": bool(self.active_waiter_pressure),
            "blocked_worker_count": int(self.blocked_worker_count),
            "clear_on_cadence": {
                "metric_update_interval_sec": float(self.metric_update_interval_sec),
                "artifact_growth_interval_sec": float(self.artifact_growth_interval_sec),
                "active_work_interval_sec": float(self.active_work_interval_sec),
            },
            "metric_scope_key": self.metric_scope_key,
            "metric_scope_change_count": int(self.metric_scope_change_count),
            "metric_scope_memory_size": len(self.metric_scope_memory),
            "active_work_signature": self.active_work_signature,
            "last_review_signature": self.last_review_signature,
            "last_review_boundary": self.last_review_boundary,
            "last_outcome": self.last_outcome,
            "last_stdout_bytes": self.last_stdout_bytes,
            "has_metric_history": bool(self.last_metric_history_text),
            "last_terminal_signal_events": int(self.last_terminal_signal_events),
        }


def normalize_clear_on(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in VALID_CLEAR_ON or text in _LEGACY_CLEAR_ON:
        return text
    return ""


def signal_satisfies_clear_on(clear_on: str, signal: ResourceReviewSignal) -> bool:
    target = normalize_clear_on(clear_on)
    if target == CLEAR_ON_METRIC_UPDATE:
        return bool(signal.metric_changed)
    if target == CLEAR_ON_ARTIFACT_GROWTH:
        return str(signal.artifact_bucket or "") == "growing"
    if target == CLEAR_ON_PROGRESS_ADVANCE:
        if str(getattr(signal, "metric_value_status", "") or "") == "below_observed_best":
            return False
        return bool(getattr(signal, "structured_progress_advanced", False))
    if target == CLEAR_ON_ACTIVE_WORK_RECOVERED:
        return bool(signal.active_work)
    if target == CLEAR_ON_OPPORTUNITY_COST_CLEARED:
        return not bool(signal.active_waiter_pressure)
    if target == CLEAR_ON_USEFUL_PROGRESS:
        return bool(signal.useful_progress)
    return False


def _updated_interval(*, previous_elapsed: float, current_elapsed: float, fallback: float) -> tuple[float, float]:
    prev = max(0.0, float(previous_elapsed or 0.0))
    cur = max(0.0, float(current_elapsed or 0.0))
    if prev > 0.0 and cur > prev:
        return cur, cur - prev
    return cur, max(0.0, float(fallback or 0.0))


def clear_on_interval_sec(state: ResourceReviewState, clear_on: str) -> float:
    target = normalize_clear_on(clear_on)
    if target == CLEAR_ON_METRIC_UPDATE:
        return max(0.0, float(state.metric_update_interval_sec or 0.0))
    if target == CLEAR_ON_ARTIFACT_GROWTH:
        return max(0.0, float(state.artifact_growth_interval_sec or 0.0))
    if target == CLEAR_ON_ACTIVE_WORK_RECOVERED:
        return max(0.0, float(state.active_work_interval_sec or 0.0))
    if target == CLEAR_ON_OPPORTUNITY_COST_CLEARED:
        return 0.0
    return 0.0


def compute_timebox_sec(
    state: ResourceReviewState,
    *,
    clear_on: str,
    heartbeat_sec: float = 60.0,
    fallback_next_review_sec: float = 300.0,
    min_timebox_sec: float = 60.0,
    max_timebox_sec: float = 1800.0,
    remaining_budget_sec: float = 0.0,
    timebox_budget_fraction: float = 0.10,
    active_waiter_pressure: bool = False,
) -> float:
    heartbeat = max(1.0, float(heartbeat_sec or 60.0))
    min_bound = max(heartbeat, float(min_timebox_sec or heartbeat))
    fallback = max(min_bound, float(fallback_next_review_sec or heartbeat))
    absolute_cap = max(min_bound, float(max_timebox_sec or 1800.0))
    budget_fraction = min(1.0, max(0.01, float(timebox_budget_fraction or 0.10)))
    remaining = max(0.0, float(remaining_budget_sec or 0.0))
    budget_cap = remaining * budget_fraction if remaining > 0.0 else absolute_cap
    contention_cap = fallback if active_waiter_pressure else absolute_cap
    max_bound = max(min_bound, min(absolute_cap, max(min_bound, budget_cap), max(min_bound, contention_cap)))
    observed = clear_on_interval_sec(state, clear_on)
    if observed <= 0.0:
        observed = fallback
    desired = max(min_bound, 1.5 * observed)
    return max(min_bound, min(desired, max_bound))


def new_review_state(job_id: str) -> ResourceReviewState:
    return ResourceReviewState(job_id=str(job_id or ""))


def advance_review_state(
    state: ResourceReviewState | None,
    signal: ResourceReviewSignal,
    *,
    config: ResourceReviewConfig | None = None,
) -> ResourceReviewState:
    cfg = (config or ResourceReviewConfig()).normalized()
    current = state or new_review_state("")
    heartbeat_index = int(current.heartbeat_index or 0) + 1
    in_warmup = heartbeat_index <= cfg.warmup_windows
    previous_scope = str(current.metric_scope_key or "")
    next_scope = str(getattr(signal, "metric_scope_key", "") or "")
    scope_changed = bool(previous_scope and next_scope and previous_scope != next_scope)

    scope_memory = {
        str(key): dict(value)
        for key, value in (current.metric_scope_memory or {}).items()
        if str(key) and isinstance(value, dict)
    }
    if scope_changed and previous_scope:
        previous_scope_state = {
            "no_useful_progress_windows": int(current.no_useful_progress_windows or 0),
            "progress_event_count_since_review": int(current.progress_event_count_since_review or 0),
            "windows_since_progress_review": int(current.windows_since_progress_review or 0),
            "proof_window_index": int(current.proof_window_index or 0),
            "failed_proof_window_count": int(current.failed_proof_window_count or 0),
            "proof_window_source": str(current.proof_window_source or ""),
            "last_metric_update_elapsed_sec": float(current.last_metric_update_elapsed_sec or 0.0),
            "metric_update_interval_sec": float(current.metric_update_interval_sec or 0.0),
        }
        if previous_scope in scope_memory or len(scope_memory) < MAX_METRIC_SCOPE_MEMORY:
            scope_memory[previous_scope] = previous_scope_state
    scope_state_available = bool(scope_changed and next_scope in scope_memory)
    new_scope_allowed = bool(scope_changed and len(scope_memory) < MAX_METRIC_SCOPE_MEMORY)
    scope_reset_allowed = bool(scope_state_available or new_scope_allowed)
    restored_scope = scope_memory.get(next_scope, {}) if scope_state_available else {}

    def _scope_int(name: str, fallback: int) -> int:
        if scope_changed and scope_reset_allowed:
            return int(restored_scope.get(name) or 0)
        return int(fallback or 0)

    def _scope_float(name: str, fallback: float) -> float:
        if scope_changed and scope_reset_allowed:
            return float(restored_scope.get(name) or 0.0)
        return float(fallback or 0.0)

    inactive_windows = 0 if signal.active_work else int(current.inactive_windows or 0) + 1
    progress_event_count = _scope_int(
        "progress_event_count_since_review",
        current.progress_event_count_since_review,
    )
    if signal.useful_progress:
        progress_event_count += 1
    windows_since_progress_review = _scope_int(
        "windows_since_progress_review",
        current.windows_since_progress_review,
    ) + 1

    timebox_id = current.timebox_id
    timebox_deadline = int(current.timebox_deadline_windows or 0)
    timebox_success = current.timebox_success_condition
    timebox_windows = int(current.timebox_windows or 0)
    timebox_started_with_waiter_pressure = bool(current.timebox_started_with_waiter_pressure)
    timebox_counts_as_proof_failure = bool(current.timebox_counts_as_proof_failure)
    timebox_failure_recorded = bool(current.timebox_failure_recorded)
    proof_window_index = _scope_int("proof_window_index", current.proof_window_index)
    failed_proof_window_count = _scope_int(
        "failed_proof_window_count",
        current.failed_proof_window_count,
    )
    proof_window_source = (
        str(restored_scope.get("proof_window_source") or "")
        if scope_changed and scope_reset_allowed
        else str(current.proof_window_source or "")
    )
    active_waiter_pressure = bool(signal.active_waiter_pressure)
    blocked_worker_count = max(0, int(getattr(signal, "blocked_worker_count", 0) or 0))
    timebox_active = bool(timebox_id)
    if scope_changed:
        timebox_id = ""
        timebox_deadline = 0
        timebox_success = ""
        timebox_windows = 0
        timebox_started_with_waiter_pressure = False
        timebox_counts_as_proof_failure = True
        timebox_failure_recorded = False
        timebox_active = False

    if timebox_active and signal_satisfies_clear_on(timebox_success, signal):
        timebox_id = ""
        timebox_deadline = 0
        timebox_success = ""
        timebox_windows = 0
        timebox_started_with_waiter_pressure = False
        timebox_counts_as_proof_failure = True
        timebox_failure_recorded = False
        proof_window_source = ""
        timebox_active = False
    elif timebox_active:
        timebox_windows += 1

    if signal.proof_reset_progress:
        failed_proof_window_count = 0
        proof_window_index = 0

    previous_no_useful = _scope_int(
        "no_useful_progress_windows",
        current.no_useful_progress_windows,
    )
    if signal.useful_progress:
        no_useful = 0
    elif in_warmup:
        no_useful = previous_no_useful
    elif signal.active_work:
        no_useful = previous_no_useful + 1
    else:
        no_useful = previous_no_useful

    metric_elapsed = _scope_float(
        "last_metric_update_elapsed_sec",
        current.last_metric_update_elapsed_sec,
    )
    metric_interval = _scope_float(
        "metric_update_interval_sec",
        current.metric_update_interval_sec,
    )
    if scope_changed and scope_reset_allowed:
        metric_elapsed = float(signal.elapsed_sec or 0.0)
    elif signal.metric_changed:
        metric_elapsed, metric_interval = _updated_interval(
            previous_elapsed=metric_elapsed,
            current_elapsed=signal.elapsed_sec,
            fallback=metric_interval,
        )
    artifact_elapsed = float(current.last_artifact_growth_elapsed_sec or 0.0)
    artifact_interval = float(current.artifact_growth_interval_sec or 0.0)
    if str(signal.artifact_bucket or "") == "growing":
        artifact_elapsed, artifact_interval = _updated_interval(
            previous_elapsed=current.last_artifact_growth_elapsed_sec,
            current_elapsed=signal.elapsed_sec,
            fallback=current.artifact_growth_interval_sec,
        )
    active_elapsed = float(current.last_active_work_elapsed_sec or 0.0)
    active_interval = float(current.active_work_interval_sec or 0.0)
    if signal.active_work:
        active_elapsed, active_interval = _updated_interval(
            previous_elapsed=current.last_active_work_elapsed_sec,
            current_elapsed=signal.elapsed_sec,
            fallback=current.active_work_interval_sec,
        )

    if timebox_active:
        bucket = TIMEBOX_ACTIVE
    elif signal.active_work and in_warmup:
        bucket = WARMUP
    elif not signal.active_work and inactive_windows > 0:
        bucket = STALL_SUSPECT
    elif signal.active_work and no_useful > 0:
        bucket = RUNNING_NO_PROGRESS
    else:
        bucket = RUNNING_HEALTHY

    return replace(
        current,
        heartbeat_index=heartbeat_index,
        last_advance_elapsed_sec=max(float(getattr(signal, "elapsed_sec", 0.0) or 0.0), float(current.last_advance_elapsed_sec or 0.0)),
        job_state_bucket=bucket,
        inactive_windows=inactive_windows,
        no_useful_progress_windows=no_useful,
        progress_event_count_since_review=progress_event_count,
        windows_since_progress_review=windows_since_progress_review,
        timebox_id=timebox_id,
        timebox_windows=timebox_windows,
        timebox_deadline_windows=timebox_deadline,
        timebox_success_condition=timebox_success,
        timebox_started_with_waiter_pressure=timebox_started_with_waiter_pressure,
        timebox_counts_as_proof_failure=timebox_counts_as_proof_failure,
        timebox_failure_recorded=timebox_failure_recorded,
        proof_window_index=proof_window_index,
        failed_proof_window_count=failed_proof_window_count,
        proof_window_source=proof_window_source,
        active_waiter_pressure=active_waiter_pressure,
        blocked_worker_count=blocked_worker_count,
        last_metric_update_elapsed_sec=metric_elapsed,
        last_artifact_growth_elapsed_sec=artifact_elapsed,
        last_active_work_elapsed_sec=active_elapsed,
        metric_update_interval_sec=metric_interval,
        artifact_growth_interval_sec=artifact_interval,
        active_work_interval_sec=active_interval,
        metric_scope_key=next_scope or previous_scope,
        metric_scope_change_count=int(current.metric_scope_change_count or 0) + (1 if scope_changed else 0),
        metric_scope_memory=scope_memory,
        active_work_signature=signal.active_work_signature,
        last_stdout_bytes=max(0, int(getattr(signal, "stdout_bytes", 0) or current.last_stdout_bytes or 0)),
        last_metric_history_text=str(getattr(signal, "metric_history_text", "") or current.last_metric_history_text or ""),
        last_terminal_signal_events=max(
            int(current.last_terminal_signal_events or 0),
            int(getattr(signal, "terminal_signal_events", 0) or 0),
        ),
    )


def next_review_boundary(state: ResourceReviewState, *, config: ResourceReviewConfig | None = None) -> ReviewBoundary | None:
    cfg = (config or ResourceReviewConfig()).normalized()
    if int(state.inactive_windows or 0) >= cfg.inactive_windows:
        return ReviewBoundary(STALL, f"inactive_windows>={cfg.inactive_windows}")
    if state.job_state_bucket == TIMEBOX_ACTIVE and bool(state.active_waiter_pressure) and not bool(state.timebox_started_with_waiter_pressure):
        return ReviewBoundary(RESOURCE_PRESSURE, "new_waiter_pressure_during_timebox")
    if state.job_state_bucket == TIMEBOX_ACTIVE and int(state.timebox_windows or 0) >= max(1, int(state.timebox_deadline_windows or cfg.timebox_windows)):
        return ReviewBoundary(TIMEBOX_EXPIRED, "timebox_expired")
    if state.job_state_bucket == TIMEBOX_ACTIVE:
        return None
    if state.job_state_bucket in {RUNNING_HEALTHY, RUNNING_NO_PROGRESS} and int(state.no_useful_progress_windows or 0) >= cfg.value_windows:
        return ReviewBoundary(ROUTE_VALUE, f"no_useful_progress_windows>={cfg.value_windows}")
    if int(state.progress_event_count_since_review or 0) > 0 and int(state.windows_since_progress_review or 0) >= cfg.progress_event_min_windows:
        return ReviewBoundary(PROGRESS_WINDOW, f"progress_window_elapsed>={cfg.progress_event_min_windows}")
    return None


def mark_review_emitted(state: ResourceReviewState, boundary: ReviewBoundary) -> ResourceReviewState:
    updates: dict[str, Any] = {
        "last_review_boundary": boundary.kind,
        "last_review_signature": state.active_work_signature,
    }
    if boundary.kind == PROGRESS_WINDOW:
        updates.update(progress_event_count_since_review=0, windows_since_progress_review=0)
    if boundary.kind == TIMEBOX_EXPIRED and not state.timebox_failure_recorded:
        updates["timebox_failure_recorded"] = True
        if state.timebox_counts_as_proof_failure:
            updates["failed_proof_window_count"] = int(state.failed_proof_window_count or 0) + 1
    return replace(state, **updates)


def apply_timebox(
    state: ResourceReviewState,
    *,
    timebox_sec: float,
    heartbeat_sec: float = 60.0,
    clear_on: str = CLEAR_ON_METRIC_UPDATE,
    active_waiter_pressure: bool = False,
    proof_window_source: str = "system_fixed",
    counts_as_proof_failure: bool | None = None,
) -> ResourceReviewState:
    target = normalize_clear_on(clear_on) or CLEAR_ON_METRIC_UPDATE
    windows = int(math.ceil(max(1.0, float(timebox_sec or 0.0)) / max(1.0, float(heartbeat_sec or 60.0))))
    if windows <= 0:
        windows = 1
    source = str(proof_window_source or "system_fixed").strip().lower()
    counts = (not bool(active_waiter_pressure)) if counts_as_proof_failure is None else bool(counts_as_proof_failure)
    return replace(
        state,
        job_state_bucket=TIMEBOX_ACTIVE,
        timebox_id=f"tb_{uuid.uuid4().hex[:10]}",
        timebox_windows=0,
        timebox_deadline_windows=windows,
        timebox_success_condition=target,
        timebox_started_with_waiter_pressure=bool(active_waiter_pressure),
        timebox_counts_as_proof_failure=counts,
        timebox_failure_recorded=False,
        proof_window_index=int(state.proof_window_index or 0) + 1,
        proof_window_source=source,
        active_waiter_pressure=bool(active_waiter_pressure),
        last_outcome="TIMEBOX",
    )


def apply_no_action(
    state: ResourceReviewState,
    *,
    observe_more_sec: float = 0.0,
    heartbeat_sec: float = 60.0,
    success_condition: str = CLEAR_ON_USEFUL_PROGRESS,
    counts_as_proof_failure: bool = False,
) -> ResourceReviewState:
    return apply_timebox(
        state,
        timebox_sec=float(observe_more_sec or 0.0),
        heartbeat_sec=heartbeat_sec,
        clear_on=success_condition or CLEAR_ON_USEFUL_PROGRESS,
        active_waiter_pressure=bool(state.active_waiter_pressure),
        proof_window_source="no_action",
        counts_as_proof_failure=bool(counts_as_proof_failure),
    )


def apply_kill(state: ResourceReviewState) -> ResourceReviewState:
    return replace(
        state,
        job_state_bucket=ACTIONED,
        timebox_id="",
        timebox_windows=0,
        timebox_deadline_windows=0,
        timebox_success_condition="",
        timebox_started_with_waiter_pressure=False,
        timebox_counts_as_proof_failure=True,
        timebox_failure_recorded=False,
        last_outcome="KILL",
    )
