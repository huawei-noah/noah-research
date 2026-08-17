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

import asyncio
import hashlib
import inspect
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_LIGHT_CPU,
    RESOURCE_LIGHT_GPU_PROBE,
    RESOURCE_READONLY_CPU,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_UNKNOWN_EXEC,
    RESOURCE_UNKNOWN_GPU_EXEC,
)
from scienceflow.solver.lnr.resource_runtime import ResourceRuntime
from scienceflow.solver.lnr.resource_runtime.review.arbiter import (
    TERMINATING_ACTIONS,
    enforce_proposal_action_allowlist,
    enforce_repeated_stall_escalation,
    fallback_policy_decision,
    main_agent_feedback,
    normalize_arbiter_decision,
    parse_arbiter_decision_text,
)
from scienceflow.solver.lnr.resource_runtime.review.arbiter_gate import (
    enforce_arbiter_kill_gate,
    proposal_advisory_fact_conflicts,
)
from scienceflow.solver.lnr.resource_runtime.review.kill_intent import (
    build_kill_intent_snapshot,
    revalidate_kill_intent as evaluate_kill_intent_revalidation,
)
from scienceflow.solver.lnr.resource_runtime.control_profile import (
    ResourceControlProfile,
    apply_control_profile_to_decision_delay,
    build_resource_control_profile,
    cap_float,
    cap_int,
)
from scienceflow.solver.lnr.resource_runtime.review.progress import classify_progress_signal
from scienceflow.safety.resource.process_lifecycle import classify_process_lifecycle
from scienceflow.solver.lnr.resource_runtime.process_liveness import build_process_liveness
from scienceflow.solver.lnr.resource_runtime.proc_inspect import read_proc_info
from scienceflow.solver.lnr.resource_runtime.utilization import process_tree_gpu_placement_snapshot
from scienceflow.solver.lnr.resource_runtime.quick_probe import classify_quick_probe_command
from scienceflow.solver.lnr.resource_runtime.review.state_generation import build_resource_state_generation
from scienceflow.solver.lnr.resource_runtime.review.execution_facts import build_execution_facts
from scienceflow.solver.lnr.resource_runtime.review.efficiency import assess_resource_efficiency
from scienceflow.solver.lnr.resource_runtime.review.research_cadence import (
    build_research_cadence_facts,
    is_comparable_live_metric,
    normalize_execution_scale,
    normalize_route_key,
    normalize_validation_protocol,
    route_metric_evidence,
    validation_protocols_comparable,
)
from scienceflow.solver.lnr.resource_runtime.metric_history import (
    metric_history_line_from_progress_signals,
    metric_history_text as compact_metric_history_text,
    update_metric_history_lines,
)
from scienceflow.solver.lnr.resource_runtime.gpu_sharing import (
    build_gpu_share_config,
    classify_cpu_pressure,
    evaluate_admission_share_trial,
    evaluate_share_phase_a,
    gpu_memory_summary,
)
from scienceflow.solver.lnr.resource_runtime.review.lease_suspect import (
    LeaseSuspectThresholds,
    active_work_counterevidence,
    classify_active_lease_suspect,
    classify_deliverable_validity,
    classify_route_viability,
)
from scienceflow.solver.lnr.resource_runtime.admission import (
    apply_admission_decision,
    normalize_admission_decision,
    parse_admission_decision_text,
    should_request_admission_llm,
)
from scienceflow.solver.lnr.estra_magent.coordinator import ESTRAMagentConfig
from scienceflow.solver.lnr.estra_magent.events import (
    MAGENT_FORK_CONSIDERED,
    MAGENT_JOIN_PACKET_READY,
    MAGENT_SIDECAR_STARTED,
    MAGENT_TRAIN_OBSERVED,
    fork_considered_payload,
    join_packet_ready_payload,
    sidecar_started_payload,
    train_observed_payload,
)
from scienceflow.solver.lnr.estra_magent.join_gate import write_join_packet
from scienceflow.solver.lnr.resource_runtime.models import GPUQueueConfig
from scienceflow.safety.resource.completion import deliverable_completion_state
from scienceflow.safety.resource import (
    KILL as RESOURCE_REVIEW_KILL,
    NO_ACTION as RESOURCE_REVIEW_NO_ACTION,
    TIMEBOX as RESOURCE_REVIEW_TIMEBOX,
    PROGRESS_WINDOW,
    RESOURCE_PRESSURE,
    ROUTE_VALUE,
    STALL,
    TIMEBOX_EXPIRED,
    ResourceReviewConfig,
    ResourceReviewState,
    advance_review_state,
    apply_kill,
    apply_no_action,
    apply_timebox,
    build_review_signal,
    compute_timebox_sec,
    mark_review_emitted,
    new_review_state,
    next_review_boundary,
    normalize_clear_on,
    normalize_value_review_outcome,
)
from scienceflow.solver.lnr.resource_runtime.post_feedback_actions import (
    decorate_post_feedback_action_after_backoff,
)
from scienceflow.solver.lnr.resource_feedback_contract import resource_feedback_text
from scienceflow.solver.lnr.resource_runtime.sidecar import run_cpu_sidecar_backfill
from scienceflow.solver.lnr.resource_runtime.soft_gpu_gate import (
    feedback_guard_is_hard_constraint,
    gpu_store_snapshot_has_lease,
    gpu_store_snapshot_has_pressure,
    normalize_gpu_ids,
    reconcile_free_gpu_pressure_blocked,
    resource_class_can_use_soft_gpu_gate,
)
from scienceflow.solver.lnr.resource_runtime.startup_trial import (
    classify_preflight_block,
    startup_policy_trial_first,
    trial_windows_for_profile,
)
from scienceflow.solver.lnr.stage.score_summary import (
    build_stage_performance_score_summary,
    metric_lower_is_better_hint,
)
from scienceflow.solver.lnr.resource_runtime.source_hints import (
    ResourceSourceHint,
    detect_resource_source_hint,
)
from scienceflow.solver.lnr.state_machine import LHRStateMachineStore
from scienceflow.solver.lnr.resource_runtime.jobs import (
    ResourceJob,
    TRACKABLE_CLASSES,
    TRAIN_CLASSES,
    TT_ALLOWED_AFTER_TIMEOUT,
)

# Preserve the historical event value during the Safety namespace migration.
# This is a compatibility token, not the name of an internal module.
_SAFETY_HEARTBEAT_SOURCE = "safety_heartbeat"


class LHRResourceObserver:
    """Rule-based resource sidecar for lnr bash jobs.

    Short/light bash commands are not written to the state/event stream. Heavy or
    unknown commands are promoted only after the time threshold, except GPU queue
    requests which are visible immediately through resource request/lease events.
    """

    def __init__(
        self,
        *,
        state_machine: LHRStateMachineStore,
        worker_id: str,
        resource_control_profile: str = "normal",
        min_register_sec: float = 600.0,
        check_interval_sec: float = 10.0,
        stalled_stdout_sec: float = 900.0,
        kill_enabled: bool = True,
        kill_mode: str = "recommend",
        monitor_agent_mode: str = "off",
        monitor_agent_min_interval_sec: float = 60.0,
        low_progress_enabled: bool = True,
        low_progress_warmup_sec: float = 900.0,
        low_progress_no_heartbeat_sec: float = 900.0,
        low_progress_no_artifact_sec: float = 900.0,
        bash_monitor_all_enabled: bool = False,
        arbiter_min_progress_windows: int = 2,
        arbiter_kill_requires_high_confidence: bool = True,
        task_resource_dir: str | Path | None = None,
        resource_runtime_enabled: bool = False,
        gpu_queue_enabled: bool = False,
        gpu_pool: list[str] | None = None,
        gpu_default_request: int = 1,
        gpu_max_request: int = 1,
        gpu_assignment: str = "env_only",
        gpu_queue_max_wait_sec: float = 1800.0,
        gpu_queue_heartbeat_sec: float = 15.0,
        gpu_max_heavy_per_gpu: int = 1,
        gpu_capacity_slots: float = 1.0,
        gpu_tt_max_per_gpu: int = 3,
        gpu_feature_max_per_gpu: int = 2,
        gpu_share_tt_with_train: bool = False,
        gpu_share_enabled: bool = False,
        gpu_share_phase: str = "observe",
        gpu_share_policy_profile: str = "conservative",
        gpu_share_memory_profile: str = "conservative",
        gpu_share_cpu_policy: str = "conservative",
        gpu_trial_admission_policy: str = "llm_grant",
        resource_startup_policy: str = "conservative",
        resource_trial_window_sec: float = 300.0,
        resource_trial_hard_review_sec: float = 900.0,
        gpu_lease_ttl_sec: float = 7200.0,
        gpu_duplicate_digest_cooldown_sec: float = 600.0,
        gpu_duplicate_digest_threshold: int = 2,
        gpu_admission_queue_enabled: bool = True,
        gpu_admission_waiter_ttl_sec: float = 900.0,
        admission_llm_enabled: bool = False,
        admission_llm_mode: str = "low_confidence",
        admission_llm_timeout_sec: float = 60.0,
        admission_decider: Any | None = None,
        stale_pressure_observe_first_enabled: bool = True,
        stale_pressure_observe_window_sec: float = 180.0,
        stale_pressure_observe_max_sec: float = 300.0,
        stale_pressure_observe_min_free_mem_gb: float = 8.0,
        stale_pressure_healthy_skip_llm: bool = True,
        gpu_pressure_yellow_hold_sec: float = 120.0,
        gpu_pressure_red_to_yellow_sec: float = 120.0,
        gpu_pressure_yellow_util_pct: float = 85.0,
        gpu_pressure_min_free_mem_gb: float = 8.0,
        gpu_pressure_yellow_free_mem_buffer_gb: float = 8.0,
        observation_enabled: bool = True,
        observation_window_sec: float = 30.0,
        observation_shadow_workspace_enabled: bool = True,
        gpu_source_hint_enabled: bool = True,
        gpu_source_hint_mode: str = "observe",
        gpu_util_observer_enabled: bool = True,
        gpu_util_sample_interval_sec: float = 30.0,
        gpu_idle_lease_guard_enabled: bool = True,
        gpu_idle_lease_warmup_sec: float = 180.0,
        gpu_idle_lease_min_samples: int = 3,
        gpu_idle_lease_util_pct: float = 1.0,
        gpu_idle_lease_mem_gb: float = 1.0,
        gpu_idle_lease_require_pressure: bool = False,
        gpu_idle_lease_action_mode: str = "release",
        resource_idle_release_admission_mode: str = "strict_exclusive",
        quick_probe_guard_enabled: bool = True,
        quick_probe_expected_runtime_sec: float = 300.0,
        quick_probe_hard_review_sec: float = 900.0,
        quick_probe_small_scope_threshold: int = 1000,
        gpu_dataloader_bottleneck_guard_enabled: bool = True,
        gpu_dataloader_bottleneck_warmup_sec: float = 600.0,
        gpu_dataloader_bottleneck_min_samples: int = 3,
        gpu_dataloader_bottleneck_util_pct: float = 15.0,
        gpu_dataloader_bottleneck_min_mem_gb: float = 2.0,
        gpu_dataloader_bottleneck_child_cpu_pct: float = 200.0,
        gpu_dataloader_bottleneck_busy_children: int = 2,
        deliverable_completion_guard_enabled: bool = True,
        deliverable_completion_warmup_sec: float = 120.0,
        deliverable_completion_settle_sec: float = 120.0,
        deliverable_completion_scan_interval_sec: float = 60.0,
        deliverable_completion_quiet_sec: float = 60.0,
        metric_health_guard_enabled: bool = True,
        metric_health_warmup_sec: float = 600.0,
        metric_health_invalid_min_events: int = 2,
        metric_health_zero_score_min_events: int = 2,
        arbiter_enabled: bool = False,
        arbiter_mode: str = "policy",
        arbiter_timeout_sec: float = 90.0,
        arbiter_decider: Any | None = None,
        main_agent_advisory_enabled: bool = False,
        main_agent_advisory_min_interval_sec: float = 600.0,
        main_agent_advisory_timeout_sec: float = 60.0,
        main_agent_advisory_decider: Any | None = None,
        arbiter_contention_review_enabled: bool = False,
        arbiter_contention_min_runtime_sec: float = 900.0,
        arbiter_contention_min_waiter_age_sec: float = 300.0,
        arbiter_contention_min_interval_sec: float = 600.0,
        research_cadence_enabled: bool = True,
        research_cadence_observe_sec: float = 300.0,
        first_comparable_metric_budget_sec: float = 900.0,
        proven_route_metric_budget_sec: float = 1800.0,
        arbiter_periodic_review_enabled: bool = False,
        arbiter_periodic_min_runtime_sec: float = 1800.0,
        arbiter_periodic_min_interval_sec: float = 900.0,
        arbiter_proposal_coalesce_window_sec: float = 60.0,
        arbiter_job_llm_call_cap: int = 6,
        arbiter_job_advisory_call_cap: int = 3,
        arbiter_job_token_cap: int = 24000,
        sidecar_enabled: bool = False,
        sidecar_min_parent_runtime_sec: float = 900.0,
        estra_magent_enabled: bool = False,
        estra_magent_sidecar_enabled: bool = False,
        estra_magent_min_parent_runtime_sec: float = 300.0,
        estra_magent_sidecar_mode: str = "cpu_only",
        estra_magent_budget_sec: float = 900.0,
        estra_magent_join_inject_parent: bool = True,
        estra_magent_join_inject_estra: bool = True,
        estra_magent_join_inject_resource_context: bool = True,
        checkpoint_submission_guard_enabled: bool = True,
        timeout_hard_gate_enabled: bool = True,
        timeout_block_train_after: int = 1,
        timeout_tt_only_after: int = 2,
        review_state_enabled: bool = True,
        review_heartbeat_sec: float = 60.0,
        review_warmup_windows: int = 10,
        review_inactive_windows: int = 3,
        review_value_windows: int = 5,
        review_progress_event_min_windows: int = 5,
        review_timebox_windows: int = 10,
        review_max_proof_windows: int = 2,
        review_min_timebox_sec: float = 60.0,
        review_max_timebox_sec: float = 1800.0,
        review_timebox_budget_fraction: float = 0.10,
    ) -> None:
        self.state_machine = state_machine
        self.worker_id = str(worker_id or "W00")
        self.control_profile: ResourceControlProfile = build_resource_control_profile(resource_control_profile)
        self.resource_control_profile = self.control_profile.name
        self.min_register_sec = cap_float(float(min_register_sec), self.control_profile.min_register_sec, minimum=0.0)
        self.check_interval_sec = max(1.0, float(check_interval_sec))
        self.stalled_stdout_sec = cap_float(float(stalled_stdout_sec), self.control_profile.stalled_stdout_sec, minimum=0.0)
        self.kill_mode = str(kill_mode or ("auto" if kill_enabled else "off")).strip().lower()
        if self.kill_mode not in {"recommend", "auto", "arbiter", "off"}:
            self.kill_mode = "recommend"
        self.kill_enabled = bool(kill_enabled) and self.kill_mode != "off"
        self.auto_kill_enabled = self.kill_enabled and self.kill_mode == "auto"
        self.arbiter_enabled = bool(arbiter_enabled) or self.kill_mode == "arbiter"
        self.arbiter_mode = str(arbiter_mode or "policy").strip().lower()
        if self.arbiter_mode not in {"policy", "llm", "off"}:
            self.arbiter_mode = "policy"
        self.arbiter_timeout_sec = max(1.0, float(arbiter_timeout_sec or 90.0))
        self.arbiter_decider = arbiter_decider
        self.main_agent_advisory_enabled = bool(main_agent_advisory_enabled)
        self.main_agent_advisory_min_interval_sec = cap_float(
            float(main_agent_advisory_min_interval_sec or 600.0),
            self.control_profile.main_agent_advisory_min_interval_sec,
            minimum=1.0,
        )
        self.main_agent_advisory_timeout_sec = max(1.0, float(main_agent_advisory_timeout_sec or 60.0))
        self.main_agent_advisory_decider = main_agent_advisory_decider
        self.arbiter_contention_review_enabled = bool(arbiter_contention_review_enabled)
        self.arbiter_contention_min_runtime_sec = max(0.0, float(arbiter_contention_min_runtime_sec or 0.0))
        self.arbiter_contention_min_waiter_age_sec = max(0.0, float(arbiter_contention_min_waiter_age_sec or 0.0))
        self.arbiter_contention_min_interval_sec = cap_float(
            float(arbiter_contention_min_interval_sec or 600.0),
            self.control_profile.arbiter_contention_min_interval_sec,
            minimum=1.0,
        )
        self.research_cadence_enabled = bool(research_cadence_enabled)
        self.research_cadence_observe_sec = max(1.0, float(research_cadence_observe_sec or 300.0))
        self.first_comparable_metric_budget_sec = max(
            self.research_cadence_observe_sec,
            float(first_comparable_metric_budget_sec or 900.0),
        )
        self.proven_route_metric_budget_sec = max(
            self.first_comparable_metric_budget_sec,
            float(proven_route_metric_budget_sec or 1800.0),
        )
        self._research_route_metrics: dict[str, dict[str, Any]] = {}
        self.arbiter_periodic_review_enabled = bool(arbiter_periodic_review_enabled)
        self.arbiter_periodic_min_runtime_sec = cap_float(
            float(arbiter_periodic_min_runtime_sec or 0.0),
            self.control_profile.arbiter_periodic_min_runtime_sec,
            minimum=0.0,
        )
        self.arbiter_periodic_min_interval_sec = cap_float(
            float(arbiter_periodic_min_interval_sec or 900.0),
            self.control_profile.arbiter_periodic_min_interval_sec,
            minimum=1.0,
        )
        self.arbiter_proposal_coalesce_window_sec = cap_float(
            float(arbiter_proposal_coalesce_window_sec or 0.0),
            self.control_profile.arbiter_proposal_coalesce_window_sec,
            minimum=0.0,
        )
        self.arbiter_job_llm_call_cap = max(0, int(arbiter_job_llm_call_cap or 0))
        self.arbiter_job_advisory_call_cap = max(0, int(arbiter_job_advisory_call_cap or 0))
        self.arbiter_job_token_cap = max(0, int(arbiter_job_token_cap or 0))
        self.admission_llm_enabled = bool(admission_llm_enabled)
        self.admission_llm_mode = str(admission_llm_mode or "low_confidence").strip().lower()
        allowed_admission_modes = {"off", "low_confidence", "always", "all", "blocked_states", "blocked", "soft_gates"}
        if self.admission_llm_mode not in allowed_admission_modes:
            self.admission_llm_mode = "low_confidence"
        self.admission_llm_timeout_sec = max(1.0, float(admission_llm_timeout_sec or 60.0))
        self.admission_decider = admission_decider
        self.stale_pressure_observe_first_enabled = bool(stale_pressure_observe_first_enabled)
        self.stale_pressure_observe_window_sec = cap_float(
            float(stale_pressure_observe_window_sec or 180.0),
            self.control_profile.stale_pressure_observe_window_sec,
            minimum=30.0,
        )
        self.stale_pressure_observe_max_sec = cap_float(
            float(stale_pressure_observe_max_sec or self.stale_pressure_observe_window_sec),
            self.control_profile.stale_pressure_observe_max_sec,
            minimum=self.stale_pressure_observe_window_sec,
        )
        self.stale_pressure_observe_min_free_mem_gb = max(0.0, float(stale_pressure_observe_min_free_mem_gb or 0.0))
        self.stale_pressure_healthy_skip_llm = bool(stale_pressure_healthy_skip_llm)
        self.estra_magent = ESTRAMagentConfig.from_values(
            enabled=bool(estra_magent_enabled),
            sidecar_enabled=bool(estra_magent_sidecar_enabled),
            min_parent_runtime_sec=float(estra_magent_min_parent_runtime_sec or 300.0),
            sidecar_mode=str(estra_magent_sidecar_mode or "cpu_only"),
            budget_sec=float(estra_magent_budget_sec or 900.0),
            join_inject_parent=bool(estra_magent_join_inject_parent),
            join_inject_estra=bool(estra_magent_join_inject_estra),
            join_inject_resource_context=bool(estra_magent_join_inject_resource_context),
        )
        legacy_sidecar_enabled = bool(sidecar_enabled)
        self.sidecar_enabled = legacy_sidecar_enabled or self.estra_magent.sidecar_allowed()
        legacy_min_runtime = max(1.0, float(sidecar_min_parent_runtime_sec or 900.0))
        if self.estra_magent.sidecar_allowed() and legacy_sidecar_enabled:
            self.sidecar_min_parent_runtime_sec = min(legacy_min_runtime, self.estra_magent.min_parent_runtime_sec)
        elif self.estra_magent.sidecar_allowed():
            self.sidecar_min_parent_runtime_sec = self.estra_magent.min_parent_runtime_sec
        else:
            self.sidecar_min_parent_runtime_sec = legacy_min_runtime
        self.checkpoint_submission_guard_enabled = bool(checkpoint_submission_guard_enabled)
        self._sidecar_jobs_started: set[str] = set()
        self._magent_observed_jobs: set[str] = set()
        self._magent_considered_jobs: set[str] = set()
        self.monitor_agent_mode = str(monitor_agent_mode or "off").strip().lower()
        if self.monitor_agent_mode not in {"off", "shadow", "advisory", "gated_kill"}:
            self.monitor_agent_mode = "off"
        self.monitor_agent_min_interval_sec = max(1.0, float(monitor_agent_min_interval_sec or 60.0))
        self.low_progress_enabled = bool(low_progress_enabled)
        self.low_progress_warmup_sec = cap_float(
            float(low_progress_warmup_sec or 0.0),
            self.control_profile.low_progress_warmup_sec,
            minimum=0.0,
        )
        self.low_progress_no_heartbeat_sec = cap_float(
            float(low_progress_no_heartbeat_sec or 0.0),
            self.control_profile.low_progress_no_heartbeat_sec,
            minimum=0.0,
        )
        self.low_progress_no_artifact_sec = cap_float(
            float(low_progress_no_artifact_sec or 0.0),
            self.control_profile.low_progress_no_artifact_sec,
            minimum=0.0,
        )
        self.bash_monitor_all_enabled = bool(bash_monitor_all_enabled)
        self.arbiter_min_progress_windows = cap_int(
            int(arbiter_min_progress_windows or 1),
            self.control_profile.arbiter_min_progress_windows,
            minimum=1,
        )
        if self.control_profile.arbiter_kill_requires_high_confidence is None:
            self.arbiter_kill_requires_high_confidence = bool(arbiter_kill_requires_high_confidence)
        else:
            self.arbiter_kill_requires_high_confidence = bool(self.control_profile.arbiter_kill_requires_high_confidence)
        self._last_monitor_agent_emit: dict[str, float] = {}
        self._last_gpu_util_emit: dict[str, float] = {}
        self._last_kill_proposal_emit: dict[str, float] = {}
        self._last_kill_proposal_id_by_dedupe: dict[str, str] = {}
        self._last_kill_proposal_boundary_by_dedupe: dict[str, dict[str, Any]] = {}
        self._last_suppressed_kill_proposal_emit: dict[str, float] = {}
        self._last_review_proposal_emit: dict[str, float] = {}
        self._review_observe_more_until: dict[str, float] = {}
        self._review_history: dict[str, dict[str, Any]] = {}
        self._last_advisory_emit: dict[str, float] = {}
        self._last_main_agent_advisory: dict[str, dict[str, Any]] = {}
        self._job_llm_call_count: dict[str, int] = {}
        self._job_advisory_call_count: dict[str, int] = {}
        self._job_llm_token_count: dict[str, int] = {}
        self._job_last_llm_metric_history_digest: dict[str, str] = {}
        self._last_idle_lease_candidate_emit: dict[str, float] = {}
        self._last_gpu_share_event_emit: dict[str, float] = {}
        self._last_admission_share_review_emit: dict[str, float] = {}
        self.gpu_share_config = build_gpu_share_config(
            enabled=bool(gpu_share_enabled),
            phase=str(gpu_share_phase or "observe"),
            policy_profile=str(gpu_share_policy_profile or "conservative"),
            memory_profile=str(gpu_share_memory_profile or "conservative"),
            cpu_policy=str(gpu_share_cpu_policy or "conservative"),
            trial_admission_policy=str(gpu_trial_admission_policy or "llm_grant"),
        )
        self.resource_startup_policy = str(resource_startup_policy or "conservative").strip().lower()
        self.resource_trial_window_sec, self.resource_trial_hard_review_sec = trial_windows_for_profile(
            profile=self.control_profile.name,
            window_sec=resource_trial_window_sec,
            hard_review_sec=resource_trial_hard_review_sec,
        )
        self.kill_proposal_cooldown_sec = float(self.control_profile.kill_proposal_cooldown_sec or 600.0)
        self._seq = 0
        self._jobs: dict[str, ResourceJob] = {}
        self._agent_backoff_waits: dict[str, dict[str, Any]] = {}
        self._last_agent_backoff_wait: dict[str, Any] | None = None
        self._pending_resource_feedback: dict[str, Any] | None = None
        self._active_plan_guards: dict[str, dict[str, Any]] = {}
        self._planning_resource_context_version: int | None = None
        self._planning_resource_pressure_generation: int | None = None
        self._feedback_seq = 0
        self._reported_resource_feedback: dict[str, dict[str, Any]] = {}
        self._gpu_queue_timeout_count = 0
        self.timeout_hard_gate_enabled = bool(timeout_hard_gate_enabled)
        self.timeout_block_train_after = max(1, int(timeout_block_train_after or 1))
        self.timeout_tt_only_after = max(self.timeout_block_train_after, int(timeout_tt_only_after or 2))
        self.review_state_enabled = bool(review_state_enabled)
        self.review_heartbeat_sec = max(1.0, float(review_heartbeat_sec or 60.0))
        self.review_config = ResourceReviewConfig(
            warmup_windows=cap_int(
                int(review_warmup_windows or 0),
                self.control_profile.review_warmup_windows,
                minimum=0,
            ),
            inactive_windows=max(1, int(review_inactive_windows or 1)),
            value_windows=cap_int(
                int(review_value_windows or 1),
                self.control_profile.review_value_windows,
                minimum=1,
            ),
            progress_event_min_windows=max(1, int(review_progress_event_min_windows or 1)),
            timebox_windows=max(1, int(review_timebox_windows or 1)),
            max_proof_windows=max(1, int(review_max_proof_windows or 2)),
            min_timebox_sec=max(1.0, float(review_min_timebox_sec or 60.0)),
            max_timebox_sec=max(1.0, float(review_max_timebox_sec or 1800.0)),
            timebox_budget_fraction=min(1.0, max(0.01, float(review_timebox_budget_fraction or 0.10))),
        ).normalized()
        self._review_states: dict[str, ResourceReviewState] = {}
        self.observation_enabled = bool(observation_enabled)
        self.observation_window_sec = max(0.05, float(observation_window_sec or 30.0))
        self.observation_shadow_workspace_enabled = bool(observation_shadow_workspace_enabled)
        self.gpu_util_observer_enabled = bool(gpu_util_observer_enabled)
        self.gpu_util_sample_interval_sec = max(1.0, float(gpu_util_sample_interval_sec or 30.0))
        self.gpu_idle_lease_guard_enabled = bool(gpu_idle_lease_guard_enabled)
        self.gpu_idle_lease_warmup_sec = cap_float(
            float(gpu_idle_lease_warmup_sec or 0.0),
            self.control_profile.gpu_idle_lease_warmup_sec,
            minimum=0.0,
        )
        self.gpu_idle_lease_min_samples = cap_int(
            int(gpu_idle_lease_min_samples or 1),
            self.control_profile.gpu_idle_lease_min_samples,
            minimum=1,
        )
        self.gpu_idle_lease_util_pct = max(0.0, float(gpu_idle_lease_util_pct or 0.0))
        self.gpu_idle_lease_mem_gb = max(0.0, float(gpu_idle_lease_mem_gb or 0.0))
        self.gpu_idle_lease_require_pressure = bool(gpu_idle_lease_require_pressure)
        self.gpu_idle_lease_action_mode = str(gpu_idle_lease_action_mode or "release").strip().lower()
        if self.gpu_idle_lease_action_mode not in {"observe", "recommend", "release", "terminate"}:
            self.gpu_idle_lease_action_mode = "release"
        self.quick_probe_guard_enabled = bool(quick_probe_guard_enabled)
        self.quick_probe_expected_runtime_sec = cap_float(
            float(quick_probe_expected_runtime_sec or 300.0),
            self.control_profile.quick_probe_expected_runtime_sec,
            minimum=30.0,
        )
        self.quick_probe_hard_review_sec = cap_float(
            float(quick_probe_hard_review_sec or 900.0),
            self.control_profile.quick_probe_hard_review_sec,
            minimum=self.quick_probe_expected_runtime_sec,
        )
        self.quick_probe_small_scope_threshold = max(1, int(quick_probe_small_scope_threshold or 1000))
        self.gpu_dataloader_bottleneck_guard_enabled = bool(gpu_dataloader_bottleneck_guard_enabled)
        self.gpu_dataloader_bottleneck_warmup_sec = cap_float(
            float(gpu_dataloader_bottleneck_warmup_sec or 0.0),
            self.control_profile.gpu_dataloader_bottleneck_warmup_sec,
            minimum=0.0,
        )
        self.gpu_dataloader_bottleneck_min_samples = cap_int(
            int(gpu_dataloader_bottleneck_min_samples or 1),
            self.control_profile.gpu_dataloader_bottleneck_min_samples,
            minimum=1,
        )
        self.gpu_dataloader_bottleneck_util_pct = max(0.0, float(gpu_dataloader_bottleneck_util_pct or 0.0))
        self.gpu_dataloader_bottleneck_min_mem_gb = max(0.0, float(gpu_dataloader_bottleneck_min_mem_gb or 0.0))
        self.gpu_dataloader_bottleneck_child_cpu_pct = max(0.0, float(gpu_dataloader_bottleneck_child_cpu_pct or 0.0))
        self.gpu_dataloader_bottleneck_busy_children = max(1, int(gpu_dataloader_bottleneck_busy_children or 1))
        self.deliverable_completion_guard_enabled = bool(deliverable_completion_guard_enabled)
        self.deliverable_completion_warmup_sec = max(0.0, float(deliverable_completion_warmup_sec or 0.0))
        self.deliverable_completion_settle_sec = max(0.0, float(deliverable_completion_settle_sec or 0.0))
        self.deliverable_completion_scan_interval_sec = max(0.0, float(deliverable_completion_scan_interval_sec or 0.0))
        self.deliverable_completion_quiet_sec = max(0.0, float(deliverable_completion_quiet_sec or 0.0))
        self.metric_health_guard_enabled = bool(metric_health_guard_enabled)
        self.metric_health_warmup_sec = max(0.0, float(metric_health_warmup_sec or 0.0))
        self.metric_health_invalid_min_events = max(1, int(metric_health_invalid_min_events or 1))
        self.metric_health_zero_score_min_events = max(1, int(metric_health_zero_score_min_events or 1))
        self.gpu_source_hint_enabled = bool(gpu_source_hint_enabled)
        self.gpu_source_hint_mode = str(gpu_source_hint_mode or "observe").strip().lower()
        if self.gpu_source_hint_mode not in {"observe", "request", "off"}:
            self.gpu_source_hint_mode = "observe"
        self.task_resource_dir = Path(task_resource_dir) if task_resource_dir is not None else None
        self.resource_runtime: ResourceRuntime | None = None
        if resource_runtime_enabled and self.task_resource_dir is not None:
            self.resource_runtime = ResourceRuntime(
                worker_id=self.worker_id,
                resource_dir=self.task_resource_dir,
                gpu_queue=GPUQueueConfig(
                    enabled=bool(gpu_queue_enabled),
                    gpu_pool=[str(x) for x in (gpu_pool or []) if str(x).strip()],
                    default_request=int(gpu_default_request or 1),
                    max_request=int(gpu_max_request or 1),
                    assignment=str(gpu_assignment or "env_only"),
                    max_wait_sec=float(gpu_queue_max_wait_sec or 0.0),
                    heartbeat_sec=float(gpu_queue_heartbeat_sec or 15.0),
                    max_heavy_per_gpu=int(gpu_max_heavy_per_gpu or 1),
                    capacity_slots=float(gpu_capacity_slots or 1.0),
                    gpu_tt_max_per_gpu=int(gpu_tt_max_per_gpu or 1),
                    gpu_feature_max_per_gpu=int(gpu_feature_max_per_gpu or 1),
                    share_tt_with_train=bool(gpu_share_tt_with_train),
                    lease_ttl_sec=float(gpu_lease_ttl_sec or 7200.0),
                    duplicate_digest_cooldown_sec=float(gpu_duplicate_digest_cooldown_sec or 600.0),
                    duplicate_digest_threshold=int(gpu_duplicate_digest_threshold or 2),
                    admission_queue_enabled=bool(gpu_admission_queue_enabled),
                    admission_waiter_ttl_sec=float(gpu_admission_waiter_ttl_sec or 900.0),
                    pressure_yellow_hold_sec=float(gpu_pressure_yellow_hold_sec or 120.0),
                    pressure_red_to_yellow_sec=float(gpu_pressure_red_to_yellow_sec or 120.0),
                    pressure_yellow_util_pct=float(gpu_pressure_yellow_util_pct or 85.0),
                    pressure_min_free_mem_gb=float(gpu_pressure_min_free_mem_gb or 0.0),
                    pressure_yellow_free_mem_buffer_gb=float(gpu_pressure_yellow_free_mem_buffer_gb or 8.0),
                    idle_release_admission_mode=str(resource_idle_release_admission_mode or "strict_exclusive"),
                ),
            )

    def set_planning_resource_context_version(
        self,
        version: int | None,
        *,
        pressure_generation: int | None = None,
        **_: Any,
    ) -> None:
        try:
            self._planning_resource_context_version = int(version) if version is not None else None
        except (TypeError, ValueError):
            self._planning_resource_context_version = None
        try:
            self._planning_resource_pressure_generation = (
                int(pressure_generation) if pressure_generation is not None else None
            )
        except (TypeError, ValueError):
            self._planning_resource_pressure_generation = None

    def planning_resource_context_version(self, **_: Any) -> int | None:
        return self._planning_resource_context_version

    def planning_resource_pressure_generation(self, **_: Any) -> int | None:
        return self._planning_resource_pressure_generation

    def lease_assignment_enabled(self, **_: Any) -> bool:
        return bool(self.resource_runtime and self.resource_runtime.lease_assignment_enabled())

    @staticmethod
    def _digest(command: str) -> str:
        return hashlib.sha256(str(command or "").encode("utf-8", errors="replace")).hexdigest()[:16]

    @staticmethod
    def _has_gpu_intent(source_hint: ResourceSourceHint | None) -> bool:
        if bool(getattr(source_hint, "command_cpu_only", False)):
            return False
        return bool(source_hint and source_hint.has_gpu_evidence)

    @staticmethod
    def _clip_score(value: float) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score != score:
            return 0.0
        return max(0.0, min(1.0, score))


    @staticmethod
    def _command_entrypoint(command: str) -> str:
        text = str(command or "")
        match = re.search(r"\b(?:python|python3|python3\.\d+)\s+([^\s;&|]+\.py)", text)
        if match:
            return match.group(1)[:240]
        match = re.search(r"(?:open|Path)\(\s*['\"]?([^'\"\)\s]+\.py)['\"]?", text)
        if match:
            return match.group(1)[:240]
        return ""

    def _research_route_key_for_job(self, job: ResourceJob) -> str:
        explicit = normalize_route_key((job.value_hint or {}).get("route_id"))
        if explicit:
            return explicit
        entrypoint = self._command_entrypoint(job.command)
        workspace = job.workspace_dir
        if not entrypoint or workspace is None:
            return ""
        root = Path(workspace).resolve(strict=False)
        path = (root / entrypoint).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError:
            return ""
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return ""
        return f"source:{digest[:16]}"

    def _research_route_metric_evidence(self, route_key: str) -> dict[str, Any]:
        route = normalize_route_key(route_key)
        evidence = dict(self._research_route_metrics.get(route) or {})
        if self.task_resource_dir is not None and route:
            persisted = route_metric_evidence(
                self.task_resource_dir.parent / "lhr_stage_performance.csv",
                route,
            )
            if int(persisted.get("comparable_metric_count") or 0) >= int(evidence.get("comparable_metric_count") or 0):
                evidence = persisted
        return {
            "route_key": route,
            "comparable_metric_count": int(evidence.get("comparable_metric_count") or 0),
            "route_metric_proven": bool(evidence.get("route_metric_proven")),
            "latest_stage_id": str(evidence.get("latest_stage_id") or ""),
            "latest_metric_value": evidence.get("latest_metric_value"),
            "source": str(evidence.get("source") or "none"),
        }

    def _update_research_route_metric(self, job: ResourceJob, signal: dict[str, Any]) -> None:
        route_key = self._research_route_key_for_job(job)
        metric = self._current_structured_metric_for_job(job)
        if not route_key or not is_comparable_live_metric(metric, saw_final_score=bool(signal.get("saw_final_score"))):
            return
        previous = dict(self._research_route_metrics.get(route_key) or {})
        self._research_route_metrics[route_key] = {
            "route_key": route_key,
            "comparable_metric_count": max(1, int(previous.get("comparable_metric_count") or 0)),
            "route_metric_proven": True,
            "latest_stage_id": str(previous.get("latest_stage_id") or ""),
            "latest_metric_value": metric.get("value") if isinstance(metric, dict) else None,
            "source": "live_comparable_metric",
        }

    @staticmethod
    def _research_cadence_eligible(job: ResourceJob) -> bool:
        if str(job.resource_class or "") in {
            RESOURCE_HEAVY_CPU_CANDIDATE,
            RESOURCE_GPU_FEATURE_EXTRACT,
            RESOURCE_GPU_LIGHT_TRAIN,
            RESOURCE_GPU_TT_LIGHT,
            RESOURCE_HEAVY_GPU_CANDIDATE,
            RESOURCE_HEAVY_GPU_TRAIN,
            RESOURCE_UNKNOWN_GPU_EXEC,
        }:
            return True
        hint = job.source_hint or ResourceSourceHint()
        return bool(hint.has_train_evidence or hint.has_feature_evidence or hint.has_tt_evidence)

    def _research_cadence_fact_card(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        progress_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        route_key = self._research_route_key_for_job(job)
        evidence = self._research_route_metric_evidence(route_key)
        metric = self._current_structured_metric_for_job(job)
        return build_research_cadence_facts(
            enabled=self.research_cadence_enabled,
            eligible=self._research_cadence_eligible(job),
            elapsed_sec=float(progress_snapshot.get("runtime_sec") or signal.get("elapsed_sec") or 0.0),
            observe_sec=self.research_cadence_observe_sec,
            first_metric_budget_sec=self.first_comparable_metric_budget_sec,
            proven_route_metric_budget_sec=self.proven_route_metric_budget_sec,
            execution_scale=normalize_execution_scale((job.value_hint or {}).get("execution_scale")),
            route_key=route_key,
            route_evidence=evidence,
            current_comparable_metric=is_comparable_live_metric(
                metric,
                saw_final_score=bool(signal.get("saw_final_score")),
            ),
            eta_to_next_comparable_metric_sec=self._float_or_none(
                progress_snapshot.get("eta_to_next_comparable_metric_sec")
            ),
            eta_confidence=str(progress_snapshot.get("eta_confidence") or "low"),
        )

    def _infer_value_hint(
        self,
        *,
        command: str,
        resource_class: str,
        source_hint: ResourceSourceHint | None,
        timeout_sec: float,
    ) -> dict[str, Any]:
        cmd = str(command or "").lower()
        hint = source_hint or ResourceSourceHint()
        cls = str(resource_class or "")
        if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN} or hint.has_train_evidence:
            stage_type = "train"
            expected = 0.55
        elif cls == RESOURCE_GPU_FEATURE_EXTRACT or hint.has_feature_evidence:
            stage_type = "feature_extract"
            expected = 0.45
        elif cls == RESOURCE_GPU_TT_LIGHT or hint.has_tt_evidence:
            stage_type = "submission_or_inference"
            expected = 0.65
        else:
            stage_type = "unknown_gpu"
            expected = 0.35
        near_submission = 0.0
        if re.search(r"\b(submission|submit|predict|inference|infer|tta|ensemble|blend)\b", cmd):
            near_submission = 0.85
            expected = max(expected, 0.65)
        elif re.search(r"\b(valid|validation|evaluate|eval|score)\b", cmd):
            near_submission = 0.45
        elif stage_type == "train":
            near_submission = 0.20
        requested = hint.requested_gpu_count or 0
        diversity = 0.2 if requested <= 1 else 0.1
        timeout = max(0.0, float(timeout_sec or 0.0))
        long_runtime = self._clip_score(timeout / 3600.0) if timeout > 0 else (0.5 if stage_type == "train" else 0.2)
        return {
            "stage_type": stage_type,
            "expected_value_score": self._clip_score(expected),
            "lineage_diversity_score": diversity,
            "near_submission_score": self._clip_score(near_submission),
            "worker_starvation_score": 0.0,
            "duplicate_penalty": 0.0,
            "timeout_history_penalty": 0.0,
            "long_runtime_penalty": self._clip_score(long_runtime),
            "requested_gpu_count": requested,
        }

    def _refine_resource_class(
        self,
        resource_class: str,
        *,
        source_hint: ResourceSourceHint | None,
    ) -> str:
        cls = str(resource_class or "")
        hint = source_hint or ResourceSourceHint()
        if bool(getattr(hint, "command_cpu_only", False)):
            if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_FEATURE_EXTRACT, RESOURCE_GPU_LIGHT_TRAIN}:
                return RESOURCE_HEAVY_CPU_CANDIDATE
            if cls == RESOURCE_UNKNOWN_GPU_EXEC:
                return RESOURCE_HEAVY_CPU_CANDIDATE if (hint.has_train_evidence or hint.has_feature_evidence) else RESOURCE_UNKNOWN_EXEC
            if cls == RESOURCE_GPU_TT_LIGHT:
                return RESOURCE_PURE_TT_CPU
            if cls == RESOURCE_LIGHT_GPU_PROBE:
                return RESOURCE_LIGHT_CPU
        has_gpu = self._has_gpu_intent(hint)
        if cls == RESOURCE_UNKNOWN_EXEC:
            if has_gpu and hint.has_train_evidence:
                return RESOURCE_HEAVY_GPU_CANDIDATE
            if has_gpu and hint.has_feature_evidence:
                return RESOURCE_GPU_FEATURE_EXTRACT
            if has_gpu and hint.has_tt_evidence:
                return RESOURCE_GPU_TT_LIGHT
            if has_gpu:
                return RESOURCE_UNKNOWN_GPU_EXEC
            return cls
        if cls == RESOURCE_UNKNOWN_GPU_EXEC:
            if hint.has_train_evidence:
                return RESOURCE_HEAVY_GPU_CANDIDATE
            if hint.has_feature_evidence:
                return RESOURCE_GPU_FEATURE_EXTRACT
            if hint.has_tt_evidence:
                return RESOURCE_GPU_TT_LIGHT
            return cls
        if cls in {RESOURCE_LIGHT_CPU, RESOURCE_LIGHT_GPU_PROBE, RESOURCE_HEAVY_CPU_CANDIDATE, RESOURCE_PURE_TT_CPU} and has_gpu:
            if hint.has_train_evidence:
                return RESOURCE_HEAVY_GPU_CANDIDATE
            if hint.has_feature_evidence:
                return RESOURCE_GPU_FEATURE_EXTRACT
            if hint.has_tt_evidence:
                return RESOURCE_GPU_TT_LIGHT
            if getattr(hint, "command_gpu_compute_evidence", False):
                return RESOURCE_UNKNOWN_GPU_EXEC
        if cls == RESOURCE_HEAVY_GPU_CANDIDATE and has_gpu and not hint.has_train_evidence:
            if hint.has_feature_evidence:
                return RESOURCE_GPU_FEATURE_EXTRACT
            if hint.has_tt_evidence:
                return RESOURCE_GPU_TT_LIGHT
        if cls == RESOURCE_GPU_TT_LIGHT and not has_gpu:
            return RESOURCE_PURE_TT_CPU
        if cls == RESOURCE_PURE_TT_CPU and has_gpu:
            return RESOURCE_GPU_TT_LIGHT
        return cls

    def _merge_resource_class(self, current: str, incoming: str, *, source_hint: ResourceSourceHint | None) -> str:
        current_cls = self._refine_resource_class(current, source_hint=source_hint)
        incoming_cls = self._refine_resource_class(incoming, source_hint=source_hint)
        if not incoming_cls:
            return current_cls
        if current_cls in {RESOURCE_GPU_TT_LIGHT, RESOURCE_GPU_FEATURE_EXTRACT, RESOURCE_PURE_TT_CPU} and incoming_cls in TRAIN_CLASSES:
            hint = source_hint or ResourceSourceHint()
            return incoming_cls if hint.has_train_evidence else current_cls
        if current_cls == RESOURCE_UNKNOWN_EXEC:
            return incoming_cls
        if current_cls == RESOURCE_UNKNOWN_GPU_EXEC and incoming_cls != RESOURCE_UNKNOWN_EXEC:
            return incoming_cls
        return current_cls or incoming_cls

    def _gpu_queue_relevant(self, resource_class: str, source_hint: ResourceSourceHint | None, gpu_ids: list[str]) -> bool:
        if self.resource_runtime is None:
            return False
        cls = str(resource_class or "")
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        queue_classes = {
            RESOURCE_HEAVY_GPU_CANDIDATE,
            RESOURCE_HEAVY_GPU_TRAIN,
            RESOURCE_GPU_FEATURE_EXTRACT,
            RESOURCE_GPU_LIGHT_TRAIN,
            RESOURCE_GPU_TT_LIGHT,
            RESOURCE_UNKNOWN_GPU_EXEC,
        }
        if ids and cls in queue_classes:
            if self._has_gpu_intent(source_hint):
                return self.resource_runtime.should_queue_gpu(resource_class=cls, gpu_ids=ids)
            hint = source_hint or ResourceSourceHint()
            has_entrypoint = bool(hint.entrypoints)
            has_stage_hint = bool(hint.has_train_evidence or hint.has_feature_evidence or hint.has_tt_evidence)
            source_checked = int(getattr(hint, "source_files_inspected", 0) or 0) > 0
            if has_entrypoint and has_stage_hint and (not source_checked or hint.has_feature_evidence or hint.has_tt_evidence):
                return self.resource_runtime.should_queue_gpu(resource_class=cls, gpu_ids=ids)
            return False
        if not self._has_gpu_intent(source_hint):
            return False
        return self.resource_runtime.should_queue_gpu(resource_class=cls, gpu_ids=ids)

    def _emit(self, event: str, job: ResourceJob, *, status: str = "", payload: dict[str, Any] | None = None) -> None:
        body = {
            "job_id": job.job_id,
            "command_digest": job.command_digest,
            "resource_class": job.resource_class,
            "gpu_ids": job.gpu_ids,
            "cpu_set": job.cpu_set,
            "timeout_sec": job.timeout_sec,
            **(payload or {}),
        }
        self.state_machine.append_event(
            event,
            task_type="resource_job",
            task_id=f"resource_job:{job.job_id}",
            status=status,
            payload=body,
        )

    def _resource_feedback_text(
        self,
        *,
        status: str,
        reason: str,
        scope: str,
        resource_mode: str,
        blocked_class: str,
        gpu_ids: list[str] | None = None,
        allowed_classes: list[str] | None = None,
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
        return resource_feedback_text(
            status=status,
            reason=reason,
            scope=scope,
            resource_mode=resource_mode,
            blocked_class=blocked_class,
            gpu_ids=gpu_ids,
            allowed_classes=allowed_classes,
            holder_job_id=holder_job_id,
            queue_position=queue_position,
            queue_len=queue_len,
            cooldown_sec=cooldown_sec,
            eta_next_train_sec=eta_next_train_sec,
            eta_confidence=eta_confidence,
            pressure_generation=pressure_generation,
            duplicate_digest_count=duplicate_digest_count,
            unlock_condition=unlock_condition,
            blocked_until_unlock=blocked_until_unlock,
            schema_state=schema_state,
            artifact_state=artifact_state,
            progress_state=progress_state,
            extra_facts=extra_facts,
        )

    @staticmethod
    def _resource_class_token(value: Any) -> str:
        return re.sub(r"[^a-z0-9_.:-]+", "_", str(value or "").strip().lower()).strip("_")

    @staticmethod
    def _safe_candidate_artifact(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts:
            return ""
        return path.as_posix()

    @classmethod
    def _resource_class_in_allowed(cls, resource_class: str, allowed_classes: list[str] | None) -> bool:
        blocked = cls._resource_class_token(resource_class)
        if not blocked:
            return False
        allowed = {cls._resource_class_token(value) for value in (allowed_classes or []) if str(value).strip()}
        return blocked in allowed

    @staticmethod
    def _hard_safety_gate_reason(reason: str) -> bool:
        return str(reason or "").strip().lower() in {
            "duplicate_digest_cooldown",
            "invalid_deliverable_schema_preflight",
            "task_gpu_boundary_preflight_violation",
            "workspace_gpu_boundary_violation",
            "boundary_violation",
        }

    def _allowed_class_bypass_result(
        self,
        job: ResourceJob,
        *,
        reason: str,
        scope: str,
        resource_mode: str,
        allowed_classes: list[str] | None,
    ) -> dict[str, Any] | None:
        if self._hard_safety_gate_reason(reason):
            return None
        if not self._resource_class_in_allowed(job.resource_class, allowed_classes):
            return None
        allowed = [str(x) for x in (allowed_classes or []) if str(x).strip()]
        self._emit(
            "resource_allowed_class_bypass",
            job,
            status="allowed",
            payload={
                "reason": "allowed_class_bypass",
                "bypassed_reason": str(reason or ""),
                "scope": str(scope or ""),
                "resource_mode": str(resource_mode or ""),
                "resource_class": job.resource_class,
                "allowed_classes": allowed,
            },
        )
        return {
            "allowed": True,
            "resource_class": job.resource_class,
            "reason": "allowed_class_bypass",
            "bypassed_reason": str(reason or ""),
            "allowed_classes": allowed,
        }

    def _soft_gate_admission_enabled(self, *, reason: str, source_reason: str = "") -> bool:
        if self._hard_safety_gate_reason(reason) or self._hard_safety_gate_reason(source_reason):
            return False
        # Monitor-first startup owns soft resource blocks. Keep the older
        # startup-time admission review only for conservative/backward-compatible
        # runs so soft gates do not become a second pre-run decision layer.
        if self._startup_trial_enabled():
            return False
        return bool(
            self.admission_llm_enabled
            and self.admission_llm_mode != "off"
            and self.admission_decider is not None
        )

    def _soft_gate_admission_review_result(
        self,
        job: ResourceJob,
        *,
        status: str,
        reason: str,
        scope: str,
        resource_mode: str,
        allowed_classes: list[str] | None = None,
        feedback: str = "",
        feedback_state: dict[str, Any] | None = None,
        eta_next_train_sec: float | None = None,
        eta_confidence: str = "low",
        cooldown_sec: float | None = None,
        retry_after_sec: float | None = None,
        unlock_condition: str = "resource_context_changed",
        blocked_until_unlock: bool = True,
        extra_facts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        gpu_ids = [str(x) for x in (job.gpu_ids or []) if str(x).strip()]
        allowed = [str(x) for x in (allowed_classes or []) if str(x).strip()]
        soft_gate = {
            "reason": str(reason or "resource_policy_gate"),
            "scope": str(scope or "task"),
            "resource_mode": str(resource_mode or "YELLOW"),
            "allowed_classes": allowed,
            "cooldown_sec": cooldown_sec,
            "retry_after_sec": retry_after_sec,
            "unlock_condition": str(unlock_condition or ""),
            "blocked_until_unlock": bool(blocked_until_unlock),
        }
        if isinstance(extra_facts, dict):
            soft_gate.update(extra_facts)
        result: dict[str, Any] = {
            "allowed": False,
            "enabled": True,
            "acquired": False,
            "requires_admission_review": True,
            "soft_gate_reason": str(reason or "resource_policy_gate"),
            "soft_gate_scope": str(scope or "task"),
            "soft_gate": soft_gate,
            "resource_class": job.resource_class,
            "policy_resource_class": job.resource_class,
            "status": str(status or "DENIED_REPLAN"),
            "admission_action": "PENDING",
            "reason": str(reason or "resource_policy_gate"),
            "resource_mode": str(resource_mode or "YELLOW"),
            "gpu_ids": gpu_ids,
            "allowed_classes": allowed,
            "requested_gpu_count": job.gpu_request_count,
            "eta_next_train_sec": eta_next_train_sec,
            "eta_confidence": str(eta_confidence or "low"),
            "retry_after_sec": retry_after_sec,
            "blocked_until_unlock": bool(blocked_until_unlock),
            "unlock_condition": str(unlock_condition or ""),
            "feedback": str(feedback or ""),
            "details": {"soft_gate": soft_gate},
        }
        if isinstance(feedback_state, dict):
            result.update({
                "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
                "feedback_state_key": feedback_state.get("feedback_state_key"),
                "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
            })
        if self.resource_runtime is not None and job.gpu_queue_relevant:
            opportunity = self.resource_runtime.admission_opportunity_facts(
                resource_class=job.resource_class,
                gpu_ids=gpu_ids,
                request_count=job.gpu_request_count,
            )
            result.update({
                "admission_opportunity": opportunity,
                "lease_grantable_by_llm": bool(opportunity.get("lease_grantable_by_llm")),
                "candidate_physical_gpus": gpu_ids,
                "allowed_physical_gpus": gpu_ids,
                "assigned_physical_gpus": [],
            })
        return result

    @staticmethod
    def _resource_feedback_state_key(
        *,
        status: str,
        reason: str,
        scope: str,
        resource_mode: str,
        blocked_class: str,
        gpu_ids: list[str] | None = None,
        allowed_classes: list[str] | None = None,
        unlock_condition: str = "",
        blocked_until_unlock: bool | None = None,
        schema_state: str = "",
        artifact_state: str = "",
        progress_state: str = "",
        deliverable_validity: str = "",
    ) -> str:
        scope_key = str(scope or "task").strip().lower()
        status_key = str(status or "DENIED_REPLAN").strip().upper()
        reason_key = str(reason or "resource_policy_gate").strip().lower()
        resource_wait_status = status_key in {"PENDING", "REPLAN", "DEFERRED", "DENIED_REPLAN", "DENIED_DUPLICATE"}
        if resource_wait_status and reason_key not in {
            "invalid_deliverable_schema_preflight",
            "task_gpu_boundary_preflight_violation",
            "workspace_gpu_boundary_violation",
            "boundary_violation",
            "duplicate_digest_cooldown",
        }:
            if "duplicate" in reason_key:
                reason_key = "duplicate_resource_wait"
            elif "schema" in reason_key:
                reason_key = "schema_resource_wait"
            elif "gpu" in reason_key or "slot" in reason_key or "lease" in reason_key or "yellow" in reason_key or "capacity" in reason_key or "class" in reason_key:
                reason_key = "gpu_resource_wait"
            else:
                reason_key = "resource_wait"
        collapse_blocked_class = scope_key in {"worker_deliverable", "worker_plan"} or reason_key in {
            "completed_deliverable_heavy_command_block",
            "invalid_deliverable_schema_preflight",
            "post_feedback_blocked_class",
            "active_resource_plan_guard",
        }
        blocked_key = "resource_work" if collapse_blocked_class else str(blocked_class or "unknown").strip().lower()
        gpu_key = "" if scope_key == "worker_deliverable" else ",".join(sorted({str(x).strip() for x in (gpu_ids or []) if str(x).strip()}))
        allowed_key = ",".join(sorted({str(x).strip().lower() for x in (allowed_classes or []) if str(x).strip()}))
        blocked_until = "" if blocked_until_unlock is None else str(bool(blocked_until_unlock)).lower()
        parts = [
            status_key,
            reason_key,
            scope_key,
            str(resource_mode or "UNKNOWN").strip().upper(),
            blocked_key,
            gpu_key,
            allowed_key,
            str(unlock_condition or "").strip().lower(),
            blocked_until,
            str(schema_state or "").strip().lower(),
            str(artifact_state or "").strip().lower(),
            str(progress_state or "").strip().lower(),
            str(deliverable_validity or "").strip().lower(),
        ]
        return "|".join(parts)

    def _dedupe_resource_feedback_for_agent(
        self,
        job: ResourceJob,
        *,
        feedback: str,
        status: str,
        reason: str,
        scope: str,
        resource_mode: str,
        blocked_class: str,
        gpu_ids: list[str] | None = None,
        allowed_classes: list[str] | None = None,
        unlock_condition: str = "",
        blocked_until_unlock: bool | None = None,
        schema_state: str = "",
        artifact_state: str = "",
        progress_state: str = "",
        deliverable_validity: str = "",
    ) -> dict[str, Any]:
        key = self._resource_feedback_state_key(
            status=status,
            reason=reason,
            scope=scope,
            resource_mode=resource_mode,
            blocked_class=blocked_class,
            gpu_ids=gpu_ids,
            allowed_classes=allowed_classes,
            unlock_condition=unlock_condition,
            blocked_until_unlock=blocked_until_unlock,
            schema_state=schema_state,
            artifact_state=artifact_state,
            progress_state=progress_state,
            deliverable_validity=deliverable_validity,
        )
        now = time.time()
        existing = self._reported_resource_feedback.get(key)
        if isinstance(existing, dict):
            repeated_count = int(existing.get("repeated_count") or 1) + 1
            existing.update(
                {
                    "repeated_count": repeated_count,
                    "last_seen_at": now,
                    "last_job_id": job.job_id,
                    "last_command_digest": job.command_digest,
                    "last_blocked_class": blocked_class,
                },
            )
            self._emit(
                "resource_feedback_suppressed",
                job,
                status="suppressed",
                payload={
                    "feedback_state_key": key,
                    "reason": reason,
                    "scope": scope,
                    "resource_mode": resource_mode,
                    "blocked_class": blocked_class,
                    "first_job_id": existing.get("first_job_id"),
                    "repeated_count": repeated_count,
                    "suppression_reason": "same resource feedback state already reported to main agent",
                },
            )
            return {
                "feedback": "",
                "feedback_suppressed": True,
                "feedback_state_key": key,
                "repeated_count": repeated_count,
            }
        self._reported_resource_feedback[key] = {
            "first_seen_at": now,
            "last_seen_at": now,
            "first_job_id": job.job_id,
            "last_job_id": job.job_id,
            "first_command_digest": job.command_digest,
            "last_command_digest": job.command_digest,
            "first_blocked_class": blocked_class,
            "last_blocked_class": blocked_class,
            "repeated_count": 1,
        }
        return {
            "feedback": feedback,
            "feedback_suppressed": False,
            "feedback_state_key": key,
            "repeated_count": 1,
        }


    def _current_pressure_generation(self) -> int:
        if self.resource_runtime is None:
            return 0
        try:
            return int(self.resource_runtime.pressure_generation() or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _resource_class_matches_guard(resource_class: str, blocked_class: str) -> bool:
        cls = str(resource_class or "")
        blocked = str(blocked_class or "")
        heavy = {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}
        if cls in heavy and blocked in heavy:
            return True
        return bool(cls and blocked and cls == blocked)

    def _light_train_share_deferral_enabled(self, resource_class: str, gpu_ids: list[str] | None) -> bool:
        return bool(
            str(resource_class or "") == RESOURCE_GPU_LIGHT_TRAIN
            and [str(x) for x in (gpu_ids or []) if str(x).strip()]
            and self.gpu_share_config.grant_active
            and self.gpu_share_config.phase in {"feature_share", "light_train"}
            and self.arbiter_enabled
            and self.arbiter_mode == "llm"
        )

    @staticmethod
    def _feedback_guard_is_hard_constraint(guard: dict[str, Any] | None) -> bool:
        return feedback_guard_is_hard_constraint(guard)

    def _can_defer_light_train_guard_to_arbiter(
        self,
        *,
        resource_class: str,
        gpu_ids: list[str] | None,
        guard: dict[str, Any] | None,
    ) -> bool:
        if not self._light_train_share_deferral_enabled(resource_class, gpu_ids):
            return False
        return not self._feedback_guard_is_hard_constraint(guard)

    def _gpu_ids_have_active_store_pressure(self, gpu_ids: list[str] | None) -> bool:
        if self.resource_runtime is None:
            return True
        ids = normalize_gpu_ids(gpu_ids)
        if not ids:
            return True
        try:
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception:
            return True
        return gpu_store_snapshot_has_pressure(snapshot, ids, include_waiters=True)

    def _gpu_ids_have_active_store_lease(self, gpu_ids: list[str] | None) -> bool:
        if self.resource_runtime is None:
            return True
        ids = normalize_gpu_ids(gpu_ids)
        if not ids:
            return True
        try:
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception:
            return True
        return gpu_store_snapshot_has_lease(snapshot, ids)

    def _soft_gpu_gate_can_clear_for_available_slot(
        self,
        *,
        resource_class: str,
        gpu_ids: list[str] | None,
        guard: dict[str, Any] | None,
    ) -> bool:
        if not self.control_profile.clear_stale_soft_gpu_gate_on_free_gpu:
            return False
        if self._feedback_guard_is_hard_constraint(guard):
            return False
        if not resource_class_can_use_soft_gpu_gate(resource_class):
            return False
        ids = normalize_gpu_ids(gpu_ids)
        if not ids or self.resource_runtime is None:
            return False
        reconcile = self.resource_runtime.reconcile_free_gpu_pressure(
            gpu_ids=ids,
            reason="observed_free_before_soft_gpu_gate",
        )
        if reconcile_free_gpu_pressure_blocked(reconcile):
            return False
        return not self._gpu_ids_have_active_store_lease(ids)

    def _can_ignore_soft_gpu_guard_for_available_slot(
        self,
        *,
        resource_class: str,
        gpu_ids: list[str] | None,
        guard: dict[str, Any] | None,
    ) -> bool:
        if self._feedback_guard_is_hard_constraint(guard):
            return False
        if not resource_class_can_use_soft_gpu_gate(resource_class):
            return False
        if self.control_profile.ignore_waiters_for_soft_gpu_gate:
            return self._soft_gpu_gate_can_clear_for_available_slot(
                resource_class=resource_class,
                gpu_ids=gpu_ids,
                guard=guard,
            )
        return not self._gpu_ids_have_active_store_pressure(gpu_ids)

    def _install_plan_guard(
        self,
        feedback: dict[str, Any],
        *,
        eta_next_train_sec: float | None = None,
        cooldown_sec: float | None = None,
    ) -> None:
        blocked = str(feedback.get("blocked_class") or "").strip()
        status = str(feedback.get("status") or "").upper()
        if not blocked or status not in {"PENDING", "REPLAN", "DEFERRED", "DENIED_REPLAN", "DENIED_DUPLICATE"}:
            return
        try:
            eta = float(eta_next_train_sec or 0.0)
        except (TypeError, ValueError):
            eta = 0.0
        try:
            cooldown = float(cooldown_sec or 0.0)
        except (TypeError, ValueError):
            cooldown = 0.0
        duration = max(eta, cooldown)
        if duration <= 0.0:
            duration = 300.0 if status in {"PENDING", "DEFERRED"} else 120.0
        duration = max(30.0, min(duration, 900.0))
        guard_id = str(feedback.get("feedback_id") or f"guard-{self.worker_id}-{self._feedback_seq:05d}")
        self._active_plan_guards[guard_id] = {
            **feedback,
            "guard_id": guard_id,
            "expires_at": time.time() + duration,
            "guard_duration_sec": duration,
            "pressure_generation": self._current_pressure_generation(),
        }

    def _active_plan_guard_decision(self, job: ResourceJob) -> dict[str, Any]:
        if not self._active_plan_guards:
            return {"blocked": False}
        if not (job.gpu_queue_relevant or self._has_gpu_intent(job.source_hint)):
            return {"blocked": False}
        now = time.time()
        current_generation = self._current_pressure_generation()
        job_gpu_ids = {str(x) for x in (job.gpu_ids or []) if str(x).strip()}
        for guard_id, raw in list(self._active_plan_guards.items()):
            if not isinstance(raw, dict):
                self._active_plan_guards.pop(guard_id, None)
                continue
            expires_at = float(raw.get("expires_at") or 0.0)
            guard_generation = int(raw.get("pressure_generation") or 0)
            if expires_at <= now or (current_generation > guard_generation and guard_generation >= 0):
                self._active_plan_guards.pop(guard_id, None)
                continue
            if not self._resource_class_matches_guard(job.resource_class, str(raw.get("blocked_class") or "")):
                continue
            if self._can_defer_light_train_guard_to_arbiter(
                resource_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                guard=raw,
            ):
                continue
            if self._can_ignore_soft_gpu_guard_for_available_slot(
                resource_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                guard=raw,
            ):
                if self.control_profile.clear_stale_soft_gpu_gate_on_free_gpu:
                    self._active_plan_guards.pop(guard_id, None)
                continue
            guard_gpu_ids = {str(x) for x in (raw.get("gpu_ids") or []) if str(x).strip()}
            if guard_gpu_ids and job_gpu_ids and not (guard_gpu_ids & job_gpu_ids):
                continue
            return {
                "blocked": True,
                "guard": raw,
                "guard_id": guard_id,
                "current_pressure_generation": current_generation,
                "remaining_sec": max(0.0, expires_at - now),
            }
        return {"blocked": False}

    def _remember_resource_feedback(
        self,
        job: ResourceJob,
        *,
        status: str,
        reason: str,
        resource_mode: str,
        blocked_class: str,
        allowed_classes: list[str] | None = None,
        eta_next_train_sec: float | None = None,
        cooldown_sec: float | None = None,
        retry_after_sec: float | None = None,
        post_feedback_action: str = "",
    ) -> None:
        self._feedback_seq += 1
        feedback = {
            "feedback_id": f"rf-{self.worker_id}-{self._feedback_seq:05d}",
            "status": str(status or ""),
            "reason": str(reason or ""),
            "resource_mode": str(resource_mode or ""),
            "blocked_class": str(blocked_class or ""),
            "allowed_classes": [str(x) for x in (allowed_classes or []) if str(x).strip()],
            "job_id": job.job_id,
            "command_digest": job.command_digest,
            "resource_class": job.resource_class,
            "gpu_ids": list(job.gpu_ids or []),
            "created_at": time.time(),
            "pressure_generation": self._current_pressure_generation(),
        }
        if retry_after_sec is not None:
            feedback["retry_after_sec"] = max(0.0, float(retry_after_sec or 0.0))
        if post_feedback_action:
            feedback["post_feedback_action"] = str(post_feedback_action or "")
        self._pending_resource_feedback = feedback
        self._install_plan_guard(
            feedback,
            eta_next_train_sec=eta_next_train_sec,
            cooldown_sec=cooldown_sec,
        )

    @staticmethod
    def _is_simple_sleep_command(command: str) -> bool:
        return bool(re.match(r"^\s*sleep\s+\d+(?:\.\d+)?[smhd]?\s*$", str(command or ""), re.IGNORECASE))

    @staticmethod
    def _is_inline_python_command(command: str) -> bool:
        return bool(re.search(r"\b(?:python|python3(?:\.\d+)?)\s+(?:-u\s+)?-c\b", str(command or "")))

    def _post_feedback_action_kind(
        self,
        *,
        command: str,
        resource_class: str,
        source_hint: ResourceSourceHint | None = None,
    ) -> str:
        cls = str(resource_class or "")
        cmd = str(command or "").strip().lower()
        if self._is_simple_sleep_command(command):
            return "sleep_backoff"
        if bool(getattr(source_hint, "command_cpu_only", False)) and cls in {
            RESOURCE_HEAVY_GPU_CANDIDATE,
            RESOURCE_HEAVY_GPU_TRAIN,
            RESOURCE_GPU_FEATURE_EXTRACT,
            RESOURCE_GPU_LIGHT_TRAIN,
            RESOURCE_GPU_TT_LIGHT,
            RESOURCE_UNKNOWN_GPU_EXEC,
            RESOURCE_HEAVY_CPU_CANDIDATE,
        }:
            return "cpu_support"
        if (
            cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}
            and self._is_inline_python_command(command)
            and not self._has_gpu_intent(source_hint)
            and not bool((source_hint or ResourceSourceHint()).entrypoints)
        ):
            return "cpu_support"
        if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
            return "gpu_train"
        if cls == RESOURCE_UNKNOWN_GPU_EXEC:
            return "unknown_gpu"
        if cls == RESOURCE_GPU_FEATURE_EXTRACT:
            return "gpu_feature_extract"
        if cls in {RESOURCE_GPU_TT_LIGHT, RESOURCE_PURE_TT_CPU}:
            return "tt_or_submission"
        if re.search(r"\b(submission|submit|predict|inference|infer|tta|ensemble|blend)\b", cmd):
            return "tt_or_submission"
        if re.match(r"^(ls|find|cat|head|tail|wc|du|df|pwd|python\s+-c\s+['\"]?print\()\b", cmd):
            return "readonly_cpu"
        return "cpu_support"

    def _emit_post_feedback_action(
        self,
        *,
        command: str,
        resource_class: str,
        command_digest: str,
        gpu_ids: list[str] | None = None,
        source_hint: ResourceSourceHint | None = None,
    ) -> dict[str, Any]:
        pending = self._pending_resource_feedback
        if not isinstance(pending, dict):
            return {}
        try:
            pending_generation = int(pending.get("pressure_generation") or 0)
        except (TypeError, ValueError):
            pending_generation = 0
        if self._current_pressure_generation() > pending_generation:
            self._pending_resource_feedback = None
            return {}
        self._pending_resource_feedback = None
        action = self._post_feedback_action_kind(command=command, resource_class=resource_class, source_hint=source_hint)
        cls = str(resource_class or "")
        action_decoration = decorate_post_feedback_action_after_backoff(
            action=action,
            pending_feedback=pending,
            last_backoff=self._last_agent_backoff_wait,
        )
        action = str(action_decoration.get("action") or action)
        blocked_class = str(pending.get("blocked_class") or "")
        mode = str(pending.get("resource_mode") or "").upper()
        allowed = [str(x) for x in (pending.get("allowed_classes") or []) if str(x).strip()]
        allowed_current_class = self._resource_class_in_allowed(cls, allowed)
        same_digest = bool(command_digest and command_digest == str(pending.get("command_digest") or ""))
        violates = False
        if same_digest and str(pending.get("status") or "").upper() == "DENIED_DUPLICATE":
            violates = True
        if blocked_class and cls == blocked_class and action not in {"sleep_backoff", "readonly_cpu", "cpu_support", "tt_or_submission"}:
            if not (
                self._can_defer_light_train_guard_to_arbiter(
                    resource_class=cls,
                    gpu_ids=gpu_ids,
                    guard=pending,
                )
                or self._can_ignore_soft_gpu_guard_for_available_slot(
                    resource_class=cls,
                    gpu_ids=gpu_ids,
                    guard=pending,
                )
            ):
                violates = True
        if mode == "RED" and cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}:
            if not (
                self._can_defer_light_train_guard_to_arbiter(
                    resource_class=cls,
                    gpu_ids=gpu_ids,
                    guard=pending,
                )
                or self._can_ignore_soft_gpu_guard_for_available_slot(
                    resource_class=cls,
                    gpu_ids=gpu_ids,
                    guard=pending,
                )
            ):
                violates = True
        if allowed and cls in {RESOURCE_GPU_TT_LIGHT, RESOURCE_PURE_TT_CPU, RESOURCE_GPU_FEATURE_EXTRACT} and cls not in allowed:
            violates = True
        hard_safety_reason = self._hard_safety_gate_reason(str(pending.get("reason") or ""))
        hard_safety_status = str(pending.get("status") or "").upper() == "DENIED_DUPLICATE"
        if allowed_current_class and not (hard_safety_reason or hard_safety_status):
            violates = False
        created_at = float(pending.get("created_at") or time.time())
        payload = {
            "feedback_id": str(pending.get("feedback_id") or ""),
            "source_job_id": str(pending.get("job_id") or ""),
            "source_status": str(pending.get("status") or ""),
            "source_reason": str(pending.get("reason") or ""),
            "source_resource_mode": str(pending.get("resource_mode") or ""),
            "source_gpu_ids": [str(x) for x in (pending.get("gpu_ids") or []) if str(x).strip()],
            "blocked_class": blocked_class,
            "allowed_classes": allowed,
            "action": action,
            "base_action": str(action_decoration.get("base_action") or action),
            "preceded_by_agent_backoff": bool(action_decoration.get("preceded_by_agent_backoff")),
            "agent_backoff_wait_id": str(action_decoration.get("agent_backoff_wait_id") or ""),
            "agent_backoff_elapsed_sec": float(action_decoration.get("agent_backoff_elapsed_sec") or 0.0),
            "agent_backoff_wake_reason": str(action_decoration.get("agent_backoff_wake_reason") or ""),
            "current_resource_class": cls,
            "current_command_digest": str(command_digest or ""),
            "same_command_digest": same_digest,
            "violates_feedback": violates,
            "elapsed_since_feedback_sec": max(0.0, time.time() - created_at),
            "retry_after_sec": pending.get("retry_after_sec"),
            "post_feedback_action_hint": str(pending.get("post_feedback_action") or ""),
        }
        self.state_machine.append_event(
            "post_feedback_action",
            task_type="resource_feedback",
            task_id=f"post_feedback:{pending.get('feedback_id') or self._feedback_seq}",
            status="violated" if violates else "observed",
            payload=payload,
        )
        return payload

    def _promote(self, job: ResourceJob, *, reason: str, elapsed_sec: float = 0.0) -> None:
        if job.visible:
            return
        job.visible = True
        self._emit(
            "resource_job_started",
            job,
            status="running",
            payload={"reason": reason, "elapsed_sec": float(elapsed_sec)},
        )
        if job.pid is not None:
            self._emit(
                "resource_lease_registered",
                job,
                status="running",
                payload={"pid": job.pid},
            )

    def job_created(
        self,
        *,
        command: str,
        inferred_class: str,
        gpu_ids: list[str] | None = None,
        cpu_set: str | None = None,
        timeout_sec: float | int | None = None,
        workspace_dir: str | Path | None = None,
        classifier_reason: str | None = None,
        value_hint_override: dict[str, Any] | None = None,
        candidate_artifact: str = "",
    ) -> str | None:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        source_hint = ResourceSourceHint()
        if self.gpu_source_hint_enabled and self.gpu_source_hint_mode != "off":
            source_hint = detect_resource_source_hint(
                command=command,
                workspace_dir=Path(workspace_dir) if workspace_dir is not None else None,
            )
        if bool(getattr(source_hint, "command_cpu_only", False)):
            ids = []
        resource_class = self._refine_resource_class(str(inferred_class or ""), source_hint=source_hint)
        command_digest = self._digest(command)
        post_feedback_gate = self._emit_post_feedback_action(
            command=command,
            resource_class=resource_class,
            command_digest=command_digest,
            gpu_ids=ids,
            source_hint=source_hint,
        )
        if resource_class not in TRACKABLE_CLASSES and not ids:
            if not self.bash_monitor_all_enabled:
                return None
            resource_class = RESOURCE_LIGHT_CPU
        value_hint = self._infer_value_hint(
            command=command,
            resource_class=resource_class,
            source_hint=source_hint,
            timeout_sec=float(timeout_sec or 0.0),
        )
        if isinstance(value_hint_override, dict):
            clean_hint: dict[str, Any] = {}
            for key, raw in value_hint_override.items():
                if key in {
                    "expected_value_score",
                    "lineage_diversity_score",
                    "near_submission_score",
                    "worker_starvation_score",
                    "duplicate_penalty",
                    "timeout_history_penalty",
                    "long_runtime_penalty",
                }:
                    clean_hint[str(key)] = self._clip_score(raw)
                elif key in {"stage_type", "reason", "route_id", "execution_scale"}:
                    clean_hint[str(key)] = str(raw)[:160]
                elif key == "requested_gpu_count":
                    try:
                        clean_hint[str(key)] = max(0, int(raw))
                    except (TypeError, ValueError):
                        pass
            value_hint.update(clean_hint)
        quick_probe = classify_quick_probe_command(
            command,
            expected_runtime_sec=self.quick_probe_expected_runtime_sec,
            hard_review_sec=self.quick_probe_hard_review_sec,
            small_scope_threshold=self.quick_probe_small_scope_threshold,
            workspace_dir=workspace_dir,
        ).to_json()
        gpu_queue_relevant = self._gpu_queue_relevant(resource_class, source_hint, ids)
        request_count = source_hint.requested_gpu_count
        if self.gpu_source_hint_mode == "request" and source_hint.source_gpu_request:
            request_count = max(request_count or 0, source_hint.source_gpu_request)
        self._seq += 1
        job = ResourceJob(
            job_id=f"{self.worker_id}:bash:{self._seq:05d}",
            command=str(command or ""),
            command_digest=command_digest,
            resource_class=resource_class,
            gpu_ids=ids,
            cpu_set=str(cpu_set or ""),
            timeout_sec=float(timeout_sec or 0.0),
            created_at=time.time(),
            workspace_dir=Path(workspace_dir).resolve(strict=False) if workspace_dir is not None else None,
            candidate_artifact=self._safe_candidate_artifact(candidate_artifact),
            gpu_queue_relevant=gpu_queue_relevant,
            gpu_request_count=request_count,
            source_hint=source_hint,
            value_hint=value_hint,
            post_feedback_gate=(post_feedback_gate if post_feedback_gate.get("violates_feedback") else {}),
            quick_probe=quick_probe if quick_probe.get("quick_probe_candidate") else {},
        )
        self._jobs[job.job_id] = job
        self._review_states[job.job_id] = new_review_state(job.job_id)
        if source_hint.hint_labels:
            self._emit(
                "resource_source_hint_detected",
                job,
                status="observed",
                payload={
                    **source_hint.to_event_payload(),
                    "classifier_reason": str(classifier_reason or ""),
                    "source_hint_mode": self.gpu_source_hint_mode,
                },
            )
        if quick_probe.get("quick_probe_candidate"):
            self._emit(
                "resource_quick_probe_detected",
                job,
                status="observed",
                payload={
                    **dict(quick_probe),
                    "classifier_reason": str(classifier_reason or ""),
                    "resource_class": resource_class,
                },
            )
        if gpu_queue_relevant:
            self._emit(
                "resource_request_created",
                job,
                status="created",
                payload={
                    "gpu_request": int(request_count or len(ids) or 1),
                    "queue_enabled": True,
                    "source_hint_mode": self.gpu_source_hint_mode,
                    "classifier_reason": str(classifier_reason or ""),
                    "value_hint": dict(value_hint),
                },
            )
        return job.job_id


    def _startup_trial_enabled(self) -> bool:
        return startup_policy_trial_first(self.resource_startup_policy)

    def _startup_trial_decision(
        self,
        job: ResourceJob,
        *,
        reason: str,
        status: str,
        scope: str,
        resource_mode: str,
        allowed_classes: list[str] | None = None,
        gpu_ids: list[str] | None = None,
        source_reason: str = "",
        extra_facts: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self._startup_trial_enabled() or not job.gpu_queue_relevant or self.resource_runtime is None:
            return None
        cls = classify_preflight_block(
            reason=reason,
            source_reason=source_reason,
            resource_class=job.resource_class,
        )
        if cls.block_class != "soft_trial":
            return None
        now = time.time()
        ids = [str(x) for x in (gpu_ids or job.gpu_ids or []) if str(x).strip()]
        state = {
            "active": True,
            "source": "monitor_first_startup_trial",
            "reason": str(reason or "soft_resource_block"),
            "source_reason": str(source_reason or ""),
            "rule_status": str(status or "DENIED_REPLAN"),
            "startup_block_class": cls.block_class,
            "started_at": now,
            "trial": True,
            "trial_window_sec": self.resource_trial_window_sec,
            "trial_hard_review_sec": self.resource_trial_hard_review_sec,
            "observe_window_sec": self.resource_trial_window_sec,
            "observe_max_sec": self.resource_trial_hard_review_sec,
            "resource_mode": str(resource_mode or "YELLOW"),
            "scope": str(scope or "worker_plan"),
            "gpu_ids": ids,
            "allowed_classes": list(allowed_classes or []),
        }
        if extra_facts:
            state["extra_facts"] = dict(extra_facts)
        job.observe_first_state = state
        payload = {
            **state,
            "blocked_class": job.resource_class,
            "resource_startup_policy": self.resource_startup_policy,
        }
        self._emit("resource_startup_trial_candidate", job, status="trial", payload=payload)
        self._emit("resource_policy_gate", job, status="trial", payload=payload)
        self.resource_runtime.record_resource_event(
            "startup_trial_candidate",
            payload={"job_id": job.job_id, "resource_type": "gpu", "result": payload},
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        return {
            "allowed": True,
            "resource_class": job.resource_class,
            "status": "OBSERVE_THEN_RUN",
            "admission_action": "OBSERVE_THEN_RUN",
            "reason": str(reason or "soft_resource_block"),
            "startup_block_class": cls.block_class,
            "resource_startup_policy": self.resource_startup_policy,
            "trial": True,
            "trial_reason": str(reason or "soft_resource_block"),
            "trial_window_sec": self.resource_trial_window_sec,
            "trial_hard_review_sec": self.resource_trial_hard_review_sec,
            "feedback_suppressed": True,
        }


    def _pressure_observe_first_eligible(
        self,
        job: ResourceJob,
        *,
        gate: str,
        pressure: dict[str, Any],
        context_is_stale: bool,
    ) -> bool:
        if not (self.stale_pressure_observe_first_enabled and job.gpu_queue_relevant and self.resource_runtime is not None):
            return False
        if str(gate or "") == "duplicate_digest_cooldown":
            return False
        try:
            timeout_count = int(pressure.get("max_queue_timeout_count") or 0)
        except (TypeError, ValueError):
            timeout_count = 0
        if timeout_count > 0:
            return False
        soft_pressure_reasons = {
            "stale_resource_context",
            "block_train_after_gpu_pressure",
            "yellow_pressure_exit_guard",
            "gpu_pressure_yellow_or_red",
        }
        if not (context_is_stale or str(gate or "") in soft_pressure_reasons):
            return False
        if job.resource_class not in {
            RESOURCE_HEAVY_GPU_CANDIDATE,
            RESOURCE_HEAVY_GPU_TRAIN,
            RESOURCE_GPU_LIGHT_TRAIN,
            RESOURCE_UNKNOWN_GPU_EXEC,
            RESOURCE_GPU_FEATURE_EXTRACT,
        }:
            return False
        return True

    def _install_pressure_observe_first(
        self,
        job: ResourceJob,
        *,
        gate: str,
        status: str,
        pressure: dict[str, Any],
        red_gpu_ids: list[str],
        allowed_classes: list[str],
        resource_context_version: int | None,
        resource_pressure_generation: int | None,
        current_pressure_generation: int | None,
        pressure_snapshot: dict[str, Any],
        digest_pressure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            pressure_generation = int(pressure_snapshot.get("generation") or pressure.get("generation") or 0)
        except (TypeError, ValueError):
            pressure_generation = 0
        state = {
            "active": True,
            "source": "stale_pressure_preflight",
            "reason": str(gate or "stale_resource_context"),
            "rule_status": str(status or "DENIED_REPLAN"),
            "started_at": time.time(),
            "observe_window_sec": self.stale_pressure_observe_window_sec,
            "observe_max_sec": self.stale_pressure_observe_max_sec,
            "healthy_skip_llm": self.stale_pressure_healthy_skip_llm,
            "min_free_mem_gb": self.stale_pressure_observe_min_free_mem_gb,
            "resource_mode": str(pressure.get("resource_mode") or "RED"),
            "gpu_ids": list(red_gpu_ids or job.gpu_ids or []),
            "allowed_classes": list(allowed_classes or []),
            "cooldown_sec": float(pressure.get("cooldown_remaining_sec") or 0.0),
            "eta_next_train_sec": float(pressure.get("eta_next_train_sec") or 0.0),
            "eta_confidence": str(pressure.get("eta_confidence") or "low"),
            "pressure_generation": pressure_generation,
            "resource_context_version": resource_context_version,
            "resource_pressure_generation": resource_pressure_generation,
            "current_pressure_generation": current_pressure_generation,
        }
        job.observe_first_state = state
        payload = {
            **state,
            "blocked_class": job.resource_class,
            "scope": "per_gpu",
            "pressure": pressure_snapshot,
            "digest_pressure": dict(digest_pressure or {}),
        }
        self._emit("resource_observe_first_candidate", job, status="observing", payload=payload)
        self._emit("resource_policy_gate", job, status="observe_first", payload=payload)
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "observe_first_candidate",
                payload={"job_id": job.job_id, "resource_type": "gpu", "result": payload},
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return {
            "allowed": True,
            "resource_class": job.resource_class,
            "status": "OBSERVE_FIRST",
            "reason": str(gate or "stale_resource_context"),
            "observe_first": True,
            "observe_window_sec": self.stale_pressure_observe_window_sec,
            "observe_max_sec": self.stale_pressure_observe_max_sec,
            "feedback_suppressed": True,
        }

    def observe_first_state(self, job_id: str | None, **_: Any) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs:
            return {"active": False}
        state = dict(self._jobs[job_id].observe_first_state or {})
        if not state.get("active"):
            return {"active": False}
        return state

    def observe_first_healthy_continue(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float = 0.0,
        stdout_age_sec: float = 0.0,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        metric_history_text: str = "",
        metric_history_line_count: int = 0,
        saw_training_progress: bool = False,
        saw_final_score: bool = False,
        current_phase: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs:
            return {"enabled": False, "reason": "job_missing"}
        job = self._jobs[job_id]
        state = dict(job.observe_first_state or {})
        if not state.get("active"):
            return {"enabled": False, "reason": "observe_first_inactive"}
        if state.get("healthy_emitted"):
            return {"enabled": False, "reason": "already_emitted"}
        evidence = {
            "stdout_lines": int(stdout_lines or 0),
            "stdout_bytes": int(stdout_bytes or 0),
            "stdout_age_sec": float(stdout_age_sec or 0.0),
            "saw_training_progress": bool(saw_training_progress),
            "saw_final_score": bool(saw_final_score),
            "current_phase": str(current_phase or ""),
            "gpu_mem_peak_gb": float(job.gpu_mem_peak_gb or 0.0),
            "idle_gpu_lease_samples": int(job.idle_gpu_lease_samples or 0),
            "dataloader_bottleneck_samples": int(job.dataloader_bottleneck_samples or 0),
            "last_gpu_util_sample": dict(job.last_gpu_util_sample or {}),
        }
        has_progress = bool(
            evidence["stdout_lines"]
            or evidence["stdout_bytes"]
            or evidence["saw_training_progress"]
            or evidence["saw_final_score"]
            or evidence["current_phase"]
            or evidence["gpu_mem_peak_gb"] > 0
        )
        health = "healthy_continue" if has_progress else "window_continue_low_confidence"
        state.update({
            "healthy_emitted": True,
            "health": health,
            "health_elapsed_sec": float(elapsed_sec or 0.0),
            "health_confidence": "medium" if has_progress else "low",
        })
        job.observe_first_state = state
        payload = {**state, "evidence": evidence, "llm_called": False}
        self._emit("resource_observe_first_healthy_continue", job, status="continued", payload=payload)
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "observe_first_healthy_continue",
                payload={"job_id": job.job_id, "resource_type": "gpu", "result": payload},
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return {"enabled": True, "health": health, "llm_called": False}

    def resource_preflight_decision(
        self,
        job_id: str | None,
        *,
        inferred_class: str,
        gpu_ids: list[str] | None = None,
        resource_context_version: int | None = None,
        resource_pressure_generation: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs:
            return {"allowed": True}
        job = self._jobs[job_id]
        incoming = str(inferred_class or job.resource_class)
        resource_class = self._merge_resource_class(job.resource_class, incoming, source_hint=job.source_hint)
        if bool(getattr(job.source_hint, "command_cpu_only", False)):
            ids = []
        else:
            ids = [str(x) for x in (gpu_ids or job.gpu_ids or []) if str(x).strip()]
        job.resource_class = resource_class
        if ids:
            job.gpu_ids = ids
        elif bool(getattr(job.source_hint, "command_cpu_only", False)):
            job.gpu_ids = []
        job.gpu_queue_relevant = self._gpu_queue_relevant(job.resource_class, job.source_hint, job.gpu_ids)
        deliverable_gate = self._completed_deliverable_preflight_gate(job)
        if deliverable_gate.get("blocked"):
            return {
                "allowed": False,
                "resource_class": job.resource_class,
                "status": str(deliverable_gate.get("status") or "DENIED_REPLAN"),
                "reason": str(deliverable_gate.get("reason") or "invalid_deliverable_schema_preflight"),
                "feedback": str(deliverable_gate.get("feedback") or ""),
                "feedback_suppressed": bool(deliverable_gate.get("feedback_suppressed")),
                "feedback_state_key": deliverable_gate.get("feedback_state_key"),
                "resource_feedback_repeated_count": deliverable_gate.get("resource_feedback_repeated_count"),
                "error": "Resource policy blocked command because deliverable schema is invalid",
                "allowed_classes": deliverable_gate.get("allowed_classes") or [],
                "deliverable_validity": deliverable_gate.get("deliverable_validity"),
                "deliverable_completion_state": deliverable_gate.get("deliverable_completion_state") or {},
            }
        context_is_stale = False
        if resource_context_version is not None:
            try:
                context_is_stale = int(resource_context_version) < int(getattr(self.state_machine, "event_count", 0) or 0)
            except (TypeError, ValueError):
                context_is_stale = False
        current_pressure_generation: int | None = None
        if self.resource_runtime is not None:
            try:
                current_pressure_generation = int(self.resource_runtime.pressure_generation() or 0)
            except (TypeError, ValueError):
                current_pressure_generation = None
        if resource_pressure_generation is not None and current_pressure_generation is not None:
            try:
                context_is_stale = context_is_stale or int(resource_pressure_generation) < current_pressure_generation
            except (TypeError, ValueError):
                pass
        if self.timeout_hard_gate_enabled and job.gpu_queue_relevant and self.resource_runtime is not None:
            pressure = self.resource_runtime.pressure_gate_decision(
                resource_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                command_digest=job.command_digest,
            )
            if isinstance(pressure, dict) and pressure.get("blocked"):
                gate = "stale_resource_context" if context_is_stale else str(pressure.get("reason") or "block_train_after_gpu_pressure")
                status = str(pressure.get("status") or "DENIED_REPLAN")
                red_gpu_ids = [str(x) for x in (pressure.get("red_gpu_ids") or pressure.get("gpu_ids") or []) if str(x).strip()]
                allowed_classes = [str(x) for x in (pressure.get("allowed_classes") or []) if str(x).strip()]
                bypass = self._allowed_class_bypass_result(
                    job,
                    reason=gate,
                    scope="per_gpu",
                    resource_mode=str(pressure.get("resource_mode") or "RED"),
                    allowed_classes=allowed_classes,
                )
                if bypass is not None:
                    return bypass
                pressure_snapshot = pressure.get("pressure") if isinstance(pressure.get("pressure"), dict) else {}
                digest_pressure: dict[str, Any] = {}
                if gate != "duplicate_digest_cooldown":
                    digest_pressure = self.resource_runtime.record_policy_gate_digest(
                        job_id=job.job_id,
                        resource_class=job.resource_class,
                        gpu_ids=job.gpu_ids,
                        command_digest=job.command_digest,
                        reason=gate,
                    )
                trial = self._startup_trial_decision(
                    job,
                    reason=gate,
                    status=status,
                    scope="per_gpu",
                    resource_mode=str(pressure.get("resource_mode") or "RED"),
                    allowed_classes=allowed_classes,
                    gpu_ids=red_gpu_ids,
                    extra_facts={
                        "pressure": pressure_snapshot,
                        "digest_pressure": digest_pressure,
                        "resource_context_version": resource_context_version,
                        "resource_pressure_generation": resource_pressure_generation,
                        "current_pressure_generation": current_pressure_generation,
                    },
                )
                if trial is not None:
                    return trial
                if self._pressure_observe_first_eligible(
                    job,
                    gate=gate,
                    pressure=pressure,
                    context_is_stale=context_is_stale,
                ):
                    return self._install_pressure_observe_first(
                        job,
                        gate=gate,
                        status=status,
                        pressure=pressure,
                        red_gpu_ids=red_gpu_ids,
                        allowed_classes=allowed_classes,
                        resource_context_version=resource_context_version,
                        resource_pressure_generation=resource_pressure_generation,
                        current_pressure_generation=current_pressure_generation,
                        pressure_snapshot=pressure_snapshot,
                        digest_pressure=digest_pressure,
                    )
                feedback = self._resource_feedback_text(
                    status=status,
                    reason=gate,
                    scope="per_gpu",
                    resource_mode=str(pressure.get("resource_mode") or "RED"),
                    blocked_class=job.resource_class,
                    gpu_ids=red_gpu_ids,
                    allowed_classes=allowed_classes,
                    cooldown_sec=float(pressure.get("cooldown_remaining_sec") or 0.0),
                    eta_next_train_sec=float(pressure.get("eta_next_train_sec") or 0.0),
                    eta_confidence=str(pressure.get("eta_confidence") or "low"),
                    unlock_condition="resource_pressure_cleared",
                    blocked_until_unlock=True,
                    pressure_generation=int(pressure_snapshot.get("generation") or pressure.get("generation") or 0),
                    duplicate_digest_count=(
                        int(pressure.get("duplicate_digest_count") or 0)
                        if pressure.get("duplicate_digest_count") is not None
                        else None
                    ),
                )
                if self._soft_gate_admission_enabled(reason=gate):
                    self._emit(
                        "resource_policy_gate",
                        job,
                        status="review",
                        payload={
                            "status": status,
                            "reason": gate,
                            "scope": "per_gpu",
                            "resource_mode": str(pressure.get("resource_mode") or "RED"),
                            "blocked_class": job.resource_class,
                            "allowed_classes": allowed_classes,
                            "requires_admission_review": True,
                            "cooldown_sec": float(pressure.get("cooldown_remaining_sec") or 0.0),
                            "eta_next_train_sec": float(pressure.get("eta_next_train_sec") or 0.0),
                            "eta_confidence": str(pressure.get("eta_confidence") or ""),
                            "pressure": pressure_snapshot,
                            "digest_pressure": digest_pressure,
                            "red_gpu_ids": red_gpu_ids,
                            "resource_context_version": resource_context_version,
                            "resource_pressure_generation": resource_pressure_generation,
                            "current_pressure_generation": current_pressure_generation,
                        },
                    )
                    return self._soft_gate_admission_review_result(
                        job,
                        status=status,
                        reason=gate,
                        scope="per_gpu",
                        resource_mode=str(pressure.get("resource_mode") or "RED"),
                        allowed_classes=allowed_classes,
                        feedback=feedback,
                        eta_next_train_sec=float(pressure.get("eta_next_train_sec") or 0.0),
                        eta_confidence=str(pressure.get("eta_confidence") or "low"),
                        cooldown_sec=float(pressure.get("cooldown_remaining_sec") or 0.0),
                        unlock_condition="resource_pressure_cleared",
                        blocked_until_unlock=True,
                        extra_facts={
                            "pressure": pressure_snapshot,
                            "digest_pressure": digest_pressure,
                            "red_gpu_ids": red_gpu_ids,
                            "duplicate_digest_count": pressure.get("duplicate_digest_count"),
                            "resource_context_version": resource_context_version,
                            "resource_pressure_generation": resource_pressure_generation,
                            "current_pressure_generation": current_pressure_generation,
                        },
                    )
                self._remember_resource_feedback(
                    job,
                    status=status,
                    reason=gate,
                    resource_mode=str(pressure.get("resource_mode") or "RED"),
                    blocked_class=job.resource_class,
                    allowed_classes=allowed_classes,
                    eta_next_train_sec=float(pressure.get("eta_next_train_sec") or 0.0),
                    cooldown_sec=float(pressure.get("cooldown_remaining_sec") or 0.0),
                )
                self._emit(
                    "resource_policy_gate",
                    job,
                    status="blocked",
                    payload={
                        "status": status,
                        "reason": gate,
                        "scope": "per_gpu",
                        "resource_mode": str(pressure.get("resource_mode") or "RED"),
                        "gpu_queue_timeout_count": int(pressure.get("max_queue_timeout_count") or 0),
                        "blocked_class": job.resource_class,
                        "allowed_after_gate": str(pressure.get("allowed_after_gate") or ""),
                        "allowed_classes": allowed_classes,
                        "cooldown_sec": float(pressure.get("cooldown_remaining_sec") or 0.0),
                        "eta_next_train_sec": float(pressure.get("eta_next_train_sec") or 0.0),
                        "eta_confidence": str(pressure.get("eta_confidence") or ""),
                        "duplicate_digest_count": pressure.get("duplicate_digest_count"),
                        "pressure": pressure_snapshot,
                        "digest_pressure": digest_pressure,
                        "red_gpu_ids": red_gpu_ids,
                        "resource_context_version": resource_context_version,
                        "resource_pressure_generation": resource_pressure_generation,
                        "current_pressure_generation": current_pressure_generation,
                    },
                )
                return {
                    "allowed": False,
                    "resource_class": job.resource_class,
                    "status": status,
                    "reason": gate,
                    "feedback": feedback,
                    "error": "Resource policy blocked command after GPU pressure update",
                }
        if job.post_feedback_gate:
            gate = dict(job.post_feedback_gate or {})
            source_gpu_ids = {str(x) for x in (gate.get("source_gpu_ids") or []) if str(x).strip()}
            current_gpu_ids = {str(x) for x in (job.gpu_ids or []) if str(x).strip()}
            same_feedback_scope = not source_gpu_ids or not current_gpu_ids or bool(source_gpu_ids & current_gpu_ids)
            if not same_feedback_scope:
                return {"allowed": True, "resource_class": job.resource_class}
            if self._soft_gpu_gate_can_clear_for_available_slot(
                resource_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                guard=gate,
            ):
                job.post_feedback_gate = {}
                self._emit(
                    "resource_soft_gate_cleared",
                    job,
                    status="allowed",
                    payload={
                        "reason": "observed_free_gpu_cleared_post_feedback_gate",
                        "scope": "worker_plan",
                        "resource_class": job.resource_class,
                        "gpu_ids": job.gpu_ids,
                        "post_feedback": gate,
                        "resource_control_profile": self.resource_control_profile,
                    },
                )
                return {
                    "allowed": True,
                    "resource_class": job.resource_class,
                    "reason": "observed_free_gpu_cleared_post_feedback_gate",
                }
            allowed_classes = [str(x) for x in (gate.get("allowed_classes") or []) if str(x).strip()]
            resource_mode = str(gate.get("source_resource_mode") or "YELLOW")
            status = "DENIED_REPLAN"
            reason = "post_feedback_blocked_class"
            bypass = self._allowed_class_bypass_result(
                job,
                reason=reason,
                scope="worker_plan",
                resource_mode=resource_mode,
                allowed_classes=allowed_classes,
            )
            if bypass is not None:
                return bypass
            source_reason = str(gate.get("source_reason") or "")
            trial = self._startup_trial_decision(
                job,
                reason=reason,
                status=status,
                scope="worker_plan",
                resource_mode=resource_mode,
                allowed_classes=allowed_classes,
                gpu_ids=job.gpu_ids,
                source_reason=source_reason,
                extra_facts={"post_feedback": gate},
            )
            if trial is not None:
                return trial
            raw_feedback = self._resource_feedback_text(
                status=status,
                reason=reason,
                scope="worker_plan",
                resource_mode=resource_mode,
                blocked_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                allowed_classes=allowed_classes,
            )
            feedback_state = self._dedupe_resource_feedback_for_agent(
                job,
                feedback=raw_feedback,
                status=status,
                reason=reason,
                scope="worker_plan",
                resource_mode=resource_mode,
                blocked_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                allowed_classes=allowed_classes,
            )
            feedback = str(feedback_state.get("feedback") or "")
            if self._soft_gate_admission_enabled(reason=reason, source_reason=source_reason):
                self._emit(
                    "resource_planner_guard",
                    job,
                    status="review",
                    payload={
                        "status": status,
                        "reason": reason,
                        "scope": "worker_plan",
                        "resource_mode": resource_mode,
                        "blocked_class": job.resource_class,
                        "allowed_classes": allowed_classes,
                        "post_feedback": gate,
                        "requires_admission_review": True,
                        "feedback_state_key": feedback_state.get("feedback_state_key"),
                        "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
                        "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
                    },
                )
                return self._soft_gate_admission_review_result(
                    job,
                    status=status,
                    reason=reason,
                    scope="worker_plan",
                    resource_mode=resource_mode,
                    allowed_classes=allowed_classes,
                    feedback=feedback,
                    feedback_state=feedback_state,
                    unlock_condition="resource_context_changed",
                    blocked_until_unlock=True,
                    extra_facts={"post_feedback": gate},
                )
            self._remember_resource_feedback(
                job,
                status=status,
                reason=reason,
                resource_mode=resource_mode,
                blocked_class=job.resource_class,
                allowed_classes=allowed_classes,
            )
            self._emit(
                "resource_planner_guard",
                job,
                status="blocked",
                payload={
                    "status": status,
                    "reason": reason,
                    "scope": "worker_plan",
                    "resource_mode": resource_mode,
                    "blocked_class": job.resource_class,
                    "allowed_classes": allowed_classes,
                    "post_feedback": gate,
                    "feedback_state_key": feedback_state.get("feedback_state_key"),
                    "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
                    "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
                },
            )
            return {
                "allowed": False,
                "resource_class": job.resource_class,
                "status": status,
                "reason": reason,
                "feedback": feedback,
                "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
                "feedback_state_key": feedback_state.get("feedback_state_key"),
                "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
                "error": "Resource planner guard blocked command after RESOURCE_FEEDBACK",
            }
        active_guard = self._active_plan_guard_decision(job)
        if active_guard.get("blocked"):
            guard = active_guard.get("guard") if isinstance(active_guard.get("guard"), dict) else {}
            allowed_classes = [str(x) for x in (guard.get("allowed_classes") or []) if str(x).strip()]
            resource_mode = str(guard.get("resource_mode") or "YELLOW")
            status = "DENIED_REPLAN"
            reason = "active_resource_plan_guard"
            bypass = self._allowed_class_bypass_result(
                job,
                reason=reason,
                scope="worker_plan",
                resource_mode=resource_mode,
                allowed_classes=allowed_classes,
            )
            if bypass is not None:
                return bypass
            source_reason = str(guard.get("source_reason") or "")
            trial = self._startup_trial_decision(
                job,
                reason=reason,
                status=status,
                scope="worker_plan",
                resource_mode=resource_mode,
                allowed_classes=allowed_classes,
                gpu_ids=job.gpu_ids,
                source_reason=source_reason,
                extra_facts={
                    "guard_id": str(active_guard.get("guard_id") or ""),
                    "current_pressure_generation": active_guard.get("current_pressure_generation"),
                    "source_feedback": guard,
                },
            )
            if trial is not None:
                return trial
            raw_feedback = self._resource_feedback_text(
                status=status,
                reason=reason,
                scope="worker_plan",
                resource_mode=resource_mode,
                blocked_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                allowed_classes=allowed_classes,
                eta_next_train_sec=float(active_guard.get("remaining_sec") or 0.0),
                eta_confidence="low",
                unlock_condition="resource_context_changed",
                blocked_until_unlock=True,
                pressure_generation=int(active_guard.get("current_pressure_generation") or 0),
            )
            feedback_state = self._dedupe_resource_feedback_for_agent(
                job,
                feedback=raw_feedback,
                status=status,
                reason=reason,
                scope="worker_plan",
                resource_mode=resource_mode,
                blocked_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                allowed_classes=allowed_classes,
                unlock_condition="resource_context_changed",
                blocked_until_unlock=True,
            )
            feedback = str(feedback_state.get("feedback") or "")
            if self._soft_gate_admission_enabled(reason=reason, source_reason=source_reason):
                self._emit(
                    "resource_planner_guard",
                    job,
                    status="review",
                    payload={
                        "status": status,
                        "reason": reason,
                        "scope": "worker_plan",
                        "resource_mode": resource_mode,
                        "blocked_class": job.resource_class,
                        "allowed_classes": allowed_classes,
                        "guard_id": str(active_guard.get("guard_id") or ""),
                        "remaining_sec": float(active_guard.get("remaining_sec") or 0.0),
                        "current_pressure_generation": active_guard.get("current_pressure_generation"),
                        "source_feedback": guard,
                        "requires_admission_review": True,
                        "feedback_state_key": feedback_state.get("feedback_state_key"),
                        "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
                        "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
                    },
                )
                return self._soft_gate_admission_review_result(
                    job,
                    status=status,
                    reason=reason,
                    scope="worker_plan",
                    resource_mode=resource_mode,
                    allowed_classes=allowed_classes,
                    feedback=feedback,
                    feedback_state=feedback_state,
                    eta_next_train_sec=float(active_guard.get("remaining_sec") or 0.0),
                    eta_confidence="low",
                    unlock_condition="resource_context_changed",
                    blocked_until_unlock=True,
                    extra_facts={
                        "guard_id": str(active_guard.get("guard_id") or ""),
                        "current_pressure_generation": active_guard.get("current_pressure_generation"),
                        "source_feedback": guard,
                    },
                )
            self._remember_resource_feedback(
                job,
                status=status,
                reason=reason,
                resource_mode=resource_mode,
                blocked_class=job.resource_class,
                allowed_classes=allowed_classes,
                eta_next_train_sec=float(active_guard.get("remaining_sec") or 0.0),
            )
            self._emit(
                "resource_planner_guard",
                job,
                status="blocked",
                payload={
                    "status": status,
                    "reason": reason,
                    "scope": "worker_plan",
                    "resource_mode": resource_mode,
                    "blocked_class": job.resource_class,
                    "allowed_classes": allowed_classes,
                    "guard_id": str(active_guard.get("guard_id") or ""),
                    "remaining_sec": float(active_guard.get("remaining_sec") or 0.0),
                    "current_pressure_generation": active_guard.get("current_pressure_generation"),
                    "source_feedback": guard,
                    "feedback_state_key": feedback_state.get("feedback_state_key"),
                    "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
                    "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
                },
            )
            return {
                "allowed": False,
                "resource_class": job.resource_class,
                "status": status,
                "reason": reason,
                "feedback": feedback,
                "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
                "feedback_state_key": feedback_state.get("feedback_state_key"),
                "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
                "error": "Resource planner guard blocked command after active RESOURCE_FEEDBACK",
            }

        if not self.timeout_hard_gate_enabled or self._gpu_queue_timeout_count <= 0:
            return {"allowed": True, "resource_class": job.resource_class}

        blocked = False
        gate = ""
        if self._gpu_queue_timeout_count >= self.timeout_tt_only_after:
            blocked = job.resource_class not in TT_ALLOWED_AFTER_TIMEOUT
            gate = "tt_only_after_queue_timeouts"
        elif self._gpu_queue_timeout_count >= self.timeout_block_train_after:
            blocked = job.resource_class in TRAIN_CLASSES or job.resource_class == RESOURCE_UNKNOWN_GPU_EXEC
            gate = "block_train_after_queue_timeout"
        if not blocked:
            return {"allowed": True, "resource_class": job.resource_class, "gate": gate}

        if gate == "tt_only_after_queue_timeouts":
            allowed = "pure_tt_cpu,gpu_tt_light,heavy_cpu_candidate,readonly_cpu,light_cpu"
            allowed_classes = [
                RESOURCE_PURE_TT_CPU,
                RESOURCE_GPU_TT_LIGHT,
                RESOURCE_HEAVY_CPU_CANDIDATE,
                RESOURCE_READONLY_CPU,
                RESOURCE_LIGHT_CPU,
            ]
        else:
            allowed = "non-training work"
            allowed_classes = [RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, "readonly_cpu", "light_cpu"]
        bypass = self._allowed_class_bypass_result(
            job,
            reason=gate,
            scope="worker_local",
            resource_mode="RED",
            allowed_classes=allowed_classes,
        )
        if bypass is not None:
            return {**bypass, "gate": gate}
        trial = self._startup_trial_decision(
            job,
            reason=gate,
            status="DENIED_REPLAN",
            scope="worker_local",
            resource_mode="RED",
            allowed_classes=allowed_classes,
            gpu_ids=job.gpu_ids,
            extra_facts={
                "gpu_queue_timeout_count": self._gpu_queue_timeout_count,
                "allowed_after_gate": allowed,
            },
        )
        if trial is not None:
            return {**trial, "gate": gate}
        feedback = self._resource_feedback_text(
            status="DENIED_REPLAN",
            reason=gate,
            scope="worker_local",
            resource_mode="RED",
            blocked_class=job.resource_class,
            gpu_ids=job.gpu_ids,
            allowed_classes=allowed_classes,
            cooldown_sec=120.0,
            eta_next_train_sec=420.0,
            eta_confidence="low",
            unlock_condition="resource_pressure_cleared",
            blocked_until_unlock=True,
        )
        if self._soft_gate_admission_enabled(reason=gate):
            self._emit(
                "resource_policy_gate",
                job,
                status="review",
                payload={
                    "status": "DENIED_REPLAN",
                    "reason": gate,
                    "scope": "worker_local",
                    "resource_mode": "RED",
                    "gpu_queue_timeout_count": self._gpu_queue_timeout_count,
                    "blocked_class": job.resource_class,
                    "allowed_after_gate": allowed,
                    "allowed_classes": allowed_classes,
                    "cooldown_sec": 120.0,
                    "eta_next_train_sec": 420.0,
                    "eta_confidence": "low",
                    "requires_admission_review": True,
                },
            )
            return self._soft_gate_admission_review_result(
                job,
                status="DENIED_REPLAN",
                reason=gate,
                scope="worker_local",
                resource_mode="RED",
                allowed_classes=allowed_classes,
                feedback=feedback,
                eta_next_train_sec=420.0,
                eta_confidence="low",
                cooldown_sec=120.0,
                unlock_condition="resource_pressure_cleared",
                blocked_until_unlock=True,
                extra_facts={
                    "gpu_queue_timeout_count": self._gpu_queue_timeout_count,
                    "allowed_after_gate": allowed,
                },
            )
        self._remember_resource_feedback(
            job,
            status="DENIED_REPLAN",
            reason=gate,
            resource_mode="RED",
            blocked_class=job.resource_class,
            allowed_classes=allowed_classes,
            eta_next_train_sec=420.0,
            cooldown_sec=120.0,
        )
        self._emit(
            "resource_policy_gate",
            job,
            status="blocked",
            payload={
                "status": "DENIED_REPLAN",
                "reason": gate,
                "scope": "worker_local",
                "resource_mode": "RED",
                "gpu_queue_timeout_count": self._gpu_queue_timeout_count,
                "blocked_class": job.resource_class,
                "allowed_after_gate": allowed,
                "allowed_classes": allowed_classes,
                "cooldown_sec": 120.0,
                "eta_next_train_sec": 420.0,
                "eta_confidence": "low",
            },
        )
        return {
            "allowed": False,
            "resource_class": job.resource_class,
            "reason": gate,
            "feedback": feedback,
            "error": "Resource policy blocked command after GPU queue timeout",
        }

    def observation_policy(
        self,
        job_id: str | None,
        *,
        inferred_class: str,
        gpu_ids: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not (self.observation_enabled and self.observation_shadow_workspace_enabled):
            return {"enabled": False, "reason": "observation_disabled"}
        if not job_id or job_id not in self._jobs:
            return {"enabled": False, "reason": "job_not_found"}
        job = self._jobs[job_id]
        incoming = str(inferred_class or job.resource_class)
        resource_class = self._merge_resource_class(job.resource_class, incoming, source_hint=job.source_hint)
        ids = [str(x) for x in (gpu_ids or job.gpu_ids or []) if str(x).strip()]
        if bool(getattr(job.source_hint, "command_cpu_only", False)):
            ids = []
        if resource_class not in {RESOURCE_UNKNOWN_EXEC, RESOURCE_UNKNOWN_GPU_EXEC}:
            return {"enabled": False, "reason": "not_unknown_exec"}
        if not ids:
            return {"enabled": False, "reason": "no_gpu_scope"}
        if job.gpu_queue_relevant:
            return {"enabled": False, "reason": "already_queue_relevant"}
        return {
            "enabled": True,
            "window_sec": self.observation_window_sec,
            "reason": "unknown_gpu_exec_shadow_observe",
            "gpu_ids": ids,
        }

    def observation_event(
        self,
        job_id: str | None,
        *,
        state: str,
        reason: str = "",
        elapsed_sec: float = 0.0,
        shadow_workspace: str = "",
        **_: Any,
    ) -> None:
        if not job_id or job_id not in self._jobs:
            return
        self._emit(
            "resource_observation",
            self._jobs[job_id],
            status=str(state or "observed"),
            payload={
                "state": str(state or ""),
                "reason": str(reason or ""),
                "elapsed_sec": float(elapsed_sec or 0.0),
                "shadow_workspace": str(shadow_workspace or ""),
            },
        )

    def observation_promoted(
        self,
        job_id: str | None,
        *,
        reason: str,
        elapsed_sec: float = 0.0,
        **_: Any,
    ) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        hint = job.source_hint or ResourceSourceHint()
        labels = list(hint.hint_labels or [])
        if "shadow_promoted" not in labels:
            labels.append("shadow_promoted")
        job.source_hint = replace(
            hint,
            command_gpu_evidence=True,
            command_train_evidence=True,
            hint_labels=labels,
        )
        job.resource_class = RESOURCE_HEAVY_GPU_CANDIDATE
        job.gpu_queue_relevant = self._gpu_queue_relevant(job.resource_class, job.source_hint, job.gpu_ids)
        self._emit(
            "resource_observation_promoted",
            job,
            status="promoted",
            payload={
                "reason": str(reason or ""),
                "elapsed_sec": float(elapsed_sec or 0.0),
                "promoted_class": job.resource_class,
                "queue_relevant": bool(job.gpu_queue_relevant),
                "source_hint": job.source_hint.to_event_payload(),
            },
        )

    def lease_registered(self, job_id: str | None, *, pid: int | None = None, pgid: int | None = None, **_: Any) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.pid = int(pid) if pid is not None else None
        job.pgid = int(pgid) if pgid is not None else job.pid
        job.pid_start_time_epoch = None
        if job.pid is not None:
            info = read_proc_info(job.pid)
            try:
                job.pid_start_time_epoch = float(info.get("start_time_epoch")) if info.get("start_time_epoch") is not None else None
            except (TypeError, ValueError):
                job.pid_start_time_epoch = None
        if job.visible:
            self._emit(
                "resource_lease_registered",
                job,
                status="running",
                payload={"pid": pid, "pgid": pgid},
            )

    def recoverable_artifact_scope_registered(
        self,
        job_id: str | None,
        *,
        run_dir: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        run_state_path: str | Path | None = None,
        **_: Any,
    ) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.run_dir = Path(run_dir).resolve(strict=False) if run_dir is not None else None
        job.run_artifact_dir = Path(artifact_dir).resolve(strict=False) if artifact_dir is not None else None
        job.run_state_path = Path(run_state_path).resolve(strict=False) if run_state_path is not None else None
        payload = {
            "run_dir": str(job.run_dir or ""),
            "artifact_dir": str(job.run_artifact_dir or ""),
            "run_state_path": str(job.run_state_path or ""),
        }
        self._emit(
            "resource_recoverable_artifact_scope_registered",
            job,
            status="registered",
            payload=payload,
        )
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_recoverable_artifact_scope_registered",
                payload={
                    "job_id": job.job_id,
                    "command_digest": job.command_digest,
                    "resource_class": job.resource_class,
                    **payload,
                },
                command_id=job.job_id,
                lease_id=job.job_id,
            )

    def resource_wait_instruction_for_bash_sleep(
        self,
        *,
        command: str = "",
        planned_sleep_sec: float | None = None,
    ) -> dict[str, Any]:
        if self.resource_runtime is None:
            return {"available": False, "reason": "resource_runtime_unavailable"}
        return self.resource_runtime.resource_wait_instruction_for_bash_sleep(
            command=command,
            planned_sleep_sec=planned_sleep_sec,
        )

    async def managed_resource_wait(
        self,
        *,
        wait_token: str,
        max_wait_sec: float | None = None,
        reason: str = "resource_busy",
    ) -> dict[str, Any]:
        if self.resource_runtime is None:
            return {
                "status": "RESOURCE_WAIT_NOT_AVAILABLE",
                "reason": "resource_runtime_unavailable",
                "feedback": "RESOURCE_FEEDBACK: RESOURCE_WAIT_NOT_AVAILABLE because resource runtime is unavailable; retry_allowed=false.\n",
            }
        return await self.resource_runtime.managed_resource_wait(
            wait_token=wait_token,
            max_wait_sec=max_wait_sec,
            reason=reason,
        )

    def queue_try_acquire(self, job_id: str | None, **kwargs: Any) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs or self.resource_runtime is None:
            return {"enabled": False, "acquired": True}
        job = self._jobs[job_id]
        incoming = str(kwargs.get("inferred_class") or job.resource_class)
        resource_class = self._merge_resource_class(job.resource_class, incoming, source_hint=job.source_hint)
        if bool(getattr(job.source_hint, "command_cpu_only", False)):
            gpu_ids = []
        else:
            gpu_ids = [str(x) for x in (kwargs.get("gpu_ids") or job.gpu_ids) if str(x).strip()]
        job.resource_class = resource_class
        if gpu_ids:
            job.gpu_ids = gpu_ids
        elif bool(getattr(job.source_hint, "command_cpu_only", False)):
            job.gpu_ids = []
        job.gpu_queue_relevant = self._gpu_queue_relevant(job.resource_class, job.source_hint, job.gpu_ids)
        if not job.gpu_queue_relevant:
            return {"enabled": False, "acquired": True}
        metadata = {
            "command_digest": job.command_digest,
            "command_excerpt": str(job.command or "")[:2000],
            "entrypoint": self._command_entrypoint(job.command),
            "resource_class": job.resource_class,
            "cpu_set": job.cpu_set,
            "gpu_request_count": job.gpu_request_count,
            "source_hint": job.source_hint.to_event_payload() if job.source_hint is not None else {},
            "value_hint": dict(job.value_hint or {}),
        }
        observe_first = dict(job.observe_first_state or {})
        if observe_first.get("active"):
            metadata.update({
                "observe_first": observe_first,
                "observe_then_run": True,
                "observe_source": str(observe_first.get("source") or "stale_pressure_preflight"),
                "observe_sec": float(observe_first.get("observe_window_sec") or self.stale_pressure_observe_window_sec),
            })
        result = self.resource_runtime.queue_try_acquire(
            job_id=job.job_id,
            resource_class=job.resource_class,
            gpu_ids=job.gpu_ids,
            request_count=job.gpu_request_count,
            metadata=metadata,
        )
        if not result.get("enabled"):
            return result
        details = result.get("details") if isinstance(result.get("details"), dict) else {}
        for raw in details.get("reaped") or []:
            if isinstance(raw, dict):
                self._emit(
                    "resource_gpu_stale_lease_reaped",
                    job,
                    status="reaped",
                    payload={"reaped_job_id": raw.get("job_id"), "gpu_ids": raw.get("gpu_ids") or []},
                )
        if result.get("acquired"):
            assigned_gpu_ids = [str(x) for x in (result.get("gpu_ids") or []) if str(x).strip()]
            if assigned_gpu_ids:
                job.gpu_ids = assigned_gpu_ids
            lease = details.get("lease") if isinstance(details.get("lease"), dict) else {}
            lease_meta = lease.get("metadata") if isinstance(lease.get("metadata"), dict) else {}
            job.lease_mode = str(lease_meta.get("lease_mode") or "exclusive")
            job.shared_primary_job_id = str(lease_meta.get("shared_primary_job_id") or "")
            if job.observe_first_state.get("active"):
                job.observe_first_state.update({
                    "lease_acquired": True,
                    "assigned_physical_gpus": assigned_gpu_ids,
                    "lease_started_at": time.time(),
                })
                result["admission_observe_then_run"] = True
                result["admission_observe_sec"] = float(
                    job.observe_first_state.get("observe_window_sec")
                    or self.stale_pressure_observe_window_sec
                )
                result["resource_trial"] = bool(job.observe_first_state.get("trial"))
                result["resource_trial_reason"] = str(job.observe_first_state.get("reason") or "")
                result["resource_trial_window_sec"] = float(
                    job.observe_first_state.get("trial_window_sec")
                    or job.observe_first_state.get("observe_window_sec")
                    or result["admission_observe_sec"]
                )
                result["resource_trial_hard_review_sec"] = float(
                    job.observe_first_state.get("trial_hard_review_sec")
                    or job.observe_first_state.get("observe_max_sec")
                    or result["resource_trial_window_sec"]
                )
                observe_payload = {
                    **dict(job.observe_first_state),
                    "gpu_ids": result.get("gpu_ids") or job.gpu_ids,
                    "assigned_physical_gpus": assigned_gpu_ids,
                    "llm_called": False,
                }
                self._emit("resource_observe_first_started", job, status="running", payload=observe_payload)
                if result.get("resource_trial"):
                    self._emit("resource_trial_started", job, status="running", payload=observe_payload)
                if self.resource_runtime is not None:
                    self.resource_runtime.record_resource_event(
                        "observe_first_started",
                        payload={"job_id": job.job_id, "resource_type": "gpu", "result": observe_payload},
                        command_id=job.job_id,
                        lease_id=job.job_id,
                    )
                    if result.get("resource_trial"):
                        self.resource_runtime.record_resource_event(
                            "resource_trial_started",
                            payload={"job_id": job.job_id, "resource_type": "gpu", "result": observe_payload},
                            command_id=job.job_id,
                            lease_id=job.job_id,
                        )
            self._emit(
                "resource_gpu_lease_acquired",
                job,
                status="acquired",
                payload={
                    "gpu_ids": result.get("gpu_ids") or job.gpu_ids,
                    "assigned_physical_gpus": result.get("assigned_physical_gpus") or result.get("gpu_ids") or job.gpu_ids,
                    "allowed_physical_gpus": result.get("allowed_physical_gpus") or [],
                    "candidate_physical_gpus": result.get("candidate_physical_gpus") or [],
                    "requested_gpu_count": result.get("requested_gpu_count"),
                    "reason": result.get("reason") or "",
                    "admission_status": result.get("status") or "GRANTED",
                    "queue_position": result.get("queue_position"),
                    "queue_len": result.get("queue_len"),
                    "policy_resource_class": result.get("policy_resource_class") or "",
                    "slot_weight": result.get("slot_weight"),
                    "capacity_slots": result.get("capacity_slots"),
                    "admission_priority_score": result.get("admission_priority_score"),
                    "expected_value_score": result.get("expected_value_score"),
                    "near_submission_score": result.get("near_submission_score"),
                    "long_runtime_penalty": result.get("long_runtime_penalty"),
                    "value_hint": result.get("value_hint") or {},
                    "pressure": result.get("pressure") or {},
                },
            )
        elif result.get("queue_started_first"):
            self._emit(
                "resource_gpu_queue_wait_started",
                job,
                status="pending",
                payload={
                    "gpu_ids": result.get("gpu_ids") or job.gpu_ids,
                    "assigned_physical_gpus": result.get("assigned_physical_gpus") or [],
                    "allowed_physical_gpus": result.get("allowed_physical_gpus") or [],
                    "candidate_physical_gpus": result.get("candidate_physical_gpus") or [],
                    "requested_gpu_count": result.get("requested_gpu_count"),
                    "reason": result.get("reason") or "gpu_slot_unavailable",
                    "max_wait_sec": result.get("max_wait_sec"),
                    "heartbeat_sec": result.get("heartbeat_sec"),
                    "policy_resource_class": result.get("policy_resource_class") or "",
                    "slot_weight": result.get("slot_weight"),
                    "capacity_slots": result.get("capacity_slots"),
                    "admission_priority_score": result.get("admission_priority_score"),
                    "expected_value_score": result.get("expected_value_score"),
                    "near_submission_score": result.get("near_submission_score"),
                    "long_runtime_penalty": result.get("long_runtime_penalty"),
                    "value_hint": result.get("value_hint") or {},
                    "share_override_candidate": bool(result.get("share_override_candidate")),
                    "share_override": result.get("share_override") or {},
                },
            )
        if str(result.get("status") or "").upper() in {"PENDING", "REPLAN", "DEFERRED", "DENIED_REPLAN"}:
            self._remember_resource_feedback(
                job,
                status=str(result.get("status") or "PENDING"),
                reason=str(result.get("reason") or "gpu_slot_unavailable"),
                resource_mode=str(result.get("resource_mode") or "YELLOW"),
                blocked_class=job.resource_class,
                allowed_classes=[str(x) for x in (result.get("allowed_classes") or []) if str(x).strip()],
                eta_next_train_sec=float(result.get("eta_next_train_sec") or 0.0),
                retry_after_sec=(float(result.get("retry_after_sec") or 0.0) if result.get("retry_after_sec") is not None else None),
                post_feedback_action=str(result.get("post_feedback_action") or ""),
            )
            self._emit(
                "resource_admission_deferred",
                job,
                status=str(result.get("status") or "pending").lower(),
                payload={
                    "status": result.get("status") or result.get("admission_action") or "PENDING",
                    "admission_action": result.get("admission_action") or result.get("status") or "PENDING",
                    "reason": result.get("reason") or "gpu_slot_unavailable",
                    "resource_mode": result.get("resource_mode") or "YELLOW",
                    "gpu_ids": result.get("gpu_ids") or job.gpu_ids,
                    "assigned_physical_gpus": result.get("assigned_physical_gpus") or [],
                    "allowed_physical_gpus": result.get("allowed_physical_gpus") or [],
                    "candidate_physical_gpus": result.get("candidate_physical_gpus") or [],
                    "requested_gpu_count": result.get("requested_gpu_count"),
                    "queue_position": result.get("queue_position"),
                    "queue_len": result.get("queue_len"),
                    "top_waiter_job_id": result.get("top_waiter_job_id"),
                    "eta_next_train_sec": result.get("eta_next_train_sec"),
                    "eta_confidence": result.get("eta_confidence") or "low",
                    "retry_after_sec": result.get("retry_after_sec"),
                    "blocked_until_unlock": bool(result.get("blocked_until_unlock")),
                    "unlock_condition": result.get("unlock_condition") or "",
                    "post_feedback_action": result.get("post_feedback_action") or "",
                    "admission_cached_backoff": bool(result.get("admission_cached_backoff")),
                    "allowed_classes": result.get("allowed_classes") or [],
                    "policy_resource_class": result.get("policy_resource_class") or "",
                    "slot_weight": result.get("slot_weight"),
                    "capacity_slots": result.get("capacity_slots"),
                    "eta_source": result.get("eta_source"),
                    "runtime_history_count": result.get("runtime_history_count"),
                    "runtime_avg_sec": result.get("runtime_avg_sec"),
                    "runtime_p80_sec": result.get("runtime_p80_sec"),
                    "active_blocker_count": result.get("active_blocker_count"),
                    "active_blocker_age_sec_max": result.get("active_blocker_age_sec_max"),
                    "admission_priority_score": result.get("admission_priority_score"),
                    "expected_value_score": result.get("expected_value_score"),
                    "near_submission_score": result.get("near_submission_score"),
                    "long_runtime_penalty": result.get("long_runtime_penalty"),
                    "value_hint": result.get("value_hint") or {},
                    "pressure": result.get("pressure") or {},
                    "share_override_candidate": bool(result.get("share_override_candidate")),
                    "share_override": result.get("share_override") or {},
                },
            )
        return result

    def _admission_task_card(self, job: ResourceJob, result: dict[str, Any]) -> dict[str, Any]:
        workspace = job.workspace_dir.resolve(strict=False) if job.workspace_dir is not None else None
        command = str(job.command or "")
        if workspace is not None:
            command = command.replace(str(workspace), ".")
        details = result.get("details") if isinstance(result.get("details"), dict) else {}
        return {
            "job_id": job.job_id,
            "worker_id": self.worker_id,
            "command_preview": command[:600],
            "command_digest": job.command_digest,
            "entrypoint": self._command_entrypoint(job.command),
            "resource_class": job.resource_class,
            "policy_resource_class": result.get("policy_resource_class") or "",
            "gpu_ids": result.get("gpu_ids") or job.gpu_ids,
            "assigned_physical_gpus": result.get("assigned_physical_gpus") or [],
            "allowed_physical_gpus": result.get("allowed_physical_gpus") or [],
            "candidate_physical_gpus": result.get("candidate_physical_gpus") or [],
            "requested_gpu_count": result.get("requested_gpu_count") or job.gpu_request_count,
            "gpu_request_count": job.gpu_request_count,
            "timeout_sec": job.timeout_sec,
            "queue_position": result.get("queue_position"),
            "queue_len": result.get("queue_len"),
            "top_waiter_job_id": result.get("top_waiter_job_id"),
            "eta_next_train_sec": result.get("eta_next_train_sec"),
            "eta_confidence": result.get("eta_confidence"),
            "admission_priority_score": result.get("admission_priority_score"),
            "value_hint": result.get("value_hint") or dict(job.value_hint or {}),
            "duplicate_digest_summary": result.get("duplicate_digest_summary") or {},
            "safety": result.get("safety") or {},
            "admission_opportunity": result.get("admission_opportunity") or {},
            "observe_first": dict(job.observe_first_state or {}),
            "requires_admission_review": bool(result.get("requires_admission_review")),
            "soft_gate_reason": result.get("soft_gate_reason") or "",
            "soft_gate_scope": result.get("soft_gate_scope") or "",
            "soft_gate": result.get("soft_gate") or {},
            "lease_grantable_by_llm": bool(result.get("lease_grantable_by_llm")),
            "blocked_until_unlock": bool(result.get("blocked_until_unlock")),
            "unlock_condition": result.get("unlock_condition") or "",
            "blocker_details": details.get("blocker_details") or {},
            "resource_mode": result.get("resource_mode") or "",
            "remaining_timeout_sec": job.timeout_sec,
        }

    @staticmethod
    def _gpu_share_decision_facts(observation: dict[str, Any], *, decision_source: str) -> dict[str, Any]:
        memory = observation.get("memory") if isinstance(observation.get("memory"), dict) else {}
        hard_gates_raw = observation.get("hard_gates") if isinstance(observation.get("hard_gates"), dict) else {}
        hard_gates = {str(key): bool(value) for key, value in hard_gates_raw.items()}
        secondary_allowed = observation.get("secondary_allowed") if isinstance(observation.get("secondary_allowed"), dict) else {}
        trial_share = observation.get("trial_share") if isinstance(observation.get("trial_share"), dict) else {}
        cpu_isolation = observation.get("cpu_isolation") if isinstance(observation.get("cpu_isolation"), dict) else {}
        primary_signal = observation.get("primary_signal") if isinstance(observation.get("primary_signal"), dict) else {}

        def as_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        return {
            "schema_version": 1,
            "decision_source": decision_source,
            "objective": "increase throughput by running revocable parallel GPU work when observed GPU headroom is idle",
            "share_eligible": bool(observation.get("share_eligible")),
            "share_candidate": bool(observation.get("share_candidate")),
            "observe_only": bool(observation.get("observe_only")),
            "hard_gates_pass": bool(hard_gates) and all(hard_gates.values()),
            "hard_gates": hard_gates,
            "primary_gpu_util_p90_pct": as_float(memory.get("gpu_util_p90_pct")),
            "primary_gpu_mem_current_gb": as_float(memory.get("gpu_mem_current_gb")),
            "primary_gpu_mem_peak_gb": as_float(memory.get("gpu_mem_peak_gb")),
            "gpu_free_mem_gb": as_float(memory.get("gpu_free_mem_gb")),
            "required_headroom_gb": as_float(memory.get("required_headroom_gb")),
            "secondary_estimated_peak_gb": as_float(memory.get("secondary_estimated_peak_gb")),
            "cpu_pressure": str(observation.get("cpu_pressure") or "unknown"),
            "cpu_isolation_mode": str(cpu_isolation.get("mode") or "unknown"),
            "cpu_isolated": bool(cpu_isolation.get("isolated")),
            "secondary_allowed": bool(secondary_allowed.get("allowed")),
            "secondary_allowed_reason": str(secondary_allowed.get("reason") or ""),
            "secondary_policy_gate": str(secondary_allowed.get("policy_gate") or ""),
            "secondary_effective_slot_weight": as_float(secondary_allowed.get("effective_slot_weight")),
            "trial_share": bool(trial_share.get("enabled")),
            "trial_primary_protected": bool(trial_share.get("primary_protected")),
            "productive_primary_signal": bool(primary_signal.get("productive_primary_signal")) if primary_signal else None,
            "llm_decision_focus": "grant sharing when these observed facts make the secondary cheap and revocable; deny or observe only for concrete memory, CPU, or progress uncertainty",
        }

    def _admission_share_review_proposal(
        self,
        job: ResourceJob,
        admission_result: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        proposal_type = "task_gpu_share_review"
        primary_id = str(observation.get("primary_job_id") or "")
        seed = f"{proposal_type}:{job.job_id}:{primary_id}:{now}"
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:8]}"
        trial_share = observation.get("trial_share") if isinstance(observation.get("trial_share"), dict) else {}
        reason_code = "task_gpu_share_review:admission_shared_trial"
        trigger_reasons = [
            "admission_blocked_waiter",
            "share_override_candidate",
            str((observation.get("secondary_allowed") or {}).get("reason") or "secondary_allowed"),
        ]
        if trial_share.get("enabled"):
            trigger_reasons.append("revocable_trial_share")
        share_decision_facts = self._gpu_share_decision_facts(observation, decision_source="admission_waiter")
        return {
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "severity": "yellow",
            "reason_code": reason_code,
            "trigger_reasons": trigger_reasons,
            "suggested_actions": ["GRANT_SHARED_GPU_LEASE", "DENY_SHARE_USE_CPU_SUPPORT", "CONTINUE_SHARED_OBSERVE"],
            "requires_llm_decision": True,
            "resource_snapshot": {
                "gpu_ids": observation.get("primary_gpu_ids") or admission_result.get("gpu_ids") or job.gpu_ids,
                "cpu_pressure": observation.get("cpu_pressure") or "unknown",
                "cpu_isolation": observation.get("cpu_isolation") or {},
                "memory": observation.get("memory") or {},
                "hard_gates": observation.get("hard_gates") or {},
                "secondary_allowed": observation.get("secondary_allowed") or {},
                "trial_share": trial_share,
                "admission_immediate_review": True,
            },
            "progress_snapshot": {
                "progress_signal": "unknown",
                "progress_confidence": "low",
                "heartbeat_source": "admission_waiter",
                "evidence_limitations": ["primary_progress_owned_by_holder_monitor"],
            },
            "blocker": {
                "job_id": primary_id,
                "worker_id": str(observation.get("primary_worker_id") or ""),
                "resource_class": str(observation.get("primary_resource_class") or ""),
                "gpu_ids": observation.get("primary_gpu_ids") or [],
                "share_role": "primary",
                "progress_signal": "unknown",
            },
            "waiters": [observation.get("waiter") or {"job_id": job.job_id, "resource_class": job.resource_class}],
            "share_observation": observation,
            "share_decision_facts": share_decision_facts,
            "share_payload": {
                "schema_version": 1,
                "phase": observation.get("phase"),
                "primary": {
                    "job_id": primary_id,
                    "worker_id": str(observation.get("primary_worker_id") or ""),
                    "resource_class": str(observation.get("primary_resource_class") or ""),
                    "gpu_ids": observation.get("primary_gpu_ids") or [],
                    "progress_signal": "unknown",
                    "progress_confidence": "low",
                },
                "waiter": observation.get("waiter") or {},
                "waiters": observation.get("waiters") or [],
                "cpu_pressure": observation.get("cpu_pressure") or "unknown",
                "cpu_isolation": observation.get("cpu_isolation") or {},
                "memory": observation.get("memory") or {},
                "hard_gates": observation.get("hard_gates") or {},
                "secondary_allowed": observation.get("secondary_allowed") or {},
                "trial_share": trial_share,
                "admission_immediate_review": True,
            },
            "admission_result": {
                "status": str(admission_result.get("status") or ""),
                "reason": str(admission_result.get("reason") or ""),
                "queue_position": admission_result.get("queue_position"),
                "queue_len": admission_result.get("queue_len"),
            },
            "decision_preview": {
                "job_id": job.job_id,
                "reason": reason_code,
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }

    def _admission_share_cpu_support_result(
        self,
        job: ResourceJob,
        admission_result: dict[str, Any],
        *,
        reason: str,
        decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = dict(admission_result or {})
        out.update({
            "enabled": True,
            "acquired": False,
            "status": "PENDING",
            "admission_action": "PENDING",
            "reason": reason or "local_gpu_busy_use_cpu_support",
            "admission_llm_reviewed": True,
            "admission_share_reviewed": True,
            "post_feedback_action": "cpu_support",
            "retry_after_sec": out.get("retry_after_sec") or 120.0,
        })
        holders = [str(x) for x in (out.get("holder_job_ids") or []) if str(x).strip()]
        wait_option: dict[str, Any] = {}
        if self.resource_runtime is not None:
            wait_option = self.resource_runtime.create_resource_wait_option(
                job_id=job.job_id,
                resource_class=str(out.get("policy_resource_class") or out.get("resource_class") or job.resource_class),
                gpu_ids=[str(x) for x in (out.get("gpu_ids") or job.gpu_ids) if str(x).strip()],
                holder_job_ids=holders,
                queue_position=(int(out.get("queue_position") or 0) if out.get("queue_position") is not None else None),
                queue_len=(int(out.get("queue_len") or 0) if out.get("queue_len") is not None else None),
                command_digest=job.command_digest,
                reason=reason or "shared GPU trial was not approved",
                max_wait_sec=min(600.0, max(60.0, float(out.get("eta_next_train_sec") or 600.0))),
            )
        out["resource_wait_option"] = wait_option
        extra_facts = {
            "post_feedback_action": "cpu_support",
            "arbiter_action": str((decision or {}).get("action") or "DENY_SHARE_USE_CPU_SUPPORT"),
        }
        if self.resource_runtime is not None:
            extra_facts.update(self.resource_runtime._resource_wait_extra_facts(wait_option))
        out["feedback"] = resource_feedback_text(
            status="LOCAL_GPU_BUSY_USE_CPU_SUPPORT",
            reason=reason or "shared GPU trial was not approved",
            scope="per_gpu",
            resource_mode=str(out.get("resource_mode") or "YELLOW"),
            blocked_class=str(out.get("policy_resource_class") or out.get("resource_class") or job.resource_class),
            gpu_ids=[str(x) for x in (out.get("gpu_ids") or job.gpu_ids) if str(x).strip()],
            allowed_classes=[str(x) for x in (out.get("allowed_classes") or []) if str(x).strip()],
            holder_job_id=holders[0] if holders else "",
            queue_position=(int(out.get("queue_position") or 0) if out.get("queue_position") is not None else None),
            queue_len=(int(out.get("queue_len") or 0) if out.get("queue_len") is not None else None),
            eta_next_train_sec=(max(0.0, float(out.get("eta_next_train_sec") or 0.0)) if out.get("eta_next_train_sec") is not None else None),
            eta_confidence=str(out.get("eta_confidence") or ""),
            unlock_condition="holder_released_or_shared_trial_granted",
            blocked_until_unlock=True,
            extra_facts=extra_facts,
        )
        return out

    async def admission_share_decide(
        self,
        job_id: str | None,
        *,
        admission_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = dict(admission_result or {})
        deterministic_trial_admission = (
            self.gpu_share_config.trial_admission_policy == "deterministic_grant_when_hard_gates_pass"
        )
        if not self.gpu_share_config.grant_active:
            return {"enabled": False, "reason": "admission_share_review_disabled"}
        if not deterministic_trial_admission and not (self.arbiter_enabled and self.arbiter_mode == "llm"):
            return {"enabled": False, "reason": "admission_share_review_disabled"}
        if not job_id or job_id not in self._jobs or self.resource_runtime is None:
            return {"enabled": False, "reason": "job_missing"}
        if not bool(result.get("share_override_candidate")):
            return {"enabled": False, "reason": "no_share_override_candidate"}
        job = self._jobs[job_id]
        now = time.time()
        cooldown_key = f"{job.job_id}:admission_share_review"
        cooldown = max(30.0, float(self.arbiter_proposal_coalesce_window_sec or 0.0))
        last = float(self._last_admission_share_review_emit.get(cooldown_key) or 0.0)
        if last and now - last < cooldown:
            return {"enabled": True, "reason": "admission_share_review_cooldown", "result": result}
        try:
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception as exc:
            return {"enabled": False, "reason": f"gpu_snapshot_failed:{type(exc).__name__}"}
        gpu_ids = [str(x) for x in (result.get("gpu_ids") or job.gpu_ids) if str(x).strip()]
        sample = self.resource_runtime.sample_gpu_util(gpu_ids=gpu_ids) if gpu_ids else {"available": False, "reason": "missing_gpu_ids", "gpus": []}
        observation = evaluate_admission_share_trial(
            cfg=self.gpu_share_config,
            admission_result=result,
            snapshot=snapshot,
            gpu_sample=sample if isinstance(sample, dict) else {},
        )
        if not observation.get("enabled"):
            return {"enabled": False, "reason": str(observation.get("reason") or "admission_share_not_enabled"), "observation": observation}
        self._last_admission_share_review_emit[cooldown_key] = now
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "admission_share_review_observed",
                payload={"job_id": job.job_id, "observation": observation, "admission_result": result},
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        if not observation.get("share_eligible"):
            return {"enabled": True, "reason": "admission_share_not_eligible", "result": result, "observation": observation}
        proposal = self._admission_share_review_proposal(job, result, observation)
        self._record_resource_review_proposal_event(job, proposal, {"elapsed_sec": 0.0})
        trial_share = observation.get("trial_share") if isinstance(observation.get("trial_share"), dict) else {}
        if deterministic_trial_admission and bool(trial_share.get("enabled")):
            decision = {
                "action": "GRANT_SHARED_GPU_LEASE",
                "reason": "hard resource gates passed for deterministic revocable admission trial",
                "confidence": "high",
                "source": "deterministic_trial_admission",
            }
            arbiter = {
                "enabled": True,
                "action": "GRANT_SHARED_GPU_LEASE",
                "decision": decision,
                "feedback": "RESOURCE_FEEDBACK: SHARED_TRIAL_STARTED because hard_resource_gates_passed; primary_protected=true; secondary_revocable=true.\n",
            }
        else:
            arbiter = await self.arbiter_decide(job.job_id, proposal=proposal, decision_preview=proposal.get("decision_preview"))
            if not isinstance(arbiter, dict) or not arbiter.get("enabled"):
                return {"enabled": True, "reason": "admission_share_arbiter_disabled", "result": result, "proposal": proposal}
            decision = arbiter.get("decision") if isinstance(arbiter.get("decision"), dict) else {}
        action = str(arbiter.get("action") or "").upper()
        if action == "GRANT_SHARED_GPU_LEASE":
            primary_id = str(observation.get("primary_job_id") or "")
            secondary_allowed = observation.get("secondary_allowed") if isinstance(observation.get("secondary_allowed"), dict) else {}
            trial_share = observation.get("trial_share") if isinstance(observation.get("trial_share"), dict) else {}
            grant = self.resource_runtime.grant_shared_gpu_lease(
                primary_job_id=primary_id,
                secondary_job_id=job.job_id,
                metadata={
                    "shared_grant_reason": "admission_task_gpu_share_review",
                    "shared_effective_slot_weight": float(secondary_allowed.get("effective_slot_weight") or 0.5),
                    "shared_policy_gate": str(secondary_allowed.get("policy_gate") or "admission_shared_trial"),
                    "trial_share": bool(trial_share.get("enabled")),
                    "trial_mode": str(trial_share.get("mode") or ""),
                    "trial_initial_observe_sec": float(trial_share.get("initial_observe_sec") or 0.0),
                    "trial_primary_protected": bool(trial_share.get("primary_protected")),
                    "admission_immediate_review": True,
                },
            )
            final = dict(result)
            final.update({
                "enabled": True,
                "acquired": bool(grant.get("acquired")),
                "status": "GRANTED" if grant.get("acquired") else "PENDING",
                "admission_action": "RUN_NOW" if grant.get("acquired") else "PENDING",
                "reason": str(grant.get("reason") or "shared_gpu_lease_granted"),
                "admission_share_reviewed": True,
                "admission_shared_trial": bool(grant.get("acquired")),
                "admission_share_action": action,
                "shared_lease_grant": grant,
            })
            if self.resource_runtime is not None:
                self.resource_runtime.record_resource_event(
                    "admission_share_decision",
                    payload={"job_id": job.job_id, "action": action, "decision": decision, "grant": grant, "proposal_id": proposal.get("proposal_id")},
                    proposal_id=str(proposal.get("proposal_id") or ""),
                    command_id=job.job_id,
                    lease_id=job.job_id,
                )
            return {"enabled": True, "action": action, "result": final, "proposal": proposal, "arbiter": arbiter}
        if action == "DENY_SHARE_USE_CPU_SUPPORT":
            final = self._admission_share_cpu_support_result(
                job,
                result,
                reason=str(decision.get("reason") or "shared GPU trial was not approved"),
                decision=decision,
            )
            if self.resource_runtime is not None:
                self.resource_runtime.record_resource_event(
                    "admission_share_decision",
                    payload={"job_id": job.job_id, "action": action, "decision": decision, "proposal_id": proposal.get("proposal_id")},
                    proposal_id=str(proposal.get("proposal_id") or ""),
                    command_id=job.job_id,
                    lease_id=job.job_id,
                )
            return {"enabled": True, "action": action, "result": final, "proposal": proposal, "arbiter": arbiter}
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "admission_share_decision",
                payload={"job_id": job.job_id, "action": action or "CONTINUE_SHARED_OBSERVE", "decision": decision, "proposal_id": proposal.get("proposal_id")},
                proposal_id=str(proposal.get("proposal_id") or ""),
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        final = dict(result)
        final["admission_share_reviewed"] = True
        final["admission_share_action"] = action or "CONTINUE_SHARED_OBSERVE"
        return {"enabled": True, "action": action or "CONTINUE_SHARED_OBSERVE", "result": final, "proposal": proposal, "arbiter": arbiter}

    async def admission_decide(
        self,
        job_id: str | None,
        *,
        admission_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.admission_llm_enabled or self.admission_llm_mode == "off" or self.admission_decider is None:
            return {"enabled": False, "reason": "admission_llm_disabled"}
        if not job_id or job_id not in self._jobs:
            return {"enabled": False, "reason": "job_missing"}
        result = dict(admission_result or {})
        if not should_request_admission_llm(result, mode=self.admission_llm_mode):
            return {"enabled": False, "reason": "rule_confident"}
        job = self._jobs[job_id]
        card = self._admission_task_card(job, result)
        raw: dict[str, Any] = {}
        source = "resource_admission_llm"
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "admission_llm_input",
                payload={
                    "actor_id": "resource_admission_arbiter",
                    "cache_namespace": "resource_admission",
                    "task_card": card,
                    "rule_result": result,
                },
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        try:
            maybe = self.admission_decider(card, result)
            if inspect.isawaitable(maybe):
                maybe = await asyncio.wait_for(maybe, timeout=self.admission_llm_timeout_sec)
            if isinstance(maybe, str):
                raw = parse_admission_decision_text(maybe)
            elif isinstance(maybe, dict):
                raw = dict(maybe)
        except Exception as exc:
            source = "admission_llm_error"
            raw = {"action": result.get("admission_action") or result.get("status") or "PENDING", "reason": f"admission_llm_failed:{type(exc).__name__}", "confidence": "low"}
        decision = normalize_admission_decision(raw, task_card=card, rule_result=result, source=source)
        result_for_apply = result
        action = str(decision.get("admission_action") or decision.get("action") or "").upper()
        if action in {"RUN_NOW", "OBSERVE_THEN_RUN"} and not result.get("acquired") and self.resource_runtime is not None:
            requested_gpu_ids = [str(x) for x in (decision.get("gpu_ids") or []) if str(x).strip()]
            opportunity = card.get("admission_opportunity") if isinstance(card.get("admission_opportunity"), dict) else {}
            if not requested_gpu_ids:
                requested_gpu_ids = [str(x) for x in (opportunity.get("grantable_gpu_ids") or result.get("candidate_physical_gpus") or result.get("allowed_physical_gpus") or job.gpu_ids) if str(x).strip()]
            request_count = len(requested_gpu_ids) if decision.get("gpu_ids") else int(result.get("requested_gpu_count") or job.gpu_request_count or len(requested_gpu_ids) or 1)
            grant = self.resource_runtime.grant_llm_admission_lease(
                job_id=job.job_id,
                resource_class=str(result.get("policy_resource_class") or result.get("resource_class") or job.resource_class),
                gpu_ids=requested_gpu_ids,
                request_count=request_count,
                metadata={
                    "command_digest": job.command_digest,
                    "command_excerpt": str(job.command or "")[:2000],
                    "entrypoint": self._command_entrypoint(job.command),
                    "value_hint": dict(job.value_hint or {}),
                    "admission_llm_decision_id": str(decision.get("decision_id") or ""),
                },
                observe_then_run=(action == "OBSERVE_THEN_RUN"),
                observe_sec=float(decision.get("observe_sec") or 0.0),
            )
            if grant.get("acquired"):
                result_for_apply = {**result, **grant, "acquired": True}
            else:
                result_for_apply = {**result, "admission_llm_lease_attempt": grant, "acquired": False}
                decision = {
                    **decision,
                    "action": "PENDING",
                    "status": "PENDING",
                    "admission_action": "PENDING",
                    "reason": "LLM approved GPU start, but atomic lease acquire failed: " + str(grant.get("reason") or "gpu_slot_unavailable"),
                }
        final_result = apply_admission_decision(result_for_apply, decision)
        if str(final_result.get("status") or "").upper() in {"PENDING", "REPLAN", "DEFERRED", "DENIED_REPLAN"}:
            allowed_classes = [str(x) for x in (final_result.get("allowed_classes") or []) if str(x).strip()]
            raw_feedback = str(final_result.get("feedback") or "")
            if not raw_feedback:
                raw_feedback = self._resource_feedback_text(
                    status=str(final_result.get("status") or "PENDING"),
                    reason=str(final_result.get("reason") or "resource_admission_review"),
                    scope=str(final_result.get("scope") or "per_gpu"),
                    resource_mode=str(final_result.get("resource_mode") or "YELLOW"),
                    blocked_class=job.resource_class,
                    gpu_ids=job.gpu_ids,
                    allowed_classes=allowed_classes,
                    eta_next_train_sec=float(final_result.get("eta_next_train_sec") or 0.0),
                    eta_confidence=str(final_result.get("eta_confidence") or "low"),
                    unlock_condition=str(final_result.get("unlock_condition") or "resource_context_changed"),
                    blocked_until_unlock=bool(final_result.get("blocked_until_unlock") if final_result.get("blocked_until_unlock") is not None else True),
                )
            feedback_state = self._dedupe_resource_feedback_for_agent(
                job,
                feedback=raw_feedback,
                status=str(final_result.get("status") or "PENDING"),
                reason=str(final_result.get("reason") or "resource_admission_review"),
                scope=str(final_result.get("scope") or "per_gpu"),
                resource_mode=str(final_result.get("resource_mode") or "YELLOW"),
                blocked_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                allowed_classes=allowed_classes,
                unlock_condition=str(final_result.get("unlock_condition") or "resource_context_changed"),
                blocked_until_unlock=bool(final_result.get("blocked_until_unlock") if final_result.get("blocked_until_unlock") is not None else True),
            )
            final_result["feedback"] = str(feedback_state.get("feedback") or "")
            final_result["feedback_suppressed"] = bool(feedback_state.get("feedback_suppressed"))
            final_result["feedback_state_key"] = feedback_state.get("feedback_state_key")
            final_result["resource_feedback_repeated_count"] = feedback_state.get("repeated_count")
            if not final_result["feedback_suppressed"]:
                self._remember_resource_feedback(
                    job,
                    status=str(final_result.get("status") or "PENDING"),
                    reason=str(final_result.get("reason") or "resource_admission_review"),
                    resource_mode=str(final_result.get("resource_mode") or "YELLOW"),
                    blocked_class=job.resource_class,
                    allowed_classes=allowed_classes,
                    eta_next_train_sec=float(final_result.get("eta_next_train_sec") or 0.0),
                    retry_after_sec=(float(final_result.get("retry_after_sec") or 0.0) if final_result.get("retry_after_sec") is not None else None),
                    post_feedback_action=str(final_result.get("post_feedback_action") or ""),
                )
        payload = {
            "actor_id": "resource_admission_arbiter",
            "cache_namespace": "resource_admission",
            "task_card": card,
            "rule_result": result,
            "decision": decision,
            "final_result": final_result,
        }
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "admission_llm_decision",
                payload=payload,
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        self._emit(
            "resource_admission_llm_decision",
            job,
            status=str(final_result.get("status") or decision.get("status") or "reviewed").lower(),
            payload=payload,
        )
        return {"enabled": True, "decision": decision, "result": final_result}

    def queue_wait_heartbeat(self, job_id: str | None, *, elapsed_sec: float, **_: Any) -> None:
        if not job_id or job_id not in self._jobs or self.resource_runtime is None:
            return
        heartbeat = self.resource_runtime.queue_wait_heartbeat(job_id=job_id, elapsed_sec=float(elapsed_sec or 0.0))
        if not heartbeat.get("emit"):
            return
        self._emit(
            "resource_gpu_queue_heartbeat",
            self._jobs[job_id],
            status="pending",
            payload={"elapsed_sec": float(elapsed_sec or 0.0)},
        )

    def lease_env_updates(self, job_id: str | None, **_: Any) -> dict[str, str]:
        if not job_id or self.resource_runtime is None:
            return {}
        return self.resource_runtime.env_updates(job_id=job_id)

    def queue_timeout(self, job_id: str | None, *, elapsed_sec: float, reason: str, **_: Any) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs.pop(job_id)
        self._gpu_queue_timeout_count += 1
        release: dict[str, Any] = {}
        pressure: dict[str, Any] = {}
        if self.resource_runtime is not None:
            release = self.resource_runtime.queue_timeout(
                job_id=job_id,
                elapsed_sec=float(elapsed_sec or 0.0),
                reason=str(reason or "queue_timeout"),
            )
            pressure = self.resource_runtime.record_queue_timeout_pressure(
                job_id=job_id,
                resource_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                elapsed_sec=float(elapsed_sec or 0.0),
                reason=str(reason or "queue_timeout"),
                command_digest=job.command_digest,
            )
        self._remember_resource_feedback(
            job,
            status="DENIED_REPLAN",
            reason=str(reason or "queue_timeout"),
            resource_mode="RED",
            blocked_class=job.resource_class,
            allowed_classes=[RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, "readonly_cpu", "light_cpu"],
            eta_next_train_sec=420.0,
            cooldown_sec=120.0,
        )
        self._emit(
            "resource_gpu_queue_timeout",
            job,
            status="timeout",
            payload={
                "elapsed_sec": float(elapsed_sec or 0.0),
                "reason": str(reason or "queue_timeout"),
                "released": bool(release.get("released")),
                "gpu_queue_timeout_count": self._gpu_queue_timeout_count,
                "pressure": pressure,
            },
        )

    def _maybe_emit_monitor_agent_shadow(self, job: ResourceJob, signal: dict[str, Any]) -> None:
        if self.monitor_agent_mode == "off" or not job.visible:
            return
        now = time.time()
        last = self._last_monitor_agent_emit.get(job.job_id, 0.0)
        if now - last < self.monitor_agent_min_interval_sec:
            return
        self._last_monitor_agent_emit[job.job_id] = now
        self._emit(
            "resource_monitor_agent_shadow",
            job,
            status="observed",
            payload={
                "mode": self.monitor_agent_mode,
                "filtered_signal": dict(signal),
                "decision_provider": "reserved_interface",
                "decision_applied": False,
            },
        )

    def _maybe_emit_gpu_util_sample(self, job: ResourceJob, *, elapsed_sec: float) -> None:
        if not self.gpu_util_observer_enabled or self.resource_runtime is None:
            return
        if not job.gpu_ids:
            return
        now = time.time()
        last = self._last_gpu_util_emit.get(job.job_id, 0.0)
        if now - last < self.gpu_util_sample_interval_sec:
            return
        self._last_gpu_util_emit[job.job_id] = now
        sample = self.resource_runtime.sample_gpu_util(gpu_ids=job.gpu_ids)
        if not isinstance(sample, dict):
            sample = {"available": False, "reason": "invalid_sample", "gpus": []}
        process_gpu_placement = self._job_process_gpu_placement(job)
        process_gpu_observed = process_gpu_placement.get("available") is True
        process_has_gpu = self._process_gpu_placement_has_usage(process_gpu_placement)
        if process_gpu_observed:
            idle_now = (not process_has_gpu) or self._gpu_util_sample_is_idle(sample, job.gpu_ids)
            low_compute_now = process_has_gpu and self._gpu_util_sample_is_low_compute_with_model(sample, job.gpu_ids)
        else:
            idle_now = self._gpu_util_sample_is_idle(sample, job.gpu_ids)
            low_compute_now = self._gpu_util_sample_is_low_compute_with_model(sample, job.gpu_ids)
        mem_summary = gpu_memory_summary(sample, job.gpu_ids, previous_peak_gb=job.gpu_mem_peak_gb)
        job.gpu_mem_peak_gb = float(mem_summary.get("gpu_mem_peak_gb") or job.gpu_mem_peak_gb or 0.0)
        job.idle_gpu_lease_samples = job.idle_gpu_lease_samples + 1 if idle_now else 0
        job.dataloader_bottleneck_samples = job.dataloader_bottleneck_samples + 1 if low_compute_now else 0
        guard_snapshot = {
            "idle_now": idle_now,
            "idle_samples": job.idle_gpu_lease_samples,
            "min_samples": self.gpu_idle_lease_min_samples,
            "util_threshold_pct": self.gpu_idle_lease_util_pct,
            "mem_threshold_gb": self.gpu_idle_lease_mem_gb,
            "dataloader_low_compute_now": low_compute_now,
            "dataloader_low_compute_samples": job.dataloader_bottleneck_samples,
            "dataloader_util_threshold_pct": self.gpu_dataloader_bottleneck_util_pct,
            "dataloader_min_mem_gb": self.gpu_dataloader_bottleneck_min_mem_gb,
            "gpu_mem_peak_gb": job.gpu_mem_peak_gb,
            "process_gpu_observed": process_gpu_observed,
            "process_has_gpu": process_has_gpu,
        }
        job.last_gpu_util_sample = {
            "elapsed_sec": float(elapsed_sec or 0.0),
            "sample": sample,
            "process_gpu_placement": process_gpu_placement,
            "idle_gpu_lease_guard": guard_snapshot,
        }
        self._emit(
            "resource_gpu_util_sampled",
            job,
            status="observed",
            payload={
                "elapsed_sec": float(elapsed_sec or 0.0),
                "sample": sample,
                "pressure": sample.get("pressure") if isinstance(sample.get("pressure"), dict) else {},
                "process_gpu_placement": process_gpu_placement,
                "idle_gpu_lease_guard": guard_snapshot,
            },
        )

    def progress_heartbeat(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float = 0.0,
        phase: str = "",
        signals: dict[str, Any] | None = None,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        metric_history_text: str = "",
        metric_history_line_count: int = 0,
        emit_reason: str = "",
        **_: Any,
    ) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        elapsed = float(elapsed_sec or 0.0)
        stdout_line_count = int(stdout_lines or 0)
        stdout_byte_count = int(stdout_bytes or 0)
        metric_text = str(metric_history_text or "").strip()
        metric_lines = int(metric_history_line_count or 0)
        signals_payload = dict(signals or {})
        previous_progress_payload = job.last_progress if isinstance(job.last_progress, dict) else {}
        structured_progress = self._structured_progress_snapshot(
            signals_payload,
            previous_progress_payload,
            current_elapsed_sec=elapsed,
        )
        structured_metric_line = metric_history_line_from_progress_signals(signals_payload)
        if structured_metric_line and not metric_text:
            existing_text = str(job.metric_history_text or "").strip()
            existing_lines = existing_text.splitlines() if existing_text else []
            merged_lines = update_metric_history_lines(existing_lines, structured_metric_line)
            metric_text = compact_metric_history_text(merged_lines)
            metric_lines = len(merged_lines)
        payload = {
            "elapsed_sec": elapsed,
            "phase": str(phase or ""),
            "signals": signals_payload,
            "stdout_lines": stdout_line_count,
            "stdout_bytes": stdout_byte_count,
            "metric_history_text": metric_text,
            "metric_history_line_count": metric_lines,
            "emit_reason": str(emit_reason or ""),
            "resource_efficiency": {
                "stdout_lines_per_sec": round(stdout_line_count / max(1.0, elapsed), 6),
                "stdout_bytes_per_sec": round(stdout_byte_count / max(1.0, elapsed), 6),
            },
        }
        if structured_progress:
            payload["structured_progress"] = structured_progress
        job.last_progress = payload
        if metric_text:
            job.metric_history_text = metric_text
            job.metric_history_line_count = metric_lines
        signals_payload = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
        artifact_payload = signals_payload.get("artifact") if isinstance(signals_payload.get("artifact"), dict) else {}
        heartbeat = signals_payload.get("heartbeat") if isinstance(signals_payload.get("heartbeat"), dict) else {}
        if artifact_payload and heartbeat:
            for key in ("route_id", "fold", "validation_protocol", "checkpoint_kind", "safe_to_resume", "mergeable"):
                value = heartbeat.get(key)
                if value not in {None, ""} and key not in artifact_payload:
                    artifact_payload[key] = value
        if artifact_payload:
            job.last_artifact_progress = payload
            job.last_recoverability = self._recoverability_from_artifact(job, artifact_payload, elapsed_sec=elapsed)
        self._emit("progress_heartbeat", job, status="observed", payload=payload)

    def stdout_heartbeat(self, job_id: str | None) -> None:
        _ = job_id

    def _structured_progress_snapshot(
        self,
        signals: dict[str, Any],
        previous_payload: dict[str, Any] | None,
        *,
        current_elapsed_sec: float,
    ) -> dict[str, Any]:
        current = self._structured_progress_entry(signals)
        if not current:
            return {}
        previous = (previous_payload or {}).get("structured_progress")
        previous = previous if isinstance(previous, dict) else {}
        previous_current = self._float_or_none(previous.get("current"))
        previous_total = self._float_or_none(previous.get("total"))
        same_unit = str(previous.get("unit") or "") == str(current.get("unit") or "")
        same_total = bool(
            previous_total is not None
            and abs(float(previous_total) - float(current["total"])) <= max(1e-9, abs(float(current["total"])) * 1e-6)
        )
        advanced = bool(same_unit and same_total and previous_current is not None and float(current["current"]) > previous_current)
        out = dict(current)
        previous_samples = int(previous.get("interval_sample_count") or 0) if same_unit and same_total else 0
        out["advanced"] = advanced
        out["interval_sample_count"] = previous_samples + 1 if advanced else previous_samples
        if same_unit and previous_current is not None:
            out["previous_current"] = previous_current
        if same_unit and previous_total is not None:
            out["previous_total"] = previous_total
        previous_elapsed = self._float_or_none((previous_payload or {}).get("elapsed_sec"))
        current_elapsed = self._float_or_none((signals.get("heartbeat") or {}).get("elapsed_s"))
        if current_elapsed is None:
            current_elapsed = max(0.0, float(current_elapsed_sec or 0.0))
        if (
            advanced
            and previous_current is not None
            and previous_elapsed is not None
            and current_elapsed is not None
            and current_elapsed > previous_elapsed
        ):
            interval_sec = current_elapsed - previous_elapsed
            rate = (float(current["current"]) - previous_current) / interval_sec
            if rate > 0.0:
                out["interval_sec"] = interval_sec
                out["rate_units_per_sec"] = rate
                out["eta_to_phase_end_sec"] = max(0.0, (float(current["total"]) - float(current["current"])) / rate)
        return out

    @staticmethod
    def _progress_unit_scope(unit: Any) -> str:
        normalized = str(unit or "").strip().lower()
        if normalized in {"batch", "batches", "step", "steps", "iteration", "iterations"}:
            return "subphase"
        if normalized in {
            "overall",
            "run",
            "runs",
            "task",
            "tasks",
            "trial",
            "trials",
            "fold",
            "folds",
            "epoch",
            "epochs",
            "round",
            "rounds",
        }:
            return "route"
        return "phase"

    def _structured_progress_entry(self, signals: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(signals, dict):
            return {}
        evidence = signals.get("progress_evidence") if isinstance(signals.get("progress_evidence"), dict) else {}
        best: dict[str, Any] = {}
        scope_rank = {"subphase": 1, "phase": 2, "route": 3}
        for unit, raw_entry in signals.items():
            if unit in {"heartbeat", "metrics", "artifact", "phase"} or not isinstance(raw_entry, dict):
                continue
            current = self._float_or_none(raw_entry.get("current"))
            total = self._float_or_none(raw_entry.get("total"))
            if current is None or total is None or total <= 0.0:
                continue
            if current < 0.0 or current > total:
                continue
            progress_scope = self._progress_unit_scope(unit)
            candidate = {
                "unit": str(unit),
                "current": current,
                "total": total,
                "progress_scope": progress_scope,
                "source": str(evidence.get("source") or raw_entry.get("source") or "unknown"),
                "evidence_trust": str(evidence.get("trust") or "unknown"),
            }
            if not best or scope_rank[progress_scope] > scope_rank[str(best["progress_scope"])]:
                best = candidate
        return best

    @staticmethod
    def _cpu_bucket(process_tree_cpu: dict[str, Any] | None) -> str:
        cpu = process_tree_cpu if isinstance(process_tree_cpu, dict) else {}
        if not cpu or not cpu.get("available", True):
            return "unknown"
        try:
            total_cpu = float(cpu.get("total_cpu_pct") or 0.0)
        except (TypeError, ValueError):
            total_cpu = 0.0
        try:
            busy_children = int(float(cpu.get("busy_child_count") or 0.0))
        except (TypeError, ValueError):
            busy_children = 0
        if total_cpu >= 300.0 or busy_children >= 2:
            return "heavy"
        if total_cpu >= 50.0 or busy_children >= 1:
            return "active"
        return "idle"

    def monitor_heartbeat(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float = 0.0,
        stdout_age_sec: float = 0.0,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        pid: int | None = None,
        returncode: int | None = None,
        process_tree_cpu: dict[str, Any] | None = None,
        source: str = "bash_guard",
        **_: Any,
    ) -> dict[str, Any]:
        if not job_id:
            return {"recorded": False, "reason": "missing_job_id"}
        if job_id not in self._jobs:
            return {"recorded": False, "reason": "missing_job"}
        job = self._jobs[job_id]
        elapsed = max(0.0, float(elapsed_sec or 0.0))
        interval = max(0.05, float(self.review_heartbeat_sec or 60.0))
        if job.monitor_heartbeat_count > 0 and elapsed < float(job.last_monitor_heartbeat_elapsed_sec or 0.0) + interval:
            return {"recorded": False, "reason": "heartbeat_interval", "next_after_sec": interval}
        if pid is not None:
            try:
                job.pid = int(pid)
                if job.pgid is None:
                    job.pgid = int(pid)
            except (TypeError, ValueError):
                pass
        if elapsed >= self.min_register_sec:
            self._promote(job, reason="monitor_heartbeat", elapsed_sec=elapsed)
        process_alive = returncode is None
        artifact_age = self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed)
        progress_age = self._elapsed_since_progress(job.last_progress, elapsed_sec=elapsed)
        recoverability = self._fresh_recoverability_for_job(job, elapsed_sec=elapsed)
        monitor_signal = {
            "elapsed_sec": elapsed,
            "stdout_age_sec": max(0.0, float(stdout_age_sec or 0.0)),
            "stdout_lines": int(stdout_lines or 0),
            "stdout_bytes": int(stdout_bytes or 0),
            "process_tree_cpu": dict(process_tree_cpu or {}),
        }
        stdout_observation = self._stdout_observation_for_job(job, monitor_signal, elapsed_sec=elapsed)
        review_state = self._review_states.get(job.job_id)
        payload = {
            "elapsed_sec": elapsed,
            "process_alive": process_alive,
            "pid": job.pid,
            "pgid": job.pgid,
            "stdout_age_sec": max(0.0, float(stdout_age_sec or 0.0)),
            "stdout_lines": int(stdout_lines or 0),
            "stdout_bytes": int(stdout_bytes or 0),
            "cpu_bucket": self._cpu_bucket(process_tree_cpu),
            "gpu_bucket": "active" if self._last_gpu_sample_active(job) else "idle_or_none",
            "artifact_age_sec": artifact_age,
            "progress_age_sec": progress_age,
            "recoverability": recoverability,
            "stdout_observation": stdout_observation,
            "heartbeat_count": int(job.monitor_heartbeat_count) + 1,
            "source": str(source or "bash_guard"),
            "process_tree_cpu": dict(process_tree_cpu or {}),
            "review_state": review_state.to_json() if review_state is not None else {},
        }
        job.monitor_heartbeat_count += 1
        job.last_monitor_heartbeat_elapsed_sec = elapsed
        self._emit("resource_monitor_heartbeat", job, status="observed", payload=payload)
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_monitor_heartbeat",
                payload=payload,
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return {"recorded": True, "heartbeat_count": job.monitor_heartbeat_count}

    def resource_monitor_gap(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float = 0.0,
        gap_sec: float = 0.0,
        stdout_age_sec: float = 0.0,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        reason: str = "",
        source: str = "bash_guard_watchdog",
        process_tree_cpu: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not job_id:
            return {"recorded": False, "reason": "missing_job_id"}
        if job_id not in self._jobs:
            return {"recorded": False, "reason": "missing_job"}
        job = self._jobs[job_id]
        payload = {
            "elapsed_sec": max(0.0, float(elapsed_sec or 0.0)),
            "gap_sec": max(0.0, float(gap_sec or 0.0)),
            "stdout_age_sec": max(0.0, float(stdout_age_sec or 0.0)),
            "stdout_lines": int(stdout_lines or 0),
            "stdout_bytes": int(stdout_bytes or 0),
            "heartbeat_count": int(job.monitor_heartbeat_count or 0),
            "last_monitor_heartbeat_elapsed_sec": float(job.last_monitor_heartbeat_elapsed_sec or 0.0),
            "reason": str(reason or "resource_monitor_gap"),
            "source": str(source or "bash_guard_watchdog"),
            "pid": job.pid,
            "pgid": job.pgid,
            "process_tree_cpu": dict(process_tree_cpu or {}),
        }
        self._emit("resource_monitor_gap", job, status="observed", payload=payload)
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_monitor_gap",
                payload=payload,
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return {"recorded": True, "gap_sec": payload["gap_sec"]}

    def resume_monitor_event(
        self,
        event: str,
        *,
        tool_name: str = "",
        tool_call_id: str = "",
        command: str = "",
        status: str = "",
        payload: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        body = {
            "event": str(event or "resume_monitor_event"),
            "tool_name": str(tool_name or ""),
            "tool_call_id": str(tool_call_id or ""),
            "command": str(command or "")[:1000],
            **dict(payload or {}),
        }
        self.state_machine.append_event(
            "resource_resume_monitor_event",
            task_type="resource_resume",
            task_id=f"resource_resume:{tool_call_id or event or 'unknown'}",
            status=str(status or event or "observed"),
            payload=body,
        )
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_resume_monitor_event",
                payload=body,
                command_id=str(tool_call_id or ""),
                lease_id=str(tool_call_id or ""),
            )

    @staticmethod
    def _elapsed_since_progress(payload: dict[str, Any], *, elapsed_sec: float) -> float:
        if not isinstance(payload, dict) or not payload:
            return max(0.0, float(elapsed_sec or 0.0))
        try:
            last_elapsed = float(payload.get("elapsed_sec") or 0.0)
        except (TypeError, ValueError):
            last_elapsed = 0.0
        if last_elapsed <= 0.0:
            return max(0.0, float(elapsed_sec or 0.0))
        return max(0.0, float(elapsed_sec or 0.0) - last_elapsed)

    @staticmethod
    def _artifact_update_is_log_like(artifact: dict[str, Any]) -> bool:
        path = str((artifact or {}).get("path") or (artifact or {}).get("name") or "").strip().lower()
        if not path:
            return False
        name = path.rsplit("/", 1)[-1]
        if name.endswith(".log"):
            return True
        return bool(name.endswith(".txt") and any(token in name for token in ("log", "stdout", "stderr", "trace")))

    @classmethod
    def _artifact_update_is_value_bearing(
        cls,
        artifact: dict[str, Any],
        *,
        deliverable_validity: str = "none",
    ) -> bool:
        if not isinstance(artifact, dict) or cls._artifact_update_is_log_like(artifact):
            return False
        path = str(artifact.get("path") or artifact.get("name") or "").strip().lower()
        if not path:
            return False
        try:
            if int(artifact.get("size_bytes") or 0) <= 0:
                return False
        except (TypeError, ValueError):
            return False
        stability = str(artifact.get("stability") or "").strip().lower()
        if stability not in {"stable", "done_marker", "run_state_confirmed"}:
            return False
        scope = str(artifact.get("artifact_scope") or "").strip().lower()
        if scope not in {"current_run", "workspace_recent"}:
            return False
        name = path.rsplit("/", 1)[-1]
        if name in {"submission.csv", "predictions.csv"}:
            return str(deliverable_validity or "").strip().lower() == "produced_valid"
        suffix = Path(name).suffix.lower()
        if suffix in {".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".joblib", ".npy", ".npz"}:
            return True
        if suffix == ".csv" and any(
            token in name for token in ("submission", "submit", "pred", "prediction", "oof", "logit", "prob")
        ):
            return True
        return False

    def _fresh_artifact_update_for_job(self, job: ResourceJob, *, elapsed_sec: float) -> dict[str, Any]:
        payload = job.last_artifact_progress if isinstance(job.last_artifact_progress, dict) else {}
        signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
        artifact = signals.get("artifact") if isinstance(signals.get("artifact"), dict) else {}
        if not artifact:
            return {}
        out = dict(artifact)
        payload_elapsed = self._float_or_none(payload.get("elapsed_sec"))
        base_age = self._float_or_none(out.get("age_sec"))
        if payload_elapsed is not None and payload_elapsed > 0.0:
            delta = max(0.0, float(elapsed_sec or 0.0) - payload_elapsed)
            out["age_sec"] = max(0.0, float(base_age or 0.0) + delta)
            out["observed_at_elapsed_sec"] = payload_elapsed
        else:
            out["age_sec"] = self._elapsed_since_progress(payload, elapsed_sec=elapsed_sec)
        out["age_refreshed_at_elapsed_sec"] = max(0.0, float(elapsed_sec or 0.0))
        out["artifact_log_like"] = self._artifact_update_is_log_like(out)
        return out

    def _fresh_recoverability_for_job(self, job: ResourceJob, *, elapsed_sec: float) -> dict[str, Any]:
        rec = dict(job.last_recoverability or {}) if isinstance(job.last_recoverability, dict) else {}
        if not rec:
            return {}
        observed = self._float_or_none(rec.get("observed_at_elapsed_sec"))
        if observed is None or observed <= 0.0:
            payload = job.last_artifact_progress if isinstance(job.last_artifact_progress, dict) else {}
            observed = self._float_or_none(payload.get("elapsed_sec"))
        delta = max(0.0, float(elapsed_sec or 0.0) - float(observed or elapsed_sec or 0.0))
        for key in ("artifact_age_sec", "last_recoverable_artifact_age_sec"):
            if rec.get(key) is None:
                continue
            age = self._float_or_none(rec.get(key))
            if age is not None:
                rec[key] = max(0.0, age + delta)
        if rec.get("artifact_age_sec") is None:
            artifact = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
            if artifact:
                rec["artifact_age_sec"] = artifact.get("age_sec")
        rec["age_refreshed_at_elapsed_sec"] = max(0.0, float(elapsed_sec or 0.0))
        return rec

    def _stdout_observation_for_job(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> dict[str, Any]:
        stdout_age = self._float_or_none(signal.get("stdout_age_sec"))
        stdout_lines = int(signal.get("stdout_lines") or 0)
        stdout_bytes = int(signal.get("stdout_bytes") or 0)
        stream_present = bool(stdout_lines > 0 or stdout_bytes > 0)
        stream_limit = max(0.0, float(self.stalled_stdout_sec or 0.0))
        stream_fresh = bool(stream_present and (stdout_age is None or stream_limit <= 0.0 or stdout_age < stream_limit))
        artifact = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
        redirected_present = bool(artifact and self._artifact_update_is_log_like(artifact))
        redirected_age = self._float_or_none(artifact.get("age_sec")) if artifact else None
        redirected_limit = max(0.0, float(self.low_progress_no_artifact_sec or self.stalled_stdout_sec or 300.0))
        if redirected_limit <= 0.0:
            redirected_limit = 300.0
        redirected_fresh = bool(redirected_present and redirected_age is not None and redirected_age < redirected_limit)
        if stream_present:
            primary = "stdout_stream"
        elif redirected_present:
            primary = "redirected_log"
        else:
            primary = "none"
        return {
            "primary_channel": primary,
            "stdout_stream": {
                "present": stream_present,
                "fresh": stream_fresh,
                "age_sec": stdout_age,
                "lines": stdout_lines,
                "bytes": stdout_bytes,
            },
            "redirected_log": {
                "present": redirected_present,
                "fresh": redirected_fresh,
                "age_sec": redirected_age,
                "path": str(artifact.get("path") or "") if artifact else "",
                "size_bytes": int(artifact.get("size_bytes") or 0) if artifact else 0,
            },
            "stalled": bool((stream_present and not stream_fresh) or (redirected_present and not redirected_fresh)),
        }

    def _job_uses_expensive_gpu(self, job: ResourceJob) -> bool:
        if bool(getattr(job.source_hint, "command_cpu_only", False)):
            return False
        cls = str(job.resource_class or "")
        if cls in {RESOURCE_PURE_TT_CPU, RESOURCE_HEAVY_CPU_CANDIDATE, RESOURCE_UNKNOWN_EXEC, RESOURCE_LIGHT_CPU}:
            return False
        if self.resource_runtime is not None and self.resource_runtime.has_active_lease(job_id=job.job_id):
            return True
        if job.gpu_queue_relevant:
            return True
        if cls in {RESOURCE_GPU_FEATURE_EXTRACT, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_GPU_TT_LIGHT, RESOURCE_UNKNOWN_GPU_EXEC}:
            return bool(job.gpu_ids) or self._has_gpu_intent(job.source_hint)
        if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
            return bool(job.gpu_ids) or self._has_gpu_intent(job.source_hint)
        return False

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if out == out else None

    def _progress_finish_feasibility(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        payload = job.last_progress if isinstance(job.last_progress, dict) else {}
        signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
        phase = str(signal.get("current_phase") or payload.get("phase") or signals.get("phase") or "").strip().lower()
        best: dict[str, Any] = {}
        heartbeat = signals.get("heartbeat") if isinstance(signals.get("heartbeat"), dict) else {}
        heartbeat_elapsed = self._float_or_none(heartbeat.get("elapsed_s"))
        payload_elapsed = self._float_or_none(payload.get("elapsed_sec"))
        elapsed = max(0.0, float(elapsed_sec or 0.0), float(payload_elapsed or 0.0), float(heartbeat_elapsed or 0.0))
        structured = payload.get("structured_progress") if isinstance(payload.get("structured_progress"), dict) else {}
        for unit, raw_entry in signals.items():
            if unit in {"heartbeat", "metrics", "artifact", "phase"} or not isinstance(raw_entry, dict):
                continue
            current = self._float_or_none(raw_entry.get("current"))
            total = self._float_or_none(raw_entry.get("total"))
            if current is None or total is None or total <= 0.0 or current < 0.0 or current > total:
                continue
            if current <= 0.0 or elapsed <= 0.0:
                continue
            structured_rate = self._float_or_none(structured.get("rate_units_per_sec"))
            structured_matches = bool(
                str(structured.get("unit") or "") == str(unit)
                and self._float_or_none(structured.get("total")) == total
                and self._float_or_none(structured.get("current")) == current
            )
            rate = float(structured_rate) if structured_matches and structured_rate and structured_rate > 0.0 else current / elapsed
            if rate <= 0.0:
                continue
            interval_samples = int(structured.get("interval_sample_count") or 0) if structured_matches else 0
            progress_source = str(structured.get("source") or raw_entry.get("source") or "unknown")
            evidence_trust = str(structured.get("evidence_trust") or "unknown")
            eta = max(0.0, (total - current) / rate)
            candidate = {
                "progress_unit": str(unit),
                "progress_scope": self._progress_unit_scope(unit),
                "progress_units_done": current,
                "progress_units_total": total,
                "progress_rate_units_per_sec": rate,
                "eta_to_current_phase_end_sec": eta,
                "eta_source": f"progress_interval:{unit}" if structured_matches and structured_rate else f"progress_heartbeat:{unit}",
                "eta_confidence": "high" if interval_samples >= 2 else "medium",
                "progress_interval_samples": interval_samples,
                "progress_source": progress_source,
                "progress_evidence_trust": evidence_trust,
            }
            candidate_rank = {"subphase": 1, "phase": 2, "route": 3}[str(candidate["progress_scope"])]
            best_rank = {"subphase": 1, "phase": 2, "route": 3}.get(str(best.get("progress_scope") or ""), 0)
            if not best or candidate_rank > best_rank or (
                candidate_rank == best_rank
                and eta > float(best.get("eta_to_current_phase_end_sec") or 0.0)
            ):
                best = candidate
        if not best:
            return {
                "progress_unit": "",
                "progress_units_done": None,
                "progress_units_total": None,
                "progress_rate_units_per_sec": None,
                "eta_to_current_phase_end_sec": None,
                "eta_to_next_comparable_metric_sec": None,
                "eta_to_deliverable_sec": None,
                "eta_source": "none",
                "eta_confidence": "low",
                "finish_feasible": "unknown",
                "finish_feasibility_reason": "eta_unavailable",
                "remaining_useful_budget_sec": None,
                "progress_fraction": None,
                "structured_progress_recent": False,
                "phase_completion_protected": False,
            }
        progress_fraction = float(best["progress_units_done"]) / max(float(best["progress_units_total"]), 1.0)
        progress_age = max(0.0, elapsed - float(payload_elapsed or 0.0))
        progress_recent = bool(payload_elapsed is not None and progress_age <= max(120.0, self.review_heartbeat_sec * 2.0))
        eta_phase = float(best.get("eta_to_current_phase_end_sec") or 0.0)
        eta_source = str(best.get("eta_source") or "")
        terminal_phase = any(token in phase for token in (
            "valid",
            "eval",
            "infer",
            "predict",
            "submission",
            "score",
            "final",
            "test",
        ))
        best.update({
            "progress_fraction": progress_fraction,
            "structured_progress_recent": progress_recent,
            "structured_progress_age_sec": progress_age,
            "phase_completion_protected": bool(
                progress_recent
                and (str(best.get("progress_scope") or "") != "subphase" or terminal_phase)
                and (
                    str(best.get("eta_confidence") or "") == "high"
                    or (
                        str(best.get("eta_confidence") or "") == "medium"
                        and int(best.get("progress_interval_samples") or 0) >= 1
                        and progress_fraction >= 0.9
                    )
                )
                and eta_phase <= 300.0
                and (eta_source.startswith("progress_interval:") or progress_fraction >= 0.9)
                and not bool(signal.get("deadline_event"))
            ),
            "phase_completion_protection_sec": 300.0,
        })
        remaining = self._float_or_none(signal.get("deadline_remaining_sec"))
        metric_phase = any(token in phase for token in ("train", "fit", "epoch", "valid", "eval", "score", "oof"))
        best["eta_to_next_comparable_metric_sec"] = (
            float(best.get("eta_to_current_phase_end_sec") or 0.0)
            if metric_phase and str(best.get("progress_scope") or "") != "subphase"
            else None
        )
        reserve = max(0.0, float(self._float_or_none(signal.get("finalization_reserve_sec")) or 0.0))
        if remaining is None:
            best.update({
                "eta_to_deliverable_sec": None,
                "finish_feasible": "unknown",
                "finish_feasibility_reason": "budget_unknown",
                "remaining_useful_budget_sec": None,
            })
            return best
        useful_budget = max(0.0, remaining - reserve)
        eta_phase = float(best.get("eta_to_current_phase_end_sec") or 0.0)
        if eta_phase > useful_budget:
            finish_feasible: bool | str = False
            reason = "eta_exceeds_remaining_useful_budget"
            eta_deliverable: float | None = eta_phase
        elif terminal_phase:
            finish_feasible = True
            reason = "terminal_phase_eta_within_remaining_useful_budget"
            eta_deliverable = eta_phase
        else:
            finish_feasible = "unknown"
            reason = "phase_eta_available_deliverable_eta_unknown"
            eta_deliverable = None
        best.update({
            "eta_to_deliverable_sec": eta_deliverable,
            "finish_feasible": finish_feasible,
            "finish_feasibility_reason": reason,
            "remaining_useful_budget_sec": useful_budget,
        })
        return best

    @staticmethod
    def _process_gpu_placement_has_usage(placement: dict[str, Any]) -> bool:
        if not isinstance(placement, dict) or placement.get("available") is not True:
            return False
        used = placement.get("used_gpu_processes")
        return bool(isinstance(used, list) and used)

    @staticmethod
    def _process_gpu_placement_mem_mb(placement: dict[str, Any]) -> float:
        if not isinstance(placement, dict) or placement.get("available") is not True:
            return 0.0
        total = 0.0
        for row in placement.get("used_gpu_processes") or []:
            if not isinstance(row, dict):
                continue
            try:
                total += max(0.0, float(row.get("used_memory_mb") or 0.0))
            except (TypeError, ValueError):
                pass
        return total

    @staticmethod
    def _process_gpu_memory_by_gpu(placement: dict[str, Any]) -> dict[str, float]:
        if not isinstance(placement, dict) or placement.get("available") is not True:
            return {}
        out: dict[str, float] = {}
        for row in placement.get("used_gpu_processes") or []:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or "").strip()
            if not gpu_id:
                continue
            try:
                used_mb = max(0.0, float(row.get("used_memory_mb") or 0.0))
            except (TypeError, ValueError):
                used_mb = 0.0
            out[gpu_id] = out.get(gpu_id, 0.0) + used_mb
        return out

    def _job_process_gpu_placement(self, job: ResourceJob) -> dict[str, Any]:
        if not job.pid:
            return {"available": False, "reason": "missing_pid", "violations": []}
        try:
            return process_tree_gpu_placement_snapshot(job.pid, job.gpu_ids)
        except Exception as exc:
            return {"available": False, "reason": type(exc).__name__, "violations": []}

    def _gpu_util_sample_is_idle(self, sample: dict[str, Any], gpu_ids: list[str]) -> bool:
        if not isinstance(sample, dict) or sample.get("available") is False:
            return False
        wanted = {str(x).strip() for x in (gpu_ids or []) if str(x).strip()}
        if not wanted:
            return False
        rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
            if gpu_id not in wanted:
                continue
            util = self._float_or_none(row.get("utilization_gpu_pct"))
            used_mb = self._float_or_none(row.get("memory_used_mb"))
            if util is None or used_mb is None:
                return False
            used_gb = max(0.0, used_mb / 1024.0)
            if util > self.gpu_idle_lease_util_pct or used_gb > self.gpu_idle_lease_mem_gb:
                return False
            seen.add(gpu_id)
        return seen == wanted

    def _gpu_util_sample_is_low_compute_with_model(self, sample: dict[str, Any], gpu_ids: list[str]) -> bool:
        if not isinstance(sample, dict) or sample.get("available") is False:
            return False
        wanted = {str(x).strip() for x in (gpu_ids or []) if str(x).strip()}
        if not wanted:
            return False
        rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
            if gpu_id not in wanted:
                continue
            util = self._float_or_none(row.get("utilization_gpu_pct"))
            used_mb = self._float_or_none(row.get("memory_used_mb"))
            if util is None or used_mb is None:
                return False
            used_gb = max(0.0, used_mb / 1024.0)
            if util > self.gpu_dataloader_bottleneck_util_pct or used_gb < self.gpu_dataloader_bottleneck_min_mem_gb:
                return False
            seen.add(gpu_id)
        return seen == wanted

    def _gpu_pressure_active_for_job(self, job: ResourceJob) -> dict[str, Any]:
        if not self.gpu_idle_lease_require_pressure:
            return {"active": True, "reason": "pressure_not_required"}
        if self.resource_runtime is None or not job.gpu_ids:
            return {"active": False, "reason": "pressure_unavailable"}
        try:
            snapshot = self.resource_runtime.pressure_snapshot(gpu_ids=job.gpu_ids)
        except Exception:
            return {"active": False, "reason": "pressure_snapshot_failed"}
        gpus = snapshot.get("gpus") if isinstance(snapshot.get("gpus"), dict) else {}
        for gpu_id, raw in gpus.items():
            if not isinstance(raw, dict):
                continue
            mode = str(raw.get("mode") or "").upper()
            queue_len = int(raw.get("queue_len") or 0)
            yellow_remaining = float(raw.get("yellow_remaining_sec") or 0.0)
            cooldown_remaining = float(raw.get("cooldown_remaining_sec") or 0.0)
            if mode in {"YELLOW", "RED"} or queue_len > 0 or yellow_remaining > 0 or cooldown_remaining > 0:
                if mode in {"YELLOW", "RED"}:
                    reason = mode.lower()
                elif cooldown_remaining > 0:
                    reason = "red_cooldown"
                elif yellow_remaining > 0:
                    reason = "yellow_hold"
                else:
                    reason = "queue_pressure"
                return {
                    "active": True,
                    "reason": reason,
                    "gpu_id": str(gpu_id),
                    "snapshot": snapshot,
                }
        return {"active": False, "reason": "no_gpu_pressure", "snapshot": snapshot}


    def _last_gpu_sample_active(self, job: ResourceJob) -> bool:
        sample_bundle = job.last_gpu_util_sample if isinstance(job.last_gpu_util_sample, dict) else {}
        placement = sample_bundle.get("process_gpu_placement") if isinstance(sample_bundle.get("process_gpu_placement"), dict) else {}
        if placement.get("available") is True and not self._process_gpu_placement_has_usage(placement):
            return False
        sample = sample_bundle.get("sample") if isinstance(sample_bundle.get("sample"), dict) else {}
        rows = sample.get("gpus") if isinstance(sample, dict) and isinstance(sample.get("gpus"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            util = self._float_or_none(row.get("utilization_gpu_pct"))
            if util is not None and util > max(1.0, float(self.gpu_idle_lease_util_pct or 0.0)):
                return True
        return False

    def _contention_context_for_job(self, job: ResourceJob) -> dict[str, Any]:
        fallback = max(
            self.review_heartbeat_sec,
            float(self.review_config.progress_event_min_windows or 1) * self.review_heartbeat_sec,
        )
        empty = {
            "active_waiter_pressure": False,
            "blocked_worker_count": 0,
            "oldest_waiter_age_sec": 0.0,
            "contention_cap_sec": fallback,
        }
        if self.resource_runtime is None:
            return empty
        try:
            if not self.resource_runtime.has_active_lease(job_id=job.job_id):
                return empty
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception:
            return empty
        now = time.time()
        waiters = self._contention_waiter_cards(job, snapshot=snapshot, now=now)
        if not waiters:
            return empty
        oldest = max(float(row.get("queue_age_sec") or 0.0) for row in waiters)
        return {
            "active_waiter_pressure": True,
            "blocked_worker_count": len(waiters),
            "oldest_waiter_age_sec": oldest,
            "contention_cap_sec": fallback,
            "waiters": waiters,
        }


    def _update_resource_efficiency_state(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        finish = self._progress_finish_feasibility(job, signal, elapsed_sec=elapsed_sec)
        progress_payload = job.last_progress if isinstance(job.last_progress, dict) else {}
        structured = progress_payload.get("structured_progress") if isinstance(progress_payload.get("structured_progress"), dict) else {}
        assessment = assess_resource_efficiency(
            previous_state=job.resource_efficiency_state,
            cpu_set=job.cpu_set,
            process_tree_cpu=signal.get("process_tree_cpu") if isinstance(signal.get("process_tree_cpu"), dict) else {},
            assigned_gpu_count=len(job.gpu_ids or []),
            gpu_expected=self._job_uses_expensive_gpu(job),
            gpu_active=self._last_gpu_sample_active(job),
            gpu_sample_available=bool(job.last_gpu_util_sample),
            runtime_sec=elapsed_sec,
            phase=str(signal.get("current_phase") or progress_payload.get("phase") or "unknown"),
            eta_to_deliverable_sec=self._float_or_none(finish.get("eta_to_deliverable_sec")),
            eta_confidence=str(finish.get("eta_confidence") or "low"),
            remaining_useful_budget_sec=self._float_or_none(finish.get("remaining_useful_budget_sec")),
            phase_completion_protected=bool(finish.get("phase_completion_protected")),
            metric_useful=self._metric_value_useful_from_signal(signal),
            deliverable_validity=str(job.deliverable_validity or "none"),
            progress_source=str(structured.get("source") or "unknown"),
            progress_evidence_trust=str(structured.get("evidence_trust") or "unknown"),
            progress_interval_samples=int(structured.get("interval_sample_count") or 0),
            llm_call_count=int(self._job_llm_call_count.get(job.job_id) or 0),
        )
        job.resource_efficiency_state = dict(assessment)
        signal["resource_efficiency"] = dict(assessment)
        return assessment

    def _advance_resource_review_state(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> tuple[ResourceReviewState, Any | None]:
        state = self._review_states.get(job.job_id) or new_review_state(job.job_id)
        if state.heartbeat_index > 0 and float(elapsed_sec or 0.0) < float(state.last_advance_elapsed_sec or 0.0) + self.review_heartbeat_sec:
            signal["resource_review_state"] = state.to_json()
            signal["resource_efficiency"] = dict(job.resource_efficiency_state or {})
            return state, None
        self._update_resource_efficiency_state(job, signal, elapsed_sec=elapsed_sec)
        artifact_update = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
        artifact_age = self._float_or_none(artifact_update.get("age_sec")) if artifact_update else self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed_sec)
        artifact_recent_window = max(self.review_heartbeat_sec * 2.0, self.check_interval_sec * 2.0)
        artifact_recent = bool(artifact_update) and float(artifact_age or 0.0) <= artifact_recent_window
        recoverability = self._fresh_recoverability_for_job(job, elapsed_sec=elapsed_sec)
        recoverable_artifact_recent = bool(recoverability.get("recoverable_artifact_on_disk")) and float(artifact_age or 0.0) <= max(
            self.review_heartbeat_sec,
            self.check_interval_sec * 2.0,
        )
        value_artifact_recent = (
            bool(artifact_update)
            and artifact_recent
            and self._artifact_update_is_value_bearing(
                artifact_update,
                deliverable_validity=str(job.deliverable_validity or "none"),
            )
        )
        artifact_grew = bool(recoverable_artifact_recent or value_artifact_recent)
        progress_payload = job.last_progress if isinstance(job.last_progress, dict) else {}
        structured_progress = progress_payload.get("structured_progress") if isinstance(progress_payload.get("structured_progress"), dict) else {}
        progress_payload_elapsed = self._float_or_none(progress_payload.get("elapsed_sec"))
        structured_progress_advanced = bool(
            structured_progress.get("advanced")
            and progress_payload_elapsed is not None
            and progress_payload_elapsed > float(state.last_advance_elapsed_sec or 0.0)
        )
        signal["structured_progress"] = structured_progress
        signal["structured_progress_advanced"] = structured_progress_advanced
        stdout_observation = self._stdout_observation_for_job(job, signal, elapsed_sec=elapsed_sec)
        signal["stdout_observation"] = stdout_observation
        stream = stdout_observation.get("stdout_stream") if isinstance(stdout_observation.get("stdout_stream"), dict) else {}
        review_stdout_lines = int(signal.get("stdout_lines") or 0) if stream.get("fresh") else 0
        review_stdout_bytes = int(signal.get("stdout_bytes") or 0) if stream.get("fresh") else 0
        review_saw_training_progress = bool(signal.get("saw_training_progress") and stream.get("fresh"))
        metric_value_useful = self._metric_value_useful_from_signal(signal)
        if structured_progress and not structured_progress_advanced:
            review_saw_training_progress = False
        if metric_value_useful is False:
            review_saw_training_progress = False
        contention_context = self._contention_context_for_job(job)
        signal["contention_context"] = contention_context
        review_signal = build_review_signal(
            elapsed_sec=elapsed_sec,
            stdout_lines=review_stdout_lines,
            stdout_bytes=review_stdout_bytes,
            previous_stdout_bytes=int(state.last_stdout_bytes or 0),
            metric_history_text=str(signal.get("metric_history_text") or job.metric_history_text or ""),
            previous_metric_history_text=str(state.last_metric_history_text or ""),
            saw_training_progress=review_saw_training_progress,
            saw_final_score=bool(signal.get("saw_final_score")),
            terminal_signal_events=int(signal.get("terminal_signal_events") or 0),
            previous_terminal_signal_events=int(state.last_terminal_signal_events or 0),
            current_phase=str(signal.get("current_phase") or ""),
            process_tree_cpu=signal.get("process_tree_cpu") if isinstance(signal.get("process_tree_cpu"), dict) else {},
            gpu_active=self._last_gpu_sample_active(job),
            gpu_unknown=not bool(job.last_gpu_util_sample),
            artifact_recent=artifact_recent,
            artifact_grew=artifact_grew,
            structured_progress_advanced=structured_progress_advanced,
            metric_value_useful=metric_value_useful,
            metric_value_status=str((signal.get("resource_metric_value") or {}).get("status") or ""),
            metric_value_delta_to_best=self._float_or_none((signal.get("resource_metric_value") or {}).get("delta_to_best")),
            metric_scope_key=str((signal.get("resource_metric_value") or {}).get("metric_scope_key") or ""),
            active_waiter_pressure=bool(contention_context.get("active_waiter_pressure")),
            blocked_worker_count=int(contention_context.get("blocked_worker_count") or 0),
        )
        state = advance_review_state(state, review_signal, config=self.review_config)
        boundary = next_review_boundary(state, config=self.review_config)
        if boundary is not None:
            state = mark_review_emitted(state, boundary)
        self._review_states[job.job_id] = state
        signal["resource_review_signal"] = review_signal.to_json()
        signal["resource_review_state"] = state.to_json()
        if boundary is not None:
            signal["resource_review_boundary"] = boundary.to_json()
            self._emit(
                "resource_review_boundary",
                job,
                status=boundary.kind,
                payload={
                    "boundary": boundary.to_json(),
                    "review_state": state.to_json(),
                    "review_signal": review_signal.to_json(),
                },
            )
            if self.resource_runtime is not None:
                self.resource_runtime.record_resource_event(
                    "resource_review_boundary",
                    payload={
                        "job_id": job.job_id,
                        "boundary": boundary.to_json(),
                        "review_state": state.to_json(),
                        "review_signal": review_signal.to_json(),
                    },
                    command_id=job.job_id,
                    lease_id=job.job_id,
                )
        return state, boundary

    def _state_machine_review_decision(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        if not (self.review_state_enabled and self.kill_enabled and job.visible):
            return {"enabled": False, "reason": "review_state_disabled_or_job_not_visible"}
        state, boundary = self._advance_resource_review_state(job, signal, elapsed_sec=elapsed_sec)
        if boundary is None:
            return {"enabled": False, "reason": "no_review_boundary", "review_state": state.to_json()}
        reason_map = {
            STALL: "active_intervention:sm_stall",
            ROUTE_VALUE: "active_intervention:sm_route_value",
            PROGRESS_WINDOW: "active_intervention:sm_progress_window",
            TIMEBOX_EXPIRED: "active_intervention:sm_timebox_expired",
            RESOURCE_PRESSURE: "active_intervention:sm_resource_pressure",
        }
        feedback_map = {
            STALL: "RESOURCE_FEEDBACK: recommend_stop_command because the monitored command has no active work across the configured review window.\n",
            ROUTE_VALUE: "RESOURCE_FEEDBACK: recommend_stop_command because active long-running work has low recent quality gain; keep the best artifact and change the method, search space, schedule, validation target, or stopping condition before continuing.\n",
            PROGRESS_WINDOW: "RESOURCE_FEEDBACK: recommend_review_command because active long-running work needs a quality-gain review; keep the best artifact and change the method, search space, schedule, validation target, or stopping condition if continuing.\n",
            TIMEBOX_EXPIRED: "RESOURCE_FEEDBACK: recommend_stop_command because the previous observe/timebox window expired without satisfying the expected resource observation condition.\n",
            RESOURCE_PRESSURE: "RESOURCE_FEEDBACK: recommend_review_command because a new waiter/resource pressure event appeared during the observation window.\n",
        }
        reason = reason_map.get(boundary.kind, "active_intervention:sm_review")
        feedback = feedback_map.get(boundary.kind, "RESOURCE_FEEDBACK: recommend_review_command because the resource review state reached a decision boundary.\n")
        decision = {
            "enabled": True,
            "terminate": False,
            "would_terminate": True,
            "recommended_action": "stop_and_replan",
            "requires_llm_decision": True,
            "arbiter_enabled": bool(self.arbiter_enabled and self.arbiter_mode != "off"),
            "reason": reason,
            "check_interval_sec": self.check_interval_sec,
            "feedback": self._append_resource_intervention_summary(
                self._recommendation_feedback(reason=reason, feedback=feedback),
                action="RECOMMEND_STOP_AND_REPLAN",
                reason=reason,
            ),
            "resource_review_boundary": boundary.to_json(),
            "resource_review_state": state.to_json(),
            "resource_review_signal": signal.get("resource_review_signal") or {},
            "contention_context": signal.get("contention_context") or {},
            "metric_history_text": str(signal.get("metric_history_text") or job.metric_history_text or ""),
            "metric_history_line_count": int(signal.get("metric_history_line_count") or job.metric_history_line_count or 0),
            "resource_metric_value": dict(signal.get("resource_metric_value") or {}),
            "terminal_signal_events": signal.get("terminal_signal_events"),
            "terminal_signal_kind": signal.get("terminal_signal_kind"),
            "deadline_event": signal.get("deadline_event"),
            "deadline_remaining_sec": signal.get("deadline_remaining_sec"),
            "finalization_reserve_sec": signal.get("finalization_reserve_sec"),
        }
        self._record_kill_proposal_event(job, decision, signal, source="resource_review_state")
        return decision

    def _update_job_progress_signal(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> dict[str, Any]:
        progress_age = self._elapsed_since_progress(job.last_progress, elapsed_sec=elapsed_sec)
        artifact_update = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
        artifact_age = self._float_or_none(artifact_update.get("age_sec")) if artifact_update else self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed_sec)
        stdout_observation = self._stdout_observation_for_job(job, signal, elapsed_sec=elapsed_sec)
        signal["stdout_observation"] = stdout_observation
        classify_signal = dict(signal)
        progress_payload = job.last_progress if isinstance(job.last_progress, dict) else {}
        structured_progress = progress_payload.get("structured_progress") if isinstance(progress_payload.get("structured_progress"), dict) else {}
        structured_progress_advanced = bool(structured_progress.get("advanced"))
        classify_signal["structured_progress"] = structured_progress
        classify_signal["structured_progress_advanced"] = structured_progress_advanced
        stream = stdout_observation.get("stdout_stream") if isinstance(stdout_observation.get("stdout_stream"), dict) else {}
        if not stream.get("fresh"):
            classify_signal["stdout_lines"] = 0
            classify_signal["stdout_bytes"] = 0
            classify_signal["saw_training_progress"] = False
        if structured_progress and not structured_progress_advanced:
            classify_signal["saw_training_progress"] = False
        if self._metric_value_useful_from_signal(signal) is False:
            classify_signal["saw_training_progress"] = False
        state = classify_progress_signal(
            classify_signal,
            progress_age_sec=progress_age,
            artifact_age_sec=float(artifact_age or 0.0),
            low_progress_warmup_sec=self.low_progress_warmup_sec,
            stalled_stdout_sec=self.stalled_stdout_sec,
            no_progress_sec=self.low_progress_no_heartbeat_sec,
            no_artifact_sec=self.low_progress_no_artifact_sec,
            idle_samples=job.idle_gpu_lease_samples,
            dataloader_samples=job.dataloader_bottleneck_samples,
            previous_signal=job.progress_signal,
            previous_windows=job.progress_signal_windows,
            min_confidence_windows=self.arbiter_min_progress_windows,
        )
        job.progress_signal = str(state.get("progress_signal") or "unknown")
        job.progress_signal_windows = int(state.get("progress_signal_windows") or 0)
        job.progress_signal_state = dict(state)
        signal.update({
            "progress_signal": job.progress_signal,
            "progress_confidence": state.get("progress_confidence"),
            "progress_signal_windows": job.progress_signal_windows,
            "progress_signal_reason": state.get("progress_signal_reason"),
            "multi_window_low_progress": bool(state.get("multi_window_low_progress")),
        })
        return state

    def _progress_confidence(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> str:
        recent = 0
        stdout_age = self._float_or_none(signal.get("stdout_age_sec"))
        stdout_observation = signal.get("stdout_observation") if isinstance(signal.get("stdout_observation"), dict) else self._stdout_observation_for_job(job, signal, elapsed_sec=elapsed_sec)
        stream = stdout_observation.get("stdout_stream") if isinstance(stdout_observation.get("stdout_stream"), dict) else {}
        if int(signal.get("stdout_lines") or 0) > 0 and bool(stream.get("fresh")) and (stdout_age is None or self.stalled_stdout_sec <= 0 or stdout_age < self.stalled_stdout_sec):
            recent += 1
        progress_age = self._elapsed_since_progress(job.last_progress, elapsed_sec=elapsed_sec)
        progress_limit = max(0.0, float(self.low_progress_no_heartbeat_sec or 0.0))
        if job.last_progress and (progress_limit <= 0 or progress_age < progress_limit):
            recent += 1
        artifact_age = self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed_sec)
        artifact_limit = max(0.0, float(self.low_progress_no_artifact_sec or 0.0))
        if job.last_artifact_progress and (artifact_limit <= 0 or artifact_age < artifact_limit):
            recent += 1
        if recent >= 2:
            return "high"
        if recent == 1:
            return "medium"
        return "low"

    def _process_liveness_for_job(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        cached = signal.get("_process_liveness") if isinstance(signal.get("_process_liveness"), dict) else None
        if cached is not None:
            return dict(cached)
        terminal_seen = bool(signal.get("terminal_signal_events") or signal.get("saw_final_score") or job.terminal_signal_seen)
        process_tree_cpu = dict(signal.get("process_tree_cpu") or {})
        liveness = build_process_liveness(
            root_pid=job.pid,
            root_pgid=job.pgid,
            expected_root_start_time_epoch=job.pid_start_time_epoch,
            exit_code_seen=bool(job.exit_code_seen),
            terminal_signal_seen=terminal_seen,
            process_tree_cpu=process_tree_cpu,
        )
        lifecycle = classify_process_lifecycle(
            process_liveness=liveness,
            process_tree_cpu=process_tree_cpu,
            tool_waiting=True,
        ).to_json()
        liveness["lifecycle_status"] = lifecycle.get("status")
        liveness["lifecycle"] = lifecycle
        if liveness.get("root_pid_alive") or liveness.get("process_group_alive") or liveness.get("live_descendant_count"):
            liveness["last_known_pid_seen_at"] = time.time()
        signal["_process_liveness"] = dict(liveness)
        return liveness

    def _current_gpu_mem_gb_for_job(self, job: ResourceJob) -> float:
        sample_bundle = job.last_gpu_util_sample if isinstance(job.last_gpu_util_sample, dict) else {}
        placement = sample_bundle.get("process_gpu_placement") if isinstance(sample_bundle.get("process_gpu_placement"), dict) else {}
        if placement.get("available") is True:
            return self._process_gpu_placement_mem_mb(placement) / 1024.0
        sample = sample_bundle.get("sample") if isinstance(sample_bundle.get("sample"), dict) else {}
        rows = sample.get("gpus") if isinstance(sample, dict) and isinstance(sample.get("gpus"), list) else []
        wanted = {str(x) for x in (job.gpu_ids or []) if str(x).strip()}
        total_mb = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
            if wanted and gpu_id not in wanted:
                continue
            try:
                total_mb += max(0.0, float(row.get("memory_used_mb") or 0.0))
            except (TypeError, ValueError):
                pass
        return total_mb / 1024.0

    def _resource_snapshot_for_job(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        gpu_rows: list[dict[str, Any]] = []
        sample = job.last_gpu_util_sample.get("sample") if isinstance(job.last_gpu_util_sample, dict) else {}
        rows = sample.get("gpus") if isinstance(sample, dict) and isinstance(sample.get("gpus"), list) else []
        placement = job.last_gpu_util_sample.get("process_gpu_placement") if isinstance(job.last_gpu_util_sample, dict) else {}
        placement_available = isinstance(placement, dict) and placement.get("available") is True
        process_mem_by_gpu = self._process_gpu_memory_by_gpu(placement) if placement_available else {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "")
            process_mem_mb = process_mem_by_gpu.get(gpu_id, 0.0)
            utilization = row.get("utilization_gpu_pct")
            memory_used = row.get("memory_used_mb")
            if placement_available:
                utilization = utilization if process_mem_mb > 0.0 else 0.0
                memory_used = process_mem_mb
            gpu_rows.append({
                "resource_type": "gpu",
                "id": gpu_id,
                "utilization_gpu_pct": utilization,
                "memory_used_mb": memory_used,
                "memory_total_mb": row.get("memory_total_mb"),
                "process_gpu_attributed": placement_available,
            })
        elapsed = max(
            0.0,
            float(signal.get("elapsed_sec") or 0.0),
            float(job.last_monitor_heartbeat_elapsed_sec or 0.0),
            float((job.last_progress or {}).get("elapsed_sec") or 0.0) if isinstance(job.last_progress, dict) else 0.0,
            float((job.last_artifact_progress or {}).get("elapsed_sec") or 0.0) if isinstance(job.last_artifact_progress, dict) else 0.0,
        )
        stdout_observation = self._stdout_observation_for_job(job, signal, elapsed_sec=elapsed)
        resources = gpu_rows + [{
            "resource_type": "cpu",
            "id": "process",
            "stdout_lines": int(signal.get("stdout_lines") or 0),
            "stdout_bytes": int(signal.get("stdout_bytes") or 0),
            "stdout_observation": stdout_observation,
            "process_tree_cpu": dict(signal.get("process_tree_cpu") or {}),
        }]
        process_liveness = self._process_liveness_for_job(job, signal)
        return {
            "resource_class_declared": job.resource_class,
            "resource_class_observed": job.resource_class,
            "primary_resource_type": "gpu" if self._job_uses_expensive_gpu(job) else "cpu",
            "recoverability": self._fresh_recoverability_for_job(job, elapsed_sec=elapsed),
            "stdout_observation": stdout_observation,
            "lease": {
                "resource_type": "gpu",
                "resource_ids": list(job.gpu_ids),
                "active": bool(self.resource_runtime and self.resource_runtime.has_active_lease(job_id=job.job_id)),
            },
            "process_liveness": process_liveness,
            "resources": resources,
        }

    def _execution_facts_for_job(
        self,
        job: ResourceJob,
        *,
        resource_snapshot: dict[str, Any],
        progress_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        gpu_expected = self._job_uses_expensive_gpu(job)
        if self.resource_runtime is not None and self.resource_runtime.has_active_lease(job_id=job.job_id):
            gpu_expected = True
        return build_execution_facts(
            resource_snapshot=resource_snapshot,
            progress_snapshot=progress_snapshot,
            command=job.command,
            source_hint=job.source_hint,
            assigned_gpu_ids=list(job.gpu_ids or []),
            gpu_expected=gpu_expected,
        )

    def _ensure_resource_proposal_facts(
        self,
        job: ResourceJob,
        proposal: dict[str, Any],
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        resource_snapshot = proposal.get("resource_snapshot") if isinstance(proposal.get("resource_snapshot"), dict) else {}
        progress_snapshot = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        if not resource_snapshot:
            resource_snapshot = self._resource_snapshot_for_job(job, signal)
            proposal["resource_snapshot"] = resource_snapshot
        if not progress_snapshot:
            progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=float(signal.get("elapsed_sec") or 0.0))
            proposal["progress_snapshot"] = progress_snapshot
        if not isinstance(proposal.get("execution_facts"), dict):
            proposal["execution_facts"] = self._execution_facts_for_job(
                job,
                resource_snapshot=resource_snapshot,
                progress_snapshot=progress_snapshot,
            )
        if not isinstance(proposal.get("research_cadence"), dict):
            proposal["research_cadence"] = self._research_cadence_fact_card(
                job,
                signal,
                progress_snapshot,
            )
        state = self._review_states.get(job.job_id)
        if state is not None and not isinstance(proposal.get("clear_on_cadence"), dict):
            proposal["clear_on_cadence"] = dict((state.to_json().get("clear_on_cadence") or {}))
        if not isinstance(proposal.get("contention_context"), dict):
            proposal["contention_context"] = dict(signal.get("contention_context") or self._contention_context_for_job(job))
        if not isinstance(proposal.get("budget_context"), dict):
            remaining = float(progress_snapshot.get("deadline_remaining_sec") or signal.get("deadline_remaining_sec") or 0.0)
            budget_cap = remaining * float(self.review_config.timebox_budget_fraction or 0.10) if remaining > 0.0 else float(self.review_config.max_timebox_sec or 1800.0)
            proposal["budget_context"] = {
                "remaining_budget_sec": remaining,
                "timebox_budget_cap_sec": budget_cap,
            }
        if not isinstance(proposal.get("state_generation"), dict):
            state_generation = build_resource_state_generation({
                "proposal_type": proposal.get("proposal_type"),
                "reason_code": proposal.get("reason_code"),
                "resource_snapshot": resource_snapshot,
                "progress_snapshot": progress_snapshot,
                "blocker": proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {},
                "gpu_ids": list(job.gpu_ids or []),
                "execution_facts": proposal.get("execution_facts"),
            })
            proposal["control_generation_key"] = state_generation.control_generation_key
            proposal["feedback_generation_key"] = state_generation.feedback_generation_key
            proposal["state_generation"] = state_generation.to_json()
        return proposal

    def _progress_snapshot_for_job(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> dict[str, Any]:
        artifact = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
        progress_state = dict(job.progress_signal_state or {})
        recoverability = self._fresh_recoverability_for_job(job, elapsed_sec=elapsed_sec)
        stdout_observation = signal.get("stdout_observation") if isinstance(signal.get("stdout_observation"), dict) else self._stdout_observation_for_job(job, signal, elapsed_sec=elapsed_sec)
        stream = stdout_observation.get("stdout_stream") if isinstance(stdout_observation.get("stdout_stream"), dict) else {}
        process_liveness = self._process_liveness_for_job(job, signal)
        finish_feasibility = self._progress_finish_feasibility(job, signal, elapsed_sec=elapsed_sec)
        return {
            "runtime_sec": float(elapsed_sec or 0.0),
            "stdout_last_line_age_sec": self._float_or_none(signal.get("stdout_age_sec")),
            "stdout_lines": int(signal.get("stdout_lines") or 0),
            "stdout_bytes": int(signal.get("stdout_bytes") or 0),
            "metric_history_text": str(signal.get("metric_history_text") or job.metric_history_text or ""),
            "metric_history_line_count": int(signal.get("metric_history_line_count") or job.metric_history_line_count or 0),
            "metric_scope_key": str((signal.get("resource_metric_value") or {}).get("metric_scope_key") or ""),
            "meaningful_stdout": bool(signal.get("saw_training_progress") and stream.get("fresh")),
            "metric_last_update_age_sec": self._elapsed_since_progress(job.last_progress, elapsed_sec=elapsed_sec),
            "artifact_last_update_age_sec": self._float_or_none(artifact.get("age_sec")) if artifact else self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed_sec),
            "artifact_updates": [artifact] if artifact else [],
            "recoverability": recoverability,
            "stdout_observation": stdout_observation,
            "recoverable_artifact_on_disk": bool(recoverability.get("recoverable_artifact_on_disk")),
            "checkpoint_on_signal_supported_observed": bool(recoverability.get("checkpoint_on_signal_supported_observed")),
            "last_recoverable_artifact_age_sec": recoverability.get("last_recoverable_artifact_age_sec"),
            "artifact_watch_pattern": str(recoverability.get("artifact_watch_pattern") or "none"),
            "stop_cost": str(recoverability.get("stop_cost") or "high"),
            "known_stage": str(signal.get("current_phase") or ""),
            "process_tree_cpu": dict(signal.get("process_tree_cpu") or {}),
            "process_liveness": process_liveness,
            "heartbeat_source": _SAFETY_HEARTBEAT_SOURCE,
            "quick_probe": dict(job.quick_probe or {}),
            "output_pattern": str((job.quick_probe or {}).get("output_pattern") or "unknown"),
            "near_submission": bool(signal.get("saw_final_score")),
            "deliverable_validity": str(job.deliverable_validity or "none"),
            "progress_signal": progress_state.get("progress_signal") or job.progress_signal or "unknown",
            "progress_confidence": progress_state.get("progress_confidence") or self._progress_confidence(job, signal, elapsed_sec=elapsed_sec),
            "progress_signal_windows": int(progress_state.get("progress_signal_windows") or job.progress_signal_windows or 0),
            "progress_signal_reason": str(progress_state.get("progress_signal_reason") or ""),
            "multi_window_low_progress": bool(progress_state.get("multi_window_low_progress")),
            "stalled_mark_count": int(job.stalled_mark_count or 0),
            "deadline_event": bool(signal.get("deadline_event")),
            "deadline_remaining_sec": float(signal.get("deadline_remaining_sec") or 0.0),
            "finalization_reserve_sec": float(signal.get("finalization_reserve_sec") or 0.0),
            **finish_feasibility,
            "resource_efficiency": dict(
                signal.get("resource_efficiency") or job.resource_efficiency_state or {}
            ),
            "phase_completion_protected": bool(
                finish_feasibility.get("phase_completion_protected")
                and not (
                    signal.get("resource_efficiency")
                    or job.resource_efficiency_state
                    or {}
                ).get("completion_grace_expired")
            ),
        }

    def _recoverability_from_artifact(
        self,
        job: ResourceJob,
        artifact: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        recoverable = bool(artifact.get("recoverable_artifact_on_disk"))
        run_state_status = str(artifact.get("run_state_status") or "").strip().lower()
        checkpoint_supported_observed = bool(
            artifact.get("run_state_confirmed") and run_state_status in {"interrupted", "completed"}
        )
        try:
            age = float(artifact.get("age_sec") or 0.0)
        except (TypeError, ValueError):
            age = 0.0
        if recoverable:
            watch_pattern = "periodic" if str(artifact.get("artifact_scope") or "") == "current_run" else "terminal"
            stop_cost = "low"
            last_recoverable_age: float | None = age
        else:
            watch_pattern = "growing" if artifact.get("candidate_artifact") else "none"
            stop_cost = "high"
            last_recoverable_age = None
        return {
            "recoverable_artifact_on_disk": recoverable,
            "checkpoint_on_signal_supported_observed": checkpoint_supported_observed,
            "last_recoverable_artifact_age_sec": last_recoverable_age,
            "artifact_watch_pattern": watch_pattern,
            "stop_cost": stop_cost,
            "artifact_scope": str(artifact.get("artifact_scope") or ""),
            "stability": str(artifact.get("stability") or ""),
            "artifact_path": str(artifact.get("path") or ""),
            "artifact_size_bytes": int(artifact.get("size_bytes") or 0),
            "artifact_age_sec": age,
            "run_state_status": run_state_status,
            "safe_to_resume": artifact.get("safe_to_resume", "unknown"),
            "route_id": str(artifact.get("route_id") or ""),
            "fold": str(artifact.get("fold") or ""),
            "validation_protocol": str(artifact.get("validation_protocol") or ""),
            "checkpoint_kind": str(artifact.get("checkpoint_kind") or ""),
            "mergeable": artifact.get("mergeable", "unknown"),
            "observed_at_elapsed_sec": float(elapsed_sec or 0.0),
            "source": "artifact_watcher",
        }

    @staticmethod
    def _waiter_gpu_ids(raw: dict[str, Any]) -> set[str]:
        return {str(x) for x in (raw.get("gpu_ids") or raw.get("candidate_gpu_ids") or []) if str(x).strip()}

    @staticmethod
    def _nested_float(raw: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            value = float(raw.get(key, default) or default)
        except (TypeError, ValueError):
            return float(default)
        return value if value == value else float(default)

    def _review_cooldown_active(self, job: ResourceJob, proposal_type: str, *, min_interval_sec: float, now: float) -> bool:
        key = f"{job.job_id}:{proposal_type}"
        observe_more_until = float(self._review_observe_more_until.get(key) or 0.0)
        if observe_more_until:
            if now < observe_more_until:
                return True
            self._review_observe_more_until.pop(key, None)
            return False
        job_last = float(self._last_review_proposal_emit.get(job.job_id) or 0.0)
        if job_last and self.arbiter_proposal_coalesce_window_sec > 0 and now - job_last < self.arbiter_proposal_coalesce_window_sec:
            return True
        last = float(self._last_review_proposal_emit.get(key) or 0.0)
        return bool(last and now - last < max(1.0, float(min_interval_sec or 1.0)))

    def _mark_review_proposal_emitted(self, job: ResourceJob, proposal_type: str, *, now: float) -> None:
        self._last_review_proposal_emit[job.job_id] = now
        self._last_review_proposal_emit[f"{job.job_id}:{proposal_type}"] = now

    def _review_history_key(self, job_id: str, proposal_type: str) -> str:
        return f"{job_id}:{proposal_type}"

    def _review_material_signature(self, proposal: dict[str, Any]) -> str:
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        blocker = proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {}
        suspect = blocker.get("active_lease_suspect") if isinstance(blocker.get("active_lease_suspect"), dict) else {}
        active_work = suspect.get("active_work_counterevidence") if isinstance(suspect.get("active_work_counterevidence"), dict) else {}
        route = progress.get("route_viability") if isinstance(progress.get("route_viability"), dict) else blocker.get("route_viability") if isinstance(blocker.get("route_viability"), dict) else {}
        cadence = proposal.get("research_cadence") if isinstance(proposal.get("research_cadence"), dict) else {}
        material = {
            "proposal_type": str(proposal.get("proposal_type") or ""),
            "reason_code": str(proposal.get("reason_code") or ""),
            "progress_signal": str(progress.get("progress_signal") or blocker.get("progress_signal") or ""),
            "progress_confidence": str(progress.get("progress_confidence") or blocker.get("progress_confidence") or ""),
            "route_viability_state": str(route.get("state") or ""),
            "resource_suspect": bool(suspect.get("resource_suspect")),
            "unknown_progress_suspect": bool(suspect.get("unknown_progress_suspect")),
            "value_suspect": bool(suspect.get("value_suspect")),
            "recent_useful_output": bool(suspect.get("recent_useful_output")),
            "active_work": bool(active_work.get("active")),
            "deadline_event": bool(progress.get("deadline_event")),
            "research_cadence_state": str(cadence.get("state") or ""),
            "route_metric_proven": bool(cadence.get("route_metric_proven")),
            "metric_budget_kind": str(cadence.get("metric_budget_kind") or ""),
        }
        return hashlib.sha1(repr(sorted(material.items())).encode("utf-8", errors="replace")).hexdigest()[:16]

    def _attach_review_history(self, proposal: dict[str, Any], *, now: float) -> dict[str, Any]:
        out = dict(proposal or {})
        job_id = str(out.get("command_id") or ((out.get("decision_preview") or {}).get("job_id") if isinstance(out.get("decision_preview"), dict) else "") or "")
        proposal_type = str(out.get("proposal_type") or "")
        if not job_id or not proposal_type:
            return out
        key = self._review_history_key(job_id, proposal_type)
        hist = dict(self._review_history.get(key) or {})
        signature = self._review_material_signature(out)
        unchanged = bool(hist.get("last_bad_signature") == signature and int(hist.get("observe_count") or 0) > 0)
        unchanged_windows = int(hist.get("unchanged_bad_fact_windows") or 0) if unchanged else 0
        repeated = bool(unchanged and int(hist.get("observe_count") or 0) >= 2 and unchanged_windows >= 2)
        waiters = out.get("waiters") if isinstance(out.get("waiters"), list) else []
        progress = out.get("progress_snapshot") if isinstance(out.get("progress_snapshot"), dict) else {}
        advisory = out.get("main_agent_advisory") if isinstance(out.get("main_agent_advisory"), dict) else {}
        advisory_preference = str(advisory.get("preference") or "").strip().lower()
        advisory_confidence = str(advisory.get("confidence") or "").strip().lower()
        advisory_safe_to_stop = advisory_preference in {"safe_to_stop", "kill_and_replan", "stop", "stop_and_replan"}
        progress_signal = str(progress.get("progress_signal") or "").strip().lower()
        reason_code = str(out.get("reason_code") or "").strip().lower()
        low_value_review = (
            str(out.get("proposal_type") or "").strip().lower() == "kill_proposal"
            and (
                progress_signal in {"stalled", "degraded", "unknown"}
                or "stalled" in reason_code
                or "low_progress" in reason_code
                or "dataloader" in reason_code
            )
        )
        advisory_opportunity_support = bool(
            repeated
            and low_value_review
            and advisory_safe_to_stop
            and advisory_confidence in {"low", "medium", "high"}
        )
        structured_support = bool(repeated and (waiters or progress.get("deadline_event") or advisory_opportunity_support))
        review_history = {
            "observe_count": int(hist.get("observe_count") or 0),
            "first_observe_at": hist.get("first_observe_at"),
            "last_action": str(hist.get("last_action") or ""),
            "last_decision_at": hist.get("last_decision_at"),
            "first_suspect_at": hist.get("first_suspect_at"),
            "unchanged_bad_fact_windows": unchanged_windows,
            "repeated_observe_support": repeated,
            "structured_opportunity_cost_support": structured_support,
            "advisory_opportunity_support": advisory_opportunity_support,
            "advisory_preference": advisory_preference,
            "advisory_confidence": advisory_confidence,
            "current_bad_signature": signature,
        }
        out["review_history"] = review_history
        if structured_support:
            out["structured_opportunity_cost_support"] = True
            if self.resource_runtime is not None:
                self.resource_runtime.record_resource_event(
                    "resource_review_observe_more_escalated",
                    payload={
                        "proposal_id": out.get("proposal_id"),
                        "proposal_type": proposal_type,
                        "job_id": job_id,
                        "review_history": review_history,
                    },
                    proposal_id=str(out.get("proposal_id") or ""),
                    command_id=job_id,
                    lease_id=job_id,
                )
        return out

    def _note_review_decision_history(self, proposal: dict[str, Any], decision: dict[str, Any]) -> None:
        job_id = str(proposal.get("command_id") or ((proposal.get("decision_preview") or {}).get("job_id") if isinstance(proposal.get("decision_preview"), dict) else "") or "")
        proposal_type = str(proposal.get("proposal_type") or "")
        if not job_id or not proposal_type:
            return
        key = self._review_history_key(job_id, proposal_type)
        now = time.time()
        action = str(decision.get("action") or "").upper()
        previous = dict(self._review_history.get(key) or {})
        signature = self._review_material_signature(proposal)
        is_observe = action in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL"}
        if is_observe:
            same = previous.get("last_bad_signature") == signature
            observe_count = int(previous.get("observe_count") or 0) + 1
            unchanged_windows = int(previous.get("unchanged_bad_fact_windows") or 0) + 1 if same else 1
            first_observe_at = previous.get("first_observe_at") or now
            first_suspect_at = previous.get("first_suspect_at")
            blocker = proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {}
            suspect = blocker.get("active_lease_suspect") if isinstance(blocker.get("active_lease_suspect"), dict) else {}
            if (suspect.get("resource_suspect") or suspect.get("unknown_progress_suspect") or suspect.get("value_suspect")) and not first_suspect_at:
                first_suspect_at = now
            self._review_history[key] = {
                "observe_count": observe_count,
                "first_observe_at": first_observe_at,
                "last_action": action,
                "last_decision_at": now,
                "first_suspect_at": first_suspect_at,
                "last_bad_signature": signature,
                "unchanged_bad_fact_windows": unchanged_windows,
            }
            return
        self._review_history[key] = {
            "observe_count": 0,
            "first_observe_at": None,
            "last_action": action,
            "last_decision_at": now,
            "first_suspect_at": previous.get("first_suspect_at"),
            "last_bad_signature": signature,
            "unchanged_bad_fact_windows": 0,
        }

    def _resource_confidence_for_review(self, job: ResourceJob, *, has_active_lease: bool, waiter_count: int = 0) -> str:
        if has_active_lease and waiter_count > 0 and job.last_gpu_util_sample:
            return "high"
        if has_active_lease and waiter_count > 0:
            return "medium"
        if has_active_lease:
            return "medium"
        return "low"

    def _lease_suspect_thresholds(self) -> LeaseSuspectThresholds:
        return LeaseSuspectThresholds(
            warmup_sec=max(0.0, float(self.gpu_idle_lease_warmup_sec or 0.0)),
            min_progress_windows=max(1, int(self.arbiter_min_progress_windows or 1)),
            low_gpu_util_pct=max(0.0, float(self.gpu_idle_lease_util_pct or 0.0)),
            min_gpu_mem_gb=max(0.0, float(self.gpu_idle_lease_mem_gb or 0.0)),
            route_viability_window_sec=max(
                float(self.low_progress_warmup_sec or 0.0),
                float(self.arbiter_periodic_min_runtime_sec or 0.0),
                float(self.arbiter_contention_min_runtime_sec or 0.0),
            ),
            unknown_progress_grace_sec=max(
                float(self.gpu_idle_lease_warmup_sec or 0.0),
                float(self.low_progress_warmup_sec or 0.0),
                float(self.arbiter_contention_min_runtime_sec or 0.0),
            ),
            unknown_progress_stale_output_sec=max(
                float(self.low_progress_no_heartbeat_sec or 0.0),
                float(self.low_progress_no_artifact_sec or 0.0),
                float(self.stalled_stdout_sec or 0.0),
                float(self.check_interval_sec or 1.0) * float(self.arbiter_min_progress_windows or 1),
            ),
        )

    def _record_v7_resource_event(self, event_type: str, job: ResourceJob, payload: dict[str, Any]) -> None:
        if self.resource_runtime is None:
            return
        self.resource_runtime.record_resource_event(
            event_type,
            payload={"schema_version": 1, **dict(payload or {})},
            command_id=job.job_id,
            lease_id=job.job_id,
        )

    def _update_v7_route_profile(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> dict[str, Any]:
        raw_state = job.last_deliverable_completion_check.get("state") if isinstance(job.last_deliverable_completion_check, dict) else {}
        state = raw_state if isinstance(raw_state, dict) else {}
        validity = classify_deliverable_validity(state)
        job.deliverable_validity = validity
        route = classify_route_viability(
            elapsed_sec=elapsed_sec,
            resource_class=job.resource_class,
            progress_snapshot=self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed_sec),
            deliverable_validity=validity,
            thresholds=self._lease_suspect_thresholds(),
        )
        job.route_viability_state = dict(route)
        job.route_viability_proven = bool(route.get("route_viability_proven"))
        return route

    def _near_deliverable_for_job(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> dict[str, Any]:
        state = job.last_deliverable_completion_check.get("state") if isinstance(job.last_deliverable_completion_check, dict) else {}
        state = state if isinstance(state, dict) else {}
        validity = classify_deliverable_validity(state)
        if validity != "none":
            job.deliverable_validity = validity
        complete = bool(validity == "produced_valid" and (state.get("complete") or signal.get("saw_final_score") or signal.get("terminal_signal_events")))
        artifact_age = self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed_sec)
        reason = "deliverable_complete" if complete else str(state.get("reason") or "unknown_without_structured_progress")
        return {
            "confidence": "high" if complete else "unknown",
            "deliverable_exists": complete,
            "deliverable_validity": validity,
            "route_viability": dict(job.route_viability_state or {}),
            "checkpoint_eta_sec": None,
            "epochs_remaining": None,
            "last_checkpoint_age_sec": artifact_age,
            "last_completion_check": dict(state),
            "source": "deliverable_completion_guard" if state else "passive_monitor",
            "reason": reason,
        }

    def _blocker_fact_card(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
        resource_confidence: str,
    ) -> dict[str, Any]:
        progress = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed_sec)
        return {
            "job_id": job.job_id,
            "worker_id": self.worker_id,
            "resource_class": job.resource_class,
            "gpu_ids": list(job.gpu_ids or []),
            "runtime_sec": float(elapsed_sec or 0.0),
            "progress_signal": progress.get("progress_signal"),
            "progress_confidence": progress.get("progress_confidence"),
            "progress_signal_windows": progress.get("progress_signal_windows"),
            "resource_confidence": resource_confidence,
            "route_viability": dict(job.route_viability_state or {}),
            "active_lease_suspect": dict(job.active_lease_suspect_state or {}),
            "process_liveness": self._process_liveness_for_job(job, signal),
            "last_gpu_util_sample": dict(job.last_gpu_util_sample or {}),
            "value_hint": dict(job.value_hint or {}),
            "recoverability": self._fresh_recoverability_for_job(job, elapsed_sec=elapsed_sec),
            "command_digest": job.command_digest,
            "command_preview": str(job.command or "")[:240],
        }

    def _contention_waiter_cards(
        self,
        job: ResourceJob,
        *,
        snapshot: dict[str, Any],
        now: float,
    ) -> list[dict[str, Any]]:
        leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
        raw_lease = leases.get(job.job_id) if isinstance(leases, dict) else None
        if not isinstance(raw_lease, dict):
            return []
        blocker_gpu_ids = {str(x) for x in (raw_lease.get("gpu_ids") or job.gpu_ids or []) if str(x).strip()}
        if not blocker_gpu_ids:
            return []
        raw_waiters = snapshot.get("waiters") if isinstance(snapshot.get("waiters"), dict) else {}
        rows: list[dict[str, Any]] = []
        for waiter_id, raw in raw_waiters.items():
            if str(waiter_id) == job.job_id or not isinstance(raw, dict):
                continue
            waiter_gpu_ids = self._waiter_gpu_ids(raw)
            if waiter_gpu_ids and not (blocker_gpu_ids & waiter_gpu_ids):
                continue
            submitted_at = self._nested_float(raw, "submitted_at", now)
            queue_age = max(0.0, now - submitted_at)
            if queue_age < self.arbiter_contention_min_waiter_age_sec:
                continue
            meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            value_hint = meta.get("value_hint") if isinstance(meta.get("value_hint"), dict) else {}
            priority = meta.get("admission_priority") if isinstance(meta.get("admission_priority"), dict) else {}
            priority_score = self._nested_float(raw, "priority_score", self._nested_float(priority, "admission_priority_score", 0.0))
            rows.append({
                "job_id": str(raw.get("job_id") or waiter_id),
                "worker_id": str(raw.get("worker_id") or ""),
                "resource_class": str(raw.get("resource_class") or ""),
                "gpu_ids": [str(x) for x in (raw.get("gpu_ids") or []) if str(x).strip()],
                "request_count": int(raw.get("request_count") or 1),
                "queue_age_sec": queue_age,
                "submitted_at": submitted_at,
                "priority_score": priority_score,
                "admission_priority_score": self._nested_float(priority, "admission_priority_score", priority_score),
                "expected_value_score": self._nested_float(value_hint, "expected_value_score", 0.0),
                "near_submission_score": self._nested_float(value_hint, "near_submission_score", 0.0),
                "value_hint": dict(value_hint),
            })
        rows.sort(key=lambda row: (-float(row.get("priority_score") or 0.0), float(row.get("submitted_at") or now), str(row.get("job_id") or "")))
        for index, row in enumerate(rows, start=1):
            row["queue_position"] = index
        return rows[:3]

    def _resource_metric_value_assessment(self, job: ResourceJob) -> dict[str, Any]:
        metric = self._current_structured_metric_for_job(job)
        if not metric:
            return {}
        live_assessment = self._live_metric_value_assessment(job, metric)
        score_context = self._score_context_fact_card()
        best = score_context.get("valid_best_score") if isinstance(score_context.get("valid_best_score"), dict) else {}
        best_value = self._float_or_none(best.get("value") if isinstance(best, dict) else None)
        if best_value is None:
            return live_assessment
        metric_protocol = str(metric.get("validation_protocol") or "").strip()
        metric_fold = str(metric.get("fold") or "").strip().lower()
        best_protocol = str(
            best.get("validation_protocol") or best.get("val_score_type") or ""
        ).strip()
        best_fold = str(best.get("fold") or "").strip().lower()
        incompatible_scope = bool(
            not validation_protocols_comparable(metric_protocol, best_protocol)
            or (bool(metric_fold or best_fold) and metric_fold != best_fold)
        )
        if incompatible_scope:
            return {**live_assessment, "global_comparison_skipped": "metric_scope_mismatch"}
        lower = bool(best.get("lower_is_better")) if isinstance(best, dict) else True
        current = float(metric["value"])
        delta_to_best = current - best_value if lower else best_value - current
        tolerance = max(1e-3, abs(best_value) * 0.003)
        if delta_to_best > tolerance:
            status = "below_observed_best"
            useful = False
        elif delta_to_best < -tolerance:
            status = "above_observed_best"
            useful = True
        else:
            status = "near_observed_best"
            useful = True
        return {
            **metric,
            "status": status,
            "useful": useful,
            "best_value": best_value,
            "best_stage": str(best.get("stage_id") or "") if isinstance(best, dict) else "",
            "best_worker": str(best.get("worker_id") or "") if isinstance(best, dict) else "",
            "best_metric_validity": str(best.get("metric_validity") or "") if isinstance(best, dict) else "",
            "lower_is_better": lower,
            "delta_to_best": delta_to_best,
            "relative_delta_to_best": delta_to_best / max(abs(best_value), 1e-12),
            "tolerance": tolerance,
            "comparison_source": "stage_performance_csv",
        }

    def _live_metric_value_assessment(self, job: ResourceJob, metric: dict[str, Any]) -> dict[str, Any]:
        name = str(metric.get("metric_name") or "").strip().lower()
        current = self._float_or_none(metric.get("value"))
        if not name or current is None:
            return {**metric, "status": "live_metric_unavailable", "useful": None}
        lower = metric_lower_is_better_hint(name)
        if lower is None:
            return {
                **metric,
                "status": "live_metric_direction_unknown",
                "useful": None,
                "comparison_source": "live_job_metric",
            }
        state_key = str(metric.get("metric_scope_key") or name)
        state = dict(job.live_metric_state.get(state_key) or {})
        signature = (
            round(float(current), 12),
            str(metric.get("progress_unit") or ""),
            self._float_or_none(metric.get("progress_current")),
            self._float_or_none(metric.get("progress_total")),
        )
        if tuple(state.get("signature") or ()) == signature and isinstance(state.get("assessment"), dict):
            return dict(state["assessment"])
        previous_best = self._float_or_none(state.get("best_value"))
        observation_count = int(state.get("observation_count") or 0) + 1
        if previous_best is None:
            best_value = float(current)
            assessment = {
                **metric,
                "status": "live_metric_baseline_established",
                "useful": None,
                "best_value": best_value,
                "lower_is_better": lower,
                "observation_count": observation_count,
                "comparison_source": "live_job_metric",
            }
        else:
            delta_to_best = float(current) - previous_best if lower else previous_best - float(current)
            tolerance = max(1e-3, abs(previous_best) * 0.003)
            improved = delta_to_best < -tolerance
            if improved:
                status = "above_live_best"
                best_value = float(current)
            elif delta_to_best > tolerance:
                status = "below_live_best"
                best_value = previous_best
            else:
                status = "live_metric_plateau"
                best_value = previous_best
            assessment = {
                **metric,
                "status": status,
                "useful": improved,
                "best_value": previous_best,
                "lower_is_better": lower,
                "delta_to_best": delta_to_best,
                "relative_delta_to_best": delta_to_best / max(abs(previous_best), 1e-12),
                "tolerance": tolerance,
                "observation_count": observation_count,
                "comparison_source": "live_job_metric",
            }
        job.live_metric_state[state_key] = {
            "signature": list(signature),
            "best_value": best_value,
            "observation_count": observation_count,
            "assessment": dict(assessment),
        }
        return assessment

    def _current_structured_metric_for_job(self, job: ResourceJob) -> dict[str, Any]:
        progress = job.last_progress if isinstance(job.last_progress, dict) else {}
        signals = progress.get("signals") if isinstance(progress.get("signals"), dict) else {}
        raw_metrics = signals.get("metrics") if isinstance(signals.get("metrics"), dict) else {}
        metric_evidence = (
            signals.get("metric_evidence")
            if isinstance(signals.get("metric_evidence"), dict)
            else {}
        )
        for raw_name, raw_value in raw_metrics.items():
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_name or "").strip().lower()).strip("_")
            if not name or name in {"loss", "train_loss", "training_loss"}:
                continue
            value = self._float_or_none(raw_value)
            if value is None:
                continue
            heartbeat = signals.get("heartbeat") if isinstance(signals.get("heartbeat"), dict) else {}
            phase = str(signals.get("phase") or heartbeat.get("phase") or progress.get("phase") or "").strip()
            route_id = str(
                heartbeat.get("route_id")
                or signals.get("route_id")
                or (job.value_hint or {}).get("route_id")
                or self._research_route_key_for_job(job)
                or "unknown"
            ).strip()
            fold = str(heartbeat.get("fold") or signals.get("fold") or "").strip()
            raw_validation_protocol = str(
                heartbeat.get("validation_protocol") or signals.get("validation_protocol") or ""
            ).strip()
            validation_protocol = (
                normalize_validation_protocol(raw_validation_protocol)
                or normalize_route_key(raw_validation_protocol).lower()
            )
            scope_values = (route_id, phase or "unknown", fold or "unknown", name, validation_protocol or "unknown")
            metric_scope_key = "|".join(re.sub(r"[^a-z0-9_.:-]+", "_", item.lower()).strip("_") or "unknown" for item in scope_values)
            return {
                "metric_name": name,
                "value": value,
                "phase": phase,
                "route_id": route_id,
                "fold": fold,
                "validation_protocol": validation_protocol,
                "metric_scope_key": metric_scope_key,
                "evidence_source": str(metric_evidence.get("source") or "unknown"),
                "evidence_trust": str(metric_evidence.get("trust") or "unknown"),
                "progress_unit": str((progress.get("structured_progress") or {}).get("unit") or ""),
                "progress_current": (progress.get("structured_progress") or {}).get("current"),
                "progress_total": (progress.get("structured_progress") or {}).get("total"),
            }
        return {}

    @staticmethod
    def _metric_value_useful_from_signal(signal: dict[str, Any]) -> bool | None:
        assessment = signal.get("resource_metric_value") if isinstance(signal.get("resource_metric_value"), dict) else {}
        useful = assessment.get("useful") if isinstance(assessment, dict) else None
        return useful if isinstance(useful, bool) else None

    def _attach_resource_metric_value_assessment(self, job: ResourceJob, signal: dict[str, Any]) -> None:
        assessment = self._resource_metric_value_assessment(job)
        if not assessment:
            return
        signal["resource_metric_value"] = assessment
        useful = assessment.get("useful")
        if isinstance(useful, bool):
            signal["metric_value_useful"] = useful

    def _score_context_fact_card(self) -> dict[str, Any]:
        if self.task_resource_dir is None:
            return {}
        path = self.task_resource_dir.parent / "lhr_stage_performance.csv"
        summary = build_stage_performance_score_summary(path, current_worker_id="")
        if not summary.get("record_count") and not summary.get("best_score"):
            return {}
        compact: dict[str, Any] = {
            "best_score": summary.get("best_score") or {},
            "valid_best_score": summary.get("valid_best_score") or {},
            "capture_gap": bool(summary.get("capture_gap")),
            "recommended_action": str(summary.get("recommended_action") or ""),
            "cheap_signal_best_score": summary.get("cheap_signal_best_score") or {},
            "record_count": summary.get("record_count") or 0,
            "valid_record_count": summary.get("valid_record_count") or 0,
        }
        for key in ("best_score", "valid_best_score", "cheap_signal_best_score"):
            value = compact.get(key)
            if isinstance(value, dict):
                value.pop("artifact_path", None)
                value.pop("snapshot_path", None)
        return compact


    def _trusted_valid_best_for_intervention(self) -> dict[str, Any]:
        score_context = self._score_context_fact_card()
        valid_best = score_context.get("valid_best_score") if isinstance(score_context.get("valid_best_score"), dict) else {}
        value = valid_best.get("value") if isinstance(valid_best, dict) else None
        if value is None or str(valid_best.get("validity") or "") != "valid_comparable":
            return {
                "current_valid_best": "unknown",
                "current_valid_best_status": "best_unreliable",
                "reason": "metric_direction_or_source_uncertain",
            }
        return {
            "current_valid_best": value,
            "current_valid_best_source": "lhr_stage_performance.csv",
            "current_valid_best_validity": "valid_comparable",
            "lower_is_better": bool(valid_best.get("lower_is_better")),
            "worker_id": str(valid_best.get("worker_id") or ""),
            "stage_id": str(valid_best.get("stage_id") or ""),
        }

    def _resource_intervention_summary_text(self, *, action: str, reason: str) -> str:
        best = self._trusted_valid_best_for_intervention()
        fields = [
            f"action={self._summary_token(action or 'resource_intervention')}",
            f"reason={self._summary_token(reason or 'resource_intervention')}",
        ]
        if best.get("current_valid_best_status") == "best_unreliable":
            fields.extend([
                "current_valid_best=unknown",
                "current_valid_best_status=best_unreliable",
                f"best_unreliable_reason={self._summary_token(best.get('reason') or 'unknown')}",
            ])
        else:
            fields.extend([
                f"current_valid_best={float(best.get('current_valid_best')):.6g}",
                f"current_valid_best_source={best.get('current_valid_best_source')}",
                f"current_valid_best_validity={best.get('current_valid_best_validity')}",
                f"lower_is_better={str(bool(best.get('lower_is_better'))).lower()}",
            ])
            if best.get("worker_id"):
                fields.append(f"best_worker={self._summary_token(best.get('worker_id'))}")
            if best.get("stage_id"):
                fields.append(f"best_stage={self._summary_token(best.get('stage_id'))}")
        return "RESOURCE_INTERVENTION_SUMMARY: " + "; ".join(fields) + ".\n"

    def _append_resource_intervention_summary(self, feedback: str, *, action: str, reason: str) -> str:
        text = str(feedback or "")
        if not text.strip() or "RESOURCE_INTERVENTION_SUMMARY:" in text:
            return text
        if "RESOURCE_FEEDBACK:" not in text:
            return text
        return text.rstrip() + "\n" + self._resource_intervention_summary_text(action=action, reason=reason)

    @staticmethod
    def _summary_token(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip()).strip("_") or "unknown"


    def _record_resource_review_proposal_event(
        self,
        job: ResourceJob,
        proposal: dict[str, Any],
        signal: dict[str, Any],
    ) -> None:
        if self.resource_runtime is None:
            return
        proposal = self._ensure_resource_proposal_facts(job, proposal, signal)
        if not isinstance(proposal.get("score_context"), dict):
            score_context = self._score_context_fact_card()
            if score_context:
                proposal["score_context"] = score_context
        if not isinstance(proposal.get("review_history"), dict):
            proposal.update(self._attach_review_history(proposal, now=time.time()))
        proposal = self._attach_budget_priority(proposal)
        proposal_id = str(proposal.get("proposal_id") or "")
        if not proposal_id:
            return
        trace_id = f"trace_{proposal_id}"
        self.resource_runtime.update_active_resource_proposal(proposal_id, proposal)
        self.resource_runtime.record_resource_event(
            "snapshot",
            payload={
                "resource_snapshot": proposal.get("resource_snapshot") or {},
                "progress_snapshot": proposal.get("progress_snapshot") or {},
                "blocker": proposal.get("blocker") or {},
                "waiters": proposal.get("waiters") or [],
                "execution_facts": proposal.get("execution_facts") or {},
                "research_cadence": proposal.get("research_cadence") or {},
            },
            trace_id=trace_id,
            proposal_id=proposal_id,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        self.resource_runtime.record_resource_event(
            "resource_review_proposal",
            payload=proposal,
            trace_id=trace_id,
            proposal_id=proposal_id,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        if proposal.get("requires_llm_decision"):
            self.resource_runtime.record_resource_event(
                "arbiter_input",
                payload={
                    "actor_id": "resource_arbiter",
                    "cache_namespace": "resource_arbiter",
                    "proposal_id": proposal_id,
                    "input_summary": {
                        "proposal_type": proposal.get("proposal_type"),
                        "reason": proposal.get("reason_code"),
                        "progress_confidence": (proposal.get("progress_snapshot") or {}).get("progress_confidence") if isinstance(proposal.get("progress_snapshot"), dict) else None,
                        "resource_confidence": (proposal.get("blocker") or {}).get("resource_confidence") if isinstance(proposal.get("blocker"), dict) else None,
                        "waiter_count": len(proposal.get("waiters") or []),
                        "intent_device_mismatch": (proposal.get("execution_facts") or {}).get("intent_device_mismatch") if isinstance(proposal.get("execution_facts"), dict) else None,
                        "observed_device": (proposal.get("execution_facts") or {}).get("observed_device") if isinstance(proposal.get("execution_facts"), dict) else None,
                    },
                },
                trace_id=trace_id,
                proposal_id=proposal_id,
                command_id=job.job_id,
                lease_id=job.job_id,
            )

    def _primary_kill_replan_candidate(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        share_blocking_reasons: list[str] = []
        progress = str(job.progress_signal or "unknown").strip().lower()
        progress_state = job.progress_signal_state if isinstance(job.progress_signal_state, dict) else {}
        windows = int(progress_state.get("progress_signal_windows") or job.progress_signal_windows or 0)
        if progress in {"stalled", "degraded"} and (windows >= self.arbiter_min_progress_windows or bool(progress_state.get("multi_window_low_progress"))):
            reason = f"progress_signal={progress}"
            reasons.append(reason)
            share_blocking_reasons.append(reason)
        if job.idle_gpu_lease_samples >= max(1, int(self.gpu_idle_lease_min_samples or 1)):
            reasons.append("idle_gpu_lease_samples_present")
            share_blocking_reasons.append("idle_gpu_lease_samples_present")
        if job.dataloader_bottleneck_samples >= max(1, int(self.gpu_dataloader_bottleneck_min_samples or 1)):
            progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=float(signal.get("elapsed_sec") or 0.0))
            active_work = active_work_counterevidence(progress_snapshot, thresholds=self._lease_suspect_thresholds())
            if active_work.get("active"):
                reasons.append("dataloader_bottleneck_samples_present")
            else:
                reasons.append("low_compute_no_active_work")
                share_blocking_reasons.append("low_compute_no_active_work")
        invalid_events = int(signal.get("invalid_metric_events") or 0)
        zero_events = int(signal.get("zero_score_events") or 0)
        if invalid_events >= self.metric_health_invalid_min_events:
            reasons.append("invalid_metric_events")
            share_blocking_reasons.append("invalid_metric_events")
        if zero_events >= self.metric_health_zero_score_min_events:
            reasons.append("zero_score_events")
            share_blocking_reasons.append("zero_score_events")
        share_blocking_set = set(share_blocking_reasons)
        return {
            "candidate": bool(reasons),
            "share_blocking_candidate": bool(share_blocking_reasons),
            "reasons": reasons,
            "share_blocking_reasons": share_blocking_reasons,
            "review_only_reasons": [reason for reason in reasons if reason not in share_blocking_set],
            "source": "v5_kill_efficiency_pipeline",
            "progress_signal": progress,
            "progress_signal_windows": windows,
            "idle_gpu_lease_samples": int(job.idle_gpu_lease_samples or 0),
            "dataloader_bottleneck_samples": int(job.dataloader_bottleneck_samples or 0),
            "invalid_metric_events": invalid_events,
            "zero_score_events": zero_events,
            "route_viability": dict(job.route_viability_state or {}),
        }

    def _record_gpu_share_event(self, event_type: str, job: ResourceJob, payload: dict[str, Any]) -> None:
        if self.resource_runtime is None:
            return
        waiter = payload.get("waiter") if isinstance(payload.get("waiter"), dict) else {}
        key = f"{event_type}:{job.job_id}:{waiter.get('job_id') or ''}"
        now = time.time()
        last = float(self._last_gpu_share_event_emit.get(key) or 0.0)
        cooldown = max(30.0, float(self.gpu_util_sample_interval_sec or 30.0))
        if last and now - last < cooldown:
            return
        self._last_gpu_share_event_emit[key] = now
        self.resource_runtime.record_resource_event(
            event_type,
            payload=payload,
            command_id=job.job_id,
            lease_id=job.job_id,
        )

    def _grant_shared_gpu_if_allowed(self, job: ResourceJob, observation: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        if self.resource_runtime is None or not self.gpu_share_config.grant_active:
            return {"enabled": False, "reason": "gpu_share_observe_only"}
        if not observation.get("share_eligible"):
            return {"enabled": False, "reason": "share_not_eligible"}
        waiter = observation.get("waiter") if isinstance(observation.get("waiter"), dict) else {}
        secondary_job_id = str(waiter.get("job_id") or "").strip()
        if not secondary_job_id:
            return {"enabled": False, "reason": "missing_secondary_job_id"}
        if not (observation.get("secondary_allowed") or {}).get("allowed"):
            return {"enabled": False, "reason": str((observation.get("secondary_allowed") or {}).get("reason") or "secondary_not_allowed")}
        secondary_allowed = observation.get("secondary_allowed") if isinstance(observation.get("secondary_allowed"), dict) else {}
        metadata = {
            "lease_mode": "shared_secondary",
            "shared_primary_job_id": job.job_id,
            "shared_primary_worker_id": self.worker_id,
            "shared_primary_resource_class": job.resource_class,
            "shared_grant_reason": "task_gpu_share_review",
            "shared_grant_phase": self.gpu_share_config.phase,
            "shared_grant_cpu_pressure": observation.get("cpu_pressure"),
            "shared_grant_memory": observation.get("memory") or {},
            "shared_grant_hard_gates": observation.get("hard_gates") or {},
            "shared_effective_resource_class": secondary_allowed.get("effective_secondary_class") or "",
            "shared_effective_slot_weight": secondary_allowed.get("effective_slot_weight"),
            "trial_share": bool((observation.get("trial_share") or {}).get("enabled")) if isinstance(observation.get("trial_share"), dict) else False,
            "trial_mode": str((observation.get("trial_share") or {}).get("mode") or "") if isinstance(observation.get("trial_share"), dict) else "",
            "trial_initial_observe_sec": float((observation.get("trial_share") or {}).get("initial_observe_sec") or 0.0) if isinstance(observation.get("trial_share"), dict) else 0.0,
            "trial_primary_protected": bool((observation.get("trial_share") or {}).get("primary_protected")) if isinstance(observation.get("trial_share"), dict) else False,
        }
        result = self.resource_runtime.grant_shared_gpu_lease(
            primary_job_id=job.job_id,
            secondary_job_id=secondary_job_id,
            metadata=metadata,
        )
        if result.get("acquired"):
            self._record_gpu_share_event(
                "shared_gpu_lease_granted",
                job,
                {**payload, "grant_result": result, "action": "GRANT_SHARED_GPU_LEASE"},
            )
            return {"enabled": True, "granted": True, "result": result}
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "gpu_share_denied_cpu_support",
                payload={**payload, "grant_result": result, "action": "DENY_SHARE_USE_CPU_SUPPORT"},
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return {"enabled": True, "granted": False, "result": result}

    def _shared_secondary_revoked_decision(self, job: ResourceJob) -> dict[str, Any]:
        if self.resource_runtime is None or str(job.lease_mode or "") != "shared_secondary":
            return {"enabled": False}
        if not self.resource_runtime.has_active_lease(job_id=job.job_id):
            return {"enabled": False}
        persisted = self.resource_runtime.persisted_lease(job_id=job.job_id)
        meta = persisted.get("metadata") if isinstance(persisted.get("metadata"), dict) else {}
        if persisted and str(meta.get("lease_mode") or "") == "shared_secondary":
            return {"enabled": False}
        payload = {
            "job_id": job.job_id,
            "primary_job_id": job.shared_primary_job_id,
            "lease_mode_before": "shared_secondary",
            "lease_mode_after": "revoked",
            "reason": "shared_secondary_lease_revoked",
        }
        self.resource_runtime.record_resource_event(
            "shared_gpu_secondary_stopped",
            payload=payload,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        return {
            "enabled": True,
            "terminate": True,
            "would_terminate": True,
            "reason": "active_intervention:shared_secondary_lease_revoked",
            "check_interval_sec": self.check_interval_sec,
            "feedback": (
                "RESOURCE_FEEDBACK: stopped_shared_secondary because the revocable shared GPU lease was withdrawn to protect the primary job. Replan or switch to CPU support work.\n"
            ),
        }

    def _shared_runtime_review_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        if self.resource_runtime is None or not self.gpu_share_config.grant_active:
            return {"enabled": False}
        persisted = self.resource_runtime.persisted_lease(job_id=job.job_id)
        meta = persisted.get("metadata") if isinstance(persisted.get("metadata"), dict) else {}
        if str(meta.get("lease_mode") or "") != "shared_primary":
            return {"enabled": False}
        secondary_ids = [str(x) for x in (meta.get("shared_secondary_job_ids") or []) if str(x).strip()]
        if not secondary_ids:
            return {"enabled": False}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        progress = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        sample = job.last_gpu_util_sample.get("sample") if isinstance(job.last_gpu_util_sample, dict) else {}
        memory = gpu_memory_summary(sample if isinstance(sample, dict) else {}, job.gpu_ids, previous_peak_gb=job.gpu_mem_peak_gb)
        cpu_pressure = classify_cpu_pressure(dict(signal.get("process_tree_cpu") or {}), cpu_policy=self.gpu_share_config.cpu_policy)
        progress_signal = str(progress.get("progress_signal") or "unknown").lower()
        windows = int(progress.get("progress_signal_windows") or 0)
        reason = ""
        if progress_signal in {"stalled", "degraded"} and windows >= max(1, int(self.gpu_share_config.revoke_min_windows or 1)):
            reason = f"primary_progress_{progress_signal}"
        elif int(signal.get("invalid_metric_events") or 0) >= self.metric_health_invalid_min_events:
            reason = "primary_invalid_metric"
        elif bool(memory.get("sample_available")) and float(memory.get("gpu_free_mem_gb") or 0.0) < self.gpu_share_config.safety_margin_gb:
            reason = "memory_headroom_below_safety_margin"
        proposal_payload = {
            "schema_version": 1,
            "proposal_type": "shared_runtime_review",
            "primary": {
                "job_id": job.job_id,
                "lease_mode": "shared_primary",
                "progress_signal": progress_signal,
                "progress_signal_windows": windows,
                "gpu_ids": list(job.gpu_ids or []),
                "resource_class": job.resource_class,
            },
            "secondary_job_ids": secondary_ids,
            "memory": memory,
            "cpu_pressure": cpu_pressure,
            "reason": reason or "continue_shared_observe",
        }
        self._record_gpu_share_event("shared_runtime_review_proposal", job, proposal_payload)
        if not reason:
            self._record_gpu_share_event(
                "shared_runtime_review_decision",
                job,
                {**proposal_payload, "action": "CONTINUE_SHARED_OBSERVE"},
            )
            return {"enabled": True, "terminate": False, "would_terminate": False, "action": "CONTINUE_SHARED_OBSERVE"}
        releases = []
        for secondary_id in secondary_ids:
            releases.append(self.resource_runtime.revoke_shared_gpu_lease(
                primary_job_id=job.job_id,
                secondary_job_id=secondary_id,
                reason=reason,
            ))
        self._record_gpu_share_event(
            "shared_runtime_review_decision",
            job,
            {**proposal_payload, "action": "STOP_SECONDARY_SHARED_JOB", "releases": releases},
        )
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "action": "STOP_SECONDARY_SHARED_JOB",
            "reason": f"shared_runtime_review:{reason}",
            "feedback": (
                f"RESOURCE_FEEDBACK: STOP_SECONDARY_SHARED_JOB because shared GPU secondary work is harming or risking the primary; reason={reason}.\n"
            ),
            "releases": releases,
        }

    def _task_gpu_share_review_proposal(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
        progress_snapshot: dict[str, Any],
        observation: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        proposal_type = "task_gpu_share_review"
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1((proposal_type + str(now)).encode('utf-8')).hexdigest()[:8]}"
        waiter = observation.get("waiter") if isinstance(observation.get("waiter"), dict) else {}
        trial_share = observation.get("trial_share") if isinstance(observation.get("trial_share"), dict) else {}
        reason_code = "task_gpu_share_review:revocable_trial_waiter" if trial_share.get("enabled") else "task_gpu_share_review:eligible_secondary_waiter"
        trigger_reasons = ["share_eligible", str((observation.get("secondary_allowed") or {}).get("reason") or "secondary_allowed")]
        if trial_share.get("enabled"):
            trigger_reasons.append("revocable_trial_share")
        share_decision_facts = self._gpu_share_decision_facts(observation, decision_source="primary_runtime")
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "severity": "yellow",
            "reason_code": reason_code,
            "trigger_reasons": trigger_reasons,
            "suggested_actions": ["GRANT_SHARED_GPU_LEASE", "DENY_SHARE_USE_CPU_SUPPORT", "CONTINUE_SHARED_OBSERVE"],
            "requires_llm_decision": True,
            "resource_snapshot": {
                "gpu_ids": list(job.gpu_ids or []),
                "cpu_pressure": observation.get("cpu_pressure"),
                "cpu_isolation": observation.get("cpu_isolation") or {},
                "memory": observation.get("memory") or {},
                "hard_gates": observation.get("hard_gates") or {},
                "secondary_allowed": observation.get("secondary_allowed") or {},
                "trial_share": trial_share,
            },
            "progress_snapshot": progress_snapshot,
            "blocker": {
                "job_id": job.job_id,
                "worker_id": self.worker_id,
                "resource_class": job.resource_class,
                "runtime_sec": elapsed_sec,
                "progress_signal": progress_snapshot.get("progress_signal"),
                "progress_confidence": progress_snapshot.get("progress_confidence"),
                "share_role": "primary",
            },
            "waiters": [waiter],
            "share_observation": observation,
            "share_decision_facts": share_decision_facts,
            "share_payload": payload,
            "decision_preview": {
                "job_id": job.job_id,
                "reason": reason_code,
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }
        return proposal

    def apply_task_gpu_share_decision(
        self,
        job_id: str | None,
        *,
        arbiter_action: str = "",
        proposal: dict[str, Any] | None = None,
        arbiter_decision: dict[str, Any] | None = None,
        elapsed_sec: float = 0.0,
        **_: Any,
    ) -> dict[str, Any]:
        action = str(arbiter_action or "").upper()
        if not job_id or job_id not in self._jobs:
            return {"enabled": False, "reason": "job_not_found", "action": action}
        job = self._jobs[job_id]
        prop = proposal if isinstance(proposal, dict) else {}
        observation = prop.get("share_observation") if isinstance(prop.get("share_observation"), dict) else {}
        payload = prop.get("share_payload") if isinstance(prop.get("share_payload"), dict) else {}
        if action == "GRANT_SHARED_GPU_LEASE":
            grant = self._grant_shared_gpu_if_allowed(job, observation, payload)
            self._record_gpu_share_event(
                "gpu_share_decision",
                job,
                {**payload, "action": action, "arbiter_decision": arbiter_decision or {}, "grant": grant},
            )
            feedback = "RESOURCE_FEEDBACK: GRANT_SHARED_GPU_LEASE accepted; a revocable secondary may run on the underused task-local GPU.\n"
            if not grant.get("granted"):
                feedback = f"RESOURCE_FEEDBACK: GRANT_SHARED_GPU_LEASE could not be applied; reason={grant.get('reason') or (grant.get('result') or {}).get('reason') or 'share_grant_failed'}.\n"
            return {"enabled": True, "action": action, "granted": bool(grant.get("granted")), "grant": grant, "feedback": feedback}
        if action in {"DENY_SHARE_USE_CPU_SUPPORT", "CONTINUE_SHARED_OBSERVE"}:
            self._record_gpu_share_event(
                "gpu_share_decision",
                job,
                {**payload, "action": action, "arbiter_decision": arbiter_decision or {}},
            )
            feedback_action = "LOCAL_GPU_BUSY_USE_CPU_SUPPORT" if action == "DENY_SHARE_USE_CPU_SUPPORT" else "CONTINUE_SHARED_OBSERVE"
            return {
                "enabled": True,
                "action": action,
                "granted": False,
                "feedback": f"RESOURCE_FEEDBACK: {feedback_action} because shared GPU grant was not approved; use CPU support or observe until the next resource review.\n",
            }
        return {"enabled": False, "reason": "share_action_not_applicable", "action": action}

    def _gpu_share_phase_a_decision(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        allow_grant: bool = True,
        allow_handoff: bool = True,
    ) -> dict[str, Any]:
        cfg = self.gpu_share_config
        if not (cfg.observe_active and self.resource_runtime is not None):
            return {"enabled": False}
        if signal.get("saw_final_score"):
            return {"enabled": False}
        if not self.resource_runtime.has_active_lease(job_id=job.job_id):
            return {"enabled": False}
        if not self._job_uses_expensive_gpu(job):
            return {"enabled": False}
        try:
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception:
            return {"enabled": False, "reason": "gpu_share_snapshot_failed"}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        sample = job.last_gpu_util_sample.get("sample") if isinstance(job.last_gpu_util_sample, dict) else {}
        if not isinstance(sample, dict):
            sample = {}
        primary_kill = self._primary_kill_replan_candidate(job, signal)
        progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        observation = evaluate_share_phase_a(
            cfg=cfg,
            primary_job_id=job.job_id,
            primary_resource_class=job.resource_class,
            primary_gpu_ids=job.gpu_ids,
            elapsed_sec=elapsed,
            progress_snapshot=progress_snapshot,
            gpu_sample=sample,
            previous_mem_peak_gb=job.gpu_mem_peak_gb,
            process_tree_cpu=dict(signal.get("process_tree_cpu") or {}),
            snapshot=snapshot,
            primary_kill_replan_candidate=primary_kill,
        )
        if not observation.get("enabled"):
            return observation
        payload = {
            "schema_version": 1,
            "phase": observation.get("phase"),
            "observe_only": bool(observation.get("observe_only")),
            "primary": {
                "job_id": job.job_id,
                "worker_id": self.worker_id,
                "resource_class": job.resource_class,
                "gpu_ids": list(job.gpu_ids or []),
                "runtime_sec": elapsed,
                "progress_signal": progress_snapshot.get("progress_signal"),
                "progress_confidence": progress_snapshot.get("progress_confidence"),
                "gpu_mem_peak_gb": job.gpu_mem_peak_gb,
            },
            "waiter": observation.get("waiter") or {},
            "waiters": observation.get("waiters") or [],
            "cpu_pressure": observation.get("cpu_pressure"),
            "cpu_isolation": observation.get("cpu_isolation") or {},
            "memory": observation.get("memory") or {},
            "hard_gates": observation.get("hard_gates") or {},
            "secondary_allowed": observation.get("secondary_allowed") or {},
            "trial_share": observation.get("trial_share") or {},
            "primary_kill_replan_candidate": primary_kill,
        }
        if primary_kill.get("candidate"):
            self._record_gpu_share_event("primary_kill_replan_candidate", job, payload)
            if primary_kill.get("share_blocking_candidate") and allow_handoff:
                review = self._periodic_efficiency_review_decision(
                    job,
                    signal,
                    force=True,
                    trigger_source="gpu_share_handoff",
                    trigger_reasons=[str(x) for x in (primary_kill.get("reasons") or [])],
                )
                if isinstance(review, dict) and review.get("enabled"):
                    return {**review, "gpu_share_handoff": True}
            if primary_kill.get("share_blocking_candidate"):
                return {
                    "enabled": True,
                    "gpu_share_handoff": bool(allow_handoff),
                    "handoff_review_enabled": False,
                    "observation": observation,
                }
        if observation.get("share_candidate"):
            self._record_gpu_share_event("gpu_share_candidate", job, payload)
        grant: dict[str, Any] = {"enabled": False}
        if observation.get("share_eligible"):
            self._record_gpu_share_event("gpu_share_eligible", job, payload)
            secondary_allowed = observation.get("secondary_allowed") if isinstance(observation.get("secondary_allowed"), dict) else {}
            requires_llm_policy_decision = bool(secondary_allowed.get("requires_llm_policy_decision"))
            if allow_grant and self.arbiter_enabled and self.arbiter_mode == "llm":
                proposal = self._task_gpu_share_review_proposal(
                    job,
                    signal,
                    elapsed_sec=elapsed,
                    progress_snapshot=progress_snapshot,
                    observation=observation,
                    payload=payload,
                )
                self._record_resource_review_proposal_event(job, proposal, signal)
                return {
                    "enabled": True,
                    "terminate": False,
                    "would_terminate": False,
                    "arbiter_review": True,
                    "requires_llm_decision": True,
                    "arbiter_enabled": True,
                    "reason": str(proposal.get("reason_code") or "task_gpu_share_review:eligible_secondary_waiter"),
                    "check_interval_sec": self.check_interval_sec,
                    "feedback": "RESOURCE_FEEDBACK: task_gpu_share_review because a queued waiter can potentially share an underused task-local GPU; arbiter review requested.\n",
                    "observation": observation,
                    "proposal": proposal,
                }
            if requires_llm_policy_decision:
                grant = {
                    "enabled": False,
                    "reason": "share_policy_requires_llm_arbiter",
                    "policy_gate": secondary_allowed.get("policy_gate") or "secondary_policy",
                }
            elif allow_grant:
                grant = self._grant_shared_gpu_if_allowed(job, observation, payload)
            else:
                grant = {"enabled": False, "reason": "post_arbiter_observe_only"}
        feedback = ""
        if grant.get("granted"):
            feedback = "RESOURCE_FEEDBACK: GRANT_SHARED_GPU_LEASE accepted; a revocable secondary may run on the underused task-local GPU.\n"
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "feedback": feedback,
            "observation": observation,
            "shared_lease_grant": grant,
        }

    def _contention_review_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        if not (self.arbiter_enabled and self.arbiter_contention_review_enabled and self.resource_runtime is not None):
            return {"enabled": False}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if elapsed < self.arbiter_contention_min_runtime_sec or signal.get("saw_final_score"):
            return {"enabled": False}
        if not self.resource_runtime.has_active_lease(job_id=job.job_id):
            return {"enabled": False}
        if not self._job_uses_expensive_gpu(job):
            return {"enabled": False}
        now = time.time()
        if self._review_cooldown_active(job, "resource_contention_review", min_interval_sec=self.arbiter_contention_min_interval_sec, now=now):
            return {"enabled": False}
        try:
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception:
            return {"enabled": False}
        waiters = self._contention_waiter_cards(job, snapshot=snapshot, now=now)
        if not waiters:
            return {"enabled": False}
        resource_confidence = self._resource_confidence_for_review(job, has_active_lease=True, waiter_count=len(waiters))
        proposal_type = "resource_contention_review"
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1((proposal_type + str(now)).encode('utf-8')).hexdigest()[:8]}"
        progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        route_viability = self._update_v7_route_profile(job, signal, elapsed_sec=elapsed)
        progress_snapshot.update({
            "route_viability": dict(route_viability or {}),
        })
        sample = job.last_gpu_util_sample.get("sample") if isinstance(job.last_gpu_util_sample, dict) else {}
        memory = gpu_memory_summary(sample if isinstance(sample, dict) else {}, job.gpu_ids, previous_peak_gb=job.gpu_mem_peak_gb)
        suspect = classify_active_lease_suspect(
            elapsed_sec=elapsed,
            has_waiter=True,
            has_active_lease=True,
            resource_class=job.resource_class,
            progress_snapshot=progress_snapshot,
            gpu_memory=memory,
            deliverable_validity="none",
            route_viability=route_viability,
            thresholds=self._lease_suspect_thresholds(),
        )
        job.active_lease_suspect_state = dict(suspect)
        trigger_reasons = [
            "active_gpu_lease",
            "pending_waiter_for_same_gpu",
            f"waiter_age_sec>={self.arbiter_contention_min_waiter_age_sec:.0f}",
        ]
        if suspect.get("resource_suspect"):
            trigger_reasons.append("active_lease_resource_suspect")
        if suspect.get("unknown_progress_suspect"):
            trigger_reasons.append("active_lease_unknown_progress_suspect")
        if suspect.get("value_suspect"):
            trigger_reasons.append("active_lease_value_suspect")
        suggested_actions = ["CONTINUE", "OBSERVE_MORE"]
        if suspect.get("resource_suspect"):
            suggested_actions.append("KILL_AND_REPLAN")
        if suspect.get("active_lease_suspect"):
            self._record_v7_resource_event(
                "active_lease_suspect",
                job,
                {
                    "proposal_type": proposal_type,
                    "waiter_count": len(waiters),
                    "waiters": waiters,
                    "active_lease_suspect": suspect,
                    "route_viability": route_viability,
                    "memory": memory,
                },
            )
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "severity": "yellow" if not suspect.get("resource_suspect") else "orange",
            "reason_code": "resource_contention:active_gpu_job_blocking_waiter",
            "trigger_reasons": trigger_reasons,
            "suggested_actions": suggested_actions,
            "requires_llm_decision": True,
            "resource_snapshot": self._resource_snapshot_for_job(job, signal),
            "progress_snapshot": progress_snapshot,
            "blocker": self._blocker_fact_card(job, signal, elapsed_sec=elapsed, resource_confidence=resource_confidence),
            "waiters": waiters,
            "decision_preview": {
                "job_id": job.job_id,
                "reason": "resource_contention_review:active_gpu_job_blocking_waiter",
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }
        proposal.update(self._attach_review_history(proposal, now=now))
        if suspect.get("unknown_progress_suspect") and proposal.get("structured_opportunity_cost_support"):
            if "KILL_AND_REPLAN" not in suggested_actions:
                suggested_actions.append("KILL_AND_REPLAN")
            if "active_lease_unknown_progress_escalated" not in trigger_reasons:
                trigger_reasons.append("active_lease_unknown_progress_escalated")
            proposal["suggested_actions"] = suggested_actions
            proposal["trigger_reasons"] = trigger_reasons
            proposal["severity"] = "orange"
        self._mark_review_proposal_emitted(job, proposal_type, now=now)
        self._record_resource_review_proposal_event(job, proposal, signal)
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "arbiter_review": True,
            "requires_llm_decision": True,
            "arbiter_enabled": True,
            "reason": "resource_contention_review:active_gpu_job_blocking_waiter",
            "check_interval_sec": self.check_interval_sec,
            "feedback": "RESOURCE_FEEDBACK: resource_contention_review because an active GPU lease is blocking a queued waiter; arbiter review requested.\n",
            "proposal": proposal,
        }

    def _research_cadence_review_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        if not (self.research_cadence_enabled and self.arbiter_enabled and self.resource_runtime is not None):
            return {"enabled": False}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not job.visible or signal.get("saw_final_score"):
            return {"enabled": False}
        progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        cadence = self._research_cadence_fact_card(job, signal, progress_snapshot)
        if not cadence.get("violation"):
            return {"enabled": False}
        now = time.time()
        proposal_type = "periodic_efficiency_review"
        min_interval = max(60.0, min(self.arbiter_periodic_min_interval_sec, self.research_cadence_observe_sec))
        if self._review_cooldown_active(job, proposal_type, min_interval_sec=min_interval, now=now):
            return {"enabled": False}
        reason_code = str(cadence.get("reason_code") or "research_cadence_exceeded")
        resource_snapshot = self._resource_snapshot_for_job(job, signal)
        blocker = self._blocker_fact_card(
            job,
            signal,
            elapsed_sec=elapsed,
            resource_confidence=self._resource_confidence_for_review(
                job,
                has_active_lease=bool(self.resource_runtime.has_active_lease(job_id=job.job_id)),
                waiter_count=0,
            ),
        )
        execution_facts = self._execution_facts_for_job(
            job,
            resource_snapshot=resource_snapshot,
            progress_snapshot=progress_snapshot,
        )
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1((reason_code + str(now)).encode('utf-8')).hexdigest()[:8]}"
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "severity": "orange",
            "reason_code": reason_code,
            "trigger_reasons": [
                "research_cadence_exceeded",
                f"metric_budget_sec={float(cadence.get('metric_budget_sec') or 0.0):.0f}",
                f"route_metric_proven={str(bool(cadence.get('route_metric_proven'))).lower()}",
            ],
            "suggested_actions": ["CONTINUE", "OBSERVE_MORE", "KILL_AND_REPLAN"],
            "requires_llm_decision": True,
            "structured_opportunity_cost_support": True,
            "resource_snapshot": resource_snapshot,
            "progress_snapshot": progress_snapshot,
            "execution_facts": execution_facts,
            "research_cadence": cadence,
            "blocker": blocker,
            "waiters": [],
            "decision_preview": {
                "job_id": job.job_id,
                "reason": reason_code,
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }
        proposal.update(self._attach_review_history(proposal, now=now))
        self._mark_review_proposal_emitted(job, proposal_type, now=now)
        self._record_resource_review_proposal_event(job, proposal, signal)
        feedback = (
            "RESOURCE_FEEDBACK: research_cadence_review because the observed or projected time to the next comparable metric "
            f"exceeds the configured value window; elapsed_sec={elapsed:.0f}; "
            f"eta_to_next_comparable_metric_sec={cadence.get('eta_to_next_comparable_metric_sec')}; "
            f"metric_budget_sec={float(cadence.get('metric_budget_sec') or 0.0):.0f}; "
            f"route_metric_proven={str(bool(cadence.get('route_metric_proven'))).lower()}.\n"
        )
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "arbiter_review": True,
            "requires_llm_decision": True,
            "arbiter_enabled": True,
            "reason": reason_code,
            "check_interval_sec": self.check_interval_sec,
            "feedback": feedback,
            "proposal": proposal,
        }

    def _periodic_efficiency_review_decision(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        force: bool = False,
        trigger_source: str = "periodic_efficiency",
        trigger_reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        if not (self.arbiter_enabled and self.resource_runtime is not None):
            return {"enabled": False}
        efficiency = (
            signal.get("resource_efficiency")
            or job.resource_efficiency_state
            or {}
        )
        efficiency_trigger = bool(
            efficiency.get("review_required")
            and not efficiency.get("review_limit_reached")
        )
        if not force and not self.arbiter_periodic_review_enabled and not efficiency_trigger:
            return {"enabled": False}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        min_runtime = (
            0.0
            if force or efficiency_trigger
            else self.arbiter_periodic_min_runtime_sec
        )
        if elapsed < min_runtime or not job.visible or signal.get("saw_final_score"):
            return {"enabled": False}
        if not self._job_is_expensive_resource_hold(job):
            return {"enabled": False}
        progress = str(job.progress_signal or "unknown")
        merged_reasons: list[str] = [str(x) for x in (trigger_reasons or []) if str(x).strip()]
        if efficiency_trigger:
            merged_reasons.append("sustained_resource_efficiency_mismatch")
            merged_reasons.extend(
                str(reason)
                for reason in efficiency.get("reason_codes") or []
                if str(reason).strip()
            )
        if progress in {"stalled", "degraded"}:
            merged_reasons.append(f"progress_signal={progress}")
        if job.idle_gpu_lease_samples > 0:
            merged_reasons.append("idle_gpu_lease_samples_present")
        if job.dataloader_bottleneck_samples > 0:
            merged_reasons.append("dataloader_bottleneck_samples_present")
        deduped_reasons = list(dict.fromkeys(merged_reasons))
        if not deduped_reasons and not force:
            return {"enabled": False}
        if not deduped_reasons:
            deduped_reasons.append("primary_kill_replan_candidate")
        now = time.time()
        if self._review_cooldown_active(job, "periodic_efficiency_review", min_interval_sec=self.arbiter_periodic_min_interval_sec, now=now):
            return {"enabled": False}
        has_lease = bool(self.resource_runtime.has_active_lease(job_id=job.job_id))
        resource_confidence = self._resource_confidence_for_review(job, has_active_lease=has_lease, waiter_count=0)
        proposal_type = "periodic_efficiency_review"
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1((proposal_type + str(now)).encode('utf-8')).hexdigest()[:8]}"
        periodic_suggested_actions = ["CONTINUE", "OBSERVE_MORE", "KILL_AND_REPLAN"]
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "severity": "yellow",
            "reason_code": (
                "gpu_share_handoff:primary_kill_replan_candidate"
                if force
                else (
                    "resource_efficiency:sustained_resource_mismatch"
                    if efficiency_trigger
                    else "periodic_efficiency:low_progress_review"
                )
            ),
            "trigger_reasons": deduped_reasons,
            "trigger_source": str(trigger_source or "periodic_efficiency"),
            "suggested_actions": periodic_suggested_actions,
            "requires_llm_decision": True,
            "resource_snapshot": self._resource_snapshot_for_job(job, signal),
            "progress_snapshot": self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed),
            "blocker": self._blocker_fact_card(job, signal, elapsed_sec=elapsed, resource_confidence=resource_confidence),
            "waiters": [],
            "decision_preview": {
                "job_id": job.job_id,
                "reason": "gpu_share_handoff:primary_kill_replan_candidate" if force else "periodic_efficiency_review:low_progress_or_low_efficiency",
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }
        proposal.update(self._attach_review_history(proposal, now=now))
        self._mark_review_proposal_emitted(job, proposal_type, now=now)
        self._record_resource_review_proposal_event(job, proposal, signal)
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "arbiter_review": True,
            "requires_llm_decision": True,
            "arbiter_enabled": True,
            "reason": "gpu_share_handoff:primary_kill_replan_candidate" if force else "periodic_efficiency_review:low_progress_or_low_efficiency",
            "check_interval_sec": self.check_interval_sec,
            "feedback": (
                "RESOURCE_FEEDBACK: PRIMARY_KILL_REPLAN_CANDIDATE because a task-local GPU share review found the current GPU holder should enter kill/replan review; arbiter review requested.\n"
                if force
                else "RESOURCE_FEEDBACK: periodic_efficiency_review because long-running resource usage has degraded signals; arbiter review requested.\n"
            ),
            "proposal": proposal,
        }


    def _quick_probe_review_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        if not (self.quick_probe_guard_enabled and self.arbiter_enabled and self.resource_runtime is not None):
            return {"enabled": False}
        quick = dict(job.quick_probe or {})
        if not quick.get("quick_probe_candidate"):
            return {"enabled": False}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        hard_review = max(1.0, float(quick.get("quick_probe_hard_review_sec") or self.quick_probe_hard_review_sec))
        if elapsed < hard_review or signal.get("saw_final_score"):
            return {"enabled": False}
        if int(signal.get("terminal_signal_events") or 0) > 0 or job.terminal_signal_seen:
            return {"enabled": False}
        if not job.visible:
            return {"enabled": False}
        now = time.time()
        min_interval = min(300.0, max(60.0, hard_review / 3.0))
        if self._review_cooldown_active(job, "quick_probe_review", min_interval_sec=min_interval, now=now):
            return {"enabled": False}
        expected = max(1.0, float(quick.get("quick_probe_expected_runtime_sec") or self.quick_probe_expected_runtime_sec))
        output_pattern = str(quick.get("output_pattern") or "unknown")
        progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        progress_snapshot.update({
            "quick_probe": quick,
            "quick_probe_candidate": True,
            "quick_probe_expected_runtime_sec": expected,
            "quick_probe_hard_review_sec": hard_review,
            "quick_probe_runtime_ratio": elapsed / expected if expected > 0 else None,
            "output_pattern": output_pattern,
            "heartbeat_source": progress_snapshot.get("heartbeat_source") or _SAFETY_HEARTBEAT_SOURCE,
        })
        resource_snapshot = self._resource_snapshot_for_job(job, signal)
        blocker = self._blocker_fact_card(
            job,
            signal,
            elapsed_sec=elapsed,
            resource_confidence=self._resource_confidence_for_review(
                job,
                has_active_lease=bool(self.resource_runtime.has_active_lease(job_id=job.job_id)),
                waiter_count=0,
            ),
        )
        execution_facts = self._execution_facts_for_job(
            job,
            resource_snapshot=resource_snapshot,
            progress_snapshot=progress_snapshot,
        )
        state_generation = build_resource_state_generation({
            "proposal_type": "quick_probe_review",
            "reason_code": "quick_probe_runtime_exceeded",
            "resource_snapshot": resource_snapshot,
            "progress_snapshot": progress_snapshot,
            "blocker": blocker,
            "gpu_ids": list(job.gpu_ids or []),
            "execution_facts": execution_facts,
        })
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1(('quick_probe_runtime_exceeded' + state_generation.control_generation_key).encode('utf-8')).hexdigest()[:8]}"
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": "quick_probe_review",
            "severity": "red",
            "reason_code": "quick_probe_runtime_exceeded",
            "trigger_reasons": [
                "quick_probe_candidate",
                f"elapsed_sec>={hard_review:.0f}",
                "terminal_signal_missing",
            ],
            "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
            "requires_llm_decision": True,
            "cooldown_break": True,
            "control_generation_key": state_generation.control_generation_key,
            "feedback_generation_key": state_generation.feedback_generation_key,
            "state_generation": state_generation.to_json(),
            "resource_snapshot": resource_snapshot,
            "progress_snapshot": progress_snapshot,
            "execution_facts": execution_facts,
            "blocker": blocker,
            "waiters": [],
            "structured_opportunity_cost_support": elapsed >= max(3600.0, hard_review * 4.0),
            "decision_preview": {
                "job_id": job.job_id,
                "reason": "quick_probe_runtime_exceeded",
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }
        proposal.update(self._attach_review_history(proposal, now=now))
        self._mark_review_proposal_emitted(job, "quick_probe_review", now=now)
        self._record_resource_review_proposal_event(job, proposal, signal)
        payload = {
            "job_id": job.job_id,
            "elapsed_sec": elapsed,
            "expected_runtime_sec": expected,
            "hard_review_sec": hard_review,
            "output_pattern": output_pattern,
            "control_generation_key": state_generation.control_generation_key,
            "feedback_generation_key": state_generation.feedback_generation_key,
            "proposal_id": proposal_id,
        }
        self.resource_runtime.record_resource_event(
            "quick_probe_forced_review",
            payload=payload,
            proposal_id=proposal_id,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "arbiter_review": True,
            "requires_llm_decision": True,
            "arbiter_enabled": True,
            "reason": "quick_probe_runtime_exceeded",
            "check_interval_sec": self.check_interval_sec,
            "feedback": (
                "RESOURCE_FEEDBACK: quick_probe_runtime_exceeded because a command classified as a short quick/probe/test run exceeded its hard review window; "
                f"elapsed_sec={elapsed:.0f}; expected_runtime_sec={expected:.0f}; hard_review_sec={hard_review:.0f}; output_pattern={output_pattern}.\n"
            ),
            "proposal": proposal,
        }

    @staticmethod
    def _int_bucket(value: Any, *, threshold: int) -> tuple[str, int]:
        try:
            number = int(value or 0)
        except (TypeError, ValueError):
            number = 0
        threshold = max(1, int(threshold or 1))
        if number <= 0:
            return "zero", 0
        if number < threshold:
            return "below_threshold", 1
        if number < threshold * 2:
            return "at_threshold", 2
        return "doubled_threshold", 3

    @staticmethod
    def _boundary_severity_rank(kind: str) -> int:
        value = str(kind or "").strip()
        if value in {STALL, TIMEBOX_EXPIRED}:
            return 4
        if value == RESOURCE_PRESSURE:
            return 3
        if value == ROUTE_VALUE:
            return 2
        if value == PROGRESS_WINDOW:
            return 1
        return 0

    def _kill_proposal_boundary_state(self, decision: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
        boundary = decision.get("resource_review_boundary") if isinstance(decision.get("resource_review_boundary"), dict) else {}
        if not boundary:
            boundary = signal.get("resource_review_boundary") if isinstance(signal.get("resource_review_boundary"), dict) else {}
        review_state = decision.get("resource_review_state") if isinstance(decision.get("resource_review_state"), dict) else {}
        if not review_state:
            review_state = signal.get("resource_review_state") if isinstance(signal.get("resource_review_state"), dict) else {}
        contention = decision.get("contention_context") if isinstance(decision.get("contention_context"), dict) else {}
        if not contention:
            contention = signal.get("contention_context") if isinstance(signal.get("contention_context"), dict) else {}
        no_useful = int(review_state.get("no_useful_progress_windows") or 0)
        no_useful_bucket, no_useful_rank = self._int_bucket(no_useful, threshold=int(self.review_config.value_windows or 1))
        blocked_workers = int(contention.get("blocked_worker_count") or review_state.get("blocked_worker_count") or 0)
        waiter_pressure = bool(contention.get("active_waiter_pressure") or review_state.get("active_waiter_pressure") or blocked_workers > 0)
        kind = str(boundary.get("kind") or "")
        return {
            "boundary_kind": kind,
            "boundary_reason": str(boundary.get("reason") or ""),
            "boundary_severity_rank": self._boundary_severity_rank(kind),
            "job_state_bucket": str(review_state.get("job_state_bucket") or ""),
            "no_useful_progress_windows": no_useful,
            "no_useful_progress_bucket": no_useful_bucket,
            "no_useful_progress_rank": no_useful_rank,
            "timebox_id": str(review_state.get("timebox_id") or ""),
            "timebox_windows": int(review_state.get("timebox_windows") or 0),
            "active_waiter_pressure": waiter_pressure,
            "blocked_worker_count": blocked_workers,
        }

    @staticmethod
    def _kill_proposal_boundary_escalated(previous: dict[str, Any] | None, current: dict[str, Any]) -> bool:
        if not previous:
            return True
        if int(current.get("boundary_severity_rank") or 0) > int(previous.get("boundary_severity_rank") or 0):
            return True
        if (
            int(current.get("no_useful_progress_windows") or 0) > int(previous.get("no_useful_progress_windows") or 0)
            and int(current.get("no_useful_progress_rank") or 0) >= 2
        ):
            return True
        if str(current.get("boundary_kind") or "") == TIMEBOX_EXPIRED:
            return True
        if bool(current.get("active_waiter_pressure")) and not bool(previous.get("active_waiter_pressure")):
            return True
        if int(current.get("blocked_worker_count") or 0) > int(previous.get("blocked_worker_count") or 0):
            return True
        return False

    def _record_kill_proposal_event(
        self,
        job: ResourceJob,
        decision: dict[str, Any],
        signal: dict[str, Any],
        *,
        source: str,
    ) -> dict[str, Any] | None:
        if self.resource_runtime is None or not decision.get("would_terminate"):
            return None
        now = time.time()
        reason = str(decision.get("reason") or "resource_guard")
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        resource_snapshot = self._resource_snapshot_for_job(job, signal)
        progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        execution_facts = self._execution_facts_for_job(
            job,
            resource_snapshot=resource_snapshot,
            progress_snapshot=progress_snapshot,
        )
        state_generation = build_resource_state_generation({
            "proposal_type": "kill_proposal",
            "reason_code": reason,
            "resource_snapshot": resource_snapshot,
            "progress_snapshot": progress_snapshot,
            "gpu_ids": list(job.gpu_ids or []),
            "execution_facts": execution_facts,
        })
        dedupe_key = f"{job.job_id}:{reason}:{state_generation.control_generation_key}"
        boundary_state = self._kill_proposal_boundary_state(decision, signal)
        previous_boundary_state = self._last_kill_proposal_boundary_by_dedupe.get(dedupe_key)
        escalated_boundary = self._kill_proposal_boundary_escalated(previous_boundary_state, boundary_state)
        last = float(self._last_kill_proposal_emit.get(dedupe_key) or 0.0)
        cooldown = max(1.0, float(self.kill_proposal_cooldown_sec or 600.0))
        cooldown_active = bool(last and now - last < cooldown)
        if cooldown_active and not escalated_boundary:
            suppressed_last = float(self._last_suppressed_kill_proposal_emit.get(dedupe_key) or 0.0)
            if now - suppressed_last >= min(60.0, cooldown):
                self._last_suppressed_kill_proposal_emit[dedupe_key] = now
                self.resource_runtime.record_resource_event(
                    "suppressed_proposal",
                    payload={
                        "reason": reason,
                        "dedupe_key": dedupe_key,
                        "control_generation_key": state_generation.control_generation_key,
                        "feedback_generation_key": state_generation.feedback_generation_key,
                        "state_generation": state_generation.to_json(),
                        "boundary_state": boundary_state,
                        "previous_boundary_state": previous_boundary_state or {},
                        "suppressed": True,
                        "suppressed_reason": "cooldown_same_unchanged_boundary",
                        "legacy_suppressed_reason": "proposal_cooldown_same_control_state",
                        "suppressed_by_proposal_id": self._last_kill_proposal_id_by_dedupe.get(dedupe_key, ""),
                        "would_trigger_arbiter": False,
                        "previous_same_kind_age_sec": now - last,
                        "cooldown_sec": cooldown,
                    },
                    command_id=job.job_id,
                    lease_id=job.job_id,
                )
            return None
        self._last_kill_proposal_emit[dedupe_key] = now
        self._last_kill_proposal_boundary_by_dedupe[dedupe_key] = dict(boundary_state)
        boundary_key = json.dumps(boundary_state, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1((reason + state_generation.control_generation_key + boundary_key).encode('utf-8', errors='replace')).hexdigest()[:8]}"
        self._last_kill_proposal_id_by_dedupe[dedupe_key] = proposal_id
        trace_id = f"trace_{proposal_id}"
        decision["proposal_id"] = proposal_id
        decision["job_id"] = job.job_id
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": "kill_proposal",
            "severity": "red" if decision.get("terminate") else "yellow",
            "reason_code": reason,
            "dedupe_key": dedupe_key,
            "control_generation_key": state_generation.control_generation_key,
            "feedback_generation_key": state_generation.feedback_generation_key,
            "state_generation": state_generation.to_json(),
            "boundary_state": boundary_state,
            "previous_boundary_state": previous_boundary_state or {},
            "cooldown_bypassed_by_escalation": bool(cooldown_active and escalated_boundary),
            "proposal_cooldown_sec": cooldown,
            "suppressed": False,
            "suppressed_reason": "",
            "requires_llm_decision": bool(decision.get("requires_llm_decision")),
            "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
            "pending_recommendations": ["release_unused_lease"] if "idle_gpu_lease" in reason else [],
            "resource_snapshot": resource_snapshot,
            "progress_snapshot": progress_snapshot,
            "execution_facts": execution_facts,
            "resource_metric_value": dict(
                signal.get("resource_metric_value")
                or decision.get("resource_metric_value")
                or {}
            ),
            "decision_preview": dict(decision),
            "source": str(source or "resource_guard"),
            "command_id": job.job_id,
        }
        proposal = self._attach_budget_priority(proposal)
        self.resource_runtime.update_active_resource_proposal(proposal_id, proposal)
        self.resource_runtime.record_resource_event(
            "snapshot",
            payload={
                "resource_snapshot": proposal["resource_snapshot"],
                "progress_snapshot": proposal["progress_snapshot"],
                "execution_facts": proposal.get("execution_facts") or {},
            },
            trace_id=trace_id,
            proposal_id=proposal_id,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        self.resource_runtime.record_resource_event(
            "kill_proposal",
            payload=proposal,
            trace_id=trace_id,
            proposal_id=proposal_id,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        if decision.get("requires_llm_decision"):
            self.resource_runtime.record_resource_event(
                "arbiter_input",
                payload={
                    "actor_id": "resource_arbiter",
                    "cache_namespace": "resource_arbiter",
                    "proposal_id": proposal_id,
                    "input_summary": {
                        "reason": reason,
                        "progress_confidence": proposal["progress_snapshot"].get("progress_confidence"),
                        "runtime_sec": elapsed,
                        "intent_device_mismatch": execution_facts.get("intent_device_mismatch"),
                        "observed_device": execution_facts.get("observed_device"),
                    },
                },
                trace_id=trace_id,
                proposal_id=proposal_id,
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        decision["proposal"] = proposal
        return proposal


    @staticmethod
    def _budget_waiter_cards(proposal: dict[str, Any]) -> list[dict[str, Any]]:
        waiters = proposal.get("waiters")
        if isinstance(waiters, list):
            return [row for row in waiters if isinstance(row, dict)]
        candidate = proposal.get("candidate")
        if isinstance(candidate, dict) and candidate:
            return [candidate]
        return []

    def _budget_priority_for_proposal(self, proposal: dict[str, Any]) -> dict[str, Any]:
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        blocker = proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {}
        suspect = blocker.get("active_lease_suspect") if isinstance(blocker.get("active_lease_suspect"), dict) else {}
        waiters = self._budget_waiter_cards(proposal)
        suggested_actions = {str(x).upper() for x in (proposal.get("suggested_actions") or [])}
        reason_code = str(proposal.get("reason_code") or "").strip().lower()
        proposal_type = str(proposal.get("proposal_type") or "").strip().lower()
        progress_signal = str(progress.get("progress_signal") or blocker.get("progress_signal") or "").strip().lower()
        progress_confidence = str(progress.get("progress_confidence") or blocker.get("progress_confidence") or "").strip().lower()
        efficiency = progress.get("resource_efficiency") if isinstance(progress.get("resource_efficiency"), dict) else {}
        queue_ages: list[float] = []
        for row in waiters:
            try:
                queue_ages.append(max(0.0, float(row.get("queue_age_sec") or 0.0)))
            except (TypeError, ValueError):
                continue
        max_queue_age = max(queue_ages) if queue_ages else 0.0
        score = 0
        reasons: list[str] = []
        job_key = self._proposal_job_id(proposal)
        metric_history_digest = self._proposal_metric_history_digest(proposal)
        metric_history_changed = bool(
            job_key
            and metric_history_digest
            and metric_history_digest != self._job_last_llm_metric_history_digest.get(job_key, "")
        )
        if metric_history_changed:
            reasons.append("new_metric_history_since_llm_review")
        if waiters:
            score += 2
            reasons.append("waiter_blocked")
        if max_queue_age >= max(1.0, float(self.arbiter_contention_min_waiter_age_sec or 0.0)):
            score += 1
            reasons.append("waiter_age_exceeds_review_threshold")
        if progress.get("deadline_event"):
            score += 2
            reasons.append("deadline_event")
        if suspect.get("resource_suspect"):
            score += 2
            reasons.append("resource_suspect")
        if suspect.get("value_suspect"):
            score += 2
            reasons.append("value_suspect")
        if suspect.get("unknown_progress_suspect"):
            score += 1
            reasons.append("unknown_progress_suspect")
        if progress_signal in {"stalled", "degraded"}:
            score += 2
            reasons.append(f"progress_{progress_signal}")
        elif progress_signal == "unknown" and progress_confidence in {"low", "unknown", ""}:
            score += 1
            reasons.append("progress_unknown_low_confidence")
        if any(token in reason_code for token in ("stalled", "low_progress", "weak_metric", "no_viability", "invalid_metric")):
            score += 2
            reasons.append("reason_indicates_low_value_or_stall")
        elif "dataloader" in reason_code:
            score += 1
            reasons.append("dataloader_bottleneck_review")
        if proposal_type in {"resource_contention_review", "task_gpu_share_review", "shared_runtime_review"}:
            score += 1
            reasons.append("resource_contention_or_share_review")
        actionable = bool(suggested_actions & {
            "KILL_AND_REPLAN",
            "GRANT_SHARED_GPU_LEASE",
            "STOP_SECONDARY_SHARED_JOB",
            "REVOKE_SHARED_LEASE",
        })
        if actionable:
            score += 1
            reasons.append("actionable_resource_decision")
        if efficiency.get("review_required") and efficiency.get("kill_authority") == "review_only":
            score += 3
            reasons.append("bounded_resource_efficiency_review")
        if not waiters and progress_signal == "active" and progress_confidence == "high":
            score -= 1
            reasons.append("active_high_confidence_no_waiter")
        if score >= 4:
            level = "high"
        elif score >= 2:
            level = "medium"
        else:
            level = "low"
        if not reasons:
            reasons.append("no_opportunity_or_value_signal")
        return {
            "level": level,
            "score": score,
            "reasons": reasons[:8],
            "waiter_count": len(waiters),
            "max_queue_age_sec": max_queue_age,
            "progress_signal": progress_signal or "unknown",
            "progress_confidence": progress_confidence or "unknown",
            "source": "deterministic_budget_router",
        }

    def _attach_budget_priority(self, proposal: dict[str, Any]) -> dict[str, Any]:
        out = dict(proposal or {})
        if not isinstance(out.get("budget_priority"), dict):
            out["budget_priority"] = self._budget_priority_for_proposal(out)
        return out

    def _llm_budget_defer_reason(self, proposal: dict[str, Any], job_key: str, *, advisory: bool) -> str:
        if not job_key:
            return ""
        budget = proposal.get("budget_priority") if isinstance(proposal.get("budget_priority"), dict) else {}
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        efficiency = progress.get("resource_efficiency") if isinstance(progress.get("resource_efficiency"), dict) else {}
        if efficiency.get("review_required") and not efficiency.get("review_limit_reached"):
            return ""
        if self._proof_debt_review_has_priority(proposal, job_key):
            return ""
        metric_history_digest = self._proposal_metric_history_digest(proposal)
        last_metric_history = self._job_last_llm_metric_history_digest.get(job_key, "")
        if not advisory and metric_history_digest and metric_history_digest != last_metric_history:
            return ""
        if str(budget.get("level") or "").lower() != "low":
            return ""
        prior_calls = int(self._job_llm_call_count.get(job_key) or 0)
        prior_advisory = int(self._job_advisory_call_count.get(job_key) or 0)
        if advisory and prior_advisory <= 0:
            return ""
        if not advisory and prior_calls <= 0:
            return ""
        return "low_priority_unchanged_or_low_opportunity"

    @staticmethod
    def _value_review_proposal_type(proposal: dict[str, Any]) -> bool:
        proposal_type = str(proposal.get("proposal_type") or "").strip().lower()
        return proposal_type in {
            "kill_proposal",
            "periodic_efficiency_review",
            "quick_probe_review",
            "resource_contention_review",
        }

    @staticmethod
    def _proposal_boundary_kind(proposal: dict[str, Any]) -> str:
        boundary = proposal.get("resource_review_boundary")
        if not isinstance(boundary, dict):
            preview = proposal.get("decision_preview") if isinstance(proposal.get("decision_preview"), dict) else {}
            boundary = preview.get("resource_review_boundary") if isinstance(preview.get("resource_review_boundary"), dict) else {}
        return str(boundary.get("kind") or "").strip().lower()

    @staticmethod
    def _advisory_commitment_clear_on(advisory: dict[str, Any]) -> str:
        preference = str(advisory.get("preference") or "").strip().lower()
        if preference not in {"continue", "timebox_continue"}:
            return ""
        commitment = str(advisory.get("commitment") or "").strip()
        expected = str(advisory.get("expected_next_artifact") or "").strip()
        if not commitment and not expected:
            return ""
        text = f"{commitment} {expected}".lower()
        metric_tokens = (
            "metric",
            "score",
            "auc",
            "rmse",
            "rmsle",
            "map",
            "f1",
            "validation",
            "valid",
            "submission",
            "submit",
            "held-out",
            "heldout",
        )
        if any(token in text for token in metric_tokens):
            return "metric_update"
        artifact_tokens = ("artifact", "checkpoint", "submission", "log", "file", "path", "csv", "npy", "pth", "pkl")
        if expected or any(token in text for token in artifact_tokens):
            return "artifact_growth"
        return ""

    def _advisory_commitment_timebox_decision(self, proposal: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        if not self._value_review_proposal_type(proposal):
            return decision
        escalation = proposal.get("resource_budget_escalation") if isinstance(proposal.get("resource_budget_escalation"), dict) else {}
        if str(escalation.get("kind") or "").strip().lower() == "final_resource_review":
            return decision
        advisory = proposal.get("main_agent_advisory") if isinstance(proposal.get("main_agent_advisory"), dict) else {}
        clear_on = self._advisory_commitment_clear_on(advisory)
        if not clear_on:
            return decision
        job_key = self._proposal_job_id(proposal)
        current_state = self._review_states.get(job_key) if job_key else None
        if (
            current_state is not None
            and current_state.in_timebox
            and normalize_clear_on(current_state.timebox_success_condition) == clear_on
        ):
            out = dict(decision)
            original_reason = str(out.get("reason") or "").strip()
            out.update({
                "action": "OBSERVE_MORE",
                "canonical_outcome": RESOURCE_REVIEW_NO_ACTION,
                "confidence": str(out.get("confidence") or "medium").strip().lower() or "medium",
                "reason_code": "advisory_timebox_already_active",
                "reason": (
                    "A negotiated resource timebox is already active for the same clear condition; "
                    "do not restart the proof window before it clears or expires."
                    + (f" Original arbiter reason: {original_reason}" if original_reason else "")
                ),
                "advisory_commitment_timebox_active": True,
                "active_timebox_id": current_state.timebox_id,
                "active_timebox_clear_on": clear_on,
                "active_timebox_windows": int(current_state.timebox_windows or 0),
                "active_timebox_deadline_windows": int(current_state.timebox_deadline_windows or 0),
            })
            out.pop("clear_on", None)
            return out
        action = str(decision.get("action") or "").strip().upper()
        canonical = str(decision.get("canonical_outcome") or "").strip().upper()
        if action in TERMINATING_ACTIONS or canonical == RESOURCE_REVIEW_KILL:
            return decision
        confidence = str(advisory.get("confidence") or decision.get("confidence") or "medium").strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        original_reason = str(decision.get("reason") or "").strip()
        out = dict(decision)
        out.update({
            "action": "OBSERVE_MORE",
            "canonical_outcome": RESOURCE_REVIEW_TIMEBOX,
            "clear_on": clear_on,
            "confidence": confidence,
            "reason_code": "main_agent_advisory_commitment_timebox",
            "reason": (
                "Main-agent advisory requested continued execution with a measurable commitment; "
                "start a negotiated timebox instead of plain defer."
                + (f" Original arbiter reason: {original_reason}" if original_reason else "")
            ),
            "advisory_commitment_timebox": True,
            "advisory_commitment": str(advisory.get("commitment") or "")[:300],
            "advisory_expected_next_artifact": str(advisory.get("expected_next_artifact") or "")[:300],
        })
        return out

    def _final_resource_review_escalation(self, proposal: dict[str, Any], job_key: str) -> dict[str, Any]:
        if not (job_key and self._value_review_proposal_type(proposal)):
            return {}
        state = self._review_states.get(job_key)
        review_state = state.to_json() if state is not None else {}
        if not review_state:
            preview = proposal.get("decision_preview") if isinstance(proposal.get("decision_preview"), dict) else {}
            review_state = preview.get("resource_review_state") if isinstance(preview.get("resource_review_state"), dict) else {}
        failed_count = int(review_state.get("failed_proof_window_count") or 0)
        max_windows = max(1, int(self.review_config.max_proof_windows or 2))
        if failed_count < max_windows:
            return {}
        boundary_kind = self._proposal_boundary_kind(proposal)
        failure_recorded = bool(review_state.get("timebox_failure_recorded"))
        if boundary_kind != TIMEBOX_EXPIRED and not failure_recorded:
            return {}
        return {
            "kind": "final_resource_review",
            "advisory_status": "exhausted",
            "failed_proof_window_count": failed_count,
            "max_proof_windows": max_windows,
            "proof_window_index": int(review_state.get("proof_window_index") or 0),
            "proof_window_source": str(review_state.get("proof_window_source") or ""),
            "timebox_success_condition": str(review_state.get("timebox_success_condition") or ""),
            "reason": "proof windows exhausted; resource arbiter must make the final resource-budget decision without another main-agent advisory",
        }

    def _with_final_resource_review_escalation(self, proposal: dict[str, Any], escalation: dict[str, Any]) -> dict[str, Any]:
        if not escalation:
            return proposal
        out = dict(proposal or {})
        out["resource_budget_escalation"] = dict(escalation)
        out["main_agent_advisory_status"] = "exhausted"
        out.setdefault("suggested_actions", ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"])
        proposal_id = str(out.get("proposal_id") or "")
        if self.resource_runtime is not None and proposal_id:
            self.resource_runtime.update_active_resource_proposal(proposal_id, out)
            self.resource_runtime.record_resource_event(
                "resource_budget_escalation",
                payload={"proposal_id": proposal_id, **dict(escalation)},
                trace_id=f"trace_{proposal_id}",
                proposal_id=proposal_id,
                command_id=self._proposal_job_id(out),
                lease_id=self._proposal_job_id(out),
            )
        return out

    def _proof_window_override_raw_decision(self, proposal: dict[str, Any], job_key: str) -> dict[str, Any] | None:
        if not (job_key and self._value_review_proposal_type(proposal)):
            return None
        boundary_kind = self._proposal_boundary_kind(proposal)
        state = self._review_states.get(job_key)
        review_state = state.to_json() if state is not None else {}
        if not review_state:
            preview = proposal.get("decision_preview") if isinstance(proposal.get("decision_preview"), dict) else {}
            review_state = preview.get("resource_review_state") if isinstance(preview.get("resource_review_state"), dict) else {}
        failed_count = int(review_state.get("failed_proof_window_count") or 0)
        proof_source = str(review_state.get("proof_window_source") or "").strip().lower()
        failure_recorded = bool(review_state.get("timebox_failure_recorded"))
        max_windows = max(1, int(self.review_config.max_proof_windows or 2))
        if failed_count >= max_windows:
            return None
        if boundary_kind != TIMEBOX_EXPIRED or not failure_recorded or failed_count <= 0:
            return None
        clear_on = normalize_clear_on(review_state.get("timebox_success_condition")) or "metric_update"
        return {
            "outcome": RESOURCE_REVIEW_TIMEBOX,
            "reason_code": "proof_window_retry",
            "reason": (
                f"The command missed proof window {failed_count}/{max_windows}; "
                "start the next bounded proof window before final resource review."
            ),
            "confidence": "medium",
            "clear_on": clear_on,
            "proof_window_override": True,
            "proof_window_retry": True,
            "proof_window_source": "proof_retry",
            "failed_proof_window_count": failed_count,
            "proof_window_source_previous": proof_source,
            "max_proof_windows": max_windows,
        }

    def _proof_debt_review_has_priority(self, proposal: dict[str, Any], job_key: str) -> bool:
        if not (job_key and self._value_review_proposal_type(proposal)):
            return False
        state = self._review_states.get(job_key)
        if state is None or int(state.failed_proof_window_count or 0) <= 0:
            return False
        boundary = self._proposal_boundary_kind(proposal)
        return boundary in {PROGRESS_WINDOW, ROUTE_VALUE}

    def _record_llm_budget_deferred_event(self, proposal: dict[str, Any], reason: str, *, advisory: bool = False) -> None:
        if self.resource_runtime is None:
            return
        proposal_id = str(proposal.get("proposal_id") or "")
        job_key = self._proposal_job_id(proposal)
        budget = proposal.get("budget_priority") if isinstance(proposal.get("budget_priority"), dict) else {}
        self.resource_runtime.record_resource_event(
            "llm_budget_deferred",
            payload={
                "proposal_id": proposal_id,
                "job_id": job_key,
                "advisory": bool(advisory),
                "reason": reason or "llm_budget_deferred",
                "budget_priority": budget,
                "llm_call_count": int(self._job_llm_call_count.get(job_key) or 0),
                "advisory_call_count": int(self._job_advisory_call_count.get(job_key) or 0),
                "llm_token_count": int(self._job_llm_token_count.get(job_key) or 0),
            },
            trace_id=f"trace_{proposal_id}" if proposal_id else "",
            proposal_id=proposal_id,
            command_id=job_key,
            lease_id=job_key,
        )

    @staticmethod
    def _proposal_job_id(proposal: dict[str, Any], fallback: str | None = None) -> str:
        preview = proposal.get("decision_preview") if isinstance(proposal.get("decision_preview"), dict) else {}
        return str(proposal.get("command_id") or proposal.get("job_id") or preview.get("job_id") or fallback or "")

    @staticmethod
    def _estimate_llm_tokens(payload: dict[str, Any]) -> int:
        try:
            text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        except Exception:
            text = str(payload or {})
        return max(1, int(len(text) / 4) + 1)

    def _llm_budget_blocked(self, job_key: str, *, estimated_tokens: int = 0, advisory: bool = False) -> dict[str, Any]:
        if not job_key:
            return {"blocked": False}
        llm_calls = int(self._job_llm_call_count.get(job_key) or 0)
        advisory_calls = int(self._job_advisory_call_count.get(job_key) or 0)
        tokens = int(self._job_llm_token_count.get(job_key) or 0)
        if self.arbiter_job_llm_call_cap > 0 and llm_calls >= self.arbiter_job_llm_call_cap:
            return {"blocked": True, "reason": "resource_arbiter_job_llm_call_cap", "llm_call_count": llm_calls}
        if advisory and self.arbiter_job_advisory_call_cap > 0 and advisory_calls >= self.arbiter_job_advisory_call_cap:
            return {"blocked": True, "reason": "resource_arbiter_job_advisory_call_cap", "advisory_call_count": advisory_calls}
        if self.arbiter_job_token_cap > 0 and tokens + max(0, int(estimated_tokens or 0)) > self.arbiter_job_token_cap:
            return {"blocked": True, "reason": "resource_arbiter_job_token_cap", "llm_token_count": tokens}
        return {"blocked": False}

    @staticmethod
    def _proposal_metric_history_digest(proposal: dict[str, Any]) -> str:
        generation = proposal.get("state_generation") if isinstance(proposal.get("state_generation"), dict) else {}
        control_fields = generation.get("control_fields") if isinstance(generation.get("control_fields"), dict) else {}
        digest = str(control_fields.get("metric_history_digest") or "").strip()
        if digest:
            return digest
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        metric_history = str(progress.get("metric_history_text") or "").strip()
        if not metric_history:
            return ""
        return hashlib.sha1(metric_history.encode("utf-8", errors="replace")).hexdigest()[:16]

    def _note_llm_call(
        self,
        job_key: str,
        *,
        estimated_tokens: int = 0,
        advisory: bool = False,
        proposal: dict[str, Any] | None = None,
    ) -> None:
        if not job_key:
            return
        self._job_llm_call_count[job_key] = int(self._job_llm_call_count.get(job_key) or 0) + 1
        if advisory:
            self._job_advisory_call_count[job_key] = int(self._job_advisory_call_count.get(job_key) or 0) + 1
        self._job_llm_token_count[job_key] = int(self._job_llm_token_count.get(job_key) or 0) + max(0, int(estimated_tokens or 0))
        metric_history_digest = self._proposal_metric_history_digest(proposal or {})
        if metric_history_digest and not advisory:
            self._job_last_llm_metric_history_digest[job_key] = metric_history_digest

    def _record_llm_budget_event(self, proposal: dict[str, Any], block: dict[str, Any], *, advisory: bool = False) -> None:
        if self.resource_runtime is None:
            return
        proposal_id = str(proposal.get("proposal_id") or "")
        job_key = self._proposal_job_id(proposal)
        self.resource_runtime.record_resource_event(
            "llm_budget_exhausted",
            payload={
                "proposal_id": proposal_id,
                "job_id": job_key,
                "advisory": bool(advisory),
                "reason": block.get("reason") or "llm_budget_exhausted",
                "llm_call_count": int(self._job_llm_call_count.get(job_key) or 0),
                "advisory_call_count": int(self._job_advisory_call_count.get(job_key) or 0),
                "llm_token_count": int(self._job_llm_token_count.get(job_key) or 0),
                "llm_call_cap": self.arbiter_job_llm_call_cap,
                "advisory_call_cap": self.arbiter_job_advisory_call_cap,
                "token_cap": self.arbiter_job_token_cap,
            },
            trace_id=f"trace_{proposal_id}" if proposal_id else "",
            proposal_id=proposal_id,
            command_id=job_key,
            lease_id=job_key,
        )

    def _advisory_scope_key(self, proposal: dict[str, Any]) -> str:
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        metric = proposal.get("resource_metric_value") if isinstance(proposal.get("resource_metric_value"), dict) else {}
        return str(metric.get("metric_scope_key") or progress.get("metric_scope_key") or "")

    def _reusable_main_agent_advisory(
        self,
        proposal: dict[str, Any],
        *,
        job_key: str,
        now: float,
    ) -> dict[str, Any]:
        cached = self._last_main_agent_advisory.get(job_key)
        if not isinstance(cached, dict):
            return {}
        advisory = cached.get("advisory") if isinstance(cached.get("advisory"), dict) else {}
        preference = str(advisory.get("preference") or "").strip().lower()
        confidence = str(advisory.get("confidence") or "").strip().lower()
        stop_preferences = {"safe_to_stop", "kill_and_replan", "stop", "stop_and_replan", "replan"}
        if preference not in stop_preferences or confidence != "high":
            return {}
        ttl_sec = max(0.0, self._nested_float(advisory, "ttl_sec", 600.0))
        if ttl_sec <= 0.0 or now - float(cached.get("captured_at") or 0.0) > ttl_sec:
            return {}
        cached_scope = str(cached.get("metric_scope_key") or "")
        current_scope = self._advisory_scope_key(proposal)
        if cached_scope and current_scope and cached_scope != current_scope:
            return {}
        cached_metric_digest = str(cached.get("metric_history_digest") or "")
        current_metric_digest = self._proposal_metric_history_digest(proposal)
        if cached_metric_digest and current_metric_digest and cached_metric_digest != current_metric_digest:
            return {}
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        metric = proposal.get("resource_metric_value") if isinstance(proposal.get("resource_metric_value"), dict) else {}
        if progress.get("near_submission") or metric.get("useful") is True:
            return {}
        if proposal_advisory_fact_conflicts(proposal, advisory):
            return {}
        reused = dict(advisory)
        reused["source"] = "main_agent_advisory_cache"
        reused["reused"] = True
        reused["reuse_age_sec"] = max(0.0, now - float(cached.get("captured_at") or now))
        return reused

    async def _maybe_attach_main_agent_advisory(self, proposal: dict[str, Any], *, job_key: str) -> dict[str, Any]:
        if not (self.main_agent_advisory_enabled and self.main_agent_advisory_decider is not None):
            return proposal
        proposal = self._attach_budget_priority(proposal)
        if proposal.get("main_agent_advisory"):
            return proposal
        now = time.time()
        reused_advisory = self._reusable_main_agent_advisory(proposal, job_key=job_key, now=now)
        if reused_advisory:
            out = dict(proposal)
            out["main_agent_advisory"] = reused_advisory
            proposal_id = str(out.get("proposal_id") or "")
            if self.resource_runtime is not None:
                self.resource_runtime.update_active_resource_proposal(proposal_id, out)
                self.resource_runtime.record_resource_event(
                    "main_agent_advisory_reused",
                    payload={
                        "proposal_id": proposal_id,
                        "job_id": job_key,
                        "advisory": reused_advisory,
                    },
                    trace_id=f"trace_{proposal_id}" if proposal_id else "",
                    proposal_id=proposal_id,
                    command_id=job_key,
                    lease_id=job_key,
                )
            return out
        defer_reason = self._llm_budget_defer_reason(proposal, job_key, advisory=True)
        if defer_reason:
            self._record_llm_budget_deferred_event(proposal, defer_reason, advisory=True)
            return proposal
        last = float(self._last_advisory_emit.get(job_key) or 0.0)
        if job_key and last and now - last < self.main_agent_advisory_min_interval_sec:
            return proposal
        estimate = self._estimate_llm_tokens(proposal)
        block = self._llm_budget_blocked(job_key, estimated_tokens=estimate, advisory=True)
        if block.get("blocked"):
            self._record_llm_budget_event(proposal, block, advisory=True)
            return proposal
        self._last_advisory_emit[job_key] = now
        self._note_llm_call(job_key, estimated_tokens=estimate, advisory=True, proposal=proposal)
        advisory: dict[str, Any]
        try:
            maybe = self.main_agent_advisory_decider(dict(proposal))
            if inspect.isawaitable(maybe):
                maybe = await asyncio.wait_for(maybe, timeout=self.main_agent_advisory_timeout_sec)
            if isinstance(maybe, str):
                try:
                    parsed = json.loads(maybe)
                except json.JSONDecodeError:
                    parsed = {"preference": "unknown", "reason": maybe[:500], "confidence": "low"}
                advisory = parsed if isinstance(parsed, dict) else {"preference": "unknown", "reason": str(maybe)[:500], "confidence": "low"}
            elif isinstance(maybe, dict):
                advisory = dict(maybe)
            else:
                advisory = {"preference": "unknown", "reason": "empty advisory response", "confidence": "low"}
        except Exception as exc:
            advisory = {"preference": "unknown", "reason": f"main_agent_advisory_failed:{type(exc).__name__}", "confidence": "low"}
        audit_payload = advisory.get("_audit") if isinstance(advisory.get("_audit"), dict) else None
        audit_status = str((audit_payload or {}).get("status") or "").strip()
        clean = {
            "preference": str(advisory.get("preference") or "unknown")[:80],
            "confidence": str(advisory.get("confidence") or "low")[:40],
            "reason": str(advisory.get("reason") or "")[:600],
            "ttl_sec": self._nested_float(advisory, "ttl_sec", 600.0),
            "source": "main_agent_advisory",
            "advisory_status": str(advisory.get("advisory_status") or audit_status or "captured")[:80],
        }
        for optional_key in (
            "advisory_mode",
            "blocked_state",
            "memory_edit_applied",
            "memory_edit_removed_tokens",
            "advisory_tool_call_rejected",
            "commitment",
            "expected_next_artifact",
            "observed_phase",
            "observed_checkpoint",
            "observed_resource_state",
            "observed_metric_status",
        ):
            if optional_key in advisory:
                clean[optional_key] = advisory.get(optional_key)
        self._last_main_agent_advisory[job_key] = {
            "advisory": dict(clean),
            "captured_at": now,
            "metric_scope_key": self._advisory_scope_key(proposal),
            "metric_history_digest": self._proposal_metric_history_digest(proposal),
        }
        out = dict(proposal)
        out["main_agent_advisory"] = clean
        proposal_id = str(out.get("proposal_id") or "")
        if self.resource_runtime is not None:
            self.resource_runtime.update_active_resource_proposal(proposal_id, out)
            if audit_payload:
                self.resource_runtime.record_resource_event(
                    "resource_advisory_audit",
                    payload={**audit_payload, "proposal_id": proposal_id, "job_id": job_key},
                    trace_id=f"trace_{proposal_id}" if proposal_id else "",
                    proposal_id=proposal_id,
                    command_id=job_key,
                    lease_id=job_key,
                )
            self.resource_runtime.record_resource_event(
                "main_agent_advisory",
                payload={"proposal_id": proposal_id, "job_id": job_key, "advisory": clean, "decision_applied": False},
                trace_id=f"trace_{proposal_id}" if proposal_id else "",
                proposal_id=proposal_id,
                command_id=job_key,
                lease_id=job_key,
            )
        return out

    def _computed_review_timebox_sec(self, job_id: str, proposal: dict[str, Any], *, clear_on: str) -> float:
        state = self._review_states.get(job_id) or new_review_state(job_id)
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        budget = proposal.get("budget_context") if isinstance(proposal.get("budget_context"), dict) else {}
        contention = proposal.get("contention_context") if isinstance(proposal.get("contention_context"), dict) else {}
        remaining = self._float_or_none(budget.get("remaining_budget_sec"))
        if remaining is None:
            remaining = self._float_or_none(progress.get("deadline_remaining_sec")) or 0.0
        active_waiter_pressure = bool(
            contention.get("active_waiter_pressure")
            or int(contention.get("blocked_worker_count") or 0) > 0
            or bool(proposal.get("waiters"))
        )
        return compute_timebox_sec(
            state,
            clear_on=clear_on,
            heartbeat_sec=self.review_heartbeat_sec,
            fallback_next_review_sec=max(
                self.review_heartbeat_sec,
                float(self.review_config.progress_event_min_windows or 1) * self.review_heartbeat_sec,
            ),
            min_timebox_sec=float(self.review_config.min_timebox_sec or 60.0),
            max_timebox_sec=float(self.review_config.max_timebox_sec or 1800.0),
            remaining_budget_sec=float(remaining or 0.0),
            timebox_budget_fraction=float(self.review_config.timebox_budget_fraction or 0.10),
            active_waiter_pressure=active_waiter_pressure,
        )


    async def arbiter_decide(
        self,
        job_id: str | None,
        *,
        proposal: dict[str, Any] | None = None,
        decision_preview: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.arbiter_enabled or self.arbiter_mode == "off":
            return {"enabled": False, "action": "DENY_KILL", "terminate": False}
        prop = dict(proposal or {})
        if not prop:
            preview = decision_preview if isinstance(decision_preview, dict) else {}
            prop = self._proposal_from_job(
                job_id,
                expected_proposal_type=str(preview.get("proposal_type") or ""),
                expected_reason=str(preview.get("reason_code") or preview.get("reason") or ""),
            )
        if not prop:
            return {"enabled": False, "action": "OBSERVE_MORE", "terminate": False, "reason": "proposal_missing"}
        job_key = self._proposal_job_id(prop, fallback=job_id)
        prop = self._attach_budget_priority(prop)
        final_escalation = self._final_resource_review_escalation(prop, job_key)
        if final_escalation:
            prop = self._with_final_resource_review_escalation(prop, final_escalation)
        raw: dict[str, Any] = {} if final_escalation else (self._proof_window_override_raw_decision(prop, job_key) or {})
        source = "proof_window_override" if raw else "policy_fallback"
        if not raw and not final_escalation:
            prop = await self._maybe_attach_main_agent_advisory(prop, job_key=job_key)
            prop = self._attach_review_history(prop, now=time.time())
        elif final_escalation:
            prop = self._attach_review_history(prop, now=time.time())
        if not raw and self.arbiter_mode == "llm" and self.arbiter_decider is not None:
            estimate = self._estimate_llm_tokens(prop)
            budget_block = {"blocked": False} if final_escalation else self._llm_budget_blocked(job_key, estimated_tokens=estimate, advisory=False)
            if budget_block.get("blocked"):
                self._record_llm_budget_event(prop, budget_block, advisory=False)
                raw = {
                    "action": "OBSERVE_MORE",
                    "reason": f"llm_budget_exhausted:{budget_block.get('reason') or 'unknown'}",
                    "confidence": "low",
                    "observe_more_sec": 600.0,
                }
                source = "llm_budget_guard"
            else:
                defer_reason = "" if final_escalation else self._llm_budget_defer_reason(prop, job_key, advisory=False)
                if defer_reason:
                    self._record_llm_budget_deferred_event(prop, defer_reason, advisory=False)
                    raw = {
                        "action": "OBSERVE_MORE",
                        "reason": f"llm_budget_deferred:{defer_reason}",
                        "confidence": "low",
                        "observe_more_sec": 600.0,
                    }
                    source = "llm_budget_deferred"
                else:
                    self._note_llm_call(job_key, estimated_tokens=estimate, advisory=False, proposal=prop)
                    try:
                        if inspect.iscoroutinefunction(self.arbiter_decider):
                            maybe = await asyncio.wait_for(self.arbiter_decider(prop), timeout=self.arbiter_timeout_sec)
                        else:
                            maybe = await asyncio.wait_for(asyncio.to_thread(self.arbiter_decider, prop), timeout=self.arbiter_timeout_sec)
                            if inspect.isawaitable(maybe):
                                maybe = await asyncio.wait_for(maybe, timeout=self.arbiter_timeout_sec)
                        if isinstance(maybe, str):
                            raw = parse_arbiter_decision_text(maybe)
                        elif isinstance(maybe, dict):
                            raw = dict(maybe)
                        source = "resource_arbiter_llm"
                    except asyncio.TimeoutError:
                        if self.resource_runtime is not None:
                            proposal_id = str(prop.get("proposal_id") or "")
                            self.resource_runtime.record_resource_event(
                                "resource_arbiter_timeout",
                                payload={
                                    "proposal_id": proposal_id,
                                    "job_id": job_key,
                                    "timeout_sec": self.arbiter_timeout_sec,
                                    "fallback_action": "OBSERVE_MORE",
                                },
                                trace_id=f"trace_{proposal_id}" if proposal_id else "",
                                proposal_id=proposal_id,
                                command_id=job_key,
                                lease_id=job_key,
                            )
                        raw = {"action": "OBSERVE_MORE", "reason": "arbiter_llm_timeout", "confidence": "low", "observe_more_sec": 300.0}
                        source = "llm_timeout_fallback"
                    except Exception as exc:
                        raw = {"action": "OBSERVE_MORE", "reason": f"arbiter_llm_failed:{type(exc).__name__}", "confidence": "low"}
                        source = "llm_error_fallback"
        if not raw:
            raw = fallback_policy_decision(prop)
            source = "policy_fallback"
        normalized = normalize_arbiter_decision(raw, proposal=prop, source=source)
        for passthrough_key in ("proof_window_source", "proof_window_retry", "proof_window_override"):
            if passthrough_key in raw and passthrough_key not in normalized:
                normalized[passthrough_key] = raw.get(passthrough_key)
        normalized = enforce_repeated_stall_escalation(normalized, prop)
        normalized = enforce_arbiter_kill_gate(
            normalized,
            prop,
            min_windows=self.arbiter_min_progress_windows,
            require_high_confidence=self.arbiter_kill_requires_high_confidence,
        )
        normalized = enforce_proposal_action_allowlist(normalized, prop)
        normalized = apply_control_profile_to_decision_delay(normalized, profile=self.control_profile)
        normalized = self._advisory_commitment_timebox_decision(prop, normalized)
        action = str(normalized.get("action") or "OBSERVE_MORE").upper()
        if action == "KILL_AND_REPLAN":
            gate_facts = normalized.get("gate") if isinstance(normalized.get("gate"), dict) else {}
            cadence = prop.get("research_cadence") if isinstance(prop.get("research_cadence"), dict) else {}
            value_review_support = bool(
                gate_facts.get("advisory_support")
                or gate_facts.get("structured_opportunity_cost_support")
                or prop.get("structured_opportunity_cost_support")
                or cadence.get("violation")
            )
            normalized["kill_intent_snapshot"] = build_kill_intent_snapshot(
                prop.get("progress_snapshot") if isinstance(prop.get("progress_snapshot"), dict) else {},
                prop.get("resource_metric_value") if isinstance(prop.get("resource_metric_value"), dict) else {},
                kill_basis="value_stagnation" if value_review_support else "liveness_stall",
            )
        if action == "MARK_STALLED_NO_KILL" and job_id and job_id in self._jobs:
            job = self._jobs[job_id]
            job.stalled_mark_count += 1
            normalized["stalled_mark_count"] = job.stalled_mark_count
        outcome = normalize_value_review_outcome(
            action,
            prop,
            canonical_outcome=str(normalized.get("canonical_outcome") or ""),
            clear_on=str(normalized.get("clear_on") or ""),
        )
        execution_outcome = outcome.outcome
        normalized["raw_action"] = action
        normalized["execution_outcome"] = execution_outcome
        normalized["value_review_outcome"] = outcome.to_json()
        review_job_key = job_key or str(job_id or "")
        if review_job_key and review_job_key in self._review_states:
            current_state = self._review_states[review_job_key]
            if outcome.outcome == RESOURCE_REVIEW_KILL:
                self._review_states[review_job_key] = apply_kill(current_state)
            elif outcome.outcome == RESOURCE_REVIEW_TIMEBOX:
                clear_on = normalize_clear_on(outcome.clear_on)
                computed_timebox_sec = self._computed_review_timebox_sec(review_job_key, prop, clear_on=clear_on)
                normalized["computed_timebox_sec"] = computed_timebox_sec
                contention = prop.get("contention_context") if isinstance(prop.get("contention_context"), dict) else {}
                active_waiter_pressure = bool(
                    contention.get("active_waiter_pressure")
                    or int(contention.get("blocked_worker_count") or 0) > 0
                    or bool(prop.get("waiters"))
                )
                proof_source = "negotiated" if normalized.get("advisory_commitment_timebox") else "system_fixed"
                if str(normalized.get("proof_window_source") or "").strip():
                    proof_source = str(normalized.get("proof_window_source") or "").strip()
                self._review_states[review_job_key] = apply_timebox(
                    current_state,
                    timebox_sec=computed_timebox_sec,
                    heartbeat_sec=self.review_heartbeat_sec,
                    clear_on=clear_on,
                    active_waiter_pressure=active_waiter_pressure,
                    proof_window_source=proof_source,
                    counts_as_proof_failure=not active_waiter_pressure,
                )
            elif outcome.suppress_main_agent_feedback:
                if current_state.in_timebox:
                    normalized["active_timebox_preserved"] = True
                    normalized.setdefault("active_timebox_id", current_state.timebox_id)
                    normalized.setdefault("active_timebox_windows", int(current_state.timebox_windows or 0))
                    normalized.setdefault("active_timebox_deadline_windows", int(current_state.timebox_deadline_windows or 0))
                    self._review_states[review_job_key] = current_state
                else:
                    delay = normalized.get("observe_more_sec") if action in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL"} else normalized.get("ttl_sec")
                    contention = prop.get("contention_context") if isinstance(prop.get("contention_context"), dict) else {}
                    active_waiter_pressure = bool(
                        contention.get("active_waiter_pressure")
                        or int(contention.get("blocked_worker_count") or 0) > 0
                        or bool(prop.get("waiters"))
                        or current_state.active_waiter_pressure
                    )
                    boundary = prop.get("resource_review_boundary") if isinstance(prop.get("resource_review_boundary"), dict) else {}
                    boundary_kind = str(boundary.get("kind") or "").strip().lower()
                    reason_code = str(prop.get("reason_code") or normalized.get("reason_code") or "").strip().lower()
                    proof_window_reason = bool(
                        boundary_kind in {STALL, ROUTE_VALUE, TIMEBOX_EXPIRED}
                        or any(token in reason_code for token in ("low_progress", "stalled", "timebox_expired", "sm_route_value", "sm_timebox_expired", "sm_stall"))
                    )
                    self._review_states[review_job_key] = apply_no_action(
                        current_state,
                        observe_more_sec=float(delay or 600.0),
                        heartbeat_sec=self.review_heartbeat_sec,
                        success_condition="useful_progress",
                        counts_as_proof_failure=bool(proof_window_reason and not active_waiter_pressure),
                    )
        if self.resource_runtime is not None:
            proposal_id_for_event = str(normalized.get("proposal_id") or prop.get("proposal_id") or "")
            self.resource_runtime.record_resource_event(
                "resource_review_outcome",
                payload={
                    "job_id": review_job_key,
                    "proposal_id": proposal_id_for_event,
                    "action": execution_outcome,
                    "execution_outcome": execution_outcome,
                    "raw_action": action,
                    "canonical_outcome": str(normalized.get("canonical_outcome") or ""),
                    "reason_code": str(normalized.get("reason_code") or ""),
                    "reason": str(normalized.get("reason") or ""),
                    "computed_timebox_sec": normalized.get("computed_timebox_sec"),
                    "outcome": outcome.to_json(),
                    "feedback_suppressed": bool(outcome.suppress_main_agent_feedback),
                },
                trace_id=f"trace_{proposal_id_for_event}" if proposal_id_for_event else "",
                proposal_id=proposal_id_for_event,
                decision_id=str(normalized.get("decision_id") or ""),
                command_id=review_job_key,
                lease_id=review_job_key,
            )
        self._record_arbiter_decision_events(prop, normalized, decision_preview=decision_preview or {})
        feedback = "" if outcome.suppress_main_agent_feedback else main_agent_feedback(normalized, proposal=prop)
        return {
            "enabled": True,
            "action": action,
            "raw_action": action,
            "execution_outcome": execution_outcome,
            "terminate": execution_outcome == RESOURCE_REVIEW_KILL,
            "observe_more": False,
            "feedback": feedback,
            "suppress_main_agent_feedback": bool(outcome.suppress_main_agent_feedback),
            "value_review_outcome": outcome.outcome,
            "computed_timebox_sec": normalized.get("computed_timebox_sec"),
            "decision": normalized,
            "proposal_id": normalized.get("proposal_id"),
            "decision_id": normalized.get("decision_id"),
        }

    def revalidate_kill_intent(
        self,
        job_id: str | None,
        *,
        arbiter_decision: dict[str, Any] | None = None,
        elapsed_sec: float,
        stdout_age_sec: float,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        metric_history_text: str = "",
        metric_history_line_count: int = 0,
        saw_training_progress: bool = False,
        saw_final_score: bool = False,
        current_phase: str = "",
        process_tree_cpu: dict[str, Any] | None = None,
        invalid_metric_events: int = 0,
        zero_score_events: int = 0,
        last_invalid_metric_text: str = "",
        last_zero_score_text: str = "",
        terminal_signal_events: int = 0,
        terminal_signal_kind: str = "",
        last_terminal_signal_text: str = "",
        deadline_event: bool = False,
        deadline_remaining_sec: float = 0.0,
        finalization_reserve_sec: float = 0.0,
        **_: Any,
    ) -> dict[str, Any]:
        decision = dict(arbiter_decision or {})
        original = decision.get("kill_intent_snapshot")
        if not job_id or job_id not in self._jobs:
            return {"enabled": False, "allow_kill": True, "reason": "job_missing"}
        job = self._jobs[job_id]
        signal = self._intervention_signal(
            elapsed_sec=float(elapsed_sec or 0.0),
            stdout_age_sec=stdout_age_sec,
            stdout_lines=stdout_lines,
            stdout_bytes=stdout_bytes,
            metric_history_text=metric_history_text,
            metric_history_line_count=metric_history_line_count,
            saw_training_progress=saw_training_progress,
            saw_final_score=saw_final_score,
            current_phase=current_phase,
            process_tree_cpu=process_tree_cpu,
            invalid_metric_events=invalid_metric_events,
            zero_score_events=zero_score_events,
            last_invalid_metric_text=last_invalid_metric_text,
            last_zero_score_text=last_zero_score_text,
            terminal_signal_events=terminal_signal_events,
            terminal_signal_kind=terminal_signal_kind,
            last_terminal_signal_text=last_terminal_signal_text,
            deadline_event=deadline_event,
            deadline_remaining_sec=deadline_remaining_sec,
            finalization_reserve_sec=finalization_reserve_sec,
        )
        self._attach_resource_metric_value_assessment(job, signal)
        self._update_job_progress_signal(job, signal, elapsed_sec=float(elapsed_sec or 0.0))
        fresh_progress = self._progress_snapshot_for_job(job, signal, elapsed_sec=float(elapsed_sec or 0.0))
        fresh_metric = signal.get("resource_metric_value") if isinstance(signal.get("resource_metric_value"), dict) else {}
        original_facts = original if isinstance(original, dict) else {}
        fresh = build_kill_intent_snapshot(
            fresh_progress,
            fresh_metric,
            kill_basis=str(original_facts.get("kill_basis") or ""),
        )
        gate = decision.get("gate") if isinstance(decision.get("gate"), dict) else {}
        result = evaluate_kill_intent_revalidation(
            original if isinstance(original, dict) else {},
            fresh,
            hard_safety=str(gate.get("kill_class") or "") == "hard_safety",
        )
        result.update({"enabled": True, "fresh_snapshot": fresh})
        if not result.get("allow_kill"):
            state = self._review_states.get(job_id)
            if state is not None:
                scope_memory = dict(state.metric_scope_memory or {})
                scope_memory.pop(str(state.metric_scope_key or ""), None)
                self._review_states[job_id] = replace(
                    state,
                    job_state_bucket="RUNNING_HEALTHY",
                    no_useful_progress_windows=0,
                    timebox_id="",
                    timebox_windows=0,
                    timebox_deadline_windows=0,
                    timebox_success_condition="",
                    timebox_failure_recorded=False,
                    proof_window_index=0,
                    failed_proof_window_count=0,
                    proof_window_source="",
                    metric_scope_memory=scope_memory,
                    last_outcome="STALE_KILL_INTENT",
                )
            self._emit("resource_kill_intent_stale", job, status="denied", payload=result)
            if self.resource_runtime is not None:
                self.resource_runtime.record_resource_event(
                    "resource_kill_intent_stale",
                    payload={"job_id": job_id, **result},
                    command_id=job_id,
                    lease_id=job_id,
                )
        return result

    def _proposal_from_job(
        self,
        job_id: str | None,
        *,
        expected_proposal_type: str = "",
        expected_reason: str = "",
    ) -> dict[str, Any] | None:
        if not job_id or self.resource_runtime is None:
            return None
        try:
            state = self.resource_runtime.unified_store._read_state_unlocked()
        except Exception:
            return None
        active = state.get("active_proposals") if isinstance(state, dict) else {}
        if not isinstance(active, dict):
            return None
        job_key = str(job_id)
        expected_type = str(expected_proposal_type or "").strip()
        expected_reason_text = str(expected_reason or "").strip()
        for proposal in active.values():
            if not isinstance(proposal, dict):
                continue
            preview = proposal.get("decision_preview") if isinstance(proposal.get("decision_preview"), dict) else {}
            command_id = str(proposal.get("command_id") or proposal.get("job_id") or preview.get("job_id") or "")
            if command_id != job_key and job_key not in str(proposal.get("proposal_id") or ""):
                continue
            if expected_type and str(proposal.get("proposal_type") or "") != expected_type:
                continue
            if expected_reason_text:
                proposal_reason = str(proposal.get("reason_code") or "").strip()
                preview_reason = str(preview.get("reason") or preview.get("reason_code") or "").strip()
                if expected_reason_text not in {proposal_reason, preview_reason}:
                    continue
            return dict(proposal)
        return None

    def _record_arbiter_decision_events(
        self,
        proposal: dict[str, Any],
        decision: dict[str, Any],
        *,
        decision_preview: dict[str, Any],
    ) -> None:
        if self.resource_runtime is None:
            return
        proposal_id = str(proposal.get("proposal_id") or decision.get("proposal_id") or "")
        decision_id = str(decision.get("decision_id") or "")
        trace_id = f"trace_{proposal_id}" if proposal_id else ""
        command_id = str(decision_preview.get("job_id") or proposal.get("command_id") or "")
        proposal_type = str(proposal.get("proposal_type") or "")
        action = str(decision.get("action") or "").upper()
        self._note_review_decision_history(proposal, decision)
        gate = decision.get("gate") if isinstance(decision.get("gate"), dict) else {}
        if action == "OBSERVE_MORE" and str(gate.get("blocked_reason") or "").startswith("stop_after_evidence_"):
            self.resource_runtime.record_resource_event(
                "stop_after_gate_downgrade",
                payload={
                    "proposal_id": proposal_id,
                    "proposal_type": proposal_type,
                    "blocked_reason": gate.get("blocked_reason"),
                    "stop_after_evidence_state": gate.get("stop_after_evidence_state"),
                    "observed_fields": gate.get("observed_fields") or [],
                },
                trace_id=trace_id,
                proposal_id=proposal_id,
                decision_id=decision_id,
                command_id=command_id,
                lease_id=command_id,
            )
        observe_more_key = f"{command_id}:{proposal_type}" if command_id and proposal_type else ""
        wait_actions = {
            "OBSERVE_MORE": ("observe_more_sec", 300.0),
            "MARK_STALLED_NO_KILL": ("observe_more_sec", 300.0),
            "CONTINUE": ("ttl_sec", 600.0),
            "DENY_KILL": ("ttl_sec", 600.0),
        }
        if observe_more_key and action in wait_actions and str(decision.get("canonical_outcome") or "").upper() != RESOURCE_REVIEW_TIMEBOX:
            delay_field, default_delay = wait_actions[action]
            try:
                delay_sec = float(decision.get(delay_field) or default_delay)
            except (TypeError, ValueError):
                delay_sec = default_delay
            delay_sec = max(1.0, delay_sec)
            next_review_at = time.time() + delay_sec
            self._review_observe_more_until[observe_more_key] = next_review_at
            self.resource_runtime.record_resource_event(
                "resource_review_observe_more_scheduled",
                payload={
                    "proposal_id": proposal_id,
                    "proposal_type": proposal_type,
                    "action": RESOURCE_REVIEW_NO_ACTION,
                    "raw_action": action,
                    "delay_field": delay_field,
                    "delay_sec": delay_sec,
                    "observe_more_sec": delay_sec if delay_field == "observe_more_sec" else None,
                    "ttl_sec": delay_sec if delay_field == "ttl_sec" else None,
                    "next_review_at": next_review_at,
                },
                trace_id=trace_id,
                proposal_id=proposal_id,
                decision_id=decision_id,
                command_id=command_id,
                lease_id=command_id,
            )
        elif observe_more_key:
            self._review_observe_more_until.pop(observe_more_key, None)
        payload = {
            "actor_id": "resource_arbiter",
            "cache_namespace": "resource_arbiter",
            "decision": dict(decision),
            "proposal_id": proposal_id,
            "decision_preview": dict(decision_preview or {}),
            "main_agent_feedback": main_agent_feedback(decision, proposal=proposal),
        }
        self.resource_runtime.record_resource_event(
            "decision",
            payload=payload,
            trace_id=trace_id,
            proposal_id=proposal_id,
            decision_id=decision_id,
            command_id=command_id,
            lease_id=command_id,
        )
        execution_outcome = str(decision.get("execution_outcome") or (RESOURCE_REVIEW_KILL if action in TERMINATING_ACTIONS else RESOURCE_REVIEW_NO_ACTION))
        execution_status = "pending_terminate" if execution_outcome == RESOURCE_REVIEW_KILL else "no_action"
        advisory = proposal.get("main_agent_advisory") if isinstance(proposal.get("main_agent_advisory"), dict) else {}
        self.resource_runtime.record_resource_event(
            "execution",
            payload={
                "action": execution_outcome,
                "execution_outcome": execution_outcome,
                "raw_action": action,
                "status": execution_status,
                "execution_status": "pending" if execution_outcome == RESOURCE_REVIEW_KILL else "no_action",
                "reason": decision.get("reason"),
                "kill_class": gate.get("kill_class"),
                "strict_gate_result": "allow" if bool(gate.get("allowed")) else "blocked",
                "strict_gate_reason": gate.get("blocked_reason"),
                "advisory_status": advisory.get("advisory_status"),
                "advisory_preference": advisory.get("preference"),
            },
            trace_id=trace_id,
            proposal_id=proposal_id,
            decision_id=decision_id,
            command_id=command_id,
            lease_id=command_id,
        )
        self.resource_runtime.record_resource_event(
            "feedback",
            payload={"message": main_agent_feedback(decision, proposal=proposal)},
            trace_id=trace_id,
            proposal_id=proposal_id,
            decision_id=decision_id,
            command_id=command_id,
            lease_id=command_id,
        )
        updated = dict(proposal)
        updated["last_decision"] = dict(decision)
        if action in TERMINATING_ACTIONS:
            updated["execution_status"] = "pending"
            self.resource_runtime.update_active_resource_proposal(proposal_id, updated)
        elif action == "DENY_KILL":
            self.resource_runtime.clear_active_resource_proposals_for_command(
                command_id,
                proposal_type=proposal_type,
                reason_code=str(proposal.get("reason_code") or ""),
            )
        else:
            self.resource_runtime.update_active_resource_proposal(proposal_id, updated)

    def _apply_kill_approval_gate(self, decision: dict[str, Any]) -> dict[str, Any]:
        if not decision.get("terminate"):
            return decision
        decision["would_terminate"] = True
        reason = str(decision.get("reason") or "resource_guard")
        if self.auto_kill_enabled:
            decision["recommended_action"] = "terminate"
            decision["requires_llm_decision"] = False
            decision["arbiter_enabled"] = False
            decision["feedback"] = self._append_resource_intervention_summary(
                str(decision.get("feedback") or ""),
                action="KILL_AND_REPLAN",
                reason=reason,
            )
            return decision
        decision["terminate"] = False
        decision["recommended_action"] = "stop_and_replan"
        decision["requires_llm_decision"] = True
        decision["arbiter_enabled"] = bool(self.arbiter_enabled and self.arbiter_mode != "off")
        decision["feedback"] = self._append_resource_intervention_summary(
            self._recommendation_feedback(
                reason=reason,
                feedback=str(decision.get("feedback") or ""),
            ),
            action="RECOMMEND_STOP_AND_REPLAN",
            reason=reason,
        )
        return decision

    @staticmethod
    def _recommendation_feedback(*, reason: str, feedback: str) -> str:
        text = str(feedback or "").strip()
        replacements = {
            "RESOURCE_FEEDBACK: terminated_idle_gpu_lease because": "RESOURCE_FEEDBACK: recommend_release_gpu_lease because",
            "RESOURCE_FEEDBACK: terminated_low_progress_command because": "RESOURCE_FEEDBACK: recommend_stop_command because",
            "RESOURCE_FEEDBACK: terminated_stalled_command because": "RESOURCE_FEEDBACK: recommend_stop_command because",
            "RESOURCE_FEEDBACK: terminated_low_signal_command because": "RESOURCE_FEEDBACK: recommend_stop_command because",
            "RESOURCE_FEEDBACK: terminated_dataloader_bottleneck because": "RESOURCE_FEEDBACK: recommend_stop_command because",
            "RESOURCE_FEEDBACK: terminated_invalid_training_metrics because": "RESOURCE_FEEDBACK: recommend_stop_command because",
        }
        for old, new in replacements.items():
            if text.startswith(old):
                text = new + text[len(old):]
                break
        if not text:
            text = f"RESOURCE_FEEDBACK: recommend_stop_command because {reason}."
        if text.endswith("."):
            text = text[:-1]
        return text + "; requires_llm_decision=true.\n"

    def _job_is_expensive_resource_hold(self, job: ResourceJob) -> bool:
        cls = str(job.resource_class or "")
        safe_after_deliverable = {
            RESOURCE_PURE_TT_CPU,
            RESOURCE_GPU_TT_LIGHT,
            RESOURCE_READONLY_CPU,
            RESOURCE_LIGHT_CPU,
        }
        if cls in safe_after_deliverable or bool(getattr(job.source_hint, "command_cpu_only", False)):
            return False
        if self._job_uses_expensive_gpu(job) or bool(job.gpu_ids):
            return True
        return cls in {RESOURCE_HEAVY_CPU_CANDIDATE, RESOURCE_UNKNOWN_EXEC, RESOURCE_UNKNOWN_GPU_EXEC}

    def _completed_deliverable_safe_command(self, job: ResourceJob) -> bool:
        command = str(job.command or "").strip()
        if not command:
            return True
        lowered = command.lower()
        compact = re.sub(r"\s+", " ", lowered).strip()

        scratch_target = re.search(r"(?:cat\s*>|tee(?:\s+-a)?)(?:\s+)(['\"]?)([^'\"\s<>|]+)\1", compact)
        if scratch_target:
            target = scratch_target.group(2).lstrip("./")
            suffix = Path(target).suffix.lower()
            if target.startswith(("tmp/", ".memory/")) and suffix in {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".log"}:
                return True

        if re.match(r"^(pwd|ls|head|tail|wc|du|df|stat|find|rg|grep|sed)(\s|$)", compact):
            return ">" not in compact or re.search(r">\s*(tmp/|\.memory/)", compact) is not None
        if re.match(r"^(mkdir\s+-p\s+)?(tmp|\.memory)(/|\s|$)", compact):
            return True

        if re.match(r"^(python3?|uv run python)\s+(-c|<<)", compact):
            heavy_markers = (
                " train.py",
                " solution.py",
                " predict.py",
                "submission.csv",
                "to_csv",
                ".fit(",
                "fit(",
                "predict_proba",
                "lightgbm",
                "xgboost",
                "catboost",
                "tensorflow",
                "torch",
                "keras",
                "pil import image",
            )
            return not any(marker in compact for marker in heavy_markers)

        return False

    def _completed_deliverable_preflight_gate(self, job: ResourceJob) -> dict[str, Any]:
        if not (self.deliverable_completion_guard_enabled and job.workspace_dir):
            return {"blocked": False}
        if self._completed_deliverable_safe_command(job):
            return {"blocked": False, "reason": "completed_deliverable_safe_command"}
        if not self._job_is_expensive_resource_hold(job):
            return {"blocked": False}
        try:
            state = deliverable_completion_state(
                job.workspace_dir,
                terminal_signal_seen=False,
                terminal_signal_kind="",
                settle_sec=self.deliverable_completion_settle_sec,
                candidate_artifact=job.candidate_artifact,
            )
        except Exception as exc:
            return {"blocked": False, "error": str(exc)}
        validity = classify_deliverable_validity(state)
        job.last_deliverable_completion_check = {"checked_at": time.time(), "state": dict(state)}
        job.deliverable_validity = validity
        if validity == "produced_invalid":
            reason = "invalid_deliverable_schema_preflight"
        else:
            return {"blocked": False, "deliverable_completion_state": dict(state), "deliverable_validity": validity}
        allowed_classes = [RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, RESOURCE_READONLY_CPU, RESOURCE_LIGHT_CPU]
        candidate_artifact = str(job.candidate_artifact or "").strip()
        unlock_condition = (
            "candidate_artifact_valid"
            if candidate_artifact and candidate_artifact != "submission.csv"
            else "submission_schema_valid"
            if validity == "produced_invalid"
            else "deliverable_state_changed"
        )
        schema_state = "invalid" if validity == "produced_invalid" else "valid"
        raw_feedback = self._append_resource_intervention_summary(
            self._resource_feedback_text(
                status="DENIED_REPLAN",
                reason=reason,
                scope="worker_deliverable",
                resource_mode="RED",
                blocked_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                allowed_classes=allowed_classes,
                unlock_condition=unlock_condition,
                blocked_until_unlock=True,
                schema_state=schema_state,
            ),
            action="DELIVERABLE_PREFLIGHT_BLOCK",
            reason=reason,
        )
        feedback_state = self._dedupe_resource_feedback_for_agent(
            job,
            feedback=raw_feedback,
            status="DENIED_REPLAN",
            reason=reason,
            scope="worker_deliverable",
            resource_mode="RED",
            blocked_class=job.resource_class,
            gpu_ids=job.gpu_ids,
            allowed_classes=allowed_classes,
            unlock_condition=unlock_condition,
            blocked_until_unlock=True,
            schema_state=schema_state,
            deliverable_validity=validity,
        )
        feedback = str(feedback_state.get("feedback") or "")
        self._remember_resource_feedback(
            job,
            status="DENIED_REPLAN",
            reason=reason,
            resource_mode="RED",
            blocked_class=job.resource_class,
            allowed_classes=allowed_classes,
        )
        self._emit(
            "resource_completed_deliverable_preflight_block",
            job,
            status="blocked",
            payload={
                "status": "DENIED_REPLAN",
                "reason": reason,
                "scope": "worker_deliverable",
                "resource_mode": "RED",
                "blocked_class": job.resource_class,
                "allowed_classes": allowed_classes,
                "deliverable_validity": validity,
                "deliverable_completion_state": dict(state),
                "candidate_artifact": candidate_artifact,
                "feedback_state_key": feedback_state.get("feedback_state_key"),
                "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
                "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
            },
        )
        return {
            "blocked": True,
            "status": "DENIED_REPLAN",
            "reason": reason,
            "feedback": feedback,
            "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
            "feedback_state_key": feedback_state.get("feedback_state_key"),
            "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
            "allowed_classes": allowed_classes,
            "deliverable_validity": validity,
            "deliverable_completion_state": dict(state),
            "candidate_artifact": candidate_artifact,
        }

    def _deliverable_completion_guard_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not (self.kill_enabled and self.deliverable_completion_guard_enabled and job.visible):
            return {"terminate": False}
        if elapsed < self.deliverable_completion_warmup_sec:
            return {"terminate": False}
        if not job.workspace_dir:
            return {"terminate": False}
        if not self._job_is_expensive_resource_hold(job):
            return {"terminate": False}
        now = time.time()
        last = float(job.last_deliverable_completion_check.get("checked_at") or 0.0)
        if last and self.deliverable_completion_scan_interval_sec > 0 and now - last < self.deliverable_completion_scan_interval_sec:
            return {
                "terminate": False,
                "deliverable_complete_samples": job.deliverable_complete_samples,
                "deliverable_completion_state": dict(job.last_deliverable_completion_check.get("state") or {}),
            }
        terminal_events = int(signal.get("terminal_signal_events") or 0)
        terminal_seen = terminal_events > 0 or bool(signal.get("saw_final_score"))
        state = deliverable_completion_state(
            job.workspace_dir,
            terminal_signal_seen=terminal_seen,
            terminal_signal_kind=str(signal.get("terminal_signal_kind") or ("final_score" if signal.get("saw_final_score") else "")),
            settle_sec=self.deliverable_completion_settle_sec,
            candidate_artifact=job.candidate_artifact,
        )
        job.last_deliverable_completion_check = {"checked_at": now, "state": dict(state)}
        job.deliverable_validity = classify_deliverable_validity(state)
        self._update_v7_route_profile(job, signal, elapsed_sec=elapsed)
        job.deliverable_complete_samples = job.deliverable_complete_samples + 1 if state.get("complete") else 0
        quiet_enough = float(signal.get("stdout_age_sec") or 0.0) >= self.deliverable_completion_quiet_sec
        is_submission = str(state.get("mode") or "") == "submission"
        if job.deliverable_validity == "produced_invalid":
            self._record_v7_resource_event(
                "invalid_deliverable_detected",
                job,
                {
                    "deliverable_completion_state": dict(state),
                    "route_viability": dict(job.route_viability_state or {}),
                    "feedback": "submission_schema_state=invalid",
                },
            )
            return {
                "terminate": False,
                "deliverable_complete_samples": job.deliverable_complete_samples,
                "deliverable_completion_state": dict(state),
                "deliverable_quiet_enough": quiet_enough,
                "feedback": self._append_resource_intervention_summary(
                    self._resource_feedback_text(
                        status="DENIED_REPLAN",
                        reason="invalid_deliverable_schema",
                        scope="worker_deliverable",
                        resource_mode="RED",
                        blocked_class=job.resource_class,
                        gpu_ids=job.gpu_ids,
                        unlock_condition="submission_schema_valid",
                        blocked_until_unlock=True,
                        schema_state="invalid",
                    ),
                    action="INVALID_DELIVERABLE_SCHEMA",
                    reason="produced_invalid",
                ),
            }
        if not state.get("complete") or (not is_submission and not quiet_enough):
            return {
                "terminate": False,
                "deliverable_complete_samples": job.deliverable_complete_samples,
                "deliverable_completion_state": dict(state),
                "deliverable_quiet_enough": quiet_enough,
            }
        return {
            "terminate": False,
            "deliverable_complete_samples": job.deliverable_complete_samples,
            "deliverable_completion_state": dict(state),
            "deliverable_quiet_enough": quiet_enough,
        }

    def _metric_health_guard_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not (self.kill_enabled and self.metric_health_guard_enabled and job.visible):
            return {"terminate": False}
        if elapsed < self.metric_health_warmup_sec or bool(signal.get("saw_final_score")):
            return {"terminate": False}
        if not (self._job_uses_expensive_gpu(job) or job.gpu_ids):
            return {"terminate": False}
        invalid_events = int(signal.get("invalid_metric_events") or 0)
        zero_score_events = int(signal.get("zero_score_events") or 0)
        if invalid_events >= self.metric_health_invalid_min_events:
            return {
                "terminate": True,
                "reason": "active_intervention:invalid_training_metrics_nan_inf",
                "invalid_metric_events": invalid_events,
                "zero_score_events": zero_score_events,
                "last_invalid_metric_text": str(signal.get("last_invalid_metric_text") or ""),
                "feedback": (
                    "RESOURCE_FEEDBACK: terminated_invalid_training_metrics because validation or training outputs contained NaN/Inf, so the run is not producing a usable model.\n"
                ),
            }
        if zero_score_events >= self.metric_health_zero_score_min_events and bool(signal.get("saw_training_progress")):
            return {
                "terminate": True,
                "reason": "active_intervention:invalid_training_metrics_zero_score",
                "invalid_metric_events": invalid_events,
                "zero_score_events": zero_score_events,
                "last_zero_score_text": str(signal.get("last_zero_score_text") or ""),
                "feedback": (
                    "RESOURCE_FEEDBACK: terminated_invalid_training_metrics because the validation score stayed at zero, so the run is not producing useful model signal.\n"
                ),
            }
        return {
            "terminate": False,
            "invalid_metric_events": invalid_events,
            "zero_score_events": zero_score_events,
        }

    def _record_idle_lease_release_candidate(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        pressure_reason: str,
        pressure: dict[str, Any],
        release_safety: dict[str, Any] | None = None,
        candidate_mode: str = "observe_only",
    ) -> None:
        now = time.time()
        cooldown = max(60.0, float(self.gpu_util_sample_interval_sec or 30.0))
        last = float(self._last_idle_lease_candidate_emit.get(job.job_id) or 0.0)
        if last and now - last < cooldown:
            return
        self._last_idle_lease_candidate_emit[job.job_id] = now
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        payload = {
            "action": "RELEASE_IDLE_LEASE",
            "mode": str(candidate_mode or "observe_only"),
            "would_release": True,
            "executed": False,
            "reason": "idle_gpu_lease_release_candidate",
            "elapsed_sec": elapsed,
            "idle_samples": job.idle_gpu_lease_samples,
            "min_samples": self.gpu_idle_lease_min_samples,
            "util_threshold_pct": self.gpu_idle_lease_util_pct,
            "mem_threshold_gb": self.gpu_idle_lease_mem_gb,
            "pressure_reason": pressure_reason,
            "pressure": dict(pressure or {}),
            "release_safety": dict(release_safety or {}),
            "last_gpu_util_sample": dict(job.last_gpu_util_sample or {}),
        }
        self._emit(
            "resource_idle_lease_release_candidate",
            job,
            status="observe_only",
            payload=payload,
        )
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_idle_lease_release_candidate",
                payload=payload,
                command_id=job.job_id,
                lease_id=job.job_id,
            )

    def _idle_lease_release_safety(self, job: ResourceJob, signal: dict[str, Any] | None = None) -> dict[str, Any]:
        hint = job.source_hint or ResourceSourceHint()
        sig = signal if isinstance(signal, dict) else {}
        phase = str(sig.get("current_phase") or "").strip().lower()
        current_mem_gb = self._current_gpu_mem_gb_for_job(job)
        idle_enough = job.idle_gpu_lease_samples >= max(1, int(self.gpu_idle_lease_min_samples or 1))
        mem_small = current_mem_gb <= max(0.0, float(self.gpu_idle_lease_mem_gb or 0.0))
        has_gpu_evidence = bool(getattr(hint, "command_gpu_compute_evidence", False)) or bool(getattr(hint, "has_gpu_evidence", False))
        known_training_phase = bool(
            any(token in phase for token in ("train", "epoch", "optimizer", "backward"))
            or (
                str(job.resource_class or "") in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN}
                and bool(getattr(hint, "has_train_evidence", False))
            )
        )
        possible_late_gpu_use = bool(known_training_phase or has_gpu_evidence)
        base = {
            "idle_samples": int(job.idle_gpu_lease_samples or 0),
            "min_samples": int(self.gpu_idle_lease_min_samples or 0),
            "current_mem_gb": current_mem_gb,
            "peak_mem_gb": float(job.gpu_mem_peak_gb or 0.0),
            "mem_threshold_gb": float(self.gpu_idle_lease_mem_gb or 0.0),
            "known_training_phase": known_training_phase,
            "has_gpu_evidence": has_gpu_evidence,
            "possible_late_gpu_use": possible_late_gpu_use,
            "phase": phase,
        }
        if not idle_enough:
            return {"safe": False, "reason": "release_window_not_satisfied", **base}
        if not mem_small:
            return {"safe": False, "reason": "gpu_mem_above_release_threshold", **base}
        if bool(getattr(hint, "command_cpu_only", False)):
            return {"safe": True, "reason": "command_cpu_only_hint", **base}
        inspected = int(getattr(hint, "source_files_inspected", 0) or 0)
        entrypoints = [str(x) for x in (getattr(hint, "entrypoints", []) or []) if str(x).strip()]
        if possible_late_gpu_use:
            return {
                "safe": True,
                "reason": "idle_small_mem_release_with_late_gpu_risk",
                "risk": "possible_late_gpu_use",
                "old_job_effect": "continues_with_existing_cuda_visible_devices",
                "new_job_policy": "admit_or_share_then_monitor_actual_contention",
                **base,
            }
        if inspected > 0:
            return {"safe": True, "reason": "source_inspected_without_gpu_evidence", "source_files_inspected": inspected, **base}
        if entrypoints:
            return {"safe": True, "reason": "entrypoint_not_inspected_but_idle_small_mem", "entrypoints": entrypoints[:5], **base}
        return {"safe": True, "reason": "command_without_gpu_evidence", **base}

    def release_idle_gpu_lease(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float = 0.0,
        reason: str = "idle_gpu_lease_release",
        feedback: str = "",
    ) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs or self.resource_runtime is None:
            return {"released": False, "reason": "missing_job_or_runtime"}
        job = self._jobs[job_id]
        safety = self._idle_lease_release_safety(job, job.last_signal)
        if not bool(safety.get("safe")):
            return {"released": False, "reason": "release_safety_failed", "release_safety": safety}
        result = self.resource_runtime.release_idle_lease(
            job_id=job.job_id,
            elapsed_sec=float(elapsed_sec or 0.0),
            reason=str(reason or "idle_gpu_lease_release"),
            resident_mem_gb=self._current_gpu_mem_gb_for_job(job),
        )
        if result.get("released"):
            job.idle_gpu_lease_samples = 0
            payload = {
                "job_id": job.job_id,
                "action": "RELEASE_IDLE_LEASE",
                "executed": True,
                "elapsed_sec": float(elapsed_sec or 0.0),
                "reason": str(reason or "idle_gpu_lease_release"),
                "feedback": str(feedback or ""),
                "release_safety": safety,
                "result": result,
            }
            self._emit("resource_idle_lease_released", job, status="released", payload=payload)
            self.resource_runtime.record_resource_event(
                "resource_guard_action",
                payload={"action": "release_idle_lease", **payload},
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return result

    def _idle_gpu_lease_guard_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not (self.kill_enabled and self.gpu_idle_lease_guard_enabled):
            return {"terminate": False}
        if not self.gpu_util_observer_enabled or self.resource_runtime is None:
            return {"terminate": False}
        if bool(signal.get("saw_final_score")):
            return {"terminate": False}
        idle_guard_classes = {
            RESOURCE_HEAVY_GPU_CANDIDATE,
            RESOURCE_HEAVY_GPU_TRAIN,
            RESOURCE_GPU_FEATURE_EXTRACT,
            RESOURCE_GPU_LIGHT_TRAIN,
            RESOURCE_GPU_TT_LIGHT,
            RESOURCE_UNKNOWN_GPU_EXEC,
        }
        if not job.gpu_ids:
            return {"terminate": False}
        if str(job.resource_class or "") not in idle_guard_classes:
            return {"terminate": False}
        if not self.resource_runtime.has_active_lease(job_id=job.job_id):
            return {"terminate": False}
        if elapsed < self.gpu_idle_lease_warmup_sec:
            return {"terminate": False, "idle_samples": job.idle_gpu_lease_samples}
        if job.idle_gpu_lease_samples < self.gpu_idle_lease_min_samples:
            return {"terminate": False, "idle_samples": job.idle_gpu_lease_samples}
        pressure = self._gpu_pressure_active_for_job(job)
        if not pressure.get("active"):
            return {
                "terminate": False,
                "idle_samples": job.idle_gpu_lease_samples,
                "pressure_reason": pressure.get("reason"),
            }
        pressure_reason = str(pressure.get("reason") or "gpu_pressure")
        release_safety = self._idle_lease_release_safety(job, signal)
        if self.gpu_idle_lease_action_mode == "observe":
            self._record_idle_lease_release_candidate(
                job,
                signal,
                pressure_reason=pressure_reason,
                pressure=pressure,
                release_safety=release_safety,
                candidate_mode="observe_only",
            )
            return {
                "terminate": False,
                "idle_samples": job.idle_gpu_lease_samples,
                "pressure_reason": pressure_reason,
                "release_candidate": True,
                "release_action": "RELEASE_IDLE_LEASE",
                "release_mode": "observe_only",
                "release_safety": release_safety,
            }
        feedback = (
            f"RESOURCE_FEEDBACK: release_idle_gpu_lease because the command held a GPU lease while sampled GPU util <= {self.gpu_idle_lease_util_pct:.1f}% and memory <= {self.gpu_idle_lease_mem_gb:.2f}GB for {job.idle_gpu_lease_samples} samples; pressure={pressure_reason}; elapsed_sec={elapsed:.1f}.\n"
        )
        if self.gpu_idle_lease_action_mode == "release" and bool(release_safety.get("safe")):
            return {
                "terminate": False,
                "release_idle_lease": True,
                "reason": f"active_intervention:idle_gpu_lease_under_pressure:{pressure_reason}",
                "idle_samples": job.idle_gpu_lease_samples,
                "pressure_reason": pressure_reason,
                "release_candidate": True,
                "release_action": "RELEASE_IDLE_LEASE",
                "release_mode": "release",
                "release_safety": release_safety,
                "feedback": feedback,
            }
        if self.gpu_idle_lease_action_mode == "release":
            self._record_idle_lease_release_candidate(
                job,
                signal,
                pressure_reason=pressure_reason,
                pressure=pressure,
                release_safety=release_safety,
                candidate_mode="blocked_by_safety",
            )
            return {
                "terminate": False,
                "idle_samples": job.idle_gpu_lease_samples,
                "pressure_reason": pressure_reason,
                "release_candidate": True,
                "release_action": "RELEASE_IDLE_LEASE",
                "release_mode": "blocked_by_safety",
                "release_safety": release_safety,
            }
        return {
            "terminate": True,
            "reason": f"active_intervention:idle_gpu_lease_under_pressure:{pressure_reason}",
            "idle_samples": job.idle_gpu_lease_samples,
            "pressure_reason": pressure_reason,
            "release_candidate": True,
            "release_action": "RELEASE_IDLE_LEASE" if self.gpu_idle_lease_action_mode == "recommend" else "KILL_AND_REPLAN",
            "release_mode": self.gpu_idle_lease_action_mode,
            "release_safety": release_safety,
            "feedback": (
                f"RESOURCE_FEEDBACK: terminated_idle_gpu_lease because the command held a GPU lease while sampled GPU util <= {self.gpu_idle_lease_util_pct:.1f}% and memory <= {self.gpu_idle_lease_mem_gb:.2f}GB for {job.idle_gpu_lease_samples} samples; pressure={pressure_reason}; elapsed_sec={elapsed:.1f}.\n"
            ),
        }

    def _low_progress_guard_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not (self.kill_enabled and self.low_progress_enabled and job.visible):
            return {"terminate": False}
        if not self._job_uses_expensive_gpu(job):
            return {"terminate": False}
        if elapsed < self.low_progress_warmup_sec or bool(signal.get("saw_final_score")):
            return {"terminate": False}
        progress_age = self._elapsed_since_progress(job.last_progress, elapsed_sec=elapsed)
        artifact_age = self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed)
        no_progress_limit = max(0.0, float(self.low_progress_no_heartbeat_sec or 0.0))
        no_artifact_limit = max(0.0, float(self.low_progress_no_artifact_sec or 0.0))
        progress_stale = no_progress_limit > 0 and progress_age >= no_progress_limit
        artifact_stale = no_artifact_limit <= 0 or artifact_age >= no_artifact_limit
        if not (progress_stale and artifact_stale):
            return {
                "terminate": False,
                "progress_age_sec": progress_age,
                "artifact_age_sec": artifact_age,
            }
        return {
            "terminate": True,
            "reason": "active_intervention:low_progress_heartbeat_stalled",
            "progress_age_sec": progress_age,
            "artifact_age_sec": artifact_age,
            "feedback": (
                "RESOURCE_FEEDBACK: terminated_low_progress_command because no epoch/iteration/metric/artifact heartbeat arrived within the configured progress threshold.\n"
            ),
        }

    def _dataloader_bottleneck_guard_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not (self.kill_enabled and self.gpu_dataloader_bottleneck_guard_enabled and job.visible):
            return {"terminate": False}
        if not self._job_uses_expensive_gpu(job):
            return {"terminate": False}
        if self.resource_runtime is None or not self.resource_runtime.has_active_lease(job_id=job.job_id):
            return {"terminate": False}
        if elapsed < self.gpu_dataloader_bottleneck_warmup_sec or bool(signal.get("saw_final_score")):
            return {"terminate": False, "dataloader_low_compute_samples": job.dataloader_bottleneck_samples}
        if job.dataloader_bottleneck_samples < self.gpu_dataloader_bottleneck_min_samples:
            return {"terminate": False, "dataloader_low_compute_samples": job.dataloader_bottleneck_samples}
        cpu = signal.get("process_tree_cpu") if isinstance(signal.get("process_tree_cpu"), dict) else {}
        child_cpu = self._float_or_none(cpu.get("child_cpu_pct")) or 0.0
        busy_children = int(cpu.get("busy_child_count") or 0)
        if child_cpu < self.gpu_dataloader_bottleneck_child_cpu_pct:
            return {
                "terminate": False,
                "dataloader_low_compute_samples": job.dataloader_bottleneck_samples,
                "child_cpu_pct": child_cpu,
                "busy_child_count": busy_children,
            }
        if busy_children < self.gpu_dataloader_bottleneck_busy_children:
            return {
                "terminate": False,
                "dataloader_low_compute_samples": job.dataloader_bottleneck_samples,
                "child_cpu_pct": child_cpu,
                "busy_child_count": busy_children,
            }
        return {
            "terminate": True,
            "reason": "active_intervention:dataloader_bottleneck_low_gpu_high_cpu",
            "dataloader_low_compute_samples": job.dataloader_bottleneck_samples,
            "child_cpu_pct": child_cpu,
            "busy_child_count": busy_children,
            "feedback": (
                "RESOURCE_FEEDBACK: terminated_dataloader_bottleneck because the command held a GPU lease and loaded model memory, but GPU utilization stayed low while DataLoader/child CPU stayed high, so the GPU was waiting on input preprocessing.\n"
            ),
        }

    def stalled_guard_decision(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float,
        stdout_age_sec: float,
        process_tree_cpu: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs:
            return {"enabled": False, "terminate": False, "would_terminate": False}
        job = self._jobs[job_id]
        elapsed = float(elapsed_sec or 0.0)
        if elapsed >= self.min_register_sec:
            self._promote(job, reason="elapsed_threshold", elapsed_sec=elapsed)
        if self.review_state_enabled:
            return {
                "enabled": False,
                "terminate": False,
                "would_terminate": False,
                "reason": "state_machine_handles_stalled_stdout",
                "check_interval_sec": self.check_interval_sec,
            }
        terminate = bool(
            self.kill_enabled
            and job.visible
            and self.stalled_stdout_sec > 0
            and float(stdout_age_sec or 0.0) >= self.stalled_stdout_sec
        )
        decision = self._apply_kill_approval_gate({
            "enabled": job.visible,
            "terminate": terminate,
            "reason": "stalled_stdout" if terminate else "",
            "check_interval_sec": self.check_interval_sec,
            "feedback": "RESOURCE_FEEDBACK: terminated_stalled_command because no useful output heartbeat arrived within the configured stalled-output threshold.\n" if terminate else "",
        })
        stalled_signal = {
            "elapsed_sec": elapsed,
            "stdout_age_sec": float(stdout_age_sec or 0.0),
            "stdout_lines": 0,
            "stdout_bytes": 0,
            "current_phase": "",
            "process_tree_cpu": dict(process_tree_cpu or {}),
        }
        self._update_job_progress_signal(job, stalled_signal, elapsed_sec=elapsed)
        self._record_kill_proposal_event(
            job,
            decision,
            stalled_signal,
            source="stalled_guard",
        )
        return decision

    @staticmethod
    def _intervention_signal(
        *,
        elapsed_sec: float,
        stdout_age_sec: float,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        metric_history_text: str = "",
        metric_history_line_count: int = 0,
        saw_training_progress: bool = False,
        saw_final_score: bool = False,
        current_phase: str = "",
        process_tree_cpu: dict[str, Any] | None = None,
        invalid_metric_events: int = 0,
        zero_score_events: int = 0,
        last_invalid_metric_text: str = "",
        last_zero_score_text: str = "",
        terminal_signal_events: int = 0,
        terminal_signal_kind: str = "",
        last_terminal_signal_text: str = "",
        deadline_event: bool = False,
        deadline_remaining_sec: float = 0.0,
        finalization_reserve_sec: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "elapsed_sec": float(elapsed_sec or 0.0),
            "stdout_age_sec": float(stdout_age_sec or 0.0),
            "stdout_lines": int(stdout_lines or 0),
            "stdout_bytes": int(stdout_bytes or 0),
            "metric_history_text": str(metric_history_text or ""),
            "metric_history_line_count": int(metric_history_line_count or 0),
            "saw_training_progress": bool(saw_training_progress),
            "saw_final_score": bool(saw_final_score),
            "current_phase": str(current_phase or ""),
            "process_tree_cpu": dict(process_tree_cpu or {}),
            "invalid_metric_events": int(invalid_metric_events or 0),
            "zero_score_events": int(zero_score_events or 0),
            "last_invalid_metric_text": str(last_invalid_metric_text or ""),
            "last_zero_score_text": str(last_zero_score_text or ""),
            "terminal_signal_events": int(terminal_signal_events or 0),
            "terminal_signal_kind": str(terminal_signal_kind or ""),
            "last_terminal_signal_text": str(last_terminal_signal_text or ""),
            "deadline_event": bool(deadline_event),
            "deadline_remaining_sec": float(deadline_remaining_sec or 0.0),
            "finalization_reserve_sec": float(finalization_reserve_sec or 0.0),
        }

    def active_intervention_decision(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float,
        stdout_age_sec: float,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        metric_history_text: str = "",
        metric_history_line_count: int = 0,
        saw_training_progress: bool = False,
        saw_final_score: bool = False,
        current_phase: str = "",
        process_tree_cpu: dict[str, Any] | None = None,
        invalid_metric_events: int = 0,
        zero_score_events: int = 0,
        last_invalid_metric_text: str = "",
        last_zero_score_text: str = "",
        terminal_signal_events: int = 0,
        terminal_signal_kind: str = "",
        last_terminal_signal_text: str = "",
        deadline_event: bool = False,
        deadline_remaining_sec: float = 0.0,
        finalization_reserve_sec: float = 0.0,
        **_: Any,
    ) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs:
            return {"enabled": False}
        job = self._jobs[job_id]
        elapsed = float(elapsed_sec or 0.0)
        signal = self._intervention_signal(
            elapsed_sec=elapsed,
            stdout_age_sec=stdout_age_sec,
            stdout_lines=stdout_lines,
            stdout_bytes=stdout_bytes,
            metric_history_text=metric_history_text,
            metric_history_line_count=metric_history_line_count,
            saw_training_progress=saw_training_progress,
            saw_final_score=saw_final_score,
            current_phase=current_phase,
            process_tree_cpu=process_tree_cpu,
            invalid_metric_events=invalid_metric_events,
            zero_score_events=zero_score_events,
            last_invalid_metric_text=last_invalid_metric_text,
            last_zero_score_text=last_zero_score_text,
            terminal_signal_events=terminal_signal_events,
            terminal_signal_kind=terminal_signal_kind,
            last_terminal_signal_text=last_terminal_signal_text,
            deadline_event=deadline_event,
            deadline_remaining_sec=deadline_remaining_sec,
            finalization_reserve_sec=finalization_reserve_sec,
        )
        if elapsed >= self.min_register_sec:
            self._promote(job, reason="elapsed_threshold", elapsed_sec=elapsed)
        self._maybe_emit_gpu_util_sample(job, elapsed_sec=elapsed)
        self._attach_resource_metric_value_assessment(job, signal)
        self._update_research_route_metric(job, signal)
        self._update_job_progress_signal(job, signal, elapsed_sec=elapsed)
        job.last_signal = signal
        self._maybe_emit_monitor_agent_shadow(job, signal)
        shared_secondary_stop = self._shared_secondary_revoked_decision(job)
        if shared_secondary_stop.get("terminate"):
            return shared_secondary_stop
        shared_runtime = self._shared_runtime_review_decision(job, signal)
        if isinstance(shared_runtime, dict) and shared_runtime.get("action") == "STOP_SECONDARY_SHARED_JOB":
            return shared_runtime
        terminate = False
        reason = ""
        feedback = ""
        deliverable_guard = self._deliverable_completion_guard_decision(job, signal)
        metric_guard = self._metric_health_guard_decision(job, signal)
        if not terminate and metric_guard.get("terminate"):
            terminate = True
            reason = str(metric_guard.get("reason") or "active_intervention:invalid_training_metrics")
            feedback = str(metric_guard.get("feedback") or "")
        if not terminate and self.review_state_enabled:
            state_decision = self._state_machine_review_decision(job, signal, elapsed_sec=elapsed)
            if isinstance(state_decision, dict) and state_decision.get("enabled"):
                return state_decision
        if not terminate and not self.review_state_enabled:
            terminate = bool(
                self.kill_enabled
                and job.visible
                and self.stalled_stdout_sec > 0
                and signal["stdout_age_sec"] >= self.stalled_stdout_sec
                and not signal["saw_final_score"]
            )
            reason = "active_intervention:stalled_or_low_signal" if terminate else ""
            feedback = (
                "RESOURCE_FEEDBACK: terminated_low_signal_command because the long-running command stopped producing useful output.\n"
                if terminate
                else ""
            )
        idle_guard = self._idle_gpu_lease_guard_decision(job, signal)
        if not terminate and idle_guard.get("terminate"):
            terminate = True
            reason = str(idle_guard.get("reason") or "active_intervention:idle_gpu_lease_under_pressure")
            feedback = str(idle_guard.get("feedback") or "")
        dataloader_guard = self._dataloader_bottleneck_guard_decision(job, signal)
        if not terminate and dataloader_guard.get("terminate"):
            terminate = True
            reason = str(dataloader_guard.get("reason") or "active_intervention:dataloader_bottleneck_low_gpu_high_cpu")
            feedback = str(dataloader_guard.get("feedback") or "")
        progress_guard = self._low_progress_guard_decision(job, signal)
        if not terminate and progress_guard.get("terminate"):
            terminate = True
            reason = str(progress_guard.get("reason") or "active_intervention:low_progress_heartbeat_stalled")
            feedback = str(progress_guard.get("feedback") or "")
        if not terminate and signal.get("deadline_event") and job.visible and self._job_is_expensive_resource_hold(job):
            terminate = True
            reason = "active_intervention:deadline_finalization_reserve"
            finish_feasibility = self._progress_finish_feasibility(job, signal, elapsed_sec=elapsed)
            if finish_feasibility.get("finish_feasible") is False:
                feedback = (
                    "RESOURCE_FEEDBACK: active_but_unaffordable because the command is still producing progress, "
                    "but observed ETA exceeds remaining useful budget and no recoverable artifact/metric/submission is guaranteed; "
                    "prefer smaller scope, cached artifacts, blend/finalize, or stop and replan.\n"
                )
            else:
                feedback = (
                    "RESOURCE_FEEDBACK: recommend_stop_command because task finalization reserve has started; "
                    "arbiter should prefer stopping work that cannot finish before deadline based on progress and efficiency evidence.\n"
                )
        decision = self._apply_kill_approval_gate({
            "enabled": bool(
                job.visible
                or deliverable_guard.get("terminate")
                or metric_guard.get("terminate")
                or idle_guard.get("terminate")
                or idle_guard.get("release_idle_lease")
                or dataloader_guard.get("terminate")
                or progress_guard.get("terminate")
            ),
            "terminate": terminate,
            "reason": reason,
            "check_interval_sec": self.check_interval_sec,
            "feedback": feedback,
            "progress_age_sec": progress_guard.get("progress_age_sec"),
            "artifact_age_sec": progress_guard.get("artifact_age_sec"),
            "idle_gpu_lease_samples": idle_guard.get("idle_samples"),
            "idle_gpu_pressure_reason": idle_guard.get("pressure_reason"),
            "idle_gpu_release_candidate": idle_guard.get("release_candidate"),
            "idle_gpu_release_action": idle_guard.get("release_action"),
            "idle_gpu_release_mode": idle_guard.get("release_mode"),
            "idle_gpu_release_safety": idle_guard.get("release_safety"),
            "release_idle_lease": idle_guard.get("release_idle_lease"),
            "dataloader_low_compute_samples": dataloader_guard.get("dataloader_low_compute_samples"),
            "dataloader_child_cpu_pct": dataloader_guard.get("child_cpu_pct"),
            "dataloader_busy_child_count": dataloader_guard.get("busy_child_count"),
            "route_viability": dict(job.route_viability_state or {}),
            "active_lease_suspect": dict(job.active_lease_suspect_state or {}),
            "terminal_signal_events": signal.get("terminal_signal_events"),
            "terminal_signal_kind": signal.get("terminal_signal_kind"),
            "last_terminal_signal_text": signal.get("last_terminal_signal_text"),
            "deadline_event": signal.get("deadline_event"),
            "deadline_remaining_sec": signal.get("deadline_remaining_sec"),
            "finalization_reserve_sec": signal.get("finalization_reserve_sec"),
            "invalid_metric_events": metric_guard.get("invalid_metric_events"),
            "zero_score_events": metric_guard.get("zero_score_events"),
            "last_invalid_metric_text": metric_guard.get("last_invalid_metric_text"),
            "last_zero_score_text": metric_guard.get("last_zero_score_text"),
        })
        if decision.get("release_idle_lease") and decision.get("feedback"):
            decision["feedback"] = self._append_resource_intervention_summary(
                str(decision.get("feedback") or ""),
                action="RELEASE_IDLE_LEASE",
                reason=str(decision.get("reason") or "idle_gpu_lease"),
            )
        self._record_kill_proposal_event(job, decision, signal, source="active_intervention")
        decision.setdefault("terminate", False)
        decision.setdefault("would_terminate", False)
        if not decision.get("terminate") and not decision.get("would_terminate"):
            review = self._quick_probe_review_decision(job, signal)
            if isinstance(review, dict) and review.get("enabled"):
                return review
            review = self._contention_review_decision(job, signal)
            if isinstance(review, dict) and review.get("enabled"):
                return review
            review = self._research_cadence_review_decision(job, signal)
            if isinstance(review, dict) and review.get("enabled"):
                return review
            review = self._periodic_efficiency_review_decision(job, signal)
            if isinstance(review, dict) and review.get("enabled"):
                return review
            share_review = self._gpu_share_phase_a_decision(job, signal)
            if isinstance(share_review, dict):
                shared_grant = share_review.get("shared_lease_grant")
                if share_review.get("arbiter_review") or (isinstance(shared_grant, dict) and shared_grant.get("granted")):
                    return share_review
        return decision

    def post_arbiter_continue_review(
        self,
        job_id: str | None,
        *,
        arbiter_action: str = "",
        elapsed_sec: float,
        stdout_age_sec: float,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        metric_history_text: str = "",
        metric_history_line_count: int = 0,
        saw_training_progress: bool = False,
        saw_final_score: bool = False,
        current_phase: str = "",
        process_tree_cpu: dict[str, Any] | None = None,
        invalid_metric_events: int = 0,
        zero_score_events: int = 0,
        last_invalid_metric_text: str = "",
        last_zero_score_text: str = "",
        terminal_signal_events: int = 0,
        terminal_signal_kind: str = "",
        last_terminal_signal_text: str = "",
        deadline_event: bool = False,
        deadline_remaining_sec: float = 0.0,
        finalization_reserve_sec: float = 0.0,
        **_: Any,
    ) -> dict[str, Any]:
        action = str(arbiter_action or "").upper()
        if action not in {"CONTINUE", "DENY_KILL", "OBSERVE_MORE", "MARK_STALLED_NO_KILL"}:
            return {"enabled": False, "reason": "arbiter_action_not_share_safe", "action": action}
        if not job_id or job_id not in self._jobs:
            return {"enabled": False, "reason": "job_not_found", "action": action}
        job = self._jobs[job_id]
        elapsed = float(elapsed_sec or 0.0)
        signal = self._intervention_signal(
            elapsed_sec=elapsed,
            stdout_age_sec=stdout_age_sec,
            stdout_lines=stdout_lines,
            stdout_bytes=stdout_bytes,
            metric_history_text=metric_history_text,
            metric_history_line_count=metric_history_line_count,
            saw_training_progress=saw_training_progress,
            saw_final_score=saw_final_score,
            current_phase=current_phase,
            process_tree_cpu=process_tree_cpu,
            invalid_metric_events=invalid_metric_events,
            zero_score_events=zero_score_events,
            last_invalid_metric_text=last_invalid_metric_text,
            last_zero_score_text=last_zero_score_text,
            terminal_signal_events=terminal_signal_events,
            terminal_signal_kind=terminal_signal_kind,
            last_terminal_signal_text=last_terminal_signal_text,
            deadline_event=deadline_event,
            deadline_remaining_sec=deadline_remaining_sec,
            finalization_reserve_sec=finalization_reserve_sec,
        )
        self._maybe_emit_gpu_util_sample(job, elapsed_sec=elapsed)
        self._attach_resource_metric_value_assessment(job, signal)
        self._update_research_route_metric(job, signal)
        self._update_job_progress_signal(job, signal, elapsed_sec=elapsed)
        job.last_signal = signal
        share_safe = action in {"CONTINUE", "DENY_KILL"}
        out = self._gpu_share_phase_a_decision(
            job,
            signal,
            allow_grant=share_safe,
            allow_handoff=share_safe,
        )
        if isinstance(out, dict) and out.get("enabled"):
            out["post_arbiter_action"] = action
            if not share_safe:
                out["observe_only_after_arbiter"] = True
        return out


    def _magent_resource_mode_for_job(self, job: ResourceJob) -> str:
        if self.resource_runtime is None or not job.gpu_ids:
            return "GREEN"
        try:
            snapshot = self.resource_runtime.pressure_snapshot(gpu_ids=job.gpu_ids)
        except Exception:
            return "UNKNOWN"
        gpus = snapshot.get("gpus") if isinstance(snapshot.get("gpus"), dict) else {}
        modes = {str(row.get("mode") or "").upper() for row in gpus.values() if isinstance(row, dict)}
        if "RED" in modes:
            return "RED"
        if "YELLOW" in modes:
            return "YELLOW"
        if modes:
            return "GREEN"
        return "UNKNOWN"

    def _record_magent_train_observed_once(
        self,
        job_id: str,
        job: ResourceJob,
        *,
        elapsed_sec: float,
        parent_state: str,
    ) -> None:
        if not self.estra_magent.enabled or self.resource_runtime is None:
            return
        if job_id in self._magent_observed_jobs:
            return
        if float(elapsed_sec or 0.0) < self.estra_magent.min_parent_runtime_sec:
            return
        self._magent_observed_jobs.add(job_id)
        self.resource_runtime.record_resource_event(
            MAGENT_TRAIN_OBSERVED,
            payload=train_observed_payload(
                worker_id=self.worker_id,
                parent_job_id=job_id,
                parent_state=parent_state,
                elapsed_sec=float(elapsed_sec or 0.0),
                resource_class=job.resource_class,
                resource_mode=self._magent_resource_mode_for_job(job),
            ),
            command_id=job_id,
            lease_id=job_id,
        )

    def _record_magent_fork_considered_once(
        self,
        job_id: str,
        *,
        decision: str,
        reason: str,
        task_type: str = "submission_checker",
    ) -> None:
        if not self.estra_magent.enabled or self.resource_runtime is None:
            return
        if job_id in self._magent_considered_jobs:
            return
        self._magent_considered_jobs.add(job_id)
        self.resource_runtime.record_resource_event(
            MAGENT_FORK_CONSIDERED,
            payload=fork_considered_payload(
                worker_id=self.worker_id,
                parent_job_id=job_id,
                decision=decision,
                reason=reason,
                task_type=task_type,
                sidecar_mode=self.estra_magent.sidecar_mode,
            ),
            command_id=job_id,
            lease_id=job_id,
        )

    def maybe_run_sidecar_backfill(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float,
        parent_state: str = "healthy_running",
    ) -> dict[str, Any]:
        if self.resource_runtime is None:
            reason = "sidecar_disabled" if not self.sidecar_enabled and not self.estra_magent.enabled else "resource_runtime_missing"
            return {"started": False, "reason": reason}
        if not job_id or job_id not in self._jobs:
            return {"started": False, "reason": "job_missing"}
        job = self._jobs[job_id]
        elapsed = float(elapsed_sec or 0.0)
        self._record_magent_train_observed_once(job_id, job, elapsed_sec=elapsed, parent_state=parent_state)
        if not self.sidecar_enabled:
            if elapsed >= self.estra_magent.min_parent_runtime_sec:
                self._record_magent_fork_considered_once(
                    job_id,
                    decision="skip",
                    reason="sidecar_disabled",
                )
            return {"started": False, "reason": "sidecar_disabled"}
        if job_id in self._sidecar_jobs_started:
            return {"started": False, "reason": "already_started"}
        if elapsed < self.sidecar_min_parent_runtime_sec:
            return {"started": False, "reason": "parent_runtime_too_short"}
        if not job.workspace_dir:
            self._record_magent_fork_considered_once(
                job_id,
                decision="skip",
                reason="workspace_missing",
            )
            return {"started": False, "reason": "workspace_missing"}
        if self.estra_magent.enabled and not self.estra_magent.sidecar_allowed() and not self.sidecar_enabled:
            self._record_magent_fork_considered_once(
                job_id,
                decision="skip",
                reason="sidecar_mode_disabled",
            )
            return {"started": False, "reason": "sidecar_mode_disabled"}

        self._sidecar_jobs_started.add(job_id)
        self._record_magent_fork_considered_once(
            job_id,
            decision="fork",
            reason="long_train_with_submission_readiness_gap",
        )
        self.resource_runtime.record_resource_event(
            "sidecar_fork_considered",
            payload={"parent_job_id": job_id, "parent_state": parent_state, "elapsed_sec": elapsed},
            command_id=job_id,
            lease_id=job_id,
        )
        try:
            report = run_cpu_sidecar_backfill(
                resource_dir=self.resource_runtime.resource_dir,
                workspace_dir=job.workspace_dir,
                parent_job_id=job_id,
                parent_state=parent_state,
                elapsed_sec=elapsed,
                parent_worker_id=self.worker_id,
                task_type="submission_checker",
                budget_sec=self.estra_magent.budget_sec,
            )
        except Exception as exc:
            self.resource_runtime.record_resource_event(
                "sidecar_discarded",
                payload={"parent_job_id": job_id, "reason": type(exc).__name__},
                command_id=job_id,
                lease_id=job_id,
            )
            return {"started": False, "reason": type(exc).__name__}
        sidecar_id = str(report.get("sidecar_id") or "")
        self.resource_runtime.record_resource_event(
            "sidecar_forked",
            payload={"parent_job_id": job_id, "sidecar_id": sidecar_id, "report_path": report.get("report_path")},
            command_id=job_id,
            lease_id=job_id,
        )
        if self.estra_magent.enabled:
            self.resource_runtime.record_resource_event(
                MAGENT_SIDECAR_STARTED,
                payload=sidecar_started_payload(
                    sidecar_id=sidecar_id,
                    parent_worker_id=self.worker_id,
                    parent_job_id=job_id,
                    budget_sec=self.estra_magent.budget_sec,
                    task_type=str(report.get("task_type") or "submission_checker"),
                    workspace=f"sidecars/{sidecar_id}/workspace" if sidecar_id else "",
                ),
                command_id=job_id,
                lease_id=job_id,
            )
        join_packet = None
        if self.estra_magent.enabled:
            join_packet = write_join_packet(
                self.resource_runtime.resource_dir,
                report,
                parent_worker_id=self.worker_id,
                inject_parent=self.estra_magent.join_inject_parent,
                inject_estra=self.estra_magent.join_inject_estra,
                inject_resource_context=self.estra_magent.join_inject_resource_context,
            )
            if join_packet:
                inject = join_packet.get("inject") if isinstance(join_packet.get("inject"), dict) else {}
                inject_targets = [name for name, enabled in inject.items() if enabled]
                self.resource_runtime.record_resource_event(
                    MAGENT_JOIN_PACKET_READY,
                    payload=join_packet_ready_payload(
                        sidecar_id=sidecar_id,
                        parent_worker_id=self.worker_id,
                        quality_gate=str(join_packet.get("quality_gate") or ""),
                        inject_targets=inject_targets,
                        artifact_count=len(join_packet.get("artifact_refs") or []),
                    ),
                    command_id=job_id,
                    lease_id=job_id,
                )
        self.resource_runtime.record_resource_event(
            "sidecar_joined",
            payload={
                "parent_job_id": job_id,
                "sidecar_id": sidecar_id,
                "quality_gate": report.get("quality_gate"),
                "observations": report.get("observations"),
                "recommended_next_action": report.get("recommended_next_action"),
                "useful_artifacts": report.get("useful_artifacts"),
                "join_packet_path": (join_packet or {}).get("packet_path") if isinstance(join_packet, dict) else "",
            },
            command_id=job_id,
            lease_id=job_id,
        )
        return {"started": True, "report": report, "join_packet": join_packet}

    def checkpoint_to_submission_guard(
        self,
        job_id: str | None,
        *,
        gap: dict[str, Any] | None,
        elapsed_sec: float,
    ) -> str:
        if not self.checkpoint_submission_guard_enabled or self.resource_runtime is None or not gap:
            return ""
        command_id = str(job_id or "")
        payload = {"job_id": command_id, "elapsed_sec": float(elapsed_sec or 0.0), **dict(gap)}
        self.resource_runtime.record_resource_event(
            "checkpoint_to_submission_guard",
            payload=payload,
            command_id=command_id,
            lease_id=command_id,
        )
        artifact_path = str(payload.get("artifact_path") or payload.get("submission_path") or "submission.csv")
        reason = str(payload.get("reason") or "checkpoint_without_submission")
        if artifact_path != "submission.csv":
            return (
                f"RESOURCE_FEEDBACK: {reason} because model/checkpoint artifacts were updated "
                f"but `{artifact_path}` was not updated.\n"
            )
        return (
            "RESOURCE_FEEDBACK: checkpoint_without_submission because model/checkpoint "
            "artifacts were updated but root submission.csv was not updated.\n"
        )

    def agent_backoff_wake_snapshot(self, **_: Any) -> dict[str, Any]:
        if self.resource_runtime is None:
            return {"pressure_generation": 0, "available": False, "reason": "resource_runtime_disabled"}
        return {
            "pressure_generation": self.resource_runtime.pressure_generation(),
            "available": False,
            "reason": "snapshot",
        }

    def agent_backoff_wake_decision(
        self,
        *,
        start_pressure_generation: int = 0,
        target_resource_class: str = RESOURCE_HEAVY_GPU_TRAIN,
        gpu_ids: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if self.resource_runtime is None:
            return {"wake": False, "wake_reason": "resource_runtime_disabled"}
        current = self.resource_runtime.pressure_generation()
        if current <= int(start_pressure_generation or 0):
            return {"wake": False, "wake_reason": "no_resource_change", "pressure_generation": current}
        availability = self.resource_runtime.resource_available_for(
            resource_class=str(target_resource_class or RESOURCE_HEAVY_GPU_TRAIN),
            gpu_ids=[str(x) for x in (gpu_ids or []) if str(x).strip()],
        )
        return {
            "wake": bool(availability.get("available")),
            "wake_reason": "resource_available" if availability.get("available") else str(availability.get("reason") or "resource_changed"),
            "pressure_generation": current,
            "availability": availability,
        }

    def agent_backoff_wait_started(
        self,
        *,
        wait_id: str,
        planned_sleep_sec: float,
        command: str = "",
        reason: str = "",
        **_: Any,
    ) -> None:
        wid = str(wait_id or "").strip()
        if not wid:
            return
        now = time.time()
        command_digest = hashlib.sha1(str(command or "").encode(errors="replace")).hexdigest()[:16]
        planned = max(0.0, float(planned_sleep_sec or 0.0))
        self._agent_backoff_waits[wid] = {
            "started_at": now,
            "planned_sleep_sec": planned,
            "command_digest": command_digest,
            "reason": str(reason or "agent_sleep_command"),
        }
        self.state_machine.append_event(
            "agent_backoff_wait_started",
            task_type="agent_backoff_wait",
            task_id=f"agent_backoff_wait:{wid}",
            status="waiting",
            payload={
                "wait_id": wid,
                "planned_sleep_sec": planned,
                "command_digest": command_digest,
                "reason": str(reason or "agent_sleep_command"),
            },
        )

    def agent_backoff_wait_finished(
        self,
        *,
        wait_id: str,
        planned_sleep_sec: float = 0.0,
        elapsed_sec: float = 0.0,
        status: str = "finished",
        wake_reason: str = "timer_elapsed",
        returncode: int | None = None,
        reason: str = "",
        **_: Any,
    ) -> None:
        wid = str(wait_id or "").strip()
        if not wid:
            return
        meta = self._agent_backoff_waits.pop(wid, {})
        planned = max(0.0, float(planned_sleep_sec or meta.get("planned_sleep_sec") or 0.0))
        elapsed = float(elapsed_sec or 0.0)
        if elapsed <= 0 and meta.get("started_at"):
            elapsed = max(0.0, time.time() - float(meta.get("started_at") or 0.0))
        finished_at = time.time()
        self._last_agent_backoff_wait = {
            "wait_id": wid,
            "planned_sleep_sec": planned,
            "elapsed_sec": max(0.0, elapsed),
            "status": str(status or "finished"),
            "wake_reason": str(wake_reason or "timer_elapsed"),
            "returncode": returncode,
            "reason": str(reason or ""),
            "command_digest": str(meta.get("command_digest") or ""),
            "started_at": meta.get("started_at"),
            "finished_at": finished_at,
        }
        self.state_machine.append_event(
            "agent_backoff_wait_finished",
            task_type="agent_backoff_wait",
            task_id=f"agent_backoff_wait:{wid}",
            status=str(status or "finished"),
            payload={
                "wait_id": wid,
                "planned_sleep_sec": planned,
                "elapsed_sec": max(0.0, elapsed),
                "wake_reason": str(wake_reason or "timer_elapsed"),
                "returncode": returncode,
                "reason": str(reason or ""),
                "command_digest": str(meta.get("command_digest") or ""),
            },
        )

    def resource_guard_action(self, job_id: str | None, *, action: str, reason: str, elapsed_sec: float, **extra: Any) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        self._promote(job, reason="guard_action", elapsed_sec=float(elapsed_sec or 0.0))
        self._remember_resource_feedback(
            job,
            status="DENIED_REPLAN",
            reason=str(reason or "resource_guard_action"),
            resource_mode="RED",
            blocked_class=job.resource_class,
            allowed_classes=[RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, "readonly_cpu", "light_cpu"],
            eta_next_train_sec=420.0,
            cooldown_sec=120.0,
        )
        action_text = str(action or "")
        executed_actions = {
            "terminate",
            "terminate_by_arbiter",
            "stop_boundary_violation",
            "gpu_orphan_cleanup",
        }
        execution_outcome = RESOURCE_REVIEW_KILL if action_text in executed_actions else RESOURCE_REVIEW_NO_ACTION
        payload = {
            "action": execution_outcome,
            "raw_action": action,
            "execution_outcome": execution_outcome,
            "reason": reason,
            "status": "DENIED_REPLAN",
            "execution_status": "executed" if action_text in executed_actions else "denied",
            "resource_mode": "RED",
            "blocked_class": job.resource_class,
            "allowed_classes": [RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, "readonly_cpu", "light_cpu"],
            "elapsed_sec": float(elapsed_sec or 0.0),
            "executed": action_text in executed_actions,
            "last_progress": dict(job.last_progress or {}),
            "last_artifact_progress": dict(job.last_artifact_progress or {}),
            "last_signal": dict(job.last_signal or {}),
        }
        for key, value in extra.items():
            if key and value is not None:
                payload[str(key)] = value
        self._emit(
            "resource_guard_action",
            job,
            status=str(action or ""),
            payload=payload,
        )
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "execution",
                payload=payload,
                command_id=job.job_id,
                lease_id=job.job_id,
            )
            if action_text in {"terminate", "terminate_by_arbiter"}:
                self.resource_runtime.record_resource_event(
                    "resource_kill_executed",
                    payload=payload,
                    command_id=job.job_id,
                    lease_id=job.job_id,
                )
                self.resource_runtime.clear_active_resource_proposals_for_command(job.job_id)

    def resource_cleanup_heartbeat(
        self,
        job_id: str | None,
        *,
        action: str,
        reason: str,
        elapsed_sec: float,
        **extra: Any,
    ) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        payload = {
            "action": str(action or "gpu_cleanup_skipped"),
            "observation_status": "OBSERVED",
            "reason": str(reason or "workspace_gpu_cleanup"),
            "elapsed_sec": float(elapsed_sec or 0.0),
        }
        for key, value in extra.items():
            if key and value is not None:
                payload[str(key)] = value
        self._emit(
            "resource_cleanup_heartbeat",
            job,
            status="observed",
            payload=payload,
        )
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_cleanup_heartbeat",
                payload={
                    "job_id": job.job_id,
                    "command_digest": job.command_digest,
                    "resource_class": job.resource_class,
                    "gpu_ids": job.gpu_ids,
                    "cpu_set": job.cpu_set,
                    "timeout_sec": job.timeout_sec,
                    **payload,
                },
                command_id=job.job_id,
                lease_id=job.job_id,
            )

    def job_finished(
        self,
        job_id: str | None,
        *,
        status: str,
        elapsed_sec: float,
        returncode: int | None = None,
        reason: str | None = None,
        **_: Any,
    ) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs.pop(job_id)
        self._review_states.pop(job_id, None)
        if self.resource_runtime is not None:
            self.resource_runtime.clear_active_resource_proposals_for_command(job_id)
        job.exit_code_seen = returncode is not None
        job.terminal_signal_seen = True
        elapsed = float(elapsed_sec or 0.0)
        release: dict[str, Any] = {}
        if self.resource_runtime is not None:
            release = self.resource_runtime.release(
                job_id=job_id,
                elapsed_sec=elapsed,
                status=str(status or "finished"),
            )
            if release.get("released"):
                self._emit(
                    "resource_gpu_lease_released",
                    job,
                    status="released",
                    payload={
                        "elapsed_sec": elapsed,
                        "release_reason": str(status or "finished"),
                        "pressure_generation": release.get("pressure_generation"),
                        "lease": release.get("lease") or {},
                    },
                )
        runtime_pressure: dict[str, Any] = {}
        status_text = str(status or "")
        reason_text = str(reason or "")
        pressure_text = f"{status_text} {reason_text}".lower()
        pressure_reason = ""
        if status_text == "resource_guard_terminated":
            pressure_reason = reason_text or "guard_terminated"
        elif "out of memory" in pressure_text or "cuda oom" in pressure_text or "oom" in pressure_text:
            pressure_reason = reason_text or status_text or "oom"
        if pressure_reason and self.resource_runtime is not None and job.gpu_ids:
            runtime_pressure = self.resource_runtime.record_runtime_pressure(
                job_id=job_id,
                resource_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                elapsed_sec=elapsed,
                reason=pressure_reason,
                command_digest=job.command_digest,
            )
            self._emit(
                "resource_gpu_runtime_pressure",
                job,
                status="red",
                payload={
                    "elapsed_sec": elapsed,
                    "reason": pressure_reason,
                    "release_reason": status_text,
                    "pressure": runtime_pressure,
                },
            )
        if not job.visible and elapsed >= self.min_register_sec:
            self._promote(job, reason="elapsed_threshold_finish", elapsed_sec=elapsed)
        if not job.visible:
            return
        self._emit(
            "resource_job_finished",
            job,
            status=str(status or "finished"),
            payload={
                "elapsed_sec": elapsed,
                "returncode": returncode,
                "reason": reason or "",
                "filtered_signal": dict(job.last_signal),
            },
        )
