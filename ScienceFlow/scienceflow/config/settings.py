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

"""Configuration dataclasses and loader for scienceflow."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from scienceflow.config.resource_modes import (
    apply_lnr_resource_control_mode_mapping as _apply_lnr_resource_control_mode_mapping,
    expand_lnr_resource_control_mode_payload,
)

_DEFAULT_CFG = Path(__file__).parent / "default.yaml"
_CONFIG_INCLUDE_KEYS: tuple[str, ...] = ("include", "includes")

# env-var → stage-field mapping (CODE_MODEL → agent.code.model, etc.); importable from this same path by ensemble replay.
STAGE_MODEL_ENV_MAP = {
    "code": "CODE_MODEL",
    "feedback": "FEEDBACK_MODEL",
}

STAGE_MODELS_ENV_MAP = {
    "code": "CODE_MODELS",
    "feedback": "FEEDBACK_MODELS",
}

STAGE_ENDPOINT_ENV_MAP = {
    "code": {
        "api_key": "CODE_API_KEY",
        "api_keys": "CODE_API_KEYS",
        "base_url": "CODE_BASE_URL",
        "base_urls": "CODE_BASE_URLS",
    },
    "feedback": {
        "api_key": "FEEDBACK_API_KEY",
        "api_keys": "FEEDBACK_API_KEYS",
        "base_url": "FEEDBACK_BASE_URL",
        "base_urls": "FEEDBACK_BASE_URLS",
    },
}


@dataclass
class StageConfig:
    model: str = "gpt-4o"
    # Optional per-worker/per-endpoint model pool. LHR workers may pin to one
    # indexed model while preserving the scalar ``model`` fallback.
    models: list[str] = field(default_factory=list)
    temp: float = 0.7
    base_url: str = ""
    api_key: str = ""
    top_p: float = 0.95
    # Optional anti-repetition bias (OpenAI-compatible APIs).
    frequency_penalty: float | None = None
    max_tokens: int = 32768
    reasoning_effort: str | None = None
    api_keys: list[str] = field(default_factory=list)
    base_urls: list[str] = field(default_factory=list)
    api_routing_mode: str = "round_robin"
    api_sticky_id: str = ""
    api_sticky_primary_index: int | None = None
    api_rate_limit_cooldown_sec: float = 60.0
    api_connection_cooldown_sec: float = 15.0
    http_timeout: int = 300
    # Extra OpenAI client default_headers (e.g. gateway tokens); per code/feedback.
    headers: dict[str, str] = field(default_factory=dict)
    # When true, pass httpx.AsyncClient(proxy=..., verify=False, Proxy-Authorization from env).
    use_proxy: bool = False


@dataclass
class AgentConfig:
    code: StageConfig = field(default_factory=StageConfig)
    feedback: StageConfig = field(default_factory=StageConfig)
    # USD prices per 1M tokens, keyed by model name.
    # Keys accepted per model: input_usd_per_1m, cached_input_usd_per_1m, output_usd_per_1m.
    llm_prices: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ExecConfig:
    fast_debug_max_samples: int = 100
    use_filtered: bool = False
    cpu_list: str = ""
    gpu_list: str = ""


@dataclass
class InitWorkspaceConfig:
    workspace_path: str = ""


@dataclass
class LnrConfig:
    """Config for the REPL-native long-horizon solver."""

    wall_clock_budget_sec: int = 3600
    max_steps: int = 200
    num_workers: int = 1
    seed: int = 0
    omp_threads_cap: int = 8
    code_organization_hint: str = "beyond_mfiles"
    init_workspace: InitWorkspaceConfig = field(default_factory=InitWorkspaceConfig)
    workspace_git_enabled: bool = True
    workspace_git_track_globs: list[str] = field(default_factory=lambda: ["*.py", "*.md"])
    workspace_git_initial_commit: bool = True
    workspace_git_auto_checkpoint: bool = True
    workspace_git_auto_review: bool = False
    ledger_filename: str = ".run_results.md"
    stage_capture_enabled: bool = True
    stage_capture_max_count: int = 0
    stage_commit_llm_timeout_sec: float = 180.0
    stage_commit_min_seconds_between: float = 20.0
    stage_commit_require_metric: bool = True
    stage_commit_text_mode: bool = True
    stage_commit_output_format: str = "text"
    stage_commit_context_mode: str = "compact"
    stage_commit_tool_choice: str = "none"
    stage_commit_experiment_state_enabled: bool = False
    stage_commit_persist_to_memory: bool = False
    stage_commit_persist_agent_write_to_memory: bool = True
    stage_commit_persist_prompt_to_memory: bool = False
    expose_runtime_context_each_round: bool = False
    metric_validity_adjudicator_enabled: bool = True
    metric_validity_adjudicator_timeout_sec: float = 60.0
    clean_repl_mode: bool = True
    metric_validation_leakage_guard_enabled: bool = True
    lnr_skill_tool_enabled: bool = False
    lnr_skill_tool_mode: str = "category_only"
    lnr_skill_category_source: str = "tasks/ml/mlebench/competition_categories.json"
    lnr_skill_category_label_field: str = "category_label"
    lnr_skill_auto_read: bool = False
    lnr_skill_auto_read_max_chars: int = 6000
    lnr_skill_visible_max: int = 1
    lnr_skill_allow_generic_wildcard: bool = False
    snapshot_dirname: str = ".snapshots"
    archive_dirname: str = "snapshots/archives"
    workspace_snapshot_enabled: bool = True
    workspace_snapshot_verify_objects: bool = True
    estra_enabled: bool = True
    estra_trigger_stage_count: int = 2
    force_estra_after_stage_count: int = 0
    force_estra_target_stage: str = ""
    force_estra_capture_duplicate_submissions: bool = False
    estra_use_main_agent_context: bool = True
    estra_peer_evidence_enabled: bool = True
    estra_peer_evidence_max_chars: int = 1400
    estra_peer_evidence_min_delta_ratio: float = 0.0
    estra_backtrack_reflection_enabled: bool = True
    estra_backtrack_reflection_max_chars: int = 1000
    estra_reflection_prompt_enabled: bool = True
    tail_summary_max_chars: int = 1200
    estra_compact_enabled: bool = True
    estra_compact_max_chars: int = 1200
    state_packet_max_chars: int = 12000
    state_packet_stage_card_max_chars: int = 420
    state_packet_archived_branch_max_chars: int = 1500
    stage_memory_folding_enabled: bool = True
    stage_memory_context_budget_chars: int = 24_000
    stage_memory_rebuild_on_stale: bool = True
    compact_on_context_limit: bool = True
    context_limit_min_messages: int = 5000
    preserve_prefix_and_eda: bool = True
    protected_eda_mode: str = "facts"
    protected_eda_facts_max_chars: int = 6000
    protected_eda_summary_max_chars: int = 8000
    protected_eda_warn_chars: int = 50_000
    worker_peer_summary_enabled: bool = True
    worker_peer_summary_max_chars: int = 2200
    merge_enabled: bool = True
    merge_mode: str = "worker_reduce"
    merge_owner_worker: str = "W00"
    merge_dirname: str = "merge"
    global_merge_wall_clock_sec: float = 900.0
    merge_prediction_file_max_bytes: int = 536_870_912
    merge_prediction_total_max_bytes: int = 2_147_483_648
    merge_required_finals: int = 3
    merge_max_finals: int = 3
    worker_dirname: str = "workers"
    resource_control_mode: str = "resource_smart_llm"
    resource_monitor_enabled: bool = True
    resource_control_profile: str = "normal"
    resource_startup_policy: str = "trial_first"
    resource_trial_window_sec: float = 300.0
    resource_trial_hard_review_sec: float = 900.0
    resource_bash_monitor_all_enabled: bool = True
    resource_bash_hard_fuse_finalization_reserve_sec: float = 900.0
    resource_monitor_min_register_sec: float = 600.0
    resource_monitor_check_interval_sec: float = 10.0
    resource_monitor_stalled_stdout_sec: float = 600.0
    resource_monitor_kill_enabled: bool = True
    resource_monitor_kill_mode: str = "arbiter"
    resource_monitor_agent_mode: str = "off"
    resource_monitor_agent_min_interval_sec: float = 60.0
    resource_monitor_low_progress_enabled: bool = True
    resource_monitor_low_progress_warmup_sec: float = 1800.0
    resource_monitor_low_progress_no_heartbeat_sec: float = 1800.0
    resource_monitor_low_progress_no_artifact_sec: float = 1800.0
    resource_review_state_enabled: bool = True
    resource_review_heartbeat_sec: float = 60.0
    resource_review_warmup_windows: int = 10
    resource_review_inactive_windows: int = 3
    resource_review_value_windows: int = 5
    resource_review_progress_event_min_windows: int = 5
    resource_review_timebox_windows: int = 10
    resource_review_max_proof_windows: int = 2
    resource_review_min_timebox_sec: float = 60.0
    resource_review_max_timebox_sec: float = 1800.0
    resource_review_timebox_budget_fraction: float = 0.10
    resource_progress_heartbeat_min_interval_sec: float = 30.0
    resource_artifact_heartbeat_scan_interval_sec: float = 30.0
    resource_artifact_recoverable_settle_sec: float = 5.0
    resource_recoverable_stop_enabled: bool = True
    resource_recoverable_stop_sigusr1_grace_sec: float = 60.0
    resource_recoverable_stop_marker_exit_grace_sec: float = 10.0
    resource_recoverable_stop_sigterm_grace_sec: float = 5.0
    resource_context_prompt_enabled: bool = True
    resource_runtime_enabled: bool = True
    resource_gpu_queue_enabled: bool = True
    resource_gpu_pool: list[str] = field(default_factory=list)
    resource_gpu_default_request: int = 1
    resource_gpu_max_request: int = 1
    resource_gpu_assignment: str = "lease"
    resource_gpu_queue_max_wait_sec: float = 1800.0
    resource_gpu_queue_heartbeat_sec: float = 15.0
    resource_gpu_max_heavy_per_gpu: int = 1
    resource_gpu_capacity_slots: float = 1.0
    resource_gpu_tt_max_per_gpu: int = 3
    resource_gpu_feature_max_per_gpu: int = 2
    resource_gpu_share_tt_with_train: bool = False
    resource_gpu_share_enabled: bool = True
    resource_gpu_share_phase: str = "observe"
    resource_gpu_share_policy_profile: str = "conservative"
    resource_gpu_share_memory_profile: str = "conservative"
    resource_gpu_share_cpu_policy: str = "conservative"
    resource_gpu_trial_admission_policy: str = "llm_grant"
    resource_gpu_lease_ttl_sec: float = 7200.0
    resource_gpu_duplicate_digest_cooldown_sec: float = 600.0
    resource_gpu_duplicate_digest_threshold: int = 2
    resource_gpu_admission_queue_enabled: bool = True
    resource_gpu_admission_waiter_ttl_sec: float = 900.0
    resource_admission_llm_enabled: bool = True
    resource_admission_llm_mode: str = "blocked_states"
    resource_admission_llm_timeout_sec: float = 60.0
    resource_stale_pressure_observe_first_enabled: bool = True
    resource_stale_pressure_observe_window_sec: float = 180.0
    resource_stale_pressure_observe_max_sec: float = 300.0
    resource_stale_pressure_observe_min_free_mem_gb: float = 8.0
    resource_stale_pressure_healthy_skip_llm: bool = True
    resource_gpu_pressure_yellow_hold_sec: float = 120.0
    resource_gpu_pressure_red_to_yellow_sec: float = 120.0
    resource_gpu_pressure_yellow_util_pct: float = 85.0
    resource_gpu_pressure_min_free_mem_gb: float = 8.0
    resource_gpu_pressure_yellow_free_mem_buffer_gb: float = 8.0
    resource_observation_enabled: bool = True
    resource_observation_window_sec: float = 30.0
    resource_observation_shadow_workspace_enabled: bool = True
    resource_gpu_source_hint_enabled: bool = True
    resource_gpu_source_hint_mode: str = "observe"
    resource_gpu_util_observer_enabled: bool = True
    resource_gpu_util_sample_interval_sec: float = 30.0
    resource_gpu_idle_lease_guard_enabled: bool = True
    resource_gpu_idle_lease_warmup_sec: float = 180.0
    resource_gpu_idle_lease_min_samples: int = 3
    resource_gpu_idle_lease_util_pct: float = 1.0
    resource_gpu_idle_lease_mem_gb: float = 1.0
    resource_gpu_idle_lease_require_pressure: bool = False
    resource_gpu_idle_lease_action_mode: str = "release"
    resource_idle_release_admission_mode: str = "strict_exclusive"
    resource_quick_probe_guard_enabled: bool = True
    resource_quick_probe_expected_runtime_sec: float = 300.0
    resource_quick_probe_hard_review_sec: float = 600.0
    resource_quick_probe_small_scope_threshold: int = 1000
    context_hygiene_compact_enabled: bool = True
    context_hygiene_max_stages_without_compact: int = 25
    context_hygiene_code_churn_stage_threshold: int = 10
    context_hygiene_large_file_repeat_threshold: int = 10
    context_hygiene_low_incremental_cache_rate: float = 0.80
    context_hygiene_low_cache_window: int = 5
    context_hygiene_min_tokens_since_compact: int = 250000
    context_hygiene_large_tool_output_chars: int = 12000
    context_hygiene_tool_output_min_interval_sec: float = 1800.0
    resource_gpu_dataloader_bottleneck_guard_enabled: bool = True
    resource_gpu_dataloader_bottleneck_warmup_sec: float = 600.0
    resource_gpu_dataloader_bottleneck_min_samples: int = 3
    resource_gpu_dataloader_bottleneck_util_pct: float = 15.0
    resource_gpu_dataloader_bottleneck_min_mem_gb: float = 2.0
    resource_gpu_dataloader_bottleneck_child_cpu_pct: float = 200.0
    resource_gpu_dataloader_bottleneck_busy_children: int = 2
    resource_deliverable_completion_guard_enabled: bool = True
    resource_deliverable_completion_warmup_sec: float = 120.0
    resource_deliverable_completion_settle_sec: float = 120.0
    resource_deliverable_completion_scan_interval_sec: float = 60.0
    resource_deliverable_completion_quiet_sec: float = 60.0
    resource_metric_health_guard_enabled: bool = True
    resource_metric_health_warmup_sec: float = 600.0
    resource_metric_health_invalid_min_events: int = 2
    resource_metric_health_zero_score_min_events: int = 2
    resource_arbiter_enabled: bool = True
    resource_arbiter_mode: str = "llm"
    resource_arbiter_timeout_sec: float = 90.0
    resource_arbiter_min_progress_windows: int = 2
    resource_arbiter_kill_requires_high_confidence: bool = True
    resource_main_agent_advisory_enabled: bool = True
    resource_main_agent_advisory_min_interval_sec: float = 600.0
    resource_main_agent_advisory_timeout_sec: float = 60.0
    resource_advisory_mode: str = "inline_memory_edit"
    resource_arbiter_contention_review_enabled: bool = False
    resource_arbiter_contention_min_runtime_sec: float = 900.0
    resource_arbiter_contention_min_waiter_age_sec: float = 300.0
    resource_arbiter_contention_min_interval_sec: float = 600.0
    resource_research_cadence_enabled: bool = True
    resource_research_cadence_observe_sec: float = 300.0
    resource_first_comparable_metric_budget_sec: float = 900.0
    resource_proven_route_metric_budget_sec: float = 1800.0
    resource_arbiter_periodic_review_enabled: bool = False
    resource_arbiter_periodic_min_runtime_sec: float = 1800.0
    resource_arbiter_periodic_min_interval_sec: float = 900.0
    resource_arbiter_proposal_coalesce_window_sec: float = 60.0
    resource_arbiter_job_llm_call_cap: int = 12
    resource_arbiter_job_advisory_call_cap: int = 3
    resource_arbiter_job_token_cap: int = 48000
    resource_sidecar_enabled: bool = False
    resource_sidecar_min_parent_runtime_sec: float = 900.0
    estra_magent_enabled: bool = False
    estra_magent_min_parent_runtime_sec: float = 300.0
    estra_magent_sidecar_enabled: bool = False
    estra_magent_sidecar_mode: str = "cpu_only"
    estra_magent_budget_sec: float = 900.0
    estra_magent_join_inject_parent: bool = True
    estra_magent_join_inject_estra: bool = True
    estra_magent_join_inject_resource_context: bool = True
    resource_checkpoint_submission_guard_enabled: bool = True
    resource_queue_timeout_hard_gate_enabled: bool = True
    resource_queue_timeout_block_train_after: int = 1
    resource_queue_timeout_tt_only_after: int = 2


@dataclass
class EvaluatorCandidateConfig:
    artifact: str = ""
    artifact_kind: str = ""
    commit_file: str = ""
    require_sha: bool = False
    scan_mode: str = "root_file"
    emit_missing_artifact_event: bool = False


@dataclass
class EvaluatorMetricConfig:
    name: str = "metric"
    lower_is_better: bool | None = None
    type: str = ""
    regex: str = ""
    json_path: str = ""
    selection_requires_direction: bool = True


@dataclass
class EvaluatorCommandConfig:
    evaluator_command: str = ""
    python_executable: str = ""
    environment_path: str = ""
    timeout_sec: float = 300.0
    cwd: str = "workspace"
    env: dict[str, str] = field(default_factory=dict)
    stdout_tail_chars: int = 4000
    stderr_tail_chars: int = 4000


@dataclass
class GateConfig:
    policy: str = "default"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorConfig:
    enabled: bool = True
    expose_wall_clock_remaining_sec: bool = False
    query_budget_scope: str = "task"
    stop_on_query_budget_exhausted: bool = False
    task_profile: str = "auto"
    backend: str = "auto"
    stage_source_mode: str = "primary"
    event_log: str = "evaluator_events.jsonl"
    cache_log: str = "evaluator_cache.jsonl"
    candidate: EvaluatorCandidateConfig = field(default_factory=EvaluatorCandidateConfig)
    metric: EvaluatorMetricConfig = field(default_factory=EvaluatorMetricConfig)
    command: EvaluatorCommandConfig = field(default_factory=EvaluatorCommandConfig)



@dataclass
class WorkspaceConfig:
    """Workspace, task identity, and global run metadata."""

    # Single-task output root. LNR workers use indexed execution roots below it
    # (workers/w00, workers/w01, ...); non-worker REPL runs use this root directly.
    task_workspace_root_dir: Path = field(default_factory=lambda: Path("."))
    # Read-only competition data root. LNR exposes this as a single flat workspace/dataset.
    # Legacy prepared/dataset_split roots are adapted by exposing their Deep child as flat data.
    input_data_dir: Path = field(default_factory=lambda: Path("."))
    # Execution directory: dataset/, submissions/, cwd for code runs.
    # Filled by prep_cfg from task_workspace_root_dir.
    workspace_dir: Path = field(default_factory=lambda: Path("./workspace"))
    log_dir: Path = field(default_factory=lambda: Path("."))
    submission_dir: Path = field(default_factory=lambda: Path("."))
    exp_id: str = ""
    mlebench_data_root_dir: str = ""
    # Optional task metadata override when leaderboard information is unavailable.
    custom_metric_name: str = ""
    custom_is_lower_better: bool | None = None
    enable_time_trace: bool = True
    max_messages: int = 100


@dataclass
class ReplConfig:
    """Generic REPL profile, data preview, and workspace-git controls."""

    # QA / generic ScienceAgent: max LLM↔tool rounds per single user request.
    qa_max_steps: int = 20
    # REPL: independent max LLM↔tool rounds per user request. 0 or negative
    # falls back to qa_max_steps for legacy manifests.
    repl_max_steps: int = 200
    # REPL profile is intentionally generic; task/domain rules should come from
    # the user query, task manifest, or skills rather than the base system prompt.
    repl_profile: str = "lite"
    # REPL tool preset. "bash_write" exposes bash/read/grep/glob/ls and performs
    # file changes through bash; "write_edit" keeps the legacy write/edit tools.
    repl_tool_preset: str = "bash_write"
    # Keep REPL system bytes stable by omitting dynamic workspace state and round
    # budget from the system prompt. Dynamic facts remain in tool results/history.
    repl_stable_system_prompt: bool = True
    # Pin a small environment-context user message at REPL session start.
    repl_pin_environment_context: bool = True
    # REPL-only optional user-side code organization preference.
    repl_code_organization_hint: str = ""
    # REPL-only workspace source control.
    repl_workspace_git_enabled: bool = True
    repl_workspace_git_track_globs: list[str] = field(default_factory=lambda: ["*.py", "*.md"])
    repl_workspace_git_initial_commit: bool = True
    repl_workspace_git_auto_checkpoint: bool = True
    repl_workspace_git_auto_review: bool = False
    repl_pin_task_description_when_auto_first_user: bool = False
    # REPL-only bash output caps.
    repl_bash_max_output_chars: int = 6000
    repl_bash_max_stream_line_chars: int = 1200
    repl_bash_observation_summary: bool = True
    # Data preview controls.
    preview_raw_files: bool = False
    data_preview_max_chars: int = 32000
    data_preview_max_items_per_dir: int = 40
    data_scan_walk_budget_dirs: int | None = 200_000
    data_scan_walk_budget_files: int | None = 500_000
    data_scan_probe_binary_dirs_budget: int = 8
    data_scan_preview_raw_dirs_budget: int = 12
    data_scan_meta_sample_max_bytes: int = 50_000
    data_scan_csv_max_rows_to_scan: int = 200_000
    data_preview_refresh_after_prep: bool = False


@dataclass
class ToolConfig:
    """ScienceAgent tool runtime, memory projection, and feedback controls."""

    # ScienceAgent: cap each execution feedback block (stdout/stderr); 0 = no cap.
    exec_feedback_max_chars: int = 5000
    # ScienceAgent: max total chars for user/assistant messages sent to LLM.
    sliding_window_budget_chars: int = 60000
    sliding_window_priority_enabled: bool = True
    mid_run_compact_enabled: bool = False
    round_budget_prompt_cap: int = 15
    # Tool runtime and sandbox.
    scienceflow_tools_sandbox: bool = True
    parallel_bash_enabled: bool = True
    parallel_llm_tool_calls: bool = True
    path_guard_extra_roots: list[Path] = field(default_factory=list)
    scienceflow_llm_stream_timeout_sec: int = 1800
    stream_repetition_detection: bool = True
    stream_repetition_window_chars: int = 4000
    stream_repetition_ngram_len: int = 150
    stream_repetition_max_repeats: int = 3
    stream_max_output_chars_soft: int = 30000
    stream_repetition_retry_max: int = 2
    llm_tool_stream_max_attempts: int = 5
    llm_tool_stream_retry_base_delay_sec: float = 1.75
    llm_tool_stream_retry_max_delay_sec: float = 20.0
    scienceflow_bash_timeout_sec: float = 14400.0
    scienceflow_bash_timeout_slow_sec: float = 14400.0
    scienceflow_stdout_max_chars: int = 32768
    # Interaction log and tool-memory projection.
    scienceflow_interaction_log_level: str = "minimal"
    scienceflow_interaction_log_full: bool = False
    scienceflow_interaction_log_color: bool = True
    scienceflow_interaction_log_llm_stream: bool = False
    scienceflow_bash_stream_to_interaction_log: bool = True
    bash_output_dedup_enabled: bool = True
    bash_output_dedup_min_repeat: int = 3
    bash_output_dedup_summary_prefix: str = "[log-dedup]"
    bash_output_dedup_apply_to_memory: bool = True
    tool_memory_compression: bool = True
    grep_max_results_lines: int = 50
    msg0_compress_body: bool = False
    bash_success_tail_lines: int = 50
    bash_success_tail_lines_solution: int = 8
    bash_success_tail_lines_test: int = 120
    bash_success_tail_lines_readonly: int = 30
    bash_success_tail_lines_install: int = 5
    read_success_max_lines: int = 200
    # Tool feedback snapshots and edit diagnostics.
    write_auto_snapshot_enabled: bool = True
    write_auto_snapshot_paths: list[str] = field(default_factory=list)
    write_auto_snapshot_code_extensions: list[str] = field(default_factory=lambda: [".py"])
    write_auto_snapshot_max_lines: int = 400
    write_auto_snapshot_max_chars: int = 8_000
    write_auto_snapshot_changed_context_lines: int = 10
    write_auto_snapshot_symbol_body_lines: int = 3
    read_overlap_guard_enabled: bool = True
    rotating_runtime_error_threshold: int = 3
    rotating_distinct_types_min: int = 2
    no_success_run_soft_threshold: int = 6
    no_success_run_hard_threshold: int = 12
    edit_short_old_str_chars: int = 40
    write_return_full_max_chars: int = 12000
    write_return_full_max_lines: int = 250
    write_return_head_tail_lines: int = 40
    edit_return_full_max_chars: int = 12000
    edit_return_full_max_lines: int = 250
    edit_return_head_tail_lines: int = 40
    edit_return_change_ctx_lines: int = 8
    edit_failure_top_k_candidates: int = 3
    edit_failure_diag_max_chars: int = 2000
    file_snapshot_latest_only: bool = True


@dataclass
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)
    exec: ExecConfig = field(default_factory=ExecConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    repl: ReplConfig = field(default_factory=ReplConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    lnr: LnrConfig = field(default_factory=LnrConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    # Optional task-profile defaults applied by CLI entrypoints before explicit
    # manifest lnr/agent patches. Keys are profile names such as "mlebench" or
    # "opt_solver"; supported child blocks are "lnr", "repl", and "evaluator".
    profile_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Set by :func:`load_cfg` to the resolved path of the YAML used as ``--config`` (or default.yaml).
    config_source_path: str = ""


_WORKSPACE_CONFIG_KEYS: tuple[str, ...] = (
    "task_workspace_root_dir",
    "input_data_dir",
    "workspace_dir",
    "log_dir",
    "submission_dir",
    "exp_id",
    "mlebench_data_root_dir",
    "custom_metric_name",
    "custom_is_lower_better",
    "enable_time_trace",
    "max_messages",
)

_REPL_CONFIG_KEYS: tuple[str, ...] = (
    "qa_max_steps",
    "repl_max_steps",
    "repl_profile",
    "repl_tool_preset",
    "repl_stable_system_prompt",
    "repl_pin_environment_context",
    "repl_code_organization_hint",
    "repl_workspace_git_enabled",
    "repl_workspace_git_track_globs",
    "repl_workspace_git_initial_commit",
    "repl_workspace_git_auto_checkpoint",
    "repl_workspace_git_auto_review",
    "repl_pin_task_description_when_auto_first_user",
    "repl_bash_max_output_chars",
    "repl_bash_max_stream_line_chars",
    "repl_bash_observation_summary",
    "preview_raw_files",
    "data_preview_max_chars",
    "data_preview_max_items_per_dir",
    "data_scan_walk_budget_dirs",
    "data_scan_walk_budget_files",
    "data_scan_probe_binary_dirs_budget",
    "data_scan_preview_raw_dirs_budget",
    "data_scan_meta_sample_max_bytes",
    "data_scan_csv_max_rows_to_scan",
    "data_preview_refresh_after_prep",
)

_TOOL_CONFIG_KEYS: tuple[str, ...] = (
    "exec_feedback_max_chars",
    "sliding_window_budget_chars",
    "sliding_window_priority_enabled",
    "mid_run_compact_enabled",
    "round_budget_prompt_cap",
    "scienceflow_tools_sandbox",
    "parallel_bash_enabled",
    "parallel_llm_tool_calls",
    "path_guard_extra_roots",
    "scienceflow_llm_stream_timeout_sec",
    "stream_repetition_detection",
    "stream_repetition_window_chars",
    "stream_repetition_ngram_len",
    "stream_repetition_max_repeats",
    "stream_max_output_chars_soft",
    "stream_repetition_retry_max",
    "llm_tool_stream_max_attempts",
    "llm_tool_stream_retry_base_delay_sec",
    "llm_tool_stream_retry_max_delay_sec",
    "scienceflow_bash_timeout_sec",
    "scienceflow_bash_timeout_slow_sec",
    "scienceflow_stdout_max_chars",
    "scienceflow_interaction_log_level",
    "scienceflow_interaction_log_full",
    "scienceflow_interaction_log_color",
    "scienceflow_interaction_log_llm_stream",
    "scienceflow_bash_stream_to_interaction_log",
    "bash_output_dedup_enabled",
    "bash_output_dedup_min_repeat",
    "bash_output_dedup_summary_prefix",
    "bash_output_dedup_apply_to_memory",
    "tool_memory_compression",
    "grep_max_results_lines",
    "msg0_compress_body",
    "bash_success_tail_lines",
    "bash_success_tail_lines_solution",
    "bash_success_tail_lines_test",
    "bash_success_tail_lines_readonly",
    "bash_success_tail_lines_install",
    "read_success_max_lines",
    "write_auto_snapshot_enabled",
    "write_auto_snapshot_paths",
    "write_auto_snapshot_code_extensions",
    "write_auto_snapshot_max_lines",
    "write_auto_snapshot_max_chars",
    "write_auto_snapshot_changed_context_lines",
    "write_auto_snapshot_symbol_body_lines",
    "read_overlap_guard_enabled",
    "rotating_runtime_error_threshold",
    "rotating_distinct_types_min",
    "no_success_run_soft_threshold",
    "no_success_run_hard_threshold",
    "edit_short_old_str_chars",
    "write_return_full_max_chars",
    "write_return_full_max_lines",
    "write_return_head_tail_lines",
    "edit_return_full_max_chars",
    "edit_return_full_max_lines",
    "edit_return_head_tail_lines",
    "edit_return_change_ctx_lines",
    "edit_failure_top_k_candidates",
    "edit_failure_diag_max_chars",
    "file_snapshot_latest_only",
)

def _config_alias(block_name: str, field_name: str) -> property:
    def _get(self: Config) -> Any:
        return getattr(getattr(self, block_name), field_name)

    def _set(self: Config, value: Any) -> None:
        setattr(getattr(self, block_name), field_name, value)

    return property(_get, _set)


for _alias_name in _WORKSPACE_CONFIG_KEYS:
    setattr(Config, _alias_name, _config_alias("workspace", _alias_name))
for _alias_name in _REPL_CONFIG_KEYS:
    setattr(Config, _alias_name, _config_alias("repl", _alias_name))
for _alias_name in _TOOL_CONFIG_KEYS:
    setattr(Config, _alias_name, _config_alias("tool", _alias_name))


def _move_legacy_top_level_keys(y: dict[str, Any], block_name: str, keys: tuple[str, ...]) -> None:
    """Move pre-nested top-level config keys into their canonical block."""
    block = y.get(block_name)
    if block is None:
        block = {}
    elif not isinstance(block, dict):
        return
    moved = False
    for key in keys:
        if key not in y:
            continue
        value = y.pop(key)
        if key not in block:
            block[key] = value
            moved = True
    if moved or block:
        y[block_name] = block


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

def _coerce_include_list(value: Any, *, source: Path) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        includes: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    f"Config include entries in {source} must be strings, "
                    f"got {type(item).__name__}"
                )
            includes.append(item)
        return includes
    raise TypeError(f"Config include in {source} must be a string or list of strings")


def _load_yaml_with_includes(path: Path, *, seen: set[Path] | None = None) -> Any:
    """Load YAML and recursively merge relative ``include`` files.

    Included files are merged first; keys in *path* override included defaults.
    The include key is loader-only and never reaches the structured config.
    """
    path = path.expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    active = seen or set()
    if resolved in active:
        chain = " -> ".join(str(p) for p in [*active, resolved])
        raise ValueError(f"Recursive config include detected: {chain}")
    active.add(resolved)

    raw = OmegaConf.load(path)
    container = OmegaConf.to_container(raw, resolve=False)
    if not isinstance(container, dict):
        active.remove(resolved)
        return raw

    include_value: Any = None
    for key in _CONFIG_INCLUDE_KEYS:
        if key in container:
            if include_value is not None:
                raise ValueError(f"Config {path} uses both include keys; use only `include`")
            include_value = container.pop(key)
    _normalize_yaml_dict(container)

    merged = OmegaConf.create({})
    for inc in _coerce_include_list(include_value, source=path):
        inc_path = Path(inc).expanduser()
        if not inc_path.is_absolute():
            inc_path = path.parent / inc_path
        merged = OmegaConf.merge(merged, _load_yaml_with_includes(inc_path, seen=active))

    merged = OmegaConf.merge(merged, OmegaConf.create(container))
    active.remove(resolved)
    return merged


def _normalize_yaml_dict(y: dict) -> None:
    """Map deprecated keys before merging into structured Config (in-place)."""
    _move_legacy_top_level_keys(y, "workspace", _WORKSPACE_CONFIG_KEYS)
    _move_legacy_top_level_keys(y, "repl", _REPL_CONFIG_KEYS)
    _move_legacy_top_level_keys(y, "tool", _TOOL_CONFIG_KEYS)

    legacy_ds = y.pop("deep_shallow", None)
    if isinstance(legacy_ds, dict):
        warnings.warn(
            "Config block `deep_shallow` was removed and is ignored.",
            DeprecationWarning,
            stacklevel=3,
        )
    legacy_data_split = y.pop("data_split", None)
    if isinstance(legacy_data_split, dict):
        warnings.warn(
            "Config block `data_split` was removed and is ignored; use `prep` "
            "with data-preparation skills instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    removed_top = [
        key
        for key in (
            "enable_deep_shallow",
            "deep_submission_dir",
            "deep_solution_dir",
            "prep_shallow_fraction",
            "hier_cold_start_step",
            "prep_light_validate_enabled",
            "prep_quality_retries",
            "prep_val_fraction",
            "prep_max_steps",
            "prep_split_only",
            "prep_python_timeout_sec",
            "prep_exec_feedback_max_chars",
            "prep_bash_timeout_sec",
        )
        if y.pop(key, None) is not None
    ]
    if removed_top:
        warnings.warn(
            "Config keys were removed and ignored: " + ", ".join(removed_top),
            DeprecationWarning,
            stacklevel=3,
        )
    ex = y.get("exec")
    if isinstance(ex, dict):
        removed = [
            key
            for key in ("exec_timeout", "draft_exec_timeout", "deep_exec_timeout")
            if ex.pop(key, None) is not None
        ]
        if removed:
            warnings.warn(
                "Config keys under `exec` were removed and ignored: " + ", ".join(removed),
                DeprecationWarning,
                stacklevel=3,
            )
    agent = y.get("agent")
    if isinstance(agent, dict) and "init_code" in agent:
        agent.pop("init_code", None)
        warnings.warn(
            "Config key `agent.init_code` is no longer supported; ignored.",
            DeprecationWarning,
            stacklevel=3,
        )
    if isinstance(agent, dict):
        if agent.pop("deep_feedback_to_shallow", None) is not None:
            warnings.warn(
                "Config key `agent.deep_feedback_to_shallow` was removed and ignored.",
                DeprecationWarning,
                stacklevel=3,
            )
        for _k in ("check", "prep"):
            if _k in agent:
                agent.pop(_k, None)
                warnings.warn(
                    f"Config key `agent.{_k}` was removed; use `agent.code` (drafts, data_prep) "
                    "and `agent.feedback` (safety summary, insights).",
                    DeprecationWarning,
                    stacklevel=3,
                )
    lnr = y.get("lnr")
    if isinstance(lnr, dict):
        _apply_lnr_resource_control_mode_mapping(lnr)
        removed_lnr = [
            key
            for key in (
                "deep_enabled",
                "ensemble_enabled",
                "stage_commit_max_turns",
                "lnr_skill_debug_log_enabled",
                "state_packet_working_note_max_chars",
                "estra_max_decisions",
                "compact_fallback_max_chars",
                "resource_main_agent_advisory_commit_transcript",
                "resource_main_agent_advisory_tool_calls",
            )
            if lnr.pop(key, None) is not None
        ]
        if removed_lnr:
            warnings.warn(
                "Config keys under `lnr` were removed and ignored: " + ", ".join(removed_lnr),
                DeprecationWarning,
                stacklevel=3,
            )
    if "data_dir" in y:
        workspace = y.setdefault("workspace", {})
        if isinstance(workspace, dict) and "input_data_dir" not in workspace:
            workspace["input_data_dir"] = y.pop("data_dir")
            warnings.warn(
                "Config key `data_dir` is deprecated; use `workspace.input_data_dir`.",
                DeprecationWarning,
                stacklevel=3,
            )
        else:
            y.pop("data_dir", None)
    if "optimizer_max_steps" in y:
        repl = y.setdefault("repl", {})
        value = y.pop("optimizer_max_steps", None)
        if isinstance(repl, dict) and "qa_max_steps" not in repl:
            repl["qa_max_steps"] = value
        warnings.warn(
            "Config key `optimizer_max_steps` is deprecated; use `repl.qa_max_steps` for ScienceAgent.",
            DeprecationWarning,
            stacklevel=3,
        )
    workspace = y.get("workspace")
    if isinstance(workspace, dict) and "workspace_dir" in workspace:
        if "task_workspace_root_dir" not in workspace:
            wd_raw = workspace.pop("workspace_dir")
            wd = Path(str(wd_raw)).expanduser()
            if wd.name == "workspace":
                parent = wd.parent
                workspace["task_workspace_root_dir"] = (
                    str(parent) if str(parent) not in ("", ".") else "."
                )
            else:
                workspace["task_workspace_root_dir"] = "."
                warnings.warn(
                    "Config key `workspace.workspace_dir` is deprecated; use "
                    "`workspace.task_workspace_root_dir` (task output root). "
                    "Could not migrate automatically (expected a path ending in "
                    f"'workspace'); using task_workspace_root_dir='.'. "
                    f"Ignored legacy workspace_dir={wd_raw!r}.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        else:
            workspace.pop("workspace_dir", None)
            warnings.warn(
                "Ignoring deprecated `workspace.workspace_dir` because "
                "`workspace.task_workspace_root_dir` is set.",
                DeprecationWarning,
                stacklevel=3,
            )


def load_cfg(path: Path | str | None = None, cli_args: bool = True) -> Config:
    """Load config from YAML → apply env → merge CLI overrides → return Config."""
    path = Path(path) if path else _DEFAULT_CFG
    raw = _load_yaml_with_includes(path)
    container = OmegaConf.to_container(raw, resolve=False)
    if isinstance(container, dict):
        _normalize_yaml_dict(container)
        raw = OmegaConf.create(container)
    schema = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(schema, raw)
    if cli_args:
        try:
            cli = OmegaConf.from_cli()
            valid = {f.name for f in Config.__dataclass_fields__.values()}
            raw = OmegaConf.to_container(cli)
            if isinstance(raw, dict):
                _normalize_yaml_dict(raw)
                overrides = {k: v for k, v in raw.items() if k in valid}
            else:
                overrides = {}
            if overrides:
                cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
        except Exception:
            pass
    OmegaConf.resolve(cfg)
    config: Config = OmegaConf.to_object(cfg)
    _apply_env(config)
    try:
        config.config_source_path = str(path.expanduser().resolve())
    except OSError:
        config.config_source_path = str(path)
    return config



_PARALLEL_MANIFEST_DEFAULTS_SKIP: frozenset[str] = frozenset(
    {
        "config",
        "workspace_base",
        "tasks",
        "defaults",
        "time_limit",
        "lnr",
        "agent",
        "gpu_list",
        "cpu_list",
    }
)


def merge_parallel_manifest_cfg_patch(
    defaults: dict[str, Any] | None, task: dict[str, Any] | None
) -> dict[str, Any]:
    """Top-level :class:`Config` keys from manifest ``defaults`` and ``task`` (task wins).

    Used by :func:`apply_repl_manifest_defaults` (``task`` empty) and by
    :class:`~scienceflow.core.parallel_runner.ParallelRunner` to pass overrides into
    prep/run subprocesses via ``SCIENCEFLOW_PARALLEL_MANIFEST_CFG_JSON``.
    """
    field_names = {f.name for f in fields(Config)}
    patch_dict: dict[str, Any] = {}
    for source in (defaults or {}, task or {}):
        for k, v in source.items():
            if k in _PARALLEL_MANIFEST_DEFAULTS_SKIP:
                continue
            patch_dict[k] = v
    if not patch_dict:
        return {}
    _normalize_yaml_dict(patch_dict)
    return {k: v for k, v in patch_dict.items() if k in field_names}


def apply_repl_manifest_defaults(
    cfg: Config,
    defaults: dict[str, Any] | None,
    task: dict[str, Any] | None = None,
) -> None:
    """Merge parallel-style manifest ``defaults:`` into *cfg* for ``repl --manifest``.

    Keys like ``repl.repl_max_steps`` / legacy ``repl_max_steps`` or
    ``tool.scienceflow_interaction_log_level`` / legacy ``scienceflow_interaction_log_level``
    live under manifest ``defaults`` or a single REPL task entry.

    Only :class:`Config` fields are merged after legacy top-level keys are moved
    into their canonical blocks; manifest-only keys (``config``,
    ``workspace_base``, ``time_limit``, nested blocks, …) are skipped.
    """
    patch_dict = merge_parallel_manifest_cfg_patch(defaults, task)
    if not patch_dict:
        return
    oc_cfg = OmegaConf.structured(cfg)
    oc_patch = OmegaConf.create(patch_dict)
    merged = OmegaConf.merge(oc_cfg, oc_patch)
    new_obj: Config = OmegaConf.to_object(merged)
    for f in fields(Config):
        setattr(cfg, f.name, getattr(new_obj, f.name))


_BUILTIN_PROFILE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "opt_solver": {
        "lnr": {"code_organization_hint": "opt_solver"},
        "repl": {"repl_code_organization_hint": "opt_solver"},
    },
    "optimization_solver": {
        "lnr": {"code_organization_hint": "opt_solver"},
        "repl": {"repl_code_organization_hint": "opt_solver"},
    },
    "artifact_solver": {
        "lnr": {"code_organization_hint": "opt_solver"},
        "repl": {"repl_code_organization_hint": "opt_solver"},
    },
}


def _profile_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _profile_override_for(
    overrides: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    target = _profile_key(profile)
    for key, value in overrides.items():
        if _profile_key(key) == target and isinstance(value, dict):
            return dict(value)
    return {}


def _merge_profile_payloads(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        for block_name, block_value in payload.items():
            if (
                isinstance(block_value, dict)
                and isinstance(merged.get(block_name), dict)
            ):
                merged[block_name] = {**merged[block_name], **block_value}
            else:
                merged[block_name] = block_value
    return merged


def _merge_profile_block(
    target: Any,
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    known = {f.name for f in fields(target)}
    if isinstance(target, LnrConfig):
        payload = expand_lnr_resource_control_mode_payload(payload)
    patch: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            warnings.warn(
                f"Ignoring unknown profile override key {label}.{key!r}",
                UserWarning,
                stacklevel=3,
            )
            continue
        patch[key] = value
    if not patch:
        return
    merged = OmegaConf.merge(OmegaConf.structured(target), OmegaConf.create(patch))
    new_obj = OmegaConf.to_object(merged)
    for f in fields(target):
        setattr(target, f.name, getattr(new_obj, f.name))


def apply_profile_overrides(cfg: Config, profile: str | None = None) -> None:
    """Apply task-profile defaults for LNR/REPL/evaluator isolation.

    This function is intentionally separate from :func:`load_cfg`: CLI callers
    invoke it after generic manifest defaults and before explicit manifest
    ``lnr``/``agent`` patches, so task-level values remain the highest priority.
    """
    package_profile = ""
    package_spec = None
    if profile is None:
        try:
            from scienceflow.core.task_package import find_task_package

            package_spec = find_task_package(str(cfg.exp_id or ""))
            package_profile = str(package_spec.profile or "") if package_spec is not None else ""
        except Exception:
            package_profile = ""
            package_spec = None
    configured_profile = str(cfg.evaluator.task_profile or "").strip()
    raw_profile = profile or package_profile or configured_profile or "auto"
    cfg.evaluator.task_profile = str(raw_profile).strip() or "auto"
    if package_spec is not None:
        if str(cfg.evaluator.backend or "").strip().lower() in {"", "auto"}:
            cfg.evaluator.backend = "task_package"
        if not str(cfg.evaluator.candidate.artifact or "").strip():
            cfg.evaluator.candidate.artifact = package_spec.artifact_path
        if not str(cfg.evaluator.candidate.artifact_kind or "").strip():
            cfg.evaluator.candidate.artifact_kind = package_spec.artifact_kind
        if str(cfg.evaluator.metric.name or "").strip() in {"", "metric"}:
            cfg.evaluator.metric.name = package_spec.metric_name
        if not str(cfg.evaluator.metric.type or "").strip():
            cfg.evaluator.metric.type = package_spec.metric_type
        if cfg.evaluator.metric.lower_is_better is None:
            cfg.evaluator.metric.lower_is_better = package_spec.lower_is_better
    key = _profile_key(raw_profile) or "auto"
    builtin = _profile_override_for(_BUILTIN_PROFILE_OVERRIDES, key)
    custom = _profile_override_for(cfg.profile_overrides or {}, key)
    merged = _merge_profile_payloads(builtin, custom)
    if not merged:
        return
    for block_name, payload in merged.items():
        if not isinstance(payload, dict):
            warnings.warn(
                f"Ignoring profile override block {key}.{block_name!r}: expected mapping",
                UserWarning,
                stacklevel=2,
            )
            continue
        if block_name == "lnr":
            _merge_profile_block(cfg.lnr, payload, label=f"{key}.lnr")
        elif block_name == "repl":
            _merge_profile_block(cfg.repl, payload, label=f"{key}.repl")
        elif block_name == "evaluator":
            _merge_profile_block(cfg.evaluator, payload, label=f"{key}.evaluator")
        else:
            warnings.warn(
                f"Ignoring unsupported profile override block {key}.{block_name!r}",
                UserWarning,
                stacklevel=2,
            )


def _apply_env(cfg: Config) -> None:
    """Read environment variables and merge into Config.

    Env vars read:
        API_KEY / API_KEYS  — single key or space/comma-separated pool
        BASE_URL / BASE_URLS — single url or space/comma-separated pool
        CODE_API_KEY(S) / FEEDBACK_API_KEY(S) — stage-specific key or pool
        CODE_BASE_URL(S) / FEEDBACK_BASE_URL(S) — stage-specific endpoint(s)
        SCIENCEFLOW_LLM_ROUTING_MODE / CODE_LLM_ROUTING_MODE / FEEDBACK_LLM_ROUTING_MODE
            — ``round_robin`` or ``sticky_failover`` for API key pools
        SCIENCEFLOW_LLM_STICKY_ID / CODE_LLM_STICKY_ID / FEEDBACK_LLM_STICKY_ID
            — stable hash seed for sticky routing when no explicit primary index is set
        SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX / CODE_LLM_STICKY_PRIMARY_INDEX / FEEDBACK_LLM_STICKY_PRIMARY_INDEX
            — explicit primary endpoint index for sticky routing
        CODE_MODEL / FEEDBACK_MODEL — per-stage model (code / feedback)
        CODE_MODELS / FEEDBACK_MODELS — comma/space-separated model pool for
            worker-indexed LHR endpoint binding
        HTTP_TIMEOUT — if set (integer seconds), overrides http_timeout for all stages
        SCIENCEFLOW_LLM_STREAM_TIMEOUT — overrides ``scienceflow_llm_stream_timeout_sec``
        SCIENCEFLOW_BASH_TIMEOUT — overrides ``scienceflow_bash_timeout_sec``
        SCIENCEFLOW_BASH_TIMEOUT_SLOW — overrides ``scienceflow_bash_timeout_slow_sec``
        SCIENCEFLOW_STDOUT_MAX_CHARS — overrides ``scienceflow_stdout_max_chars`` (0 = no stdout cap)
        SCIENCEFLOW_INTERACTION_LOG_LEVEL — ``minimal`` / ``normal`` / ``verbose``; sets ``scienceflow_interaction_log_level``
        SCIENCEFLOW_INTERACTION_LOG_FULL — if 1/true/yes/on, sets ``scienceflow_interaction_log_full``
        SCIENCEFLOW_INTERACTION_LOG_COLOR — 1/true/yes/on → ``scienceflow_interaction_log_color`` True;
            0/false/no/off → False (overrides default-on)
        SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM — 1/true/yes/on → ``scienceflow_interaction_log_llm_stream`` True;
            0/false/no/off → False (mirror raw LLM streaming chunks into interaction.log)
        SCIENCEFLOW_BASH_STREAM_TO_INTERACTION_LOG — toggle filtered bash stream mirror into interaction.log
        SCIENCEFLOW_TOOL_MEMORY_COMPRESSION — 0/false/no/off disables ``tool_memory_compression``;
            1/true/yes/on enables (default on)
        SCIENCEFLOW_PARALLEL_BASH_ENABLED — 0/false/no/off disables ``parallel_bash_enabled``;
            1/true/yes/on enables (default on)
        SCIENCEFLOW_PARALLEL_LLM_TOOL_CALLS — 0/false/no/off sets ``parallel_llm_tool_calls`` False
            (API may return at most one tool call per turn); 1/true/yes/on enables (default on)
        SCIENCEFLOW_BASH_SUCCESS_TAIL_LINES — integer; last N lines of bash stdout kept in LLM memory
        PATH_GUARD_EXTRA_ROOTS — extra absolute path prefixes for ScienceAgent PathGuard
            (colon/semicolon/comma/whitespace-separated); merged with YAML ``path_guard_extra_roots``
        MLEBENCH_DATA_ROOT_DIR — path to mlebench data root (optional)
        CUSTOM_METRIC_NAME — optional; mirrors ``Config.custom_metric_name`` for grading hints
        CUSTOM_IS_LOWER_BETTER — optional; ``1``/``true``/``yes`` → True, ``0``/``false`` → False
        SCIENCEFLOW_EXP_ID — competition / experiment slug (set by ``parallel`` subprocesses;
            overrides ``Config.exp_id`` when non-empty)
    When a stage has ``use_proxy: true`` (YAML), ``scienceflow.core.llm_http`` also reads
    ``http_proxy`` / ``HTTP_PROXY``, ``USER_NAME``, ``USER_PWD`` (not merged here).
    """
    mlebench_dir = os.environ.get("MLEBENCH_DATA_ROOT_DIR", "").strip()
    if mlebench_dir and not cfg.mlebench_data_root_dir:
        cfg.mlebench_data_root_dir = mlebench_dir

    exp_id_env = os.environ.get("SCIENCEFLOW_EXP_ID", "").strip()
    if exp_id_env:
        cfg.exp_id = exp_id_env

    cm = os.environ.get("CUSTOM_METRIC_NAME", "").strip()
    if cm:
        cfg.custom_metric_name = cm
        cfg.evaluator.metric.name = cm
    cilb = os.environ.get("CUSTOM_IS_LOWER_BETTER", "").strip().lower()
    if cilb in ("1", "true", "yes", "on"):
        cfg.custom_is_lower_better = True
        cfg.evaluator.metric.lower_is_better = True
    elif cilb in ("0", "false", "no", "off"):
        cfg.custom_is_lower_better = False
        cfg.evaluator.metric.lower_is_better = False

    api_keys = _split_env("API_KEYS")
    api_key = os.environ.get("API_KEY", "").strip()
    base_urls = _split_env("BASE_URLS")
    base_url = os.environ.get("BASE_URL", "").strip()

    http_timeout_env = os.environ.get("HTTP_TIMEOUT", "").strip()
    http_timeout_override: int | None = None
    if http_timeout_env:
        try:
            http_timeout_override = int(http_timeout_env)
        except ValueError:
            pass

    for name in ("code", "feedback"):
        stage: StageConfig = getattr(cfg.agent, name)

        # model: env CODE_MODEL > yaml
        env_model = os.environ.get(STAGE_MODEL_ENV_MAP[name], "").strip()
        if env_model:
            stage.model = env_model
        env_models = _split_env(STAGE_MODELS_ENV_MAP[name])
        if env_models:
            stage.models = env_models

        endpoint_env = STAGE_ENDPOINT_ENV_MAP[name]
        stage_api_keys = _split_env(endpoint_env["api_keys"])
        stage_api_key = os.environ.get(endpoint_env["api_key"], "").strip()
        stage_base_urls = _split_env(endpoint_env["base_urls"])
        stage_base_url = os.environ.get(endpoint_env["base_url"], "").strip()

        # keys: YAML stage config > stage-specific env > generic env
        if not (stage.api_keys or stage.api_key):
            if stage_api_keys:
                stage.api_keys = stage_api_keys
            elif stage_api_key:
                stage.api_key = stage_api_key
            elif api_keys:
                stage.api_keys = api_keys
            elif api_key:
                stage.api_key = api_key

        # urls: YAML stage config > stage-specific env > generic env
        if not (stage.base_urls or stage.base_url):
            if stage_base_urls:
                stage.base_urls = stage_base_urls
            elif stage_base_url:
                stage.base_url = stage_base_url
            elif base_urls:
                stage.base_urls = base_urls
            elif base_url:
                stage.base_url = base_url

        if http_timeout_override is not None:
            stage.http_timeout = http_timeout_override

        upper = name.upper()
        routing_mode = (
            os.environ.get(f"{upper}_LLM_ROUTING_MODE", "").strip()
            or os.environ.get("SCIENCEFLOW_LLM_ROUTING_MODE", "").strip()
        )
        if routing_mode:
            stage.api_routing_mode = routing_mode
        sticky_id = (
            os.environ.get(f"{upper}_LLM_STICKY_ID", "").strip()
            or os.environ.get("SCIENCEFLOW_LLM_STICKY_ID", "").strip()
        )
        if sticky_id:
            stage.api_sticky_id = sticky_id
        sticky_primary_raw = (
            os.environ.get(f"{upper}_LLM_STICKY_PRIMARY_INDEX", "").strip()
            or os.environ.get("SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX", "").strip()
        )
        if sticky_primary_raw:
            try:
                stage.api_sticky_primary_index = int(sticky_primary_raw)
            except ValueError:
                pass
        rate_limit_raw = (
            os.environ.get(f"{upper}_LLM_RATE_LIMIT_COOLDOWN_SEC", "").strip()
            or os.environ.get("SCIENCEFLOW_LLM_RATE_LIMIT_COOLDOWN_SEC", "").strip()
        )
        if rate_limit_raw:
            try:
                stage.api_rate_limit_cooldown_sec = float(rate_limit_raw)
            except ValueError:
                pass
        connection_raw = (
            os.environ.get(f"{upper}_LLM_CONNECTION_COOLDOWN_SEC", "").strip()
            or os.environ.get("SCIENCEFLOW_LLM_CONNECTION_COOLDOWN_SEC", "").strip()
        )
        if connection_raw:
            try:
                stage.api_connection_cooldown_sec = float(connection_raw)
            except ValueError:
                pass

    _scienceflow_llm = os.environ.get("SCIENCEFLOW_LLM_STREAM_TIMEOUT", "").strip()
    if _scienceflow_llm:
        try:
            cfg.scienceflow_llm_stream_timeout_sec = int(_scienceflow_llm)
        except ValueError:
            pass
    _scienceflow_bash = os.environ.get("SCIENCEFLOW_BASH_TIMEOUT", "").strip()
    if _scienceflow_bash:
        try:
            cfg.scienceflow_bash_timeout_sec = float(_scienceflow_bash)
        except ValueError:
            pass
    _scienceflow_bash_slow = os.environ.get("SCIENCEFLOW_BASH_TIMEOUT_SLOW", "").strip()
    if _scienceflow_bash_slow:
        try:
            cfg.scienceflow_bash_timeout_slow_sec = float(_scienceflow_bash_slow)
        except ValueError:
            pass
    _scienceflow_stdout = os.environ.get("SCIENCEFLOW_STDOUT_MAX_CHARS", "").strip()
    if _scienceflow_stdout:
        try:
            cfg.scienceflow_stdout_max_chars = int(_scienceflow_stdout)
        except ValueError:
            pass

    _scienceflow_interaction_log = os.environ.get("SCIENCEFLOW_INTERACTION_LOG_FULL", "").strip().lower()
    if _scienceflow_interaction_log in ("1", "true", "yes", "on"):
        cfg.scienceflow_interaction_log_full = True

    _scienceflow_log_level = os.environ.get("SCIENCEFLOW_INTERACTION_LOG_LEVEL", "").strip().lower()
    if _scienceflow_log_level:
        if _scienceflow_log_level in ("minimal", "m"):
            cfg.scienceflow_interaction_log_level = "minimal"
        elif _scienceflow_log_level in ("normal", "n"):
            cfg.scienceflow_interaction_log_level = "normal"
        elif _scienceflow_log_level in ("verbose", "v"):
            cfg.scienceflow_interaction_log_level = "verbose"

    _scienceflow_interaction_color = os.environ.get("SCIENCEFLOW_INTERACTION_LOG_COLOR", "").strip().lower()
    if _scienceflow_interaction_color in ("1", "true", "yes", "on"):
        cfg.scienceflow_interaction_log_color = True
    elif _scienceflow_interaction_color in ("0", "false", "no", "off"):
        cfg.scienceflow_interaction_log_color = False

    _scienceflow_llm_stream_log = os.environ.get("SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM", "").strip().lower()
    if _scienceflow_llm_stream_log in ("1", "true", "yes", "on"):
        cfg.scienceflow_interaction_log_llm_stream = True
    elif _scienceflow_llm_stream_log in ("0", "false", "no", "off"):
        cfg.scienceflow_interaction_log_llm_stream = False

    _scienceflow_bash_stream = os.environ.get("SCIENCEFLOW_BASH_STREAM_TO_INTERACTION_LOG", "").strip().lower()
    if _scienceflow_bash_stream in ("1", "true", "yes", "on"):
        cfg.scienceflow_bash_stream_to_interaction_log = True
    elif _scienceflow_bash_stream in ("0", "false", "no", "off"):
        cfg.scienceflow_bash_stream_to_interaction_log = False

    _tmc = os.environ.get("SCIENCEFLOW_TOOL_MEMORY_COMPRESSION", "").strip().lower()
    if _tmc in ("0", "false", "no", "off"):
        cfg.tool_memory_compression = False
    elif _tmc in ("1", "true", "yes", "on"):
        cfg.tool_memory_compression = True

    _npb = os.environ.get("SCIENCEFLOW_PARALLEL_BASH_ENABLED", "").strip().lower()
    if _npb in ("0", "false", "no", "off"):
        cfg.parallel_bash_enabled = False
    elif _npb in ("1", "true", "yes", "on"):
        cfg.parallel_bash_enabled = True

    _nplc = os.environ.get("SCIENCEFLOW_PARALLEL_LLM_TOOL_CALLS", "").strip().lower()
    if _nplc in ("0", "false", "no", "off"):
        cfg.parallel_llm_tool_calls = False
    elif _nplc in ("1", "true", "yes", "on"):
        cfg.parallel_llm_tool_calls = True

    _bash_tail = os.environ.get("SCIENCEFLOW_BASH_SUCCESS_TAIL_LINES", "").strip()
    if _bash_tail:
        try:
            cfg.bash_success_tail_lines = int(_bash_tail)
        except ValueError:
            pass

    _pg_extra = os.environ.get("PATH_GUARD_EXTRA_ROOTS", "").strip()
    if _pg_extra:
        import re as _re

        seen_pg = {Path(p).resolve() for p in cfg.path_guard_extra_roots}
        for chunk in _re.split(r"[;:,\s]+", _pg_extra):
            s = chunk.strip()
            if not s:
                continue
            p = Path(s).expanduser().resolve()
            if p not in seen_pg:
                cfg.path_guard_extra_roots.append(p)
                seen_pg.add(p)


def _split_env(var: str) -> list[str]:
    """Split an env var by comma or whitespace, return non-empty items."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return []
    import re
    return [s.strip() for s in re.split(r"[,\s]+", raw) if s.strip()]


def _stage_endpoint_pairs(stage: StageConfig, *, stage_name: str) -> set[tuple[str, str]]:
    keys = [str(x).strip() for x in (stage.api_keys or []) if str(x).strip()]
    if not keys and str(stage.api_key or "").strip():
        keys = [str(stage.api_key).strip()]
    if not keys:
        return set()

    urls = [str(x).strip().rstrip("/") for x in (stage.base_urls or []) if str(x).strip()]
    if not urls and str(stage.base_url or "").strip():
        urls = [str(stage.base_url).strip().rstrip("/")]
    if not urls:
        urls = [""] * len(keys)
    elif len(urls) == 1 and len(keys) > 1:
        urls = urls * len(keys)
    elif len(urls) != len(keys):
        raise ValueError(
            f"agent.{stage_name} has {len(keys)} API key(s) but {len(urls)} base URL(s); "
            "provide one base URL to broadcast or the same count as keys.",
        )
    return set(zip(urls, keys))


def _expand_parallel_manifest_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_parallel_manifest_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _expand_parallel_manifest_value(v) for k, v in value.items()}
    return value


def _apply_parallel_manifest_dataclass_patch(
    target: Any,
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    known = {f.name for f in fields(target)}
    for key, value in payload.items():
        if key not in known:
            warnings.warn(
                f"Ignoring unknown {label} key from parallel manifest: {key!r}",
                UserWarning,
                stacklevel=2,
            )
            continue
        cur = getattr(target, key)
        value = _expand_parallel_manifest_value(value)
        try:
            if isinstance(cur, bool):
                setattr(target, key, bool(value))
            elif isinstance(cur, int):
                setattr(target, key, int(value))
            elif isinstance(cur, float):
                setattr(target, key, float(value))
            elif isinstance(cur, str):
                setattr(target, key, str(value))
            elif isinstance(cur, list):
                if not isinstance(value, list):
                    raise TypeError(f"expected list for {key!r}")
                setattr(target, key, list(value))
            elif isinstance(cur, dict):
                if not isinstance(value, dict):
                    raise TypeError(f"expected mapping for {key!r}")
                setattr(target, key, dict(value))
            else:
                setattr(target, key, value)
        except (TypeError, ValueError) as e:
            warnings.warn(
                f"Could not apply {label}.{key}={value!r} from parallel manifest: {e}",
                RuntimeWarning,
                stacklevel=2,
            )


def apply_parallel_manifest_agent_overrides(cfg: Config) -> None:
    """Apply ``agent`` keys from parallel manifest (child env ``SCIENCEFLOW_PARALLEL_AGENT_JSON``).

    Set by :class:`scienceflow.core.parallel_runner.ParallelRunner` when the manifest
    defines ``agent:``. Top-level ``AgentConfig`` fields and nested
    ``agent.code`` / ``agent.feedback`` ``StageConfig`` fields are supported.
    """
    raw = os.environ.get("SCIENCEFLOW_PARALLEL_AGENT_JSON", "").strip()
    if not raw:
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"SCIENCEFLOW_PARALLEL_AGENT_JSON is invalid JSON: {e}; ignoring.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if not isinstance(payload, dict):
        warnings.warn(
            "SCIENCEFLOW_PARALLEL_AGENT_JSON must be a JSON object; ignoring.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    ag = cfg.agent
    nested_stage_keys = {"code", "feedback"}
    known = {f.name for f in fields(AgentConfig)} - nested_stage_keys
    for key, value in payload.items():
        if key in nested_stage_keys:
            if not isinstance(value, dict):
                warnings.warn(
                    f"Ignoring agent.{key} from parallel manifest: expected mapping, "
                    f"got {type(value).__name__}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            _apply_parallel_manifest_dataclass_patch(
                getattr(ag, key),
                value,
                label=f"agent.{key}",
            )
            continue
        if key not in known:
            warnings.warn(
                f"Ignoring unknown agent key from parallel manifest: {key!r}",
                UserWarning,
                stacklevel=2,
            )
            continue
        cur = getattr(ag, key)
        value = _expand_parallel_manifest_value(value)
        try:
            if isinstance(cur, bool):
                setattr(ag, key, bool(value))
            elif isinstance(cur, int):
                setattr(ag, key, int(value))
            elif isinstance(cur, float):
                setattr(ag, key, float(value))
            elif isinstance(cur, str):
                setattr(ag, key, str(value))
            else:
                setattr(ag, key, value)
        except (TypeError, ValueError) as e:
            warnings.warn(
                f"Could not apply agent.{key}={value!r} from parallel manifest: {e}",
                RuntimeWarning,
                stacklevel=2,
            )



def _apply_parallel_manifest_lnr_payload(
    lnr: LnrConfig,
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    known = {f.name for f in fields(LnrConfig)}
    payload = expand_lnr_resource_control_mode_payload(payload)
    for key, value in payload.items():
        if key not in known:
            warnings.warn(
                f"Ignoring unknown {label} key from parallel manifest: {key!r}",
                UserWarning,
                stacklevel=2,
            )
            continue
        cur = getattr(lnr, key)
        try:
            if isinstance(cur, bool):
                setattr(lnr, key, bool(value))
            elif isinstance(cur, int):
                setattr(lnr, key, int(value))
            elif isinstance(cur, float):
                setattr(lnr, key, float(value))
            elif isinstance(cur, str):
                setattr(lnr, key, str(value))
            elif isinstance(cur, list):
                if not isinstance(value, list):
                    raise TypeError(f"expected list for {key!r}")
                setattr(lnr, key, list(value))
            elif is_dataclass(cur) and isinstance(value, dict):
                nested_known = {f.name for f in fields(cur)}
                for nested_key, nested_value in value.items():
                    if nested_key not in nested_known:
                        warnings.warn(
                            f"Ignoring unknown {label}.{key} key from parallel manifest: {nested_key!r}",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue
                    nested_cur = getattr(cur, nested_key)
                    if isinstance(nested_cur, bool):
                        setattr(cur, nested_key, bool(nested_value))
                    elif isinstance(nested_cur, int):
                        setattr(cur, nested_key, int(nested_value))
                    elif isinstance(nested_cur, float):
                        setattr(cur, nested_key, float(nested_value))
                    elif isinstance(nested_cur, str):
                        setattr(cur, nested_key, str(nested_value))
                    else:
                        setattr(cur, nested_key, nested_value)
            else:
                setattr(lnr, key, value)
        except (TypeError, ValueError) as e:
            warnings.warn(
                f"Could not apply {label}.{key}={value!r} from parallel manifest: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

def apply_parallel_manifest_lnr_overrides(cfg: Config) -> None:
    """Apply lnr manifest keys for type: lnr children."""
    raw = os.environ.get("SCIENCEFLOW_PARALLEL_LNR_JSON", "").strip()
    if not raw:
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"SCIENCEFLOW_PARALLEL_LNR_JSON is invalid JSON: {e}; ignoring lnr overrides.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if not isinstance(payload, dict):
        warnings.warn(
            "SCIENCEFLOW_PARALLEL_LNR_JSON must be a JSON object; ignoring.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    _apply_parallel_manifest_lnr_payload(cfg.lnr, payload, label="lnr")

def _sanitize_config_dict_for_yaml(obj: Any) -> Any:
    """Convert Path / nested structures for ``yaml.safe_dump``."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_config_dict_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_config_dict_for_yaml(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_sanitize_config_dict_for_yaml(v) for v in obj)
    return obj


def dump_resolved_config_yaml(cfg: Config, dest: Path | None = None) -> None:
    """Write the effective :class:`Config` as YAML (paths as strings).

    *dest* defaults to ``<task_workspace_root_dir>/resolved_config.yaml``.
    Skip if env ``SCIENCEFLOW_SKIP_RESOLVED_CONFIG_YAML`` is truthy.
    """
    if os.environ.get("SCIENCEFLOW_SKIP_RESOLVED_CONFIG_YAML", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    root = Path(cfg.task_workspace_root_dir).expanduser().resolve()
    out = dest if dest is not None else root / "resolved_config.yaml"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(cfg)
        data = _sanitize_config_dict_for_yaml(data)
        header = (
            "# scienceflow resolved_config.yaml — effective Config after load, env, CLI, then prep_cfg.\n"
            "# Field config_source_path is the YAML used with --config when applicable.\n"
            "# api_key fields may contain secrets; treat this file as sensitive.\n"
        )
        with open(out, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(
                data,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
    except Exception as e:
        warnings.warn(
            f"Could not write resolved_config.yaml to {out}: {e}",
            RuntimeWarning,
            stacklevel=2,
        )


def prep_cfg(cfg: Config) -> Config:
    """Derive execution workspace_dir, log_dir, submission paths; create directories."""
    root = Path(cfg.task_workspace_root_dir).expanduser().resolve()
    workspace_override = getattr(cfg, "_workspace_dir_override", None)
    inner = (
        Path(workspace_override).expanduser().resolve()
        if workspace_override is not None and str(workspace_override).strip()
        else root
    )
    cfg.workspace_dir = inner
    log_dir_override = getattr(cfg, "_log_dir_override", None)
    cfg.log_dir = (
        Path(log_dir_override).expanduser().resolve()
        if log_dir_override is not None and str(log_dir_override).strip()
        else root / "logs"
    )
    cfg.submission_dir = inner / "submissions"

    for d in (cfg.log_dir, inner, cfg.submission_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    roots: list[Path] = []
    seen_r: set[Path] = set()
    for p in getattr(cfg, "path_guard_extra_roots", None) or []:
        s = str(p).strip()
        if not s:
            continue
        rp = Path(s).expanduser().resolve()
        if rp not in seen_r:
            roots.append(rp)
            seen_r.add(rp)
    cfg.path_guard_extra_roots = roots

    lnr = getattr(cfg, "lnr", None)
    if lnr is not None and bool(getattr(lnr, "compact_on_context_limit", False)):
        try:
            floor = int(getattr(lnr, "context_limit_min_messages", 0) or 0)
        except (TypeError, ValueError):
            floor = 0
        if floor > 0 and int(getattr(cfg, "max_messages", 0) or 0) < floor:
            cfg.max_messages = floor

    dump_resolved_config_yaml(cfg)
    return cfg
