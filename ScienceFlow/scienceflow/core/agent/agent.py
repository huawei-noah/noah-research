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

"""ScienceAgent - function-calling agent over local workspace tools."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import deque
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable, Sequence

from deepcraft_agent import BaseAgent
from deepcraft_core import Message
from deepcraft_core.tool import ToolResult
from pydantic import ConfigDict

from scienceflow.core.agent.shared.constants import _WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD
from scienceflow.core.agent.run_control.embedded_fullrun import EmbeddedFullRunMixin
from scienceflow.core.agent.runtime.llm_stream import LLMStreamMixin
from scienceflow.core.agent.runtime.lnr_hooks import LNRHooksMixin
from scienceflow.core.agent.runtime.recovery import RecoveryMixin
from scienceflow.core.agent.runtime.run_loop import RunLoopMixin
from scienceflow.core.agent.prompts.system_prompt import (
    SystemPromptMixin,
    _code_agent_core_prompt,
    _default_system_prompt,
)
from scienceflow.core.agent.tool_exec.parallel import ParallelToolExecMixin
from scienceflow.core.agent.tool_exec.sequential import SequentialRecoveryMixin
from scienceflow.core.agent.tool_exec.single import SingleToolExecMixin
from scienceflow.core.agent.tools.tool_guards import (
    BashPythonSourceDumpGuard,
    BashRepeatFailureGuard,
    EditFailureGuard,
    ExploreStreakGuard,
    GuardManager,
    NoProgressHardStopGuard,
    NoSuccessfulSolutionRunGuard,
    RepeatedRuntimeErrorGuard,
    RuntimeErrorGuard,
    SingleReadStreakGuard,
    WriteFailureGuard,
    WriteNudgeGuard,
    WriteRepeatGuard,
)
from scienceflow.core.tools.file_utils import PathGuard
from scienceflow.core.agent.prompts.write_coaching import WriteCoachingMixin
from scienceflow.core.agent.registry import register_agent
from scienceflow.core.agent.run_policy import DefaultPolicy, RunPolicy
from scienceflow.core.tools import create_tool_collection
from scienceflow.core.mem.memory_context import MemoryContextManager, trim_tool_feedback_for_llm_context
from scienceflow.core.agent.tools.tool_output_artifacts import (
    ToolOutputArtifactStore,
    attach_tool_output_reference,
    raw_tool_result_text,
    reduce_tool_feedback_for_memory,
)
from scienceflow.core.agent.memory.resource_feedback_memory import ResourceFeedbackMemoryDeduper
from scienceflow.core.agent.memory.memory_utils import (
    _assistant_message_from_api,
    inject_thought_into_tool_params,
)
from scienceflow.core.skills.registry import SkillRegistry
from scienceflow.core.agent.io.interaction_log_policy import resolve_interaction_log_policy
from scienceflow.utils.workspace_interaction_log import (
    attach_workspace_interaction_logger,
    build_interaction_log_context_tag,
    set_interaction_log_context_tag,
)

if TYPE_CHECKING:
    from scienceflow.ui.console import RichUI

_logger = logging.getLogger("scienceflow")
_ON_LLM_CALL_UNSET = object()
_HOST_ABSOLUTE_VISIBLE_PATH_RE = re.compile(
    r"(?<![\w.])/(?:home|work|mnt|kaggle|tmp)(?:/[^\s\"'`<>),;]*)?",
)
_WSP_ABSOLUTE_VISIBLE_PREFIX_RE = re.compile(r"/\S+/wsp/[0-9a-f]{32}/")



@register_agent("science")
class ScienceAgent(
    RunLoopMixin,
    RecoveryMixin,
    SingleToolExecMixin,
    SequentialRecoveryMixin,
    ParallelToolExecMixin,
    EmbeddedFullRunMixin,
    WriteCoachingMixin,
    LNRHooksMixin,
    LLMStreamMixin,
    SystemPromptMixin,
    BaseAgent,
):
    """Multi-step LLM agent using OpenAI-style function calling and local tools."""

    max_steps: int = 30

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        *,
        workspace_dir: str | Path = ".",
        sandbox: bool = True,
        ui: RichUI | None = None,
        max_steps: int | None = None,
        exec_feedback_max_chars: int = 5000,
        scienceflow_stdout_max_chars: int = 0,
        sliding_window_budget_chars: int = 60_000,
        pinned_budget_ratio: float = 0.45,
        sliding_window_priority_enabled: bool = True,
        mid_run_compact_enabled: bool = False,
        llm_stream_timeout_sec: int = 1800,
        llm_tool_stream_max_attempts: int = 5,
        llm_tool_stream_retry_base_delay_sec: float = 1.75,
        llm_tool_stream_retry_max_delay_sec: float = 20.0,
        stream_repetition_detection: bool = True,
        stream_repetition_window_chars: int = 4000,
        stream_repetition_ngram_len: int = 150,
        stream_repetition_max_repeats: int = 3,
        stream_max_output_chars_soft: int = 30000,
        stream_repetition_retry_max: int = 2,
        bash_timeout_sec: float = 1800.0,
        bash_timeout_slow_sec: float = 1800.0,
        bash_max_output_chars: int = 8000,
        bash_max_stream_line_chars: int = 2400,
        bash_observation_summary_enabled: bool = False,
        system_prompt_hook: Callable[[str], str] | None = None,
        system_prompt_core: str | None = None,
        path_guard_extra_roots: Sequence[str | Path] | None = None,
        extra_env: dict[str, str] | None = None,
        readonly_dirs: Sequence[str] | None = None,
        run_policy: RunPolicy | None = None,
        embedded_full_run_enabled: bool = False,
        skip_embedded_duplicate_when_progressive_bare_success: bool = True,
        embedded_full_run_update_result_md: bool = True,
        embedded_full_run_timeout_sec: float | None = None,
        fullrun_output_tail_stdout_lines: int = 50,
        fullrun_output_tail_stderr_lines: int = 0,
        fullrun_output_tail_max_chars: int = 4000,
        quick_test_extrapolation_enabled: bool = True,
        quick_test_extrapolation_budget_sec: float | None = None,
        quick_test_extrapolation_rows: int = 20,
        fullrun_epoch_watchdog_enabled: bool = True,
        fullrun_epoch_watchdog_budget_sec: float | None = None,
        interaction_log_level: str = "normal",
        interaction_log_full: bool = False,
        interaction_log_color: bool = True,
        interaction_log_llm_stream: bool = False,
        interaction_log_layout: str = "flat",
        bash_stream_to_interaction_log: bool = True,
        bash_output_dedup_enabled: bool = True,
        bash_output_dedup_min_repeat: int = 3,
        bash_output_dedup_summary_prefix: str = "[log-dedup]",
        bash_output_dedup_apply_to_memory: bool = True,
        tool_memory_compression: bool = True,
        include_write_edit_tools: bool = True,
        tool_display_paths_relative: bool = True,
        tool_display_root: str | Path | None = None,
        write_return_full_max_chars: int = 12000,
        write_return_full_max_lines: int = 250,
        write_return_head_tail_lines: int = 40,
        edit_return_full_max_chars: int = 12000,
        edit_return_full_max_lines: int = 250,
        edit_return_head_tail_lines: int = 40,
        edit_return_change_ctx_lines: int = 8,
        edit_failure_top_k_candidates: int = 3,
        edit_failure_diag_max_chars: int = 2000,
        file_snapshot_latest_only: bool = True,
        grep_max_results_lines: int = 50,
        bash_success_tail_lines: int = 20,
        bash_success_tail_lines_solution: int = 8,
        bash_success_tail_lines_test: int = 120,
        bash_success_tail_lines_readonly: int = 30,
        bash_success_tail_lines_install: int = 5,
        read_success_max_lines: int = 200,
        write_auto_snapshot_enabled: bool = True,
        write_auto_snapshot_paths: Sequence[str] | None = None,
        write_auto_snapshot_code_extensions: Sequence[str] | None = None,
        write_auto_snapshot_max_lines: int = 400,
        write_auto_snapshot_max_chars: int = 8_000,
        write_auto_snapshot_changed_context_lines: int = 10,
        write_auto_snapshot_symbol_body_lines: int = 3,
        read_overlap_guard_enabled: bool = True,
        msg0_compress_body: bool = False,
        on_llm_call: Callable[[dict[str, Any]], None] | None = None,
        debug_dynamic_steps_enabled: bool = False,
        debug_step_boost: int = 0,
        debug_max_steps_cap: int = 0,
        round_budget_prompt_cap: int = 0,
        parallel_bash_enabled: bool = True,
        parallel_llm_tool_calls: bool = True,
        lnr_explore_streak_inject_after: int = 0,
        lnr_single_read_streak_inject_after: int = 0,
        no_progress_hard_stop_after: int = 0,
        no_progress_hardstop_guard_grace_rounds: int = 1,
        repeated_runtime_error_threshold: int = 2,
        rotating_runtime_error_threshold: int = 3,
        rotating_distinct_types_min: int = 2,
        no_success_run_soft_threshold: int = 6,
        no_success_run_hard_threshold: int = 12,
        edit_short_old_str_chars: int = 40,
        lnr_category_skill_text: str = "",
        lnr_phase_skill_registry: dict[str, str] | None = None,
        lnr_phase_skill_descriptions: dict[str, str] | None = None,
        lnr_skill_inject_max_chars: int = 12000,
        lnr_periodic_no_solution_every: int = 0,
        lnr_llm_turns_log_enabled: bool = False,
        lnr_llm_turns_log_path: str | Path | None = None,
        sft_data_log_path: str | Path | None = None,
        mlebench_validate_after_embedded_full_run: bool = False,
        mlebench_data_dir: str | None = None,
        mlebench_exp_id: str | None = None,
        lnr_mlebench_validate_enabled: bool = True,
        lnr_run_control_max_fix_rounds: int = 5,
        # LNR only: end ``run()`` after a bare solution run passes run-control checks.
        lnr_stop_after_bare_solution_success: bool = False,
        lnr_superloop_enabled: bool = False,
        # Teleport mode (search-wide system prompt + 4-class fork policy). When != "off",
        # the per-agent ``_lnr_phase_header`` is suppressed (kept empty) so guard/periodic
        # nudge messages do not re-inject ``## Phase Header`` blocks into chat — the
        # search-wide system prompt already conveys phase-agnostic rules.
        teleport_mode: str = "off",
        # REPL/code-agent mode: keep the system prompt byte-stable by omitting
        # dynamic file-state and round-budget suffixes. Dynamic state remains in
        # tool results and workspace files instead of the system prefix.
        stable_system_prompt: bool = False,
        workspace_git_auto_checkpoint_enabled: bool = False,
        workspace_git_track_globs: Sequence[str] | None = None,
        # Teleport mode (F3): seed for ``_search_round_offset`` so a freshly constructed
        # ScienceAgent on a B/C-class fork (or first-fork fallback when persistent agent
        # registry is empty) keeps the system prompt's ``Round X/Y`` line monotonically
        # increasing instead of resetting to 1. Caller (clone.py / core.py) passes the
        # parent's accumulated round count; persistent-agent path uses ``swap_workspace``
        # which manages the offset internally.
        search_round_offset: int = 0,
        skill_registry: SkillRegistry | None = None,
        task_type: str | None = None,
        skill_allow_names: Sequence[str] | None = None,
        skill_tool_mode: str = "all",
        skill_allow_generic_wildcard: bool = True,
        skill_visible_max: int = 0,
        resource_observer: Any | None = None,
        interaction_log_session: str | None = None,
        interaction_log_phase: str | None = None,
        lnr_exploit_ensemble_mode: bool = False,
        lnr_hide_future_stages: bool = False,
        **kwargs: Any,
    ) -> None:
        ws = Path(workspace_dir).resolve()
        include_write_edit_tools = bool(include_write_edit_tools)
        if "systemPrompt" not in kwargs or kwargs["systemPrompt"] is None:
            kwargs["systemPrompt"] = _default_system_prompt(
                parallel_bash_enabled=bool(parallel_bash_enabled),
                bash_file_write_mode=not include_write_edit_tools,
            )
        tools = create_tool_collection(
            ws,
            sandbox=sandbox,
            path_guard_extra_roots=path_guard_extra_roots,
            max_bash_output_chars=int(bash_max_output_chars),
            max_bash_stream_line_chars=int(bash_max_stream_line_chars),
            bash_observation_summary_enabled=bool(bash_observation_summary_enabled),
            bash_timeout_sec=bash_timeout_sec,
            bash_timeout_slow_sec=bash_timeout_slow_sec,
            extra_env=extra_env,
            readonly_dirs=readonly_dirs,
            resource_observer=resource_observer,
            skill_registry=skill_registry,
            task_type=task_type,
            skill_allow_names=skill_allow_names,
            skill_tool_mode=skill_tool_mode,
            skill_allow_generic_wildcard=skill_allow_generic_wildcard,
            skill_visible_max=skill_visible_max,
            include_write_edit_tools=include_write_edit_tools,
            tool_display_paths_relative=tool_display_paths_relative,
            tool_display_root=tool_display_root,
            write_return_full_max_chars=int(write_return_full_max_chars),
            write_return_full_max_lines=int(write_return_full_max_lines),
            write_return_head_tail_lines=int(write_return_head_tail_lines),
            edit_return_full_max_chars=int(edit_return_full_max_chars),
            edit_return_full_max_lines=int(edit_return_full_max_lines),
            edit_return_head_tail_lines=int(edit_return_head_tail_lines),
            edit_return_change_ctx_lines=int(edit_return_change_ctx_lines),
            edit_failure_top_k_candidates=int(edit_failure_top_k_candidates),
            edit_failure_diag_max_chars=int(edit_failure_diag_max_chars),
            grep_max_results_lines=int(grep_max_results_lines),
        )
        kwargs.setdefault("availableTools", tools)
        super().__init__(**kwargs)
        self._include_write_edit_tools = include_write_edit_tools
        self._tool_display_paths_relative = bool(tool_display_paths_relative)
        self._tool_display_root = (
            Path(tool_display_root).resolve() if tool_display_root is not None else ws
        )
        self._tools_with_thought = inject_thought_into_tool_params(
            self.availableTools.to_params(),
        )
        self._workspace_dir = ws
        self._ui = ui
        self._ui_step = 0
        self._skill_registry = skill_registry
        self._task_type = task_type
        self._skill_allow_names = tuple(skill_allow_names or ())
        self._skill_tool_mode = str(skill_tool_mode or "all")
        self._skill_allow_generic_wildcard = bool(skill_allow_generic_wildcard)
        self._skill_visible_max = int(skill_visible_max or 0)
        self._resource_observer = resource_observer
        self._bash_timeout_sec = float(bash_timeout_sec)
        self._bash_timeout_slow_sec = float(bash_timeout_slow_sec)
        self._bash_max_output_chars = int(bash_max_output_chars)
        self._bash_max_stream_line_chars = int(bash_max_stream_line_chars)
        self._bash_observation_summary_enabled = bool(bash_observation_summary_enabled)
        self._extra_env: dict[str, str] | None = dict(extra_env) if extra_env else None
        self._embedded_full_run_enabled = bool(embedded_full_run_enabled)
        self._skip_embedded_duplicate_when_progressive_bare_success = bool(
            skip_embedded_duplicate_when_progressive_bare_success,
        )
        self._embedded_full_run_update_result_md = bool(embedded_full_run_update_result_md)
        self._embedded_full_run_timeout_sec = embedded_full_run_timeout_sec
        self._fullrun_output_tail_stdout_lines = int(fullrun_output_tail_stdout_lines)
        self._fullrun_output_tail_stderr_lines = int(fullrun_output_tail_stderr_lines)
        self._fullrun_output_tail_max_chars = int(fullrun_output_tail_max_chars)
        self._quick_test_extrapolation_enabled = bool(quick_test_extrapolation_enabled)
        self._quick_test_extrapolation_budget_sec = quick_test_extrapolation_budget_sec
        self._quick_test_extrapolation_rows = int(quick_test_extrapolation_rows)
        self._fullrun_epoch_watchdog_enabled = bool(fullrun_epoch_watchdog_enabled)
        self._fullrun_epoch_watchdog_budget_sec = fullrun_epoch_watchdog_budget_sec
        self._embedded_full_run_done = False
        self._consecutive_write_syntax_fails: int = 0
        self._consecutive_infra_errors: int = 0
        self._initial_solution_sha: str | None = None
        if max_steps is not None:
            self.max_steps = int(max_steps)
        self._exec_feedback_max_chars = int(exec_feedback_max_chars)
        self._scienceflow_stdout_max_chars = int(scienceflow_stdout_max_chars)
        self._llm_stream_timeout_sec = int(llm_stream_timeout_sec)
        self._llm_tool_stream_max_attempts = max(1, int(llm_tool_stream_max_attempts))
        self._llm_tool_stream_retry_base_delay_sec = float(
            llm_tool_stream_retry_base_delay_sec,
        )
        self._llm_tool_stream_retry_max_delay_sec = float(
            llm_tool_stream_retry_max_delay_sec,
        )
        self._stream_repetition_detection = bool(stream_repetition_detection)
        self._stream_repetition_window_chars = max(256, int(stream_repetition_window_chars))
        self._stream_repetition_ngram_len = max(32, int(stream_repetition_ngram_len))
        self._stream_repetition_max_repeats = max(2, int(stream_repetition_max_repeats))
        self._stream_max_output_chars_soft = int(stream_max_output_chars_soft)
        self._stream_repetition_retry_max = max(0, int(stream_repetition_retry_max))
        self._last_stream_guard_reason: str | None = None
        self._last_stream_guard_detail: str = ""
        self._last_stream_guard_chars: int = 0
        self._tool_memory_compression = bool(tool_memory_compression)
        self._file_snapshot_latest_only = bool(file_snapshot_latest_only)
        self._mid_run_compact_enabled = bool(mid_run_compact_enabled)
        self._bash_output_dedup_enabled = bool(bash_output_dedup_enabled)
        self._bash_output_dedup_min_repeat = int(bash_output_dedup_min_repeat)
        self._bash_output_dedup_summary_prefix = str(bash_output_dedup_summary_prefix)
        self._bash_output_dedup_apply_to_memory = bool(bash_output_dedup_apply_to_memory)
        self._grep_max_results_lines = int(grep_max_results_lines)
        self._interaction_log_layout = (
            "split"
            if str(interaction_log_layout or "flat").strip().lower() in {"split", "lhr_split"}
            else "flat"
        )
        self._memory_ctx = MemoryContextManager(
            self.memory,
            self._workspace_dir,
            budget_chars=int(sliding_window_budget_chars),
            path_guard_extra_roots=path_guard_extra_roots,
            tool_memory_compression=self._tool_memory_compression,
            bash_success_tail_lines=int(bash_success_tail_lines),
            bash_success_tail_lines_solution=int(bash_success_tail_lines_solution),
            bash_success_tail_lines_test=int(bash_success_tail_lines_test),
            bash_success_tail_lines_readonly=int(bash_success_tail_lines_readonly),
            bash_success_tail_lines_install=int(bash_success_tail_lines_install),
            read_success_max_lines=int(read_success_max_lines),
            sliding_window_priority_enabled=bool(sliding_window_priority_enabled),
            pinned_budget_ratio=float(pinned_budget_ratio),
            bash_output_dedup_apply_to_memory=self._bash_output_dedup_apply_to_memory,
            bash_output_dedup_min_repeat=self._bash_output_dedup_min_repeat,
            bash_output_dedup_summary_prefix=self._bash_output_dedup_summary_prefix,
            write_auto_snapshot_enabled=bool(write_auto_snapshot_enabled),
            write_auto_snapshot_paths=write_auto_snapshot_paths,
            write_auto_snapshot_code_extensions=write_auto_snapshot_code_extensions,
            write_auto_snapshot_max_lines=int(write_auto_snapshot_max_lines),
            write_auto_snapshot_max_chars=int(write_auto_snapshot_max_chars),
            write_auto_snapshot_changed_context_lines=int(
                write_auto_snapshot_changed_context_lines,
            ),
            write_auto_snapshot_symbol_body_lines=int(write_auto_snapshot_symbol_body_lines),
            read_overlap_guard_enabled=bool(read_overlap_guard_enabled),
            msg0_compress_body=bool(msg0_compress_body),
        )
        self._tool_output_artifacts = self._make_tool_output_artifact_store(self._workspace_dir)
        self._resource_feedback_memory_deduper = ResourceFeedbackMemoryDeduper()
        self._system_prompt_hook = system_prompt_hook
        self._system_prompt_core = (
            system_prompt_core
            if system_prompt_core is not None
            else (
                _code_agent_core_prompt(
                    bash_file_write_mode=not self._include_write_edit_tools,
                )
                if bool(stable_system_prompt)
                and str(teleport_mode or "off") in ("", "off")
                else ""
            )
        )
        self._ws_interaction_log = attach_workspace_interaction_logger(
            self._workspace_dir,
            color=bool(interaction_log_color),
            layout=self._interaction_log_layout,
        )
        _ilog_sess = (interaction_log_session or "draft").strip().lower() or "draft"
        _ilog_tag = build_interaction_log_context_tag(_ilog_sess, interaction_log_phase)
        self._lnr_interaction_log_session = _ilog_sess
        _ilp = (
            str(interaction_log_phase).strip().lower()
            if interaction_log_phase is not None and str(interaction_log_phase).strip()
            else ""
        )
        self._lnr_interaction_log_phase: str | None = _ilp or None
        self._lnr_exploit_ensemble_mode = bool(lnr_exploit_ensemble_mode)
        self._lnr_hide_future_stages = bool(lnr_hide_future_stages)
        self._teleport_mode = str(teleport_mode or "off")
        self._stable_system_prompt = bool(stable_system_prompt)
        self._workspace_git_auto_checkpoint_enabled = bool(
            workspace_git_auto_checkpoint_enabled,
        )
        self._workspace_git_track_globs = tuple(workspace_git_track_globs or ("*.py", "*.md"))
        # LNR/REPL agents do not inject legacy phase headers into chat.
        self._lnr_phase_header = ""
        self._interaction_log_ctx_token: Any = set_interaction_log_context_tag(_ilog_tag)
        self._run_policy: RunPolicy = (
            run_policy if run_policy is not None else DefaultPolicy(recovery_rounds=3)
        )
        self._interaction_log_policy = resolve_interaction_log_policy(
            level=interaction_log_level,
            legacy_full=bool(interaction_log_full),
            legacy_llm_stream=bool(interaction_log_llm_stream),
            bash_output_dedup_enabled=self._bash_output_dedup_enabled,
            bash_output_dedup_min_repeat=self._bash_output_dedup_min_repeat,
            bash_output_dedup_summary_prefix=self._bash_output_dedup_summary_prefix,
        )
        self._interaction_log_full = self._interaction_log_policy.tool_call_full
        self._interaction_log_llm_stream = self._interaction_log_policy.llm_stream_to_file
        self._bash_stream_to_interaction_log = bool(bash_stream_to_interaction_log)
        self.last_run_tokens_in: int = 0
        self.last_run_tokens_out: int = 0
        self.last_run_tokens_cached: int = 0
        self.last_run_llm_calls: int = 0
        self.last_run_compact_tokens_in: int = 0
        self.last_run_compact_tokens_out: int = 0
        self.last_run_compact_tokens_cached: int = 0
        self.last_run_compact_llm_calls: int = 0
        self.last_run_route_tokens_in: int = 0
        self.last_run_route_tokens_out: int = 0
        self.last_run_route_tokens_cached: int = 0
        self.last_run_route_llm_calls: int = 0
        self._run_tokens_in: int = 0
        self._run_tokens_out: int = 0
        self._run_tokens_cached: int = 0
        self._run_llm_calls: int = 0
        self._run_route_tokens_in: int = 0
        self._run_route_tokens_out: int = 0
        self._run_route_tokens_cached: int = 0
        self._run_route_llm_calls: int = 0
        self._run_compact_tokens_in: int = 0
        self._run_compact_tokens_out: int = 0
        self._run_compact_tokens_cached: int = 0
        self._run_compact_llm_calls: int = 0
        self._on_llm_call: Callable[[dict[str, Any]], None] | None = on_llm_call
        self._call_seq: int = 0
        self._current_round: int = 0
        self._round_budget_enabled: bool = True
        # Teleport: cumulative round count carried across ScienceAgent instances within
        # the same tree-search lineage. Each new ScienceAgent constructed for a fork
        # starts with offset = sum-of-rounds-in-prior-nodes; system prompt then shows
        # ``Round (offset + current)/(offset + max_steps)``. Default 0 = legacy behavior
        # (per-node counter starting at 1). Mutated by orchestrator at fork point or
        # seeded via the ``search_round_offset`` constructor kwarg (F3).
        self._search_round_offset: int = max(0, int(search_round_offset or 0))
        self._repl_session_run_index: int | None = None
        self._debug_dynamic_steps_enabled = bool(debug_dynamic_steps_enabled)
        self._debug_step_boost = int(debug_step_boost)
        self._debug_max_steps_cap = int(debug_max_steps_cap)
        self._round_budget_prompt_cap = int(round_budget_prompt_cap)
        self._effective_max_steps: int = int(self.max_steps)
        self._parallel_bash_enabled = bool(parallel_bash_enabled)
        self._parallel_llm_tool_calls = bool(parallel_llm_tool_calls)
        self._lnr_explore_streak_inject_after = int(lnr_explore_streak_inject_after)
        self._lnr_single_read_streak_inject_after = int(lnr_single_read_streak_inject_after)
        self._lnr_category_skill_text = str(lnr_category_skill_text or "")
        self._lnr_phase_skill_registry: dict[str, str] = dict(lnr_phase_skill_registry or {})
        self._lnr_phase_skill_descriptions: dict[str, str] = dict(
            lnr_phase_skill_descriptions or {},
        )
        self._lnr_injected_skills: set[str] = set()
        self._lnr_skill_inject_max_chars = int(lnr_skill_inject_max_chars)
        self._lnr_periodic_no_solution_every = int(lnr_periodic_no_solution_every)
        self._lnr_fresh_hint_injected: bool = False
        self._lnr_llm_turns_log_enabled = bool(lnr_llm_turns_log_enabled)
        self._mlebench_validate_after_embedded_full_run = bool(
            mlebench_validate_after_embedded_full_run,
        )
        _mdd = mlebench_data_dir
        self._mlebench_data_dir = str(_mdd).strip() if isinstance(_mdd, str) and _mdd.strip() else None
        _me = mlebench_exp_id
        self._mlebench_exp_id = str(_me).strip() if isinstance(_me, str) and _me.strip() else None
        self._lnr_mlebench_validate_enabled = bool(lnr_mlebench_validate_enabled)
        self._lnr_run_control_max_fix_rounds = int(lnr_run_control_max_fix_rounds)
        self._lnr_stop_after_bare_solution_success = bool(lnr_stop_after_bare_solution_success)
        self._lnr_result_md_after_success_pending = False
        self._lnr_superloop_enabled = bool(lnr_superloop_enabled)
        self._run_control_user_injections = 0
        self._lnr_snapshot_ok = False
        self._lnr_snapshot_reason = ""
        self._lnr_last_valid_solution_sha = ""
        self._lnr_last_valid_solution_rel_path = "solution.py"
        self._lnr_last_valid_submission_sha = ""
        self._lnr_last_valid_bash_cmd = ""
        self._lnr_last_valid_bare_run_is_local = False
        self._lnr_stage_journal_pending = False
        self._lnr_committed_ledger_step_count = None
        self._lnr_ledger_committed = False
        self._lnr_ledger_step_count = 0
        self._lnr_ledger_new_entries = 0
        self._lnr_stage_commit_guard_failed = False
        _nlp = lnr_llm_turns_log_path
        self._lnr_llm_turns_log_path: Path | None = (
            Path(_nlp).resolve() if _nlp else None
        )
        _sftp = sft_data_log_path
        self._sft_data_log_path: Path | None = (
            Path(_sftp).resolve() if _sftp else None
        )
        self._sandbox = bool(sandbox)
        self._path_guard_extra_roots = tuple(
            Path(str(p)).expanduser().resolve()
            for p in (path_guard_extra_roots or ())
            if str(p).strip()
        )
        self._recent_read_history: deque[tuple[str, str]] = deque(maxlen=3)
        self._last_read_sha_by_path: dict[str, str] = {}

        def _reset_solution_write_syntax_fail_counter() -> None:
            self._consecutive_write_syntax_fails = 0

        self._guard_manager = GuardManager(
            [
                WriteNudgeGuard(
                    get_current_solution_sha=self._sha256_of_solution,
                    get_initial_solution_sha=lambda: self._initial_solution_sha,
                    on_solution_write_success=_reset_solution_write_syntax_fail_counter,
                ),
                WriteFailureGuard(),
                WriteRepeatGuard(),
                RuntimeErrorGuard(
                    rotating_runtime_error_threshold=int(rotating_runtime_error_threshold),
                    rotating_distinct_types_min=int(rotating_distinct_types_min),
                ),
                EditFailureGuard(short_old_str_chars=int(edit_short_old_str_chars)),
                NoSuccessfulSolutionRunGuard(
                    soft_threshold=int(no_success_run_soft_threshold),
                    hard_threshold=int(no_success_run_hard_threshold),
                ),
                BashRepeatFailureGuard(),
                RepeatedRuntimeErrorGuard(
                    same_error_threshold=int(repeated_runtime_error_threshold),
                ),
                SingleReadStreakGuard(
                    threshold=int(lnr_single_read_streak_inject_after),
                    parallel_llm_tool_calls_enabled=bool(parallel_llm_tool_calls),
                ),
                BashPythonSourceDumpGuard(),
                ExploreStreakGuard(
                    threshold=int(lnr_explore_streak_inject_after),
                    get_current_solution_sha=self._sha256_of_solution,
                    get_initial_solution_sha=lambda: self._initial_solution_sha,
                    category_skill_text=self._lnr_category_skill_text,
                    max_skill_chars=self._lnr_skill_inject_max_chars,
                ),
                NoProgressHardStopGuard(
                    threshold=int(no_progress_hard_stop_after),
                    get_current_solution_sha=self._sha256_of_solution,
                    get_initial_solution_sha=lambda: self._initial_solution_sha,
                    get_embedded_metric_token=self._embedded_full_run_metric_token,
                    grace_rounds_after_peer_inject=int(no_progress_hardstop_guard_grace_rounds),
                ),
            ],
            phase_header=self._lnr_phase_header,
        )

    def _maybe_workspace_git_auto_checkpoint(
        self,
        tool_name: str,
        tool_result: ToolResult,
    ) -> None:
        """Checkpoint tracked source/docs after successful REPL tool activity."""
        if not bool(getattr(self, "_workspace_git_auto_checkpoint_enabled", False)):
            return
        try:
            from scienceflow.utils.workspace_git import auto_checkpoint_workspace_source

            result = auto_checkpoint_workspace_source(
                self._workspace_dir,
                enabled=True,
                track_globs=getattr(self, "_workspace_git_track_globs", ("*.py", "*.md")),
                tool_name=tool_name,
                tool_error=bool(tool_result.error),
            )
        except Exception as exc:
            self._log_warning("[workspace-git] auto checkpoint failed: %s", exc)
            return
        if not result.ready:
            if result.message:
                self._log_warning("[workspace-git] auto checkpoint unavailable: %s", result.message)
            return
        if result.committed or result.submission_snapshot:
            parts = []
            if result.committed:
                parts.append(f"commit={result.commit_sha[:12]}")
            if result.submission_snapshot:
                parts.append(f"submission={result.submission_snapshot}")
            if result.metric_value is not None:
                parts.append(f"metric={result.metric_value:.6g}")
            self._log_info("[workspace-git] auto checkpoint %s", " ".join(parts))

    def set_repl_session_run_index(self, n: int | None) -> None:
        """Set outer REPL session counter for ``[repl-run N]`` logs (None to disable)."""
        self._repl_session_run_index = n

    def _workspace_relative_path_mode_enabled(self) -> bool:
        return bool(getattr(self, "_workspace_relative_path_mode", False))

    def _path_hygiene_enabled(self) -> bool:
        return self._workspace_relative_path_mode_enabled() or (
            str(getattr(self, "_teleport_mode", "off") or "off") != "off"
        )

    @staticmethod
    def _host_abs_placeholder(value: str) -> str:
        raw = str(value or "")
        name = Path(raw.rstrip("/ ")).name or raw.strip("/") or "path"
        return f"<host-path:{name}>"

    @staticmethod
    def _rewrite_abs_prefix_to_label(text: str, root: Path, label: str) -> str:
        root_s = str(root).rstrip("/")
        if not root_s or root_s == "/" or root_s not in text:
            return text
        text = text.replace(root_s + "/", label.rstrip("/") + "/")
        text = text.replace(root_s, label.rstrip("/"))
        return text

    def _get_agent_hidden_workspace_filenames(self) -> tuple[str, ...]:
        raw = getattr(self, "_agent_hidden_workspace_filenames", ())
        if isinstance(raw, str):
            raw = (raw,)
        names: list[str] = []
        try:
            iterator = iter(raw)
        except TypeError:
            iterator = iter(())
        for item in iterator:
            name = Path(str(item or "").replace("\\", "/")).name.strip()
            if name and name not in names:
                names.append(name)
        return tuple(names)

    def _normalize_hidden_workspace_path(self, value: str) -> str:
        raw = str(value or "").replace("\\", "/").strip().strip("'\"")
        while raw.startswith("./"):
            raw = raw[2:]
        raw = raw.lstrip("/")
        return raw.rstrip("/")

    def _get_agent_hidden_workspace_path_prefixes(self) -> tuple[str, ...]:
        raw = getattr(self, "_agent_hidden_workspace_path_prefixes", ())
        if isinstance(raw, str):
            raw = (raw,)
        prefixes: list[str] = []
        try:
            iterator = iter(raw)
        except TypeError:
            iterator = iter(())
        for item in iterator:
            prefix = self._normalize_hidden_workspace_path(str(item or ""))
            if prefix and prefix not in prefixes:
                prefixes.append(prefix)
        return tuple(prefixes)

    def _apply_agent_hidden_path_denials(self, prefixes: Sequence[str | Path]) -> None:
        denied = [
            self._normalize_hidden_workspace_path(str(prefix or ""))
            for prefix in prefixes
        ]
        denied = [prefix for prefix in denied if prefix]
        tool_map = getattr(getattr(self, "availableTools", None), "tool_map", {}) or {}
        for tool in tool_map.values():
            if hasattr(tool, "path_guard_denied_prefixes"):
                try:
                    tool.path_guard_denied_prefixes = list(denied)
                except Exception:
                    continue

    def _text_mentions_hidden_workspace_prefix(self, text: str, prefix: str) -> bool:
        raw = str(text or "").replace("\\", "/")
        prefix = str(prefix or "").strip("/")
        if not raw or not prefix:
            return False
        escaped = re.escape(prefix)
        pattern = rf"(^|[\s'\"=;:&|(<>{{}}\[])(?:\./)?{escaped}(?=($|[\s'\"=;:&|/)<>}}\]]))"
        return re.search(pattern, raw) is not None

    def _hide_agent_hidden_workspace_filename_mentions(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        out = text
        for prefix in sorted(self._get_agent_hidden_workspace_path_prefixes(), key=len, reverse=True):
            escaped = re.escape(prefix)
            out = re.sub(
                rf"(?<![\w./-])(?:\./)?{escaped}(?=($|[\s'\"=;:&|/)<>}}\]]))",
                "<hidden-control-path>",
                out,
            )
        for name in self._get_agent_hidden_workspace_filenames():
            out = out.replace(name, "<hidden-control-file>")
        return out

    def _path_mentions_hidden_workspace_file(self, value: str) -> bool:
        raw = str(value or "").replace("\\", "/").strip().strip("'\"")
        if not raw:
            return False
        norm = self._normalize_hidden_workspace_path(raw)
        for prefix in self._get_agent_hidden_workspace_path_prefixes():
            if norm == prefix or norm.startswith(prefix + "/") or norm.endswith("/" + prefix) or f"/{prefix}/" in norm:
                return True
        for name in self._get_agent_hidden_workspace_filenames():
            if norm == name or norm.endswith("/" + name) or raw == name or raw.endswith("/" + name):
                return True
        return False

    def _tool_request_mentions_hidden_workspace_file(self, tool_name: str, args: dict[str, Any]) -> bool:
        prefixes = self._get_agent_hidden_workspace_path_prefixes()
        names = self._get_agent_hidden_workspace_filenames()
        if not prefixes and not names:
            return False
        if tool_name in {"read", "write", "edit", "ls", "grep"}:
            if self._path_mentions_hidden_workspace_file(str(args.get("path") or "")):
                return True
        if tool_name == "bash":
            cmd = str(args.get("command") or "")
            if any(self._text_mentions_hidden_workspace_prefix(cmd, prefix) for prefix in prefixes):
                return True
            return any(name in cmd for name in names)
        return False

    def _hide_agent_hidden_workspace_file_lines(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        names = self._get_agent_hidden_workspace_filenames()
        prefixes = self._get_agent_hidden_workspace_path_prefixes()
        if not names and not prefixes:
            return text
        kept = []
        removed = False
        for line in text.splitlines():
            if any(name in line for name in names) or any(self._text_mentions_hidden_workspace_prefix(line, prefix) for prefix in prefixes):
                removed = True
                continue
            kept.append(line)
        if removed:
            kept.append("[hidden control file entries omitted]")
        return "\n".join(kept)

    def _sanitize_agent_visible_paths(self, text: str) -> str:
        """Rewrite model-visible host paths to workspace-relative or opaque labels."""
        if not isinstance(text, str) or not text:
            return text
        if not self._path_hygiene_enabled():
            return self._hide_agent_hidden_workspace_filename_mentions(text)
        from scienceflow.core.agent_runtime import rewrite_workspace_abs_to_relative

        out = rewrite_workspace_abs_to_relative(text, str(self._workspace_dir))
        out = _WSP_ABSOLUTE_VISIBLE_PREFIX_RE.sub("./", out)

        dataset_link = self._workspace_dir / "dataset"
        try:
            if dataset_link.exists():
                out = self._rewrite_abs_prefix_to_label(out, dataset_link.resolve(), "dataset")
        except OSError:
            pass

        if self._workspace_relative_path_mode_enabled():
            out = _HOST_ABSOLUTE_VISIBLE_PATH_RE.sub(
                lambda m: self._host_abs_placeholder(m.group(0)),
                out,
            )
        return self._hide_agent_hidden_workspace_filename_mentions(out)

    def _sanitize_message_agent_visible_paths(self, message: Message) -> Message:
        """Sanitize message content and assistant tool-call arguments before memory use."""
        if not self._path_hygiene_enabled():
            return message
        from deepcraft_core.tool.base import Function as ToolFunction

        updates: dict[str, Any] = {}
        if isinstance(message.content, str):
            content = self._sanitize_agent_visible_paths(message.content)
            if content != message.content:
                updates["content"] = content
        if message.tool_calls:
            new_calls = []
            changed = False
            for tc in message.tool_calls:
                args = tc.function.arguments
                new_args = self._sanitize_agent_visible_paths(args) if isinstance(args, str) else args
                if new_args != args:
                    changed = True
                    new_calls.append(
                        tc.model_copy(
                            update={
                                "function": ToolFunction(
                                    name=tc.function.name,
                                    arguments=new_args,
                                ),
                            },
                        ),
                    )
                else:
                    new_calls.append(tc)
            if changed:
                updates["tool_calls"] = new_calls
        if not updates:
            return message
        return message.model_copy(update=updates)

    def _add_assistant_api_message(self, assistant_msg: Any) -> None:
        self.memory.add_message(
            self._sanitize_message_agent_visible_paths(_assistant_message_from_api(assistant_msg)),
        )

    def sanitize_existing_memory_agent_visible_paths(self) -> int:
        """Rewrite already-loaded chat memory for opt-in workspace-relative path mode."""
        if not self._path_hygiene_enabled():
            return 0
        try:
            records = self.memory.chat_history_memory.retrieve(window_size=None)
        except Exception:
            return 0
        if not records:
            return 0
        messages = []
        changed = False
        for record in records:
            try:
                message = record.memory_record.message
            except Exception:
                continue
            new_message = self._sanitize_message_agent_visible_paths(message)
            if new_message is not message:
                changed = True
            messages.append(new_message)
        if not changed or not messages:
            return 0
        try:
            rewrite = getattr(getattr(self, "_memory_ctx", None), "rewrite_messages", None)
            if callable(rewrite):
                rewrite(messages)
            else:
                self.memory.chat_history_memory.storage.clear()
                for message in messages:
                    self.memory.add_message(message)
            return len(messages)
        except Exception:
            return 0

    def _maybe_normalize_abs_path_to_workspace_relative(self, path_value: str) -> str | None:
        try:
            p = Path(path_value).expanduser()
        except (TypeError, ValueError):
            return None
        if not p.is_absolute():
            return None
        try:
            rel = p.resolve(strict=False).relative_to(self._workspace_dir)
            return rel.as_posix() or "."
        except (OSError, ValueError):
            pass
        dataset_link = self._workspace_dir / "dataset"
        try:
            if dataset_link.exists():
                rel = p.resolve(strict=False).relative_to(dataset_link.resolve())
                return f"dataset/{rel.as_posix()}" if rel.as_posix() else "dataset"
        except (OSError, ValueError):
            pass
        return None

    def _maybe_normalize_tool_input_paths(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Normalize model-emitted path args to workspace-relative form when enabled."""
        if not isinstance(args, dict):
            return args
        if not self._path_hygiene_enabled():
            return args
        path_val = args.get("path")
        if isinstance(path_val, str):
            new_path: str | None = None
            if path_val.startswith("/workspace/"):
                new_path = "./" + path_val[len("/workspace/") :]
            elif path_val == "/workspace":
                new_path = "."
            elif self._workspace_relative_path_mode_enabled():
                new_path = self._maybe_normalize_abs_path_to_workspace_relative(path_val)
            if new_path is not None and new_path != path_val:
                args = dict(args)
                args["path"] = new_path
        return args

    def _maybe_rewrite_tool_result_paths(self, tool_result: ToolResult) -> ToolResult:
        """Rewrite model-visible tool output paths to stable relative forms."""
        if tool_result is None or not self._path_hygiene_enabled():
            return tool_result
        out = tool_result.output
        new_out = self._sanitize_agent_visible_paths(out) if isinstance(out, str) else out
        err = tool_result.error
        new_err = self._sanitize_agent_visible_paths(err) if isinstance(err, str) else err
        if new_out is out and new_err is err:
            return tool_result
        # Mutate in place when possible (ToolResult is pydantic; fall back to copy)
        try:
            tool_result.output = new_out  # type: ignore[assignment]
            tool_result.error = new_err  # type: ignore[assignment]
            return tool_result
        except Exception:
            try:
                return tool_result.model_copy(update={"output": new_out, "error": new_err})
            except Exception:
                return tool_result

    def swap_workspace(
        self,
        *,
        workspace_dir: str | Path,
        path_guard_extra_roots: Sequence[str | Path] | None = None,
        readonly_dirs: Sequence[str] | None = None,
        run_policy: RunPolicy | None = None,
        max_steps_override: int | None = None,
        bash_timeout_sec: float | None = None,
        bash_timeout_slow_sec: float | None = None,
        extra_env: dict[str, str] | None = None,
        skill_registry: SkillRegistry | None = None,
        task_type: str | None = None,
        skill_allow_names: Sequence[str] | None = None,
        skill_tool_mode: str | None = None,
        skill_allow_generic_wildcard: bool | None = None,
        skill_visible_max: int | None = None,
        resource_observer: Any | None = None,
        interaction_log_color: bool | None = None,
        interaction_log_session: str | None = None,
        interaction_log_phase: str | None = None,
        copy_memory_storage_to_new_workspace: bool = True,
        sliding_window_budget_chars: int | None = None,
        lnr_llm_turns_log_path: str | Path | None = None,
        sft_data_log_path: str | Path | None = None,
        on_llm_call: Callable[[dict[str, Any]], None] | None | object = _ON_LLM_CALL_UNSET,
    ) -> None:
        """Teleport: swap the underlying workspace **without** reconstructing the agent.

        Used when *teleport_mode* is on for **A-class fork** continuations
        (same-branch / explore↔exploit_explore). The LLM's view (``systemPrompt``,
        ``_system_prompt_hook``, ``memory`` chat history, all ``_consecutive_*``
        counters, pinned messages) is preserved; only orchestrator-side state
        (workspace dir, tool collection, path guard, readonly dirs, run policy,
        interaction log file, optional max_steps / bash_timeout) is rebuilt.

        Round counter behavior: ``_current_round`` is added into
        ``_search_round_offset`` and reset to 0. In teleport mode this counter
        is used for logs and legacy callers, while the dynamic Round Budget is
        deliberately kept out of the system prompt for KV-cache stability.

        Memory storage is hot-swapped so future appends persist under the new
        node's ``.agent_memory/Draft/short_term.json`` instead of the old
        node's. The in-memory record list is preserved by copying records over.

        B/C-class forks should NOT use this method — they want a fresh
        ScienceAgent with deep-copied parent memory + functional first-user msg
        (see :func:`run_agent_session_clone` teleport branch).
        """
        # Late imports to avoid widening module-import surface.
        from deepcraft_core.storage.kv_storage import JsonKeyValueStorage

        new_ws = Path(workspace_dir).resolve()
        old_ws = self._workspace_dir
        old_round = int(self._current_round)
        # 1. Round budget bookkeeping: accumulate prior counter into search-wide offset.
        self._search_round_offset = int(getattr(self, "_search_round_offset", 0) or 0) + old_round
        self._current_round = 0
        # Trace hooks are node-scoped. Persistent A_TELEPORT agents keep the
        # Python object, so explicitly rebind the hook and restart call_seq for
        # the child node when the caller supplies a new hook.
        self._call_seq = 0
        if on_llm_call is not _ON_LLM_CALL_UNSET:
            self._on_llm_call = on_llm_call  # type: ignore[assignment]
        if max_steps_override is not None:
            self.max_steps = int(max_steps_override)
            self._effective_max_steps = int(max_steps_override)

        # 2. Tool collection: rebuild with new workspace + (optionally new) extra roots / readonly_dirs.
        new_extra_roots = (
            list(path_guard_extra_roots) if path_guard_extra_roots is not None else None
        )
        new_readonly = list(readonly_dirs) if readonly_dirs is not None else None
        new_extra_env = dict(extra_env) if extra_env is not None else self._extra_env
        if skill_registry is None:
            skill_registry = getattr(self, "_skill_registry", None)
        if task_type is None:
            task_type = getattr(self, "_task_type", None)
        if skill_allow_names is None:
            skill_allow_names = getattr(self, "_skill_allow_names", ())
        if skill_tool_mode is None:
            skill_tool_mode = getattr(self, "_skill_tool_mode", "all")
        if skill_allow_generic_wildcard is None:
            skill_allow_generic_wildcard = getattr(self, "_skill_allow_generic_wildcard", True)
        if skill_visible_max is None:
            skill_visible_max = getattr(self, "_skill_visible_max", 0)
        if resource_observer is None:
            resource_observer = getattr(self, "_resource_observer", None)
        if bash_timeout_sec is not None:
            self._bash_timeout_sec = float(bash_timeout_sec)
        if bash_timeout_slow_sec is not None:
            self._bash_timeout_slow_sec = float(bash_timeout_slow_sec)
        elif not hasattr(self, "_bash_timeout_slow_sec"):
            self._bash_timeout_slow_sec = float(self._bash_timeout_sec)
        old_display_root = getattr(self, "_tool_display_root", old_ws)
        old_display_root_path = Path(old_display_root).resolve()
        new_display_root = new_ws if old_display_root_path == old_ws else old_display_root_path
        new_tools = create_tool_collection(
            new_ws,
            sandbox=True,
            path_guard_extra_roots=new_extra_roots,
            max_bash_output_chars=int(getattr(self, "_bash_max_output_chars", 8000)),
            max_bash_stream_line_chars=int(
                getattr(self, "_bash_max_stream_line_chars", 2400),
            ),
            bash_observation_summary_enabled=bool(
                getattr(self, "_bash_observation_summary_enabled", False),
            ),
            bash_timeout_sec=float(self._bash_timeout_sec),
            bash_timeout_slow_sec=float(self._bash_timeout_slow_sec),
            extra_env=new_extra_env,
            readonly_dirs=new_readonly,
            resource_observer=resource_observer,
            skill_registry=skill_registry,
            task_type=task_type,
            skill_allow_names=skill_allow_names,
            skill_tool_mode=str(skill_tool_mode or "all"),
            skill_allow_generic_wildcard=bool(skill_allow_generic_wildcard),
            skill_visible_max=int(skill_visible_max or 0),
            include_write_edit_tools=bool(
                getattr(self, "_include_write_edit_tools", True),
            ),
            tool_display_paths_relative=bool(
                getattr(self, "_tool_display_paths_relative", True),
            ),
            tool_display_root=new_display_root,
            grep_max_results_lines=int(getattr(self, "_grep_max_results_lines", 50)),
        )
        self.availableTools = new_tools
        self._tools_with_thought = inject_thought_into_tool_params(
            self.availableTools.to_params(),
        )
        self._workspace_dir = new_ws
        self._extra_env = new_extra_env
        self._skill_registry = skill_registry
        self._task_type = task_type
        self._skill_allow_names = tuple(skill_allow_names or ())
        self._skill_tool_mode = str(skill_tool_mode or "all")
        self._skill_allow_generic_wildcard = bool(skill_allow_generic_wildcard)
        self._skill_visible_max = int(skill_visible_max or 0)
        self._resource_observer = resource_observer
        self._tool_display_root = new_display_root
        self._tool_output_artifacts = self._make_tool_output_artifact_store(new_ws)
        self._resource_feedback_memory_deduper = ResourceFeedbackMemoryDeduper()

        # 3. MemoryContextManager workspace + path guard.
        if hasattr(self, "_memory_ctx") and self._memory_ctx is not None:
            self._memory_ctx._workspace = new_ws
            if sliding_window_budget_chars is not None:
                self._memory_ctx._budget_chars = max(
                    0,
                    int(sliding_window_budget_chars),
                )
                self._memory_ctx._last_window_k = 0
            try:
                self._memory_ctx._guard = PathGuard(
                    new_ws,
                    extra_roots=new_extra_roots,
                )
            except Exception:
                # PathGuard mismatch is non-fatal; tools have their own guards.
                pass

        # 4. Memory storage hot-swap so future appends land in the new ``.agent_memory``.
        if (
            copy_memory_storage_to_new_workspace
            and getattr(self, "memory", None) is not None
            and getattr(self.memory, "chat_history_memory", None) is not None
        ):
            try:
                from scienceflow.core.agent_runtime import (
                    select_prefix_safe_agent_memory_records,
                    write_agent_memory_record_files,
                )

                storage = self.memory.chat_history_memory.storage
                old_storage_path = Path(
                    str(
                        getattr(storage, "json_path", None)
                        or getattr(storage, "path", None)
                        or ""
                    )
                )
                old_agent_dir = old_storage_path.parent if str(old_storage_path) else None
                max_messages = int(getattr(self.memory, "max_messages", 0) or 0)
                if old_agent_dir is not None and old_agent_dir.is_dir():
                    old_records = select_prefix_safe_agent_memory_records(
                        old_agent_dir,
                        max_messages=max_messages,
                    )
                else:
                    old_records = list(storage.load() or [])
            except Exception:
                old_records = []
            new_agent_dir = new_ws / ".agent_memory" / "Draft"
            new_st = new_agent_dir / "short_term.json"
            new_agent_dir.mkdir(parents=True, exist_ok=True)
            try:
                if old_records:
                    write_agent_memory_record_files(
                        new_agent_dir,
                        old_records,
                        write_long_term=True,
                    )
                else:
                    new_st.touch(exist_ok=True)
                new_storage = JsonKeyValueStorage(path=str(new_st), mode="a")
                self.memory.chat_history_memory.storage = new_storage
                # Re-calibrate _num_records so future ``exceed_memory_limit`` decisions match.
                self.memory.chat_history_memory._num_records = len(old_records)
            except Exception:
                _logger.warning(
                    "[teleport] swap_workspace memory storage swap failed; keeping old storage",
                    exc_info=True,
                )

        # 5. Run policy.
        if run_policy is not None:
            self._run_policy = run_policy

        # 6. Interaction log: re-attach to new workspace logs/.
        try:
            self._ws_interaction_log = attach_workspace_interaction_logger(
                new_ws,
                color=bool(interaction_log_color)
                if interaction_log_color is not None
                else False,
                layout=getattr(self, "_interaction_log_layout", "flat"),
            )
        except Exception:
            _logger.warning("[teleport] interaction-log re-attach failed", exc_info=True)
        if interaction_log_session is not None:
            self._lnr_interaction_log_session = (
                (interaction_log_session or "draft").strip().lower() or "draft"
            )
        if interaction_log_phase is not None:
            _ilp = str(interaction_log_phase).strip().lower()
            self._lnr_interaction_log_phase = _ilp or None
        if interaction_log_session is not None or interaction_log_phase is not None:
            try:
                _ilog_tag = build_interaction_log_context_tag(
                    self._lnr_interaction_log_session,
                    self._lnr_interaction_log_phase,
                )
                self._interaction_log_ctx_token = set_interaction_log_context_tag(_ilog_tag)
            except Exception:
                pass

        if lnr_llm_turns_log_path is not None:
            self._lnr_llm_turns_log_path = (
                Path(lnr_llm_turns_log_path).resolve()
                if lnr_llm_turns_log_path
                else None
            )
        if sft_data_log_path is not None:
            self._sft_data_log_path = (
                Path(sft_data_log_path).resolve()
                if sft_data_log_path
                else None
            )

        # 7. Per-node embedded full-run state - clear so the new node's safety policy can run.
        self._embedded_full_run_done = False
        self._lnr_snapshot_ok = False
        self._lnr_snapshot_reason = ""
        self._lnr_last_valid_solution_sha = ""
        self._lnr_last_valid_solution_rel_path = "solution.py"
        self._lnr_last_valid_submission_sha = ""
        self._lnr_last_valid_bash_cmd = ""
        self._lnr_last_valid_bare_run_is_local = False
        self._lnr_committed_ledger_step_count = None
        self._lnr_ledger_committed = False
        self._lnr_ledger_step_count = 0
        self._lnr_ledger_new_entries = 0
        self._lnr_stage_commit_guard_failed = False
        self._lnr_stage_journal_pending = False
        self._run_control_user_injections = 0
        self._lnr_result_md_after_success_pending = False
        # A_TELEPORT continuations run in clone-style mode. Fresh clone agents already
        # disable mid-run LLM compaction; persistent agents must do the same here or a
        # child can stall inside compact(mid_run=True) after the inherited transcript
        # barely exceeds the sliding-window budget. The fork-boundary compression +
        # priority sliding window are the intended budget controls for teleport.
        self._mid_run_compact_enabled = False
        self._mid_run_compacted = False

        _logger.info(
            "[teleport] swap_workspace: %s -> %s (round_offset=%d -> %d)",
            old_ws,
            new_ws,
            self._search_round_offset - old_round,
            self._search_round_offset,
        )


    def _make_tool_output_artifact_store(
        self,
        workspace_dir: str | Path,
        *,
        log_dir_override: str | Path | None = None,
    ) -> ToolOutputArtifactStore:
        if getattr(self, "_interaction_log_layout", "flat") == "split":
            return ToolOutputArtifactStore(
                workspace_dir,
                output_parts=("interaction", "tool_outputs"),
                mirror_parts=("traj_interaction", "tool_outputs"),
                log_dir_override=log_dir_override,
            )
        return ToolOutputArtifactStore(workspace_dir, log_dir_override=log_dir_override)

    def _prepare_tool_feedback_for_memory(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_result: ToolResult,
        *,
        guard_coaching: str = "",
    ) -> str:
        """Build final tool feedback for chat memory and persist raw output artifact.

        Bash streaming is already complete before this method is called; this only
        touches the final ToolResult text that would otherwise be written to memory.
        """
        raw_text = raw_tool_result_text(tool_result)
        hidden_request = self._tool_request_mentions_hidden_workspace_file(tool_name, args)
        ref = None
        store = getattr(self, "_tool_output_artifacts", None)
        if store is not None and raw_text:
            ref = store.reserve(tool_name, raw_text)
            if ref is not None and not store.write_raw(ref, raw_text):
                ref = None

        if hidden_request:
            feedback = "[hidden control file output omitted from agent memory]"
        else:
            feedback = self._memory_ctx.record_tool_result(
                tool_name,
                args,
                tool_result,
                raw_id=ref.raw_id if ref is not None else "",
            )
        if guard_coaching:
            feedback = feedback + "\n\n" + guard_coaching

        feedback, reducer = reduce_tool_feedback_for_memory(
            tool_name=tool_name,
            args=args,
            feedback=feedback,
            raw_text=raw_text,
            tool_error=bool(getattr(tool_result, "error", None)),
        )
        feedback, resource_deduped = self._dedup_resource_feedback_for_memory(feedback)
        if resource_deduped:
            reducer = f"{reducer}+resource_feedback_dedup"

        if ref is not None:
            feedback = attach_tool_output_reference(
                feedback,
                raw_id=ref.raw_id,
                raw_chars=ref.raw_chars,
                reducer=reducer,
            )

        feedback = trim_tool_feedback_for_llm_context(
            feedback,
            max_chars=self._exec_feedback_max_chars,
        )
        if not hidden_request:
            feedback = self._hide_agent_hidden_workspace_file_lines(feedback)
        feedback = self._hide_agent_hidden_workspace_filename_mentions(feedback)

        if ref is not None:
            store.append_index(
                ref,
                reducer_name=reducer,
                compressed_chars=len(feedback),
            )
            append_stage_index = getattr(store, "append_stage_index", None)
            if callable(append_stage_index):
                append_stage_index(
                    ref,
                    args=args,
                    reducer_name=reducer,
                    compressed_chars=len(feedback),
                    tool_error=bool(getattr(tool_result, "error", None)),
                )

        return feedback

    def _dedup_resource_feedback_for_memory(self, feedback: str) -> tuple[str, bool]:
        deduper = getattr(self, "_resource_feedback_memory_deduper", None)
        if deduper is None:
            deduper = ResourceFeedbackMemoryDeduper()
            self._resource_feedback_memory_deduper = deduper
        return deduper.reduce(feedback)

    def _log_iteration_header(self, round_one_based: int) -> None:
        # Teleport mode F5: when ``_search_round_offset > 0`` the search-wide round
        # counter shown to the LLM is ``round_one_based + offset`` / ``max_steps + offset``.
        # Mirror that into the audit log so external readers (interaction.log /
        # scienceflow.log) see the same monotone series rather than a confusing
        # per-node ``Iteration 1/50`` reset.
        _offset = int(getattr(self, "_search_round_offset", 0) or 0)
        _eff_total = self._effective_max_steps + _offset
        _disp_round = round_one_based + _offset
        if self._repl_session_run_index is not None:
            self._log_info(
                "[repl-run %d] ====== Agent Iteration %d/%d =======",
                self._repl_session_run_index,
                _disp_round,
                _eff_total,
            )
        elif _offset > 0:
            self._log_info(
                "====== Agent Iteration %d/%d (search round) =======",
                _disp_round,
                _eff_total,
            )
        else:
            self._log_info(
                "====== Agent Iteration %d/%d =======",
                round_one_based,
                self._effective_max_steps,
            )

    def _sanitize_log_format(self, msg: str, args: tuple[object, ...]) -> tuple[str, tuple[object, ...]]:
        if not self._workspace_relative_path_mode_enabled():
            return msg, args
        try:
            rendered = (msg % args) if args else str(msg)
        except Exception:
            rendered = " ".join([str(msg), *(str(a) for a in args)])
        return "%s", (self._sanitize_agent_visible_paths(rendered),)

    def _log_info(self, msg: str, *args: object) -> None:
        msg, args = self._sanitize_log_format(msg, args)
        _logger.info(msg, *args)
        if self._ws_interaction_log is not None:
            self._ws_interaction_log.info(msg, *args)

    def _log_warning(self, msg: str, *args: object) -> None:
        msg, args = self._sanitize_log_format(msg, args)
        _logger.warning(msg, *args)
        if self._ws_interaction_log is not None:
            self._ws_interaction_log.warning(msg, *args)

    def _pop_and_log_thought(self, args: dict[str, Any]) -> None:
        """Remove ``thought`` from tool args (not passed to tools) and log it."""

        raw = args.pop("thought", None)
        if raw is None:
            return
        s = str(raw).strip()
        if s:
            self._log_info("[thought] %s", s)

    def _next_step(self) -> int:
        self._ui_step += 1
        return self._ui_step

    def _sync_last_run_token_totals(self) -> None:
        """Copy cumulative run counters to ``last_run_*`` (incl. compact subset)."""
        self.last_run_tokens_in = self._run_tokens_in
        self.last_run_tokens_out = self._run_tokens_out
        self.last_run_tokens_cached = self._run_tokens_cached
        self.last_run_llm_calls = self._run_llm_calls
        self.last_run_route_tokens_in = self._run_route_tokens_in
        self.last_run_route_tokens_out = self._run_route_tokens_out
        self.last_run_route_tokens_cached = self._run_route_tokens_cached
        self.last_run_route_llm_calls = self._run_route_llm_calls
        self.last_run_compact_tokens_in = self._run_compact_tokens_in
        self.last_run_compact_tokens_out = self._run_compact_tokens_out
        self.last_run_compact_tokens_cached = self._run_compact_tokens_cached
        self.last_run_compact_llm_calls = self._run_compact_llm_calls

    def _accumulate_compact_llm_into_run_counters(self) -> None:
        """After ``MemoryContextManager.compact`` (``llm.ask``), fold tokens into run totals."""
        try:
            ti = getattr(self.llm, "_last_call_input_tokens", None)
            to = getattr(self.llm, "_last_call_output_tokens", None)
            tc = getattr(self.llm, "_last_call_input_cached_tokens", None)
            if ti is not None:
                v = int(ti or 0)
                self._run_tokens_in += v
                self._run_compact_tokens_in += v
            if to is not None:
                v = int(to or 0)
                self._run_tokens_out += v
                self._run_compact_tokens_out += v
            if tc is not None:
                v = int(tc or 0)
                self._run_tokens_cached += v
                self._run_compact_tokens_cached += v
            self._run_llm_calls += 1
            self._run_compact_llm_calls += 1
        except (TypeError, ValueError):
            pass

    @property
    def file_state_summary(self) -> str:
        s = self._memory_ctx.file_state_summary
        return s if s else "(no files tracked)"

    def pin_message(self, msg: Message) -> None:
        """Pin an L1 user message so it is always included before the sliding window."""
        self._memory_ctx.pin_message(msg)

    def _sha256_of_solution(self) -> str | None:
        """Hex digest of workspace ``solution.py`` if present; else None."""
        p = self._workspace_dir / "solution.py"
        if not p.is_file():
            return None
        try:
            h = hashlib.sha256()
            h.update(p.read_bytes())
            return h.hexdigest()
        except OSError:
            return None

    def _sha256_of_workspace_file(self, rel_path: str) -> str | None:
        p = self._workspace_dir / rel_path
        if not p.is_file():
            return None
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            return None

    def _normalize_valid_run_source_rel_path(self, rel_path: str | None = None) -> str:
        raw = str(rel_path or "solution.py").replace("\\", "/").strip().lstrip("/")
        if not raw:
            return "solution.py"
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts:
            return "solution.py"
        return str(path)

    def _lnr_mark_valid_bare_run(self, *, bash_cmd: str = "", solution_rel_path: str | None = None) -> None:
        """Bind the latest run-control-ready run to the current source/submission files."""
        source_rel = self._normalize_valid_run_source_rel_path(solution_rel_path)
        self._lnr_last_valid_solution_sha = self._sha256_of_workspace_file(source_rel) or ""
        self._lnr_last_valid_solution_rel_path = source_rel
        self._lnr_last_valid_submission_sha = (
            self._sha256_of_workspace_file("submission.csv") or ""
        )
        self._lnr_last_valid_bash_cmd = str(bash_cmd or "")
        self._lnr_snapshot_ok = True
        self._lnr_last_valid_bare_run_is_local = True

    def _lnr_restore_valid_bare_run_from_snapshot(self) -> bool:
        """Restore a clone-carried run-control-ready run when file hashes still match."""
        from scienceflow.utils.node_paths import find_node_log_path

        p = find_node_log_path(self._workspace_dir, "fullrun_tail_snapshot.json")
        if not p.is_file():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return False
        if not isinstance(data, dict):
            return False
        if data.get("exit_code") not in (0, "0"):
            return False
        if data.get("validation_ok") is False:
            return False
        if str(data.get("submission_status") or "") == "pending_evaluator":
            return False
        try:
            metric = float(data.get("metric_value"))
        except (TypeError, ValueError):
            return False
        if metric != metric or metric in (float("inf"), float("-inf")):
            return False
        expected_solution = str(data.get("solution_sha") or "").strip()
        if not expected_solution:
            return False
        source_rel = self._normalize_valid_run_source_rel_path(data.get("solution_path") or "solution.py")
        current_solution = self._sha256_of_workspace_file(source_rel) or ""
        if current_solution != expected_solution:
            return False
        expected_submission = str(data.get("submission_sha") or "").strip()
        current_submission = self._sha256_of_workspace_file("submission.csv") or ""
        if expected_submission and current_submission != expected_submission:
            return False
        self._lnr_last_valid_solution_sha = current_solution
        self._lnr_last_valid_solution_rel_path = source_rel
        self._lnr_last_valid_submission_sha = current_submission
        self._lnr_last_valid_bash_cmd = str(data.get("bash_cmd") or "")
        self._lnr_snapshot_ok = True
        self._lnr_snapshot_reason = "restored_gate_tail_snapshot"
        self._lnr_last_valid_bare_run_is_local = False
        return True

    def _lnr_has_current_valid_bare_run(self) -> bool:
        """True only if current files still match the last run-control-ready bare run."""
        if (
            not bool(getattr(self, "_lnr_snapshot_ok", False))
            or not str(getattr(self, "_lnr_last_valid_solution_sha", "") or "")
        ):
            self._lnr_restore_valid_bare_run_from_snapshot()
        if not bool(getattr(self, "_lnr_snapshot_ok", False)):
            return False
        expected_solution = str(getattr(self, "_lnr_last_valid_solution_sha", "") or "")
        # Backward-compatible for tests/legacy agents constructed before this stamp existed.
        if not hasattr(self, "_lnr_last_valid_solution_sha"):
            return True
        if not expected_solution:
            return False
        source_rel = self._normalize_valid_run_source_rel_path(
            getattr(self, "_lnr_last_valid_solution_rel_path", "solution.py"),
        )
        if (self._sha256_of_workspace_file(source_rel) or "") != expected_solution:
            return False
        expected_submission = str(getattr(self, "_lnr_last_valid_submission_sha", "") or "")
        if expected_submission and (
            (self._sha256_of_workspace_file("submission.csv") or "") != expected_submission
        ):
            return False
        return True

    def _lnr_invalidate_current_valid_run(self, reason: str) -> None:
        self._lnr_snapshot_ok = False
        self._lnr_snapshot_reason = reason
        self._lnr_last_valid_solution_sha = ""
        self._lnr_last_valid_solution_rel_path = "solution.py"
        self._lnr_last_valid_submission_sha = ""
        self._lnr_last_valid_bash_cmd = ""
        self._lnr_last_valid_bare_run_is_local = False
        self._lnr_stage_journal_pending = False
        self._lnr_result_md_after_success_pending = False
        result_md = self._workspace_dir / "result.md"
        if result_md.exists():
            try:
                result_md.unlink()
                _logger.info("[run-control] removed stale result.md after %s", reason)
            except OSError as exc:
                _logger.warning(
                    "[run-control] failed to remove stale result.md after %s: %s",
                    reason,
                    exc,
                )

    def _lnr_adjust_policy_injection(self, msg: str) -> str:
        if (
            bool(getattr(self, "_lnr_stop_after_bare_solution_success", False))
            and "result.md does not exist yet" in (msg or "")
            and not self._lnr_has_current_valid_bare_run()
        ):
            return (
                "result.md does not exist yet, but the current `solution.py` has no "
                "matching accepted full run in this node. First modify `solution.py` "
                "if needed, then run `python3 solution.py` with the full-training path "
                "active. Only after run-control checks pass should you write result.md. "
                "Use a tool NOW."
            )
        return msg

    def _embedded_full_run_metric_token(self) -> str | None:
        """Stable token for successful embedded full-run metric (if available)."""
        from scienceflow.utils.node_paths import find_node_log_path

        p = find_node_log_path(self._workspace_dir, "embedded_full_run_result.json")
        if not p.is_file():
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        mv = payload.get("metric_value")
        if mv is None:
            return None
        mn = str(payload.get("metric_name") or "").strip() or "metric"
        return f"{mn}:{mv}"

    def _write_productivity_snapshot(self) -> None:
        """Persist per-run productivity counters for solver-side node triage."""
        guard_mgr = getattr(self, "_guard_manager", None)
        if guard_mgr is None:
            return
        counts = guard_mgr.productivity_counters()
        write_n = int(counts.get("write_success_count", 0))
        edit_n = int(counts.get("edit_success_count", 0))
        payload = {
            "write_count": write_n,
            "edit_count": edit_n,
            "write_success_count": write_n,
            "edit_success_count": edit_n,
            "initial_solution_sha": self._initial_solution_sha,
            "final_solution_sha": self._sha256_of_solution(),
            "had_metric": self._embedded_full_run_metric_token() is not None,
        }
        out = self._workspace_dir / ".agent_memory" / "productivity.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _reset_edit_read_guard_state(self) -> None:
        """Clear read fingerprints for :class:`EditFailureGuard` hard re-read policy."""
        self._recent_read_history.clear()
        self._last_read_sha_by_path.clear()

    @staticmethod
    def _normalize_rel_workspace_path(path: str) -> str:
        return str(path or "").replace("\\", "/").lstrip("/")

    def _resolve_workspace_path(self, rel: str) -> Path | None:
        try:
            return PathGuard(
                self._workspace_dir,
                enabled=self._sandbox,
                extra_roots=self._path_guard_extra_roots,
            ).resolve(rel)
        except ValueError:
            return None

    def _record_read_for_edit_guard(self, rel: str) -> None:
        norm = self._normalize_rel_workspace_path(rel)
        resolved = self._resolve_workspace_path(norm)
        if resolved is None:
            return
        sha = self._sha256_file(resolved)
        if sha is None:
            return
        self._last_read_sha_by_path[norm] = sha
        self._recent_read_history.append((norm, sha))

    def _record_successful_file_view_for_edit_guard(
        self,
        name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> None:
        """Treat successful read/write/edit results as a fresh file view for edit safety."""
        if result.error or name not in {"read", "write", "edit"}:
            return
        path = str(args.get("path") or "")
        if path:
            self._record_read_for_edit_guard(path)

    def seed_edit_guard(self, rel: str) -> None:
        """Pre-seed read-before-edit guard for a path (e.g. clone parent solution on disk)."""
        self._record_read_for_edit_guard(rel)

    def _should_block_edit_for_stale_read(self, rel: str) -> bool:
        """True when ``edit`` must be blocked until a successful ``read`` matches on-disk content."""
        norm = self._normalize_rel_workspace_path(rel)
        resolved = self._resolve_workspace_path(norm)
        if resolved is None:
            return True
        cur = self._sha256_file(resolved)
        if cur is None:
            return True
        if cur == self._last_read_sha_by_path.get(norm):
            return False
        for p, s in self._recent_read_history:
            if p == norm and s == cur:
                return False
        return True

    async def _execute_tool_maybe_edit_guard(
        self,
        name: str,
        args: dict[str, Any],
    ) -> ToolResult:
        """Run non-bash tools; block ``edit`` until a recent ``read`` matches file contents."""
        # Teleport: normalize defensive ``/workspace/...`` prefix to ``./...`` so the
        # tool layer (rooted at the current node directory) sees a workable path.
        args = self._maybe_normalize_tool_input_paths(name, args)
        if name == "edit":
            rel = str(args.get("path") or "")
            if self._should_block_edit_for_stale_read(rel):
                return ToolResult(
                    error=(
                        f"Edit blocked: re-read `{rel}` (full or around target) first; "
                        "file changed or not recently read."
                    ),
                )
        result = await self.availableTools.execute(
            name=name,
            tool_input=args,
        )
        if not isinstance(result, ToolResult):
            result = ToolResult(output=str(result))
        self._record_successful_file_view_for_edit_guard(name, args, result)
        return result

    @staticmethod
    def _sha256_file(path: Path) -> str | None:
        if not path.is_file():
            return None
        try:
            h = hashlib.sha256()
            h.update(path.read_bytes())
            return h.hexdigest()
        except OSError:
            return None

    def _parent_solution_path(self) -> Path | None:
        """Improve nodes: parent's ``solution.py`` on disk (sibling workspace), if known."""
        from scienceflow.utils.node_paths import find_node_context_path

        ws = self._workspace_dir
        ctx_path = find_node_context_path(ws)
        if ctx_path.is_file():
            try:
                ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                ctx = {}
            else:
                pid = ctx.get("parent_node_id") or ctx.get("inherit_parent_id")
                if isinstance(pid, str) and pid.strip():
                    p = (ws.parent / pid.strip() / "solution.py").resolve()
                    if p.is_file():
                        return p
        leg = ws / "parent_workspace" / "solution.py"
        return leg if leg.is_file() else None

    @staticmethod
    def classify_tool_error_for_budget(
        name: str,
        args: dict[str, Any],
        tool_result: ToolResult,
    ) -> str:
        """``protocol`` = user/tool-args mistakes; ``runtime`` = command execution failures."""
        err = (tool_result.error or "") or ""
        el = err.lower()
        if name == "bash":
            return "runtime"
        if name == "write":
            if "syntax check failed" in el:
                return "protocol"
            c = args.get("content")
            if isinstance(c, str) and len(c) < _WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD:
                return "protocol"
            return "runtime"
        if name == "grep":
            if "timed out" in el or "timeout" in el:
                return "runtime"
            return "protocol"
        if name in ("read", "edit"):
            return "protocol"
        return "runtime"

    async def compact(self, *, mid_run: bool = False) -> str:
        """Run LLM summarization and reset chat history (see REPL /compact).

        *mid_run* uses a shorter, debugging-oriented summary prompt (in-session continuation).
        """
        t0 = time.time()
        try:
            out = await self._memory_ctx.compact(self.llm, mid_run=mid_run)
        except BaseException:
            self._record_llm_call("compact", time.time() - t0, None, "error", recovery=False)
            raise
        if (out or "").strip() == "Nothing to compact.":
            return out
        dur_ok = time.time() - t0
        self._accumulate_compact_llm_into_run_counters()
        self._sync_last_run_token_totals()
        self._record_llm_call("compact", dur_ok, None, "ok", recovery=False)
        # compact() replaces chat history with a short summary; reset the monotone
        # window-k so the next build_messages_for_llm_with_stats re-evaluates from 0.
        try:
            self._memory_ctx._last_window_k = 0
        except AttributeError:
            pass
        return out

    async def compact_inband(self, *, mid_run: bool = True) -> str:
        """Compact REPL history through the normal system prompt, not a summarizer prompt."""
        t0 = time.time()
        compact_messages, total_messages = self._memory_ctx.build_inband_compact_messages(
            mid_run=mid_run,
        )
        if total_messages <= 0 or not compact_messages:
            return "Nothing to compact."
        if not hasattr(self.llm, "ask_tool_stream"):
            return "Compact failed: LLM backend has no ask_tool_stream()."

        try:
            assistant_msg = await self._ask_tool_stream_guarded(
                messages=compact_messages,
                system_msgs=[Message.system_message(self._build_system_prompt())],
                timeout=self._llm_stream_timeout_sec,
                tools=self._tools_with_thought,
                tool_choice="none",
                parallel_tool_calls=self._parallel_llm_tool_calls,
                collect_all_tool_calls=True,
            )
        except BaseException:
            self._record_llm_call(
                "compact_inband",
                time.time() - t0,
                None,
                "error",
                recovery=False,
            )
            raise

        dur_ok = time.time() - t0
        self._accumulate_compact_llm_into_run_counters()
        self._sync_last_run_token_totals()
        self._record_llm_call(
            "compact_inband",
            dur_ok,
            None,
            "ok",
            recovery=False,
            turn_kind="compact",
        )

        tool_calls = getattr(assistant_msg, "tool_calls", None) or []
        if tool_calls:
            return "Compact failed: summary turn emitted tool call(s)."
        summary = (getattr(assistant_msg, "content", None) or "").strip()
        if not summary:
            summary = (getattr(assistant_msg, "reasoning_content", None) or "").strip()
        recent_messages = 0 if bool(
            getattr(self, "_lnr_compact_on_context_threshold", False),
        ) else 8
        return self._memory_ctx.replace_history_with_compacted_summary(
            summary,
            recent_messages=recent_messages,
        )
