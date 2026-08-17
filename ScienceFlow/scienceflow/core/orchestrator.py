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

import os
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from deepcraft_core import Message

from scienceflow.config.settings import Config, prep_cfg
from scienceflow.core.runtime_env import make_path_env
from scienceflow.core.llm_http import aclose_llm_clients as _aclose_llm_clients
from scienceflow.core.message_bus import AsyncMessageBus
from scienceflow.utils.workspace_git import normalize_workspace_git_track_globs

from scienceflow.core.agent.run_policy import AutoContinuePolicy, DefaultPolicy

if TYPE_CHECKING:
    from scienceflow.core.agent import ScienceAgent
    from scienceflow.core.agent.run_policy import RunPolicy

logger = logging.getLogger("scienceflow")


def _repl_environment_context_prompt(workspace_dir: str | Path) -> str:
    """Stable REPL environment context, generated once at session start."""
    shell = (os.environ.get("SHELL") or "bash").rsplit("/", 1)[-1] or "bash"
    current_date = datetime.now(timezone.utc).date().isoformat()
    return (
        "<environment_context>\n"
        f"  <cwd>{Path(workspace_dir).resolve()}</cwd>\n"
        f"  <shell>{shell}</shell>\n"
        f"  <current_date>{current_date}</current_date>\n"
        "  <timezone>Etc/UTC</timezone>\n"
        "</environment_context>"
    )


def _plain_repl_auto_continue_session(run_policy: object, teleport_mode: str) -> bool:
    """True for plain REPL code-agent sessions that should mimic generic CLI context."""
    return isinstance(run_policy, AutoContinuePolicy) and str(teleport_mode or "off") in ("", "off")


def _repl_code_agent_append_prompt() -> str:
    """Stable REPL-only system suffix aligned with a generic code agent."""
    return (
        "\n\n## Interactive REPL Code Agent\n"
        "You are in one continuous workspace session. Inspect files, modify files, "
        "run commands, validate outcomes, and continue from the current state.\n"
        "- Treat the fixed user task as authoritative; do not rewrite, summarize, "
        "or replace it.\n"
        "- Prefer small reversible changes with direct validation when changing "
        "the workspace.\n"
        "- For existing files, prefer targeted shell-level changes over "
        "whole-file rewrites. Use whole-file write mainly for new files or a "
        "clearly small replacement.\n"
        "- If you need exact code anchors, use targeted reads with offsets or "
        "limits instead of repeatedly reading full files.\n"
        "- Use workspace files and recorded outcomes as durable memory instead "
        "of relying on long conversation history.\n"
        "- For greetings, thanks, small talk, or questions answerable without "
        "workspace facts, reply naturally without tools.\n"
        "- Do not introduce planning systems, stage machines, or multi-agent "
        "search in REPL mode.\n"
    )


def _repl_bash_file_write_prompt() -> str:
    """REPL-only tool contract when file writes are performed through bash."""
    return (
        "\n\n## REPL Bash File Writes\n"
        "Available tools in this REPL session are `bash`, `read`, `grep`, `glob`, "
        "and `ls`. Use `bash` as the file-change tool.\n"
        "Create and modify files with `bash` so the complete content or exact "
        "rewrite operation remains in the tool-call arguments.\n"
        "- Preserve useful working artifacts before risky changes when practical.\n"
        "- For small existing-file changes, read exact anchors when needed and make "
        "one targeted shell-level change.\n"
        "- After a successful file-modifying bash command, validate with checks or "
        "runs instead of re-reading only to confirm the write.\n"
        "- Use `ls` / `glob` for file discovery, `grep` for text search, and `read` "
        "for exact file content. Use bash for shell execution, installs, validation "
        "runs, artifact preservation, and file changes.\n"
        "- A bash observation may be summarized, deduplicated, or truncated. Do not "
        "treat a compact bash result as complete ground truth when exact details matter.\n"
        "- If bash output was compacted or too broad, recover exact facts with a "
        "narrower command, `grep`, or paged `read` calls against a workspace file.\n"
        "- For verbose or long-running commands, write the full log to a workspace "
        "file and print only key metrics, exit status, artifact paths, and a compact tail.\n"
        "- Do not run verbose training or validation as `command | tail` when that "
        "is the only copy of the output. Save the full output to a workspace log "
        "first, then print extracted metrics and a compact tail.\n"
        "- During iterative optimization, record the current best metric, "
        "command/configuration, and artifact path in a workspace ledger. Record "
        "each substantive attempt, including failed attempts that change direction.\n"
        "- After a training/validation command returns, update that ledger before "
        "starting another risky experiment. If the metric improves, preserve the "
        "matching artifact(s) as the best-known candidate immediately. If a better "
        "training run has not produced the final artifact yet, run the matching "
        "prediction/export command and preserve that output first.\n"
        "- Before finishing a timed optimization run, promote the best known "
        "artifact to the expected final artifact path and align any metric file "
        "with that promoted artifact.\n"
        "- Keep bash output compact; the command arguments carry the durable file "
        "content.\n"
    )


def _append_repl_bash_file_write_prompt(text: str) -> str:
    block = _repl_bash_file_write_prompt()
    marker = "## REPL Bash File Writes"
    if marker in (text or ""):
        return text or ""
    return (text or "").rstrip() + block


def _repl_code_organization_prompt(hint: str | None) -> str:
    """Return an optional REPL user-side code organization hint."""
    raw = str(hint or "").strip()
    if not raw:
        return ""
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key in {"0", "false", "no", "none", "off", "disabled"}:
        return ""
    if key in {
        "opt_solver",
        "optimization_solver",
        "artifact_solver",
        "generic_evaluator",
    }:
        return (
            "## Code organization preference\n\n"
            "Prefer solver-oriented workspace files for optimization tasks. "
            "A single `solution.py` is acceptable for a tiny baseline, but when "
            "the implementation grows, keep parsing, solver/search logic, "
            "validation, and artifact writing in clear functions or modules.\n\n"
            "Use the task's configured runtime and evaluator contract as the "
            "authority. Write the configured candidate artifact path exactly, "
            "and do not invent ML submission files or self-scored benchmark "
            "contracts unless the task description explicitly requests them.\n\n"
            "When iteration is expensive, preserve the best known candidate "
            "artifact before risky experiments, keep lightweight validation "
            "probes reproducible, and make resume state depend on workspace files "
            "rather than hidden process memory."
        )
    if key in {
        "beyond_mfiles",
        "beyond_multifile",
        "reusable_predict",
        "train_predict_boundary",
        "predict_reuse",
        "artifact_reuse",
    }:
        return (
            "## Code organization preference\n\n"
            "Prefer a reusable train/predict boundary over a fixed file count. "
            "A single `solution.py` is acceptable for a tiny or cheap baseline, "
            "but keep clear functions for loading data, preprocessing or feature "
            "extraction, training, prediction, and writing `submission.csv`.\n\n"
            "When training is expensive, artifacts are reusable, or the task is "
            "CV/NLP/audio with costly model fitting, split the workflow into "
            "runnable entrypoints:\n"
            "- `train.py`: train or tune once, save checkpoints/weights and every "
            "artifact required for inference, including tokenizer/vectorizer, "
            "label encoders, scalers, feature columns, thresholds, configs, and "
            "validation metrics.\n"
            "- `predict.py`: load saved artifacts and regenerate the root-level "
            "`submission.csv` without retraining. Keep this path deterministic "
            "and fast enough for repeated submit/postprocess checks.\n"
            "- `util.py` or shared functions: own preprocessing, feature extraction, "
            "metrics, path helpers, and submission schema validation so train and "
            "predict use identical transforms.\n\n"
            "After a strong model exists, prefer low-cost predict-only iterations "
            "before retraining: threshold tuning, calibration, postprocessing, "
            "test-time augmentation/inference settings, batch-size fixes, artifact "
            "loading checks, and submission formatting validation. Retrain only "
            "when the expected gain justifies the remaining budget.\n"
            "Do not duplicate train/test feature logic. If a transform changes, "
            "update the shared function or `util.py` first and keep train and "
            "predict aligned through that shared path."
        )
    if key in {
        "mfiles",
        "mle_mfiles",
        "mle_multifile",
        "multi_file",
        "multifile",
        "train_predict_submit_util",
        "train_predict_util",
    }:
        return (
            "## Code organization preference\n\n"
            "Prefer a small multi-file ML solution layout when it fits the task:\n"
            "- `util.py`: own shared data loading and feature engineering. Put the "
            "`build_features` / transform functions, metrics, seeds, and path helpers "
            "here so training and prediction use one identical feature pipeline.\n"
            "- `train.py`: train or tune models, record validation metrics, and save "
            "reusable artifacts by importing feature helpers from `util.py`.\n"
            "- `predict.py`: load saved artifacts, generate predictions deterministically, "
            "reuse `util.py` feature helpers, and write the required `submission.csv`.\n\n"
            "Do not duplicate feature-engineering code separately in training and "
            "prediction files. If a feature changes, update `util.py` first and keep "
            "both entrypoints aligned through that shared function.\n"
            "Keep entrypoints runnable from the shell and keep interfaces simple. "
            "Only add an extra submit/wrapper file if the task or runner genuinely "
            "needs a separate command; otherwise let `predict.py` be the submission "
            "entrypoint. If a single runnable entrypoint is required, provide a thin "
            "compatible wrapper that calls the multi-file pipeline. "
            "Do not over-split a tiny baseline when a simpler layout is clearly "
            "more reliable under the time budget."
        )
    return "## Code organization preference\n\n" + raw


def _repl_workspace_git_prompt(
    enabled: bool,
    track_globs: list[str] | tuple[str, ...] | None = None,
    *,
    auto_review: bool = False,
    auto_checkpoint: bool = True,
) -> str:
    """Return a short user-side hint for workspace-local source control."""
    if not enabled:
        return ""
    globs = ", ".join(f"`{p}`" for p in normalize_workspace_git_track_globs(track_globs))
    mode = (
        "Source checkpoints and submission snapshots are created automatically."
        if auto_checkpoint
        else "Source checkpoints are available in this local repository."
    )
    text = (
        "## Workspace source checkpoints\n\n"
        f"{mode} The repository tracks source/docs only ({globs}, plus "
        "`.gitignore`); datasets, model weights, logs, and submissions are not "
        "tracked by git.\n"
        "You may inspect checkpoints with `git log --oneline -- '*.py' '*.md'`, "
        "`git status --short`, and `git diff -- '*.py' '*.md'`. If an experiment "
        "breaks the source, you may restore tracked source files with "
        "`git restore --source=<commit> -- '*.py' '*.md'`. Do not create commits "
        "manually; keep focusing on modeling and validation."
    )
    if not auto_review:
        return text
    return (
        text
        + "\n"
        "Auto-review is enabled: before finalizing after a worse or broken "
        "experiment, inspect `git diff` and restore the best tracked source "
        "checkpoint if needed. Do not commit manually."
    )


class Orchestrator:
    """Main entry point for REPL-native long-horizon tasks."""

    def __init__(self, cfg: Config):
        self.cfg = prep_cfg(cfg)
        self.bus = AsyncMessageBus()

    def make_llm_call_tracer(
        self,
        *,
        node_id: str = "repl",
        process_id: int | str | None = None,
        fork_class: str | None = None,
        detail_prefix: str = "mode=repl",
    ) -> Callable[[dict[str, Any]], None] | None:
        """Build an ``on_llm_call`` hook that writes rows to ``scienceflow_time_trace.csv``."""
        if not bool(getattr(self.cfg, "enable_time_trace", True)):
            return None
        from scienceflow.utils.time_trace import TimeTracer, merge_trace_details

        tracer = TimeTracer(
            Path(self.cfg.log_dir),
            enabled=True,
            price_table_config=getattr(getattr(self.cfg, "agent", None), "llm_prices", None),
        )

        def _hook(payload: dict[str, Any]) -> None:
            op = str(payload.get("operation", "llm"))
            dur = float(payload.get("duration_sec", 0.0) or 0.0)
            status = str(payload.get("status", "ok"))
            seq = payload.get("call_seq", "")
            rnd = payload.get("round_idx")
            rec = bool(payload.get("recovery", False))
            model = (payload.get("model") or "") or ""
            turn_kind = payload.get("turn_kind")
            first_tool = payload.get("first_tool_name")
            first_bash_kind = payload.get("first_bash_kind")
            detail = merge_trace_details(
                detail_prefix,
                f"llm_role={str(payload.get('llm_role') or 'code')[:40]}",
                f"call_seq={seq}",
                f"round={rnd}" if rnd is not None else "round=-",
                f"recovery={int(rec)}",
                f"turn={turn_kind}" if isinstance(turn_kind, str) and turn_kind.strip() else "",
                f"tool={str(first_tool)[:40]}"
                if isinstance(first_tool, str) and first_tool.strip()
                else "",
                f"bash_kind={str(first_bash_kind)[:40]}"
                if isinstance(first_bash_kind, str) and first_bash_kind.strip()
                else "",
                f"model={str(model)[:40]}" if model else "",
                max_len=200,
            )
            tracer.record(
                "llm_api",
                op,
                dur,
                status=status,
                node_id=node_id,
                process_id=str(process_id) if process_id is not None else "",
                tokens_input=payload.get("tokens_input"),
                tokens_output=payload.get("tokens_output"),
                tokens_cached=payload.get("tokens_cached"),
                ttft_sec=payload.get("ttft_sec"),
                tpot_ms=payload.get("tpot_ms"),
                pool_index=payload.get("pool_index"),
                failover_count=payload.get("failover_count"),
                detail=detail,
                fork_class=fork_class,
            )

        return _hook

    def create_science_agent(
        self,
        *,
        ui: object | None = None,
        task_description: str | None = None,
        run_policy: RunPolicy | None = None,
        system_prompt: str | None = None,
        system_prompt_hook: Callable[[str], str] | None = None,
        append_repl_system_prompt: bool | None = None,
        pin_task_description: bool = True,
        max_steps_override: int | None = None,
        on_llm_call: Callable[[dict[str, Any]], None] | None = None,
        memory_dir_override: str | Path | None = None,
        load_existing_memory: bool = False,
        memory_agent_name: str = "ScienceAgent",
        teleport_mode: str = "off",
        repl_bash_write_mode: bool = False,
        stable_system_prompt: bool | None = None,
        pin_environment_context: bool | None = None,
        code_organization_hint: str | None = None,
        workspace_git_enabled: bool = False,
        workspace_git_track_globs: list[str] | tuple[str, ...] | None = None,
        workspace_git_auto_review: bool = False,
        workspace_git_auto_checkpoint: bool = False,
        bash_max_output_chars_override: int | None = None,
        bash_max_stream_line_chars_override: int | None = None,
        bash_observation_summary_override: bool | None = None,
        interaction_log_layout: str = "flat",
        extra_env_override: dict[str, str] | None = None,
        skill_registry: object | None = None,
        task_type: str | None = None,
        skill_allow_names: list[str] | tuple[str, ...] | None = None,
        skill_tool_mode: str = "all",
        skill_allow_generic_wildcard: bool = True,
        skill_visible_max: int = 0,
        llm_stage_override: object | None = None,
    ) -> ScienceAgent:
        """Construct a single ScienceAgent instance (REPL reuses it for persistent memory)."""
        from scienceflow.core.agent import ScienceAgent
        from scienceflow.core.agent_runtime import create_agent_memory, load_agent_memory, _build_llm

        built = _build_llm(llm_stage_override or self.cfg.agent.code)
        memory_dir = (
            Path(memory_dir_override).expanduser().resolve(strict=False)
            if memory_dir_override is not None
            else Path(self.cfg.log_dir) / "agent_memory"
        )
        memory_dir.mkdir(parents=True, exist_ok=True)
        if load_existing_memory:
            memory = load_agent_memory(memory_dir, memory_agent_name, self.cfg.max_messages)
        else:
            memory = create_agent_memory(memory_dir, memory_agent_name, self.cfg.max_messages)
        rp: RunPolicy = (
            run_policy if run_policy is not None else DefaultPolicy(recovery_rounds=3)
        )
        hook: Callable[[str], str] | None = system_prompt_hook
        if repl_bash_write_mode:
            prior_hook = hook

            def _bash_write_hook(base: str) -> str:
                updated = prior_hook(base) if prior_hook is not None else base
                return _append_repl_bash_file_write_prompt(updated)

            hook = _bash_write_hook
        td_stripped = (
            str(task_description).strip() if task_description and str(task_description).strip() else ""
        )
        append_repl = (
            system_prompt_hook is None
            if append_repl_system_prompt is None
            else bool(append_repl_system_prompt)
        )
        # REPL code-agent guidance now lives in a separate stable system core
        # message. Keep the legacy suffix helper available for tests/docs, but
        # do not append it into the runtime/tool system contract.
        _ = append_repl
        plain_repl = _plain_repl_auto_continue_session(rp, teleport_mode)
        stable_system = (
            isinstance(rp, AutoContinuePolicy)
            if stable_system_prompt is None
            else bool(stable_system_prompt)
        )
        agent_max_steps = int(max_steps_override) if max_steps_override is not None else int(self.cfg.qa_max_steps)
        if agent_max_steps <= 0:
            agent_max_steps = int(self.cfg.qa_max_steps)
        bash_max_output_chars = (
            int(bash_max_output_chars_override)
            if bash_max_output_chars_override is not None
            else int(getattr(self.cfg, "bash_max_output_chars", 8000) or 8000)
        )
        if bash_max_output_chars <= 0:
            bash_max_output_chars = 8000
        bash_max_stream_line_chars = (
            int(bash_max_stream_line_chars_override)
            if bash_max_stream_line_chars_override is not None
            else int(getattr(self.cfg, "bash_max_stream_line_chars", 2400) or 2400)
        )
        if bash_max_stream_line_chars <= 0:
            bash_max_stream_line_chars = 2400
        bash_observation_summary_enabled = (
            bool(bash_observation_summary_override)
            if bash_observation_summary_override is not None
            else False
        )
        pin_env = plain_repl if pin_environment_context is None else bool(pin_environment_context)
        write_auto_snapshot_enabled = bool(getattr(self.cfg, "write_auto_snapshot_enabled", True))
        if plain_repl:
            # The full write/edit payload already lives in assistant tool-call
            # arguments, like shell/patch-style calls. Keep the following tool
            # result short instead of echoing code through an auto-snapshot.
            write_auto_snapshot_enabled = False
        agent = ScienceAgent(
            llm=built,
            memory=memory,
            workspace_dir=self.cfg.workspace_dir,
            sandbox=self.cfg.scienceflow_tools_sandbox,
            path_guard_extra_roots=self.cfg.path_guard_extra_roots,
            ui=ui,
            max_steps=agent_max_steps,
            exec_feedback_max_chars=self.cfg.exec_feedback_max_chars,
            scienceflow_stdout_max_chars=self.cfg.scienceflow_stdout_max_chars,
            sliding_window_budget_chars=self.cfg.sliding_window_budget_chars,
            mid_run_compact_enabled=bool(self.cfg.mid_run_compact_enabled),
            llm_stream_timeout_sec=self.cfg.scienceflow_llm_stream_timeout_sec,
            llm_tool_stream_max_attempts=self.cfg.llm_tool_stream_max_attempts,
            llm_tool_stream_retry_base_delay_sec=self.cfg.llm_tool_stream_retry_base_delay_sec,
            llm_tool_stream_retry_max_delay_sec=self.cfg.llm_tool_stream_retry_max_delay_sec,
            stream_repetition_detection=bool(self.cfg.stream_repetition_detection),
            stream_repetition_window_chars=int(self.cfg.stream_repetition_window_chars),
            stream_repetition_ngram_len=int(self.cfg.stream_repetition_ngram_len),
            stream_repetition_max_repeats=int(self.cfg.stream_repetition_max_repeats),
            stream_max_output_chars_soft=int(self.cfg.stream_max_output_chars_soft),
            stream_repetition_retry_max=int(self.cfg.stream_repetition_retry_max),
            bash_timeout_sec=self.cfg.scienceflow_bash_timeout_sec,
            bash_timeout_slow_sec=self.cfg.scienceflow_bash_timeout_slow_sec,
            bash_max_output_chars=bash_max_output_chars,
            bash_max_stream_line_chars=bash_max_stream_line_chars,
            bash_observation_summary_enabled=bash_observation_summary_enabled,
            interaction_log_level=self.cfg.scienceflow_interaction_log_level,
            interaction_log_full=self.cfg.scienceflow_interaction_log_full,
            interaction_log_color=self.cfg.scienceflow_interaction_log_color,
            interaction_log_llm_stream=self.cfg.scienceflow_interaction_log_llm_stream,
            interaction_log_layout=interaction_log_layout,
            tool_memory_compression=self.cfg.tool_memory_compression,
            include_write_edit_tools=not bool(repl_bash_write_mode),
            msg0_compress_body=bool(getattr(self.cfg, "msg0_compress_body", False)),
            write_return_full_max_chars=int(self.cfg.write_return_full_max_chars),
            write_return_full_max_lines=int(self.cfg.write_return_full_max_lines),
            write_return_head_tail_lines=int(self.cfg.write_return_head_tail_lines),
            edit_return_full_max_chars=int(self.cfg.edit_return_full_max_chars),
            edit_return_full_max_lines=int(self.cfg.edit_return_full_max_lines),
            edit_return_head_tail_lines=int(self.cfg.edit_return_head_tail_lines),
            edit_return_change_ctx_lines=int(self.cfg.edit_return_change_ctx_lines),
            edit_failure_top_k_candidates=int(self.cfg.edit_failure_top_k_candidates),
            edit_failure_diag_max_chars=int(self.cfg.edit_failure_diag_max_chars),
            file_snapshot_latest_only=bool(self.cfg.file_snapshot_latest_only),
            grep_max_results_lines=int(getattr(self.cfg, "grep_max_results_lines", 50)),
            bash_success_tail_lines=self.cfg.bash_success_tail_lines,
            bash_success_tail_lines_solution=int(
                getattr(self.cfg, "bash_success_tail_lines_solution", 8),
            ),
            bash_success_tail_lines_test=int(
                getattr(self.cfg, "bash_success_tail_lines_test", 120),
            ),
            bash_success_tail_lines_readonly=int(
                getattr(self.cfg, "bash_success_tail_lines_readonly", 30),
            ),
            bash_success_tail_lines_install=int(
                getattr(self.cfg, "bash_success_tail_lines_install", 5),
            ),
            read_success_max_lines=int(getattr(self.cfg, "read_success_max_lines", 200) or 200),
            write_auto_snapshot_enabled=write_auto_snapshot_enabled,
            write_auto_snapshot_paths=getattr(self.cfg, "write_auto_snapshot_paths", None),
            write_auto_snapshot_code_extensions=getattr(
                self.cfg,
                "write_auto_snapshot_code_extensions",
                None,
            ),
            write_auto_snapshot_max_lines=int(
                getattr(self.cfg, "write_auto_snapshot_max_lines", 400) or 400,
            ),
            write_auto_snapshot_max_chars=int(
                getattr(self.cfg, "write_auto_snapshot_max_chars", 8_000) or 8_000,
            ),
            write_auto_snapshot_changed_context_lines=int(
                getattr(self.cfg, "write_auto_snapshot_changed_context_lines", 10) or 10,
            ),
            write_auto_snapshot_symbol_body_lines=int(
                getattr(self.cfg, "write_auto_snapshot_symbol_body_lines", 3) or 3,
            ),
            read_overlap_guard_enabled=bool(getattr(self.cfg, "read_overlap_guard_enabled", True)),
            bash_output_dedup_enabled=bool(self.cfg.bash_output_dedup_enabled),
            bash_output_dedup_min_repeat=int(self.cfg.bash_output_dedup_min_repeat),
            bash_output_dedup_summary_prefix=str(self.cfg.bash_output_dedup_summary_prefix),
            bash_output_dedup_apply_to_memory=bool(self.cfg.bash_output_dedup_apply_to_memory),
            parallel_bash_enabled=self.cfg.parallel_bash_enabled,
            parallel_llm_tool_calls=self.cfg.parallel_llm_tool_calls,
            extra_env=make_path_env(extra_env_override or {}),
            skill_registry=skill_registry,
            task_type=task_type,
            skill_allow_names=skill_allow_names,
            skill_tool_mode=skill_tool_mode,
            skill_allow_generic_wildcard=skill_allow_generic_wildcard,
            skill_visible_max=skill_visible_max,
            systemPrompt=system_prompt,
            system_prompt_hook=hook,
            run_policy=rp,
            on_llm_call=on_llm_call,
            teleport_mode=teleport_mode,
            stable_system_prompt=stable_system,
            workspace_git_auto_checkpoint_enabled=bool(
                workspace_git_enabled and workspace_git_auto_checkpoint,
            ),
            workspace_git_track_globs=workspace_git_track_globs,
        )
        if pin_env:
            env_prompt = _repl_environment_context_prompt(self.cfg.workspace_dir)
            agent.pin_message(Message.user_message(env_prompt))
        code_org = _repl_code_organization_prompt(code_organization_hint)
        if code_org:
            agent.pin_message(Message.user_message(code_org))
        workspace_git_prompt = _repl_workspace_git_prompt(
            workspace_git_enabled,
            workspace_git_track_globs,
            auto_review=workspace_git_auto_review,
            auto_checkpoint=workspace_git_auto_checkpoint,
        )
        if workspace_git_prompt:
            agent.pin_message(Message.user_message(workspace_git_prompt))
        if td_stripped and bool(pin_task_description):
            agent.pin_message(Message.user_message("## Task description\n\n" + td_stripped))
        return agent

    async def run_science_task(self, request: str, *, ui: object | None = None) -> str:
        """One-shot ScienceAgent run (new agent + memory each call)."""
        agent = self.create_science_agent(ui=ui)
        try:
            return await agent.run(request) or "(no text output)"
        finally:
            await _aclose_llm_clients(agent.llm)

    async def run_lnr_task(
        self,
        task_desc: str,
        resume: bool = False,
        resume_step: int | None = None,
    ) -> dict:
        """Run the REPL-native long-horizon stage-capture solver."""
        _ = resume, resume_step
        from scienceflow.solver.lnr import LnrSolver

        solver = LnrSolver(
            task_desc=task_desc,
            cfg=self.cfg,
            orchestrator=self,
        )
        return await solver.run()

    async def run(
        self,
        task_desc: str,
        task_type: str = "lnr",
        *,
        ui: "object | None" = None,
        resume: bool = False,
        resume_step: int | None = None,
    ) -> dict | str:
        """Main entry: REPL-native long-horizon solver only."""
        _ = ui, resume, resume_step
        tt = (task_type or "lnr").strip().lower()
        if tt != "lnr":
            raise ValueError(
                f"Unknown task_type {task_type!r} (expected 'lnr')",
            )
        return await self.run_lnr_task(task_desc)
