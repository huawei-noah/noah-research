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

"""Run shell commands in workspace with timeouts and output trimming."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from deepcraft_core.tool import BaseTool, ToolResult

from scienceflow.core.runtime_env import make_path_env
from scienceflow.core.subprocess_utils import spawn_shell, terminate_tree, terminate_tree_recoverable
from scienceflow.core.tools.resource_timeout_feedback import build_command_timeout_feedback
from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
    classify_bash_command,
)
from scienceflow.core.tools.shadow_workspace import ShadowWorkspaceManager
from scienceflow.safety.resource.signals import scan_output_health
from scienceflow.solver.lnr.resource_runtime.utilization import (
    logical_cuda_ordinals_for_assignment as _logical_cuda_ordinals_for_assignment,
    normalize_cuda_visible_devices_for_task_pool as _normalize_cuda_visible_devices_for_task_pool,
    parse_visible_gpu_ids as _parse_visible_gpu_ids,
    process_tree_cpu_snapshot as _process_tree_cpu_snapshot,
    process_tree_gpu_placement_snapshot as _process_tree_gpu_placement_snapshot,
    replace_leading_cuda_visible_devices as _replace_leading_cuda_visible_devices,
)
from scienceflow.solver.lnr.resource_runtime.gpu_feedback import (
    build_gpu_boundary_feedback as _build_gpu_boundary_feedback,
)
from scienceflow.solver.lnr.resource_runtime.metric_history import (
    metric_history_text as _metric_history_text,
    update_metric_history_lines as _update_metric_history_lines,
)
from scienceflow.solver.lnr.resource_runtime.workspace_gpu_guard import (
    cleanup_workspace_gpu_processes as _cleanup_workspace_gpu_processes,
)
from pydantic import ConfigDict, Field
from scienceflow.core.tools.bash.guards import (
    _command_executes_under_readonly_dir,
    _format_cpu_set_compact,
    _has_silent_redirect,
    _infer_timeout,
    _leading_cd_abs_path_missing,
    _parse_cpu_set_string,
    background_resource_command_blocked_error,
    dangerous_delete_command_blocked_error,
    global_filesystem_scan_blocked_error,
    hidden_workspace_path_usage_blocked_error,
    interactive_stdin_blocked_error,
    mixed_file_write_execution_blocked_error,
    normalize_bash_command_for_agent,
    privilege_escalation_blocked_error,
    process_control_blocked_error,
    shared_python_env_write_blocked_error,
    truncated_resource_output_blocked_error,
    workspace_scope_path_blocked_error,
)
from scienceflow.core.tools.bash.output import (
    _dedup_repeated_blocks,
    _distill_tracebacks,
    _has_masked_python_traceback,
    _maybe_lossless_observation_summary,
    _sanitize_model_visible_output_paths,
    _trim_output,
)
from scienceflow.core.tools.bash.signals import (
    _is_gpu_visibility_probe,
    _latest_artifact_snapshot,
    _parse_leading_sleep_command,
    _parse_progress_signals,
    _parse_resource_context_version,
    _parse_resource_pressure_generation,
    _parse_resource_value_hint,
)
from scienceflow.core.tools.bash.spawn_feedback import build_spawn_failure_tool_result

logger = logging.getLogger("scienceflow")

_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+.*(-\w*f\w*|--force|--recursive)\b.*\s/\s*$"),
    re.compile(r"\brm\s+.*\s/\s*$"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*\bof=/dev/"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"),  # fork bomb
]

_FINAL_VALIDATION_SCORE_RE = re.compile(
    r"Final\s+Validation\s+Score\s*:\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
)
_TRAINING_PROGRESS_RE = re.compile(
    r"\b(train|training|fit|fitting|epoch|epochs|fold|folds|iter|iteration|"
    r"batch|loss|auc|label)\b|\(\s*\d+\s*/\s*\d+\s*\)",
    re.IGNORECASE,
)
_VALIDATION_OR_INFERENCE_RE = re.compile(
    r"\b(validating|validation|evaluate|evaluating|inference|predict|predicting|"
    r"submission|saved submission)\b",
    re.IGNORECASE,
)
_OOM_RE = re.compile(r"\b(cuda\s+out\s+of\s+memory|out\s+of\s+memory|oom)\b", re.IGNORECASE)


def _queue_label_for_resource_class(resource_class: str) -> tuple[str, str]:
    cls = str(resource_class or "")
    if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
        return "GPU train", "gpu_train"
    if cls == RESOURCE_GPU_LIGHT_TRAIN:
        return "GPU light train", "gpu_light_train"
    if cls == RESOURCE_GPU_TT_LIGHT:
        return "GPU TT", "gpu_tt"
    if cls == RESOURCE_GPU_FEATURE_EXTRACT:
        return "GPU feature", "gpu_feature"
    if cls == RESOURCE_UNKNOWN_GPU_EXEC:
        return "GPU unknown", "gpu_unknown"
    return "compute", "compute"


def _executed_resource_termination_feedback(
    feedback: str,
    *,
    action: str,
) -> str:
    """Make an executed termination unambiguous to the owning agent."""
    action_text = str(action or "").upper()
    marker = "The command has already been terminated and is no longer running. Do not wait for it."
    detail = str(feedback or "").strip()
    if action_text == "KILL_AND_REPLAN" and detail.startswith(
        "Resource arbiter approved kill because"
    ):
        detail = detail.replace(
            "Resource arbiter approved kill because",
            "Resource arbiter has already terminated this command because",
            1,
        )
    return marker if not detail else marker + "\n" + detail




























def _resource_queue_feedback(
    elapsed: float,
    max_wait_sec: float,
    *,
    queue_label: str,
    queue_key: str,
) -> str:
    return (
        f"RESOURCE_FEEDBACK: queue_wait_aborted because the {queue_label.lower()} slot "
        f"stayed busy for {elapsed:.1f}s, exceeding the {max_wait_sec:.0f}s queue budget; "
        f"reason={queue_key}_queue_wait_exceeded.\n"
    )




def _resource_call(observer: Any | None, method: str, *args: Any, **kwargs: Any) -> Any:
    if observer is None:
        return None
    fn = getattr(observer, method, None)
    if fn is None:
        return None
    if kwargs:
        try:
            sig = inspect.signature(fn)
            has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except (TypeError, ValueError):
            pass
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.debug("[bash-resource] observer.%s failed", method, exc_info=True)
        return None


def _kill_revalidation_allows_termination(result: Any, *, hard_safety: bool) -> bool:
    if hard_safety:
        return True
    return bool(
        isinstance(result, dict)
        and result.get("enabled") is True
        and result.get("allow_kill") is True
    )


async def _resource_async_call(observer: Any | None, method: str, *args: Any, **kwargs: Any) -> Any:
    result = _resource_call(observer, method, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _resource_policy_preflight_result(
    observer: Any | None,
    job_id: str | None,
    *,
    inferred_class: str,
    gpu_ids: list[str],
    elapsed_sec: float,
    resource_context_version: int | None = None,
    resource_pressure_generation: int | None = None,
) -> ToolResult | None:
    if not job_id:
        return None
    decision = _resource_call(
        observer,
        "resource_preflight_decision",
        job_id,
        inferred_class=inferred_class,
        gpu_ids=gpu_ids,
        resource_context_version=resource_context_version,
        resource_pressure_generation=resource_pressure_generation,
    )
    if not isinstance(decision, dict) or decision.get("allowed", True):
        return None
    if decision.get("requires_admission_review"):
        reviewed = await _maybe_review_resource_admission(observer, job_id, decision)
        if isinstance(reviewed, dict):
            blocked_result = _resource_admission_tool_result(
                observer,
                job_id,
                reviewed,
                elapsed_sec=float(elapsed_sec or 0.0),
            )
            if blocked_result is not None:
                return blocked_result
            status = str(reviewed.get("status") or "").upper()
            if reviewed.get("acquired") or status not in _RESOURCE_ADMISSION_BLOCKED_STATUSES:
                return None
            decision = reviewed
    feedback = str(decision.get("feedback") or "").strip()
    feedback_suppressed = bool(decision.get("feedback_suppressed"))
    if not feedback and not feedback_suppressed:
        feedback = (
            f"RESOURCE_FEEDBACK: resource_policy_blocked because {decision.get('reason') or 'resource_policy_gate'}.\n"
        )
    error = str(decision.get("error") or "Resource policy blocked command")
    if feedback_suppressed:
        error = ""
    _resource_call(
        observer,
        "job_finished",
        job_id,
        status="resource_policy_blocked",
        elapsed_sec=float(elapsed_sec or 0.0),
        reason=str(decision.get("reason") or "resource_policy_gate"),
    )
    return ToolResult(output=feedback, error=error or None)


_RESOURCE_ADMISSION_BLOCKED_STATUSES = {"PENDING", "REPLAN", "DEFERRED", "DENIED_REPLAN", "DENIED_DUPLICATE"}


async def _maybe_review_resource_admission(
    observer: Any | None,
    job_id: str | None,
    decision: dict[str, Any],
) -> dict[str, Any]:
    if str(decision.get("status") or "").upper() not in _RESOURCE_ADMISSION_BLOCKED_STATUSES:
        return decision
    share_review = await _resource_async_call(
        observer,
        "admission_share_decide",
        job_id,
        admission_result=decision,
    )
    if isinstance(share_review, dict) and share_review.get("enabled") and isinstance(share_review.get("result"), dict):
        decision = dict(share_review["result"])
        if decision.get("acquired") or str(decision.get("status") or "").upper() not in _RESOURCE_ADMISSION_BLOCKED_STATUSES:
            return decision
        if decision.get("admission_llm_reviewed"):
            return decision
    review = await _resource_async_call(
        observer,
        "admission_decide",
        job_id,
        admission_result=decision,
    )
    if isinstance(review, dict) and review.get("enabled") and isinstance(review.get("result"), dict):
        return dict(review["result"])
    return decision


def _resource_admission_tool_result(
    observer: Any | None,
    job_id: str | None,
    decision: dict[str, Any],
    *,
    elapsed_sec: float,
) -> ToolResult | None:
    status_upper = str(decision.get("status") or "").upper()
    if decision.get("acquired") or status_upper not in _RESOURCE_ADMISSION_BLOCKED_STATUSES:
        return None
    wait_option = decision.get("resource_wait_option") if isinstance(decision.get("resource_wait_option"), dict) else {}
    wait_offered = bool(wait_option.get("offered"))
    if status_upper == "PENDING" and not decision.get("admission_llm_reviewed") and not wait_offered:
        return None
    feedback = str(decision.get("feedback") or "").strip()
    feedback_suppressed = bool(decision.get("feedback_suppressed"))
    if not feedback and feedback_suppressed:
        feedback = ""
    elif not feedback:
        feedback = (
            f"RESOURCE_FEEDBACK: {str(decision.get('status') or 'PENDING')} because {str(decision.get('reason') or 'gpu_slot_unavailable')}.\n"
        )
    status = str(decision.get("status") or "PENDING").lower()
    _resource_call(
        observer,
        "job_finished",
        job_id,
        status=f"resource_admission_{status}",
        elapsed_sec=max(0.0, float(elapsed_sec or 0.0)),
        reason=str(decision.get("reason") or "gpu_slot_unavailable"),
    )
    return ToolResult(output=feedback, error=(None if feedback_suppressed else "Resource admission pending"))


async def _wait_for_resource_queue(
    observer: Any | None,
    job_id: str | None,
    *,
    inferred_class: str,
    gpu_ids: list[str],
    on_output: Callable[[str], None] | None,
) -> ToolResult | None:
    decision = _resource_call(
        observer,
        "queue_try_acquire",
        job_id,
        inferred_class=inferred_class,
        gpu_ids=gpu_ids,
    )
    if not isinstance(decision, dict):
        return None
    if not decision.get("enabled") or decision.get("acquired", True):
        return None
    decision = await _maybe_review_resource_admission(observer, job_id, decision)
    blocked_result = _resource_admission_tool_result(observer, job_id, decision, elapsed_sec=0.0)
    if blocked_result is not None:
        return blocked_result

    queue_label, queue_key = _queue_label_for_resource_class(inferred_class)
    max_wait_sec = max(0.0, float(decision.get("max_wait_sec") or 0.0))
    heartbeat_sec = max(1.0, float(decision.get("heartbeat_sec") or 15.0))
    wait_start = time.time()
    last_emit_m = 0.0
    heartbeat_active = False
    try:
        while True:
            elapsed = time.time() - wait_start
            if max_wait_sec > 0 and elapsed >= max_wait_sec:
                reason = f"{queue_key}_queue_wait_exceeded_{max_wait_sec:.0f}s"
                _resource_call(
                    observer,
                    "queue_timeout",
                    job_id,
                    elapsed_sec=elapsed,
                    reason=reason,
                )
                if on_output is not None and heartbeat_active:
                    on_output("\r\033[K")
                block = (
                    f"[queue_timeout, {elapsed:.1f}s]\n"
                    f"{queue_label} queue wait exceeded after {max_wait_sec:.0f}s"
                    f"\n{_resource_queue_feedback(elapsed, max_wait_sec, queue_label=queue_label, queue_key=queue_key)}"
                )
                return ToolResult(output=block, error=f"{queue_label} queue wait exceeded")

            now_m = time.monotonic()
            if on_output is not None and now_m - last_emit_m >= heartbeat_sec:
                on_output(
                    f"\r\033[K[bash queued] waiting for {queue_label.lower()} slot, "
                    f"{elapsed:.0f}s elapsed]",
                )
                heartbeat_active = True
                last_emit_m = now_m
            _resource_call(
                observer,
                "queue_wait_heartbeat",
                job_id,
                elapsed_sec=elapsed,
            )
            await asyncio.sleep(min(0.25, max(0.05, max_wait_sec - elapsed if max_wait_sec else 0.25)))
            decision = _resource_call(
                observer,
                "queue_try_acquire",
                job_id,
                inferred_class=inferred_class,
                gpu_ids=gpu_ids,
            )
            if not isinstance(decision, dict):
                if on_output is not None and heartbeat_active:
                    on_output("\r\033[K")
                return None
            decision = await _maybe_review_resource_admission(observer, job_id, decision)
            blocked_result = _resource_admission_tool_result(observer, job_id, decision, elapsed_sec=time.time() - wait_start)
            if blocked_result is not None:
                if on_output is not None and heartbeat_active:
                    on_output("\r\033[K")
                return blocked_result
            if not decision.get("enabled") or decision.get("acquired", True):
                if on_output is not None and heartbeat_active:
                    on_output("\r\033[K")
                return None
    except asyncio.CancelledError:
        elapsed = time.time() - wait_start
        _resource_call(
            observer,
            "queue_timeout",
            job_id,
            elapsed_sec=elapsed,
            reason="cancelled_while_queued",
        )
        raise


async def _read_stream_limited(
    stream: asyncio.StreamReader | None,
    *,
    limit_chars: int = 4000,
) -> str:
    if stream is None:
        return ""
    chunks: list[str] = []
    seen = 0
    while True:
        data = await stream.readline()
        if not data:
            break
        text = data.decode(errors="replace")
        if seen < limit_chars:
            room = max(0, limit_chars - seen)
            chunks.append(text[:room])
            seen += len(text)
    return "".join(chunks)


async def _observe_unknown_command(
    *,
    observer: Any | None,
    job_id: str | None,
    inferred_class: str,
    gpu_ids: list[str],
    cmd: str,
    workspace_dir: Path,
    proc_env: dict[str, str],
    stream_limit: int,
    on_output: Callable[[str], None] | None,
) -> tuple[str | None, ToolResult | None]:
    policy = _resource_call(
        observer,
        "observation_policy",
        job_id,
        inferred_class=inferred_class,
        gpu_ids=gpu_ids,
    )
    if not isinstance(policy, dict) or not policy.get("enabled"):
        return None, None

    window_sec = max(0.05, float(policy.get("window_sec") or 30.0))
    manager = ShadowWorkspaceManager(
        copy_file_max_bytes=20 * 1024 * 1024,
        readonly_dir_names=("dataset", "data", "input"),
        protected_globs=(
            "submission.csv",
            "result.md",
            "*.ckpt",
            "*.joblib",
            "*.npy",
            "*.npz",
            "*.pkl",
            "*.pt",
            "*.pth",
            "*logit*",
            "*pred*.csv",
            "checkpoints/**",
            "models/**",
            "dataset/**",
            "data/**",
            "input/**",
        ),
    )
    try:
        shadow = manager.create(workspace_dir, job_id or "unknown")
    except Exception as exc:
        _resource_call(
            observer,
            "observation_event",
            job_id,
            state="shadow_create_failed",
            reason=type(exc).__name__,
        )
        return None, None

    if on_output is not None:
        on_output("[bash observing] running isolated trial before replay\n")
    _resource_call(
        observer,
        "observation_event",
        job_id,
        state="started",
        shadow_workspace=str(shadow.shadow_workspace),
    )

    start = time.time()
    timed_out = False
    proc = None
    try:
        proc = await spawn_shell(
            cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(shadow.shadow_workspace),
            env=proc_env,
            limit=int(stream_limit),
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    _read_stream_limited(proc.stdout),
                    _read_stream_limited(proc.stderr),
                ),
                timeout=window_sec,
            )
            await proc.wait()
        except asyncio.TimeoutError:
            timed_out = True
            await terminate_tree(proc)
            await proc.wait()

        elapsed = time.time() - start
        ok, changed = shadow.validate_real_workspace_unchanged()
        shadow_ok, shadow_changed = shadow.validate_shadow_protected_unchanged()
        if not ok or not shadow_ok:
            prefix = "real" if not ok else "shadow"
            details = changed if not ok else shadow_changed
            reason = "shadow_isolation_failed:" + prefix + ":" + ",".join(details[:5])
            _resource_call(
                observer,
                "observation_event",
                job_id,
                state="shadow_isolation_failed",
                reason=reason,
                elapsed_sec=elapsed,
                shadow_workspace=str(shadow.shadow_workspace),
            )
            _resource_call(
                observer,
                "job_finished",
                job_id,
                status="shadow_isolation_failed",
                returncode=getattr(proc, "returncode", None),
                elapsed_sec=elapsed,
                reason=reason,
            )
            return None, ToolResult(error=reason)

        if timed_out:
            reason = f"observation_window_exceeded_{window_sec:.0f}s"
            _resource_call(
                observer,
                "observation_promoted",
                job_id,
                reason=reason,
                elapsed_sec=elapsed,
            )
            if on_output is not None:
                on_output("[bash observing] command promoted to queued heavy replay\n")
            return RESOURCE_HEAVY_GPU_CANDIDATE, None

        _resource_call(
            observer,
            "observation_event",
            job_id,
            state="completed_replay_real",
            reason="trial_completed",
            elapsed_sec=elapsed,
            shadow_workspace=str(shadow.shadow_workspace),
        )
        return None, None
    except asyncio.CancelledError:
        if proc is not None:
            await terminate_tree(proc)
        raise
    except Exception as exc:
        elapsed = time.time() - start
        _resource_call(
            observer,
            "observation_event",
            job_id,
            state="trial_failed",
            reason=type(exc).__name__,
            elapsed_sec=elapsed,
            shadow_workspace=str(shadow.shadow_workspace),
        )
        return None, None
    finally:
        try:
            shadow.cleanup()
        except Exception:
            logger.debug("[bash-resource] shadow cleanup failed", exc_info=True)

# Matches explicit stderr-silencing redirects that users add deliberately.
# When present the caller chose to suppress error output, so empty stdout +
# non-zero rc is expected (file-not-found, glob miss, etc.) — NOT infra failure.




































































# Split on shell list separators while keeping delimiters (for per-segment pip rewrite / priv checks).

# Leading shell env assignments: VAR=value (quoted or unquoted) before the real command.




































class BashTool(BaseTool):
    """Execute a bash one-liner / pipeline in the workspace directory."""

    name: str = "bash"
    description: str = (
        "Execute a shell command with cwd already set to the task workspace. "
        "Prefer this for shell execution, package installs, quick listings, "
        "validation/training runs, artifact preservation, and one-off shell tasks. "
        "Use workspace-relative paths (e.g. dataset/) or run `pwd` first; do not "
        "assume external notebook paths or Kaggle paths like /mnt/data exist on "
        "this machine. IMPORTANT: cwd is already the workspace, so do not prefix "
        "commands with an absolute-path cd. Run workspace commands directly, such "
        "as `python3 solution.py` or `python3 train.py` when relevant. For large "
        "observations, print compact summaries or write verbose logs to workspace "
        "files instead of dumping raw output. "
        "For verbose training or validation, save a full workspace log first; do not "
        "use `command | tail` as the only output record."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Shell command to run (bash -c semantics via subprocess shell). "
                    "cwd is already set to the workspace; do not use an absolute "
                    "path prefix before running scripts. Use workspace-relative "
                    "paths (e.g. dataset/) or run `pwd` to confirm location. If "
                    "output may be large, write the full log to a workspace file "
                    "and print a compact summary/tail; do not pipe verbose "
                    "training or validation directly to `tail` as the only output record. "
                    "For long jobs, use SCIENCEFLOW_RUN_ARTIFACT_DIR and "
                    "SCIENCEFLOW_RUN_STATE_PATH when present: atomically save "
                    "periodic checkpoints/partials and handle SIGUSR1 by writing "
                    "status=interrupted before clean exit."
                ),
            },
        },
        "required": ["command"],
    }
    workspace_dir: Path = Field(...)
    max_output_chars: int = Field(default=8000)
    # Cap each stdout/stderr line sent to on_output (REPL live stream); 0 = no per-line cap.
    max_stream_line_chars: int = Field(default=2400)
    bash_timeout_sec: float = Field(default=1800.0)
    bash_timeout_slow_sec: float = Field(default=1800.0)
    # Optional task-level hard fuse. When set, BashTool caps command runtime by
    # remaining task wall clock minus finalization reserve; resource arbiter still
    # receives deadline events before the deterministic timeout fires.
    bash_hard_fuse_deadline_monotonic: float = Field(default=0.0)
    bash_hard_fuse_finalization_reserve_sec: float = Field(default=0.0)
    # REPL live stream: after this many seconds with no stdout/stderr line, emit progress updates.
    bash_heartbeat_idle_sec: float = Field(default=5.0)
    # Seconds between heartbeat updates; 0 disables. Each new subprocess line resets idle (see plan).
    bash_heartbeat_interval_sec: float = Field(default=10.0)
    # Minimum seconds between resource progress_heartbeat events for noisy training logs.
    # The first and final meaningful progress summaries are still emitted.
    resource_progress_heartbeat_min_interval_sec: float = Field(default=30.0)
    # Minimum seconds between workspace artifact scans while a command is running.
    resource_artifact_heartbeat_scan_interval_sec: float = Field(default=30.0)
    # Minimum age for a same-size/same-mtime artifact to be considered stable.
    resource_artifact_recoverable_settle_sec: float = Field(default=5.0)
    # Recoverable resource stops send SIGUSR1 first and wait for a clean marker before hard kill.
    resource_recoverable_stop_enabled: bool = Field(default=True)
    resource_recoverable_stop_sigusr1_grace_sec: float = Field(default=60.0)
    resource_recoverable_stop_marker_exit_grace_sec: float = Field(default=10.0)
    resource_recoverable_stop_sigterm_grace_sec: float = Field(default=5.0)
    # Optional extra environment variables injected into every subprocess (e.g. CUDA_VISIBLE_DEVICES).
    extra_env: Optional[dict[str, str]] = Field(default=None)
    # Basenames under workspace_dir that must not be used as execution targets (e.g. parent_workspace).
    readonly_dirs: list[str] = Field(default_factory=list)
    # Extra absolute roots (dataset symlink targets, pretrained_models_dir, etc.) that are also
    # allowed as ``cd`` targets — mirrors PathGuard.extra_roots used by read/write/edit/grep/glob/ls.
    path_guard_extra_roots: list[Path] = Field(default_factory=list)
    path_guard_denied_prefixes: list[str] = Field(default_factory=list)
    # asyncio StreamReader buffer limit (bytes).  Default 64 KiB is too small for
    # processes that produce very long lines (tqdm \r, large JSON, etc.).
    subprocess_stream_limit: int = Field(default=50 * 1024 * 1024)
    # Collapse consecutive repeated stdout/stderr line-blocks (1–4 lines) when a block
    # repeats at least this many times. 0 = disable (keep raw subprocess output).
    dedup_min_repeat: int = Field(default=3)
    # Collapse site-packages / non-workspace frames inside Python tracebacks in bash output.
    distill_tracebacks: bool = Field(default=True)
    # Strip symlink arrow targets (`` -> /real/path``) from bash output so the LLM does not
    # see host-specific absolute paths that leak from ``ls -la`` / ``find`` output.
    strip_symlink_targets: bool = Field(default=True)
    # When enabled, large successful observation outputs (ls/find/cat/head/...) are
    # represented as exact head/tail summaries while the full output is carried in
    # ToolResult.system for raw artifact storage. The command itself is never rewritten.
    observation_summary_enabled: bool = Field(default=False)
    # Opt-in guard for solvers that require model-visible workspace-relative
    # paths only. Blocks host absolute paths and parent-directory escapes in
    # bash commands before execution.
    forbid_host_absolute_paths: bool = Field(default=False)
    # Optional observe-only resource hook used by LNR. It must never affect command
    # execution or the agent-visible tool schema.
    resource_observer: Any | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _apply_hard_fuse_timeout(self, timeout_sec: float) -> float:
        timeout = max(1.0, float(timeout_sec or 1.0))
        try:
            deadline = float(self.bash_hard_fuse_deadline_monotonic or 0.0)
        except (TypeError, ValueError):
            deadline = 0.0
        if deadline <= 0.0:
            return timeout
        try:
            reserve = max(0.0, float(self.bash_hard_fuse_finalization_reserve_sec or 0.0))
        except (TypeError, ValueError):
            reserve = 0.0
        _ = reserve
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return 1.0
        return max(1.0, min(timeout, remaining))

    def _deadline_state(self) -> dict[str, float | bool]:
        try:
            deadline = float(self.bash_hard_fuse_deadline_monotonic or 0.0)
        except (TypeError, ValueError):
            deadline = 0.0
        try:
            reserve = max(0.0, float(self.bash_hard_fuse_finalization_reserve_sec or 0.0))
        except (TypeError, ValueError):
            reserve = 0.0
        if deadline <= 0.0:
            return {"deadline_event": False, "deadline_remaining_sec": 0.0, "finalization_reserve_sec": reserve}
        remaining = max(0.0, deadline - time.monotonic())
        return {
            "deadline_event": bool(reserve > 0.0 and remaining <= reserve),
            "deadline_remaining_sec": remaining,
            "finalization_reserve_sec": reserve,
        }

    async def execute(
        self,
        command: str,
        *,
        on_output: Callable[[str], None] | None = None,
        **kwargs,
    ) -> ToolResult:
        kwargs.pop("config", None)
        raw_cmd = (command or "").strip()
        cmd, pip_rewritten = normalize_bash_command_for_agent(raw_cmd)
        if pip_rewritten:
            logger.info(
                "[bash-normalize] reason=pip_rewritten_to_uv_pip before=%r after=%r",
                raw_cmd,
                cmd,
            )
        if not cmd:
            return ToolResult(error="Empty command")

        priv_err = privilege_escalation_blocked_error(cmd)
        if priv_err:
            return ToolResult(error=priv_err)
        process_control_err = process_control_blocked_error(cmd)
        if process_control_err:
            return ToolResult(error=process_control_err)
        interactive_err = interactive_stdin_blocked_error(cmd)
        if interactive_err:
            return ToolResult(error=interactive_err)
        if bool(self.forbid_host_absolute_paths):
            workspace_scope_err = workspace_scope_path_blocked_error(
                cmd,
                workspace_dir=self.workspace_dir,
                allowed_roots=self.path_guard_extra_roots,
            )
            if workspace_scope_err:
                return ToolResult(error=workspace_scope_err)
        global_scan_err = global_filesystem_scan_blocked_error(cmd)
        if global_scan_err:
            return ToolResult(error=global_scan_err)
        mixed_write_exec_err = mixed_file_write_execution_blocked_error(cmd)
        if mixed_write_exec_err:
            return ToolResult(error=mixed_write_exec_err)
        background_resource_err = background_resource_command_blocked_error(cmd)
        if background_resource_err:
            return ToolResult(error=background_resource_err)
        delete_err = dangerous_delete_command_blocked_error(cmd)
        if delete_err:
            return ToolResult(error=delete_err)
        truncated_output_err = truncated_resource_output_blocked_error(cmd)
        if truncated_output_err:
            return ToolResult(error=truncated_output_err)
        env_write_err = shared_python_env_write_blocked_error(cmd)
        if env_write_err:
            return ToolResult(error=env_write_err)
        hidden_path_err = hidden_workspace_path_usage_blocked_error(
            cmd,
            self.path_guard_denied_prefixes,
        )
        if hidden_path_err:
            return ToolResult(error=hidden_path_err)

        for pat in _DANGEROUS_PATTERNS:
            if pat.search(cmd):
                return ToolResult(error=f"Blocked potentially dangerous command: {cmd!r}")

        blocked = _command_executes_under_readonly_dir(cmd, list(self.readonly_dirs or []))
        if blocked:
            return ToolResult(
                error=(
                    f"Blocked: executing scripts under read-only directory {blocked!r} is not allowed. "
                    "Read files with read/grep/glob/ls or non-executing bash (e.g. cat, head) instead."
                )
            )

        cd_err = _leading_cd_abs_path_missing(
            cmd,
            workspace_dir=self.workspace_dir,
            allowed_roots=self.path_guard_extra_roots,
        )
        if cd_err:
            return ToolResult(error=cd_err)

        sleep_prefix = _parse_leading_sleep_command(cmd) if self.resource_observer is not None else None
        sleep_prefix_command = cmd
        if sleep_prefix is not None:
            wait_instruction = _resource_call(
                self.resource_observer,
                "resource_wait_instruction_for_bash_sleep",
                command=sleep_prefix_command,
                planned_sleep_sec=float(sleep_prefix[0]),
            )
            if isinstance(wait_instruction, dict) and wait_instruction.get("available"):
                feedback = str(wait_instruction.get("feedback") or "").strip()
                if not feedback:
                    feedback = (
                        "RESOURCE_FEEDBACK: RESOURCE_WAIT_REQUIRED because use resource_wait tool instead of bash sleep; "
                        "wait_tool=resource_wait; bash_sleep_allowed=false; retry_allowed=false.\n"
                    )
                return ToolResult(output=feedback, error="Use resource_wait tool")
        timeout = _infer_timeout(
            sleep_prefix[1] if sleep_prefix and sleep_prefix[1] else cmd,
            float(self.bash_timeout_sec),
            float(self.bash_timeout_slow_sec),
        )
        timeout = self._apply_hard_fuse_timeout(timeout)
        agent_backoff_sleep_sec = None
        agent_backoff_wait_id = ""
        agent_backoff_finished = False
        ws = Path(self.workspace_dir).resolve()
        start = time.time()
        start_wall = start

        # Always merge PATH/VIRTUAL_ENV from make_path_env so ``python`` in bash matches the
        # ScienceFlow process (e.g. uv run .venv), even if BashTool.extra_env was not set.
        proc_env: dict[str, str] = {**os.environ, **make_path_env()}
        if self.extra_env:
            proc_env = {**proc_env, **self.extra_env}
        task_physical_gpu_pool = _parse_visible_gpu_ids(
            proc_env.get("SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL")
            or proc_env.get("SCIENCEFLOW_RESOLVED_CUDA_DEVICES")
            or proc_env.get("CUDA_VISIBLE_DEVICES")
        )
        gpu_ids = _parse_visible_gpu_ids(proc_env.get("CUDA_VISIBLE_DEVICES"))
        resource_lease_assignment = bool(
            _resource_call(self.resource_observer, "lease_assignment_enabled")
        )
        if resource_lease_assignment and "CUDA_VISIBLE_DEVICES" in proc_env and not _is_gpu_visibility_probe(cmd):
            # In lease mode, inherited worker CUDA visibility is only a physical GPU pool hint.
            # Child processes receive a concrete physical GPU subset only after admission grants a lease.
            proc_env["CUDA_VISIBLE_DEVICES"] = ""

        if sleep_prefix is not None:
            sleep_sec, after_sleep_cmd = sleep_prefix
            agent_backoff_wait_id = (
                f"sleep:{int(start * 1000)}:"
                f"{hashlib.sha1(sleep_prefix_command.encode(errors='replace')).hexdigest()[:12]}"
            )
            wake_snapshot = _resource_call(self.resource_observer, "agent_backoff_wake_snapshot")
            start_pressure_generation = 0
            if isinstance(wake_snapshot, dict):
                try:
                    start_pressure_generation = int(wake_snapshot.get("pressure_generation") or 0)
                except (TypeError, ValueError):
                    start_pressure_generation = 0
            _resource_call(
                self.resource_observer,
                "agent_backoff_wait_started",
                wait_id=agent_backoff_wait_id,
                planned_sleep_sec=float(sleep_sec),
                command=sleep_prefix_command,
                reason=("agent_sleep_prefix_command" if after_sleep_cmd else "agent_sleep_command"),
                pressure_generation=start_pressure_generation,
            )
            wait_status = "success"
            wake_reason = "timer_elapsed"
            finish_reason = ""
            remaining = float(sleep_sec)
            try:
                poll_sec = max(0.05, float(proc_env.get("_SCIENCEFLOW_AGENT_BACKOFF_POLL_SEC", "15") or 15.0))
            except (TypeError, ValueError):
                poll_sec = 15.0
            try:
                while remaining > 0:
                    chunk = min(remaining, poll_sec)
                    await asyncio.sleep(max(0.0, chunk))
                    remaining -= chunk
                    wake = _resource_call(
                        self.resource_observer,
                        "agent_backoff_wake_decision",
                        start_pressure_generation=start_pressure_generation,
                        target_resource_class=RESOURCE_HEAVY_GPU_TRAIN,
                        gpu_ids=gpu_ids,
                    )
                    if isinstance(wake, dict) and wake.get("wake"):
                        wait_status = "woken"
                        wake_reason = str(wake.get("wake_reason") or "resource_available")
                        finish_reason = "resource_backoff_wake"
                        break
            except asyncio.CancelledError:
                _resource_call(
                    self.resource_observer,
                    "agent_backoff_wait_finished",
                    wait_id=agent_backoff_wait_id,
                    planned_sleep_sec=float(sleep_sec),
                    elapsed_sec=max(0.0, time.time() - start),
                    status="cancelled",
                    wake_reason="cancelled",
                    reason="cancelled",
                )
                agent_backoff_finished = True
                raise
            _resource_call(
                self.resource_observer,
                "agent_backoff_wait_finished",
                wait_id=agent_backoff_wait_id,
                planned_sleep_sec=float(sleep_sec),
                elapsed_sec=max(0.0, time.time() - start),
                status=wait_status,
                wake_reason=wake_reason,
                reason=finish_reason,
            )
            agent_backoff_finished = True
            if not after_sleep_cmd:
                return ToolResult(output="", error=None)
            cmd = after_sleep_cmd
            timeout = _infer_timeout(
                cmd,
                float(self.bash_timeout_sec),
                float(self.bash_timeout_slow_sec),
            )
            timeout = self._apply_hard_fuse_timeout(timeout)
            start = time.time()
            start_wall = start

        cpu_set = (proc_env.pop("_SCIENCEFLOW_CPU_SET", "") or "").strip()
        if cpu_set:
            # Guard: drop CPU IDs that exceed the host core count before passing
            # them to taskset.  taskset exits 1 immediately for non-existent CPUs,
            # making every bash invocation fail with no useful output.
            host_cores = os.cpu_count() or 0
            if host_cores > 0:
                valid_ids = [
                    c for c in _parse_cpu_set_string(cpu_set) if c < host_cores
                ]
                oob_ids = [
                    c for c in _parse_cpu_set_string(cpu_set) if c >= host_cores
                ]
                if oob_ids:
                    logger.warning(
                        "[bash-tool] cpu_set %r has %d out-of-range ID(s) "
                        "(host has %d cores); dropping them before taskset",
                        cpu_set, len(oob_ids), host_cores,
                    )
                if valid_ids:
                    cpu_set = _format_cpu_set_compact(valid_ids)
                    cmd = f"taskset -c {cpu_set} bash -c {shlex.quote(cmd)}"
                else:
                    logger.warning(
                        "[bash-tool] cpu_set %r is entirely out-of-range "
                        "(host has %d cores); skipping taskset affinity",
                        cpu_set, host_cores,
                    )
                    cpu_set = ""
            else:
                cmd = f"taskset -c {cpu_set} bash -c {shlex.quote(cmd)}"

        classified = classify_bash_command(cmd)
        resource_value_hint = _parse_resource_value_hint(cmd, proc_env)
        resource_job_id = _resource_call(
            self.resource_observer,
            "job_created",
            command=cmd,
            inferred_class=classified.resource_class,
            gpu_ids=gpu_ids,
            cpu_set=cpu_set or None,
            timeout_sec=timeout,
            workspace_dir=ws,
            classifier_reason=classified.reason,
            value_hint_override=resource_value_hint,
            candidate_artifact=str(proc_env.get("SCIENCEFLOW_CANDIDATE_ARTIFACT") or ""),
        )
        run_dir: Path | None = None
        run_artifact_dir: Path | None = None
        run_state_path: Path | None = None
        if resource_job_id:
            safe_job_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(resource_job_id)).strip("_") or "job"
            run_dir = ws / ".scienceflow_runs" / safe_job_id
            run_artifact_dir = run_dir / "artifacts"
            run_state_path = run_dir / "run_state.json"
            try:
                run_artifact_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.debug("[bash-resource] failed to create run artifact dir", exc_info=True)
                run_dir = None
                run_artifact_dir = None
                run_state_path = None
            if run_dir is not None and run_artifact_dir is not None and run_state_path is not None:
                proc_env.update({
                    "SCIENCEFLOW_RUN_ID": str(resource_job_id),
                    "SCIENCEFLOW_RUN_DIR": str(run_dir),
                    "SCIENCEFLOW_RUN_ARTIFACT_DIR": str(run_artifact_dir),
                    "SCIENCEFLOW_RUN_STATE_PATH": str(run_state_path),
                    "SCIENCEFLOW_RECOVERABLE_STOP": "1",
                    "SCIENCEFLOW_STOP_SIGNAL": "SIGUSR1",
                })
                _resource_call(
                    self.resource_observer,
                    "recoverable_artifact_scope_registered",
                    resource_job_id,
                    run_dir=run_dir,
                    artifact_dir=run_artifact_dir,
                    run_state_path=run_state_path,
                )
        effective_resource_class = classified.resource_class
        resource_context_version = _parse_resource_context_version(cmd, proc_env)
        if resource_context_version is None:
            observer_context_version = _resource_call(
                self.resource_observer,
                "planning_resource_context_version",
            )
            try:
                resource_context_version = (
                    int(observer_context_version)
                    if observer_context_version is not None
                    else None
                )
            except (TypeError, ValueError):
                resource_context_version = None
        resource_pressure_generation = _parse_resource_pressure_generation(cmd, proc_env)
        if resource_pressure_generation is None:
            observer_pressure_generation = _resource_call(
                self.resource_observer,
                "planning_resource_pressure_generation",
            )
            try:
                resource_pressure_generation = (
                    int(observer_pressure_generation)
                    if observer_pressure_generation is not None
                    else None
                )
            except (TypeError, ValueError):
                resource_pressure_generation = None
        policy_result = await _resource_policy_preflight_result(
            self.resource_observer,
            resource_job_id,
            inferred_class=effective_resource_class,
            gpu_ids=gpu_ids,
            elapsed_sec=time.time() - start,
            resource_context_version=resource_context_version,
            resource_pressure_generation=resource_pressure_generation,
        )
        if policy_result is not None:
            return policy_result

        promoted_class, observation_result = await _observe_unknown_command(
            observer=self.resource_observer,
            job_id=resource_job_id,
            inferred_class=classified.resource_class,
            gpu_ids=gpu_ids,
            cmd=cmd,
            workspace_dir=ws,
            proc_env=proc_env,
            stream_limit=int(self.subprocess_stream_limit),
            on_output=on_output,
        )
        if observation_result is not None:
            return observation_result
        if promoted_class:
            effective_resource_class = promoted_class
            policy_result = await _resource_policy_preflight_result(
                self.resource_observer,
                resource_job_id,
                inferred_class=effective_resource_class,
                gpu_ids=gpu_ids,
                elapsed_sec=time.time() - start,
                resource_context_version=resource_context_version,
                resource_pressure_generation=resource_pressure_generation,
            )
            if policy_result is not None:
                return policy_result

        queue_result = await _wait_for_resource_queue(
            self.resource_observer,
            resource_job_id,
            inferred_class=effective_resource_class,
            gpu_ids=gpu_ids,
            on_output=on_output,
        )
        if queue_result is not None:
            return queue_result
        lease_env_updates = _resource_call(
            self.resource_observer,
            "lease_env_updates",
            resource_job_id,
        )
        if isinstance(lease_env_updates, dict):
            clean_updates = {
                str(k): str(v)
                for k, v in lease_env_updates.items()
                if str(k).strip() and v is not None
            }
            if clean_updates:
                if "CUDA_VISIBLE_DEVICES" in clean_updates:
                    leased_cuda = clean_updates["CUDA_VISIBLE_DEVICES"]
                    normalized_cuda, normalized_gpu_ids, remapped_cuda, normalize_reason = (
                        _normalize_cuda_visible_devices_for_task_pool(leased_cuda, task_physical_gpu_pool)
                    )
                    clean_updates["CUDA_VISIBLE_DEVICES"] = normalized_cuda
                    clean_updates.setdefault("SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL", normalized_cuda)
                    clean_updates.setdefault(
                        "SCIENCEFLOW_ASSIGNED_CUDA_LOGICAL",
                        _logical_cuda_ordinals_for_assignment(normalized_gpu_ids),
                    )
                    if task_physical_gpu_pool:
                        clean_updates.setdefault(
                            "SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL",
                            ",".join(task_physical_gpu_pool),
                        )
                    if remapped_cuda:
                        clean_updates["SCIENCEFLOW_CUDA_VISIBLE_NORMALIZED_FROM"] = leased_cuda
                    if normalized_gpu_ids and task_physical_gpu_pool and not set(normalized_gpu_ids).issubset(set(task_physical_gpu_pool)):
                        reason = "task_gpu_boundary_preflight_violation"
                        feedback = _build_gpu_boundary_feedback(
                            actual_gpu_ids=normalized_gpu_ids,
                            allowed_gpu_ids=task_physical_gpu_pool,
                            command_scope="lease",
                        ) + "\n"
                        _resource_call(
                            self.resource_observer,
                            "resource_guard_action",
                            resource_job_id,
                            action="stop_boundary_violation",
                            reason=reason,
                            elapsed_sec=max(0.0, time.time() - start),
                            placement={
                                "allowed_gpu_ids": task_physical_gpu_pool,
                                "leased_cuda_visible_devices": leased_cuda,
                                "normalized_cuda_visible_devices": normalized_cuda,
                                "normalize_reason": normalize_reason,
                            },
                        )
                        _resource_call(
                            self.resource_observer,
                            "job_finished",
                            resource_job_id,
                            status="resource_boundary_violation",
                            elapsed_sec=max(0.0, time.time() - start),
                            reason=reason,
                        )
                        return ToolResult(output=feedback, error="Resource boundary violation")
                    leased_cuda = normalized_cuda
                    gpu_ids = normalized_gpu_ids
                    cmd = _replace_leading_cuda_visible_devices(cmd, leased_cuda)
                proc_env.update(clean_updates)

        # Queue wait is intentionally outside the bash runtime budget.
        start = time.time()
        start_wall = start

        def _finish_agent_backoff_wait(
            *,
            status: str,
            wake_reason: str,
            elapsed_sec: float | None = None,
            returncode: int | None = None,
            reason: str = "",
        ) -> None:
            nonlocal agent_backoff_finished
            if not agent_backoff_wait_id or agent_backoff_finished:
                return
            agent_backoff_finished = True
            _resource_call(
                self.resource_observer,
                "agent_backoff_wait_finished",
                wait_id=agent_backoff_wait_id,
                planned_sleep_sec=float(agent_backoff_sleep_sec or 0.0),
                elapsed_sec=float(time.time() - start if elapsed_sec is None else elapsed_sec),
                status=status,
                wake_reason=wake_reason,
                returncode=returncode,
                reason=reason,
            )

        if agent_backoff_sleep_sec is not None:
            agent_backoff_wait_id = (
                f"sleep:{int(start * 1000)}:"
                f"{hashlib.sha1(cmd.encode(errors='replace')).hexdigest()[:12]}"
            )
            wake_snapshot = _resource_call(self.resource_observer, "agent_backoff_wake_snapshot")
            start_pressure_generation = 0
            if isinstance(wake_snapshot, dict):
                try:
                    start_pressure_generation = int(wake_snapshot.get("pressure_generation") or 0)
                except (TypeError, ValueError):
                    start_pressure_generation = 0
            _resource_call(
                self.resource_observer,
                "agent_backoff_wait_started",
                wait_id=agent_backoff_wait_id,
                planned_sleep_sec=float(agent_backoff_sleep_sec),
                command=cmd,
                reason="agent_sleep_command",
                pressure_generation=start_pressure_generation,
            )
            remaining = float(agent_backoff_sleep_sec)
            try:
                poll_sec = max(0.05, float(proc_env.get("_SCIENCEFLOW_AGENT_BACKOFF_POLL_SEC", "15") or 15.0))
            except (TypeError, ValueError):
                poll_sec = 15.0
            while remaining > 0:
                chunk = min(remaining, poll_sec)
                await asyncio.sleep(max(0.0, chunk))
                remaining -= chunk
                wake = _resource_call(
                    self.resource_observer,
                    "agent_backoff_wake_decision",
                    start_pressure_generation=start_pressure_generation,
                    target_resource_class=RESOURCE_HEAVY_GPU_TRAIN,
                    gpu_ids=gpu_ids,
                )
                if isinstance(wake, dict) and wake.get("wake"):
                    elapsed_sleep = time.time() - start
                    _finish_agent_backoff_wait(
                        status="woken",
                        wake_reason=str(wake.get("wake_reason") or "resource_available"),
                        elapsed_sec=elapsed_sleep,
                        reason="resource_backoff_wake",
                    )
                    _resource_call(
                        self.resource_observer,
                        "job_finished",
                        resource_job_id,
                        status="success",
                        returncode=0,
                        elapsed_sec=elapsed_sleep,
                        reason="agent_backoff_woken",
                    )
                    return ToolResult(output="", error=None)
            elapsed_sleep = time.time() - start
            _finish_agent_backoff_wait(
                status="success",
                wake_reason="timer_elapsed",
                elapsed_sec=elapsed_sleep,
                reason="",
            )
            _resource_call(
                self.resource_observer,
                "job_finished",
                resource_job_id,
                status="success",
                returncode=0,
                elapsed_sec=elapsed_sleep,
                reason="agent_backoff_elapsed",
            )
            return ToolResult(output="", error=None)
        try:
            proc = await spawn_shell(
                cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ws),
                env=proc_env,
                limit=int(self.subprocess_stream_limit),
            )
        except Exception as exc:
            elapsed = time.time() - start
            _finish_agent_backoff_wait(
                status="spawn_failed",
                wake_reason="spawn_failed",
                elapsed_sec=elapsed,
                reason=type(exc).__name__,
            )
            _resource_call(
                self.resource_observer,
                "job_finished",
                resource_job_id,
                status="spawn_failed",
                elapsed_sec=elapsed,
                reason=type(exc).__name__,
            )
            logger.warning("[bash-tool] subprocess spawn failed: %s", exc)
            return build_spawn_failure_tool_result(exc, elapsed_sec=elapsed)
        _resource_call(
            self.resource_observer,
            "lease_registered",
            resource_job_id,
            pid=getattr(proc, "pid", None),
            pgid=getattr(proc, "pid", None),
            gpu_ids=gpu_ids,
            cpu_set=cpu_set or None,
        )
        observe_first_state = _resource_call(
            self.resource_observer,
            "observe_first_state",
            resource_job_id,
        )
        if not isinstance(observe_first_state, dict):
            observe_first_state = {}
        observe_first_active = bool(observe_first_state.get("active"))
        try:
            observe_first_window_sec = max(1.0, float(observe_first_state.get("observe_window_sec") or 0.0))
        except (TypeError, ValueError):
            observe_first_window_sec = 0.0
        observe_first_health_emitted = [False]

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        # Live stream caps (ToolResult still gets full captured output, then _trim_output).
        stream_state: dict[str, int | bool] = {"sent": 0, "truncated": False}
        line_cap = int(self.max_stream_line_chars)
        total_cap = int(self.max_output_chars)
        idle_after = float(self.bash_heartbeat_idle_sec)
        interval_hb = float(self.bash_heartbeat_interval_sec)
        io_touch = [time.monotonic()]
        last_hb_emit = [0.0]
        heartbeat_active = [False]
        guard_reason: list[str | None] = [None]
        guard_feedback: list[str | None] = [None]
        guard_recommendation_emitted: list[bool] = [False]
        guard_observe_more_until: list[float] = [0.0]
        placement_audit_last: list[float] = [0.0]
        guard_task: asyncio.Task[None] | None = None
        guard_watchdog_task: asyncio.Task[None] | None = None
        last_monitor_heartbeat_wall: list[float] = [0.0]
        guard_watchdog_restarts: list[int] = [0]
        stream_stats: dict[str, Any] = {
            "stdout_lines": 0,
            "stdout_bytes": 0,
            "saw_training_progress": False,
            "saw_final_score": False,
            "current_phase": "",
            "invalid_metric_events": 0,
            "zero_score_events": 0,
            "last_invalid_metric_text": "",
            "last_zero_score_text": "",
            "terminal_signal_events": 0,
            "terminal_signal_kind": "",
            "last_terminal_signal_text": "",
            "last_progress_signals": {},
            "last_progress_emitted": {},
            "metric_history_lines": [],
            "metric_history_text": "",
            "last_artifact": {},
            "last_artifact_emitted": {},
            "last_artifact_emitted_key": (),
        }
        last_progress_emit = [0.0]
        last_artifact_scan = [0.0]
        artifact_watch_cache: dict[str, Any] = {}
        progress_min_interval = max(
            0.0,
            float(self.resource_progress_heartbeat_min_interval_sec or 0.0),
        )
        artifact_scan_interval = max(
            0.05,
            float(self.resource_artifact_heartbeat_scan_interval_sec or 30.0),
        )

        def _touch_io() -> None:
            io_touch[0] = time.monotonic()
            last_hb_emit[0] = 0.0

        def _maybe_emit_progress_heartbeat(
            progress_signals: dict[str, Any],
            *,
            elapsed_sec: float | None = None,
            force: bool = False,
            reason: str = "progress",
        ) -> None:
            if not progress_signals:
                return
            now_progress = time.time()
            last_emitted = stream_stats.get("last_progress_emitted") or {}
            changed = progress_signals != last_emitted
            if last_progress_emit[0] <= 0.0:
                emit_reason = "initial"
            elif force and changed:
                emit_reason = reason or "final"
            elif changed and (
                progress_min_interval <= 0.0
                or now_progress - last_progress_emit[0] >= progress_min_interval
            ):
                emit_reason = "interval"
            else:
                return
            last_progress_emit[0] = now_progress
            stream_stats["last_progress_emitted"] = dict(progress_signals)
            _resource_call(
                self.resource_observer,
                "progress_heartbeat",
                resource_job_id,
                elapsed_sec=max(
                    0.0,
                    float(elapsed_sec)
                    if elapsed_sec is not None
                    else now_progress - start,
                ),
                phase=str(stream_stats.get("current_phase") or "training"),
                signals=progress_signals,
                stdout_lines=int(stream_stats["stdout_lines"]),
                stdout_bytes=int(stream_stats["stdout_bytes"]),
                metric_history_text=str(stream_stats.get("metric_history_text") or ""),
                metric_history_line_count=len(stream_stats.get("metric_history_lines") or []),
                emit_reason=emit_reason,
            )

        def _update_resource_stream_stats(text: str, *, stderr: bool = False) -> None:
            if not text:
                return
            if not stderr:
                stream_stats["stdout_bytes"] = int(stream_stats["stdout_bytes"]) + len(
                    text.encode(errors="replace"),
                )
                lines = text.splitlines() or [text]
                stream_stats["stdout_lines"] = int(stream_stats["stdout_lines"]) + len(lines)
            if _FINAL_VALIDATION_SCORE_RE.search(text):
                stream_stats["saw_final_score"] = True
                stream_stats["current_phase"] = "final_scoring"
            if _TRAINING_PROGRESS_RE.search(text):
                stream_stats["saw_training_progress"] = True
                if not stream_stats.get("current_phase"):
                    stream_stats["current_phase"] = "training"
            if _VALIDATION_OR_INFERENCE_RE.search(text):
                stream_stats["current_phase"] = "validation_or_inference"
            health = scan_output_health(text)
            metric_history_lines = _update_metric_history_lines(
                stream_stats.get("metric_history_lines") or [],
                text,
            )
            if metric_history_lines != stream_stats.get("metric_history_lines"):
                stream_stats["metric_history_lines"] = metric_history_lines
                stream_stats["metric_history_text"] = _metric_history_text(metric_history_lines)
            if health.get("invalid_metric_events"):
                stream_stats["invalid_metric_events"] = int(stream_stats.get("invalid_metric_events") or 0) + int(
                    health.get("invalid_metric_events") or 0,
                )
                stream_stats["last_invalid_metric_text"] = str(health.get("last_invalid_metric_text") or "")
                stream_stats["saw_training_progress"] = True
            if health.get("zero_score_events"):
                stream_stats["zero_score_events"] = int(stream_stats.get("zero_score_events") or 0) + int(
                    health.get("zero_score_events") or 0,
                )
                stream_stats["last_zero_score_text"] = str(health.get("last_zero_score_text") or "")
                stream_stats["saw_training_progress"] = True
            if health.get("terminal_signal_events"):
                stream_stats["terminal_signal_events"] = int(stream_stats.get("terminal_signal_events") or 0) + int(
                    health.get("terminal_signal_events") or 0,
                )
                stream_stats["terminal_signal_kind"] = str(health.get("terminal_signal_kind") or "")
                stream_stats["last_terminal_signal_text"] = str(health.get("last_terminal_signal_text") or "")
            progress_signals = _parse_progress_signals(text)
            if progress_signals:
                stream_stats["saw_training_progress"] = True
                stream_stats["last_progress_signals"] = progress_signals
                heartbeat_phase = str(progress_signals.get("phase") or "").strip()
                if heartbeat_phase:
                    stream_stats["current_phase"] = heartbeat_phase
                elif not stream_stats.get("current_phase"):
                    stream_stats["current_phase"] = "training"
                _maybe_emit_progress_heartbeat(
                    progress_signals,
                    reason="scienceflow_hb" if "heartbeat" in progress_signals else "progress",
                )

        def _maybe_emit_artifact_progress(elapsed_sec: float, *, force: bool = False) -> None:
            now_m = time.monotonic()
            if not force and now_m - last_artifact_scan[0] < artifact_scan_interval:
                return
            last_artifact_scan[0] = now_m
            artifact = _latest_artifact_snapshot(
                ws,
                started_at=start_wall,
                run_artifact_dir=run_artifact_dir,
                run_state_path=run_state_path,
                stability_cache=artifact_watch_cache,
                settle_sec=float(self.resource_artifact_recoverable_settle_sec or 0.0),
            )
            if not artifact:
                return
            artifact_key = (
                str(artifact.get("path") or ""),
                float(artifact.get("mtime") or 0.0),
                int(artifact.get("size_bytes") or 0),
                str(artifact.get("stability") or ""),
                bool(artifact.get("recoverable_artifact_on_disk")),
            )
            stream_stats["last_artifact"] = artifact
            if artifact_key == stream_stats.get("last_artifact_emitted_key"):
                return
            stream_stats["last_artifact_emitted"] = artifact
            stream_stats["last_artifact_emitted_key"] = artifact_key
            signals = dict(stream_stats.get("last_progress_signals") or {})
            signals["artifact"] = artifact
            stream_stats["last_progress_signals"] = signals
            _resource_call(
                self.resource_observer,
                "progress_heartbeat",
                resource_job_id,
                elapsed_sec=max(0.0, float(elapsed_sec or 0.0)),
                phase=str(stream_stats.get("current_phase") or "artifact_update"),
                signals=signals,
                stdout_lines=int(stream_stats["stdout_lines"]),
                stdout_bytes=int(stream_stats["stdout_bytes"]),
                emit_reason="artifact",
            )

        workspace_cleanup_done = [False]

        def _recoverable_stop_marker_seen() -> bool:
            if run_state_path is None:
                return False
            try:
                stat = run_state_path.stat()
            except OSError:
                return False
            if stat.st_mtime + 1.0 < start_wall or stat.st_size <= 0:
                return False
            try:
                state = json.loads(run_state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return False
            if not isinstance(state, dict):
                return False
            return str(state.get("status") or "").strip().lower() in {"interrupted", "completed"}

        async def _terminate_resource_guard_process() -> None:
            if not self.resource_recoverable_stop_enabled:
                await terminate_tree(proc)
                return
            await terminate_tree_recoverable(
                proc,
                marker_check=_recoverable_stop_marker_seen,
                sigusr1_grace=float(self.resource_recoverable_stop_sigusr1_grace_sec or 0.0),
                marker_exit_grace=float(self.resource_recoverable_stop_marker_exit_grace_sec or 0.0),
                sigterm_grace=float(self.resource_recoverable_stop_sigterm_grace_sec or 0.0),
            )

        async def _run_workspace_gpu_cleanup(*, reason: str, force: bool = False) -> dict[str, Any]:
            if workspace_cleanup_done[0] and not force:
                return {}
            cleanup_gpu_ids = task_physical_gpu_pool or gpu_ids
            if not cleanup_gpu_ids or self.resource_observer is None:
                return {}
            workspace_cleanup_done[0] = True
            try:
                cleanup = await asyncio.to_thread(
                    _cleanup_workspace_gpu_processes,
                    workspace_dir=ws,
                    allowed_gpu_ids=cleanup_gpu_ids,
                    task_started_at=start_wall,
                    reason=reason,
                    dry_run=False,
                    sigterm_grace_sec=0.5,
                )
            except Exception:
                logger.debug("[bash-resource] workspace gpu cleanup failed", exc_info=True)
                return {}
            if not isinstance(cleanup, dict):
                return {}
            killed_count = int(cleanup.get("killed_count") or 0)
            violation_count = int(cleanup.get("violation_count") or 0)
            skipped = cleanup.get("skipped") if isinstance(cleanup.get("skipped"), list) else []
            interesting_skips = [
                row for row in skipped
                if str((row or {}).get("skip_reason") or "") not in {
                    "pid_disappeared",
                    "workspace_mismatch",
                    "process_started_before_task",
                }
            ]
            if killed_count > 0:
                action = "gpu_orphan_cleanup"
            elif violation_count > 0:
                action = "stop_boundary_violation"
            elif interesting_skips:
                _resource_call(
                    self.resource_observer,
                    "resource_cleanup_heartbeat",
                    resource_job_id,
                    action="gpu_cleanup_skipped",
                    reason=reason,
                    elapsed_sec=max(0.0, time.time() - start),
                    placement=cleanup,
                )
                return cleanup
            else:
                return cleanup
            _resource_call(
                self.resource_observer,
                "resource_guard_action",
                resource_job_id,
                action=action,
                reason=reason,
                elapsed_sec=max(0.0, time.time() - start),
                placement=cleanup,
            )
            return cleanup

        async def _emit_line(text: str, prefix: str) -> None:
            """Forward a decoded line to *on_output* with truncation bookkeeping."""
            if on_output is None or stream_state["truncated"]:
                return
            if heartbeat_active[0]:
                on_output("\r\033[K")
                heartbeat_active[0] = False
            show = _sanitize_model_visible_output_paths(
                text,
                ws,
                self.path_guard_extra_roots or (),
                strip_symlink_targets=bool(self.strip_symlink_targets),
            )
            if line_cap > 0 and len(show) > line_cap:
                show = show[:line_cap] + "… [line truncated]\n"
            chunk = f"{prefix}{show}"
            if total_cap > 0:
                if stream_state["sent"] + len(chunk) > total_cap:
                    room = total_cap - int(stream_state["sent"])
                    if room > 120:
                        on_output(chunk[:room])
                    shown = int(stream_state["sent"]) + min(len(chunk), max(0, room))
                    on_output(
                        f"{prefix}… [live stdout truncated: displayed ~{shown} chars "
                        f"of stream mirror; full capture still used for tool result "
                        f"(trimmed to max_output_chars={total_cap})]\n",
                    )
                    stream_state["truncated"] = True
                    return
            stream_state["sent"] = int(stream_state["sent"]) + len(chunk)
            on_output(chunk)

        async def _drain_stream(
            stream: asyncio.StreamReader | None,
            parts: list[str],
            prefix: str,
        ) -> None:
            if stream is None:
                return
            _READ_CHUNK = 1024 * 1024  # 1 MiB fallback chunk size
            while True:
                try:
                    line = await stream.readline()
                except ValueError:
                    # LimitOverrunError (wrapped as ValueError): a single line
                    # exceeded the StreamReader buffer.  Fall back to chunk reads
                    # so we never lose output or crash the agent loop.
                    while True:
                        chunk_bytes = await stream.read(_READ_CHUNK)
                        if not chunk_bytes:
                            break
                        text = chunk_bytes.decode(errors="replace")
                        _touch_io()
                        _update_resource_stream_stats(text, stderr=bool(prefix))
                        _resource_call(
                            self.resource_observer,
                            "stdout_heartbeat",
                            resource_job_id,
                        )
                        parts.append(text)
                        await _emit_line(text, prefix)
                    break
                if not line:
                    break
                text = line.decode(errors="replace")
                _touch_io()
                _update_resource_stream_stats(text, stderr=bool(prefix))
                _resource_call(
                    self.resource_observer,
                    "stdout_heartbeat",
                    resource_job_id,
                )
                parts.append(text)
                await _emit_line(text, prefix)

        hb_task: asyncio.Task[None] | None = None
        if on_output is not None and interval_hb > 0:

            async def _heartbeat() -> None:
                try:
                    while True:
                        await asyncio.sleep(0.25)
                        now_m = time.monotonic()
                        wall_elapsed = time.time() - start_wall
                        remaining = max(0.0, timeout - wall_elapsed)
                        if now_m - io_touch[0] < idle_after:
                            continue
                        if last_hb_emit[0] > 0 and (now_m - last_hb_emit[0]) < interval_hb:
                            continue
                        last_hb_emit[0] = now_m
                        on_output(
                            f"\r\033[K[bash … {wall_elapsed:.0f}s elapsed, "
                            f"{remaining:.0f}s until timeout]",
                        )
                        heartbeat_active[0] = True
                except asyncio.CancelledError:
                    raise

            hb_task = asyncio.create_task(_heartbeat())

        async def _resource_guard() -> None:
            try:
                interval = 0.25
                while True:
                    await asyncio.sleep(max(0.05, interval))
                    if getattr(proc, "returncode", None) is not None:
                        return
                    elapsed_now = time.time() - start
                    stdout_age = time.monotonic() - io_touch[0]
                    _maybe_emit_artifact_progress(elapsed_now)
                    if (
                        observe_first_active
                        and observe_first_window_sec > 0
                        and not observe_first_health_emitted[0]
                        and elapsed_now >= observe_first_window_sec
                    ):
                        observe_first_health_emitted[0] = True
                        _resource_call(
                            self.resource_observer,
                            "observe_first_healthy_continue",
                            resource_job_id,
                            elapsed_sec=elapsed_now,
                            stdout_age_sec=stdout_age,
                            stdout_lines=int(stream_stats["stdout_lines"]),
                            stdout_bytes=int(stream_stats["stdout_bytes"]),
                            metric_history_text=str(stream_stats.get("metric_history_text") or ""),
                            metric_history_line_count=len(stream_stats.get("metric_history_lines") or []),
                            saw_training_progress=bool(stream_stats["saw_training_progress"]),
                            saw_final_score=bool(stream_stats["saw_final_score"]),
                            current_phase=str(stream_stats.get("current_phase") or ""),
                        )
                    placement_interval = 5.0 if elapsed_now < 120.0 else 15.0
                    if gpu_ids and time.monotonic() - placement_audit_last[0] >= placement_interval:
                        placement_audit_last[0] = time.monotonic()
                        placement = _process_tree_gpu_placement_snapshot(getattr(proc, "pid", None), gpu_ids)
                        violations = placement.get("violations") if isinstance(placement, dict) else []
                        if violations:
                            reason = "task_gpu_boundary_violation"
                            guard_reason[0] = reason
                            used_ids = sorted({
                                str(row.get("gpu_id") or "")
                                for row in violations
                                if isinstance(row, dict) and str(row.get("gpu_id") or "").strip()
                            })
                            feedback = _build_gpu_boundary_feedback(
                                actual_gpu_ids=used_ids,
                                allowed_gpu_ids=gpu_ids,
                                command_scope="command",
                            )
                            guard_feedback[0] = feedback
                            _resource_call(
                                self.resource_observer,
                                "resource_guard_action",
                                resource_job_id,
                                action="stop_boundary_violation",
                                reason=reason,
                                elapsed_sec=elapsed_now,
                                placement=placement,
                            )
                            if on_output is not None:
                                if heartbeat_active[0]:
                                    on_output("\r\033[K")
                                    heartbeat_active[0] = False
                                on_output(f"[resource guard] stopping boundary violation after {elapsed_now:.0f}s: {reason}\n")
                                on_output(feedback.rstrip() + "\n")
                            await terminate_tree(proc)
                            await _run_workspace_gpu_cleanup(
                                reason="workspace_cleanup_after_boundary_violation",
                                force=True,
                            )
                            return
                    parent_state = "stalled_but_alive" if stdout_age >= max(60.0, float(self.bash_heartbeat_idle_sec)) else "healthy_running"
                    _resource_call(
                        self.resource_observer,
                        "maybe_run_sidecar_backfill",
                        resource_job_id,
                        elapsed_sec=elapsed_now,
                        parent_state=parent_state,
                    )
                    process_tree_cpu = _process_tree_cpu_snapshot(getattr(proc, "pid", None))
                    monitor_result = _resource_call(
                        self.resource_observer,
                        "monitor_heartbeat",
                        resource_job_id,
                        elapsed_sec=elapsed_now,
                        stdout_age_sec=stdout_age,
                        stdout_lines=int(stream_stats["stdout_lines"]),
                        stdout_bytes=int(stream_stats["stdout_bytes"]),
                        pid=getattr(proc, "pid", None),
                        returncode=getattr(proc, "returncode", None),
                        process_tree_cpu=process_tree_cpu,
                        source="bash_guard",
                    )
                    if isinstance(monitor_result, dict) and monitor_result.get("recorded"):
                        last_monitor_heartbeat_wall[0] = time.time()
                    decision = _resource_call(
                        self.resource_observer,
                        "stalled_guard_decision",
                        resource_job_id,
                        inferred_class=effective_resource_class,
                        gpu_ids=gpu_ids,
                        elapsed_sec=elapsed_now,
                        stdout_age_sec=stdout_age,
                        process_tree_cpu=process_tree_cpu,
                    )
                    if not isinstance(decision, dict) or not decision.get("enabled"):
                        decision = {}
                    interval = float(decision.get("check_interval_sec") or interval or 1.0)
                    if not decision.get("terminate") and not decision.get("would_terminate"):
                        decision = _resource_call(
                            self.resource_observer,
                            "active_intervention_decision",
                            resource_job_id,
                            inferred_class=effective_resource_class,
                            gpu_ids=gpu_ids,
                            elapsed_sec=elapsed_now,
                            stdout_age_sec=stdout_age,
                            stdout_lines=int(stream_stats["stdout_lines"]),
                            stdout_bytes=int(stream_stats["stdout_bytes"]),
                            metric_history_text=str(stream_stats.get("metric_history_text") or ""),
                            metric_history_line_count=len(stream_stats.get("metric_history_lines") or []),
                            saw_training_progress=bool(stream_stats["saw_training_progress"]),
                            saw_final_score=bool(stream_stats["saw_final_score"]),
                            current_phase=str(stream_stats.get("current_phase") or ""),
                            invalid_metric_events=int(stream_stats.get("invalid_metric_events") or 0),
                            zero_score_events=int(stream_stats.get("zero_score_events") or 0),
                            last_invalid_metric_text=str(stream_stats.get("last_invalid_metric_text") or ""),
                            last_zero_score_text=str(stream_stats.get("last_zero_score_text") or ""),
                            terminal_signal_events=int(stream_stats.get("terminal_signal_events") or 0),
                            terminal_signal_kind=str(stream_stats.get("terminal_signal_kind") or ""),
                            last_terminal_signal_text=str(stream_stats.get("last_terminal_signal_text") or ""),
                            process_tree_cpu=process_tree_cpu,
                            **self._deadline_state(),
                        )
                        if not isinstance(decision, dict) or not decision.get("enabled"):
                            continue
                        interval = float(decision.get("check_interval_sec") or interval or 1.0)
                    if (
                        decision.get("would_terminate")
                        or decision.get("arbiter_review")
                        or decision.get("requires_llm_decision")
                    ) and not decision.get("terminate"):
                        review_only = bool(decision.get("arbiter_review")) and not bool(decision.get("would_terminate"))
                        if guard_observe_more_until[0] and time.time() < guard_observe_more_until[0]:
                            continue
                        if guard_observe_more_until[0] and time.time() >= guard_observe_more_until[0]:
                            guard_observe_more_until[0] = 0.0
                        internal_arbiter_review = bool(decision.get("arbiter_enabled") and decision.get("requires_llm_decision"))
                        if guard_recommendation_emitted[0] and not review_only and not internal_arbiter_review:
                            continue
                        reason = str(decision.get("reason") or "resource_guard_recommendation")
                        feedback = str(decision.get("feedback") or "").strip()
                        if decision.get("arbiter_enabled"):
                            arbiter = await _resource_async_call(
                                self.resource_observer,
                                "arbiter_decide",
                                resource_job_id,
                                proposal=decision.get("proposal"),
                                decision_preview=decision,
                            )
                            if isinstance(arbiter, dict) and arbiter.get("enabled"):
                                action = str(arbiter.get("action") or "").upper()
                                arbiter_feedback = str(arbiter.get("feedback") or "").strip()
                                if arbiter_feedback:
                                    feedback = arbiter_feedback
                                    guard_feedback[0] = feedback
                                if action == "GRANT_SHARED_GPU_LEASE":
                                    grant = _resource_call(
                                        self.resource_observer,
                                        "apply_task_gpu_share_decision",
                                        resource_job_id,
                                        arbiter_action=action,
                                        proposal=decision.get("proposal"),
                                        arbiter_decision=arbiter.get("decision"),
                                        elapsed_sec=elapsed_now,
                                    )
                                    _resource_call(
                                        self.resource_observer,
                                        "resource_guard_action",
                                        resource_job_id,
                                        action="grant_shared_gpu_lease" if isinstance(grant, dict) and grant.get("granted") else "grant_shared_gpu_lease_denied",
                                        reason=reason,
                                        elapsed_sec=elapsed_now,
                                    )
                                    if on_output is not None:
                                        if heartbeat_active[0]:
                                            on_output("\r\033[K")
                                            heartbeat_active[0] = False
                                        state = "granted shared GPU lease" if isinstance(grant, dict) and grant.get("granted") else "could not grant shared GPU lease"
                                        on_output(f"[resource arbiter] {state} after {elapsed_now:.0f}s: {reason}\n")
                                        grant_feedback = str((grant or {}).get("feedback") or feedback).strip() if isinstance(grant, dict) else feedback
                                        if grant_feedback:
                                            on_output(grant_feedback.rstrip() + "\n")
                                    guard_observe_more_until[0] = time.time() + 30.0
                                    continue
                                if action == "RELEASE_IDLE_LEASE":
                                    release = _resource_call(
                                        self.resource_observer,
                                        "release_idle_gpu_lease",
                                        resource_job_id,
                                        elapsed_sec=elapsed_now,
                                        reason=reason,
                                        feedback=feedback,
                                    )
                                    _resource_call(
                                        self.resource_observer,
                                        "resource_guard_action",
                                        resource_job_id,
                                        action="release_idle_lease" if isinstance(release, dict) and release.get("released") else "release_idle_lease_denied",
                                        reason=reason,
                                        elapsed_sec=elapsed_now,
                                    )
                                    if on_output is not None:
                                        if heartbeat_active[0]:
                                            on_output("\r\033[K")
                                            heartbeat_active[0] = False
                                        state = "released idle GPU lease" if isinstance(release, dict) and release.get("released") else "could not release idle GPU lease"
                                        on_output(f"[resource arbiter] {state} after {elapsed_now:.0f}s: {reason}\n")
                                        if feedback:
                                            on_output(feedback.rstrip() + "\n")
                                    guard_observe_more_until[0] = time.time() + 300.0
                                    continue
                                if action in {"KILL_AND_REPLAN", "STOP_BOUNDARY_VIOLATION"}:
                                    guard_reason[0] = reason
                                    guard_action = {
                                        "STOP_BOUNDARY_VIOLATION": "stop_boundary_violation",
                                    }.get(action, "terminate_by_arbiter")
                                    decision_payload = arbiter.get("decision") if isinstance(arbiter.get("decision"), dict) else {}
                                    gate_payload = decision_payload.get("gate") if isinstance(decision_payload.get("gate"), dict) else {}
                                    if action == "KILL_AND_REPLAN":
                                        _maybe_emit_artifact_progress(elapsed_now, force=True)
                                        fresh_process_tree_cpu = _process_tree_cpu_snapshot(getattr(proc, "pid", None))
                                        revalidation = _resource_call(
                                            self.resource_observer,
                                            "revalidate_kill_intent",
                                            resource_job_id,
                                            arbiter_decision=decision_payload,
                                            elapsed_sec=elapsed_now,
                                            stdout_age_sec=stdout_age,
                                            stdout_lines=int(stream_stats["stdout_lines"]),
                                            stdout_bytes=int(stream_stats["stdout_bytes"]),
                                            metric_history_text=str(stream_stats.get("metric_history_text") or ""),
                                            metric_history_line_count=len(stream_stats.get("metric_history_lines") or []),
                                            saw_training_progress=bool(stream_stats["saw_training_progress"]),
                                            saw_final_score=bool(stream_stats["saw_final_score"]),
                                            current_phase=str(stream_stats.get("current_phase") or ""),
                                            invalid_metric_events=int(stream_stats.get("invalid_metric_events") or 0),
                                            zero_score_events=int(stream_stats.get("zero_score_events") or 0),
                                            last_invalid_metric_text=str(stream_stats.get("last_invalid_metric_text") or ""),
                                            last_zero_score_text=str(stream_stats.get("last_zero_score_text") or ""),
                                            terminal_signal_events=int(stream_stats.get("terminal_signal_events") or 0),
                                            terminal_signal_kind=str(stream_stats.get("terminal_signal_kind") or ""),
                                            last_terminal_signal_text=str(stream_stats.get("last_terminal_signal_text") or ""),
                                            process_tree_cpu=fresh_process_tree_cpu,
                                            **self._deadline_state(),
                                        )
                                        hard_safety = str(gate_payload.get("kill_class") or "") == "hard_safety"
                                        if not _kill_revalidation_allows_termination(
                                            revalidation,
                                            hard_safety=hard_safety,
                                        ):
                                            revalidation_payload = revalidation if isinstance(revalidation, dict) else {}
                                            revalidation_available = bool(
                                                revalidation_payload.get("enabled") is True
                                                and "allow_kill" in revalidation_payload
                                            )
                                            revalidation_reason = str(
                                                revalidation_payload.get("reason")
                                                or (
                                                    "stale_kill_intent"
                                                    if revalidation_available
                                                    else "kill_revalidation_unavailable"
                                                )
                                            )
                                            protective_changes = revalidation_payload.get("protective_changes") or []
                                            guard_reason[0] = ""
                                            guard_observe_more_until[0] = time.time() + 120.0
                                            _resource_call(
                                                self.resource_observer,
                                                "resource_guard_action",
                                                resource_job_id,
                                                action=(
                                                    "stale_kill_intent_denied"
                                                    if revalidation_available
                                                    else "kill_revalidation_unavailable"
                                                ),
                                                reason=revalidation_reason,
                                                elapsed_sec=elapsed_now,
                                                protective_changes=protective_changes,
                                            )
                                            if on_output is not None:
                                                if heartbeat_active[0]:
                                                    on_output("\r\033[K")
                                                    heartbeat_active[0] = False
                                                detail = ",".join(str(item) for item in protective_changes)
                                                if not detail:
                                                    detail = revalidation_reason
                                                on_output(
                                                    f"[resource arbiter] kill denied by revalidation after "
                                                    f"{elapsed_now:.0f}s: {detail}\n"
                                                )
                                            continue
                                    await _terminate_resource_guard_process()
                                    _resource_call(
                                        self.resource_observer,
                                        "resource_guard_action",
                                        resource_job_id,
                                        action=guard_action,
                                        reason=reason,
                                        elapsed_sec=elapsed_now,
                                        arbiter_decision_id=decision_payload.get("decision_id"),
                                        proposal_id=decision_payload.get("proposal_id") or arbiter.get("proposal_id"),
                                        kill_class=gate_payload.get("kill_class"),
                                        strict_gate_result="allow" if bool(gate_payload.get("allowed")) else "blocked",
                                        strict_gate_reason=gate_payload.get("blocked_reason"),
                                        advisory_support=gate_payload.get("advisory_support"),
                                        captured_advisory=gate_payload.get("captured_advisory"),
                                    )
                                    if on_output is not None:
                                        if heartbeat_active[0]:
                                            on_output("\r\033[K")
                                            heartbeat_active[0] = False
                                        verb = "KILL EXECUTED; command process has terminated"
                                        if action == "STOP_BOUNDARY_VIOLATION":
                                            verb = "STOP EXECUTED; command process has terminated"
                                        on_output(f"[resource arbiter] {verb} after {elapsed_now:.0f}s: {reason}\n")
                                        executed_feedback = _executed_resource_termination_feedback(
                                            feedback,
                                            action=action,
                                        )
                                        on_output(executed_feedback.rstrip() + "\n")
                                    return
                                if action in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL", "CONTINUE", "DENY_KILL", "CONTINUE_SHARED_OBSERVE", "DENY_SHARE_USE_CPU_SUPPORT"}:
                                    if action in {"DENY_SHARE_USE_CPU_SUPPORT", "CONTINUE_SHARED_OBSERVE"}:
                                        _resource_call(
                                            self.resource_observer,
                                            "apply_task_gpu_share_decision",
                                            resource_job_id,
                                            arbiter_action=action,
                                            proposal=decision.get("proposal"),
                                            arbiter_decision=arbiter.get("decision"),
                                            elapsed_sec=elapsed_now,
                                        )
                                    arbiter_decision = arbiter.get("decision", {}) if isinstance(arbiter.get("decision"), dict) else {}
                                    execution_outcome = str(arbiter.get("execution_outcome") or arbiter_decision.get("execution_outcome") or "").upper()
                                    if execution_outcome == "TIMEBOX":
                                        ttl = arbiter.get("computed_timebox_sec") or arbiter_decision.get("computed_timebox_sec")
                                    elif action in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL", "CONTINUE_SHARED_OBSERVE"}:
                                        ttl = arbiter_decision.get("observe_more_sec")
                                    else:
                                        ttl = arbiter_decision.get("ttl_sec")
                                    observe_more_sec = float(ttl or 300.0)
                                    guard_observe_more_until[0] = time.time() + max(1.0, observe_more_sec)
                                else:
                                    guard_recommendation_emitted[0] = True
                                arbiter_decision = arbiter.get("decision", {}) if isinstance(arbiter.get("decision"), dict) else {}
                                execution_outcome = str(arbiter.get("execution_outcome") or arbiter_decision.get("execution_outcome") or "").upper()
                                action_name = "arbiter_timebox" if execution_outcome == "TIMEBOX" else {
                                    "OBSERVE_MORE": "arbiter_observe_more",
                                    "MARK_STALLED_NO_KILL": "arbiter_mark_stalled_no_kill",
                                    "CONTINUE": "arbiter_continue",
                                    "DENY_KILL": "arbiter_deny_kill",
                                    "CONTINUE_SHARED_OBSERVE": "arbiter_continue_shared_observe",
                                    "DENY_SHARE_USE_CPU_SUPPORT": "arbiter_deny_share_use_cpu_support",
                                }.get(action, "arbiter_no_action")
                                _resource_call(
                                    self.resource_observer,
                                    "resource_guard_action",
                                    resource_job_id,
                                    action=action_name,
                                    reason=reason,
                                    elapsed_sec=elapsed_now,
                                )
                                suppress_feedback = bool(arbiter.get("suppress_main_agent_feedback"))
                                if on_output is not None and not suppress_feedback:
                                    if heartbeat_active[0]:
                                        on_output("\r\033[K")
                                        heartbeat_active[0] = False
                                    on_output(f"[resource arbiter] {action.lower()} after {elapsed_now:.0f}s: {reason}\n")
                                    if feedback:
                                        on_output(feedback.rstrip() + "\n")
                                if action in {"CONTINUE", "DENY_KILL", "OBSERVE_MORE", "MARK_STALLED_NO_KILL"}:
                                    post_share = _resource_call(
                                        self.resource_observer,
                                        "post_arbiter_continue_review",
                                        resource_job_id,
                                        arbiter_action=action,
                                        inferred_class=effective_resource_class,
                                        gpu_ids=gpu_ids,
                                        elapsed_sec=elapsed_now,
                                        stdout_age_sec=stdout_age,
                                        stdout_lines=int(stream_stats["stdout_lines"]),
                                        stdout_bytes=int(stream_stats["stdout_bytes"]),
                                        metric_history_text=str(stream_stats.get("metric_history_text") or ""),
                                        metric_history_line_count=len(stream_stats.get("metric_history_lines") or []),
                                        saw_training_progress=bool(stream_stats["saw_training_progress"]),
                                        saw_final_score=bool(stream_stats["saw_final_score"]),
                                        current_phase=str(stream_stats.get("current_phase") or ""),
                                        invalid_metric_events=int(stream_stats.get("invalid_metric_events") or 0),
                                        zero_score_events=int(stream_stats.get("zero_score_events") or 0),
                                        last_invalid_metric_text=str(stream_stats.get("last_invalid_metric_text") or ""),
                                        last_zero_score_text=str(stream_stats.get("last_zero_score_text") or ""),
                                        terminal_signal_events=int(stream_stats.get("terminal_signal_events") or 0),
                                        terminal_signal_kind=str(stream_stats.get("terminal_signal_kind") or ""),
                                        last_terminal_signal_text=str(stream_stats.get("last_terminal_signal_text") or ""),
                                        process_tree_cpu=_process_tree_cpu_snapshot(getattr(proc, "pid", None)),
                                        **self._deadline_state(),
                                    )
                                    if isinstance(post_share, dict) and post_share.get("arbiter_review") and post_share.get("arbiter_enabled"):
                                        share_arbiter = await _resource_async_call(
                                            self.resource_observer,
                                            "arbiter_decide",
                                            resource_job_id,
                                            proposal=post_share.get("proposal"),
                                            decision_preview=post_share,
                                        )
                                        if isinstance(share_arbiter, dict) and share_arbiter.get("enabled"):
                                            share_action = str(share_arbiter.get("action") or "").upper()
                                            if share_action in {"GRANT_SHARED_GPU_LEASE", "DENY_SHARE_USE_CPU_SUPPORT", "CONTINUE_SHARED_OBSERVE"}:
                                                share_apply = _resource_call(
                                                    self.resource_observer,
                                                    "apply_task_gpu_share_decision",
                                                    resource_job_id,
                                                    arbiter_action=share_action,
                                                    proposal=post_share.get("proposal"),
                                                    arbiter_decision=share_arbiter.get("decision"),
                                                    elapsed_sec=elapsed_now,
                                                )
                                                if on_output is not None:
                                                    if heartbeat_active[0]:
                                                        on_output("\r\033[K")
                                                        heartbeat_active[0] = False
                                                    state = "granted shared GPU lease" if isinstance(share_apply, dict) and share_apply.get("granted") else share_action.lower()
                                                    on_output(f"[resource arbiter] {state} after {elapsed_now:.0f}s: task_gpu_share_review\n")
                                                    share_feedback = str((share_apply or {}).get("feedback") or share_arbiter.get("feedback") or "").strip() if isinstance(share_apply, dict) else str(share_arbiter.get("feedback") or "").strip()
                                                    if share_feedback:
                                                        on_output(share_feedback.rstrip() + "\n")
                                continue
                        guard_recommendation_emitted[0] = True
                        if feedback:
                            guard_feedback[0] = feedback
                        _resource_call(
                            self.resource_observer,
                            "resource_guard_action",
                            resource_job_id,
                            action="recommend_stop",
                            reason=reason,
                            elapsed_sec=elapsed_now,
                        )
                        if on_output is not None:
                            if heartbeat_active[0]:
                                on_output("\r\033[K")
                                heartbeat_active[0] = False
                            on_output(
                                f"[resource guard] recommends LLM review "
                                f"after {elapsed_now:.0f}s: {reason}; command not terminated\n",
                            )
                            if feedback:
                                on_output(feedback.rstrip() + "\n")
                        continue
                    if decision.get("release_idle_lease") and not decision.get("terminate"):
                        reason = str(decision.get("reason") or "active_intervention:idle_gpu_lease_release")
                        feedback = str(decision.get("feedback") or "").strip()
                        release = _resource_call(
                            self.resource_observer,
                            "release_idle_gpu_lease",
                            resource_job_id,
                            elapsed_sec=elapsed_now,
                            reason=reason,
                            feedback=feedback,
                        )
                        if on_output is not None:
                            if heartbeat_active[0]:
                                on_output("\r\033[K")
                                heartbeat_active[0] = False
                            state = "released idle GPU lease" if isinstance(release, dict) and release.get("released") else "could not release idle GPU lease"
                            on_output(f"[resource guard] {state} after {elapsed_now:.0f}s: {reason}\n")
                            if feedback:
                                on_output(feedback.rstrip() + "\n")
                        guard_observe_more_until[0] = time.time() + 300.0
                        continue
                    if not decision.get("terminate"):
                        continue
                    reason = str(decision.get("reason") or "stalled_heavy")
                    guard_reason[0] = reason
                    feedback = str(decision.get("feedback") or "").strip()
                    if feedback:
                        guard_feedback[0] = feedback
                    _resource_call(
                        self.resource_observer,
                        "resource_guard_action",
                        resource_job_id,
                        action="terminate",
                        reason=reason,
                        elapsed_sec=elapsed_now,
                    )
                    if on_output is not None:
                        if heartbeat_active[0]:
                            on_output("\r\033[K")
                            heartbeat_active[0] = False
                        on_output(
                            f"[resource guard] terminating command "
                            f"after {elapsed_now:.0f}s: {reason}\n",
                        )
                        if feedback:
                            on_output(feedback.rstrip() + "\n")
                    await _terminate_resource_guard_process()
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("[bash-resource] stalled guard failed", exc_info=True)

        async def _resource_guard_watchdog() -> None:
            nonlocal guard_task
            poll_sec = max(1.0, min(30.0, float(self.bash_heartbeat_interval_sec or 5.0)))
            try:
                observer_interval = float(
                    getattr(self.resource_observer, "review_heartbeat_sec", 60.0) or 60.0,
                )
            except (TypeError, ValueError):
                observer_interval = 60.0
            stale_after_sec = max(2.0 * poll_sec, 2.0 * observer_interval)
            last_gap_emit_wall: list[float] = [0.0]

            async def _record_gap_and_review(gap_reason: str, gap_sec: float) -> bool:
                elapsed_now = time.time() - start
                stdout_age = time.monotonic() - io_touch[0]
                process_tree_cpu = _process_tree_cpu_snapshot(getattr(proc, "pid", None))
                _resource_call(
                    self.resource_observer,
                    "resource_monitor_gap",
                    resource_job_id,
                    elapsed_sec=elapsed_now,
                    gap_sec=gap_sec,
                    stdout_age_sec=stdout_age,
                    stdout_lines=int(stream_stats["stdout_lines"]),
                    stdout_bytes=int(stream_stats["stdout_bytes"]),
                    reason=gap_reason,
                    source="bash_guard_watchdog",
                    process_tree_cpu=process_tree_cpu,
                )
                decision = _resource_call(
                    self.resource_observer,
                    "active_intervention_decision",
                    resource_job_id,
                    inferred_class=effective_resource_class,
                    gpu_ids=gpu_ids,
                    elapsed_sec=elapsed_now,
                    stdout_age_sec=stdout_age,
                    stdout_lines=int(stream_stats["stdout_lines"]),
                    stdout_bytes=int(stream_stats["stdout_bytes"]),
                    metric_history_text=str(stream_stats.get("metric_history_text") or ""),
                    metric_history_line_count=len(stream_stats.get("metric_history_lines") or []),
                    saw_training_progress=bool(stream_stats["saw_training_progress"]),
                    saw_final_score=bool(stream_stats["saw_final_score"]),
                    current_phase=str(stream_stats.get("current_phase") or ""),
                    invalid_metric_events=int(stream_stats.get("invalid_metric_events") or 0),
                    zero_score_events=int(stream_stats.get("zero_score_events") or 0),
                    last_invalid_metric_text=str(stream_stats.get("last_invalid_metric_text") or ""),
                    last_zero_score_text=str(stream_stats.get("last_zero_score_text") or ""),
                    terminal_signal_events=int(stream_stats.get("terminal_signal_events") or 0),
                    terminal_signal_kind=str(stream_stats.get("terminal_signal_kind") or ""),
                    last_terminal_signal_text=str(stream_stats.get("last_terminal_signal_text") or ""),
                    process_tree_cpu=process_tree_cpu,
                    **self._deadline_state(),
                )
                if not isinstance(decision, dict):
                    decision = {}
                if decision.get("terminate"):
                    reason = str(decision.get("reason") or "resource_monitor_gap")
                    feedback = str(decision.get("feedback") or "").strip()
                    guard_reason[0] = reason
                    if feedback:
                        guard_feedback[0] = feedback
                    _resource_call(
                        self.resource_observer,
                        "resource_guard_action",
                        resource_job_id,
                        action="terminate_by_monitor_watchdog",
                        reason=reason,
                        elapsed_sec=elapsed_now,
                    )
                    if on_output is not None:
                        if heartbeat_active[0]:
                            on_output("\r\033[K")
                            heartbeat_active[0] = False
                        on_output(f"[resource watchdog] terminating command after {elapsed_now:.0f}s: {reason}\n")
                        if feedback:
                            on_output(feedback.rstrip() + "\n")
                    await _terminate_resource_guard_process()
                    return True
                if decision.get("would_terminate") or decision.get("arbiter_review") or decision.get("requires_llm_decision"):
                    _resource_call(
                        self.resource_observer,
                        "resource_guard_action",
                        resource_job_id,
                        action="monitor_gap_review_requested",
                        reason=str(decision.get("reason") or gap_reason),
                        elapsed_sec=elapsed_now,
                    )
                return False

            while True:
                await asyncio.sleep(poll_sec)
                if getattr(proc, "returncode", None) is not None:
                    return
                now_wall = time.time()
                last_wall = last_monitor_heartbeat_wall[0] or start
                gap_sec = max(0.0, now_wall - last_wall)
                task = guard_task
                if task is None or not task.done():
                    if gap_sec >= stale_after_sec and now_wall - last_gap_emit_wall[0] >= stale_after_sec:
                        last_gap_emit_wall[0] = now_wall
                        if await _record_gap_and_review("resource_monitor_heartbeat_stale", gap_sec):
                            return
                    continue
                try:
                    exc = task.exception()
                except asyncio.CancelledError:
                    return
                exc_name = type(exc).__name__ if exc is not None else "completed_without_process_exit"
                guard_watchdog_restarts[0] += 1
                gap_reason = f"resource_guard_stopped:{exc_name}"
                if await _record_gap_and_review(gap_reason, gap_sec):
                    return
                guard_task = asyncio.create_task(_resource_guard())

        guard_task = asyncio.create_task(_resource_guard())
        guard_watchdog_task = asyncio.create_task(_resource_guard_watchdog())

        try:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        _drain_stream(proc.stdout, stdout_parts, ""),
                        _drain_stream(proc.stderr, stderr_parts, "[stderr] "),
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                await terminate_tree(proc)
                await _run_workspace_gpu_cleanup(reason="workspace_cleanup_after_timeout")
                _finish_agent_backoff_wait(
                    status="timeout",
                    wake_reason="timeout",
                    elapsed_sec=time.time() - start,
                    returncode=getattr(proc, "returncode", None),
                    reason=f"timeout_after_{timeout:.0f}s",
                )
                _resource_call(
                    self.resource_observer,
                    "job_finished",
                    resource_job_id,
                    status="timeout",
                    returncode=getattr(proc, "returncode", None),
                    elapsed_sec=time.time() - start,
                    reason=f"timeout_after_{timeout:.0f}s",
                )
                timeout_feedback = build_command_timeout_feedback(
                    timeout_sec=float(timeout),
                    resource_class=str(effective_resource_class or classified.resource_class or ""),
                    gpu_ids=gpu_ids,
                    saw_progress=bool(stream_stats.get("saw_training_progress")),
                    saw_artifact=bool(stream_stats.get("last_artifact") or stream_stats.get("last_artifact_emitted")),
                    current_phase=str(stream_stats.get("current_phase") or ""),
                )
                return ToolResult(
                    output=timeout_feedback,
                    error=f"Command timed out after {timeout:.0f}s",
                    system=cmd,
                )
            except asyncio.CancelledError:
                await terminate_tree(proc)
                await _run_workspace_gpu_cleanup(reason="workspace_cleanup_after_cancelled")
                _finish_agent_backoff_wait(
                    status="cancelled",
                    wake_reason="cancelled",
                    elapsed_sec=time.time() - start,
                    returncode=getattr(proc, "returncode", None),
                    reason="cancelled",
                )
                _resource_call(
                    self.resource_observer,
                    "job_finished",
                    resource_job_id,
                    status="cancelled",
                    returncode=getattr(proc, "returncode", None),
                    elapsed_sec=time.time() - start,
                    reason="cancelled",
                )
                raise

            await proc.wait()
        finally:
            if hb_task is not None and not hb_task.done():
                hb_task.cancel()
                try:
                    await hb_task
                except asyncio.CancelledError:
                    pass
            if guard_task is not None and not guard_task.done():
                guard_task.cancel()
                try:
                    await guard_task
                except asyncio.CancelledError:
                    pass
            if guard_watchdog_task is not None and not guard_watchdog_task.done():
                guard_watchdog_task.cancel()
                try:
                    await guard_watchdog_task
                except asyncio.CancelledError:
                    pass
            await _run_workspace_gpu_cleanup(reason="workspace_cleanup_after_command_finally")
            if on_output is not None and heartbeat_active[0]:
                on_output("\r\033[K")
                heartbeat_active[0] = False
            # Remove from live registry now that the process has exited (or was killed).
            from scienceflow.core.subprocess_utils import _LIVE_PGIDS
            _LIVE_PGIDS.discard(proc.pid)

        elapsed = time.time() - start
        _maybe_emit_progress_heartbeat(
            dict(stream_stats.get("last_progress_signals") or {}),
            elapsed_sec=elapsed,
            force=True,
            reason="final",
        )
        _maybe_emit_artifact_progress(elapsed, force=True)
        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        parts = [stdout]
        if stderr.strip():
            parts.append(f"[stderr]\n{stderr.rstrip()}")
        output = "\n".join(parts).rstrip()
        captured_output = output
        output = _dedup_repeated_blocks(output, min_repeat=self.dedup_min_repeat)
        output = _distill_tracebacks(
            output,
            self.workspace_dir,
            enabled=self.distill_tracebacks,
        )
        output = _sanitize_model_visible_output_paths(
            output,
            ws,
            self.path_guard_extra_roots or (),
            strip_symlink_targets=bool(self.strip_symlink_targets),
        )
        rc = proc.returncode if proc.returncode is not None else -1
        _finish_agent_backoff_wait(
            status="success" if rc == 0 else "failed",
            wake_reason="timer_elapsed" if rc == 0 else "non_zero_exit",
            elapsed_sec=elapsed,
            returncode=rc,
            reason="" if rc == 0 else f"non_zero_exit_{rc}",
        )
        if _has_masked_python_traceback(cmd, output, rc):
            output = (
                output.rstrip()
                + "\n[masked-zero-exit] Python traceback detected despite shell exit 0; "
                "treating this as failure because a pipeline likely hid the Python exit code."
            )
            rc = 1
        raw_output_for_artifact: str | None = None
        output, raw_output_for_artifact = _maybe_lossless_observation_summary(
            command=cmd,
            output=output,
            enabled=bool(self.observation_summary_enabled),
            max_chars=int(self.max_output_chars),
            returncode=rc,
        )
        if raw_output_for_artifact is None:
            output = _trim_output(output, self.max_output_chars)
        checkpoint_feedback = ""
        if str(effective_resource_class or classified.resource_class or "") in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_FEATURE_EXTRACT, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}:
            gap = None
            try:
                from scienceflow.solver.lnr.resource_runtime.sidecar import checkpoint_submission_gap

                gap = checkpoint_submission_gap(
                    workspace_dir=ws,
                    started_at=start_wall,
                    candidate_artifact=str((self.extra_env or {}).get("SCIENCEFLOW_CANDIDATE_ARTIFACT") or ""),
                )
            except Exception:
                logger.debug("[bash-resource] checkpoint_submission_gap failed", exc_info=True)
            if gap:
                checkpoint_feedback = str(_resource_call(
                    self.resource_observer,
                    "checkpoint_to_submission_guard",
                    resource_job_id,
                    gap=gap,
                    elapsed_sec=elapsed,
                ) or "").strip()
                if checkpoint_feedback:
                    output = (output.rstrip() + "\n" + checkpoint_feedback).strip()
        if guard_reason[0]:
            _resource_call(
                self.resource_observer,
                "job_finished",
                resource_job_id,
                status="resource_guard_terminated",
                returncode=rc,
                elapsed_sec=elapsed,
                reason=guard_reason[0],
            )
            header = f"resource_guard_terminated, exit={rc}, {elapsed:.1f}s"
            feedback = (guard_feedback[0] or "").strip()
            pieces = [f"[{header}]"]
            if feedback:
                pieces.append(feedback)
            if output.strip():
                pieces.append(output)
            block = "\n".join(pieces).rstrip()
            reason_text = str(guard_reason[0] or "")
            if "boundary_violation" in reason_text:
                error = "Resource guard stopped boundary violation"
            elif reason_text.startswith("active_intervention:"):
                error = "Resource guard terminated slow progress"
            else:
                error = "Resource guard terminated stalled heavy"
            return ToolResult(output=block, error=error)
        oom_detected = bool(rc != 0 and _OOM_RE.search(captured_output or output or ""))
        finish_status = "success" if rc == 0 else "oom" if oom_detected else "failed"
        finish_reason = None if rc == 0 else "oom_detected" if oom_detected else f"non_zero_exit_{rc}"
        _resource_call(
            self.resource_observer,
            "job_finished",
            resource_job_id,
            status=finish_status,
            returncode=rc,
            elapsed_sec=elapsed,
            reason=finish_reason,
        )

        header = f"exit={rc}, {elapsed:.1f}s"

        # Detect fast exits (< 0.5 s) with no output and non-zero rc.
        # Two distinct cases require different LLM guidance:
        #
        # 1. Benign "quiet rc" — the caller deliberately silenced stderr
        #    (`2>/dev/null`) OR the command returned rc=2 (the standard GNU
        #    exit code for "no such file / unmatched glob").  Both are normal
        #    outcomes; reporting them as infra failures confuses the LLM and
        #    incorrectly blocks budget expansion.
        #
        # 2. True infra failure — taskset/affinity killed the process before it
        #    could produce any output, with rc that is not a typical "not found"
        #    code and no explicit silence.  Keep the [infra-error] marker so
        #    downstream guards (tool_exec_*.py) can suppress budget expansion
        #    while keeping LNR alive until the wall-clock budget expires.
        if rc != 0 and elapsed < 0.5 and not output.strip():
            _is_quiet = _has_silent_redirect(cmd) or rc == 2
            if _is_quiet:
                no_output_hint = (
                    f"[no-output] command exited rc={rc} with no captured output. "
                    "This typically means the targets do not exist "
                    "(e.g. missing file or unmatched glob), or stderr was explicitly "
                    "silenced (`2>/dev/null` / `&>/dev/null`). "
                    "This is **not** an infrastructure failure; "
                    "do not retry purely to \"verify the env\"."
                )
                logger.info(
                    "[bash-tool] quiet-rc detected: rc=%d elapsed=%.2fs cmd=%r",
                    rc, elapsed, cmd,
                )
                block = f"[{header}]\n{no_output_hint}"
                return ToolResult(output=block, error=f"non-zero exit code {rc}")
            else:
                infra_hint = (
                    "[infra-error] Process exited immediately (< 0.5 s) with no output. "
                    "This is likely an infrastructure failure (e.g. taskset/affinity "
                    "misconfiguration), NOT a bug in your code. "
                    "Do NOT keep retrying the same command. "
                    "If this persists across multiple attempts, report it as an "
                    "environment issue and stop wasting budget on bash retries."
                )
                logger.warning(
                    "[bash-tool] fast-fail detected: rc=%d elapsed=%.2fs cmd=%r",
                    rc, elapsed, cmd,
                )
                block = f"[{header}]\n{infra_hint}"
                return ToolResult(output=block, error=f"non-zero exit code {rc}")

        block = f"[{header}]\n{output}"
        if rc != 0:
            return ToolResult(output=block, error=f"non-zero exit code {rc}")
        if raw_output_for_artifact is not None:
            return ToolResult(output=block, system=f"[{header}]\n{raw_output_for_artifact}")
        return ToolResult(output=block)
