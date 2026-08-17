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

"""Parallel multi-task runner with per-task CPU/GPU isolation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from scienceflow.config.settings import (
    expand_lnr_resource_control_mode_payload,
    merge_parallel_manifest_cfg_patch,
)
from scienceflow.core.task_package import description_path_for_task
from scienceflow.core.runtime_env import resolve_project_python
from scienceflow.core.subprocess_utils import (
    _LIVE_PGIDS,
    kill_all_live_pgids,
    spawn_exec,
    terminate_tree,
)
from scienceflow.solver.lnr.resource_runtime.workspace_gpu_guard import cleanup_workspace_gpu_processes
from scienceflow.utils.llm_config_summary import (
    format_llm_config_summary,
    merge_llm_config_summary,
    summarize_llm_config_from_agent_patch,
    summarize_llm_config_from_env,
)
from scienceflow.utils.resource_utils import parse_cpu_list, parse_gpu_list_auto, select_least_used_gpu

logger = logging.getLogger("scienceflow")


def _format_cpu_set_compact(cpu_ids: list[int]) -> str:
    """Format CPU ids as a compact taskset-compatible range string."""
    vals = sorted({int(c) for c in cpu_ids if int(c) >= 0})
    if not vals:
        return ""
    ranges: list[str] = []
    start = prev = vals[0]
    for cur in vals[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cur
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _positive_int_or_none(value: Any) -> int | None:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


_VALID_PHASES = frozenset({"run", "prep"})
_BUDGET_DONE_STATUS = "budget_done"


def _scienceflow_repo_root() -> Path:
    """Project root (parent of the installed ``scienceflow`` package directory)."""
    # scienceflow/core/parallel_runner.py -> parents[2] == repo root (contains ``tasks/``)
    return Path(__file__).resolve().parents[2]


def _resolve_parallel_log_dir_override(log_dir: str | Path | None) -> Path | None:
    """Resolve optional legacy subprocess log dir and reject repo-root parallel_logs."""
    if log_dir is None or not str(log_dir).strip():
        return None
    raw = Path(log_dir).expanduser()
    if raw.parts and raw.parts[0] == "parallel_logs":
        raise ValueError(
            "--log-dir under repo-root parallel_logs is no longer supported; "
            "parallel subprocess logs are stored in each task workspace task_logs/.",
        )
    policy_path = raw if raw.is_absolute() else _scienceflow_repo_root() / raw
    resolved = policy_path.resolve(strict=False)
    forbidden = (_scienceflow_repo_root() / "parallel_logs").resolve(strict=False)
    try:
        inside_forbidden = resolved == forbidden or resolved.is_relative_to(forbidden)
    except ValueError:
        inside_forbidden = False
    if inside_forbidden:
        raise ValueError(
            "--log-dir under repo-root parallel_logs is no longer supported; "
            "parallel subprocess logs are stored in each task workspace task_logs/.",
        )
    return raw


def _resolve_parallel_task_text(exp_id: str, raw_task: Any) -> str:
    """Inline manifest ``task`` wins; otherwise load categorized task text."""
    if raw_task is not None:
        text = str(raw_task).strip()
        if text:
            return text
    default_path = description_path_for_task(exp_id, tasks_root=_scienceflow_repo_root() / "tasks")
    if default_path is None:
        raise ValueError(
            f"Task {exp_id!r}: empty or missing inline `task` and no default description file "
            f"registered by tasks/**/{exp_id}/task.yaml "
            f"(add `task: |` in the manifest or create that file).",
        )
    return default_path.read_text(encoding="utf-8")


def _safe_filename(s: str, max_len: int = 200) -> str:
    """Sanitize a string for use as a single path component (run_id / exp_id segments)."""
    t = re.sub(r"[^a-zA-Z0-9._-]", "_", s)[:max_len]
    return t if t else "_"


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _manifest_string_list(raw: Any) -> list[str]:
    """Normalize a manifest scalar/list CSV-ish value into a list of non-empty strings."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    text = str(raw).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"[,\s]+", text) if s.strip()]


def _manifest_string_csv(raw: Any) -> str:
    return ",".join(_manifest_string_list(raw))


def _env_key_pool_size(env: dict[str, str]) -> int:
    """Return the inherited key pool size used by code LLM routing."""
    for name in ("CODE_API_KEYS", "API_KEYS", "FEEDBACK_API_KEYS"):
        size = len(_manifest_string_list(env.get(name, "")))
        if size > 0:
            return size
    return 0


def _env_sticky_primary_index(env: dict[str, str], task_index: int) -> int | None:
    pool_size = _env_key_pool_size(env)
    if pool_size <= 0:
        return None
    return max(0, int(task_index)) % pool_size


def _manifest_endpoint_pairs(
    keys: list[str],
    *,
    base_url: str = "",
    base_urls: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Build ``(base_url, api_key)`` pairs with the same broadcast semantics as parse_key_env."""
    if not keys:
        return []
    urls = list(base_urls or [])
    fallback_url = str(base_url or "").strip()
    if not urls and fallback_url:
        urls = [fallback_url]
    if not urls:
        urls = [""] * len(keys)
    elif len(urls) == 1 and len(keys) > 1:
        urls = urls * len(keys)
    elif len(urls) != len(keys):
        raise ValueError(
            f"Mismatch: {len(keys)} api_keys but {len(urls)} base_urls. "
            "Provide one base_url/base_urls entry to broadcast or the same count as keys.",
        )
    return list(zip(urls, keys))


def _rotate_endpoint_pairs_for_primary(
    pairs: list[tuple[str, str]], primary_index: int,
) -> list[tuple[str, str]]:
    if not pairs:
        return []
    idx = primary_index % len(pairs)
    return pairs[idx:] + pairs[:idx]


def _manifest_task_run_id(task: dict[str, Any], idx: int, exp_id: str) -> str:
    """Resolve per-task run id. Optional key ``run_id``; not inherited from defaults.

    When omitted or blank, defaults to ``exp_id`` (backward compatible).
    """
    if "run_id" not in task:
        return exp_id
    raw = task["run_id"]
    s = str(raw).strip() if raw is not None else ""
    return s if s else exp_id


def _resolve_task_workspace(
    defaults: dict[str, Any],
    task: dict[str, Any],
    idx: int,
    run_id: str,
    exp_id: str,
) -> str:
    """Resolve workspace path: explicit ``workspace`` wins, else ``workspace_base/run_id/exp_id``."""
    explicit_ws = str(task.get("workspace", "") or "").strip()
    if explicit_ws:
        return explicit_ws
    task_base = str(task.get("workspace_base", "") or "").strip()
    default_base = str(defaults.get("workspace_base", "") or "").strip()
    workspace_base = task_base or default_base
    if workspace_base:
        base = Path(workspace_base).expanduser()
        return str(base / _safe_filename(run_id) / _safe_filename(exp_id))
    raise ValueError(
        f"tasks[{idx}]: need non-empty 'workspace' or 'workspace_base' "
        f"(defaults or task) for run_id={run_id!r}, exp_id={exp_id!r}",
    )


def resolve_manifest_task_workspace(
    defaults: dict[str, Any],
    task: dict[str, Any],
    idx: int,
    exp_id: str,
) -> str:
    """Public helper: same workspace resolution as :class:`ParallelRunner` (for monitor, tests)."""
    run_id = _manifest_task_run_id(task, idx, exp_id)
    return _resolve_task_workspace(defaults, task, idx, run_id, exp_id)


def _manifest_task_exp_id(task: dict[str, Any], idx: int) -> str:
    """Resolve per-task experiment / competition id (manifest key ``exp_id``).

    Legacy key ``name`` is still accepted but deprecated.
    """
    if "exp_id" in task:
        raw = task["exp_id"]
    elif "name" in task:
        warnings.warn(
            "Parallel manifest key 'name' is deprecated; use 'exp_id' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        raw = task["name"]
    else:
        raise ValueError(
            f"tasks[{idx}]: missing required key 'exp_id' "
            "(replace legacy 'name' with 'exp_id').",
        )
    s = str(raw).strip() if raw is not None else ""
    if not s:
        raise ValueError(f"tasks[{idx}]: 'exp_id' must be non-empty")
    return s


def _manifest_input_data_dir(defaults: dict[str, Any], task: dict[str, Any]) -> str:
    """Resolve read-only input root: same as Config.input_data_dir / ``prep --input-data-dir`` / ``-d``.

    Prefer ``input_data_dir``; accept legacy manifest key ``data_dir`` (task overrides
    defaults, first non-empty wins).
    """
    for src in (
        task.get("input_data_dir"),
        task.get("data_dir"),
        defaults.get("input_data_dir"),
        defaults.get("data_dir"),
    ):
        if src is not None and str(src).strip():
            return str(src).strip()
    return ""


def _manifest_scienceflow_interaction_log_full(
    defaults: dict[str, Any], task: dict[str, Any]
) -> bool | None:
    """Resolve ``scienceflow_interaction_log_full`` (task overrides defaults). None = do not touch env."""
    if "scienceflow_interaction_log_full" in task:
        return bool(task["scienceflow_interaction_log_full"])
    if "scienceflow_interaction_log_full" in defaults:
        return bool(defaults["scienceflow_interaction_log_full"])
    return None


def _manifest_scienceflow_interaction_log_color(
    defaults: dict[str, Any], task: dict[str, Any]
) -> bool | None:
    """Resolve ``scienceflow_interaction_log_color`` (task overrides defaults). None = do not touch env."""
    if "scienceflow_interaction_log_color" in task:
        return bool(task["scienceflow_interaction_log_color"])
    if "scienceflow_interaction_log_color" in defaults:
        return bool(defaults["scienceflow_interaction_log_color"])
    return None


def _manifest_scienceflow_interaction_log_llm_stream(
    defaults: dict[str, Any], task: dict[str, Any]
) -> bool | None:
    """Resolve ``scienceflow_interaction_log_llm_stream`` (task overrides defaults). None = do not touch env."""
    if "scienceflow_interaction_log_llm_stream" in task:
        return bool(task["scienceflow_interaction_log_llm_stream"])
    if "scienceflow_interaction_log_llm_stream" in defaults:
        return bool(defaults["scienceflow_interaction_log_llm_stream"])
    return None


def _manifest_scienceflow_interaction_log_level(
    defaults: dict[str, Any], task: dict[str, Any]
) -> str | None:
    """Resolve ``scienceflow_interaction_log_level`` (task overrides defaults). None = do not touch env."""
    if "scienceflow_interaction_log_level" in task:
        raw = task["scienceflow_interaction_log_level"]
        return str(raw).strip().lower() if raw is not None else None
    if "scienceflow_interaction_log_level" in defaults:
        raw = defaults["scienceflow_interaction_log_level"]
        return str(raw).strip().lower() if raw is not None else None
    return None


def _manifest_run_type(
    defaults: dict[str, Any],
    task: dict[str, Any],
    idx: int,
    exp_id: str,
) -> str:
    """Resolve phase=run child CLI type. REPL-only builds accept one solver type."""
    raw_type = task.get("type", defaults.get("type", "lnr"))
    run_type = str(raw_type).strip().lower() if raw_type is not None else "lnr"
    if run_type != "lnr":
        raise ValueError(
            f"tasks[{idx}] ({exp_id!r}): invalid type {run_type!r} (use 'lnr')",
        )
    return run_type


def _manifest_merge_lnr(
    manifest: dict[str, Any],
    defaults: dict[str, Any],
    task: dict[str, Any],
    idx: int,
    exp_id: str,
) -> dict[str, Any]:
    """Merge lnr blocks: manifest root < defaults < task."""

    def _as_block(raw: Any, where: str) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        raise TypeError(
            f"tasks[{idx}] ({exp_id!r}): lnr at {where} must be a mapping, "
            f"got {type(raw).__name__}",
        )

    root = expand_lnr_resource_control_mode_payload(_as_block(manifest.get("lnr"), "manifest root"))
    dflt = expand_lnr_resource_control_mode_payload(_as_block(defaults.get("lnr"), "defaults"))
    tblk = expand_lnr_resource_control_mode_payload(_as_block(task.get("lnr"), "task"))
    return {**root, **dflt, **tblk}

def _manifest_merge_agent(
    manifest: dict[str, Any],
    defaults: dict[str, Any],
    task: dict[str, Any],
    idx: int,
    exp_id: str,
) -> dict[str, Any]:
    """Merge ``agent`` blocks (top-level AgentConfig scalars): manifest root < defaults < task."""

    def _as_block(raw: Any, where: str) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        raise TypeError(
            f"tasks[{idx}] ({exp_id!r}): agent at {where} must be a mapping, "
            f"got {type(raw).__name__}",
        )

    root = _as_block(manifest.get("agent"), "manifest root")
    dflt = _as_block(defaults.get("agent"), "defaults")
    tblk = _as_block(task.get("agent"), "task")
    return {**root, **dflt, **tblk}


def _validate_parallel_manifest(tasks: list[TaskSpec]) -> None:
    """Fail fast before spawning subprocesses (prep needs input_data_dir)."""
    errors: list[str] = []
    seen_run: dict[str, int] = {}
    for i, spec in enumerate(tasks):
        rid = spec.run_id or spec.exp_id
        if rid in seen_run:
            errors.append(
                f"Duplicate run_id {rid!r} at tasks[{seen_run[rid]}] and tasks[{i}]. "
                "Add explicit distinct `run_id` values per task.",
            )
        else:
            seen_run[rid] = i
    for spec in tasks:
        if spec.phase not in _VALID_PHASES:
            errors.append(
                f"{spec.exp_id}: invalid phase {spec.phase!r} (use 'run' or 'prep')",
            )
            continue
        if spec.phase == "prep":
            if not spec.input_data_dir.strip():
                errors.append(
                    f"{spec.exp_id}: phase=prep requires non-empty input_data_dir "
                    f"(manifest key input_data_dir, or legacy data_dir)",
                )
                continue
            p = Path(spec.input_data_dir).expanduser()
            if not p.is_dir():
                errors.append(
                    f"{spec.exp_id}: input_data_dir is not an existing directory: {p}",
                )
        elif spec.input_data_dir.strip():
            p = Path(spec.input_data_dir).expanduser()
            if not p.is_dir():
                errors.append(
                    f"{spec.exp_id}: input_data_dir is not an existing directory: {p}",
                )
    if errors:
        raise ValueError(
            "Parallel manifest validation failed:\n- " + "\n- ".join(errors),
        )


def _task_state_log_dir(spec: "TaskSpec") -> Path:
    """Return the root-level state log directory for a parallel task."""
    root = Path(spec.workspace).expanduser()
    return root / "task_logs" if str(spec.type or "").strip().lower() == "lnr" else root / "logs"


def _parallel_state_file_candidates(task_root: str, *, task_type: str = "") -> list[Path]:
    """Return state.json candidates for the current layout policy."""
    root = Path(task_root).expanduser()
    if str(task_type or "").strip().lower() == "lnr":
        return [root / "task_logs" / "state.json"]
    return [
        root / "logs" / "state.json",
        root / "workspace" / "state.json",
        root / "state.json",
    ]


def _parallel_state_file_for_read(task_root: str, *, task_type: str = "") -> Path | None:
    for p in _parallel_state_file_candidates(task_root, task_type=task_type):
        if p.is_file():
            return p
    return None


def _ensure_workspace_for_state_write(workspace: str) -> Path | None:
    """Return directory where state.json may be written, or None if unsafe/unwritable.

    ``mkdir(parents=True)`` on paths like ``/path/to/ws`` tries to create ``/path`` first,
    which typically raises PermissionError. Only create missing segments when some
    ancestor already exists below the filesystem root.
    """
    ws = Path(workspace).expanduser()
    if ws.exists():
        return ws if ws.is_dir() else None
    target = ws.resolve(strict=False)
    p = target
    while not p.exists():
        if p.parent == p:
            return None
        if p.parent == Path("/"):
            logger.warning(
                "workspace %r is missing prefix %s under /; refusing to mkdir "
                "(use an existing directory you own)",
                workspace,
                p,
            )
            return None
        p = p.parent
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("cannot create workspace %r: %s", workspace, e)
        return None
    return target


def _write_gpu_assignment_json(spec: TaskSpec, resolved_gpu: str) -> None:
    """Persist resolved CUDA device IDs under the task state log directory."""
    base = _ensure_workspace_for_state_write(spec.workspace)
    if base is None:
        return
    logs_dir = _task_state_log_dir(spec)
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("[%s] cannot create logs for gpu_assignment.json: %s", spec.exp_id, e)
        return
    raw = (spec.gpu_list_raw or spec.gpu_list or "").strip()
    payload: dict[str, Any] = {
        "resolved_gpu": resolved_gpu if resolved_gpu else None,
        "auto": bool(spec.gpu_auto),
        "raw_gpu_list": raw if raw else None,
    }
    try:
        (logs_dir / "gpu_assignment.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("[%s] Failed to write gpu_assignment.json: %s", spec.exp_id, e)


def _read_resolved_gpu_from_assignment(logs_dir: Path) -> str:
    """Best-effort read of ``resolved_gpu`` from ``gpu_assignment.json``."""
    p = logs_dir / "gpu_assignment.json"
    if not p.is_file():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    r = data.get("resolved_gpu")
    return str(r).strip() if r is not None else ""


_EXTERNAL_LLM_FAILURE_KINDS = {"llm_quota_error", "llm_api_error"}


def _parallel_failure_kind_from_text(text: str) -> str:
    lower = str(text or "").lower()
    if not lower.strip():
        return ""
    if (
        "llm_quota_error" in lower
        or "insufficient balance" in lower
        or "pre_consume_token_quota_failed" in lower
        or "token quota is not enough" in lower
        or "permissiondeniederror" in lower
        or "error code: 402" in lower
        or "error code: 403" in lower
    ):
        return "llm_quota_error"
    if (
        "context_compact_failed" in lower
        or "compact did not fit context" in lower
        or "omitted-history main-agent request" in lower
    ):
        return "context_compact_failed"
    if (
        "llm_api_error" in lower
        or "apistatuserror" in lower
        or "authenticationerror" in lower
        or "badrequesterror" in lower
        or "reasoning_content" in lower
        or "invalid_request_error" in lower
        or "error code: 400" in lower
        or "error code: 401" in lower
    ):
        return "llm_api_error"
    if "time budget exhausted" in lower:
        return "time_budget_expired"
    if "timeout" in lower:
        return "worker_timeout"
    return ""


def _read_log_tail(path: str | Path, *, max_bytes: int = 200_000) -> str:
    try:
        p = Path(path)
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _parallel_failure_kind(result: "TaskResult") -> str:
    if result.status == "success":
        return ""
    direct = _parallel_failure_kind_from_text(result.error)
    if direct:
        return direct
    return _parallel_failure_kind_from_text(_read_log_tail(result.log_file))


def _charged_elapsed_for_resume(result: "TaskResult", failure_kind: str) -> float:
    if result.status in {"failed", "error"} and failure_kind in _EXTERNAL_LLM_FAILURE_KINDS:
        return 0.0
    return float(result.elapsed_sec or 0.0)


def _workspace_lnr_event_files(workspace: str) -> list[Path]:
    task_logs = Path(workspace).expanduser() / "task_logs"
    return [
        task_logs / "lhr_events.jsonl",
        task_logs / "lhr_coordinator_events.jsonl",
    ]


def _workspace_lnr_done_payload(workspace: str) -> dict[str, Any]:
    for events_path in _workspace_lnr_event_files(workspace):
        try:
            lines = events_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines[-500:]):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "multi_worker_done":
                continue
            payload = event.get("payload")
            if isinstance(payload, dict):
                return payload
            return event if isinstance(event, dict) else {}
    return {}


def _workspace_lnr_result_status(workspace: str) -> str:
    payload = _workspace_lnr_done_payload(workspace)
    return str(payload.get("status") or "").strip().lower()


def _workspace_lnr_failure_kind(workspace: str) -> str:
    payload = _workspace_lnr_done_payload(workspace)
    if not payload:
        return ""
    for key in ("failure_kind", "worker_error_kinds", "stop_reason"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                kind = _parallel_failure_kind_from_text(str(item))
                if kind:
                    return kind
        else:
            kind = _parallel_failure_kind_from_text(str(value or ""))
            if kind:
                return kind
    for result in payload.get("worker_results") or []:
        if isinstance(result, dict):
            kind = _parallel_failure_kind_from_text(str(result.get("error") or ""))
            if kind:
                return kind
    return ""


def _state_completed_but_child_failed(workspace: str) -> bool:
    status = _workspace_lnr_result_status(workspace)
    return bool(status and status not in {"success", "completed"})


@dataclass
class TaskSpec:
    exp_id: str
    task: str
    workspace: str
    manifest_index: int = 0
    run_id: str = ""
    cpu_list: str = ""
    gpu_list: str = ""
    config: str = ""
    type: str = "lnr"
    time_limit: int = 3600
    api_key: str = ""
    api_keys: str = ""
    base_url: str = ""
    base_urls: str = ""
    llm_routing_mode: str = ""
    llm_sticky_id: str = ""
    llm_sticky_primary_index: int | None = None
    # phase "run" -> `cli run`; phase "prep" -> `cli prep`.
    phase: str = "run"
    input_data_dir: str = ""
    num_drafts: int = 1
    draft_parallel: int = 1
    # Merged from manifest lnr; passed via SCIENCEFLOW_PARALLEL_LNR_JSON.
    lnr_patch: dict[str, Any] | None = None
    # Merged from manifest ``agent`` (root / defaults / task); top-level AgentConfig scalars via SCIENCEFLOW_PARALLEL_AGENT_JSON.
    agent_patch: dict[str, Any] | None = None
    # Top-level :class:`Config` keys from manifest ``defaults`` + ``task`` (task wins); passed via env to prep/run.
    manifest_cfg_patch: dict[str, Any] | None = None
    # When set, child env ``SCIENCEFLOW_INTERACTION_LOG_FULL`` is forced on/off (see ``_exec_subprocess``).
    scienceflow_interaction_log_full: bool | None = None
    # When set, child env ``SCIENCEFLOW_INTERACTION_LOG_COLOR`` is forced on/off.
    scienceflow_interaction_log_color: bool | None = None
    # When set, child env ``SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM`` is forced 1/0.
    scienceflow_interaction_log_llm_stream: bool | None = None
    # When set, child env ``SCIENCEFLOW_INTERACTION_LOG_LEVEL`` is set (minimal|normal|verbose).
    scienceflow_interaction_log_level: str | None = None
    # Auto GPU selection: when True, gpu_list is resolved at subprocess launch
    # by picking the GPU(s) with the most free memory from gpu_auto_candidates.
    gpu_auto: bool = False
    gpu_auto_count: int = 1
    gpu_auto_candidates: list[int] = field(default_factory=list)
    # Original manifest ``gpu_list`` string (e.g. ``auto:2``) for gpu_assignment.json.
    gpu_list_raw: str = ""


@dataclass
class TaskResult:
    exp_id: str
    run_id: str = ""
    status: str = "pending"
    exit_code: int | None = None
    elapsed_sec: float = 0.0
    error: str = ""
    log_file: str = ""


class ParallelRunner:
    """Launch multiple ScienceFlow tasks as isolated subprocesses."""

    def __init__(
        self,
        manifest_path: str | Path,
        max_concurrent: int | None = None,
        log_dir: str | Path | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.log_dir = _resolve_parallel_log_dir_override(log_dir)
        self._manifest: dict[str, Any] = {}
        self._tasks: list[TaskSpec] = []
        self._max_concurrent = max_concurrent
        self._resume_budget_policy = "remaining"
        self._gpu_auto_lock = asyncio.Lock()
        self._gpu_auto_assigned: set[int] = set()
        self._load_manifest()

    def _load_manifest(self) -> None:
        with open(self.manifest_path) as f:
            self._manifest = yaml.safe_load(f) or {}

        defaults = self._manifest.get("defaults") or {}

        global_time_limit = self._manifest.get("time_limit", 3600)
        self._resume_budget_policy = str(
            self._manifest.get("resume_budget_policy", "remaining") or "remaining"
        ).strip().lower()
        if self._resume_budget_policy not in {"remaining", "fresh"}:
            raise ValueError("resume_budget_policy must be one of: remaining, fresh")
        if self._max_concurrent is None:
            self._max_concurrent = self._manifest.get("max_concurrent", 4)

        global_api_keys = _manifest_string_list(self._manifest.get("api_keys", []))
        global_base_url = self._manifest.get("base_url", defaults.get("base_url", ""))
        global_base_urls = _manifest_string_list(self._manifest.get("base_urls", []))
        global_endpoint_pairs = _manifest_endpoint_pairs(
            global_api_keys,
            base_url=str(global_base_url or ""),
            base_urls=global_base_urls,
        )

        default_phase = str(defaults.get("phase", "run")).strip().lower() or "run"

        raw_tasks = self._manifest.get("tasks", [])
        if raw_tasks is None:
            raw_tasks = []
        for idx, t in enumerate(raw_tasks):
            if not isinstance(t, dict):
                raise TypeError(f"tasks[{idx}] must be a mapping, got {type(t).__name__}")

            phase = str(t.get("phase", default_phase)).strip().lower() or "run"
            input_data_dir = _manifest_input_data_dir(defaults, t)
            exp_id = _manifest_task_exp_id(t, idx)
            run_id = _manifest_task_run_id(t, idx, exp_id)
            workspace = _resolve_task_workspace(defaults, t, idx, run_id, exp_id)
            task_api_key = str(t.get("api_key", "") or "").strip()
            task_api_keys = _manifest_string_csv(t.get("api_keys", ""))
            task_base_url = str(t.get("base_url", global_base_url) or "").strip()
            task_base_urls = _manifest_string_csv(
                t.get("base_urls", ",".join(global_base_urls) if global_base_urls else ""),
            )
            llm_routing_mode = str(
                t.get("llm_routing_mode", self._manifest.get("llm_routing_mode", ""))
                or "",
            ).strip()
            llm_sticky_id = str(
                t.get("llm_sticky_id", self._manifest.get("llm_sticky_id", ""))
                or "",
            ).strip()
            llm_sticky_primary_index: int | None = None
            if not task_api_key and not task_api_keys and global_endpoint_pairs:
                primary_index = idx % len(global_endpoint_pairs)
                ordered_pairs = _rotate_endpoint_pairs_for_primary(
                    global_endpoint_pairs,
                    primary_index,
                )
                task_api_key = ordered_pairs[0][1]
                task_base_url = ordered_pairs[0][0]
                if len(ordered_pairs) > 1:
                    task_api_keys = ",".join(key for _, key in ordered_pairs)
                    urls = [url for url, _ in ordered_pairs]
                    if len(set(urls)) == 1:
                        task_base_url = urls[0]
                        task_base_urls = ""
                    else:
                        task_base_urls = ",".join(urls)
                    llm_routing_mode = llm_routing_mode or "sticky_failover"
                    llm_sticky_id = llm_sticky_id or run_id or exp_id
                    llm_sticky_primary_index = 0

            if phase == "run":
                run_type = _manifest_run_type(defaults, t, idx, exp_id)
            else:
                run_type = "lnr"

            lnr_patch = _manifest_merge_lnr(self._manifest, defaults, t, idx, exp_id)
            lnr_patch_arg: dict[str, Any] | None = lnr_patch if lnr_patch else None
            agent_patch = _manifest_merge_agent(self._manifest, defaults, t, idx, exp_id)
            agent_patch_arg: dict[str, Any] | None = agent_patch if agent_patch else None
            mcfg = merge_parallel_manifest_cfg_patch(defaults, t)
            manifest_cfg_patch_arg: dict[str, Any] | None = mcfg if mcfg else None
            interaction_log_full = _manifest_scienceflow_interaction_log_full(defaults, t)
            interaction_log_color = _manifest_scienceflow_interaction_log_color(defaults, t)
            interaction_log_llm_stream = _manifest_scienceflow_interaction_log_llm_stream(defaults, t)
            interaction_log_level = _manifest_scienceflow_interaction_log_level(defaults, t)

            raw_gpu = str(t.get("gpu_list", "") or "")
            gpu_disabled = raw_gpu.strip().lower() in {"none", "cpu", "off", "disabled", "-1"}
            is_auto, auto_count, auto_candidates = (
                (False, 0, None) if gpu_disabled else parse_gpu_list_auto(raw_gpu)
            )
            auto_count_eff = max(1, auto_count)
            if is_auto and run_type == "lnr":
                # Bare "auto" (no explicit count) scales to LNR worker count for isolation.
                bare_auto = raw_gpu.strip().lower() == "auto"
                if bare_auto:
                    _workers = 2
                    _shared = False
                    if lnr_patch_arg:
                        try:
                            _workers = int(lnr_patch_arg.get("num_workers", _workers) or _workers)
                        except (TypeError, ValueError):
                            _workers = 2
                    if (not _shared) and _workers > 1:
                        auto_count_eff = max(auto_count_eff, _workers)
            self._tasks.append(
                TaskSpec(
                    exp_id=exp_id,
                    task=_resolve_parallel_task_text(exp_id, t.get("task")),
                    workspace=workspace,
                    manifest_index=idx,
                    run_id=run_id,
                    cpu_list=t.get("cpu_list", ""),
                    gpu_list="" if (is_auto or gpu_disabled) else raw_gpu,
                    config=t.get("config", defaults.get("config", "")),
                    type=run_type,
                    time_limit=t.get("time_limit", global_time_limit),
                    api_key=task_api_key,
                    api_keys=task_api_keys,
                    base_url=task_base_url,
                    base_urls=task_base_urls,
                    llm_routing_mode=llm_routing_mode,
                    llm_sticky_id=llm_sticky_id,
                    llm_sticky_primary_index=llm_sticky_primary_index,
                    phase=phase,
                    input_data_dir=input_data_dir,
                    num_drafts=int(t.get("num_drafts", defaults.get("num_drafts", 1)) or 1),
                    draft_parallel=int(
                        t.get("draft_parallel", defaults.get("draft_parallel", 1)) or 1,
                    ),
                    lnr_patch=lnr_patch_arg,
                    agent_patch=agent_patch_arg,
                    manifest_cfg_patch=manifest_cfg_patch_arg,
                    scienceflow_interaction_log_full=interaction_log_full,
                    scienceflow_interaction_log_color=interaction_log_color,
                    scienceflow_interaction_log_llm_stream=interaction_log_llm_stream,
                    scienceflow_interaction_log_level=interaction_log_level,
                    gpu_auto=is_auto,
                    gpu_auto_count=auto_count_eff,
                    gpu_auto_candidates=auto_candidates,
                    gpu_list_raw=raw_gpu,
                )
            )

        _validate_parallel_manifest(self._tasks)

    @property
    def resume_enabled(self) -> bool:
        return self._manifest.get("resume", False)

    async def run_all(self) -> list[TaskResult]:
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(self._max_concurrent)
        coros = [self._run_task(spec, sem) for spec in self._tasks]
        try:
            results = await asyncio.gather(*coros, return_exceptions=True)
        except (asyncio.CancelledError, KeyboardInterrupt):
            self._write_interrupted_states(self._tasks, error="interrupted by user")
            # Sweep any process groups that are still alive before propagating.
            kill_all_live_pgids(sig=signal.SIGTERM)
            await asyncio.sleep(3)
            kill_all_live_pgids(sig=signal.SIGKILL)
            raise

        final: list[TaskResult] = []
        for r in results:
            if isinstance(r, Exception):
                final.append(TaskResult(exp_id="unknown", run_id="unknown", status="error", error=str(r)))
            else:
                final.append(r)
        return final

    def _task_subprocess_log_file(self, spec: TaskSpec) -> Path:
        if self.log_dir is not None:
            return self.log_dir / f"{_safe_filename(spec.run_id)}.log"
        return _task_state_log_dir(spec) / "parallel_subprocess.log"

    async def _run_task(self, spec: TaskSpec, sem: asyncio.Semaphore) -> TaskResult:
        log_file = self._task_subprocess_log_file(spec)
        result = TaskResult(
            exp_id=spec.exp_id,
            run_id=spec.run_id,
            log_file=str(log_file),
        )

        if self.resume_enabled:
            remaining = self._check_resume(spec)
            if remaining is not None and remaining <= 0:
                spec = self._with_resume_remaining_budget(spec, 0)
                result.status = "skipped"
                self._write_state(spec, result, resolved_gpu="")
                return result
            if remaining is not None:
                spec = self._with_resume_remaining_budget(spec, remaining)

        resolved_gpu = ""
        interrupted_exc: BaseException | None = None
        async with sem:
            result.status = "running"
            t0 = time.monotonic()
            try:
                exit_code, resolved_gpu = await self._exec_subprocess(spec, result.log_file)
                result.exit_code = exit_code
                result.status = "success" if exit_code == 0 else "failed"
            except asyncio.TimeoutError:
                result.status = _BUDGET_DONE_STATUS
                result.error = f"time budget exhausted after {spec.time_limit}s"
            except (asyncio.CancelledError, KeyboardInterrupt) as e:
                result.status = "stopped_by_user"
                result.error = type(e).__name__
                interrupted_exc = e
            except Exception as e:
                result.status = "error"
                result.error = str(e)
            finally:
                result.elapsed_sec = round(time.monotonic() - t0, 1)

        self._write_state(spec, result, resolved_gpu=resolved_gpu)
        if interrupted_exc is not None:
            raise interrupted_exc
        return result

    @staticmethod
    def child_cli_argv(spec: TaskSpec) -> list[str]:
        """Build ``python -m scienceflow.cli …`` argv (no ``taskset`` prefix); used by tests and `_exec_subprocess`."""
        child_python, _child_bin = resolve_project_python()
        argv: list[str] = [
            child_python,
            "-m",
            "scienceflow.cli",
        ]
        if spec.phase == "prep":
            argv.extend(
                [
                    "prep",
                    "--task",
                    spec.task,
                    "--workspace",
                    spec.workspace,
                    "--input-data-dir",
                    str(Path(spec.input_data_dir).expanduser()),
                ],
            )
            if spec.config:
                argv.extend(["--config", spec.config])
        else:
            argv.extend(
                [
                    "run",
                    "--task",
                    spec.task,
                    "--workspace",
                    spec.workspace,
                    "--type",
                    spec.type,
                ],
            )
            if spec.config:
                argv.extend(["--config", spec.config])
            if spec.input_data_dir.strip():
                argv.extend(
                    [
                        "--input-data-dir",
                        str(Path(spec.input_data_dir).expanduser()),
                    ],
                )
        return argv

    async def _exec_subprocess(self, spec: TaskSpec, log_file: str) -> tuple[int, str]:
        cmd: list[str] = []
        effective_cpu_list = spec.cpu_list or ""
        if effective_cpu_list:
            # Validate that every requested CPU exists on this host before passing
            # them to ``taskset``.  An out-of-range ID causes taskset to exit 1
            # immediately — killing all bash commands in the child process.
            host_cores = os.cpu_count() or 0
            parsed_ids = parse_cpu_list(effective_cpu_list)
            valid_ids = [c for c in parsed_ids if c < host_cores]
            oob_ids = [c for c in parsed_ids if c >= host_cores]
            if oob_ids:
                logger.warning(
                    "[parallel-runner] task=%s: dropping %d out-of-range CPU(s) "
                    "%s from cpu_list %r (host has %d cores)",
                    spec.run_id, len(oob_ids), oob_ids[:8],
                    effective_cpu_list, host_cores,
                )
            if valid_ids:
                # Re-format as compact range string understood by taskset
                effective_cpu_list = _format_cpu_set_compact(valid_ids)
                cmd.extend(["taskset", "-c", effective_cpu_list])
            else:
                logger.warning(
                    "[parallel-runner] task=%s: cpu_list %r entirely out-of-range "
                    "for this host (%d cores); skipping taskset",
                    spec.run_id, spec.cpu_list, host_cores,
                )
                effective_cpu_list = ""
        cmd.extend(self.child_cli_argv(spec))

        # exec.cpu_list/gpu_list must not be passed as trailing argv: Click rejects
        # unknown args before run(). Isolation uses taskset + CUDA_VISIBLE_DEVICES
        # above; inner-process affinity reads cfg.exec (default empty → inherit env).

        env = os.environ.copy()
        child_python, child_bin = resolve_project_python()
        if child_bin:
            env["PATH"] = child_bin + ":" + env.get("PATH", "")
        child_py = Path(child_python)
        if child_py.parent.name == "bin" and child_py.parent.parent.name == ".venv":
            env["VIRTUAL_ENV"] = str(child_py.parent.parent)
        if effective_cpu_list:
            env["SCIENCEFLOW_CPU_LIST"] = effective_cpu_list
        if spec.manifest_cfg_patch:
            env["SCIENCEFLOW_PARALLEL_MANIFEST_CFG_JSON"] = json.dumps(spec.manifest_cfg_patch)
        if (
            spec.lnr_patch
            and spec.phase == "run"
            and spec.type == "lnr"
        ):
            env["SCIENCEFLOW_PARALLEL_LNR_JSON"] = json.dumps(spec.lnr_patch)
        if spec.scienceflow_interaction_log_full is True:
            env["SCIENCEFLOW_INTERACTION_LOG_FULL"] = "1"
        elif spec.scienceflow_interaction_log_full is False:
            env.pop("SCIENCEFLOW_INTERACTION_LOG_FULL", None)

        if spec.scienceflow_interaction_log_color is True:
            env["SCIENCEFLOW_INTERACTION_LOG_COLOR"] = "1"
        elif spec.scienceflow_interaction_log_color is False:
            env.pop("SCIENCEFLOW_INTERACTION_LOG_COLOR", None)

        if spec.scienceflow_interaction_log_llm_stream is True:
            env["SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM"] = "1"
        elif spec.scienceflow_interaction_log_llm_stream is False:
            env["SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM"] = "0"

        if spec.scienceflow_interaction_log_level:
            env["SCIENCEFLOW_INTERACTION_LOG_LEVEL"] = spec.scienceflow_interaction_log_level

        resolved_gpu = spec.gpu_list
        _assigned_auto_gpus: list[int] = []
        gpu_disabled = (spec.gpu_list_raw or "").strip().lower() in {
            "none",
            "cpu",
            "off",
            "disabled",
            "-1",
        }
        if gpu_disabled:
            env["SCIENCEFLOW_DISABLE_GPU"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        if spec.gpu_auto:
            async with self._gpu_auto_lock:
                candidates = spec.gpu_auto_candidates or None
                chosen = select_least_used_gpu(
                    candidates=candidates,
                    exclude=self._gpu_auto_assigned,
                    count=spec.gpu_auto_count,
                )
                if chosen is not None:
                    resolved_gpu = chosen
                    chosen_ids = [
                        int(g.strip()) for g in chosen.split(",") if g.strip()
                    ]
                    _assigned_auto_gpus = [
                        g for g in chosen_ids if g not in self._gpu_auto_assigned
                    ]
                    for g in _assigned_auto_gpus:
                        self._gpu_auto_assigned.add(g)
                    if len(_assigned_auto_gpus) < len(chosen_ids):
                        logger.info(
                            "[%s] GPU stacked: %s (some already assigned to other tasks)",
                            spec.run_id,
                            chosen,
                        )
                else:
                    logger.warning(
                        "[%s] gpu_list=auto but nvidia-smi unavailable; "
                        "clearing CUDA_VISIBLE_DEVICES to let CUDA auto-select",
                        spec.run_id,
                    )
                    env.pop("CUDA_VISIBLE_DEVICES", None)

        if resolved_gpu and not gpu_disabled:
            env["CUDA_VISIBLE_DEVICES"] = resolved_gpu
            env["SCIENCEFLOW_RESOLVED_CUDA_DEVICES"] = resolved_gpu
            env["SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL"] = resolved_gpu
        if effective_cpu_list:
            env["SCIENCEFLOW_TASK_CPU_LIST"] = effective_cpu_list
        if spec.lnr_patch and spec.phase == "run" and spec.type == "lnr":
            effective_lnr_patch = dict(spec.lnr_patch)
            if resolved_gpu and not gpu_disabled and not effective_lnr_patch.get("resource_gpu_pool"):
                effective_lnr_patch["resource_gpu_pool"] = [g.strip() for g in resolved_gpu.split(",") if g.strip()]
            env["SCIENCEFLOW_PARALLEL_LNR_JSON"] = json.dumps(effective_lnr_patch)
        if spec.api_key:
            env["API_KEY"] = spec.api_key
        if spec.api_keys:
            env["API_KEYS"] = spec.api_keys
        elif spec.api_key:
            env.pop("API_KEYS", None)
        if spec.base_url:
            env["BASE_URL"] = spec.base_url
        if spec.base_urls:
            env["BASE_URLS"] = spec.base_urls
        elif spec.base_url:
            env.pop("BASE_URLS", None)
        inherited_key_pool = bool(
            (not spec.api_key)
            and (not spec.api_keys)
            and any(
                env.get(name, "").strip()
                for name in ("API_KEYS", "CODE_API_KEYS", "FEEDBACK_API_KEYS")
            )
        )
        inherited_sticky_primary_index: int | None = None
        if inherited_key_pool:
            inherited_sticky_primary_index = _env_sticky_primary_index(
                env,
                spec.manifest_index,
            )
        if spec.llm_routing_mode:
            env["SCIENCEFLOW_LLM_ROUTING_MODE"] = spec.llm_routing_mode
        elif inherited_key_pool:
            env.setdefault("SCIENCEFLOW_LLM_ROUTING_MODE", "sticky_failover")
        if spec.llm_sticky_id:
            env["SCIENCEFLOW_LLM_STICKY_ID"] = spec.llm_sticky_id
        elif inherited_key_pool:
            env.setdefault("SCIENCEFLOW_LLM_STICKY_ID", spec.run_id or spec.exp_id)
        if spec.llm_sticky_primary_index is not None:
            env["SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX"] = str(
                spec.llm_sticky_primary_index,
            )
        elif inherited_sticky_primary_index is not None:
            env.setdefault(
                "SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX",
                str(inherited_sticky_primary_index),
            )
        env["SCIENCEFLOW_EXP_ID"] = spec.exp_id
        if spec.run_id != spec.exp_id:
            env["SCIENCEFLOW_RUN_ID"] = spec.run_id
        else:
            env.pop("SCIENCEFLOW_RUN_ID", None)
        apatch = spec.agent_patch
        if apatch and spec.phase in {"run", "prep"}:
            env["SCIENCEFLOW_PARALLEL_AGENT_JSON"] = json.dumps(apatch)

        key_mode = "pool"
        if spec.api_keys and spec.llm_routing_mode:
            key_mode = f"pool:{spec.llm_routing_mode}"
        elif spec.api_key:
            key_mode = "assigned"
        elif inherited_key_pool:
            primary = env.get("SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX", "").strip()
            key_mode = "env:sticky_failover"
            if primary:
                key_mode = f"{key_mode}:primary={primary}"
        elif not spec.api_keys:
            key_mode = "env"
        key_info = f"keys={key_mode}"
        gpu_info = f"gpu={resolved_gpu}" + (" (auto)" if spec.gpu_auto else "")
        llm_config = summarize_llm_config_from_env(env, source="parallel_runner_env")
        agent_llm_config = summarize_llm_config_from_agent_patch(spec.agent_patch)
        if agent_llm_config:
            llm_config = merge_llm_config_summary(
                llm_config,
                agent_llm_config,
                source="parallel_runner_env+agent_patch",
            )
        llm_info = format_llm_config_summary(llm_config).replace("\n", "; ") or "llm=unknown"
        logger.info(
            f"[{spec.run_id}] ({spec.exp_id}) Starting phase={spec.phase} cpu={spec.cpu_list} "
            f"{gpu_info} {key_info} llm={llm_info}",
        )
        _write_gpu_assignment_json(spec, resolved_gpu)
        self._write_running_state(spec, resolved_gpu=resolved_gpu, llm_config=llm_config, log_file=log_file)

        task_started_at = time.time()
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as lf:
                proc = await spawn_exec(
                    *cmd,
                    stdout=lf,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                )
                try:
                    await asyncio.wait_for(proc.wait(), timeout=spec.time_limit)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    await terminate_tree(proc)
                    raise
                finally:
                    _LIVE_PGIDS.discard(proc.pid)
        finally:
            if resolved_gpu and not gpu_disabled:
                try:
                    cleanup = await asyncio.to_thread(
                        cleanup_workspace_gpu_processes,
                        workspace_dir=spec.workspace,
                        allowed_gpu_ids=[g.strip() for g in str(resolved_gpu).split(",") if g.strip()],
                        task_started_at=task_started_at,
                        reason="parallel_task_finish",
                        dry_run=False,
                        sigterm_grace_sec=0.5,
                    )
                    if (
                        int(cleanup.get("killed_count") or 0) > 0
                        or int(cleanup.get("violation_count") or 0) > 0
                        or any(str((row or {}).get("skip_reason") or "") not in {"workspace_mismatch", "process_started_before_task"} for row in cleanup.get("skipped") or [])
                    ):
                        with open(log_file, "a") as lf:
                            lf.write("\n[parallel-runner] workspace_gpu_cleanup ")
                            lf.write(json.dumps(cleanup, ensure_ascii=False, sort_keys=True)[:4000])
                            lf.write("\n")
                except Exception:
                    logger.debug("[parallel-runner] workspace gpu cleanup failed", exc_info=True)
            # Release GPU assignment so subsequent tasks can reuse these GPUs.
            # Runs on success, timeout, and any other exception.
            if _assigned_auto_gpus:
                async with self._gpu_auto_lock:
                    for g in _assigned_auto_gpus:
                        self._gpu_auto_assigned.discard(g)

        return proc.returncode, resolved_gpu

    def _check_resume(self, spec: TaskSpec) -> int | None:
        """Return remaining seconds, or None if no prior state."""
        state_file = _parallel_state_file_for_read(spec.workspace, task_type=spec.type)
        if state_file is None:
            return None
        try:
            state = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        child_failed = _state_completed_but_child_failed(spec.workspace)
        if state.get("status") == "completed" and not child_failed:
            stop_reason = str(state.get("stop_reason") or state.get("failure_kind") or "").strip()
            if stop_reason not in {"budget_expired", "time_budget_expired"}:
                return 0

        if getattr(self, "_resume_budget_policy", "remaining") == "fresh":
            return int(spec.time_limit)

        if child_failed and _workspace_lnr_failure_kind(spec.workspace) in _EXTERNAL_LLM_FAILURE_KINDS:
            charged_elapsed = 0
        else:
            charged_elapsed = state.get("charged_elapsed_sec", state.get("elapsed_sec", 0))
        return max(0, self._logical_resume_budget_sec(spec) - int(float(charged_elapsed or 0)))

    @staticmethod
    def _logical_resume_budget_sec(spec: TaskSpec) -> int:
        """Return the user-facing budget that resume should spend down.

        Parallel manifests often set ``time_limit`` slightly above the LNR wall
        clock to give the child process shutdown headroom. Resume accounting
        should spend down the LNR budget, not that outer cushion.
        """
        budget = _positive_int_or_none(spec.time_limit) or 0
        if spec.phase == "run" and spec.type == "lnr" and isinstance(spec.lnr_patch, dict):
            lnr_budget = _positive_int_or_none(spec.lnr_patch.get("wall_clock_budget_sec"))
            if lnr_budget is not None:
                return min(budget, lnr_budget) if budget > 0 else lnr_budget
        return budget

    @staticmethod
    def _with_resume_remaining_budget(spec: TaskSpec, remaining_sec: int) -> TaskSpec:
        remaining = max(0, int(remaining_sec))
        updates: dict[str, Any] = {"time_limit": remaining}
        if spec.phase == "run" and spec.type == "lnr":
            lnr_patch = dict(spec.lnr_patch or {})
            if "wall_clock_budget_sec" in lnr_patch:
                lnr_patch["wall_clock_budget_sec"] = remaining
            updates["lnr_patch"] = lnr_patch or None
        return TaskSpec(**{**spec.__dict__, **updates})

    def _write_interrupted_states(self, specs: list[TaskSpec], *, error: str) -> None:
        for spec in specs:
            self._write_interrupted_state(spec, error=error)

    def _write_interrupted_state(self, spec: TaskSpec, *, error: str) -> None:
        base = _ensure_workspace_for_state_write(spec.workspace)
        if base is None:
            logger.warning(
                f"[{spec.exp_id}] Skipped writing interrupted state.json; fix workspace path: {spec.workspace!r}"
            )
            return
        logs_dir = _task_state_log_dir(spec)
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"[{spec.exp_id}] cannot create logs directory: {e}")
            return
        state_file = logs_dir / "state.json"
        try:
            state = _load_json_file(state_file)
            now = time.time()
            started_at = float(state.get("run_started_at") or 0.0)
            segment_elapsed = max(0.0, now - started_at) if started_at > 0 else 0.0
            prior_charged = float(
                state.get("resume_prior_charged_elapsed_sec")
                if state.get("resume_prior_charged_elapsed_sec") is not None
                else state.get("charged_elapsed_sec")
                or 0.0
            )
            state.update(
                {
                    "exp_id": spec.exp_id,
                    "run_id": spec.run_id,
                    "status": "stopped_by_user",
                    "elapsed_sec": round(segment_elapsed, 1),
                    "segment_elapsed_sec": round(segment_elapsed, 1),
                    "charged_elapsed_sec": prior_charged + segment_elapsed,
                    "resume_prior_charged_elapsed_sec": prior_charged,
                    "resume_total_budget_sec": prior_charged + float(spec.time_limit or 0),
                    "exit_code": None,
                    "error": error,
                    "time_limit_sec": spec.time_limit,
                    "stopped_at": now,
                }
            )
            state_file.write_text(json.dumps(state, indent=2))
        except OSError as e:
            logger.warning(f"[{spec.exp_id}] Failed to write interrupted state: {e}")

    def _write_running_state(
        self,
        spec: TaskSpec,
        *,
        resolved_gpu: str = "",
        llm_config: dict[str, Any] | None = None,
        log_file: str = "",
    ) -> None:
        base = _ensure_workspace_for_state_write(spec.workspace)
        if base is None:
            logger.warning(
                f"[{spec.exp_id}] Skipped writing running state.json; fix workspace path: {spec.workspace!r}"
            )
            return
        logs_dir = _task_state_log_dir(spec)
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"[{spec.exp_id}] cannot create logs directory: {e}")
            return
        state_file = logs_dir / "state.json"
        try:
            state = _load_json_file(state_file)
            prior_charged = float(state.get("charged_elapsed_sec") or 0.0)
            total_budget = prior_charged + float(spec.time_limit or 0)
            state.update(
                {
                    "exp_id": spec.exp_id,
                    "run_id": spec.run_id,
                    "status": "running",
                    "elapsed_sec": 0.0,
                    "segment_elapsed_sec": 0.0,
                    "charged_elapsed_sec": prior_charged,
                    "resume_prior_charged_elapsed_sec": prior_charged,
                    "resume_total_budget_sec": total_budget,
                    "run_started_at": time.time(),
                    "exit_code": None,
                    "error": "",
                    "time_limit_sec": spec.time_limit,
                }
            )
            rg = resolved_gpu or _read_resolved_gpu_from_assignment(logs_dir)
            if rg:
                state["resolved_gpu"] = rg
            if spec.gpu_auto:
                state["gpu_auto"] = True
            raw_gl = (spec.gpu_list_raw or "").strip()
            if raw_gl:
                state["raw_gpu_list"] = raw_gl
            if llm_config:
                state["llm_config"] = llm_config
            for stale_key in ("failure_kind", "resume_retriable", "stop_reason"):
                state.pop(stale_key, None)
            policy = getattr(self, "_resume_budget_policy", "remaining")
            if policy != "remaining":
                state["resume_budget_policy"] = policy
            else:
                state.pop("resume_budget_policy", None)
            if log_file:
                state["log_file"] = str(log_file)
            state_file.write_text(json.dumps(state, indent=2))
        except OSError as e:
            logger.warning(f"[{spec.exp_id}] Failed to write running state: {e}")

    def _write_state(self, spec: TaskSpec, result: TaskResult, *, resolved_gpu: str = "") -> None:
        base = _ensure_workspace_for_state_write(spec.workspace)
        if base is None:
            logger.warning(
                f"[{spec.exp_id}] Skipped writing state.json; fix workspace path: {spec.workspace!r}"
            )
            return
        logs_dir = _task_state_log_dir(spec)
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"[{spec.exp_id}] cannot create logs directory: {e}")
            return
        state_file = logs_dir / "state.json"
        try:
            existing_state = _load_json_file(state_file)
            rg = resolved_gpu or _read_resolved_gpu_from_assignment(logs_dir)
            failure_kind = _parallel_failure_kind(result)
            lnr_done_payload = _workspace_lnr_done_payload(spec.workspace) if spec.type == "lnr" else {}
            lnr_stop_reason = str(lnr_done_payload.get("stop_reason") or "").strip()
            prior_charged = float(existing_state.get("resume_prior_charged_elapsed_sec") or 0.0)
            segment_charged_elapsed_sec = _charged_elapsed_for_resume(result, failure_kind)
            charged_elapsed_sec = prior_charged + segment_charged_elapsed_sec
            state = {
                "exp_id": spec.exp_id,
                "run_id": spec.run_id,
                "status": "completed" if result.status == "success" else result.status,
                "elapsed_sec": result.elapsed_sec,
                "segment_elapsed_sec": result.elapsed_sec,
                "charged_elapsed_sec": charged_elapsed_sec,
                "resume_prior_charged_elapsed_sec": prior_charged,
                "resume_total_budget_sec": prior_charged + float(spec.time_limit or 0),
                "exit_code": result.exit_code,
                "error": result.error,
                "time_limit_sec": spec.time_limit,
            }
            llm_config = existing_state.get("llm_config")
            if isinstance(llm_config, dict) and llm_config:
                state["llm_config"] = llm_config
            log_file = result.log_file or str(existing_state.get("log_file") or "")
            if log_file:
                state["log_file"] = log_file
            policy = getattr(self, "_resume_budget_policy", "remaining")
            if policy != "remaining":
                state["resume_budget_policy"] = policy
            if result.status == "success" and lnr_stop_reason:
                state["stop_reason"] = lnr_stop_reason
            elif failure_kind == "time_budget_expired" and result.status == _BUDGET_DONE_STATUS:
                state["stop_reason"] = failure_kind
            elif failure_kind:
                state["failure_kind"] = failure_kind
                state["resume_retriable"] = result.status in {"failed", "error"}
                if failure_kind in _EXTERNAL_LLM_FAILURE_KINDS:
                    state["resume_budget_policy"] = "external_llm_failure_not_charged"
            if rg:
                state["resolved_gpu"] = rg
            if spec.gpu_auto:
                state["gpu_auto"] = True
            raw_gl = (spec.gpu_list_raw or "").strip()
            if raw_gl:
                state["raw_gpu_list"] = raw_gl
            state_file.write_text(json.dumps(state, indent=2))
        except OSError as e:
            logger.warning(f"[{spec.exp_id}] Failed to write state: {e}")

    @staticmethod
    def format_summary(results: list[TaskResult]) -> str:
        lines = [
            f"{'run_id':<32} {'Status':<10} {'Time':>8} {'Exit':>5}",
            "-" * 60,
        ]
        for r in results:
            time_str = f"{r.elapsed_sec:.0f}s" if r.elapsed_sec else "-"
            exit_str = str(r.exit_code) if r.exit_code is not None else "-"
            rid = r.run_id or r.exp_id
            lines.append(f"{rid:<32} {r.status:<10} {time_str:>8} {exit_str:>5}")
            if r.error:
                lines.append(f"  error: {r.error}")
        return "\n".join(lines)
