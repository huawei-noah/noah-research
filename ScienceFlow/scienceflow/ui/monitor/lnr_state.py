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

"""Build monitor state from current LNR task artifacts.

The legacy monitor tails ``logs/monitor_state.json``.  Modern LNR runs keep the
authoritative state in task logs instead:

* ``task_logs/lhr_stage_performance.csv``
* ``task_logs/resource/resource_events.jsonl``
* ``task_logs/workers/w*/scienceflow_time_trace.csv``

This module adapts those files into a compact state dict for the Rich monitor.
It is intentionally read-only and best-effort: missing or malformed files should
degrade the display, not affect a running experiment.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
from collections import Counter
from collections.abc import Iterable

import yaml
from pathlib import Path
from typing import Any

from scienceflow.utils.llm_cost import (
    estimate_llm_cost_usd,
    extract_model_from_trace_detail,
    load_price_table,
)
from scienceflow.utils.llm_config_summary import (
    format_llm_config_summary,
    merge_observed_models,
    observed_models_from_log_file,
    summarize_llm_config_from_resolved_config,
)
from scienceflow.utils.time_trace import TRACE_FILENAME


def _time_trace_paths(task_root: Path) -> list[Path]:
    directories = [task_root / "task_logs"]
    directories.extend(task_root.glob("task_logs/workers/w*"))
    directories.extend(task_root.glob("workers/w*/logs"))
    paths: list[Path] = []
    for directory in directories:
        candidate = directory / TRACE_FILENAME
        if candidate.is_file():
            paths.append(candidate)
    return paths
from scienceflow.solver.lnr.stage.score_summary import (
    build_score_summary,
    infer_stage_rows_lower_is_better,
    score_record_from_stage_performance_row,
)


_MAX_RESOURCE_EVENTS = 5000
_RECENT_JSONL_TAIL_BYTES = 4 * 1024 * 1024
_TERMINAL_DISPLAY_STATUSES = {
    "finished",
    "failed",
    "early_stop",
    "budget_expired",
    "budget_done",
    "steps_completed",
    "timeout",
    "skipped",
    "stopped_by_user",
}
_PARALLEL_FINAL_STATUSES = {"finished", "failed", "budget_done", "timeout", "skipped", "stopped_by_user"}


def load_lnr_task_state(task_root: Path) -> dict[str, Any]:
    """Return a monitor state dict for an LNR task workspace."""
    root = Path(task_root)
    if not (root / "task_logs").is_dir():
        return {}

    stages = _load_csv_rows(root / "task_logs" / "lhr_stage_performance.csv")
    stage_summary = _summarize_stages(stages)
    event_paths = _lnr_event_paths(root)
    resource_summary = _summarize_resource_events([
        root / "task_logs" / "resource" / "resource_events.jsonl",
        *event_paths,
    ])
    resource_current = _load_current_resource_snapshot(root)
    estra_summary = _merge_estra_summaries(
        _summarize_estra_events(event_paths),
        _summarize_worker_estra_states(root),
    )
    trace_summary = _summarize_time_traces(root)
    budget_sec = _load_wall_clock_budget_sec(root)
    newest_mtime = _newest_mtime(root, stages)

    active_jobs = resource_summary.get("active_jobs", [])
    worker_states = _filter_stale_wait_states(
        resource_summary.get("worker_states", []),
        resource_current,
    )
    live_process_summary = _live_process_summary_for_root(root)
    running_process_count = int(live_process_summary.get("count") or 0)
    final_state = _load_parallel_task_state(root)
    budget_sec = _effective_wall_clock_budget_sec(budget_sec, final_state)
    llm_config = _load_llm_config_summary(root, final_state=final_state, trace_summary=trace_summary)
    run_status = _load_run_status(root, final_state=final_state)
    live = running_process_count > 0 or run_status in {"run", "running"}
    run_started_at = _load_live_run_started_at(root) if live else None
    elapsed_sec = _elapsed_from_live_process_summary(live_process_summary, final_state) if live else None
    if elapsed_sec is None and not live:
        elapsed_sec = _elapsed_from_final_state(final_state)
    if elapsed_sec is None:
        elapsed_sec = _elapsed_from_resource_range(resource_summary, live=live, run_started_at=run_started_at)
    progress_ratio = (elapsed_sec / budget_sec) if budget_sec and elapsed_sec is not None else None
    status = _display_status(run_status=run_status, running_process_count=running_process_count, active_jobs=active_jobs)
    if status in _TERMINAL_DISPLAY_STATUSES:
        active_jobs = []
        worker_states = []
    return {
        "monitor_kind": "lnr_task",
        "status": status,
        "task_name": root.name,
        "exp_id": root.name,
        "task_root": str(root),
        "timestamp": _format_timestamp(newest_mtime),
        "elapsed_sec": elapsed_sec,
        "total_sec": budget_sec,
        "run_started_at": run_started_at,
        "remaining_sec": (max(0.0, budget_sec - elapsed_sec) if budget_sec and elapsed_sec is not None else None),
        "progress_ratio": (min(1.0, max(0.0, progress_ratio)) if progress_ratio is not None else None),
        "stage_count": stage_summary["stage_count"],
        "candidate_stage_count": stage_summary["candidate_stage_count"],
        "stages_by_worker": stage_summary["stages_by_worker"],
        "candidate_stages_by_worker": stage_summary["candidate_stages_by_worker"],
        "ready_stage_count": stage_summary["ready_stage_count"],
        "eligible_stage_count": stage_summary["eligible_stage_count"],
        "duplicate_stage_count": stage_summary["duplicate_stage_count"],
        "low_validity_stage_count": stage_summary["low_validity_stage_count"],
        "latest_stage": stage_summary["latest_stage"],
        "latest_stage_by_worker": stage_summary["latest_stage_by_worker"],
        "best_raw_metric": stage_summary["best_raw_metric"],
        "best_valid_metric": stage_summary["best_valid_metric"],
        "best_raw_candidate": stage_summary["best_raw_candidate"],
        "best_valid_candidate": stage_summary["best_valid_candidate"],
        "best_raw_validity": stage_summary["best_raw_validity"],
        "best_valid_validity": stage_summary["best_valid_validity"],
        "lower_is_better": stage_summary["lower_is_better"],
        "active_jobs": active_jobs,
        "active_job_count": len(active_jobs),
        "worker_states": worker_states,
        "running_process_count": running_process_count,
        "resource_counts": resource_summary["counts"],
        "resource_current": resource_current,
        "resource_recent": resource_summary["recent"],
        "resource_outcomes": resource_summary["outcomes"],
        "resource_actionable_outcomes": resource_summary["actionable_outcomes"],
        "advisory_preferences": resource_summary["advisory_preferences"],
        "review_boundaries": resource_summary["review_boundaries"],
        "estra_decision_count": estra_summary["decision_count"],
        "estra_switch_count": estra_summary["switch_count"],
        "estra_redirect_count": estra_summary["redirect_count"],
        "estra_continue_count": estra_summary["continue_count"],
        "estra_current_continue_count": estra_summary["current_continue_count"],
        "estra_current_redirect_count": estra_summary["current_redirect_count"],
        "estra_stage_continue_count": estra_summary["stage_continue_count"],
        "estra_stage_redirect_count": estra_summary["stage_redirect_count"],
        "estra_invalid_count": estra_summary["invalid_count"],
        "estra_compact_count": estra_summary["compact_count"],
        "estra_raw_decision_count": estra_summary["raw_decision_count"],
        "estra_repeat_decision_count": estra_summary["repeat_decision_count"],
        "estra_text_only_check_count": estra_summary["text_only_check_count"],
        "estra_context_check_count": estra_summary["context_check_count"],
        "estra_text_only_trigger_count": estra_summary["text_only_trigger_count"],
        "estra_context_trigger_count": estra_summary["context_trigger_count"],
        "total_tokens_in": trace_summary["tokens_in"],
        "total_tokens_out": trace_summary["tokens_out"],
        "total_tokens_cached": trace_summary["tokens_cached"],
        "total_llm_calls": trace_summary["llm_calls"],
        "total_llm_cost_usd": trace_summary["cost_usd"],
        "llm_cost_known_calls": trace_summary["cost_known_calls"],
        "llm_cache_rate": trace_summary["cache_rate"],
        "llm_config": llm_config,
        "llm_config_text": format_llm_config_summary(llm_config),
        "observed_llm_models": llm_config.get("observed_models", []) if isinstance(llm_config, dict) else [],
        "max_ttft_sec": trace_summary["max_ttft_sec"],
        "max_tpot_ms": trace_summary["max_tpot_ms"],
    }


def _load_current_resource_snapshot(task_root: Path) -> dict[str, Any]:
    """Return current lease/waiter counts without changing resource state."""
    resource_dir = Path(task_root) / "task_logs" / "resource"
    gpu_leases = _load_json_dict(resource_dir / "gpu_leases.json")
    resource_state = _load_json_dict(resource_dir / "resource_state.json")
    has_snapshot = bool(gpu_leases or resource_state)
    lease_count = max(
        _collection_len(gpu_leases.get("leases")),
        _collection_len(resource_state.get("leases")),
    )
    waiter_count = max(
        _collection_len(gpu_leases.get("waiters")),
        _collection_len(resource_state.get("waiters")),
    )
    waiter_workers = sorted(
        _worker_ids_from_collection(gpu_leases.get("waiters"))
        | _worker_ids_from_collection(resource_state.get("waiters"))
    )
    return {
        "has_snapshot": has_snapshot,
        "active_lease_count": lease_count,
        "active_waiter_count": waiter_count,
        "active_waiter_workers": waiter_workers,
    }


def _filter_stale_wait_states(worker_states: Any, current: dict[str, Any]) -> list[dict[str, Any]]:
    states = [s for s in worker_states if isinstance(s, dict)]
    if not current.get("has_snapshot"):
        return states
    waiter_count = int(current.get("active_waiter_count") or 0)
    if waiter_count <= 0:
        return [s for s in states if s.get("kind") != "wait"]
    waiter_workers = {
        str(w).upper()
        for w in current.get("active_waiter_workers") or []
        if str(w).strip()
    }
    if not waiter_workers:
        return states
    return [
        s
        for s in states
        if s.get("kind") != "wait" or str(s.get("worker_id") or "").upper() in waiter_workers
    ]


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _collection_len(value: Any) -> int:
    if isinstance(value, (dict, list, tuple, set)):
        return len(value)
    return 0


def _worker_ids_from_collection(value: Any) -> set[str]:
    workers: set[str] = set()
    items: Iterable[Any]
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = value
    else:
        return workers
    for item in items:
        key: Any = ""
        payload: Any = item
        if isinstance(item, tuple) and len(item) == 2:
            key, payload = item
        candidates = [key]
        if isinstance(payload, dict):
            candidates.extend(
                [
                    payload.get("worker_id"),
                    payload.get("worker"),
                    payload.get("job_id"),
                    payload.get("command_id"),
                ]
            )
        for candidate in candidates:
            worker = _worker_id_from_text(candidate)
            if worker:
                workers.add(worker)
    return workers


def _worker_id_from_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if text.startswith("W") and len(text) >= 3 and text[1:3].isdigit():
        return text[:3]
    for part in text.replace(":", " ").replace("/", " ").split():
        if part.startswith("W") and len(part) >= 3 and part[1:3].isdigit():
            return part[:3]
    return ""


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""

def _live_process_summary_for_root(task_root: Path) -> dict[str, Any]:
    """Return live process count and oldest start time for *task_root*.

    Direct resume runs are launched from the project checkout, so their cwd is
    not inside the task workspace. They still carry ``--workspace <task_root>``
    in argv. Check both cwd and cmdline so the monitor does not display a
    historical terminal status while a resume process is active.  When a resume
    is active, progress should be relative to the new process lifetime, not the
    old task's completed state.
    """
    root_text = str(task_root)
    proc = Path("/proc")
    boot_time = _linux_boot_time()
    count = 0
    oldest_start: float | None = None
    try:
        entries = list(proc.iterdir())
    except OSError:
        return {"count": 0, "oldest_start_ts": None}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        matched = False
        try:
            cwd = (entry / "cwd").readlink()
        except OSError:
            cwd = None
        if cwd is not None and root_text in str(cwd):
            matched = True
        else:
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                    "utf-8",
                    errors="ignore",
                )
            except OSError:
                cmdline = ""
            matched = root_text in cmdline
        if not matched:
            continue
        count += 1
        start_ts = _proc_start_timestamp(entry, boot_time=boot_time)
        if start_ts is not None and (oldest_start is None or start_ts < oldest_start):
            oldest_start = start_ts
    return {"count": count, "oldest_start_ts": oldest_start}


def _count_live_processes_for_root(task_root: Path) -> int:
    return int(_live_process_summary_for_root(task_root).get("count") or 0)


def _elapsed_from_live_process_summary(
    summary: dict[str, Any],
    final_state: dict[str, Any] | None = None,
) -> float | None:
    start_ts = _to_float(summary.get("oldest_start_ts"))
    if start_ts is None:
        return None
    segment_elapsed = max(0.0, _dt.datetime.now().timestamp() - start_ts)
    prior_elapsed = _resume_prior_charged_elapsed_sec(final_state or {})
    return prior_elapsed + segment_elapsed


def _linux_boot_time() -> float | None:
    try:
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, IndexError, ValueError):
        return None
    return _dt.datetime.now().timestamp() - uptime


def _proc_start_timestamp(proc_entry: Path, *, boot_time: float | None) -> float | None:
    if boot_time is None:
        return None
    try:
        text = (proc_entry / "stat").read_text(encoding="utf-8", errors="ignore")
        after_comm = text[text.rfind(")") + 2 :].split()
        start_ticks = float(after_comm[19])
        ticks_per_sec = float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    except (OSError, IndexError, KeyError, TypeError, ValueError):
        return None
    if ticks_per_sec <= 0:
        return None
    return boot_time + (start_ticks / ticks_per_sec)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def _summarize_stages(rows: list[dict[str, str]]) -> dict[str, Any]:
    records = [rec for row in rows if (rec := score_record_from_stage_performance_row(row)) is not None]
    score_summary = build_score_summary(records)
    best_raw = score_summary.get("best_score") if isinstance(score_summary.get("best_score"), dict) else {}
    best_valid = score_summary.get("valid_best_score") if isinstance(score_summary.get("valid_best_score"), dict) else {}
    lower = _summary_lower_is_better(rows, best_raw, best_valid)
    with_metric = [r for r in rows if _to_float(r.get("metric_value")) is not None]
    ready = [
        r for r in with_metric
        if _truthy(r.get("candidate_ready")) and _truthy(r.get("selection_eligible"))
    ]
    stage_keys = [_stage_count_key(row) for row in rows]
    stage_keys = [key for key in stage_keys if key is not None]
    unique_stage_keys = set(stage_keys)
    stages_by_worker = Counter(worker for worker, _ in unique_stage_keys)
    candidate_stages_by_worker = Counter((r.get("worker_id") or "?").upper() for r in rows)
    return {
        "stage_count": len(unique_stage_keys) if unique_stage_keys else len(rows),
        "candidate_stage_count": len(rows),
        "stages_by_worker": dict(stages_by_worker or candidate_stages_by_worker),
        "candidate_stages_by_worker": dict(candidate_stages_by_worker),
        "ready_stage_count": len([r for r in rows if _truthy(r.get("candidate_ready"))]),
        "eligible_stage_count": len(ready),
        "duplicate_stage_count": len([
            r for r in rows
            if (r.get("duplicate_submission_of_stage") or r.get("duplicate_submission_of_snapshot_id"))
        ]),
        "low_validity_stage_count": len([
            r for r in rows
            if str(r.get("metric_validity") or "").strip().lower() in {"low", "invalid"}
        ]),
        "latest_stage": _stage_brief(rows[-1]) if rows else {},
        "latest_stage_by_worker": _latest_stage_by_worker(rows),
        "best_raw_metric": best_raw.get("value"),
        "best_raw_candidate": best_raw.get("candidate_id") or "",
        "best_raw_validity": best_raw.get("metric_validity") or "",
        "best_valid_metric": best_valid.get("value"),
        "best_valid_candidate": best_valid.get("candidate_id") or "",
        "best_valid_validity": best_valid.get("metric_validity") or "",
        "lower_is_better": lower,
    }


def _stage_count_key(row: dict[str, str]) -> tuple[str, str] | None:
    worker = str(row.get("worker_id") or "?").strip().upper() or "?"
    stage_id = str(row.get("stage_id") or "").strip()
    if _looks_like_stage_id(stage_id):
        return worker, stage_id
    candidate_id = str(row.get("candidate_id") or "").strip()
    candidate_stage = candidate_id.rsplit(":", 1)[-1] if candidate_id else ""
    if _looks_like_stage_id(candidate_stage):
        return worker, candidate_stage
    return None


def _looks_like_stage_id(value: str) -> bool:
    return len(value) > 1 and value[0].upper() == "S" and value[1:].isdigit()


def _summarize_resource_events(paths: Path | Iterable[Path]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    advisory: Counter[str] = Counter()
    boundaries: Counter[str] = Counter()
    recent: list[dict[str, Any]] = []
    latest_jobs: dict[str, dict[str, Any]] = {}
    latest_waits: dict[str, dict[str, Any]] = {}
    first_event_at: float | None = None
    last_event_at: float | None = None

    if isinstance(paths, Path):
        event_paths = [paths]
    else:
        event_paths = [Path(p) for p in paths]

    for obj in _iter_recent_jsonl_many(event_paths):
        created_at = _to_float(obj.get("created_at") or obj.get("timestamp"))
        if created_at is not None:
            first_event_at = created_at if first_event_at is None else min(first_event_at, created_at)
            last_event_at = created_at if last_event_at is None else max(last_event_at, created_at)
        event_type = str(obj.get("event_type") or obj.get("event") or "unknown")
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        counts[event_type] += 1

        outcome = payload.get("execution_outcome") or payload.get("outcome")
        if outcome:
            outcomes[str(outcome)] += 1
        boundary = payload.get("resource_review_boundary") or payload.get("review_boundary")
        if isinstance(boundary, dict) and boundary.get("kind"):
            boundaries[str(boundary["kind"])] += 1
        elif boundary:
            boundaries[str(boundary)] += 1

        adv = payload.get("advisory") or payload.get("parsed_response")
        if isinstance(adv, dict) and adv.get("preference"):
            advisory[str(adv["preference"])] += 1

        command_id = str(obj.get("command_id") or payload.get("job_id") or "")
        if command_id and event_type in {
            "snapshot",
            "kill_proposal",
            "resource_kill_executed",
            "resource_review_outcome",
            "progress_heartbeat",
            "resource_monitor_heartbeat",
            "resource_job_finished",
            "resource_source_hint_detected",
        }:
            current = _job_summary(obj, payload)
            latest_jobs[command_id] = _merge_job_summary(latest_jobs.get(command_id), current)

        if event_type in {"admission_pending", "managed_resource_wait_offered"}:
            wait = _wait_summary(obj, payload)
            worker_id = str(wait.get("worker_id") or "").upper()
            if worker_id:
                previous = latest_waits.get(worker_id)
                if previous is None or float(wait.get("observed_at") or 0.0) >= float(previous.get("observed_at") or 0.0):
                    latest_waits[worker_id] = wait

        if event_type in {
            "resource_kill_executed",
            "resource_review_outcome",
            "main_agent_advisory",
            "execution",
            "admission_pending",
            "managed_resource_wait_offered",
        }:
            recent.append(_recent_event(obj, payload))

    active_job_candidates = [
        j for j in latest_jobs.values()
        if _job_is_active(j) and not _job_is_stale_for_display(j)
    ]
    active_jobs = _select_active_jobs_for_display(active_job_candidates)
    worker_states = _worker_states(active_job_candidates, latest_waits)
    return {
        "counts": dict(counts),
        "outcomes": dict(outcomes),
        "actionable_outcomes": {
            key: value for key, value in outcomes.items() if key != "NO_ACTION"
        },
        "advisory_preferences": dict(advisory),
        "review_boundaries": dict(boundaries),
        "active_jobs": active_jobs[:6],
        "worker_states": worker_states,
        "recent": recent[-8:],
        "first_event_at": first_event_at,
        "last_event_at": last_event_at,
    }


def _summarize_estra_events(paths: Path | Iterable[Path]) -> dict[str, int]:
    actions: Counter[str] = Counter()
    axes: Counter[str] = Counter()
    requested_compact_count = 0
    actual_compact_count = 0
    invalid_count = 0
    raw_decision_count = 0
    repeat_decision_count = 0
    text_only_check_count = 0
    context_check_count = 0
    text_only_trigger_count = 0
    context_trigger_count = 0
    seen_decisions: set[tuple[str, str, str, str, str, str]] = set()
    if isinstance(paths, Path):
        events = _iter_recent_jsonl(paths)
    else:
        events = _iter_recent_jsonl_many(paths)
    for obj in events:
        event_type = str(obj.get("event") or obj.get("event_type") or "")
        if event_type == "text_only_estra_check":
            text_only_check_count += 1
            continue
        if event_type in {"context_limit_estra_check", "cache_hygiene_compact_triggered"}:
            context_check_count += 1
            continue
        if event_type in {"estra_keep_current_compacted", "estra_memory_compacted"}:
            actual_compact_count += 1
            continue
        if event_type != "estra_decision":
            continue
        raw_decision_count += 1
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        action = str(payload.get("action") or "").strip()
        startpoint = str(payload.get("startpoint") or ("previous_stage" if action == "switch_stage" else "current_workspace"))
        intent = str(payload.get("intent") or ("redirect" if action == "keep_but_redirect" else "continue"))
        decision_key = _estra_decision_key(
            obj,
            payload,
            action=action,
            startpoint=startpoint,
            intent=intent,
        )
        if decision_key in seen_decisions:
            repeat_decision_count += 1
            continue
        seen_decisions.add(decision_key)
        if action in {"switch_stage", "keep_but_redirect", "keep_current"}:
            actions[action] += 1
        else:
            invalid_count += 1
        if startpoint == "previous_stage" and intent == "redirect":
            axes["stage_redirect"] += 1
        elif startpoint == "previous_stage":
            axes["stage_continue"] += 1
        elif intent == "redirect":
            axes["current_redirect"] += 1
        else:
            axes["current_continue"] += 1
        trigger_source = str(payload.get("trigger_source") or "").strip()
        if trigger_source == "text_only":
            text_only_trigger_count += 1
        elif trigger_source in {"context_limit", "context_hygiene"}:
            context_trigger_count += 1
        if bool(payload.get("compact")):
            requested_compact_count += 1
    compact_count = actual_compact_count or requested_compact_count
    return {
        "decision_count": int(sum(actions.values()) + invalid_count),
        "raw_decision_count": int(raw_decision_count),
        "repeat_decision_count": int(repeat_decision_count),
        "switch_count": int(actions.get("switch_stage", 0)),
        "redirect_count": int(actions.get("keep_but_redirect", 0)),
        "continue_count": int(actions.get("keep_current", 0)),
        "current_continue_count": int(axes.get("current_continue", 0)),
        "current_redirect_count": int(axes.get("current_redirect", 0)),
        "stage_continue_count": int(axes.get("stage_continue", 0)),
        "stage_redirect_count": int(axes.get("stage_redirect", 0)),
        "invalid_count": int(invalid_count),
        "compact_count": int(compact_count),
        "text_only_check_count": int(text_only_check_count),
        "context_check_count": int(context_check_count),
        "text_only_trigger_count": int(text_only_trigger_count),
        "context_trigger_count": int(context_trigger_count),
    }


def _estra_decision_key(
    obj: dict[str, Any],
    payload: dict[str, Any],
    *,
    action: str,
    startpoint: str,
    intent: str,
) -> tuple[str, str, str, str, str, str]:
    worker = str(obj.get("worker_id") or payload.get("worker_id") or "?").strip().upper() or "?"
    stage = _first_non_empty(
        payload.get("latest_stage"),
        payload.get("target_stage"),
        obj.get("latest_stage"),
        payload.get("stage_id"),
        obj.get("stage_id"),
        payload.get("stage_count"),
        obj.get("stage_count"),
    )
    if not stage:
        ts = _to_float(obj.get("timestamp") or obj.get("created_at"))
        stage = f"ts:{ts:.3f}" if ts is not None else "unknown"
    trigger = str(payload.get("trigger_source") or "").strip()
    return (worker, str(stage).strip().upper(), action, startpoint, intent, trigger)


def _summarize_worker_estra_states(task_root: Path) -> dict[str, int]:
    """Return estra counters from live worker state files.

    Worker-level ``lhr_state.json`` can be fresher than the task-level event
    rollup while a run is active.  The monitor is read-only, so this is used as
    a display fallback only; it does not backfill or mutate task state.
    """

    summary = _empty_estra_summary()
    field_map = {
        "decision_count": "estra_decisions",
        "switch_count": "estra_switch_count",
        "redirect_count": "estra_redirect_count",
        "continue_count": "estra_continue_count",
        "current_continue_count": "estra_current_continue_count",
        "current_redirect_count": "estra_current_redirect_count",
        "stage_continue_count": "estra_stage_continue_count",
        "stage_redirect_count": "estra_stage_redirect_count",
        "invalid_count": "estra_invalid_decisions",
        "compact_count": "estra_compact_count",
    }
    paths = list(Path(task_root).glob("task_logs/workers/w*/lhr_state.json"))
    paths.extend(Path(task_root).glob("workers/w*/logs/lhr_state.json"))
    for path in sorted(paths):
        data = _load_json_dict(path)
        if not data:
            continue
        for out_key, state_key in field_map.items():
            summary[out_key] += _to_int(data.get(state_key)) or 0
    return summary


def _merge_estra_summaries(event_summary: dict[str, int], worker_summary: dict[str, int]) -> dict[str, int]:
    """Prefer event counters, but fill stale/missing action counts from workers."""

    merged = _empty_estra_summary()
    merged.update({key: int(event_summary.get(key, 0) or 0) for key in merged})
    action_keys = (
        "decision_count",
        "switch_count",
        "redirect_count",
        "continue_count",
        "current_continue_count",
        "current_redirect_count",
        "stage_continue_count",
        "stage_redirect_count",
        "invalid_count",
        "compact_count",
    )
    has_event_actions = any(int(event_summary.get(key, 0) or 0) for key in ("raw_decision_count", *action_keys))
    for key in action_keys:
        if has_event_actions:
            merged[key] = int(event_summary.get(key, 0) or 0)
        else:
            merged[key] = int(worker_summary.get(key, 0) or 0)
    if not has_event_actions and merged.get("decision_count"):
        merged["raw_decision_count"] = int(merged.get("decision_count") or 0)
    return merged


def _empty_estra_summary() -> dict[str, int]:
    return {
        "decision_count": 0,
        "raw_decision_count": 0,
        "repeat_decision_count": 0,
        "switch_count": 0,
        "redirect_count": 0,
        "continue_count": 0,
        "current_continue_count": 0,
        "current_redirect_count": 0,
        "stage_continue_count": 0,
        "stage_redirect_count": 0,
        "invalid_count": 0,
        "compact_count": 0,
        "text_only_check_count": 0,
        "context_check_count": 0,
        "text_only_trigger_count": 0,
        "context_trigger_count": 0,
    }

def _load_wall_clock_budget_sec(task_root: Path) -> float | None:
    path = task_root / "resolved_config.yaml"
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None
    lnr = data.get("lnr") if isinstance(data, dict) else None
    if not isinstance(lnr, dict):
        return None
    value = _to_float(lnr.get("wall_clock_budget_sec"))
    return value if value and value > 0 else None


def _effective_wall_clock_budget_sec(config_budget_sec: float | None, final_state: dict[str, Any]) -> float | None:
    total_budget = _to_float(final_state.get("resume_total_budget_sec"))
    if total_budget is not None and total_budget > 0:
        if config_budget_sec is None or config_budget_sec <= 0:
            return total_budget
        return max(config_budget_sec, total_budget)
    return config_budget_sec


def _resume_prior_charged_elapsed_sec(final_state: dict[str, Any]) -> float:
    value = _to_float(final_state.get("resume_prior_charged_elapsed_sec"))
    if value is None:
        return 0.0
    return max(0.0, value)


def _load_llm_price_config(task_root: Path) -> dict[str, Any]:
    path = task_root / "resolved_config.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return {}
    agent = data.get("agent") if isinstance(data, dict) else None
    prices = agent.get("llm_prices") if isinstance(agent, dict) else None
    return prices if isinstance(prices, dict) else {}


def _load_llm_config_summary(
    task_root: Path,
    *,
    final_state: dict[str, Any],
    trace_summary: dict[str, Any],
) -> dict[str, Any]:
    state_llm = final_state.get("llm_config") if isinstance(final_state, dict) else None
    if isinstance(state_llm, dict) and state_llm:
        summary = dict(state_llm)
    else:
        summary = _load_llm_config_from_resolved_config(task_root)
    log_models = observed_models_from_log_file(final_state.get("log_file")) if isinstance(final_state, dict) else []
    observed = merge_observed_models(
        trace_summary.get("observed_models") if isinstance(trace_summary, dict) else [],
        log_models,
        summary.get("observed_models") if isinstance(summary, dict) else [],
    )
    if observed:
        summary = dict(summary) if summary else {"source": "observed"}
        summary["observed_models"] = observed
    return summary


def _load_llm_config_from_resolved_config(task_root: Path) -> dict[str, Any]:
    path = task_root / "resolved_config.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return {}
    return summarize_llm_config_from_resolved_config(data if isinstance(data, dict) else {})


def _load_parallel_task_state(task_root: Path) -> dict[str, Any]:
    path = task_root / "task_logs" / "state.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_run_status(task_root: Path, *, final_state: dict[str, Any] | None = None) -> str:
    final_status = _status_from_final_state(final_state or {})
    if final_status in _PARALLEL_FINAL_STATUSES:
        return final_status
    path = task_root / "task_logs" / "lhr_state.json"
    if not path.is_file():
        return final_status
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return final_status
    status = str(data.get("run_status") or data.get("status") or "").strip().lower()
    return status or final_status


def _load_live_run_started_at(task_root: Path) -> float | None:
    state_path = task_root / "task_logs" / "lhr_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        state = {}
    if isinstance(state, dict):
        started_at = _to_float(state.get("started_at") or state.get("run_started_at"))
        if started_at is not None and started_at > 0:
            return started_at

    multi_worker_starts: list[float] = []
    worker_starts: list[float] = []
    for path in _lnr_event_paths(task_root):
        for obj in _iter_jsonl(path):
            event = str(obj.get("event") or obj.get("event_type") or "")
            payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
            payload_event = str(payload.get("event") or "")
            ts = _to_float(obj.get("timestamp") or obj.get("created_at"))
            if ts is None:
                continue
            if event == "multi_worker_start" or payload_event == "multi_worker_start":
                multi_worker_starts.append(ts)
            elif event == "worker_start" or payload_event == "worker_start":
                worker_starts.append(ts)
    if multi_worker_starts:
        return max(multi_worker_starts)
    if worker_starts:
        return max(worker_starts)
    return None


def _status_from_final_state(final_state: dict[str, Any]) -> str:
    status = str(final_state.get("status") or "").strip().lower()
    error = str(final_state.get("error") or "").strip().lower()
    if status == "completed":
        return "finished"
    if status == "timeout" and error.startswith("exceeded "):
        return "budget_done"
    if status in {"budget_done", "timeout", "failed", "skipped", "stopped_by_user"}:
        return status
    return ""


def _elapsed_from_final_state(final_state: dict[str, Any]) -> float | None:
    value = _to_float(final_state.get("charged_elapsed_sec"))
    if value is None:
        value = _to_float(final_state.get("elapsed_sec"))
    return value if value is not None and value >= 0 else None


def _display_status(*, run_status: str, running_process_count: int, active_jobs: list[dict[str, Any]]) -> str:
    status = str(run_status or "").strip().lower()
    if running_process_count > 0 and status in _PARALLEL_FINAL_STATUSES:
        return "run_resume"
    if status in _TERMINAL_DISPLAY_STATUSES:
        return status
    if status in {"run", "running"} or running_process_count > 0:
        return "run"
    if active_jobs:
        return "tracked"
    return "unknown"


def _lnr_event_paths(task_root: Path) -> list[Path]:
    paths = [task_root / "task_logs" / "lhr_events.jsonl"]
    paths.extend(sorted(task_root.glob("task_logs/workers/w*/lhr_events.jsonl")))
    paths.extend(sorted(task_root.glob("workers/w*/logs/lhr_events.jsonl")))
    return paths


def _elapsed_from_resource_range(
    resource_summary: dict[str, Any],
    *,
    live: bool = False,
    run_started_at: float | None = None,
) -> float | None:
    first = _to_float(run_started_at) if live else None
    if first is None:
        first = _to_float(resource_summary.get("first_event_at"))
    last = _to_float(resource_summary.get("last_event_at"))
    if first is None:
        return None
    end = _dt.datetime.now().timestamp() if live else last
    if end is None:
        return None
    return max(0.0, end - first)


def _summarize_time_traces(task_root: Path) -> dict[str, Any]:
    tokens_in = tokens_out = tokens_cached = calls = 0
    cost_usd = 0.0
    cost_known_calls = 0
    model_counts: Counter[str] = Counter()
    price_table = load_price_table(_load_llm_price_config(task_root))
    max_ttft_sec: float | None = None
    max_tpot_ms: float | None = None
    paths = _time_trace_paths(task_root)
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        for row in _load_csv_rows(path):
            calls += 1
            ti = _first_int(row, ("tokens_input", "input_tokens"))
            to = _first_int(row, ("tokens_output", "output_tokens"))
            tc = _first_int(row, ("tokens_cached", "cached_tokens"))
            tokens_in += ti
            tokens_out += to
            tokens_cached += tc
            detail_s = str(row.get("detail") or "")
            model_name = extract_model_from_trace_detail(detail_s)
            if model_name:
                model_counts[model_name] += 1
            category_s = str(row.get("category") or "")
            cost_scope = category_s == "llm_api" or "token_scope=llm_api" in detail_s or not category_s
            if cost_scope:
                row_cost = _first_float(row, ("llm_cost_usd", "cost_usd"))
                if row_cost is None:
                    row_cost = estimate_llm_cost_usd(
                        model=extract_model_from_trace_detail(detail_s),
                        tokens_input=ti,
                        tokens_output=to,
                        tokens_cached=tc,
                        price_table=price_table,
                    )
                if row_cost is not None:
                    cost_usd += float(row_cost)
                    cost_known_calls += 1
            max_ttft_sec = _max_optional(max_ttft_sec, _first_float(row, ("ttft_sec", "ttft")))
            max_tpot_ms = _max_optional(max_tpot_ms, _first_float(row, ("tpot_ms", "tpot")))
    cache_rate = (tokens_cached / tokens_in) if tokens_in > 0 else None
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_cached": tokens_cached,
        "llm_calls": calls,
        "cost_usd": cost_usd if cost_known_calls else None,
        "cost_known_calls": cost_known_calls,
        "cache_rate": cache_rate,
        "observed_models": [name for name, _ in model_counts.most_common()],
        "max_ttft_sec": max_ttft_sec,
        "max_tpot_ms": max_tpot_ms,
    }


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _iter_recent_jsonl(path: Path, max_events: int = _MAX_RESOURCE_EVENTS) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > _RECENT_JSONL_TAIL_BYTES:
                start = max(0, size - _RECENT_JSONL_TAIL_BYTES)
                f.seek(start)
                raw_lines = f.read().splitlines()
                if start > 0 and raw_lines:
                    raw_lines = raw_lines[1:]
            else:
                raw_lines = f.read().splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for raw_line in raw_lines[-max_events:]:
        line = raw_line.decode("utf-8", errors="ignore")
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _iter_recent_jsonl_many(paths: Iterable[Path], max_events: int = _MAX_RESOURCE_EVENTS) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for obj in _iter_recent_jsonl(Path(path), max_events=max_events):
            try:
                key = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError):
                key = repr(obj)
            if key in seen:
                continue
            seen.add(key)
            out.append(obj)
    out.sort(key=lambda obj: _to_float(obj.get("timestamp") or obj.get("created_at")) or 0.0)
    return out[-max_events:]


def _heartbeat_progress_summary(
    payload: dict[str, Any],
    *,
    event_type: str,
    observed_at: float,
) -> dict[str, Any]:
    if event_type != "progress_heartbeat":
        return {}
    signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
    heartbeat = signals.get("heartbeat") if isinstance(signals.get("heartbeat"), dict) else {}
    structured = payload.get("structured_progress")
    if not isinstance(structured, dict):
        structured = {}
    progress = heartbeat.get("progress") if isinstance(heartbeat.get("progress"), dict) else {}
    progress_unit = ""
    if not progress:
        for key in ("progress", "batch", "step", "epoch", "nb", "items", "rows"):
            candidate = signals.get(key)
            if isinstance(candidate, dict) and ("current" in candidate or "total" in candidate):
                progress = candidate
                progress_unit = key
                break

    current = _to_float(structured.get("current"))
    if current is None:
        current = _to_float(progress.get("current"))
    total = _to_float(structured.get("total"))
    if total is None:
        total = _to_float(progress.get("total"))
    unit = str(
        structured.get("unit")
        or progress.get("unit")
        or heartbeat.get("unit")
        or progress_unit
        or ""
    )
    advanced: bool | None = None
    for raw in (structured.get("advanced"), progress.get("advanced"), heartbeat.get("advanced")):
        if isinstance(raw, bool):
            advanced = raw
            break

    metric_name = ""
    metric_value: float | None = None
    metrics = signals.get("metrics") if isinstance(signals.get("metrics"), dict) else {}
    for key, value in metrics.items():
        numeric = _to_float(value)
        if numeric is not None:
            metric_name = str(key)
            metric_value = numeric
            break

    phase = str(payload.get("phase") or payload.get("current_phase") or heartbeat.get("phase") or "")
    summary: dict[str, Any] = {}
    if current is not None:
        summary["progress_current"] = current
    if total is not None:
        summary["progress_total"] = total
    if unit:
        summary["progress_unit"] = unit
    if advanced is not None:
        summary["progress_advanced"] = advanced
    if metric_name:
        summary["metric_name"] = metric_name
    if metric_value is not None:
        summary["metric_value"] = metric_value
    if phase:
        summary["heartbeat_phase"] = phase
    if summary:
        summary["heartbeat_observed_at"] = observed_at
    return summary


def _job_summary(obj: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    event_type = str(obj.get("event_type") or obj.get("event") or "")
    command_id = str(obj.get("command_id") or payload.get("job_id") or "")
    observed_at = _to_float(obj.get("created_at") or obj.get("timestamp")) or 0.0
    outcome = str(payload.get("execution_outcome") or payload.get("outcome") or "").strip().upper()
    finished = event_type in {"resource_job_finished", "resource_kill_executed"} or (
        event_type == "resource_review_outcome" and outcome in {"KILL", "TERMINATED", "STOPPED"}
    )
    signal = payload.get("resource_review_signal")
    if not isinstance(signal, dict):
        signal = {}
    progress = payload.get("progress_snapshot")
    if not isinstance(progress, dict):
        progress = {}
    if not progress and isinstance(payload.get("filtered_signal"), dict):
        progress = payload.get("filtered_signal") or {}
    if not progress and command_id and ("elapsed_sec" in payload or "metric_history_line_count" in payload):
        progress = payload
    resource = payload.get("resource_snapshot")
    if not isinstance(resource, dict):
        resource = {}
    source_hint = payload.get("source_hint")
    if not isinstance(source_hint, dict):
        source_hint = resource.get("source_hint") if isinstance(resource.get("source_hint"), dict) else {}
    if not signal and command_id and progress:
        signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
        cpu_bucket = str(payload.get("cpu_bucket") or "")
        gpu_bucket = str(payload.get("gpu_bucket") or "")
        process_alive = bool(payload.get("process_alive"))
        monitor_active = event_type == "resource_monitor_heartbeat" and process_alive
        signal = {
            "active_work": monitor_active or not finished,
            "useful_progress": bool(signals or _to_int(progress.get("metric_history_line_count")) or _to_int(payload.get("stdout_lines"))),
            "metric_changed": bool(signals.get("metrics") if isinstance(signals, dict) else False),
            "gpu_bucket": gpu_bucket,
            "cpu_bucket": cpu_bucket,
        }
    liveness = progress.get("process_liveness") or resource.get("process_liveness")
    if not isinstance(liveness, dict):
        liveness = {}
    alive = liveness.get("status") in {"alive", "inconsistent"}
    liveness_known = bool(liveness.get("status")) or "process_alive" in payload
    if "process_alive" in payload:
        alive = bool(payload.get("process_alive"))
    if finished:
        alive = False
    summary = {
        "worker_id": obj.get("worker_id") or payload.get("worker_id") or "?",
        "command_id": command_id,
        "event_type": event_type,
        "observed_at": observed_at,
        "runtime_sec": _to_float(
            progress.get("runtime_sec")
            or progress.get("elapsed_sec")
            or payload.get("runtime_sec")
            or payload.get("elapsed_sec")
        ) or 0.0,
        "active_work": bool(signal.get("active_work")) and not finished,
        "useful_progress": bool(signal.get("useful_progress")),
        "gpu_bucket": str(signal.get("gpu_bucket") or ""),
        "cpu_bucket": str(signal.get("cpu_bucket") or ""),
        "metric_changed": bool(signal.get("metric_changed")),
        "metric_lines": _to_int(progress.get("metric_history_line_count")) or 0,
        "stdout_lines": _to_int(progress.get("stdout_lines") or payload.get("stdout_lines")) or 0,
        "alive": alive,
        "liveness_known": liveness_known,
        "finished": finished,
        "reason": str(signal.get("reason") or payload.get("reason_code") or ""),
        "command_excerpt": str(payload.get("command_excerpt") or resource.get("command_excerpt") or ""),
        "entrypoint": str(payload.get("entrypoint") or resource.get("entrypoint") or ""),
        "resource_class": str(payload.get("resource_class") or resource.get("resource_class") or ""),
        "known_stage": str(
            progress.get("known_stage")
            or progress.get("current_phase")
            or payload.get("known_stage")
            or payload.get("current_phase")
            or ""
        ),
        "progress_signal": str(progress.get("progress_signal") or ""),
        "progress_confidence": str(progress.get("progress_confidence") or ""),
        "output_pattern": str(progress.get("output_pattern") or ""),
        "recoverable_artifact_on_disk": bool(progress.get("recoverable_artifact_on_disk")),
        "stop_cost": str(progress.get("stop_cost") or ""),
        "finish_feasible": progress.get("finish_feasible"),
        "source_hint": source_hint,
    }
    summary.update(_heartbeat_progress_summary(payload, event_type=event_type, observed_at=observed_at))
    return summary


def _select_active_jobs_for_display(jobs: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    """Keep fresh per-worker jobs before older long-running historical records."""
    if not jobs:
        return []

    latest_by_worker: dict[str, dict[str, Any]] = {}
    for job in jobs:
        worker = str(job.get("worker_id") or "?").upper()
        previous = latest_by_worker.get(worker)
        if previous is None or _job_seen_at(job) >= _job_seen_at(previous):
            latest_by_worker[worker] = job

    selected: list[dict[str, Any]] = sorted(
        latest_by_worker.values(),
        key=lambda job: (_job_seen_at(job), float(job.get("runtime_sec") or 0.0)),
        reverse=True,
    )
    selected_ids = {str(job.get("command_id") or id(job)) for job in selected}
    for job in sorted(
        jobs,
        key=lambda item: (_job_seen_at(item), float(item.get("runtime_sec") or 0.0)),
        reverse=True,
    ):
        if len(selected) >= limit:
            break
        job_id = str(job.get("command_id") or id(job))
        if job_id in selected_ids:
            continue
        selected.append(job)
        selected_ids.add(job_id)
    return selected[:limit]


def _job_seen_at(job: dict[str, Any]) -> float:
    return max(
        float(job.get("heartbeat_observed_at") or 0.0),
        float(job.get("observed_at") or 0.0),
    )


def _job_is_active(job: dict[str, Any]) -> bool:
    if job.get("finished"):
        return False
    if job.get("alive"):
        return True
    if job.get("liveness_known"):
        return False
    if str(job.get("event_type") or "") == "progress_heartbeat":
        return bool(job.get("useful_progress")) and float(job.get("runtime_sec") or 0.0) > 0.0
    return float(job.get("runtime_sec") or 0.0) > 0.0


_REAL_EPOCH_MIN_TS = 1_600_000_000.0
_STALE_ACTIVE_JOB_DISPLAY_SEC = 900.0


def _job_is_stale_for_display(job: dict[str, Any]) -> bool:
    seen_at = _job_seen_at(job)
    if seen_at < _REAL_EPOCH_MIN_TS:
        return False
    return _dt.datetime.now().timestamp() - seen_at > _STALE_ACTIVE_JOB_DISPLAY_SEC


_HEARTBEAT_CONTEXT_KEYS = {
    "heartbeat_observed_at",
    "heartbeat_phase",
    "progress_current",
    "progress_total",
    "progress_unit",
    "progress_advanced",
    "metric_name",
    "metric_value",
}
_STALE_HEARTBEAT_CONTEXT_GAP_SEC = 600.0


def _can_inherit_job_context(previous: dict[str, Any], current: dict[str, Any], key: str) -> bool:
    if key not in _HEARTBEAT_CONTEXT_KEYS:
        return True
    previous_seen = _job_seen_at(previous)
    current_seen = _job_seen_at(current)
    if previous_seen <= 0.0 or current_seen <= 0.0:
        return True
    return current_seen - previous_seen <= _STALE_HEARTBEAT_CONTEXT_GAP_SEC


def _merge_job_summary(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return current
    context_keys = {
        "command_excerpt",
        "entrypoint",
        "resource_class",
        "known_stage",
        "progress_signal",
        "progress_confidence",
        "output_pattern",
        "stop_cost",
        "source_hint",
        "heartbeat_observed_at",
        "heartbeat_phase",
        "progress_current",
        "progress_total",
        "progress_unit",
        "progress_advanced",
        "metric_name",
        "metric_value",
    }
    context_only = (
        not current.get("finished")
        and not current.get("alive")
        and float(current.get("runtime_sec") or 0.0) <= 0.0
        and current.get("event_type") == "resource_source_hint_detected"
    )
    if context_only:
        merged = dict(previous)
        for key in context_keys:
            value = current.get(key)
            if value not in (None, "", {}, []):
                merged[key] = value
        # A resume run can reuse command ids.  Context-only events are useful
        # labels, but they must not make old heartbeat/progress evidence look
        # current for a later command with the same id.
        if not any(previous.get(key) not in (None, "", {}, []) for key in _HEARTBEAT_CONTEXT_KEYS):
            merged["observed_at"] = max(
                float(previous.get("observed_at") or 0.0),
                float(current.get("observed_at") or 0.0),
            )
        return merged
    merged = dict(current)
    for key in context_keys:
        if merged.get(key) in (None, "", {}, []) and _can_inherit_job_context(previous, current, key):
            value = previous.get(key)
            if value not in (None, "", {}, []):
                merged[key] = value
    return merged


def _wait_summary(obj: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "worker_id": obj.get("worker_id") or payload.get("worker_id") or "",
        "command_id": obj.get("command_id") or payload.get("job_id") or "",
        "event_type": obj.get("event_type") or obj.get("event") or "",
        "observed_at": _to_float(
            obj.get("created_at") or obj.get("timestamp") or payload.get("created_at")
        ) or 0.0,
        "reason": payload.get("reason_code") or payload.get("reason") or "",
        "resource_class": payload.get("resource_class") or "",
        "queue_position": _to_int(payload.get("queue_position")),
        "queue_len": _to_int(payload.get("queue_len")),
    }


def _worker_states(active_jobs: list[dict[str, Any]], latest_waits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    jobs_by_worker: dict[str, dict[str, Any]] = {}
    for job in active_jobs:
        worker = str(job.get("worker_id") or "?").upper()
        if not worker or worker == "?":
            continue
        previous = jobs_by_worker.get(worker)
        if previous is None or _worker_status_job_key(job) > _worker_status_job_key(previous):
            jobs_by_worker[worker] = job

    by_worker = {
        worker: _worker_state_from_job(job)
        for worker, job in jobs_by_worker.items()
    }
    for worker, wait in latest_waits.items():
        wait_state = _worker_state_from_wait(wait)
        previous = by_worker.get(worker)
        if previous is None or _observed_at(wait_state) >= _observed_at(previous):
            by_worker[worker] = wait_state
    return [by_worker[key] for key in sorted(by_worker)]


def _worker_status_job_key(job: dict[str, Any]) -> tuple[int, int, int, float, float]:
    """Choose stable per-worker status without letting short probes hide long work."""
    runtime = float(job.get("runtime_sec") or 0.0)
    long_bucket = 2 if runtime >= 1800.0 else (1 if runtime >= 900.0 else 0)
    useful_bucket = 1 if _job_has_useful_status_signal(job) else 0
    alive_bucket = 1 if bool(job.get("alive")) else 0
    return (alive_bucket, long_bucket, useful_bucket, _job_seen_at(job), runtime)


def _job_has_useful_status_signal(job: dict[str, Any]) -> bool:
    return bool(
        job.get("useful_progress")
        or job.get("metric_changed")
        or int(job.get("stdout_lines") or 0) > 0
        or int(job.get("metric_lines") or 0) > 0
        or str(job.get("known_stage") or "").strip()
        or str(job.get("progress_signal") or "").strip()
        or str(job.get("heartbeat_phase") or "").strip()
    )


def _worker_state_from_wait(wait: dict[str, Any]) -> dict[str, Any]:
    return {
        "worker_id": str(wait.get("worker_id") or "?").upper(),
        "command_id": wait.get("command_id") or "",
        "kind": "wait",
        "label": "WAIT",
        "runtime_sec": 0.0,
        "observed_at": float(wait.get("observed_at") or 0.0),
        "reason": wait.get("reason") or "",
        "resource_class": wait.get("resource_class") or "",
        "queue_position": wait.get("queue_position"),
        "queue_len": wait.get("queue_len"),
    }


def _worker_state_from_job(job: dict[str, Any]) -> dict[str, Any]:
    kind, label = _worker_job_kind(job)
    return {
        "worker_id": str(job.get("worker_id") or "?").upper(),
        "command_id": job.get("command_id") or "",
        "kind": kind,
        "label": label,
        "runtime_sec": float(job.get("runtime_sec") or 0.0),
        "observed_at": float(job.get("observed_at") or 0.0),
        "gpu_bucket": job.get("gpu_bucket") or "",
        "cpu_bucket": job.get("cpu_bucket") or "",
        "useful_progress": bool(job.get("useful_progress")),
        "metric_changed": bool(job.get("metric_changed")),
        "stdout_lines": int(job.get("stdout_lines") or 0),
        "metric_lines": int(job.get("metric_lines") or 0),
        "resource_class": job.get("resource_class") or "",
        "known_stage": job.get("known_stage") or "",
        "progress_signal": job.get("progress_signal") or "",
        "finish_feasible": job.get("finish_feasible"),
    }


def _worker_job_kind(job: dict[str, Any]) -> tuple[str, str]:
    text = " ".join(
        str(part or "")
        for part in (
            job.get("command_excerpt"),
            job.get("entrypoint"),
            job.get("resource_class"),
            job.get("known_stage"),
            job.get("reason"),
        )
    ).lower()
    source_hint = job.get("source_hint") if isinstance(job.get("source_hint"), dict) else {}
    runtime = float(job.get("runtime_sec") or 0.0)
    inference_tokens = ("predict", "infer", "inference", "submission", "submit", "ensemble", "test.py")
    if any(token in text for token in inference_tokens):
        return "inference", "INF"
    if any(token in text for token in ("valid", "eval", "score", "oof")):
        return "validation", "VAL"
    train_hint = (
        any(token in text for token in ("train", "fit", "trainer", "epoch", "heavy_gpu_train", "heavy_cpu_train"))
        or bool(source_hint.get("command_train_evidence"))
        or bool(source_hint.get("source_train_evidence"))
    )
    if train_hint:
        return ("long_train", "LTRN") if runtime >= 900.0 else ("train", "TRN")
    if runtime >= 1800.0:
        return "long_run", "LRUN"
    return "run", "RUN"


def _observed_at(value: dict[str, Any]) -> float:
    return float(value.get("observed_at") or 0.0)


def _recent_event(obj: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    adv = payload.get("advisory") or payload.get("parsed_response")
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    gate = decision.get("gate") if isinstance(decision.get("gate"), dict) else {}
    preference = payload.get("advisory_preference")
    advisory_status = payload.get("advisory_status")
    advisory_confidence = payload.get("advisory_confidence")
    if isinstance(adv, dict):
        preference = preference or adv.get("preference")
        advisory_status = advisory_status or adv.get("advisory_status") or adv.get("status")
        advisory_confidence = advisory_confidence or adv.get("confidence")
    boundary = payload.get("resource_review_boundary")
    boundary_kind = boundary.get("kind") if isinstance(boundary, dict) else boundary
    gate_allowed = gate.get("allowed")
    strict_gate_result = payload.get("strict_gate_result")
    if not strict_gate_result and gate_allowed is not None:
        strict_gate_result = "allow" if bool(gate_allowed) else "blocked"
    strict_gate_reason = payload.get("strict_gate_reason") or gate.get("blocked_reason") or gate.get("reason") or ""
    outcome = payload.get("execution_outcome") or payload.get("outcome") or decision.get("execution_outcome") or decision.get("canonical_outcome") or ""
    return {
        "event_type": obj.get("event_type") or obj.get("event") or "",
        "worker_id": obj.get("worker_id") or payload.get("worker_id") or "",
        "command_id": obj.get("command_id") or payload.get("job_id") or "",
        "outcome": outcome,
        "reason": payload.get("reason_code") or payload.get("reason") or decision.get("reason_code") or decision.get("reason") or "",
        "boundary": boundary_kind or "",
        "preference": preference or "",
        "advisory_status": advisory_status or "",
        "advisory_confidence": advisory_confidence or "",
        "strict_gate_result": strict_gate_result or "",
        "strict_gate_reason": strict_gate_reason or "",
        "decision_applied": payload.get("decision_applied", ""),
    }


def _latest_stage_by_worker(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        worker = (row.get("worker_id") or "?").upper()
        latest[worker] = _stage_brief(row)
    return latest


def _stage_brief(row: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id") or "",
        "worker_id": row.get("worker_id") or "",
        "metric_value": _to_float(row.get("metric_value")),
        "metric_validity": row.get("metric_validity") or "",
        "candidate_ready": row.get("candidate_ready") or "",
        "selection_eligible": row.get("selection_eligible") or "",
        "evaluator_backend": row.get("evaluator_backend") or "",
        "evaluator_status": row.get("evaluator_status") or "",
        "artifact_sha": row.get("artifact_sha") or row.get("submission_sha") or "",
    }


def _summary_lower_is_better(
    rows: list[dict[str, str]],
    best_raw: dict[str, Any],
    best_valid: dict[str, Any],
) -> bool:
    for score in (best_valid, best_raw):
        if isinstance(score.get("lower_is_better"), bool):
            return bool(score["lower_is_better"])
    return infer_stage_rows_lower_is_better(rows)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _first_int(row: dict[str, str], keys: tuple[str, ...]) -> int:
    for key in keys:
        val = _to_int(row.get(key))
        if val is not None:
            return val
    return 0


def _first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        val = _to_float(row.get(key))
        if val is not None:
            return val
    return None


def _max_optional(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    if current is None:
        return value
    return max(current, value)


def _to_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _newest_mtime(task_root: Path, rows: list[dict[str, str]]) -> float:
    candidates = [
        task_root / "task_logs" / "lhr_stage_performance.csv",
        task_root / "task_logs" / "lhr_events.jsonl",
        task_root / "task_logs" / "resource" / "resource_events.jsonl",
    ]
    candidates.extend(_time_trace_paths(task_root))
    mtimes = []
    for path in candidates:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    return max(mtimes) if mtimes else _dt.datetime.now().timestamp()


def _format_timestamp(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")
