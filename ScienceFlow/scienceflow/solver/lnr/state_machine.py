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

import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scienceflow.solver.lnr.stage.score_summary import (
    build_score_summary,
    read_stage_performance_records,
    score_record_from_stage_payload,
)


LHR_EVENTS_JSONL = "lhr_events.jsonl"
LHR_STATE_JSON = "lhr_state.json"
_RESOURCE_EVENT_NAMES = {
    "resource_source_hint_detected",
    "resource_request_created",
    "resource_gpu_queue_wait_started",
    "resource_gpu_queue_heartbeat",
    "resource_gpu_queue_timeout",
    "resource_policy_gate",
    "resource_planner_guard",
    "resource_admission_deferred",
    "resource_admission_llm_decision",
    "resource_gpu_lease_acquired",
    "resource_gpu_lease_released",
    "resource_gpu_runtime_pressure",
    "resource_gpu_stale_lease_reaped",
    "resource_gpu_util_sampled",
    "progress_heartbeat",
    "agent_backoff_wait_started",
    "agent_backoff_wait_finished",
    "post_feedback_action",
}


def _now_ts() -> float:
    return time.time()


def _utc_iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts if ts is not None else _now_ts())))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise


def _atomic_write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(_json_safe(event), ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
    except (OSError, json.JSONDecodeError):
        return out
    return out


def _event_dedupe_key(event: dict[str, Any]) -> str:
    return json.dumps(_json_safe(event), ensure_ascii=False, sort_keys=True)


def _string_list(value: Any, *, limit: int = 16) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in value:
        text = str(raw or "").strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _resource_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    source_hint = payload.get("source_hint") if isinstance(payload.get("source_hint"), dict) else {}
    sample = payload.get("sample") if isinstance(payload.get("sample"), dict) else {}
    gpu_util = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
    placement = payload.get("placement") if isinstance(payload.get("placement"), dict) else {}
    placement_violations = placement.get("violations") if isinstance(placement.get("violations"), list) else []
    summary: dict[str, Any] = {
        "event": str(event.get("event") or ""),
        "worker_id": str(event.get("worker_id") or "W00"),
        "status": str(event.get("status") or ""),
        "timestamp": float(event.get("timestamp") or 0.0),
        "timestamp_utc": str(event.get("timestamp_utc") or ""),
        "job_id": str(payload.get("job_id") or ""),
        "command_digest": str(payload.get("command_digest") or ""),
        "resource_class": str(payload.get("resource_class") or ""),
        "gpu_ids": _string_list(payload.get("gpu_ids")),
        "cpu_set": str(payload.get("cpu_set") or ""),
        "reason": str(payload.get("reason") or payload.get("release_reason") or ""),
        "feedback_status": str(payload.get("status") or ""),
        "admission_action": str(payload.get("admission_action") or ""),
        "scope": str(payload.get("scope") or ""),
        "resource_mode": str(payload.get("resource_mode") or ""),
        "blocked_class": str(payload.get("blocked_class") or ""),
        "allowed_classes": _string_list(payload.get("allowed_classes")),
        "cooldown_sec": payload.get("cooldown_sec"),
        "eta_next_train_sec": payload.get("eta_next_train_sec"),
        "eta_confidence": str(payload.get("eta_confidence") or ""),
        "eta_source": str(payload.get("eta_source") or ""),
        "runtime_history_count": payload.get("runtime_history_count"),
        "runtime_avg_sec": payload.get("runtime_avg_sec"),
        "runtime_p80_sec": payload.get("runtime_p80_sec"),
        "active_blocker_count": payload.get("active_blocker_count"),
        "active_blocker_age_sec_max": payload.get("active_blocker_age_sec_max"),
        "admission_priority_score": payload.get("admission_priority_score"),
        "expected_value_score": payload.get("expected_value_score"),
        "near_submission_score": payload.get("near_submission_score"),
        "long_runtime_penalty": payload.get("long_runtime_penalty"),
        "value_hint": payload.get("value_hint") if isinstance(payload.get("value_hint"), dict) else {},
        "resource_context_version": payload.get("resource_context_version"),
        "resource_pressure_generation": payload.get("resource_pressure_generation"),
        "current_pressure_generation": payload.get("current_pressure_generation"),
        "pressure_generation": payload.get("pressure_generation"),
        "queue_position": payload.get("queue_position"),
        "queue_len": payload.get("queue_len"),
        "top_waiter_job_id": str(payload.get("top_waiter_job_id") or ""),
        "duplicate_digest_count": payload.get("duplicate_digest_count"),
        "phase": str(payload.get("phase") or ""),
        "signals": payload.get("signals") if isinstance(payload.get("signals"), dict) else {},
        "resource_efficiency": payload.get("resource_efficiency") if isinstance(payload.get("resource_efficiency"), dict) else {},
        "last_progress": payload.get("last_progress") if isinstance(payload.get("last_progress"), dict) else {},
        "last_artifact_progress": payload.get("last_artifact_progress") if isinstance(payload.get("last_artifact_progress"), dict) else {},
        "elapsed_sec": payload.get("elapsed_sec"),
        "planned_sleep_sec": payload.get("planned_sleep_sec"),
        "wake_reason": payload.get("wake_reason"),
        "wait_id": str(payload.get("wait_id") or ""),
        "post_feedback_action": str(payload.get("action") or ""),
        "violates_feedback": payload.get("violates_feedback"),
        "source_reason": str(payload.get("source_reason") or ""),
        "source_resource_mode": str(payload.get("source_resource_mode") or ""),
        "current_resource_class": str(payload.get("current_resource_class") or ""),
        "elapsed_since_feedback_sec": payload.get("elapsed_since_feedback_sec"),
        "gpu_request": payload.get("gpu_request") or payload.get("requested_gpu_count"),
        "hint_labels": _string_list(payload.get("hint_labels") or source_hint.get("hint_labels")),
        "entrypoints": _string_list(payload.get("entrypoints") or source_hint.get("entrypoints"), limit=8),
        "actual_gpu_ids": sorted({
            str(row.get("gpu_id") or "")
            for row in placement_violations
            if isinstance(row, dict) and str(row.get("gpu_id") or "").strip()
        }),
        "allowed_gpu_ids": _string_list(placement.get("allowed_gpu_ids") or placement.get("allowed_physical_gpus")),
        "cleanup_killed_count": placement.get("killed_count"),
        "cleanup_skipped_count": placement.get("skipped_count"),
    }
    if gpu_util:
        summary["gpu_util"] = [
            {
                "gpu_id": str(row.get("gpu_id") or row.get("index") or ""),
                "utilization_gpu_pct": row.get("utilization_gpu_pct"),
                "memory_used_mb": row.get("memory_used_mb"),
                "memory_total_mb": row.get("memory_total_mb"),
            }
            for row in gpu_util[:8]
            if isinstance(row, dict)
        ]
        summary["sample_available"] = bool(sample.get("available"))
    return {k: v for k, v in summary.items() if v not in ("", [], None)}


def _float_or_zero(value: Any) -> float:
    try:
        out = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out else 0.0


def _percentile(values: list[float], q: float) -> float:
    clean = sorted(v for v in (_float_or_zero(x) for x in values) if v >= 0.0)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * min(1.0, max(0.0, float(q)))
    lo = int(pos)
    hi = min(lo + 1, len(clean) - 1)
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def _queue_wait_stats(durations: list[float], *, waits: int, timeouts: int) -> dict[str, Any]:
    clean = [max(0.0, _float_or_zero(x)) for x in durations]
    wait_count = max(0, int(waits or 0))
    timeout_count = max(0, int(timeouts or 0))
    return {
        "resource_gpu_queue_waits_completed": len(clean),
        "resource_gpu_queue_wait_sec_total": round(sum(clean), 3),
        "resource_gpu_queue_wait_sec_max": round(max(clean, default=0.0), 3),
        "resource_gpu_queue_wait_sec_p50": round(_percentile(clean, 0.50), 3),
        "resource_gpu_queue_wait_sec_p95": round(_percentile(clean, 0.95), 3),
        "resource_gpu_queue_timeout_rate": round(timeout_count / wait_count, 3) if wait_count else 0.0,
    }


def _record_queue_wait_duration(
    values_by_worker: dict[str, list[float]],
    worker_id: str,
    elapsed_sec: float,
) -> None:
    elapsed = max(0.0, _float_or_zero(elapsed_sec))
    values_by_worker.setdefault(str(worker_id or "W00"), []).append(elapsed)


def _append_gpu_util_rows(rows_by_gpu: dict[str, list[dict[str, Any]]], event: dict[str, Any]) -> None:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    sample = payload.get("sample") if isinstance(payload.get("sample"), dict) else {}
    rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
    ts = _float_or_zero(event.get("timestamp"))
    for row in rows:
        if not isinstance(row, dict):
            continue
        gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
        if not gpu_id:
            continue
        used_gb = _float_or_zero(row.get("memory_used_mb")) / 1024.0
        total_gb = _float_or_zero(row.get("memory_total_mb")) / 1024.0
        rows_by_gpu.setdefault(gpu_id, []).append({
            "timestamp": ts,
            "utilization_gpu_pct": _float_or_zero(row.get("utilization_gpu_pct")),
            "memory_used_gb": used_gb,
            "memory_total_gb": total_gb,
            "free_mem_gb": max(0.0, total_gb - used_gb) if total_gb > 0 else 0.0,
        })


def _gpu_util_recent_summary(rows_by_gpu: dict[str, list[dict[str, Any]]], *, window_sec: float = 60.0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    window = max(1.0, float(window_sec or 60.0))
    for gpu_id in sorted(rows_by_gpu):
        rows = [r for r in rows_by_gpu.get(gpu_id, []) if isinstance(r, dict)]
        if not rows:
            continue
        latest_ts = max(_float_or_zero(r.get("timestamp")) for r in rows)
        cutoff = latest_ts - window
        recent = [r for r in rows if _float_or_zero(r.get("timestamp")) >= cutoff]
        if not recent:
            recent = [max(rows, key=lambda r: _float_or_zero(r.get("timestamp")))]
        util = [_float_or_zero(r.get("utilization_gpu_pct")) for r in recent]
        free = [_float_or_zero(r.get("free_mem_gb")) for r in recent]
        used = [_float_or_zero(r.get("memory_used_gb")) for r in recent]
        latest = max(recent, key=lambda r: _float_or_zero(r.get("timestamp")))
        out.append({
            "gpu_id": gpu_id,
            "sample_count": len(rows),
            "recent_sample_count": len(recent),
            "window_sec": round(window, 3),
            "util_p50_60s": round(_percentile(util, 0.50), 3),
            "util_p95_60s": round(_percentile(util, 0.95), 3),
            "free_mem_gb_min_60s": round(min(free, default=0.0), 3),
            "free_mem_gb_p05_60s": round(_percentile(free, 0.05), 3),
            "memory_used_gb_p95_60s": round(_percentile(used, 0.95), 3),
            "memory_total_gb_last": round(_float_or_zero(latest.get("memory_total_gb")), 3),
            "last_sample_at": _float_or_zero(latest.get("timestamp")),
            "last_sample_at_utc": _utc_iso(_float_or_zero(latest.get("timestamp"))) if latest.get("timestamp") else "",
        })
    return out


def _sorted_resource_pressure_states(states: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(states[k]) for k in sorted(states)]


def _resource_pressure_entries(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    pressure = payload.get("pressure") if isinstance(payload.get("pressure"), dict) else {}
    entries: dict[str, dict[str, Any]] = {}
    raw_gpus = pressure.get("gpus")
    if isinstance(raw_gpus, dict):
        for raw_id, raw_entry in raw_gpus.items():
            entry = dict(raw_entry if isinstance(raw_entry, dict) else {})
            gpu_id = str(entry.get("gpu_id") or raw_id or "").strip()
            if gpu_id:
                entries[gpu_id] = entry
    ids = _string_list(pressure.get("gpu_ids") or payload.get("red_gpu_ids"))
    for gpu_id in ids:
        entries.setdefault(gpu_id, {})
    if not entries:
        return {}
    ts = _float_or_zero(event.get("timestamp"))
    et = str(event.get("event") or "")
    reason = str(payload.get("reason") or pressure.get("reason") or "")
    now = _now_ts()
    for gpu_id, entry in entries.items():
        entry["gpu_id"] = gpu_id
        entry.setdefault("mode", pressure.get("mode") or ("RED" if et == "resource_gpu_queue_timeout" else ""))
        if pressure.get("generation") is not None:
            entry.setdefault("generation", pressure.get("generation"))
        if pressure.get("cooldown_sec") is not None:
            entry.setdefault("cooldown_sec", pressure.get("cooldown_sec"))
        if pressure.get("red_until") is not None:
            entry.setdefault("red_until", pressure.get("red_until"))
        entry.setdefault("last_event", et)
        entry.setdefault("last_reason", reason)
        entry.setdefault("last_worker_id", str(event.get("worker_id") or "W00"))
        entry.setdefault("last_job_id", str(payload.get("job_id") or ""))
        entry.setdefault("last_resource_class", str(payload.get("resource_class") or payload.get("blocked_class") or ""))
        entry["updated_at"] = _float_or_zero(entry.get("updated_at")) or ts
        red_until = _float_or_zero(entry.get("red_until"))
        yellow_until = _float_or_zero(entry.get("yellow_until"))
        if red_until or yellow_until:
            if red_until > now:
                entry["mode"] = "RED"
            elif yellow_until > now:
                entry["mode"] = "YELLOW"
            else:
                entry["mode"] = "GREEN"
            entry["cooldown_remaining_sec"] = max(0.0, red_until - now)
            entry["yellow_remaining_sec"] = max(0.0, yellow_until - now)
    return entries


def _apply_resource_event_to_views(
    event: dict[str, Any],
    *,
    active_leases: dict[str, dict[str, Any]],
    pending_jobs: dict[str, dict[str, Any]],
    last_events: list[dict[str, Any]],
    pressure_states: dict[str, dict[str, Any]] | None = None,
    max_last: int = 12,
) -> None:
    et = str(event.get("event") or "")
    if et not in _RESOURCE_EVENT_NAMES:
        return
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    job_id = str(payload.get("job_id") or "")
    summary = _resource_event_summary(event)
    last_events.append(summary)
    if len(last_events) > max(1, int(max_last or 12)):
        del last_events[: len(last_events) - max(1, int(max_last or 12))]
    if pressure_states is not None:
        for gpu_id, entry in _resource_pressure_entries(event).items():
            pressure_states[gpu_id] = entry
    if not job_id:
        return
    if et == "resource_gpu_queue_wait_started":
        pending_jobs[job_id] = summary
    elif et == "resource_gpu_lease_acquired":
        active_leases[job_id] = summary
        pending_jobs.pop(job_id, None)
    elif et in {"resource_gpu_queue_timeout", "resource_gpu_lease_released", "resource_gpu_runtime_pressure", "resource_admission_deferred"}:
        pending_jobs.pop(job_id, None)
        active_leases.pop(job_id, None)


def build_lhr_state_from_events(
    events: list[dict[str, Any]],
    *,
    worker_count: int = 1,
    run_status: str = "",
    ledger_filename: str = ".run_results.md",
) -> dict[str, Any]:
    worker_stage_ids: dict[str, set[str]] = {}
    worker_status: dict[str, str] = {}
    worker_last: dict[str, float] = {}
    worker_estra_decision: dict[str, int] = {}
    worker_estra_invalid: dict[str, int] = {}
    worker_estra_continue: dict[str, int] = {}
    worker_estra_redirect: dict[str, int] = {}
    worker_estra_switch_decision: dict[str, int] = {}
    worker_estra_current_continue: dict[str, int] = {}
    worker_estra_current_redirect: dict[str, int] = {}
    worker_estra_stage_continue: dict[str, int] = {}
    worker_estra_stage_redirect: dict[str, int] = {}
    worker_estra_switch: dict[str, int] = {}
    worker_estra_compact: dict[str, int] = {}
    worker_resource: dict[str, int] = {}
    worker_guard: dict[str, int] = {}
    worker_cleanup_heartbeat: dict[str, int] = {}
    worker_monitor_agent: dict[str, int] = {}
    worker_resource_request: dict[str, int] = {}
    worker_resource_source_hint: dict[str, int] = {}
    worker_gpu_queue_wait: dict[str, int] = {}
    worker_gpu_queue_timeout: dict[str, int] = {}
    worker_gpu_queue_wait_durations: dict[str, list[float]] = {}
    queue_wait_started_at: dict[str, tuple[str, float]] = {}
    worker_policy_gate: dict[str, int] = {}
    worker_planner_guard: dict[str, int] = {}
    worker_admission_deferred: dict[str, int] = {}
    worker_gpu_lease_acquired: dict[str, int] = {}
    worker_gpu_lease_released: dict[str, int] = {}
    worker_gpu_util_sample: dict[str, int] = {}
    worker_progress_heartbeat: dict[str, int] = {}
    progress_last: dict[str, Any] = {}
    worker_agent_backoff_wait: dict[str, int] = {}
    worker_agent_backoff_wait_sec_total: dict[str, float] = {}
    worker_agent_backoff_wait_sec_max: dict[str, float] = {}
    worker_post_feedback_action: dict[str, int] = {}
    post_feedback_action_counts: dict[str, int] = {}
    post_feedback_violations = 0
    agent_backoff_active: dict[str, dict[str, Any]] = {}
    agent_backoff_wake_reasons: dict[str, int] = {}
    resource_active_leases: dict[str, dict[str, Any]] = {}
    resource_pending_gpu_jobs: dict[str, dict[str, Any]] = {}
    resource_gpu_pressure_states: dict[str, dict[str, Any]] = {}
    resource_gpu_util_rows: dict[str, list[dict[str, Any]]] = {}
    resource_last_events: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None
    global_best_as_of = 0.0
    multi_worker_started_at = 0.0
    worker_started_at = 0.0

    def is_better(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
        try:
            metric = float(candidate.get("metric_value"))
        except (TypeError, ValueError):
            return False
        if metric != metric:
            return False
        if current is None:
            return True
        try:
            cur = float(current.get("metric_value"))
        except (TypeError, ValueError):
            return True
        lower = bool(candidate.get("lower_is_better") is not False)
        return metric < cur if lower else metric > cur

    for event in sorted(events, key=lambda e: float(e.get("timestamp") or 0.0)):
        et = str(event.get("event") or "")
        worker_id = str(event.get("worker_id") or "W00")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        ts = float(event.get("timestamp") or 0.0)
        worker_last[worker_id] = max(worker_last.get(worker_id, 0.0), ts)
        _apply_resource_event_to_views(
            event,
            active_leases=resource_active_leases,
            pending_jobs=resource_pending_gpu_jobs,
            last_events=resource_last_events,
            pressure_states=resource_gpu_pressure_states,
        )
        payload_event = str(payload.get("event") or "")
        if et == "multi_worker_start" or payload_event == "multi_worker_start":
            multi_worker_started_at = max(multi_worker_started_at, ts)
        elif et == "worker_start" or payload_event == "worker_start":
            worker_started_at = max(worker_started_at, ts)
        if et == "worker_status":
            worker_status[worker_id] = str(event.get("status") or payload.get("status") or "")
        if et == "stage_captured":
            sid = str(payload.get("stage_id") or "")
            if sid:
                worker_stage_ids.setdefault(worker_id, set()).add(sid)
            record = score_record_from_stage_payload(payload, worker_id=worker_id)
            if record is not None and record.valid_comparable:
                candidate = {
                    "candidate_id": record.candidate_id or (f"{worker_id}:{sid}" if sid else worker_id),
                    "worker_id": record.worker_id or worker_id,
                    "stage_id": record.stage_id or sid,
                    "snapshot_id": payload.get("snapshot_id") or "",
                    "metric_value": record.value,
                    "metric_name": record.metric_name,
                    "lower_is_better": record.lower_is_better,
                    "validation_ok": record.validation_ok,
                    "metric_validity": record.metric_validity,
                    "validity": record.validity,
                    "evaluator_backend": record.evaluator_backend,
                    "evaluator_status": record.evaluator_status,
                }
                if is_better(candidate, global_best):
                    global_best = candidate
                    global_best_as_of = ts
        elif et == "estra_stage_switched":
            worker_estra_switch[worker_id] = worker_estra_switch.get(worker_id, 0) + 1
        elif et == "estra_decision":
            worker_estra_decision[worker_id] = worker_estra_decision.get(worker_id, 0) + 1
            action = str(payload.get("action") or "")
            startpoint = str(payload.get("startpoint") or ("previous_stage" if action == "switch_stage" else "current_workspace"))
            intent = str(payload.get("intent") or ("redirect" if action == "keep_but_redirect" else "continue"))
            valid_estra_action = action in {"keep_current", "keep_but_redirect", "switch_stage"}
            if action == "keep_current":
                worker_estra_continue[worker_id] = worker_estra_continue.get(worker_id, 0) + 1
            elif action == "keep_but_redirect":
                worker_estra_redirect[worker_id] = worker_estra_redirect.get(worker_id, 0) + 1
            elif action == "switch_stage":
                worker_estra_switch_decision[worker_id] = worker_estra_switch_decision.get(worker_id, 0) + 1
            elif action:
                worker_estra_invalid[worker_id] = worker_estra_invalid.get(worker_id, 0) + 1
            if valid_estra_action:
                if startpoint == "previous_stage" and intent == "redirect":
                    worker_estra_stage_redirect[worker_id] = worker_estra_stage_redirect.get(worker_id, 0) + 1
                elif startpoint == "previous_stage":
                    worker_estra_stage_continue[worker_id] = worker_estra_stage_continue.get(worker_id, 0) + 1
                elif intent == "redirect":
                    worker_estra_current_redirect[worker_id] = worker_estra_current_redirect.get(worker_id, 0) + 1
                else:
                    worker_estra_current_continue[worker_id] = worker_estra_current_continue.get(worker_id, 0) + 1
        elif et == "estra_keep_current_compacted":
            worker_estra_compact[worker_id] = worker_estra_compact.get(worker_id, 0) + 1
        elif et == "resource_job_started":
            worker_resource[worker_id] = worker_resource.get(worker_id, 0) + 1
        elif et == "resource_guard_action":
            worker_guard[worker_id] = worker_guard.get(worker_id, 0) + 1
        elif et == "resource_cleanup_heartbeat":
            worker_cleanup_heartbeat[worker_id] = worker_cleanup_heartbeat.get(worker_id, 0) + 1
        elif et == "resource_monitor_agent_shadow":
            worker_monitor_agent[worker_id] = worker_monitor_agent.get(worker_id, 0) + 1
        elif et == "resource_request_created":
            worker_resource_request[worker_id] = worker_resource_request.get(worker_id, 0) + 1
        elif et == "resource_source_hint_detected":
            worker_resource_source_hint[worker_id] = worker_resource_source_hint.get(worker_id, 0) + 1
        elif et == "resource_gpu_queue_wait_started":
            worker_gpu_queue_wait[worker_id] = worker_gpu_queue_wait.get(worker_id, 0) + 1
            job_id = str(payload.get("job_id") or event.get("task_id") or "")
            if job_id:
                queue_wait_started_at[job_id] = (worker_id, ts)
        elif et == "resource_gpu_queue_timeout":
            worker_gpu_queue_timeout[worker_id] = worker_gpu_queue_timeout.get(worker_id, 0) + 1
            job_id = str(payload.get("job_id") or event.get("task_id") or "")
            started = queue_wait_started_at.pop(job_id, None) if job_id else None
            elapsed = _float_or_zero(payload.get("elapsed_sec"))
            if elapsed <= 0.0 and started is not None:
                elapsed = max(0.0, ts - _float_or_zero(started[1]))
            _record_queue_wait_duration(worker_gpu_queue_wait_durations, worker_id, elapsed)
        elif et == "resource_policy_gate":
            worker_policy_gate[worker_id] = worker_policy_gate.get(worker_id, 0) + 1
        elif et == "resource_planner_guard":
            worker_planner_guard[worker_id] = worker_planner_guard.get(worker_id, 0) + 1
        elif et == "resource_admission_deferred":
            worker_admission_deferred[worker_id] = worker_admission_deferred.get(worker_id, 0) + 1
        elif et == "resource_gpu_lease_acquired":
            worker_gpu_lease_acquired[worker_id] = worker_gpu_lease_acquired.get(worker_id, 0) + 1
            job_id = str(payload.get("job_id") or event.get("task_id") or "")
            started = queue_wait_started_at.pop(job_id, None) if job_id else None
            if started is not None:
                start_worker, start_ts = started
                _record_queue_wait_duration(worker_gpu_queue_wait_durations, start_worker, ts - _float_or_zero(start_ts))
        elif et == "resource_gpu_lease_released":
            worker_gpu_lease_released[worker_id] = worker_gpu_lease_released.get(worker_id, 0) + 1
        elif et == "resource_gpu_util_sampled":
            worker_gpu_util_sample[worker_id] = worker_gpu_util_sample.get(worker_id, 0) + 1
            _append_gpu_util_rows(resource_gpu_util_rows, event)
        elif et == "progress_heartbeat":
            worker_progress_heartbeat[worker_id] = worker_progress_heartbeat.get(worker_id, 0) + 1
            progress_last = _resource_event_summary(event)
        elif et == "agent_backoff_wait_started":
            worker_agent_backoff_wait[worker_id] = worker_agent_backoff_wait.get(worker_id, 0) + 1
            wait_id = str(payload.get("wait_id") or event.get("task_id") or "")
            if wait_id:
                agent_backoff_active[wait_id] = _resource_event_summary(event)
        elif et == "agent_backoff_wait_finished":
            elapsed = _float_or_zero(payload.get("elapsed_sec"))
            worker_agent_backoff_wait_sec_total[worker_id] = worker_agent_backoff_wait_sec_total.get(worker_id, 0.0) + elapsed
            worker_agent_backoff_wait_sec_max[worker_id] = max(worker_agent_backoff_wait_sec_max.get(worker_id, 0.0), elapsed)
            wake = str(payload.get("wake_reason") or "unknown")
            agent_backoff_wake_reasons[wake] = agent_backoff_wake_reasons.get(wake, 0) + 1
            wait_id = str(payload.get("wait_id") or event.get("task_id") or "")
            if wait_id:
                agent_backoff_active.pop(wait_id, None)
        elif et == "post_feedback_action":
            worker_post_feedback_action[worker_id] = worker_post_feedback_action.get(worker_id, 0) + 1
            action = str(payload.get("action") or "unknown")
            post_feedback_action_counts[action] = post_feedback_action_counts.get(action, 0) + 1
            if bool(payload.get("violates_feedback")):
                post_feedback_violations += 1
        elif et == "estra_invalid":
            worker_estra_invalid[worker_id] = worker_estra_invalid.get(worker_id, 0) + 1

    worker_ids = sorted(
        set(worker_stage_ids)
        | set(worker_status)
        | set(worker_last)
        | set(worker_resource)
        | set(worker_guard)
        | set(worker_cleanup_heartbeat)
        | set(worker_monitor_agent)
        | set(worker_resource_request)
        | set(worker_resource_source_hint)
        | set(worker_gpu_queue_wait)
        | set(worker_gpu_queue_timeout)
        | set(worker_policy_gate)
        | set(worker_planner_guard)
        | set(worker_admission_deferred)
        | set(worker_gpu_lease_acquired)
        | set(worker_gpu_lease_released)
        | set(worker_gpu_util_sample)
        | set(worker_progress_heartbeat)
        | set(worker_agent_backoff_wait)
        | set(worker_agent_backoff_wait_sec_total)
        | set(worker_post_feedback_action)
        | {"W00"}
    )
    workers: dict[str, Any] = {}
    for idx, worker_id in enumerate(worker_ids):
        last = worker_last.get(worker_id, 0.0)
        workers[worker_id] = {
            "worker_id": worker_id,
            "worker_index": idx,
            "status": worker_status.get(worker_id) or "unknown",
            "stage_count": len(worker_stage_ids.get(worker_id, set())),
            "estra_decisions": worker_estra_decision.get(worker_id, 0),
            "estra_invalid_decisions": worker_estra_invalid.get(worker_id, 0),
            "estra_continue_count": worker_estra_continue.get(worker_id, 0),
            "estra_redirect_count": worker_estra_redirect.get(worker_id, 0),
            "estra_switch_count": worker_estra_switch_decision.get(worker_id, 0),
            "estra_current_continue_count": worker_estra_current_continue.get(worker_id, 0),
            "estra_current_redirect_count": worker_estra_current_redirect.get(worker_id, 0),
            "estra_stage_continue_count": worker_estra_stage_continue.get(worker_id, 0),
            "estra_stage_redirect_count": worker_estra_stage_redirect.get(worker_id, 0),
            "estra_switch_stage_count": worker_estra_switch.get(worker_id, 0),
            "estra_compact_count": worker_estra_compact.get(worker_id, 0),
            "resource_jobs": worker_resource.get(worker_id, 0),
            "resource_guard_actions": worker_guard.get(worker_id, 0),
            "resource_cleanup_heartbeats": worker_cleanup_heartbeat.get(worker_id, 0),
            "resource_monitor_agent_events": worker_monitor_agent.get(worker_id, 0),
            "resource_requests": worker_resource_request.get(worker_id, 0),
            "resource_source_hints": worker_resource_source_hint.get(worker_id, 0),
            "resource_gpu_queue_waits": worker_gpu_queue_wait.get(worker_id, 0),
            "resource_gpu_queue_timeouts": worker_gpu_queue_timeout.get(worker_id, 0),
            **_queue_wait_stats(
                worker_gpu_queue_wait_durations.get(worker_id, []),
                waits=worker_gpu_queue_wait.get(worker_id, 0),
                timeouts=worker_gpu_queue_timeout.get(worker_id, 0),
            ),
            "resource_policy_gates": worker_policy_gate.get(worker_id, 0),
            "resource_planner_guards": worker_planner_guard.get(worker_id, 0),
            "resource_admission_deferred": worker_admission_deferred.get(worker_id, 0),
            "resource_gpu_lease_acquired": worker_gpu_lease_acquired.get(worker_id, 0),
            "resource_gpu_lease_released": worker_gpu_lease_released.get(worker_id, 0),
            "resource_gpu_util_samples": worker_gpu_util_sample.get(worker_id, 0),
            "progress_heartbeats": worker_progress_heartbeat.get(worker_id, 0),
            "agent_backoff_waits": worker_agent_backoff_wait.get(worker_id, 0),
            "post_feedback_actions": worker_post_feedback_action.get(worker_id, 0),
            "agent_backoff_wait_sec_total": round(worker_agent_backoff_wait_sec_total.get(worker_id, 0.0), 3),
            "agent_backoff_wait_sec_max": round(worker_agent_backoff_wait_sec_max.get(worker_id, 0.0), 3),
            "last_event_at": last,
            "last_event_at_utc": _utc_iso(last) if last else "",
        }
    now = _now_ts()
    started_at = multi_worker_started_at or worker_started_at or 0.0
    status = str(run_status or "").strip()
    if not status:
        statuses = {str(w.get("status") or "") for w in workers.values()}
        status = "finished" if statuses and statuses <= {"finished", "success"} else "running"
    return {
        "schema_version": 1,
        "solver": "lnr",
        "run_status": status,
        "started_at": started_at,
        "started_at_utc": _utc_iso(started_at) if started_at else "",
        "worker_count": max(1, int(worker_count or len(workers) or 1)),
        "workers": workers,
        "global_best": global_best or {},
        "global_best_as_of": global_best_as_of,
        "global_best_as_of_utc": _utc_iso(global_best_as_of) if global_best_as_of else "",
        "stage_count": sum(len(v) for v in worker_stage_ids.values()),
        "estra_decisions": sum(worker_estra_decision.values()),
        "estra_invalid_decisions": sum(worker_estra_invalid.values()),
        "estra_continue_count": sum(worker_estra_continue.values()),
        "estra_redirect_count": sum(worker_estra_redirect.values()),
        "estra_switch_count": sum(worker_estra_switch_decision.values()),
        "estra_current_continue_count": sum(worker_estra_current_continue.values()),
        "estra_current_redirect_count": sum(worker_estra_current_redirect.values()),
        "estra_stage_continue_count": sum(worker_estra_stage_continue.values()),
        "estra_stage_redirect_count": sum(worker_estra_stage_redirect.values()),
        "estra_start_current_count": sum(worker_estra_current_continue.values()) + sum(worker_estra_current_redirect.values()),
        "estra_start_stage_count": sum(worker_estra_stage_continue.values()) + sum(worker_estra_stage_redirect.values()),
        "estra_intent_continue_count": sum(worker_estra_current_continue.values()) + sum(worker_estra_stage_continue.values()),
        "estra_intent_redirect_count": sum(worker_estra_current_redirect.values()) + sum(worker_estra_stage_redirect.values()),
        "estra_switch_stage_count": sum(worker_estra_switch.values()),
        "estra_compact_count": sum(worker_estra_compact.values()),
        "resource_jobs": sum(worker_resource.values()),
        "resource_guard_actions": sum(worker_guard.values()),
        "resource_cleanup_heartbeats": sum(worker_cleanup_heartbeat.values()),
        "resource_monitor_agent_events": sum(worker_monitor_agent.values()),
        "resource_requests": sum(worker_resource_request.values()),
        "resource_source_hints": sum(worker_resource_source_hint.values()),
        "resource_gpu_queue_waits": sum(worker_gpu_queue_wait.values()),
        "resource_gpu_queue_timeouts": sum(worker_gpu_queue_timeout.values()),
        **_queue_wait_stats(
            [duration for durations in worker_gpu_queue_wait_durations.values() for duration in durations],
            waits=sum(worker_gpu_queue_wait.values()),
            timeouts=sum(worker_gpu_queue_timeout.values()),
        ),
        "resource_policy_gates": sum(worker_policy_gate.values()),
        "resource_planner_guards": sum(worker_planner_guard.values()),
        "resource_admission_deferred": sum(worker_admission_deferred.values()),
        "resource_gpu_lease_acquired": sum(worker_gpu_lease_acquired.values()),
        "resource_gpu_lease_released": sum(worker_gpu_lease_released.values()),
        "resource_gpu_util_samples": sum(worker_gpu_util_sample.values()),
        "progress_heartbeats": sum(worker_progress_heartbeat.values()),
        "progress_last": progress_last,
        "agent_backoff_waits": sum(worker_agent_backoff_wait.values()),
        "post_feedback_actions": sum(worker_post_feedback_action.values()),
        "post_feedback_action_counts": dict(sorted(post_feedback_action_counts.items())),
        "post_feedback_violations": post_feedback_violations,
        "agent_backoff_wait_sec_total": round(sum(worker_agent_backoff_wait_sec_total.values()), 3),
        "agent_backoff_wait_sec_max": round(max(worker_agent_backoff_wait_sec_max.values(), default=0.0), 3),
        "agent_backoff_wait_active": list(agent_backoff_active.values()),
        "agent_backoff_wake_reasons": dict(sorted(agent_backoff_wake_reasons.items())),
        "resource_active_leases": list(resource_active_leases.values()),
        "resource_pending_gpu_jobs": list(resource_pending_gpu_jobs.values()),
        "resource_gpu_pressure_states": _sorted_resource_pressure_states(resource_gpu_pressure_states),
        "resource_gpu_util_recent": _gpu_util_recent_summary(resource_gpu_util_rows),
        "resource_last_events": resource_last_events[-12:],
        "event_count": len(events),
        "ledger_filename": ledger_filename,
        "generated_at": now,
        "generated_at_utc": _utc_iso(now),
        "eventual_consistency": "coordinator aggregation; final aggregation runs before merge/reduce completion",
    }


def _backfill_lhr_state_from_stage_performance(state: dict[str, Any], stage_csv: Path) -> dict[str, Any]:
    records = read_stage_performance_records(stage_csv)
    if not records:
        return state
    out = dict(state)
    record_count = len(records)
    if _int_or_zero(out.get("stage_count")) < record_count:
        out["stage_count"] = record_count
        workers = out.get("workers") if isinstance(out.get("workers"), dict) else {}
        if workers:
            out["workers"] = workers
            stage_counts = Counter(
                rec.worker_id or _worker_id_from_candidate(rec.candidate_id) or "W00"
                for rec in records
            )
            for worker_id, count in stage_counts.items():
                worker = workers.get(worker_id)
                if isinstance(worker, dict) and _int_or_zero(worker.get("stage_count")) < count:
                    worker["stage_count"] = count
    best = build_score_summary(records).get("valid_best_score")
    if isinstance(best, dict) and best:
        out["global_best"] = _score_summary_best_to_lhr_global_best(best)
        as_of = _path_mtime(stage_csv)
        if as_of:
            out["global_best_as_of"] = as_of
            out["global_best_as_of_utc"] = _utc_iso(as_of)
    return out


def _score_summary_best_to_lhr_global_best(best: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": best.get("candidate_id") or "",
        "worker_id": best.get("worker_id") or _worker_id_from_candidate(best.get("candidate_id")) or "",
        "stage_id": best.get("stage_id") or "",
        "snapshot_id": "",
        "metric_value": best.get("value"),
        "metric_name": best.get("metric_name") or "",
        "lower_is_better": best.get("lower_is_better"),
        "validation_ok": best.get("validation_ok"),
        "metric_validity": best.get("metric_validity") or "",
        "score_source": "lhr_stage_performance.csv",
    }


def _worker_id_from_candidate(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("W") and len(text) >= 3 and text[1:3].isdigit():
        return text[:3]
    return ""


def _int_or_zero(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _path_mtime(path: Path) -> float:
    try:
        return float(Path(path).stat().st_mtime)
    except OSError:
        return 0.0


class LHRStateMachineStore:
    """Small state/event sidecar for the lnr solver.

    This class is intentionally independent from LNR/MCTS. It mirrors LHR
    control events into one stable stream and maintains a compact monitor state.
    """

    def __init__(
        self,
        *,
        log_dir: Path,
        worker_id: str,
        worker_index: int = 0,
        worker_count: int = 1,
        ledger_filename: str = ".run_results.md",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.worker_id = str(worker_id or "W00")
        self.worker_index = int(worker_index or 0)
        self.worker_count = max(1, int(worker_count or 1))
        self.ledger_filename = str(ledger_filename or "run_results.md")
        self.event_count = 0
        self.stage_count = 0
        self.estra_decision_count = 0
        self.estra_invalid_decision_count = 0
        self.estra_continue_count = 0
        self.estra_redirect_count = 0
        self.estra_switch_count = 0
        self.estra_current_continue_count = 0
        self.estra_current_redirect_count = 0
        self.estra_stage_continue_count = 0
        self.estra_stage_redirect_count = 0
        self.estra_switch_stage_count = 0
        self.estra_compact_count = 0
        self.resource_job_count = 0
        self.resource_guard_action_count = 0
        self.resource_cleanup_heartbeat_count = 0
        self.resource_monitor_agent_event_count = 0
        self.resource_request_count = 0
        self.resource_source_hint_count = 0
        self.resource_gpu_queue_wait_count = 0
        self.resource_gpu_queue_timeout_count = 0
        self.resource_gpu_queue_wait_durations: list[float] = []
        self._resource_gpu_queue_wait_started_at: dict[str, float] = {}
        self.resource_policy_gate_count = 0
        self.resource_planner_guard_count = 0
        self.resource_admission_deferred_count = 0
        self.resource_gpu_lease_acquired_count = 0
        self.resource_gpu_lease_released_count = 0
        self.resource_gpu_util_sample_count = 0
        self.progress_heartbeat_count = 0
        self.progress_last: dict[str, Any] = {}
        self.agent_backoff_wait_count = 0
        self.agent_backoff_wait_sec_total = 0.0
        self.agent_backoff_wait_sec_max = 0.0
        self.post_feedback_action_count = 0
        self.post_feedback_action_counts: dict[str, int] = {}
        self.post_feedback_violation_count = 0
        self.agent_backoff_wait_active: dict[str, dict[str, Any]] = {}
        self.agent_backoff_wake_reasons: dict[str, int] = {}
        self.resource_active_leases: dict[str, dict[str, Any]] = {}
        self.resource_pending_gpu_jobs: dict[str, dict[str, Any]] = {}
        self.resource_gpu_pressure_states: dict[str, dict[str, Any]] = {}
        self.resource_gpu_util_rows: dict[str, list[dict[str, Any]]] = {}
        self.resource_last_events: list[dict[str, Any]] = []
        self.global_best: dict[str, Any] | None = None
        self.global_best_as_of = 0.0
        self.run_status = "created"
        self.started_at = _now_ts()
        self.last_event_at = self.started_at

    @property
    def events_path(self) -> Path:
        return self.log_dir / LHR_EVENTS_JSONL

    @property
    def state_path(self) -> Path:
        return self.log_dir / LHR_STATE_JSON

    def _base_state(self) -> dict[str, Any]:
        now = _now_ts()
        best = dict(self.global_best or {})
        return {
            "schema_version": 1,
            "solver": "lnr",
            "run_status": self.run_status,
            "started_at": self.started_at,
            "started_at_utc": _utc_iso(self.started_at),
            "worker_count": self.worker_count,
            "workers": {
                self.worker_id: {
                    "worker_id": self.worker_id,
                    "worker_index": self.worker_index,
                    "status": self.run_status,
                    "stage_count": self.stage_count,
                    "estra_decisions": self.estra_decision_count,
                    "estra_invalid_decisions": self.estra_invalid_decision_count,
                    "estra_continue_count": self.estra_continue_count,
                    "estra_redirect_count": self.estra_redirect_count,
                    "estra_switch_count": self.estra_switch_count,
                    "estra_current_continue_count": self.estra_current_continue_count,
                    "estra_current_redirect_count": self.estra_current_redirect_count,
                    "estra_stage_continue_count": self.estra_stage_continue_count,
                    "estra_stage_redirect_count": self.estra_stage_redirect_count,
                    "estra_switch_stage_count": self.estra_switch_stage_count,
                    "estra_compact_count": self.estra_compact_count,
                    "resource_jobs": self.resource_job_count,
                    "resource_guard_actions": self.resource_guard_action_count,
                    "resource_cleanup_heartbeats": self.resource_cleanup_heartbeat_count,
                    "resource_monitor_agent_events": self.resource_monitor_agent_event_count,
                    "resource_requests": self.resource_request_count,
                    "resource_source_hints": self.resource_source_hint_count,
                    "resource_gpu_queue_waits": self.resource_gpu_queue_wait_count,
                    "resource_gpu_queue_timeouts": self.resource_gpu_queue_timeout_count,
                    **_queue_wait_stats(
                        self.resource_gpu_queue_wait_durations,
                        waits=self.resource_gpu_queue_wait_count,
                        timeouts=self.resource_gpu_queue_timeout_count,
                    ),
                    "resource_policy_gates": self.resource_policy_gate_count,
                    "resource_planner_guards": self.resource_planner_guard_count,
                    "resource_admission_deferred": self.resource_admission_deferred_count,
                    "resource_gpu_lease_acquired": self.resource_gpu_lease_acquired_count,
                    "resource_gpu_lease_released": self.resource_gpu_lease_released_count,
                    "resource_gpu_util_samples": self.resource_gpu_util_sample_count,
                    "progress_heartbeats": self.progress_heartbeat_count,
                    "agent_backoff_waits": self.agent_backoff_wait_count,
                    "post_feedback_actions": self.post_feedback_action_count,
                    "agent_backoff_wait_sec_total": round(self.agent_backoff_wait_sec_total, 3),
                    "agent_backoff_wait_sec_max": round(self.agent_backoff_wait_sec_max, 3),
                    "last_event_at": self.last_event_at,
                    "last_event_at_utc": _utc_iso(self.last_event_at),
                }
            },
            "global_best": best,
            "global_best_as_of": self.global_best_as_of,
            "global_best_as_of_utc": _utc_iso(self.global_best_as_of) if self.global_best_as_of else "",
            "stage_count": self.stage_count,
            "estra_decisions": self.estra_decision_count,
            "estra_invalid_decisions": self.estra_invalid_decision_count,
            "estra_continue_count": self.estra_continue_count,
            "estra_redirect_count": self.estra_redirect_count,
            "estra_switch_count": self.estra_switch_count,
            "estra_current_continue_count": self.estra_current_continue_count,
            "estra_current_redirect_count": self.estra_current_redirect_count,
            "estra_stage_continue_count": self.estra_stage_continue_count,
            "estra_stage_redirect_count": self.estra_stage_redirect_count,
            "estra_start_current_count": self.estra_current_continue_count + self.estra_current_redirect_count,
            "estra_start_stage_count": self.estra_stage_continue_count + self.estra_stage_redirect_count,
            "estra_intent_continue_count": self.estra_current_continue_count + self.estra_stage_continue_count,
            "estra_intent_redirect_count": self.estra_current_redirect_count + self.estra_stage_redirect_count,
            "estra_switch_stage_count": self.estra_switch_stage_count,
            "estra_compact_count": self.estra_compact_count,
            "resource_jobs": self.resource_job_count,
            "resource_guard_actions": self.resource_guard_action_count,
            "resource_cleanup_heartbeats": self.resource_cleanup_heartbeat_count,
            "resource_monitor_agent_events": self.resource_monitor_agent_event_count,
            "resource_requests": self.resource_request_count,
            "resource_source_hints": self.resource_source_hint_count,
            "resource_gpu_queue_waits": self.resource_gpu_queue_wait_count,
            "resource_gpu_queue_timeouts": self.resource_gpu_queue_timeout_count,
            **_queue_wait_stats(
                self.resource_gpu_queue_wait_durations,
                waits=self.resource_gpu_queue_wait_count,
                timeouts=self.resource_gpu_queue_timeout_count,
            ),
            "resource_policy_gates": self.resource_policy_gate_count,
            "resource_planner_guards": self.resource_planner_guard_count,
            "resource_admission_deferred": self.resource_admission_deferred_count,
            "resource_gpu_lease_acquired": self.resource_gpu_lease_acquired_count,
            "resource_gpu_lease_released": self.resource_gpu_lease_released_count,
            "resource_gpu_util_samples": self.resource_gpu_util_sample_count,
            "progress_heartbeats": self.progress_heartbeat_count,
            "progress_last": dict(self.progress_last or {}),
            "agent_backoff_waits": self.agent_backoff_wait_count,
            "post_feedback_actions": self.post_feedback_action_count,
            "post_feedback_action_counts": dict(sorted(self.post_feedback_action_counts.items())),
            "post_feedback_violations": self.post_feedback_violation_count,
            "agent_backoff_wait_sec_total": round(self.agent_backoff_wait_sec_total, 3),
            "agent_backoff_wait_sec_max": round(self.agent_backoff_wait_sec_max, 3),
            "agent_backoff_wait_active": list(self.agent_backoff_wait_active.values()),
            "agent_backoff_wake_reasons": dict(sorted(self.agent_backoff_wake_reasons.items())),
            "resource_active_leases": list(self.resource_active_leases.values()),
            "resource_pending_gpu_jobs": list(self.resource_pending_gpu_jobs.values()),
            "resource_gpu_pressure_states": _sorted_resource_pressure_states(self.resource_gpu_pressure_states),
            "resource_gpu_util_recent": _gpu_util_recent_summary(self.resource_gpu_util_rows),
            "resource_last_events": self.resource_last_events[-12:],
            "event_count": self.event_count,
            "ledger_filename": self.ledger_filename,
            "generated_at": now,
            "generated_at_utc": _utc_iso(now),
            "eventual_consistency": "single-writer state; coordinator aggregation is added in a later phase",
        }

    def write_state(self) -> None:
        _atomic_write_json(self.state_path, self._base_state())

    def mark_run_status(self, status: str, *, payload: dict[str, Any] | None = None) -> None:
        status = str(status or "").strip() or self.run_status
        self.run_status = status
        self.append_event(
            "worker_status",
            task_type="repl_search",
            task_id=f"repl_search:{self.worker_id}",
            status=status,
            payload=payload or {},
        )

    def append_event(
        self,
        event_type: str,
        *,
        task_type: str = "",
        task_id: str = "",
        status: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        now = _now_ts()
        event = {
            "schema_version": 1,
            "timestamp": now,
            "timestamp_utc": _utc_iso(now),
            "event": str(event_type or "event"),
            "task_type": str(task_type or ""),
            "task_id": str(task_id or ""),
            "status": str(status or ""),
            "worker_id": self.worker_id,
            "worker_index": self.worker_index,
            "payload": _json_safe(payload or {}),
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self.event_count += 1
        self.last_event_at = now
        self._update_counters(event)
        self.write_state()

    @classmethod
    def aggregate_logs(
        cls,
        *,
        output_log_dir: Path,
        input_log_dirs: list[Path],
        worker_count: int = 1,
        run_status: str = "",
        ledger_filename: str = ".run_results.md",
    ) -> dict[str, Any]:
        seen: set[str] = set()
        events: list[dict[str, Any]] = []
        for log_dir in input_log_dirs:
            for event in _read_events(Path(log_dir) / LHR_EVENTS_JSONL):
                key = _event_dedupe_key(event)
                if key in seen:
                    continue
                seen.add(key)
                events.append(event)
        events.sort(key=lambda e: float(e.get("timestamp") or 0.0))
        out_dir = Path(output_log_dir)
        _atomic_write_jsonl(out_dir / LHR_EVENTS_JSONL, events)
        state = build_lhr_state_from_events(
            events,
            worker_count=worker_count,
            run_status=run_status,
            ledger_filename=ledger_filename,
        )
        state = _backfill_lhr_state_from_stage_performance(
            state,
            out_dir / "lhr_stage_performance.csv",
        )
        _atomic_write_json(out_dir / LHR_STATE_JSON, state)
        return state

    def _update_counters(self, event: dict[str, Any]) -> None:
        et = str(event.get("event") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        _apply_resource_event_to_views(
            event,
            active_leases=self.resource_active_leases,
            pending_jobs=self.resource_pending_gpu_jobs,
            last_events=self.resource_last_events,
            pressure_states=self.resource_gpu_pressure_states,
        )
        if et == "stage_captured":
            self.stage_count += 1
            self._maybe_update_best(payload)
        elif et == "estra_decision":
            self.estra_decision_count += 1
            action = str(payload.get("action") or "")
            startpoint = str(payload.get("startpoint") or ("previous_stage" if action == "switch_stage" else "current_workspace"))
            intent = str(payload.get("intent") or ("redirect" if action == "keep_but_redirect" else "continue"))
            valid_estra_action = action in {"keep_current", "keep_but_redirect", "switch_stage"}
            if action == "keep_current":
                self.estra_continue_count += 1
            elif action == "keep_but_redirect":
                self.estra_redirect_count += 1
            elif action == "switch_stage":
                self.estra_switch_count += 1
            if valid_estra_action:
                if startpoint == "previous_stage" and intent == "redirect":
                    self.estra_stage_redirect_count += 1
                elif startpoint == "previous_stage":
                    self.estra_stage_continue_count += 1
                elif intent == "redirect":
                    self.estra_current_redirect_count += 1
                else:
                    self.estra_current_continue_count += 1
        elif et == "estra_invalid":
            self.estra_invalid_decision_count += 1
        elif et == "estra_stage_switched":
            self.estra_switch_stage_count += 1
        elif et == "estra_keep_current_compacted":
            self.estra_compact_count += 1
        elif et == "resource_job_started":
            self.resource_job_count += 1
        elif et == "resource_guard_action":
            self.resource_guard_action_count += 1
        elif et == "resource_cleanup_heartbeat":
            self.resource_cleanup_heartbeat_count += 1
        elif et == "resource_monitor_agent_shadow":
            self.resource_monitor_agent_event_count += 1
        elif et == "resource_request_created":
            self.resource_request_count += 1
        elif et == "resource_source_hint_detected":
            self.resource_source_hint_count += 1
        elif et == "resource_gpu_queue_wait_started":
            self.resource_gpu_queue_wait_count += 1
            job_id = str(payload.get("job_id") or event.get("task_id") or "")
            if job_id:
                self._resource_gpu_queue_wait_started_at[job_id] = _float_or_zero(event.get("timestamp"))
        elif et == "resource_gpu_queue_timeout":
            self.resource_gpu_queue_timeout_count += 1
            job_id = str(payload.get("job_id") or event.get("task_id") or "")
            started_at = self._resource_gpu_queue_wait_started_at.pop(job_id, 0.0) if job_id else 0.0
            elapsed = _float_or_zero(payload.get("elapsed_sec"))
            if elapsed <= 0.0 and started_at > 0.0:
                elapsed = _float_or_zero(event.get("timestamp")) - started_at
            self.resource_gpu_queue_wait_durations.append(max(0.0, elapsed))
        elif et == "resource_policy_gate":
            self.resource_policy_gate_count += 1
        elif et == "resource_planner_guard":
            self.resource_planner_guard_count += 1
        elif et == "resource_admission_deferred":
            self.resource_admission_deferred_count += 1
        elif et == "resource_gpu_lease_acquired":
            self.resource_gpu_lease_acquired_count += 1
            job_id = str(payload.get("job_id") or event.get("task_id") or "")
            started_at = self._resource_gpu_queue_wait_started_at.pop(job_id, 0.0) if job_id else 0.0
            if started_at > 0.0:
                self.resource_gpu_queue_wait_durations.append(
                    max(0.0, _float_or_zero(event.get("timestamp")) - started_at)
                )
        elif et == "resource_gpu_lease_released":
            self.resource_gpu_lease_released_count += 1
        elif et == "resource_gpu_util_sampled":
            self.resource_gpu_util_sample_count += 1
            _append_gpu_util_rows(self.resource_gpu_util_rows, event)
        elif et == "progress_heartbeat":
            self.progress_heartbeat_count += 1
            self.progress_last = _resource_event_summary(event)
        elif et == "agent_backoff_wait_started":
            self.agent_backoff_wait_count += 1
            wait_id = str(payload.get("wait_id") or event.get("task_id") or "")
            if wait_id:
                self.agent_backoff_wait_active[wait_id] = _resource_event_summary(event)
        elif et == "agent_backoff_wait_finished":
            elapsed = _float_or_zero(payload.get("elapsed_sec"))
            self.agent_backoff_wait_sec_total += elapsed
            self.agent_backoff_wait_sec_max = max(self.agent_backoff_wait_sec_max, elapsed)
            wake = str(payload.get("wake_reason") or "unknown")
            self.agent_backoff_wake_reasons[wake] = self.agent_backoff_wake_reasons.get(wake, 0) + 1
            wait_id = str(payload.get("wait_id") or event.get("task_id") or "")
            if wait_id:
                self.agent_backoff_wait_active.pop(wait_id, None)
        elif et == "post_feedback_action":
            self.post_feedback_action_count += 1
            action = str(payload.get("action") or "unknown")
            self.post_feedback_action_counts[action] = self.post_feedback_action_counts.get(action, 0) + 1
            if bool(payload.get("violates_feedback")):
                self.post_feedback_violation_count += 1

    def _maybe_update_best(self, payload: dict[str, Any]) -> None:
        record = score_record_from_stage_payload(payload, worker_id=self.worker_id)
        if record is None or not record.valid_comparable:
            return
        metric = record.value
        if metric != metric:
            return
        current_metric = None
        if self.global_best:
            try:
                current_metric = float(self.global_best.get("metric_value"))
            except (TypeError, ValueError):
                current_metric = None
        lower_is_better = record.lower_is_better
        better = current_metric is None or (metric < current_metric if lower_is_better else metric > current_metric)
        if not better:
            return
        now = _now_ts()
        self.global_best = {
            "candidate_id": record.candidate_id or f"{self.worker_id}:{payload.get('stage_id') or ''}",
            "worker_id": record.worker_id or self.worker_id,
            "stage_id": record.stage_id or payload.get("stage_id") or "",
            "snapshot_id": payload.get("snapshot_id") or "",
            "metric_value": metric,
            "metric_name": record.metric_name,
            "lower_is_better": lower_is_better,
            "validation_ok": record.validation_ok,
            "metric_validity": record.metric_validity,
            "validity": record.validity,
            "evaluator_backend": record.evaluator_backend,
            "evaluator_status": record.evaluator_status,
        }
        self.global_best_as_of = now
