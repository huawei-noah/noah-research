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

"""Build monitor trace data from existing ScienceFlow monitor artifacts."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from typing import Any

from scienceflow.solver.lnr.stage.score_summary import (
    ScoreRecord,
    metric_lower_is_better_hint,
    score_record_from_stage_performance_row,
)
from scienceflow.ui.monitor.helpers import parse_monitor_manifest
from scienceflow.ui.monitor.lnr_state import load_lnr_task_state
from scienceflow.ui.monitor_trace.leaderboard import (
    leaderboard_with_position,
    load_manifest_leaderboards,
)
from scienceflow.ui.monitor_trace.models import (
    BestPoint,
    Leaderboard,
    MonitorTraceReport,
    TaskTrace,
    TraceEvent,
    TracePoint,
)

_RECENT_JSONL_TAIL_BYTES = 4 * 1024 * 1024
_MAX_EVENTS_PER_TASK = 200


def build_monitor_trace_report(manifest_path: str | Path) -> MonitorTraceReport:
    manifest = Path(manifest_path).expanduser().resolve()
    leaderboards = load_manifest_leaderboards(manifest)
    tasks = [
        build_task_trace(
            run_id=run_id,
            gpu_list=gpu_list,
            monitor_state_path=state_path,
            leaderboard=leaderboards.get(run_id),
        )
        for run_id, gpu_list, state_path in parse_monitor_manifest(manifest)
    ]
    return MonitorTraceReport(
        schema_version=1,
        generated_at_utc=_utc_now_iso(),
        manifest_path=str(manifest),
        tasks=tasks,
    )


def build_task_trace(
    *,
    run_id: str,
    gpu_list: str,
    monitor_state_path: Path,
    leaderboard: Leaderboard | None = None,
) -> TaskTrace:
    task_root = _task_root_from_monitor_state_path(monitor_state_path)
    state = load_lnr_task_state(task_root)
    rows = _load_csv_rows(task_root / "task_logs" / "lhr_stage_performance.csv")
    points, x_mode, lower = _build_points(rows)
    lower, direction_source, direction_conflict = _resolve_lower_is_better(
        lower,
        state,
        rows=rows,
        leaderboard=leaderboard,
        task_root=task_root,
    )
    best_points, best_kind = _build_best_points(points, lower=lower)
    events = _load_trace_events(task_root, x_mode=x_mode)
    metric_name = points[-1].metric_name if points else str(state.get("metric_name") or "metric")
    summary = _build_summary(
        state,
        points,
        best_kind,
        lower=lower,
        direction_source=direction_source,
        direction_conflict=direction_conflict,
    )
    leaderboard = leaderboard_with_position(leaderboard, summary=summary, lower=lower)
    return TaskTrace(
        run_id=run_id,
        exp_id=str(state.get("exp_id") or task_root.name),
        gpu_list=gpu_list,
        task_root=str(task_root),
        status=str(state.get("status") or "unknown"),
        metric_name=metric_name,
        lower_is_better=lower,
        x_mode=x_mode,
        summary=summary,
        points=points,
        best_points=best_points,
        events=events,
        leaderboard=leaderboard,
    )


def _task_root_from_monitor_state_path(path: Path) -> Path:
    p = Path(path).expanduser()
    if p.name == "monitor_state.json" and p.parent.name == "logs":
        return p.parent.parent
    return p.parent


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def _build_points(rows: list[dict[str, str]]) -> tuple[list[TracePoint], str, bool]:
    records: list[tuple[dict[str, str], ScoreRecord]] = []
    for row in rows:
        rec = score_record_from_stage_performance_row(row)
        if rec is not None:
            records.append((row, rec))
    if not records:
        return [], "row_order", True

    parsed_times = [_parse_row_time(row) for row, _ in records]
    use_time = sum(1 for value in parsed_times if value is not None) >= 2
    x_values = _filled_time_values(parsed_times) if use_time else []
    lower = _majority_lower_is_better([rec.lower_is_better for _, rec in records])
    points: list[TracePoint] = []
    for idx, ((row, rec), timestamp) in enumerate(zip(records, parsed_times, strict=False), start=1):
        x_value = x_values[idx - 1] if use_time else float(rec.row_order or idx)
        label_time = x_value if use_time else timestamp
        points.append(
            TracePoint(
                row_order=rec.row_order or idx,
                x_value=x_value,
                x_label=_x_label(row, rec, label_time, fallback=idx),
                metric=rec.value,
                metric_name=rec.metric_name,
                worker_id=rec.worker_id,
                stage_id=rec.stage_id,
                candidate_id=rec.candidate_id,
                validity=rec.validity,
                metric_validity=rec.metric_validity,
                valid_comparable=rec.valid_comparable,
                candidate_ready=rec.candidate_ready,
            )
        )
    return points, "wall_clock" if use_time else "row_order", lower


def _build_best_points(points: list[TracePoint], *, lower: bool) -> tuple[list[BestPoint], str]:
    valid_points = [p for p in points if p.valid_comparable]
    if not valid_points:
        return [], "none"
    source_ids = {id(p) for p in valid_points}
    current: TracePoint | None = None
    best: list[BestPoint] = []
    for point in points:
        if id(point) in source_ids and _is_better(
            point.metric,
            current.metric if current else None,
            lower=lower,
        ):
            current = point
        if current is not None:
            best.append(
                BestPoint(
                    x_value=point.x_value,
                    x_label=point.x_label,
                    metric=current.metric,
                    stage_id=current.stage_id,
                    kind="valid_best",
                )
            )
    return best, "valid_best"


def _majority_lower_is_better(votes: list[bool]) -> bool:
    if not votes:
        return True
    true_count = votes.count(True)
    false_count = votes.count(False)
    if true_count == false_count:
        return votes[-1]
    return true_count > false_count


def _filled_time_values(values: list[float | None]) -> list[float]:
    known = [(idx, value) for idx, value in enumerate(values) if value is not None]
    if not known:
        return [float(idx + 1) for idx in range(len(values))]
    default_step = _median_positive_step(known) or 60.0
    out: list[float] = []
    for idx, value in enumerate(values):
        if value is not None:
            out.append(value)
            continue
        prev = next(((i, v) for i, v in reversed(known) if i < idx), None)
        nxt = next(((i, v) for i, v in known if i > idx), None)
        if prev is not None and nxt is not None and nxt[0] > prev[0]:
            ratio = (idx - prev[0]) / (nxt[0] - prev[0])
            out.append(prev[1] + ((nxt[1] - prev[1]) * ratio))
        elif prev is not None:
            out.append(prev[1] + ((idx - prev[0]) * default_step))
        elif nxt is not None:
            out.append(nxt[1] - ((nxt[0] - idx) * default_step))
        else:
            out.append(float(idx + 1))
    return out


def _median_positive_step(known: list[tuple[int, float]]) -> float | None:
    steps = [
        (cur_value - prev_value) / (cur_idx - prev_idx)
        for (prev_idx, prev_value), (cur_idx, cur_value) in zip(known, known[1:], strict=False)
        if cur_idx > prev_idx and cur_value > prev_value
    ]
    if not steps:
        return None
    steps.sort()
    return steps[len(steps) // 2]


def _is_better(candidate: float, current: float | None, *, lower: bool) -> bool:
    if current is None:
        return True
    return candidate < current if lower else candidate > current


def _parse_row_time(row: dict[str, Any]) -> float | None:
    for key in ("created_at_utc", "timestamp_utc", "timestamp", "created_at"):
        value = row.get(key)
        if value in (None, ""):
            continue
        parsed = _parse_time_value(value)
        if parsed is not None:
            return parsed
    return None


def _parse_time_value(value: Any) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        if text.replace(".", "", 1).isdigit():
            return float(text)
        path_time = _mtime_from_path_text(text)
        if path_time is not None:
            return path_time
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _mtime_from_path_text(text: str) -> float | None:
    if not (text.startswith("/") or text.startswith("~")):
        return None
    try:
        path = Path(text).expanduser()
        return path.stat().st_mtime if path.exists() else None
    except OSError:
        return None


def _x_label(row: dict[str, Any], rec: ScoreRecord, timestamp: float | None, *, fallback: int) -> str:
    stage = rec.stage_id or rec.candidate_id or f"row-{fallback}"
    if timestamp is None:
        return stage
    return f"{_format_time_label(timestamp)} {stage}"


def _format_time_label(timestamp: float) -> str:
    try:
        return dt.datetime.fromtimestamp(timestamp, tz=dt.UTC).strftime("%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return str(timestamp)


def _resolve_lower_is_better(
    default: bool,
    state: dict[str, Any],
    *,
    rows: list[dict[str, str]],
    leaderboard: Leaderboard | None,
    task_root: Path,
) -> tuple[bool, str, bool]:
    if leaderboard is not None and isinstance(leaderboard.lower_is_better, bool):
        return leaderboard.lower_is_better, "leaderboard", False

    votes = [
        rec.lower_is_better
        for row in rows
        if (rec := score_record_from_stage_performance_row(row)) is not None
    ]
    direction_conflict = len(set(votes)) > 1
    task_hint = _task_description_direction_hint(task_root)
    if task_hint is not None:
        return task_hint, "task_description", direction_conflict or any(vote != task_hint for vote in votes)

    if direction_conflict:
        semantic_votes = [
            hint
            for row in rows
            if (hint := _row_metric_direction_hint(row)) is not None
        ]
        if semantic_votes and len(set(semantic_votes)) == 1:
            return semantic_votes[0], "metric_text_after_conflict", True
        true_count = votes.count(True)
        false_count = votes.count(False)
        if true_count != false_count:
            return true_count > false_count, "stage_majority_after_conflict", True
        # The latest capture is only a last resort when neither metric semantics
        # nor the explicit stage votes resolve the conflict.
        return votes[-1], "latest_stage_after_unresolved_conflict", True

    raw = state.get("lower_is_better")
    if isinstance(raw, bool):
        return raw, "monitor_state", False
    if votes:
        return default, "stage_consensus", False
    return default, "default", False


def _task_description_direction_hint(task_root: Path) -> bool | None:
    candidates = [task_root / "description.md"]
    for workers_dir in (task_root / "workers", task_root / "lhr_workers"):
        if not workers_dir.is_dir():
            continue
        candidates.extend(sorted(workers_dir.glob("*/workspace/description.md")))
        candidates.extend(sorted(workers_dir.glob("*/workspace/dataset/description.md")))
    for path in candidates:
        try:
            hint = metric_lower_is_better_hint(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if hint is not None:
            return hint
    return None


def _row_metric_direction_hint(row: dict[str, str]) -> bool | None:
    for key in (
        "metric_name",
        "brief",
        "metric_validity_note",
        "metric_source_note",
        "why",
    ):
        hint = metric_lower_is_better_hint(str(row.get(key) or ""))
        if hint is not None:
            return hint
    return None


def _build_summary(
    state: dict[str, Any],
    points: list[TracePoint],
    best_kind: str,
    *,
    lower: bool,
    direction_source: str,
    direction_conflict: bool,
) -> dict[str, Any]:
    best_raw = _best_trace_point(points, lower=lower)
    best_valid = _best_trace_point(
        [point for point in points if point.valid_comparable],
        lower=lower,
    )
    best_raw_metric = best_raw.metric if best_raw is not None else None
    best_valid_metric = best_valid.metric if best_valid is not None else None
    return {
        "stage_count": state.get("stage_count") or len({(p.worker_id, p.stage_id) for p in points}),
        "candidate_stage_count": state.get("candidate_stage_count") or len(points),
        "best_raw_metric": best_raw_metric,
        "best_valid_metric": best_valid_metric,
        "best_kind": best_kind,
        "metric_direction_source": direction_source,
        "metric_direction_conflict": direction_conflict,
        "elapsed_sec": state.get("elapsed_sec"),
        "remaining_sec": state.get("remaining_sec"),
        "estra_decision_count": state.get("estra_decision_count"),
        "estra_switch_count": state.get("estra_switch_count"),
        "resource_outcomes": state.get("resource_outcomes") or {},
        "llm_display_text": _format_llm_display(state),
        "total_llm_cost_usd": state.get("total_llm_cost_usd"),
    }


def _best_trace_point(points: list[TracePoint], *, lower: bool) -> TracePoint | None:
    if not points:
        return None
    return min(points, key=lambda point: point.metric) if lower else max(points, key=lambda point: point.metric)


def _format_llm_display(state: dict[str, Any]) -> str:
    config = state.get("llm_config") if isinstance(state.get("llm_config"), dict) else {}
    fallback_text = str(state.get("llm_config_text") or "")
    lines: list[str] = []
    for stage_name in ("code", "feedback"):
        stage = config.get(stage_name) if isinstance(config, dict) else None
        model = _model_from_stage_summary(stage) if isinstance(stage, dict) else ""
        if not model:
            model = _model_from_formatted_llm_text(fallback_text, stage_name)
        if model:
            lines.append(f"{stage_name}={model}")
    return "\n".join(lines)


def _model_from_stage_summary(stage: dict[str, Any]) -> str:
    model = str(stage.get("model") or "").strip()
    if model:
        return model
    models = stage.get("models") if isinstance(stage.get("models"), list) else []
    return str(models[0]).strip() if models else ""


def _model_from_formatted_llm_text(text: str, stage_name: str) -> str:
    prefix = f"{stage_name}="
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(prefix):
            continue
        value = line[len(prefix):].strip().split(None, 1)[0]
        return value.strip()
    return ""


def _load_trace_events(task_root: Path, *, x_mode: str) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for path in [
        task_root / "task_logs" / "lhr_events.jsonl",
        task_root / "task_logs" / "resource" / "resource_events.jsonl",
    ]:
        for item in _iter_recent_jsonl(path):
            event = _event_from_json(item, x_mode=x_mode)
            if event is not None:
                events.append(event)
    return events[-_MAX_EVENTS_PER_TASK:]


def _iter_recent_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > _RECENT_JSONL_TAIL_BYTES:
                f.seek(size - _RECENT_JSONL_TAIL_BYTES)
                f.readline()
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in data.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _event_from_json(item: dict[str, Any], *, x_mode: str) -> TraceEvent | None:
    raw_kind = str(item.get("event") or item.get("event_type") or "").strip()
    kind = _normalize_event_kind(raw_kind, item)
    if not kind:
        return None
    timestamp = _parse_time_value(
        item.get("timestamp") or item.get("created_at_utc") or item.get("time"),
    )
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    label = _event_label(kind, raw_kind, payload)
    x_value = timestamp if x_mode == "wall_clock" else None
    x_label = _format_time_label(timestamp) if timestamp is not None else ""
    return TraceEvent(x_value=x_value, x_label=x_label, kind=kind, label=label)


def _normalize_event_kind(raw_kind: str, item: dict[str, Any]) -> str:
    lowered = raw_kind.lower()
    if "estra" in lowered:
        return "estra"
    if "kill" in lowered or "timeout" in lowered or "timebox" in lowered:
        return "guard"
    if lowered == "resource_review_outcome":
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        outcome = str(payload.get("execution_outcome") or "").lower()
        if outcome and outcome not in {"no_action", "none"}:
            return "guard"
    if "resume" in lowered:
        return "resume"
    return ""


def _event_label(kind: str, raw_kind: str, payload: dict[str, Any]) -> str:
    if kind == "estra":
        action = str(payload.get("action") or payload.get("intent") or "").strip()
        return f"estra {action}".strip()
    if kind == "guard":
        outcome = str(payload.get("execution_outcome") or payload.get("action") or "").strip()
        return f"guard {outcome}".strip()
    return raw_kind or kind


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
