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

"""Leaderboard baseline loading and rank projection for monitor trace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from scienceflow.ui.monitor_trace.models import Leaderboard, LeaderboardEntry


def load_manifest_leaderboards(manifest: Path) -> dict[str, Leaderboard]:
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    table = _load_leaderboard_table(manifest, data)
    out: dict[str, Leaderboard] = {}
    for idx, task in enumerate(data.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        exp_id = str(task.get("exp_id") or f"task-{idx}").strip()
        run_id = str(task.get("run_id") or exp_id).strip()
        leaderboard = _leaderboard_from_config(task.get("leaderboard"))
        if leaderboard is None:
            leaderboard = table.get(run_id) or table.get(exp_id)
        if leaderboard is None:
            continue
        if run_id:
            out[run_id] = leaderboard
        if exp_id:
            out.setdefault(exp_id, leaderboard)
    return out


def leaderboard_with_position(
    leaderboard: Leaderboard | None,
    *,
    summary: dict[str, Any],
    lower: bool,
) -> Leaderboard | None:
    if leaderboard is None:
        return None
    our_score, score_source = _best_summary_score(summary, leaderboard.score_source)
    if our_score is None:
        return leaderboard
    direction_lower = leaderboard.lower_is_better if leaderboard.lower_is_better is not None else lower
    ordered = sorted(
        leaderboard.entries,
        key=lambda entry: entry.score,
        reverse=not direction_lower,
    )
    better_entries = [
        entry
        for entry in ordered
        if _is_better(entry.score, our_score, lower=direction_lower)
    ]
    rank1 = ordered[0] if ordered else None
    next_better = better_entries[-1] if better_entries else None
    return Leaderboard(
        title=leaderboard.title,
        metric_name=leaderboard.metric_name,
        source_url=leaderboard.source_url,
        lower_is_better=leaderboard.lower_is_better,
        display_limit=leaderboard.display_limit,
        entries=leaderboard.entries,
        our_score=our_score,
        our_score_source=score_source,
        estimated_rank=(len(better_entries) + 1) if ordered else None,
        gap_to_rank1=_gap_to_target(rank1.score, our_score, lower=direction_lower)
        if rank1
        else None,
        gap_to_next=_gap_to_target(next_better.score, our_score, lower=direction_lower)
        if next_better
        else None,
        rank_basis=leaderboard.rank_basis,
        score_source=leaderboard.score_source,
    )


def _load_leaderboard_table(manifest: Path, manifest_data: dict[str, Any]) -> dict[str, Leaderboard]:
    path = _leaderboard_table_path(manifest, manifest_data)
    if path is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    raw_items = data.get("leaderboards") or data.get("tasks") or {}
    out: dict[str, Leaderboard] = {}
    if isinstance(raw_items, dict):
        iterable = raw_items.items()
    elif isinstance(raw_items, list):
        iterable = (
            (str(item.get("exp_id") or item.get("run_id") or ""), item)
            for item in raw_items
            if isinstance(item, dict)
        )
    else:
        return {}
    for key, raw in iterable:
        if not isinstance(raw, dict):
            continue
        leaderboard = _leaderboard_from_config(raw)
        if leaderboard is None:
            continue
        primary = str(key or raw.get("exp_id") or raw.get("run_id") or "").strip()
        aliases = [
            primary,
            str(raw.get("exp_id") or "").strip(),
            str(raw.get("run_id") or "").strip(),
        ]
        for alias in aliases:
            if alias:
                out[alias] = leaderboard
    return out


def _leaderboard_table_path(manifest: Path, data: dict[str, Any]) -> Path | None:
    keys = ("leaderboards_path", "leaderboard_table", "leaderboard_table_path")
    raw = next((data.get(key) for key in keys if data.get(key)), None)
    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    if not raw and isinstance(defaults, dict):
        raw = defaults.get("leaderboards_path") or defaults.get("leaderboard_table")
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    by_manifest = (manifest.parent / path).resolve()
    if by_manifest.exists():
        return by_manifest
    return (Path.cwd() / path).resolve()


def _leaderboard_from_config(raw: Any) -> Leaderboard | None:
    if not isinstance(raw, dict):
        return None
    raw_entries = raw.get("entries") or raw.get("top") or []
    entries: list[LeaderboardEntry] = []
    for idx, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            continue
        score = _float_or_none(item.get("score"))
        if score is None:
            continue
        rank = _int_or_none(item.get("rank")) or (idx + 1)
        user = str(item.get("user") or item.get("username") or f"rank-{rank}").strip()
        entries.append(
            LeaderboardEntry(
                rank=rank,
                user=user,
                score=score,
                submitted_at=str(item.get("submitted_at") or item.get("submittedAt") or "").strip(),
            ),
        )
    if not entries:
        return None
    return Leaderboard(
        title=str(raw.get("title") or "Public leaderboard").strip(),
        metric_name=str(raw.get("metric_name") or raw.get("metric") or "metric").strip(),
        source_url=str(raw.get("source_url") or raw.get("source") or "").strip(),
        lower_is_better=_bool_or_none(raw.get("lower_is_better")),
        display_limit=max(1, _int_or_none(raw.get("display_limit") or raw.get("top_n")) or 3),
        entries=entries,
        rank_basis=str(raw.get("rank_basis") or "known_public_entries").strip(),
        score_source=str(raw.get("score_source") or "").strip(),
    )


def _best_summary_score(summary: dict[str, Any], preferred: str = "") -> tuple[float | None, str]:
    scores = {
        "best_valid": _float_or_none(summary.get("best_valid_metric")),
        "best_raw": _float_or_none(summary.get("best_raw_metric")),
    }
    preferred = str(preferred or "").strip()
    if preferred in scores and scores[preferred] is not None:
        return scores[preferred], preferred
    if scores["best_valid"] is not None:
        return scores["best_valid"], "best_valid"
    if scores["best_raw"] is not None:
        return scores["best_raw"], "best_raw"
    return None, ""


def _gap_to_target(target: float, current: float, *, lower: bool) -> float:
    return current - target if lower else target - current


def _is_better(candidate: float, current: float | None, *, lower: bool) -> bool:
    if current is None:
        return True
    return candidate < current if lower else candidate > current


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None
