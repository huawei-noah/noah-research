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

"""Compact target-metric trajectory extraction for resource reviews."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

DEFAULT_MAX_LINES = 12
DEFAULT_MAX_LINE_CHARS = 220
DEFAULT_MAX_TEXT_CHARS = 1800

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TARGET_METRIC_RE = re.compile(
    r"\b("
    r"final\s+validation\s+score|"
    r"(?:avg|mean)?[_\s:-]*(?:levenshtein|edit[_\s-]*distance)|"
    r"(?:val|valid|validation|oof|cv|fold)[_\s:-]*(?:auc|score|loss|logloss|rmse|rmsle|mae|acc(?:uracy)?|f1|kt|kendall(?:_tau)?)|"
    r"(?:auc|score|rmse|rmsle|mae|logloss|accuracy|acc|f1|precision|recall|kt|kendall(?:_tau)?)"
    r")\b\s*(?:[:=]\s*)?[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
    re.IGNORECASE,
)
_NOISE_RE = re.compile(
    r"("
    r"\b(?:warning|warn|deprecatedwarning|futurewarning|userwarning)\b|"
    r"\btraceback\b|"
    r"^\s*file\s+\".*\",\s+line\s+\d+|"
    r"\b(?:download|extract|unzip|copying|cache|cached)\b|"
    r"\d+%\|.*\||"
    r"\bit/s\b|"
    r"\bETA\b"
    r")",
    re.IGNORECASE,
)
_LOSS_ONLY_METRIC_NAMES = frozenset({"loss", "train_loss", "training_loss"})


def _clean_line(line: str) -> str:
    cleaned = _ANSI_RE.sub("", str(line or ""))
    cleaned = cleaned.replace("\r", " ").strip()
    return re.sub(r"\s+", " ", cleaned)


def is_metric_history_line(line: str) -> bool:
    """Return true when a line contains a target metric, not only train loss/noise."""

    cleaned = _clean_line(line)
    if not cleaned:
        return False
    has_target = bool(_TARGET_METRIC_RE.search(cleaned))
    if not has_target:
        return False
    if _NOISE_RE.search(cleaned) and not has_target:
        return False
    # Loss-only training lines are intentionally excluded by the target metric
    # pattern. Validation loss or logloss lines still match and remain useful.
    return True


def extract_metric_history_lines(
    text: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
) -> list[str]:
    """Extract compact target-metric lines from raw stdout/stderr text."""

    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw_line in str(text).replace("\r", "\n").splitlines():
        line = _clean_line(raw_line)
        if not is_metric_history_line(line):
            continue
        if len(line) > max_line_chars:
            line = line[: max(0, max_line_chars - 3)].rstrip() + "..."
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    if max_lines > 0 and len(out) > max_lines:
        out = out[-max_lines:]
    return out


def update_metric_history_lines(
    existing: Iterable[str] | None,
    text: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
) -> list[str]:
    """Append newly extracted target metric lines while preserving order and deduping."""

    combined: list[str] = []
    seen: set[str] = set()
    for line in list(existing or []) + extract_metric_history_lines(
        text,
        max_lines=max_lines,
        max_line_chars=max_line_chars,
    ):
        cleaned = _clean_line(line)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append(cleaned)
    if max_lines > 0 and len(combined) > max_lines:
        combined = combined[-max_lines:]
    return combined


def metric_history_line_from_progress_signals(
    signals: dict[str, Any] | None,
    *,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
) -> str:
    """Render structured progress heartbeat metrics as one metric-history line."""

    if not isinstance(signals, dict):
        return ""
    raw_metrics = signals.get("metrics")
    if not isinstance(raw_metrics, dict):
        return ""

    metrics: list[tuple[str, float]] = []
    for raw_name, raw_value in raw_metrics.items():
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_name or "").strip().lower()).strip("_")
        if not name or name in _LOSS_ONLY_METRIC_NAMES:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value != value:
            continue
        metrics.append((name, value))
    if not metrics:
        return ""

    def _format_number(value: Any) -> str:
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return str(value).strip()

    heartbeat = signals.get("heartbeat") if isinstance(signals.get("heartbeat"), dict) else {}
    phase = str(signals.get("phase") or heartbeat.get("phase") or "").strip()
    parts = ["SCIENCEFLOW_HB"]
    if phase:
        parts.append(f"phase={phase}")

    for unit, raw_entry in signals.items():
        if unit in {"heartbeat", "metrics", "artifact", "phase"} or not isinstance(raw_entry, dict):
            continue
        if "current" not in raw_entry:
            continue
        current = _format_number(raw_entry.get("current"))
        total = raw_entry.get("total")
        if total is not None:
            parts.append(f"progress={current}/{_format_number(total)}")
        else:
            parts.append(f"progress={current}")
        parts.append(f"unit={str(unit or 'progress').strip() or 'progress'}")
        break

    parts.extend(f"{name}={value:.12g}" for name, value in metrics)
    line = _clean_line(" ".join(parts))
    if len(line) > max_line_chars:
        line = line[: max(0, max_line_chars - 3)].rstrip() + "..."
    return line if is_metric_history_line(line) else ""


def metric_history_text(
    lines: Iterable[str] | None,
    *,
    max_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    """Render metric history lines for fact cards without expanding normal bash output."""

    text = "\n".join(str(line).strip() for line in lines or [] if str(line).strip())
    if max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:].lstrip()
    return text
