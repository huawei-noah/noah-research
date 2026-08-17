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

import re
import shlex
from typing import Any


_PROGRESS_INDEX_RE = re.compile(
    r"\b(epoch|epochs|iter|iteration|step|batch)\s*[:=]?\s*(\d+(?:\.\d+)?)"
    r"(?:\s*/\s*(\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
_PROGRESS_METRIC_RE = re.compile(
    r"\b(loss|val_loss|valid_loss|auc|val_auc|valid_auc|acc|accuracy|score|rmse|rmsle|"
    r"avg_levenshtein|mean_levenshtein|levenshtein|edit_distance|mean_edit_distance)"
    r"\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_SCIENCEFLOW_HEARTBEAT_LINE_RE = re.compile(r"(^|\s)SCIENCEFLOW_HB\s+v=1\b")
_TQDM_PROGRESS_RE = re.compile(
    r"(?:^|\r|\n|\s)(?:\d{1,3}\s*%\|[^\r\n]*?\|\s*)?"
    r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?=\s*\[)",
    re.IGNORECASE,
)
_COUNTED_WORK_VERBS = r"extracted|processed|scored|predicted|generated|completed|loaded|written"
_COUNTED_WORK_SUFFIX_RE = re.compile(
    rf"(?:^|\r|\n|\s|\[)(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\]?"
    rf"\s*({_COUNTED_WORK_VERBS})\b",
    re.IGNORECASE,
)
_COUNTED_WORK_PREFIX_RE = re.compile(
    rf"\b({_COUNTED_WORK_VERBS})\b\s*[:=]?\s*"
    r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

AGENT_REPORTED = "agent_reported"


def parse_progress_signals(text: str) -> dict[str, Any]:
    """Extract progress hints while preserving their untrusted stdout provenance."""

    signals: dict[str, Any] = {}
    raw = str(text or "")
    heartbeat = _parse_scienceflow_heartbeat(raw)
    if heartbeat:
        signals["heartbeat"] = heartbeat
        phase = str(heartbeat.get("phase") or "").strip()
        if phase:
            signals["phase"] = phase
        progress = heartbeat.get("progress")
        if isinstance(progress, dict):
            unit = str(heartbeat.get("unit") or "progress").strip() or "progress"
            signals[unit] = progress
            _set_progress_evidence(signals, source="scienceflow_hb")
        metrics = heartbeat.get("metrics")
        if isinstance(metrics, dict) and metrics:
            signals["metrics"] = metrics
            _set_metric_evidence(signals, source="scienceflow_hb")

    for match in _PROGRESS_INDEX_RE.finditer(raw):
        name = match.group(1).lower()
        if name == "epochs":
            name = "epoch"
        elif name == "iteration":
            name = "iter"
        current = float(match.group(2))
        total = float(match.group(3)) if match.group(3) is not None else None
        entry: dict[str, Any] = {"current": current}
        if total is not None:
            entry["total"] = total
        signals[name] = entry
        _set_progress_evidence(signals, source="stdout_pattern")

    for match in _TQDM_PROGRESS_RE.finditer(raw):
        current = float(match.group(1))
        total = float(match.group(2))
        if total > 0.0 and current <= total:
            signals.setdefault("items", {"current": current, "total": total, "source": "tqdm"})
            _set_progress_evidence(signals, source="tqdm")

    for match in _COUNTED_WORK_SUFFIX_RE.finditer(raw):
        _set_counted_progress(
            signals,
            verb=match.group(3),
            current=match.group(1),
            total=match.group(2),
        )
    for match in _COUNTED_WORK_PREFIX_RE.finditer(raw):
        _set_counted_progress(
            signals,
            verb=match.group(1),
            current=match.group(2),
            total=match.group(3),
        )

    metrics: dict[str, float] = {}
    for match in _PROGRESS_METRIC_RE.finditer(raw):
        key = match.group(1).lower()
        try:
            metrics[key] = float(match.group(2))
        except ValueError:
            continue
    if metrics:
        merged_metrics = dict(signals.get("metrics") or {})
        merged_metrics.update(metrics)
        signals["metrics"] = merged_metrics
        _set_metric_evidence(signals, source="stdout_pattern")
    return signals


def _set_counted_progress(
    signals: dict[str, Any],
    *,
    verb: str,
    current: str,
    total: str,
) -> None:
    current_value = float(current)
    total_value = float(total)
    if total_value <= 0.0 or current_value > total_value:
        return
    signals["items"] = {
        "current": current_value,
        "total": total_value,
        "source": str(verb or "processed").lower(),
    }
    _set_progress_evidence(signals, source="stdout_pattern")


def _set_progress_evidence(signals: dict[str, Any], *, source: str) -> None:
    existing = signals.get("progress_evidence")
    if isinstance(existing, dict) and existing.get("source") == "scienceflow_hb":
        return
    signals["progress_evidence"] = {"source": source, "trust": AGENT_REPORTED}


def _set_metric_evidence(signals: dict[str, Any], *, source: str) -> None:
    existing = signals.get("metric_evidence")
    if isinstance(existing, dict) and existing.get("source") == "scienceflow_hb":
        return
    signals["metric_evidence"] = {"source": source, "trust": AGENT_REPORTED}


def _parse_scienceflow_heartbeat(text: str) -> dict[str, Any]:
    heartbeat: dict[str, Any] = {}
    for line in str(text or "").splitlines():
        if not _SCIENCEFLOW_HEARTBEAT_LINE_RE.search(line):
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        fields: dict[str, str] = {}
        for part in parts:
            if part == "SCIENCEFLOW_HB" or "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key.strip():
                fields[key.strip()] = value.strip()
        if fields.get("v") != "1":
            continue
        heartbeat = {
            "format": "SCIENCEFLOW_HB",
            "version": 1,
            "raw": line.strip()[:500],
            "evidence_trust": AGENT_REPORTED,
        }
        phase = fields.get("phase")
        if phase:
            heartbeat["phase"] = phase
        scoped_fields = {
            "route_id": fields.get("route_id") or fields.get("route"),
            "fold": fields.get("fold"),
            "validation_protocol": fields.get("validation_protocol") or fields.get("protocol"),
            "checkpoint_kind": fields.get("checkpoint_kind"),
            "safe_to_resume": fields.get("safe_to_resume"),
            "mergeable": fields.get("mergeable"),
        }
        for key, raw_value in scoped_fields.items():
            value = str(raw_value or "").strip()
            if value:
                heartbeat[key] = value[:160]
        tick = _float_or_none(fields.get("tick"))
        if tick is not None:
            heartbeat["tick"] = tick
        elapsed_s = _float_or_none(fields.get("elapsed_s"))
        if elapsed_s is not None:
            heartbeat["elapsed_s"] = elapsed_s
        unit = fields.get("unit")
        if unit:
            heartbeat["unit"] = unit
        progress = _parse_nf_progress(fields.get("progress"))
        if progress:
            heartbeat["progress"] = progress
        metrics: dict[str, float] = {}
        metric = _parse_nf_metric(fields.get("metric"))
        if metric:
            metrics.update(metric)
        loss = _float_or_none(fields.get("loss"))
        if loss is not None:
            metrics["loss"] = loss
        if metrics:
            heartbeat["metrics"] = metrics
        artifact = fields.get("artifact")
        if artifact and artifact.lower() not in {"none", "na", "n/a", "null", "-"}:
            heartbeat["artifact_path"] = artifact
    return heartbeat


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _parse_nf_progress(value: str | None) -> dict[str, Any]:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"na", "n/a", "none", "?", "-"}:
        return {}
    current_raw, sep, total_raw = raw.partition("/")
    current = _float_or_none(current_raw)
    if current is None:
        return {}
    progress: dict[str, Any] = {"current": current}
    if sep:
        total = _float_or_none(total_raw)
        if total is not None:
            progress["total"] = total
    return progress


def _parse_nf_metric(value: str | None) -> dict[str, float]:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"na", "n/a", "none", "?", "-"}:
        return {}
    name, sep, metric_value = raw.partition(":")
    if not sep:
        name, sep, metric_value = raw.partition("=")
    if not sep:
        return {}
    metric = _float_or_none(metric_value)
    if metric is None:
        return {}
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip().lower()).strip("_")
    return {key: metric} if key else {}
