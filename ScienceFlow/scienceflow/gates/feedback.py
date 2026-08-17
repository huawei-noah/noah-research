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

"""Agent-visible evaluator feedback formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _event_value(event: Mapping[str, Any], key: str, default: object = "") -> object:
    if key in event:
        value = event.get(key, default)
        if value not in (None, ""):
            return value
    extra = event.get("extra")
    if isinstance(extra, Mapping) and key in extra:
        return extra.get(key, default)
    return default


def _compact_text(value: object, *, max_chars: int = 900) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    compact = "\n".join(lines).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def _bool_label(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def format_invalid_evaluator_feedback(
    metric_event: Mapping[str, Any] | None,
    *,
    candidate_artifact: str = "",
) -> str:
    """Format invalid artifact feedback without MLEBench-specific terminology."""
    if not isinstance(metric_event, Mapping):
        return ""
    backend = str(metric_event.get("evaluator_backend") or "").strip()
    if not backend:
        return ""

    artifact = str(metric_event.get("artifact_path") or candidate_artifact or "").strip()
    status = str(metric_event.get("evaluator_status") or "unknown").strip()
    reason = str(
        metric_event.get("gate_reason_code")
        or metric_event.get("metric_validity_reason_code")
        or _event_value(metric_event, "reason_code")
        or ""
    ).strip()
    message = str(
        metric_event.get("gate_message")
        or metric_event.get("metric_note")
        or metric_event.get("metric_source_note")
        or _event_value(metric_event, "message")
        or ""
    ).strip()
    stderr_tail = _compact_text(
        _event_value(metric_event, "stderr_tail")
        or _event_value(metric_event, "evaluator_stderr_tail"),
    )
    stdout_tail = _compact_text(
        _event_value(metric_event, "stdout_tail")
        or _event_value(metric_event, "evaluator_stdout_tail"),
    )
    detail = stderr_tail or stdout_tail

    lines = [
        "EVALUATOR_INVALID_ARTIFACT",
        f"- artifact: {artifact or 'unknown'}",
        f"- evaluator_backend: {backend}",
        f"- status: {status}",
        f"- candidate_ready: {_bool_label(metric_event.get('candidate_ready'))}",
        f"- selection_eligible: {_bool_label(metric_event.get('selection_eligible'))}",
    ]
    if reason:
        lines.append(f"- reason: {reason}")
    if message:
        lines.append(f"- message: {_compact_text(message, max_chars=300)}")
    if detail:
        label = "stderr_tail" if stderr_tail else "stdout_tail"
        lines.append(f"- {label}: {_compact_text(detail, max_chars=700)}")
    lines.append(
        "- required_fix: produce a valid candidate artifact that satisfies the task "
        "evaluator, then rerun the configured evaluator before treating the result "
        "as progress."
    )
    return "\n".join(lines)
