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
import re
import time
from dataclasses import dataclass
from typing import Any

_LARGE_CODE_RE = re.compile(r"(?:^|[\s/])(?P<name>train|predict|util|solution)\.py\b", re.IGNORECASE)


@dataclass(frozen=True)
class ContextHygieneDecision:
    should_compact: bool
    reason: str
    facts: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"should_compact": self.should_compact, "reason": self.reason, "facts": dict(self.facts)}


def evaluate_context_hygiene_compact(
    *,
    stage_count_since_last_compact: int,
    total_input_tokens_since_last_compact: int,
    cache_rates: list[float],
    large_file_touch_counts: dict[str, int],
    large_tool_output_count: int,
    seconds_since_last_compact: float,
    active_high_risk_bash: bool,
    snapshot_can_preserve_current_state: bool,
    max_stages_without_compact: int = 25,
    code_churn_stage_threshold: int = 10,
    large_file_repeat_threshold: int = 10,
    low_incremental_cache_rate: float = 0.80,
    low_cache_window: int = 5,
    min_tokens_since_compact: int = 5_000_000,
    tool_output_min_interval_sec: float = 1800.0,
) -> ContextHygieneDecision:
    stage_count = max(0, int(stage_count_since_last_compact or 0))
    input_tokens = max(0, int(total_input_tokens_since_last_compact or 0))
    repeat_threshold = max(1, int(large_file_repeat_threshold or 1))
    repeated_files = {
        str(name): int(count)
        for name, count in (large_file_touch_counts or {}).items()
        if int(count or 0) >= repeat_threshold
    }
    code_churn_primary = bool(repeated_files) and stage_count >= max(1, int(code_churn_stage_threshold or 1))
    stage_count_primary = stage_count >= max(1, int(max_stages_without_compact or 1))
    primary_gate = bool(code_churn_primary or stage_count_primary)

    valid_rates = [float(x) for x in (cache_rates or []) if _valid_rate(x)]
    window = max(1, int(low_cache_window or 1))
    recent_rates = valid_rates[-window:]
    low_cache_avg = sum(recent_rates) / len(recent_rates) if recent_rates else None
    low_cache_evidence = bool(
        len(recent_rates) >= window
        and low_cache_avg is not None
        and low_cache_avg < float(low_incremental_cache_rate or 0.80)
        and input_tokens >= max(0, int(min_tokens_since_compact or 0))
    )
    repeated_tool_output_evidence = bool(
        int(large_tool_output_count or 0) >= 3
        and float(seconds_since_last_compact or 0.0) >= max(0.0, float(tool_output_min_interval_sec or 0.0))
    )
    same_file_churn_evidence = bool(repeated_files)
    evidence_gate = bool(low_cache_evidence or repeated_tool_output_evidence or same_file_churn_evidence)
    safety = {
        "no_active_high_risk_bash": not bool(active_high_risk_bash),
        "snapshot_can_preserve_current_state": bool(snapshot_can_preserve_current_state),
    }
    safety_gate = all(safety.values())
    facts = {
        "stage_count_since_last_compact": stage_count,
        "total_input_tokens_since_last_compact": input_tokens,
        "cache_rates_considered": recent_rates,
        "low_cache_avg": low_cache_avg,
        "large_file_touch_counts": dict(large_file_touch_counts or {}),
        "repeated_large_files": repeated_files,
        "large_tool_output_count": int(large_tool_output_count or 0),
        "seconds_since_last_compact": float(seconds_since_last_compact or 0.0),
        "primary_gate": primary_gate,
        "primary_reasons": [
            reason
            for reason, enabled in {
                "repeated_same_file_churn": code_churn_primary,
                "stage_count_threshold": stage_count_primary,
            }.items()
            if enabled
        ],
        "evidence_gate": evidence_gate,
        "evidence_reasons": [
            reason
            for reason, enabled in {
                "low_incremental_cache": low_cache_evidence,
                "repeated_tool_output": repeated_tool_output_evidence,
                "repeated_same_file_churn": same_file_churn_evidence,
            }.items()
            if enabled
        ],
        "safety_gate": safety_gate,
        "safety": safety,
    }
    if primary_gate and evidence_gate and safety_gate:
        return ContextHygieneDecision(True, "context_hygiene_low_cache_or_churn", facts)
    missing: list[str] = []
    if not primary_gate:
        missing.append("primary_gate")
    if not evidence_gate:
        missing.append("evidence_gate")
    if not safety_gate:
        missing.append("safety_gate")
    return ContextHygieneDecision(False, "missing_" + "_and_".join(missing), facts)


def large_code_file_touch_counts(messages: list[Any], *, max_messages: int = 240) -> dict[str, int]:
    counts: dict[str, int] = {}
    for msg in list(messages or [])[-max(1, int(max_messages or 1)):]:
        text = _message_text(msg)
        if not text:
            continue
        for match in _LARGE_CODE_RE.finditer(text):
            name = f"{match.group('name').lower()}.py"
            counts[name] = counts.get(name, 0) + 1
    return counts


def large_tool_output_count(messages: list[Any], *, min_chars: int = 12_000, max_messages: int = 240) -> int:
    count = 0
    for msg in list(messages or [])[-max(1, int(max_messages or 1)):]:
        role = str(getattr(msg, "role", "") or "")
        if role != "tool":
            continue
        if len(_message_text(msg)) >= max(1, int(min_chars or 1)):
            count += 1
    return count


def _message_text(msg: Any) -> str:
    chunks: list[str] = []
    content = getattr(msg, "content", "")
    if content is not None:
        chunks.append(str(content))
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        try:
            chunks.append(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True, default=str))
        except (TypeError, ValueError):
            chunks.append(str(tool_calls))
    return "\n".join(chunks)


def _valid_rate(value: Any) -> bool:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return out == out and 0.0 <= out <= 1.0


def now_seconds() -> float:
    return time.time()
