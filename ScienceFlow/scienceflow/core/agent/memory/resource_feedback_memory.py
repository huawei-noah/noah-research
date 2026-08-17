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

"""RESOURCE_FEEDBACK reduction for agent memory hygiene.

Raw resource events remain append-only in audit/tool-output logs. Repeated
resource feedback is folded out of main-agent chat memory so resource control
does not pollute the research trace. ``summary_text()`` is retained for
diagnostics and legacy callers, but run-loop memory no longer receives a live
resource state slot.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass

RESOURCE_STATE_SUMMARY_MARKER = "[RESOURCE_STATE_SUMMARY_SLOT]"


_RESOURCE_FEEDBACK_RE = re.compile(
    r"RESOURCE_FEEDBACK:\s*(?P<status>[A-Za-z0-9_:-]+)\s+because\s+(?P<reason>[^;\n.]+)",
)
_RESOURCE_RESEARCH_SIGNAL_RE = re.compile(r"RESOURCE_RESEARCH_SIGNAL:\s*(?P<body>[^\n]+)")
_KV_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_:-]{1,64})\s*[=:]\s*([^;\n]+)")

_RESEARCH_STATE_KEYS = set(
    """
    valid_best valid_best_score current_valid_best current_valid_best_status best_score best_validity
    submission_validity valid_submission route_viability route_viability_proven produced_invalid
    timebox_id timebox_state timebox_deadline_sec deadline_sec intervention_action
    resource_intervention_action stage_id current_stage
    outcome confidence review_facts resource_facts progress_facts budget_facts value_facts execution_focus
    """.split()
)


@dataclass
class ResourceFeedbackRecord:
    status: str
    reason: str
    blocked_class: str
    allowed_classes: str
    resource_mode: str
    feedback_scope: str
    gpu_ids: str
    holder_job_id: str
    eta_bucket: str
    eta_confidence: str
    unlock_condition: str
    blocked_until_unlock: str
    schema_state: str
    artifact_state: str
    progress_state: str
    wait_state: str
    wait_tool: str
    wait_token: str
    wait_max_sec: str
    wait_reason: str
    resource_wait_allowed: str
    retry_allowed: str
    same_command_retry_allowed: str
    research_fingerprint: tuple[tuple[str, str], ...]

    @property
    def dedup_key(self) -> tuple[str, ...]:
        if self.feedback_scope == "worker_deliverable":
            return (
                self.status,
                self.reason,
                self.allowed_classes,
                self.resource_mode,
                self.feedback_scope,
                self.unlock_condition,
                self.blocked_until_unlock,
                self.schema_state,
                self.artifact_state,
                self.progress_state,
                self.wait_state,
                self.resource_wait_allowed,
                self.retry_allowed,
                self.same_command_retry_allowed,
            )
        return (
            self.status,
            self.reason,
            self.blocked_class,
            self.allowed_classes,
            self.resource_mode,
            self.feedback_scope,
            self.gpu_ids,
            self.holder_job_id,
            self.eta_bucket,
            self.eta_confidence,
            self.unlock_condition,
            self.blocked_until_unlock,
            self.schema_state,
            self.artifact_state,
            self.progress_state,
            self.wait_state,
            self.resource_wait_allowed,
            self.retry_allowed,
            self.same_command_retry_allowed,
        )


@dataclass
class _FeedbackState:
    record: ResourceFeedbackRecord
    research_fingerprint: tuple[tuple[str, str], ...]
    repeat_count: int = 1


class ResourceFeedbackMemoryDeduper:
    """Fold repeated resource feedback out of main-agent chat memory."""

    def __init__(self, *, max_keys: int = 128) -> None:
        self._max_keys = max(8, int(max_keys))
        self._states: OrderedDict[tuple[str, ...], _FeedbackState] = OrderedDict()

    def reduce(self, feedback: str) -> tuple[str, bool]:
        records = parse_resource_feedback_records(feedback)
        if not records:
            return feedback, False

        for record in records:
            self._reduce_record(record)
        return "", True

    def _reduce_record(self, record: ResourceFeedbackRecord) -> None:
        key = record.dedup_key
        state = self._states.get(key)
        if state is None or state.research_fingerprint != record.research_fingerprint:
            self._states[key] = _FeedbackState(record, record.research_fingerprint)
            self._states.move_to_end(key)
            self._trim()
            return

        state.record = record
        state.repeat_count += 1
        self._states.move_to_end(key)

    def summary_text(self, *, max_entries: int = 8) -> str:
        states = list(self._states.values())[-max(1, int(max_entries)):]
        if not states:
            return ""
        lines = [RESOURCE_STATE_SUMMARY_MARKER, "RESOURCE_STATE_SUMMARY:"]
        for state in states:
            lines.append(resource_state_summary_line(state.record, state.repeat_count))
        return "\n".join(lines).rstrip() + "\n"

    def _trim(self) -> None:
        while len(self._states) > self._max_keys:
            self._states.popitem(last=False)

def parse_resource_feedback_record(feedback: str) -> ResourceFeedbackRecord | None:
    records = parse_resource_feedback_records(feedback)
    return records[0] if records else None

def parse_resource_feedback_records(feedback: str) -> list[ResourceFeedbackRecord]:
    text = str(feedback or "")
    markers: list[tuple[int, str, re.Match[str]]] = []
    markers.extend((match.start(), "feedback", match) for match in _RESOURCE_FEEDBACK_RE.finditer(text))
    markers.extend((match.start(), "research", match) for match in _RESOURCE_RESEARCH_SIGNAL_RE.finditer(text))
    if not markers:
        return []
    markers.sort(key=lambda item: item[0])
    records: list[ResourceFeedbackRecord] = []
    for index, (start, kind, match) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        segment = text[start:end]
        if kind == "feedback":
            records.append(_resource_feedback_record_from_match(segment, match))
        else:
            records.append(_resource_research_signal_record_from_match(segment, match))
    return records

def _resource_research_signal_record_from_match(segment: str, match: re.Match[str]) -> ResourceFeedbackRecord:
    body = match.group("body") if "body" in match.groupdict() else segment
    kv = _parse_key_values(body)
    outcome = _clean_token(kv.get("outcome") or kv.get("action") or "resource_review").upper()
    return ResourceFeedbackRecord(
        status="RESEARCH_SIGNAL",
        reason=outcome,
        blocked_class="unknown",
        allowed_classes="",
        resource_mode="SIGNAL",
        feedback_scope="research",
        gpu_ids="",
        holder_job_id="",
        eta_bucket="",
        eta_confidence="",
        unlock_condition="",
        blocked_until_unlock="",
        schema_state="",
        artifact_state="",
        progress_state=_clean_token(kv.get("progress_facts") or kv.get("outcome") or ""),
        wait_state="unknown",
        wait_tool="",
        wait_token="",
        wait_max_sec="",
        wait_reason="",
        resource_wait_allowed="",
        retry_allowed="",
        same_command_retry_allowed="",
        research_fingerprint=tuple(
            sorted(
                (key, _clean_value(value))
                for key, value in kv.items()
                if key in _RESEARCH_STATE_KEYS and _clean_value(value)
            ),
        ),
    )


def _resource_feedback_record_from_match(segment: str, match: re.Match[str]) -> ResourceFeedbackRecord:
    kv = _parse_key_values(segment)
    return ResourceFeedbackRecord(
        status=_clean_token(match.group("status")).upper(),
        reason=_clean_value(match.group("reason")),
        blocked_class=_clean_token(kv.get("blocked") or kv.get("blocked_class") or "unknown"),
        allowed_classes=_clean_csv(kv.get("allowed") or kv.get("allowed_classes") or ""),
        resource_mode=_clean_token(kv.get("mode") or kv.get("resource_mode") or "UNKNOWN").upper(),
        feedback_scope=_clean_token(kv.get("scope") or kv.get("feedback_scope") or "task"),
        gpu_ids=_clean_csv(kv.get("gpu") or kv.get("gpu_ids") or ""),
        holder_job_id=_clean_token(kv.get("holder") or kv.get("holder_job_id") or ""),
        eta_bucket=_clean_token(kv.get("eta_bucket") or ""),
        eta_confidence=_clean_token(kv.get("eta_confidence") or ""),
        unlock_condition=_clean_token(kv.get("unlock_condition") or ""),
        blocked_until_unlock=_clean_bool(kv.get("blocked_until_unlock") or ""),
        schema_state=_clean_token(kv.get("schema_state") or ""),
        artifact_state=_clean_token(kv.get("artifact_state") or ""),
        progress_state=_clean_token(kv.get("progress_state") or ""),
        wait_state=_wait_state(kv),
        wait_tool=_clean_token(kv.get("wait_tool") or ""),
        wait_token=_clean_token(kv.get("wait_token") or ""),
        wait_max_sec=_clean_numeric_token(kv.get("wait_max_sec") or ""),
        wait_reason=_clean_token(kv.get("wait_reason") or ""),
        resource_wait_allowed=_clean_bool(kv.get("resource_wait_allowed") or ""),
        retry_allowed=_clean_bool(kv.get("retry_allowed") or ""),
        same_command_retry_allowed=_clean_bool(kv.get("same_command_retry_allowed") or ""),
        research_fingerprint=tuple(
            sorted(
                (key, _clean_value(value))
                for key, value in kv.items()
                if key in _RESEARCH_STATE_KEYS and _clean_value(value)
            ),
        ),
    )

def resource_state_summary_line(record: ResourceFeedbackRecord, repeat_count: int) -> str:
    details = [
        f"status={record.status}",
        f"mode={record.resource_mode}",
        f"repeat_count={max(1, int(repeat_count))}",
    ]
    for key, value in (
        ("scope", record.feedback_scope),
        ("blocked", record.blocked_class),
        ("allowed", record.allowed_classes),
        ("gpu", record.gpu_ids),
        ("holder", record.holder_job_id),
        ("eta_bucket", record.eta_bucket),
        ("eta_confidence", record.eta_confidence),
        ("unlock_condition", record.unlock_condition),
        ("blocked_until_unlock", record.blocked_until_unlock),
        ("schema_state", record.schema_state),
        ("artifact_state", record.artifact_state),
        ("progress_state", record.progress_state),
        ("wait", record.wait_state),
        ("wait_tool", record.wait_tool),
        ("wait_token", record.wait_token),
        ("wait_max_sec", record.wait_max_sec),
        ("wait_reason", record.wait_reason),
        ("resource_wait_allowed", record.resource_wait_allowed),
        ("retry_allowed", record.retry_allowed),
        ("same_command_retry_allowed", record.same_command_retry_allowed),
        *record.research_fingerprint,
    ):
        if value and value != "unknown":
            details.append(f"{key}={value}")
    return f"- {record.reason}: {'; '.join(details)}."


def summarize_repeated_resource_feedback(record: ResourceFeedbackRecord, repeat_count: int) -> str:
    return RESOURCE_STATE_SUMMARY_MARKER + "\nRESOURCE_STATE_SUMMARY:\n" + resource_state_summary_line(record, repeat_count) + "\n"


# Backward-compatible name for tests and older callers that import it directly.
compact_repeated_resource_feedback = summarize_repeated_resource_feedback

def _parse_key_values(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in _KV_RE.finditer(text):
        key = _clean_token(match.group(1)).lower()
        value = _clean_value(match.group(2))
        if key and value:
            out[key] = value
    return out


def _clean_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip()).strip("_") or "unknown"


def _clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().rstrip(".")).strip()


def _clean_bool(value: str) -> str:
    token = _clean_token(value).lower()
    return token if token in {"true", "false"} else ""


def _clean_numeric_token(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "unknown"
    try:
        number = float(raw)
    except ValueError:
        return _clean_token(raw)
    if number != number:
        return "unknown"
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _clean_csv(value: str) -> str:
    tokens = [_clean_token(part) for part in str(value or "").split(",") if str(part).strip()]
    tokens = [token for token in tokens if token != "unknown"]
    return ",".join(sorted(dict.fromkeys(tokens))) or "unknown"


def _wait_state(kv: dict[str, str]) -> str:
    allowed = _clean_bool(kv.get("resource_wait_allowed") or "")
    if allowed == "false":
        return "unavailable"
    action_options = {part.strip() for part in str(kv.get("action_options") or "").split(",")}
    if (
        _clean_token(kv.get("wait_tool") or "") == "resource_wait"
        or _clean_token(kv.get("wait_token") or "") != "unknown"
        or "resource_wait" in action_options
    ):
        return "available"
    if allowed == "true":
        return "available"
    return "unknown"
