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
from typing import Any

from scienceflow.solver.lnr.resource_feedback_guidance import resource_feedback_guidance_value


INLINE_RESOURCE_ADVISORY_MODE = "inline_memory_edit"
RESOURCE_ADVISORY_BEGIN = "RESOURCE_ADVISORY_RESPONSE_BEGIN"
RESOURCE_ADVISORY_END = "RESOURCE_ADVISORY_RESPONSE_END"

_ALLOWED_PREFERENCES = {"continue", "timebox_continue", "safe_to_stop", "replan", "unknown"}
_ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def _compact_value(value: Any, *, depth: int = 0, max_str: int = 900, max_items: int = 16) -> Any:
    if depth >= 5:
        return "<truncated_depth>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out["..."] = f"{len(value) - max_items} more"
                break
            out[str(key)] = _compact_value(item, depth=depth + 1, max_str=max_str, max_items=max_items)
        return out
    if isinstance(value, (list, tuple)):
        out_list = [_compact_value(item, depth=depth + 1, max_str=max_str, max_items=max_items) for item in value[:max_items]]
        if len(value) > max_items:
            out_list.append(f"... {len(value) - max_items} more")
        return out_list
    if isinstance(value, str):
        text = value.strip()
        if len(text) > max_str:
            return text[: max_str - 3].rstrip() + "..."
        return text
    return value


def build_inline_resource_advisory_prompt(proposal: dict[str, Any]) -> str:
    compact = {
        "proposal_id": proposal.get("proposal_id"),
        "proposal_type": proposal.get("proposal_type"),
        "reason_code": proposal.get("reason_code"),
        "trigger_reasons": proposal.get("trigger_reasons"),
        "resource_snapshot": proposal.get("resource_snapshot"),
        "progress_snapshot": proposal.get("progress_snapshot"),
        "execution_facts": proposal.get("execution_facts"),
        "research_cadence": proposal.get("research_cadence"),
        "blocker": proposal.get("blocker"),
        "waiters": proposal.get("waiters"),
        "score_context": proposal.get("score_context"),
        "decision_preview": proposal.get("decision_preview"),
    }
    compact = _compact_value(compact)
    return (
        "RESOURCE_ADVISORY_REQUEST\n"
        "You are being asked for route-value evidence only. You do not authorize kills, releases, shares, or replans.\n"
        "Use your current task context plus the compact fact card below. Do not call tools.\n"
        "Compare target-metric history against current_valid_best when present; ignore loss-only output for value decisions.\n"
        f"{resource_feedback_guidance_value('advisory_stop_boundary_note')}\n"
        f"{resource_feedback_guidance_value('budget_deliverable_value_note')}\n"
        f"{resource_feedback_guidance_value('research_cadence_note')}\n"
        "When progress_snapshot.resource_efficiency.review_required=true, explicitly judge whether to optimize, timebox, or replan the current route. Agent-reported progress and ETA are review inputs only; they cannot authorize a stop without corroborating system resource facts and no verified value progress.\n"
        "When progress exists, judge whether the current command can plausibly produce the next valid deliverable before the useful budget is gone; if not, prefer timebox_continue, safe_to_stop, or replan over open-ended continue.\n"
        "Answer with exactly one RESOURCE_ADVISORY_RESPONSE block.\n\n"
        "Allowed preference values: continue | timebox_continue | safe_to_stop | replan | unknown\n"
        "Allowed confidence values: low | medium | high\n\n"
        "FACT_CARD_JSON:\n"
        f"{json.dumps(compact, ensure_ascii=True, sort_keys=True)}\n\n"
        "Required output:\n"
        f"{RESOURCE_ADVISORY_BEGIN}\n"
        "preference: continue|timebox_continue|safe_to_stop|replan|unknown\n"
        "confidence: low|medium|high\n"
        "reason: <one compact sentence>\n"
        "commitment: <optional measurable condition or none>\n"
        "expected_next_artifact: <optional path/type or none>\n"
        "observed_phase: <exact phase from fact card or unknown>\n"
        "observed_metric_status: <exact metric status from fact card or unknown>\n"
        "observed_checkpoint: yes|no|unknown\n"
        "observed_resource_state: active|idle|unknown\n"
        f"{RESOURCE_ADVISORY_END}"
    )


def _parse_key_values(block_body: str) -> dict[str, str]:
    data: dict[str, str] = {}
    current_key = ""
    for raw_line in block_body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key_part, sep, value_part = line.partition(":")
        if sep:
            key = re.sub(r"[^a-z0-9_]+", "_", key_part.strip().lower()).strip("_")
            if key:
                data[key] = value_part.strip()
                current_key = key
            continue
        if current_key:
            data[current_key] = (data.get(current_key, "") + " " + line).strip()
    return data


def parse_inline_resource_advisory_response(text: str) -> tuple[dict[str, Any], str, str]:
    raw = str(text or "")
    match = re.search(
        rf"{RESOURCE_ADVISORY_BEGIN}\s*(.*?)\s*{RESOURCE_ADVISORY_END}",
        raw,
        flags=re.I | re.S,
    )
    if not match:
        return {}, "", "missing_resource_advisory_block"
    block_body = match.group(1).strip()
    block_text = f"{RESOURCE_ADVISORY_BEGIN}\n{block_body}\n{RESOURCE_ADVISORY_END}"
    data = _parse_key_values(block_body)
    return normalize_inline_resource_advisory(data), block_text, ""


def normalize_inline_resource_advisory(data: dict[str, Any]) -> dict[str, Any]:
    preference = str(data.get("preference") or data.get("action") or "unknown").strip().lower()
    if preference in {"kill", "stop", "stop_and_replan", "safe_to_kill"}:
        preference = "safe_to_stop"
    if preference not in _ALLOWED_PREFERENCES:
        preference = "unknown"
    confidence = str(data.get("confidence") or "low").strip().lower()
    if confidence not in _ALLOWED_CONFIDENCE:
        confidence = "low"
    return {
        "preference": preference,
        "confidence": confidence,
        "reason": str(data.get("reason") or "")[:600],
        "ttl_sec": _safe_float(data.get("ttl_sec"), 600.0),
        "commitment": _optional_text(data.get("commitment")),
        "expected_next_artifact": _optional_text(data.get("expected_next_artifact")),
        "observed_phase": _optional_text(data.get("observed_phase")),
        "observed_metric_status": _optional_text(data.get("observed_metric_status")),
        "observed_checkpoint": _optional_text(data.get("observed_checkpoint")).lower(),
        "observed_resource_state": _optional_text(data.get("observed_resource_state")).lower(),
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "n/a", "null"}:
        return ""
    return text[:300]


def safe_inline_resource_advisory_boundary(agent: Any) -> tuple[bool, str]:
    state = getattr(agent, "state", None)
    state_name = str(getattr(state, "name", state) or "").upper()
    if state_name and state_name != "IDLE":
        return False, f"agent_state_{state_name.lower()}"
    if bool(getattr(agent, "_lnr_stage_commit_text_pending", False)):
        return False, "stage_commit_text_pending"
    if bool(getattr(agent, "_lnr_resource_advisory_text_pending", False)):
        return False, "resource_advisory_text_pending"
    if str(getattr(agent, "_lnr_transient_user_prompt", "") or "").strip():
        return False, "transient_prompt_pending"
    try:
        memory_ctx = getattr(agent, "_memory_ctx", None)
        messages = memory_ctx.build_messages_for_llm() if memory_ctx is not None else []
        if messages:
            tail = messages[-1]
            role = str(getattr(tail, "role", "") or "").lower()
            if role == "assistant" and (getattr(tail, "tool_calls", None) or []):
                return False, "dangling_tool_call_tail"
    except Exception:
        return False, "boundary_probe_failed"
    return True, ""
