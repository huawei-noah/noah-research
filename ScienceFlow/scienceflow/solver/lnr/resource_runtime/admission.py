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

import hashlib
import json
import re
from typing import Any

from scienceflow.solver.lnr.resource_feedback_contract import resource_feedback_text


_ACTIONS = {"RUN_NOW", "OBSERVE_THEN_RUN", "PENDING", "REPLAN"}
_START_ACTIONS = {"RUN_NOW", "OBSERVE_THEN_RUN"}
_BLOCKED_ACTIONS = {"PENDING", "REPLAN", "DEFERRED", "DENIED_REPLAN", "DENIED_DUPLICATE"}


def should_request_admission_llm(rule_result: dict[str, Any], *, mode: str = "low_confidence") -> bool:
    """Return true when rule admission would benefit from an isolated LLM review."""
    if not isinstance(rule_result, dict) or not rule_result.get("enabled"):
        return False
    if rule_result.get("acquired"):
        return False
    status = str(rule_result.get("status") or rule_result.get("admission_action") or "").upper()
    if status not in _BLOCKED_ACTIONS:
        return False
    mode = str(mode or "low_confidence").strip().lower()
    if mode in {"off", "false", "none"}:
        return False
    if mode in {"always", "all", "blocked_states", "blocked", "soft_gates"}:
        return True
    if rule_result.get("requires_admission_review"):
        return True
    if rule_result.get("admission_cached_backoff"):
        return False
    opportunity = rule_result.get("admission_opportunity") if isinstance(rule_result.get("admission_opportunity"), dict) else {}
    if opportunity.get("lease_grantable_by_llm") or opportunity.get("partial_gpu_candidates"):
        return True
    if rule_result.get("lease_grantable_by_llm"):
        return True
    reason = str(rule_result.get("reason") or "")
    priority = _float(rule_result.get("admission_priority_score"))
    duplicate = rule_result.get("duplicate_digest_summary") if isinstance(rule_result.get("duplicate_digest_summary"), dict) else {}
    duplicate_count = int(duplicate.get("duplicate_count") or 0)
    queue_position = int(rule_result.get("queue_position") or 0)
    requested_multi_gpu = int((rule_result.get("details") or {}).get("request_count") or 0) > 1
    if status == "REPLAN":
        return priority >= 0.35 or reason in {"duplicate_gpu_command_under_contention", "multi_gpu_pending_wait_exceeded"}
    if reason in {"gpu_memory_below_admission_reserve", "admission_waiter_ahead"}:
        return priority >= 0.5
    if duplicate_count > 0:
        return priority >= 0.35
    if requested_multi_gpu:
        return True
    if queue_position and queue_position <= 2 and priority >= 0.25:
        return True
    return False


def build_resource_admission_prompt(task_card: dict[str, Any], rule_result: dict[str, Any]) -> str:
    card_json = _compact_json(task_card, max_chars=9000)
    rule_json = _compact_json(_public_rule_result(rule_result), max_chars=5000)
    return (
        "You are ResourceAdmissionArbiter. Decide whether this GPU task should start now, remain pending, "
        "or be rejected for replan.\n"
        "Return exactly one JSON object and nothing else. Do not call tools, write files, use markdown, "
        "or include hidden chain-of-thought.\n\n"
        "Allowed actions:\n"
        "- RUN_NOW: start now after the runtime can atomically acquire the selected GPU lease.\n"
        "- OBSERVE_THEN_RUN: start now with a short early runtime review when footprint confidence is low.\n"
        "- PENDING: the task has value but should wait without starting bash or holding GPU.\n"
        "- REPLAN: the task should not wait; the main agent should change the command or plan.\n\n"
        "Hard constraints:\n"
        "- Never approve RUN_NOW/OBSERVE_THEN_RUN when lease_grantable_by_llm is false.\n"
        "- For partial GPU capacity, prefer OBSERVE_THEN_RUN when free memory appears sufficient but footprint confidence is low.\n"
        "- Pick gpu_ids only from candidate_physical_gpus or partial_gpu_candidates.\n"
        "- Prefer PENDING over REPLAN for non-duplicate valuable work with enough budget.\n"
        "- Prefer REPLAN for duplicate heavy GPU commands under contention or work unlikely to finish.\n"
        "- Keep the reason to one short sentence.\n\n"
        "Output schema:\n"
        '{"action":"PENDING","reason":"...","confidence":"low|medium|high","pending_ttl_sec":300,"gpu_ids":["0"],"observe_sec":180}\n\n'
        "ResourceTaskCard:\n"
        f"{card_json}\n\n"
        "Rule admission result:\n"
        f"{rule_json}\n"
    )


def parse_admission_decision_text(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def normalize_admission_decision(
    raw: dict[str, Any],
    *,
    task_card: dict[str, Any],
    rule_result: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    rule_status = str(rule_result.get("status") or rule_result.get("admission_action") or "PENDING").upper()
    if rule_status == "GRANTED":
        rule_status = "RUN_NOW"
    action = str(raw.get("action") or raw.get("admission_action") or "").strip().upper()
    if action not in _ACTIONS:
        action = rule_status if rule_status in _ACTIONS else "PENDING"
    reason = _one_line(raw.get("reason") or rule_result.get("reason") or "resource_admission_review")
    acquired = bool(rule_result.get("acquired"))
    opportunity = task_card.get("admission_opportunity") if isinstance(task_card.get("admission_opportunity"), dict) else {}
    lease_grantable = bool(acquired or rule_result.get("lease_grantable_by_llm") or opportunity.get("lease_grantable_by_llm"))
    if action in _START_ACTIONS and not lease_grantable:
        action = "PENDING"
        reason = "LLM requested GPU start, but no atomic lease grant path was available; keeping task pending"
    confidence = str(raw.get("confidence") or "medium").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    pending_ttl_sec = _float(raw.get("pending_ttl_sec"), default=0.0)
    decision_id_seed = f"{task_card.get('job_id','')}:{action}:{reason}:{source}"
    decision_id = hashlib.sha1(decision_id_seed.encode("utf-8", errors="replace")).hexdigest()[:12]
    return {
        "action": action,
        "status": action if action not in _START_ACTIONS else "GRANTED",
        "admission_action": action,
        "reason": reason,
        "confidence": confidence,
        "pending_ttl_sec": max(0.0, pending_ttl_sec),
        "source": str(source or "resource_admission_llm"),
        "decision_id": "ad_" + decision_id,
        "rule_status": rule_status,
        "rule_reason": str(rule_result.get("reason") or ""),
        "gpu_ids": _selected_gpu_ids(raw, task_card=task_card, rule_result=rule_result),
        "observe_sec": max(0.0, _float(raw.get("observe_sec"), default=0.0)),
        "lease_grantable_by_llm": lease_grantable,
    }


def apply_admission_decision(rule_result: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    out = dict(rule_result or {})
    action = str(decision.get("admission_action") or decision.get("action") or "").upper()
    if action in _START_ACTIONS and not out.get("acquired"):
        action = "PENDING"
    status = "GRANTED" if action in _START_ACTIONS else action
    reason = str(decision.get("reason") or out.get("reason") or "resource_admission_review")
    out.update(
        {
            "status": status,
            "admission_action": action,
            "reason": reason,
            "admission_llm_reviewed": True,
            "admission_llm_source": decision.get("source") or "resource_admission_llm",
            "admission_llm_confidence": decision.get("confidence") or "",
            "admission_llm_decision_id": decision.get("decision_id") or "",
        }
    )
    if decision.get("admission_action") == "OBSERVE_THEN_RUN" and out.get("acquired"):
        out["admission_observe_then_run"] = True
        out["admission_observe_sec"] = max(0.0, _float(decision.get("observe_sec"), default=0.0))
    if action in {"PENDING", "REPLAN"}:
        out["feedback"] = admission_feedback_from_result(out)
    return out


def admission_feedback_from_result(result: dict[str, Any]) -> str:
    action = str(result.get("admission_action") or result.get("status") or "PENDING").upper()
    if action == "GRANTED":
        action = "RUN_NOW"
    holders = [str(x) for x in (result.get("holder_job_ids") or []) if str(x).strip()]
    extra_facts = {
        "admission_priority_score": "%.4f" % _float(result.get("admission_priority_score")) if result.get("admission_priority_score") is not None else "",
        "llm_confidence": result.get("admission_llm_confidence") or "",
    }
    if result.get("retry_after_sec") is not None:
        extra_facts["retry_after_sec"] = "%.0f" % max(0.0, _float(result.get("retry_after_sec")))
    if result.get("post_feedback_action"):
        extra_facts["post_feedback_action"] = str(result.get("post_feedback_action") or "")
    if result.get("admission_cached_backoff"):
        extra_facts["cached_backoff"] = "true"
    return resource_feedback_text(
        status=action,
        reason=_one_line(result.get("reason") or "resource_admission"),
        scope="per_gpu",
        resource_mode=str(result.get("resource_mode") or "YELLOW"),
        blocked_class=str(result.get("policy_resource_class") or result.get("resource_class") or "gpu_task"),
        gpu_ids=[str(x) for x in (result.get("gpu_ids") or []) if str(x).strip()],
        allowed_classes=[str(x) for x in (result.get("allowed_classes") or []) if str(x).strip()],
        holder_job_id=holders[0] if holders else "",
        queue_position=(int(result.get("queue_position") or 0) if result.get("queue_position") is not None else None),
        queue_len=(int(result.get("queue_len") or 0) if result.get("queue_len") is not None else None),
        eta_next_train_sec=(max(0.0, _float(result.get("eta_next_train_sec"))) if result.get("eta_next_train_sec") is not None else None),
        eta_confidence=str(result.get("eta_confidence") or ""),
        unlock_condition="holder_released" if holders else "resource_slot_available",
        blocked_until_unlock=True,
        extra_facts=extra_facts,
    )


def _public_rule_result(rule_result: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "status",
        "admission_action",
        "reason",
        "resource_mode",
        "gpu_ids",
        "assigned_physical_gpus",
        "allowed_physical_gpus",
        "candidate_physical_gpus",
        "requested_gpu_count",
        "resource_class",
        "policy_resource_class",
        "slot_weight",
        "capacity_slots",
        "queue_position",
        "queue_len",
        "top_waiter_job_id",
        "eta_next_train_sec",
        "eta_confidence",
        "eta_source",
        "allowed_classes",
        "admission_priority_score",
        "expected_value_score",
        "lineage_diversity_score",
        "near_submission_score",
        "worker_starvation_score",
        "duplicate_penalty",
        "timeout_history_penalty",
        "long_runtime_penalty",
        "duplicate_digest_summary",
        "safety",
        "admission_opportunity",
        "lease_grantable_by_llm",
        "requires_admission_review",
        "soft_gate_reason",
        "soft_gate_scope",
        "soft_gate",
        "blocked_until_unlock",
        "unlock_condition",
        "feedback_state_key",
        "resource_feedback_repeated_count",
    }
    return {key: value for key, value in dict(rule_result or {}).items() if key in allowed}


def _compact_json(value: Any, *, max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 32)].rstrip() + "...[truncated]"


def _one_line(value: Any, *, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def _selected_gpu_ids(raw: dict[str, Any], *, task_card: dict[str, Any], rule_result: dict[str, Any]) -> list[str]:
    selected = raw.get("gpu_ids") or raw.get("selected_gpu_ids") or []
    ids = [str(x).strip() for x in selected if str(x).strip()] if isinstance(selected, list) else []
    allowed = _candidate_gpu_set(task_card, rule_result)
    out: list[str] = []
    seen: set[str] = set()
    for gpu_id in ids:
        if allowed and gpu_id not in allowed:
            continue
        if gpu_id not in seen:
            seen.add(gpu_id)
            out.append(gpu_id)
    return out


def _candidate_gpu_set(task_card: dict[str, Any], rule_result: dict[str, Any]) -> set[str]:
    values: list[Any] = []
    for container in (task_card, rule_result):
        for key in ("candidate_physical_gpus", "allowed_physical_gpus", "gpu_ids"):
            raw = container.get(key) if isinstance(container, dict) else []
            if isinstance(raw, list):
                values.extend(raw)
        opportunity = container.get("admission_opportunity") if isinstance(container, dict) and isinstance(container.get("admission_opportunity"), dict) else {}
        raw_candidates = opportunity.get("partial_gpu_candidates") if isinstance(opportunity, dict) else []
        if isinstance(raw_candidates, list):
            for row in raw_candidates:
                if isinstance(row, dict):
                    values.append(row.get("gpu_id"))
                else:
                    values.append(row)
    return {str(x).strip() for x in values if str(x).strip()}
