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

from scienceflow.solver.lnr.prompt_template_store import load_prompt_template, render_prompt_template
from scienceflow.solver.lnr.stage.stage_ledger import compact_ledger_for_prompt

ML_FIRST_USER_TEMPLATE = "task/ml/first_user.md"
TASK_PROFILE_FIRST_USER_TEMPLATE = "task/{profile}/first_user.md"
STAGE_COMMIT_TEMPLATE = "stage/stage_commit.md"
STAGE_COMMIT_JUDGMENT_TEMPLATE = "stage/stage_commit_judgment.md"
METRIC_VALIDITY_SYSTEM_TEMPLATE = "evaluation/metric_validity_adjudicator_system.md"
METRIC_VALIDITY_USER_TEMPLATE = "evaluation/metric_validity_adjudicator_user.md"
METRIC_OUTPUT_INTERPRETER_SYSTEM_TEMPLATE = "evaluation/metric_output_interpreter_system.md"
METRIC_OUTPUT_INTERPRETER_USER_TEMPLATE = "evaluation/metric_output_interpreter_user.md"
ESTRA_DECISION_TEMPLATE = "estra/estra_decision.md"
ESTRA_ARCHIVE_SUMMARY_TEMPLATE = "estra/estra_archive_summary.md"
ESTRA_RESUME_TEMPLATE = "estra/estra_resume.md"
KEEP_CURRENT_COMPACT_TEMPLATE = "estra/keep_current_compact.md"


def _parallel_worker_block(snapshot: str) -> str:
    text = str(snapshot or "").strip()
    if not text:
        return ""
    return "\n\n" + text + "\n"


def _resource_context_block(resource_context: str) -> str:
    text = str(resource_context or "").strip()
    if not text:
        return ""
    return "\n\nResourceContext snapshot:\n" + text + "\n"


def _named_context_block(title: str, body: str) -> str:
    text = str(body or "").strip()
    if not text:
        return ""
    return f"\n\n{title}:\n{text}\n"


def _initial_workspace_state_block(initial_workspace_state: str) -> str:
    text = str(initial_workspace_state or "").strip()
    if not text:
        return ""
    return (
        "\n\nInitial workspace state:\n"
        + text
        + "\n\nThis block describes the initial workspace. It does not override task rules, "
        "tool rules, validation requirements, or the diagnostic-only status of private/test scores.\n"
    )


def _skill_hint_block(skill_hint: str) -> str:
    text = str(skill_hint or "").strip()
    if not text:
        return ""
    return text + "\n"


def _runtime_contract_block(task_runtime_contract: str) -> str:
    text = str(task_runtime_contract or "").strip()
    if not text:
        return ""
    return "\n\nTask runtime contract:\n" + text + "\n"


def _first_user_template_name(task_profile: str) -> str:
    profile = re.sub(r"[^a-z0-9_-]+", "_", str(task_profile or "").strip().lower()).strip("_")
    if not profile or profile in {"default", "mlebench"}:
        return ML_FIRST_USER_TEMPLATE
    candidate = TASK_PROFILE_FIRST_USER_TEMPLATE.format(profile=profile)
    try:
        load_prompt_template(candidate)
    except ValueError:
        return ML_FIRST_USER_TEMPLATE
    return candidate


def _seed_block(seed: int | None) -> str:
    try:
        value = int(seed or 0)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    return (
        f"Base experiment seed: {value}. Use this for validation splits, model random_state, "
        "and other stochastic choices; worker processes may receive deterministic offsets.\n"
    )


def _worker_identity_block(worker_id: str) -> str:
    wid = str(worker_id or "").strip().upper()
    if not wid:
        return ""
    return f"Worker identity: {wid}. Use this ID when labeling worker-local notes, artifacts, and stage references.\n"


def _compact_prompt_field(value: str, *, max_chars: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _estra_research_note_block(
    *,
    continuation_action: str = "keep_current",
    exploration_summary: str = "",
    bottleneck: str = "",
    evidence: str = "",
    missing_evidence: str = "",
    decision_reason: str = "",
    redirect_focus: str = "",
) -> str:
    action = str(continuation_action or "keep_current").strip() or "keep_current"
    startpoint = "previous_stage" if action == "switch_stage" else "current_workspace"
    intent = "redirect" if action == "keep_but_redirect" else "continue"
    rows: list[str] = [f"ESTRA continuation action: {action}", f"ESTRA startpoint: {startpoint}", f"ESTRA intent: {intent}"]
    fields = [
        ("Exploration summary", exploration_summary),
        ("Bottleneck", bottleneck),
        ("Evidence", evidence),
        ("Missing evidence", missing_evidence),
        ("Decision reason", decision_reason),
        ("Redirect focus", redirect_focus),
    ]
    for label, value in fields:
        compact = _compact_prompt_field(str(value or ""), max_chars=180)
        if compact:
            rows.append(f"{label}: {compact}")
    if len(rows) == 1 and action == "keep_current":
        return ""
    if action == "keep_but_redirect":
        rows.append("Use this note to redirect the next stage around the stated bottleneck; do not repeat shallow local tweaks.")
    return "ESTRA research note:\n" + "\n".join(rows) + "\n\n"


def build_first_user_prompt(
    task_desc: str,
    *,
    wall_clock_budget_sec: int,
    seed: int | None = None,
    worker_id: str = "",
    parallel_worker_snapshot: str = "",
    resource_context: str = "",
    skill_hint: str = "",
    evaluator_contract: str = "",
    task_profile: str = "mlebench",
    task_runtime_contract: str = "",
    initial_workspace_state: str = "",
) -> str:
    budget = max(60, int(wall_clock_budget_sec or 0))
    task = (task_desc or "").strip() or "Complete the machine-learning task in this workspace."
    contract = str(evaluator_contract or "").strip()
    if contract:
        task = task.rstrip() + "\n\nEvaluator artifact contract:\n" + contract
    return render_prompt_template(
        _first_user_template_name(task_profile),
        budget=budget,
        parallel_worker_block=_parallel_worker_block(parallel_worker_snapshot),
        resource_context_block=_resource_context_block(resource_context),
        seed_block=_seed_block(seed),
        worker_identity_block=_worker_identity_block(worker_id),
        skill_hint_block=_skill_hint_block(skill_hint),
        runtime_contract_block=_runtime_contract_block(task_runtime_contract),
        initial_workspace_state_block=_initial_workspace_state_block(initial_workspace_state),
        task=task,
    )


def build_stage_commit_prompt(
    *,
    stage_id: str,
    ledger_filename: str,
    metric_event: dict[str, Any],
    existing_ledger: str,
) -> str:
    metric_json = json.dumps(metric_event, ensure_ascii=False, sort_keys=True)
    prior = compact_ledger_for_prompt(existing_ledger, max_chars=7000)
    prior_block = prior if prior else "(no prior stages)"
    return render_prompt_template(
        STAGE_COMMIT_TEMPLATE,
        stage_id=stage_id,
        ledger_filename=ledger_filename,
        metric_json=metric_json,
        prior_block=prior_block,
    )


def build_stage_commit_judgment_prompt(
    *,
    stage_id: str,
    ledger_filename: str,
    metric_event: dict[str, Any],
    existing_ledger: str,
) -> str:
    metric_json = json.dumps(metric_event, ensure_ascii=False, sort_keys=True)
    prior = compact_ledger_for_prompt(existing_ledger, max_chars=7000)
    prior_block = prior if prior else "(no prior stages)"
    return render_prompt_template(
        STAGE_COMMIT_JUDGMENT_TEMPLATE,
        stage_id=stage_id,
        ledger_filename=ledger_filename,
        metric_json=metric_json,
        prior_block=prior_block,
    )

def build_metric_validity_adjudicator_prompt(facts: dict[str, Any]) -> str:
    fact_json = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    return render_prompt_template(
        METRIC_VALIDITY_USER_TEMPLATE,
        fact_json=fact_json,
    )


def build_metric_validity_adjudicator_system_prompt() -> str:
    return render_prompt_template(METRIC_VALIDITY_SYSTEM_TEMPLATE)


def build_metric_output_interpreter_prompt(facts: dict[str, Any]) -> str:
    fact_json = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    return render_prompt_template(
        METRIC_OUTPUT_INTERPRETER_USER_TEMPLATE,
        fact_json=fact_json,
    )


def build_metric_output_interpreter_system_prompt() -> str:
    return render_prompt_template(METRIC_OUTPUT_INTERPRETER_SYSTEM_TEMPLATE)


def build_estra_prompt(
    *,
    ledger_filename: str,
    ledger_text: str,
    latest_stage: str,
    switch_candidate_stages: list[str],
    stage_checkpoint_context: str = "",
    peer_route_evidence: str = "",
    backtrack_reflection: str = "",
    resource_context: str = "",
) -> str:
    latest = str(latest_stage or "").strip().upper() or "(none)"
    candidates = ", ".join(str(s).upper() for s in switch_candidate_stages) or "(none)"
    ledger = compact_ledger_for_prompt(ledger_text, max_chars=9000) or "(empty)"
    checkpoint_context = (stage_checkpoint_context or "").strip()
    checkpoint_block = (
        "\nStage checkpoint context:\n"
        f"{checkpoint_context}\n"
        if checkpoint_context
        else ""
    )
    peer_block = _named_context_block("Peer route evidence", peer_route_evidence)
    backtrack_block = _named_context_block("Backtrack reflection", backtrack_reflection)
    resource_block = _resource_context_block(resource_context)
    return render_prompt_template(
        ESTRA_DECISION_TEMPLATE,
        latest_stage=latest,
        switch_candidates=candidates,
        ledger_filename=ledger_filename,
        ledger=ledger,
        trailing_blocks="\n" + checkpoint_block + peer_block + backtrack_block + resource_block,
    )


def build_estra_archive_summary_prompt(
    *,
    target_stage: str,
    terminal_stage: str,
    estra_reason: str,
    deterministic_summary: str,
) -> str:
    summary = (deterministic_summary or "").strip()
    if len(summary) > 3000:
        summary = summary[:2960].rstrip() + "\n... [deterministic tail truncated]"
    return render_prompt_template(
        ESTRA_ARCHIVE_SUMMARY_TEMPLATE,
        target_stage=str(target_stage or "").strip().upper() or "(unknown)",
        terminal_stage=str(terminal_stage or "").strip().upper() or "(unknown)",
        estra_reason=_compact_prompt_field(estra_reason, max_chars=240),
        deterministic_summary=summary or "(no abandoned tail stages after target)",
    )


def _base_stage_memory_policy(base_stage: str) -> str:
    stage = str(base_stage or "").strip().upper()
    if stage:
        return f"The original {stage} base stage is preserved as the stable EDA and first-solution foundation."
    return "No stable base stage is available; rely on the state packet and workspace files as the source of truth."


def build_estra_resume_prompt(
    *,
    target_stage: str,
    next_stage: str,
    ledger_filename: str,
    base_stage: str = "",
    compact_summary: str,
    state_packet: str = "",
    parallel_worker_snapshot: str = "",
    resource_context: str = "",
) -> str:
    _ = ledger_filename
    summary = (compact_summary or "").strip()
    if len(summary) > 1600:
        summary = summary[:1560].rstrip() + "\n... [compact summary truncated]"
    summary_block = summary if summary else "(no abandoned trajectory summary was available)"
    packet = (state_packet or "").strip()
    packet_block = (
        "LHR state packet (source of truth for continuation):\n"
        f"{packet}\n\n"
        if packet
        else ""
    )
    return render_prompt_template(
        ESTRA_RESUME_TEMPLATE,
        target_stage=target_stage,
        next_stage=next_stage,
        state_packet_block=packet_block,
        summary_block=summary_block,
        base_stage_memory_policy=_base_stage_memory_policy(base_stage),
        parallel_worker_block=_parallel_worker_block(parallel_worker_snapshot),
        resource_context_block=_resource_context_block(resource_context),
    )


def build_keep_current_compact_prompt(
    *,
    terminal_stage: str,
    next_stage: str,
    compact_summary: str,
    base_stage: str = "",
    state_packet: str = "",
    parallel_worker_snapshot: str = "",
    resource_context: str = "",
    continuation_action: str = "keep_current",
    exploration_summary: str = "",
    bottleneck: str = "",
    evidence: str = "",
    missing_evidence: str = "",
    decision_reason: str = "",
    redirect_focus: str = "",
) -> str:
    summary = (compact_summary or "").strip()
    if len(summary) > 1600:
        summary = summary[:1560].rstrip() + "\n... [compact summary truncated]"
    summary_block = summary if summary else "(no keep_current trajectory summary was available)"
    packet = (state_packet or "").strip()
    packet_block = (
        "LHR state packet (source of truth for continuation):\n"
        f"{packet}\n\n"
        if packet
        else ""
    )
    return render_prompt_template(
        KEEP_CURRENT_COMPACT_TEMPLATE,
        terminal_stage=terminal_stage,
        next_stage=next_stage,
        base_stage_memory_policy=_base_stage_memory_policy(base_stage),
        state_packet_block=packet_block,
        summary_block=summary_block,
        parallel_worker_block=_parallel_worker_block(parallel_worker_snapshot),
        resource_context_block=_resource_context_block(resource_context),
        estra_research_note_block=_estra_research_note_block(
            continuation_action=continuation_action,
            exploration_summary=exploration_summary,
            bottleneck=bottleneck,
            evidence=evidence,
            missing_evidence=missing_evidence,
            decision_reason=decision_reason,
            redirect_focus=redirect_focus,
        ),
    )
