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

"""Adapters from evaluator events to existing LNR stage facts."""

from __future__ import annotations

from typing import Any

from scienceflow.gates.evaluator.models import MetricEvent


def metric_event_to_stage_facts(event: MetricEvent) -> dict[str, Any]:
    """Return stage-performance-compatible facts for one evaluator event.

    This function intentionally performs field mapping only. Policy decisions
    such as whether to prefer these facts over legacy stage capture are owned by
    the LNR integration layer via ``stage_source_mode``.
    """

    artifact_kind = ""
    if isinstance(event.extra, dict):
        artifact_kind = str(event.extra.get("artifact_kind") or "")
    facts: dict[str, Any] = {
        "metric_value": event.metric_value,
        "metric_name": event.metric_name,
        "lower_is_better": event.lower_is_better,
        "validation_ok": event.validation_ok,
        "candidate_ready": event.candidate_ready,
        "selection_eligible": event.selection_eligible,
        "metric_validity": event.metric_validity,
        "metric_validity_reason_code": event.metric_validity_reason_code,
        "artifact_path": event.artifact_path,
        "artifact_sha": event.artifact_sha,
        "artifact_kind": artifact_kind,
        "evaluator_backend": event.evaluator_backend,
        "evaluator_status": event.evaluator_status,
        "task_profile": event.task_profile,
        "val_score_type": event.metric_type,
        "metric_source_note": event.metric_note,
        "run_time_sec": event.run_time_sec,
    }
    if isinstance(event.extra, dict):
        facts["evaluator_stdout_tail"] = str(event.extra.get("stdout_tail") or "")
        facts["evaluator_stderr_tail"] = str(event.extra.get("stderr_tail") or "")
        facts["metric_authoritative"] = bool(
            event.extra.get("metric_authoritative", False)
        )
    deliverable_role = str(event.extra.get("deliverable_role") or "") if isinstance(event.extra, dict) else ""
    if deliverable_role == "submission_csv":
        facts["submission_status"] = event.evaluator_status
        facts["submission_sha"] = event.artifact_sha
        facts["submission_snapshot"] = event.artifact_path
    return facts


def merge_adjudicated_stage_facts(
    legacy: dict[str, Any],
    evaluator_facts: dict[str, Any],
) -> dict[str, Any]:
    """Overlay evaluator validation/artifact facts without replacing legacy metric."""

    out = dict(legacy or {})
    for key in (
        "validation_ok",
        "candidate_ready",
        "selection_eligible",
        "metric_validity",
        "metric_validity_reason_code",
        "artifact_path",
        "artifact_sha",
        "artifact_kind",
        "evaluator_backend",
        "evaluator_status",
        "task_profile",
        "metric_authoritative",
        "submission_status",
        "submission_sha",
        "gate_metric_validity",
        "gate_policy",
        "gate_policy_version",
        "gate_action",
        "gate_accepted",
        "gate_reason_code",
        "gate_message",
        "_gate_evaluated",
    ):
        if key in evaluator_facts:
            out[key] = evaluator_facts[key]
    note = str(evaluator_facts.get("metric_source_note") or "").strip()
    if note:
        out["metric_source_note"] = note
    return out


def merge_primary_stage_facts(
    legacy: dict[str, Any],
    evaluator_facts: dict[str, Any],
) -> dict[str, Any]:
    """Use evaluator metric facts while preserving non-evaluator run metadata."""

    out = dict(legacy or {})
    out.update(evaluator_facts)
    for key in (
        "solution_sha",
        "source_commit_sha",
        "source_changed",
        "semantic_source_changed",
        "capture_type",
        "workspace_git_stage_id",
        "workspace_git_ready",
        "workspace_git_message",
    ):
        if key in legacy and key not in out:
            out[key] = legacy[key]
    return out
