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

"""Authoritative Gate transaction that owns evaluator execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

from scienceflow.core.bash_solution_cmd import command_sets_subsample_row_env
from scienceflow.gates.evaluator.manager import EvaluatorManager
from scienceflow.gates.evaluator.models import (
    EvaluationOutcome,
    EvaluationRequest,
    MetricEvent,
)
from scienceflow.gates.policy import GateManager


_STAGE_RESULT_TRIGGERS = frozenset({"stage", "stage_end"})


class GateService:
    """Evaluate candidate artifacts and return authoritative Gate outcomes."""

    def __init__(
        self,
        evaluator_manager: EvaluatorManager | None = None,
        gate_manager: GateManager | None = None,
    ) -> None:
        self.evaluator_manager = evaluator_manager or EvaluatorManager.default()
        self.gate_manager = gate_manager or GateManager.default()

    @property
    def manager(self) -> EvaluatorManager:
        """Return the evaluator plugin manager owned by this Gate."""

        return self.evaluator_manager

    @classmethod
    def default(cls) -> "GateService":
        return cls(EvaluatorManager.default(), GateManager.default())

    def build_prompt_contract(self, request: EvaluationRequest) -> str:
        ctx = request.context
        backend = self.evaluator_manager.get(
            self.evaluator_manager.backend_name_for_context(ctx)
        )
        if backend is None:
            return ""
        return str(backend.build_prompt_contract(ctx) or "").strip()

    def evaluate(self, request: EvaluationRequest) -> list[EvaluationOutcome]:
        ctx = request.context
        outcomes: list[EvaluationOutcome] = []
        for raw_event in self.evaluator_manager.evaluate_workspace(ctx):
            outcomes.append(self._evaluate_event(request, raw_event))

        # Stage admission has two independent candidate sources: an evaluator-
        # verified artifact, or the validated result of a complete run.  Final
        # selection deliberately has no result-signal fallback: merge/export
        # still requires the real task artifact.
        if str(request.trigger or "").strip().lower() in _STAGE_RESULT_TRIGGERS:
            result_event = _stage_result_signal_event(request)
            result_outcome = (
                self._evaluate_event(request, result_event)
                if result_event is not None
                else None
            )
            if len(outcomes) == 1 and outcomes[0].decision.accepted:
                return outcomes
            if result_outcome is not None and result_outcome.decision.accepted:
                return [result_outcome]
            if not outcomes and result_outcome is not None:
                return [result_outcome]
        return outcomes

    def _evaluate_event(
        self,
        request: EvaluationRequest,
        raw_event: MetricEvent,
    ) -> EvaluationOutcome:
        ctx = request.context
        decision, gate_trace = self.gate_manager.decide_with_trace(
            ctx,
            raw_event,
            trigger=request.trigger,
        )
        extra = dict(raw_event.extra or {})
        extra["gate"] = gate_trace
        event = replace(
            raw_event,
            candidate_ready=decision.candidate_ready,
            selection_eligible=decision.selection_eligible,
            extra=extra,
        )
        return EvaluationOutcome(event=event, decision=decision)


def _stage_result_signal_event(request: EvaluationRequest) -> MetricEvent | None:
    """Normalize the existing full-run metric snapshot as a Stage candidate.

    This is intentionally not a generic stdout trigger.  The LNR integration
    supplies a ``metric_event`` only after its full-run/validation contract has
    produced a workspace snapshot.  Pilot, quick, debug, and row-subsampled
    runs remain observations and cannot become Stages.
    """

    ctx = request.context
    metadata = ctx.metadata if isinstance(ctx.metadata, Mapping) else {}
    raw = metadata.get("metric_event")
    if not isinstance(raw, Mapping) or raw.get("metric_value") in (None, ""):
        return None
    try:
        metric_value = float(raw.get("metric_value"))
    except (TypeError, ValueError):
        metric_value = None
    finite = metric_value is not None and math.isfinite(metric_value)
    validation_ok = raw.get("validation_ok") is True
    selection_eligible = _boolish(raw.get("selection_eligible"), default=False)
    non_full_reason = _non_full_result_reason(raw)
    stage_ready = bool(
        finite
        and validation_ok
        and selection_eligible
        and not non_full_reason
    )
    metric_name = str(raw.get("metric_name") or "metric")
    solution_sha = str(raw.get("solution_sha") or "").strip()
    candidate_suffix = solution_sha or str(ctx.stage_id or "result")
    note = str(
        non_full_reason
        or raw.get("validation_issue")
        or raw.get("metric_validity_note")
        or raw.get("selection_note")
        or "validated complete-run result signal"
    )
    extra = {
        "signal_kind": "full_run_result",
        "artifact_ready": False,
        "solution_sha": solution_sha,
        "snapshot_path": str(raw.get("snapshot_path") or ""),
        "execution_scale": str(raw.get("execution_scale") or ""),
        "execution_mode": str(raw.get("execution_mode") or ""),
    }
    return MetricEvent(
        candidate_id=f"result:{ctx.worker_id}:{candidate_suffix}",
        worker_id=ctx.worker_id,
        stage_id=ctx.stage_id,
        metric_value=metric_value,
        metric_name=metric_name,
        lower_is_better=_optional_bool(raw.get("lower_is_better")),
        validation_ok=validation_ok,
        candidate_ready=stage_ready,
        selection_eligible=bool(selection_eligible and stage_ready),
        metric_validity=str(raw.get("metric_validity") or "low"),
        metric_validity_reason_code=str(
            raw.get("metric_validity_reason_code") or "full_run_result_signal"
        ),
        artifact_path="",
        artifact_sha="",
        evaluator_backend="result_signal",
        evaluator_status="valid_full_run_result" if stage_ready else "non_stage_result",
        metric_type=str(raw.get("val_score_type") or raw.get("metric_type") or ""),
        task_profile=ctx.task_profile,
        metric_note=note,
        run_time_sec=_optional_float(raw.get("run_time_sec")),
        extra=extra,
    )


def _non_full_result_reason(raw: Mapping[str, object]) -> str:
    scale = str(raw.get("execution_scale") or "").strip().lower().replace("-", "_")
    if scale == "pilot":
        return "pilot result is not eligible for Stage admission"
    command = str(raw.get("bash_cmd") or "").strip()
    if command_sets_subsample_row_env(command):
        return "row-subsampled or quick-test result is not eligible for Stage admission"
    return ""


def _boolish(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
