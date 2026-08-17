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

import math
from pathlib import Path

import pytest

from scienceflow.gates.evaluator import EvalContext, GateDecision, MetricEvent
from scienceflow.gates import (
    GateManager,
    format_invalid_evaluator_feedback,
    legacy_score_contract_enabled,
    resolve_gate_config,
)


def _context(*, cfg: object = None, task_id: str = "test-task") -> EvalContext:
    return EvalContext(
        task_profile="test",
        task_id=task_id,
        task_root=Path("."),
        workspace=Path("."),
        worker_id="W00",
        cfg={} if cfg is None else cfg,
    )


def _event(**overrides: object) -> MetricEvent:
    values: dict[str, object] = {
        "candidate_id": "candidate",
        "worker_id": "W00",
        "stage_id": "S01",
        "metric_value": 1.0,
        "metric_name": "score",
        "lower_is_better": False,
        "validation_ok": True,
        "candidate_ready": True,
        "selection_eligible": True,
        "metric_validity": "high",
        "metric_validity_reason_code": "authoritative",
        "artifact_path": "artifact.json",
        "artifact_sha": "abc",
        "evaluator_backend": "test",
        "evaluator_status": "success",
        "metric_type": "benchmark",
    }
    values.update(overrides)
    return MetricEvent(**values)


def _legacy_default_decision(
    event: MetricEvent,
    *,
    trigger: str,
    require_direction: bool = True,
    minimum_validity: str = "high",
) -> GateDecision:
    failure: tuple[str, str] | None = None
    if not event.validation_ok:
        failure = ("validation_failed", event.metric_note or "candidate validation failed")
    elif not event.candidate_ready:
        failure = ("candidate_not_ready", event.metric_note or "candidate artifact is not ready")
    elif event.metric_value is None:
        failure = ("metric_missing", "evaluator did not produce a primary metric")
    elif not math.isfinite(float(event.metric_value)):
        failure = ("metric_not_finite", "evaluator primary metric is not finite")
    elif require_direction and event.lower_is_better is None:
        failure = ("metric_direction_missing", "primary metric direction is not configured")
    else:
        ranks = {"low": 0, "medium": 1, "high": 2}
        actual = str(event.metric_validity or "low").strip().lower()
        if ranks.get(actual, -1) < ranks.get(minimum_validity, 2):
            failure = (
                "metric_validity_below_gate",
                f"metric validity {actual!r} is below required {minimum_validity!r}",
            )
    ready = bool(event.candidate_ready and event.validation_ok)
    if failure is None:
        return GateDecision("accept", True, ready, True, "eligible")
    reason, message = failure
    return GateDecision(
        "reject" if trigger in {"final", "finalize", "global_merge"} else "retry",
        False,
        ready,
        False,
        reason,
        message,
    )


def test_legacy_score_contract_disabled_for_artifact_opt_solver() -> None:
    assert not legacy_score_contract_enabled(
        task_profile="opt_solver",
        evaluator_backend="artifact_command",
        candidate_artifact="artifacts/best_solution.json",
    )


def test_legacy_score_contract_enabled_for_mlebench_submission() -> None:
    assert legacy_score_contract_enabled(
        task_profile="mlebench",
        evaluator_backend="task_package",
        candidate_artifact="submission.csv",
    )


def test_invalid_evaluator_feedback_uses_nested_stderr_tail() -> None:
    feedback = format_invalid_evaluator_feedback(
        {
            "artifact_path": "artifacts/best_solution.json",
            "candidate_ready": False,
            "selection_eligible": False,
            "evaluator_backend": "artifact_command",
            "evaluator_status": "evaluator_error",
            "metric_note": "evaluator command exited with 2",
            "metric_validity_reason_code": "evaluator_nonzero_exit",
            "stderr_tail": "",
            "extra": {
                "stderr_tail": "chromosome length must be 541 for 181 tomatoes, got 130",
            },
        },
        candidate_artifact="artifacts/best_solution.json",
    )

    assert "EVALUATOR_INVALID_ARTIFACT" in feedback
    assert "artifact_command" in feedback
    assert "chromosome length must be 541" in feedback
    assert "submission.csv" not in feedback
    assert "Final Validation Score" not in feedback


def test_ratio_task_keeps_default_policy_for_behavior_parity() -> None:
    ctx = _context(
        task_id="ratio-minimization",
        cfg={
            "gate": {
                "policy": "not-allowed-to-replace-task-policy",
                "params": {"minimum_metric_validity": "medium"},
            }
        },
    )

    name, params = resolve_gate_config(ctx)

    assert name == "default"
    assert params == {"minimum_metric_validity": "medium"}


@pytest.mark.parametrize(
    ("overrides", "trigger", "require_direction", "minimum_validity"),
    [
        ({}, "stage_end", True, "high"),
        ({"validation_ok": False}, "stage_end", True, "high"),
        ({"candidate_ready": False}, "stage_end", True, "high"),
        ({"metric_value": None}, "stage_end", True, "high"),
        ({"metric_value": float("inf")}, "stage_end", True, "high"),
        ({"lower_is_better": None}, "stage_end", True, "high"),
        ({"lower_is_better": None}, "stage_end", False, "high"),
        ({"metric_validity": "medium"}, "stage_end", True, "high"),
        ({"metric_validity": "medium"}, "stage_end", True, "medium"),
        ({"validation_ok": False}, "global_merge", True, "high"),
    ],
)
def test_default_plugin_matches_legacy_gate_decision(
    overrides: dict[str, object],
    trigger: str,
    require_direction: bool,
    minimum_validity: str,
) -> None:
    event = _event(**overrides)
    cfg = {
        "evaluator": {"metric": {"selection_requires_direction": require_direction}},
        "gate": {"params": {"minimum_metric_validity": minimum_validity}},
    }

    actual = GateManager.default().decide(_context(cfg=cfg), event, trigger=trigger)
    expected = _legacy_default_decision(
        event,
        trigger=trigger,
        require_direction=require_direction,
        minimum_validity=minimum_validity,
    )

    assert actual == expected


def test_unknown_policy_fails_closed_and_final_trigger_rejects() -> None:
    ctx = _context(cfg={"gate": {"policy": "missing_plugin"}})

    decision = GateManager.default().decide(ctx, _event(), trigger="finalize")

    assert decision.action == "reject"
    assert decision.reason_code == "gate_policy_unknown"


def test_invalid_param_fails_closed() -> None:
    ctx = _context(cfg={"gate": {"params": {"unexpected": True}}})

    decision = GateManager.default().decide(ctx, _event(), trigger="stage_end")

    assert decision.action == "retry"
    assert decision.reason_code == "gate_invalid_config"


def test_optimization_policy_rejects_constraint_violation() -> None:
    ctx = _context(
        cfg={
            "gate": {
                "policy": "optimization_feasibility",
                "params": {"max_constraint_violation": 1e-6},
            }
        }
    )

    decision = GateManager.default().decide(
        ctx,
        _event(extra={"constraint_violation": 1e-3}),
        trigger="stage_end",
    )

    assert decision.action == "retry"
    assert decision.reason_code == "constraint_violation_above_gate"


def test_plugin_cannot_bypass_core_invariant() -> None:
    class AlwaysAccept:
        name = "always_accept"
        version = "test"

        def decide(self, ctx, event, *, trigger, params):
            _ = ctx, event, trigger, params
            return GateDecision(
                action="accept",
                accepted=True,
                candidate_ready=True,
                selection_eligible=True,
                reason_code="forced",
            )

    manager = GateManager()
    manager.register(AlwaysAccept())
    ctx = _context(cfg={"gate": {"policy": "always_accept"}})

    decision = manager.decide(
        ctx,
        _event(validation_ok=False),
        trigger="stage_end",
    )

    assert decision.action == "retry"
    assert decision.reason_code == "validation_failed"


def test_duplicate_registration_is_rejected() -> None:
    class Plugin:
        name = "duplicate"
        version = "1"

        def decide(self, ctx, event, *, trigger, params):
            raise AssertionError("not called")

    manager = GateManager()
    manager.register(Plugin())

    with pytest.raises(ValueError, match="already registered"):
        manager.register(Plugin())


@pytest.mark.parametrize(
    ("returned", "reason_code"),
    [
        ("not-a-decision", "gate_invalid_decision"),
        (RuntimeError("broken plugin"), "gate_policy_exception"),
    ],
)
def test_bad_plugin_result_fails_closed(returned: object, reason_code: str) -> None:
    class BadPlugin:
        name = "bad"
        version = "1"

        def decide(self, ctx, event, *, trigger, params):
            _ = ctx, event, trigger, params
            if isinstance(returned, Exception):
                raise returned
            return returned

    manager = GateManager()
    manager.register(BadPlugin())
    ctx = _context(cfg={"gate": {"policy": "bad"}})

    decision = manager.decide(ctx, _event(), trigger="stage_end")

    assert decision.action == "retry"
    assert decision.reason_code == reason_code
