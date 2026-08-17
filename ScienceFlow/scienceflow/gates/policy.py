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

"""Lightweight gate plugin contract, registry, and built-in policies."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Any, Mapping, Protocol, runtime_checkable

from scienceflow.core.task_package import find_task_package
from scienceflow.gates.invariants import (
    check_core_invariants,
    failure_decision,
)
from scienceflow.gates.evaluator.models import EvalContext, GateDecision, MetricEvent

_LEGACY_SCORE_PROFILES = {"", "default", "ml", "mlebench", "kaggle"}
_LEGACY_SCORE_BACKENDS = {"", "default", "ml", "mlebench", "kaggle", "task_package"}
_ARTIFACT_PROFILES = {"artifact_command", "artifact_solver", "opt_solver", "optimization_solver"}
_ARTIFACT_BACKENDS = {"artifact_command"}
_VALIDITY_RANK = {"low": 0, "medium": 1, "high": 2}


@runtime_checkable
class GatePolicy(Protocol):
    """Minimal contract implemented by trusted, explicitly registered plugins."""

    name: str
    version: str

    def decide(
        self,
        ctx: EvalContext,
        event: MetricEvent,
        *,
        trigger: str,
        params: Mapping[str, Any],
    ) -> GateDecision: ...


class DefaultGatePolicy:
    """Default metric eligibility policy used when a task names no plugin."""

    name = "default"
    version = "1"

    def decide(
        self,
        ctx: EvalContext,
        event: MetricEvent,
        *,
        trigger: str,
        params: Mapping[str, Any],
    ) -> GateDecision:
        allowed = {"minimum_metric_validity"}
        _reject_unknown_params(params, allowed)
        required_validity = str(
            params.get("minimum_metric_validity", "high") or "high"
        ).strip().lower()
        if required_validity not in _VALIDITY_RANK:
            raise ValueError(
                "minimum_metric_validity must be one of: low, medium, high"
            )
        if event.metric_value is None:
            return failure_decision(
                event,
                trigger=trigger,
                reason_code="metric_missing",
                message="evaluator did not produce a primary metric",
            )
        if _runtime_metric_direction_required(ctx) and event.lower_is_better is None:
            return failure_decision(
                event,
                trigger=trigger,
                reason_code="metric_direction_missing",
                message="primary metric direction is not configured",
            )
        actual_validity = str(event.metric_validity or "low").strip().lower()
        if _VALIDITY_RANK.get(actual_validity, -1) < _VALIDITY_RANK[required_validity]:
            return failure_decision(
                event,
                trigger=trigger,
                reason_code="metric_validity_below_gate",
                message=(
                    f"metric validity {actual_validity!r} is below required "
                    f"{required_validity!r}"
                ),
            )
        return _accept(event)


class OptimizationFeasibilityGatePolicy(DefaultGatePolicy):
    """Default metric checks plus an optional optimization feasibility bound."""

    name = "optimization_feasibility"
    version = "1"

    def decide(
        self,
        ctx: EvalContext,
        event: MetricEvent,
        *,
        trigger: str,
        params: Mapping[str, Any],
    ) -> GateDecision:
        allowed = {"minimum_metric_validity", "max_constraint_violation"}
        _reject_unknown_params(params, allowed)
        base_params = {
            key: value for key, value in params.items() if key == "minimum_metric_validity"
        }
        base = super().decide(
            ctx,
            event,
            trigger=trigger,
            params=base_params,
        )
        if not base.accepted:
            return base
        try:
            maximum = float(params.get("max_constraint_violation", 1e-6))
            actual = float(event.extra.get("constraint_violation", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("constraint violation values must be numeric") from exc
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError("max_constraint_violation must be finite and non-negative")
        if not math.isfinite(actual) or actual > maximum:
            return failure_decision(
                event,
                trigger=trigger,
                reason_code="constraint_violation_above_gate",
                message=f"constraint violation {actual} exceeds allowed {maximum}",
            )
        return base


class GateManager:
    """Explicit registry and fail-closed dispatcher for trusted gate policies."""

    def __init__(self) -> None:
        self._policies: dict[str, GatePolicy] = {}

    @classmethod
    def default(cls) -> "GateManager":
        manager = cls()
        manager.register(DefaultGatePolicy())
        manager.register(OptimizationFeasibilityGatePolicy())
        return manager

    def register(self, policy: GatePolicy) -> None:
        name = normalize_gate_name(getattr(policy, "name", ""))
        version = str(getattr(policy, "version", "") or "").strip()
        if not name or not version or not isinstance(policy, GatePolicy):
            raise ValueError("gate policy must provide name, version, and decide()")
        if name in self._policies:
            raise ValueError(f"gate policy already registered: {name!r}")
        self._policies[name] = policy

    def decide(
        self,
        ctx: EvalContext,
        event: MetricEvent,
        *,
        trigger: str,
    ) -> GateDecision:
        decision, _ = self.decide_with_trace(ctx, event, trigger=trigger)
        return decision

    def decide_with_trace(
        self,
        ctx: EvalContext,
        event: MetricEvent,
        *,
        trigger: str,
    ) -> tuple[GateDecision, dict[str, Any]]:
        name, params = resolve_gate_config(ctx)
        policy = self._policies.get(name)
        invariant_failure = check_core_invariants(event, trigger=trigger)
        if invariant_failure is not None:
            decision = invariant_failure
            version = str(getattr(policy, "version", "unknown"))
            return decision, _trace(name, version, params, trigger, decision)
        if policy is None:
            decision = failure_decision(
                event,
                trigger=trigger,
                reason_code="gate_policy_unknown",
                message=f"gate policy is not registered: {name!r}",
            )
            return decision, _trace(name, "unknown", params, trigger, decision)
        try:
            decision = policy.decide(
                ctx,
                event,
                trigger=trigger,
                params=params,
            )
        except (TypeError, ValueError) as exc:
            decision = failure_decision(
                event,
                trigger=trigger,
                reason_code="gate_invalid_config",
                message=str(exc),
            )
        except Exception as exc:
            decision = failure_decision(
                event,
                trigger=trigger,
                reason_code="gate_policy_exception",
                message=f"gate policy {name!r} failed: {exc}",
            )
        if not _valid_decision(decision):
            decision = failure_decision(
                event,
                trigger=trigger,
                reason_code="gate_invalid_decision",
                message=f"gate policy {name!r} returned an invalid decision",
            )
        return decision, _trace(name, policy.version, params, trigger, decision)


def decide_metric_event(
    ctx: EvalContext,
    event: MetricEvent,
    *,
    trigger: str,
) -> GateDecision:
    """Compatibility entry point backed by the default plugin manager."""

    return _DEFAULT_GATE_MANAGER.decide(ctx, event, trigger=trigger)


def resolve_gate_config(ctx: EvalContext) -> tuple[str, dict[str, Any]]:
    """Resolve task policy plus task/run parameters with run parameters last."""

    task_gate: Mapping[str, Any] = {}
    try:
        spec = find_task_package(ctx.task_id)
    except Exception:
        spec = None
    if spec is not None and isinstance(spec.config.get("gate"), Mapping):
        task_gate = spec.config["gate"]
    runtime_gate = _value(ctx.cfg, "gate", {})
    if not isinstance(runtime_gate, Mapping):
        runtime_gate = _object_mapping(runtime_gate)

    task_policy = normalize_gate_name(task_gate.get("policy"))
    runtime_policy = normalize_gate_name(runtime_gate.get("policy"))
    name = task_policy or runtime_policy or "default"
    params: dict[str, Any] = {}
    for source in (task_gate.get("params"), runtime_gate.get("params")):
        if isinstance(source, Mapping):
            params.update(source)
    return name, params


def _accept(event: MetricEvent) -> GateDecision:
    ready = bool(event.candidate_ready and event.validation_ok)
    return GateDecision(
        action="accept",
        accepted=True,
        candidate_ready=ready,
        selection_eligible=True,
        reason_code="eligible",
    )


def _valid_decision(decision: object) -> bool:
    if not isinstance(decision, GateDecision):
        return False
    if decision.action == "accept":
        return bool(decision.accepted and decision.candidate_ready and decision.selection_eligible)
    if decision.action in {"retry", "reject"}:
        return not decision.accepted and not decision.selection_eligible
    return False


def _trace(
    name: str,
    version: str,
    params: Mapping[str, Any],
    trigger: str,
    decision: GateDecision,
) -> dict[str, Any]:
    return {
        "policy": name,
        "version": version,
        "params": dict(params),
        "trigger": str(trigger or "stage_end"),
        "decision": decision.to_dict(),
    }


def _reject_unknown_params(params: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(str(key) for key in params if key not in allowed)
    if unknown:
        raise ValueError(f"unknown gate params: {', '.join(unknown)}")


def _value(obj: object, key: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _object_mapping(obj: object) -> Mapping[str, Any]:
    if obj is None:
        return {}
    return {
        key: getattr(obj, key)
        for key in ("policy", "params")
        if hasattr(obj, key)
    }


def _runtime_metric_direction_required(ctx: EvalContext) -> bool:
    """Preserve the pre-plugin default gate's direction requirement."""

    evaluator = _value(ctx.cfg, "evaluator", None)
    metric = _value(evaluator, "metric", None)
    value = _value(metric, "selection_requires_direction", None)
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return True


def normalize_gate_name(value: object) -> str:
    """Normalize profile/backend names for deterministic gate decisions."""
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _clean_artifact_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    return str(PurePosixPath(text))


def legacy_score_contract_enabled(
    *,
    task_profile: object = "",
    evaluator_backend: object = "",
    candidate_artifact: object = "",
) -> bool:
    """Return whether legacy MLEBench score/submission checks may run.

    The legacy gate enforces root ``submission.csv`` and ``Final Validation Score``.
    Generic artifact tasks must not inherit those semantics; they are validated by
    their configured evaluator backend instead.
    """
    profile = normalize_gate_name(task_profile)
    backend = normalize_gate_name(evaluator_backend)
    artifact = _clean_artifact_path(candidate_artifact)

    if profile in _ARTIFACT_PROFILES or backend in _ARTIFACT_BACKENDS:
        return False
    if profile not in _LEGACY_SCORE_PROFILES:
        return False
    if backend not in _LEGACY_SCORE_BACKENDS:
        return False
    return artifact in {"", "submission.csv"}


_DEFAULT_GATE_MANAGER = GateManager.default()
