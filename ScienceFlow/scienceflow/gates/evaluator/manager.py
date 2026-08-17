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

"""Evaluator backend registry and dispatcher."""

from __future__ import annotations

import logging
from typing import Protocol

from scienceflow.gates.evaluator.models import (
    CandidateRef,
    EvalContext,
    EvaluationResult,
    MetricEvent,
)

logger = logging.getLogger("scienceflow")


class EvaluatorBackend(Protocol):
    name: str

    def prepare_workspace(self, ctx: EvalContext) -> None:
        ...

    def build_prompt_contract(self, ctx: EvalContext) -> str:
        ...

    def detect_candidates(self, ctx: EvalContext) -> list[CandidateRef]:
        ...

    def evaluate(self, ctx: EvalContext, candidate: CandidateRef) -> EvaluationResult:
        ...

    def to_metric_event(
        self,
        ctx: EvalContext,
        candidate: CandidateRef,
        result: EvaluationResult,
    ) -> MetricEvent:
        ...


class EvaluatorManager:
    """Small registry wrapper for task evaluator backends."""

    def __init__(self, backends: list[EvaluatorBackend] | None = None) -> None:
        self._backends: dict[str, EvaluatorBackend] = {}
        for backend in backends or []:
            self.register(backend)

    @classmethod
    def default(cls) -> "EvaluatorManager":
        from scienceflow.gates.evaluator.backends.artifact_command import ArtifactCommandBackend
        from scienceflow.gates.evaluator.backends.task_package import TaskPackageBackend

        return cls([TaskPackageBackend(), ArtifactCommandBackend()])

    def register(self, backend: EvaluatorBackend) -> None:
        name = str(getattr(backend, "name", "") or "").strip()
        if not name:
            raise ValueError("evaluator backend must define a non-empty name")
        self._backends[name] = backend

    def get(self, name: str) -> EvaluatorBackend | None:
        return self._backends.get(str(name or "").strip())

    def backend_name_for_context(self, ctx: EvalContext) -> str:
        cfg_value = _evaluator_config_value(ctx.cfg, "backend", "") or _config_value(ctx.cfg, "evaluator_backend", "")
        if str(cfg_value or "").strip().lower() not in {"", "auto"}:
            return str(cfg_value)
        try:
            from scienceflow.core.task_package import find_task_package

            spec = find_task_package(ctx.task_id)
        except Exception:
            spec = None
        if spec is not None and spec.evaluator_entrypoint:
            return "task_package"
        profile = str(
            ctx.task_profile
            or _evaluator_config_value(ctx.cfg, "task_profile", "")
            or _config_value(ctx.cfg, "task_profile", "auto")
            or "auto"
        )
        if profile in {"", "auto", "default", "generic"}:
            return ""
        if profile in {"ml", "mlebench", "kaggle"}:
            return "task_package"
        return profile

    def evaluate_workspace(self, ctx: EvalContext) -> list[MetricEvent]:
        if _evaluator_config_value(ctx.cfg, "enabled", True) is False:
            return []
        name = self.backend_name_for_context(ctx)
        backend = self.get(name)
        if backend is None:
            logger.debug("[evaluator] backend %s not registered", name)
            return []
        events: list[MetricEvent] = []
        try:
            candidates = backend.detect_candidates(ctx)
        except Exception:
            logger.debug("[evaluator] candidate detection failed for %s", name, exc_info=True)
            return []
        for candidate in candidates:
            try:
                result = backend.evaluate(ctx, candidate)
                events.append(backend.to_metric_event(ctx, candidate, result))
            except Exception as exc:
                logger.debug(
                    "[evaluator] evaluation failed backend=%s candidate=%s",
                    name,
                    candidate.candidate_id,
                    exc_info=True,
                )
                try:
                    failure = _exception_result(ctx, backend=name, exc=exc)
                    events.append(backend.to_metric_event(ctx, candidate, failure))
                except Exception:
                    logger.debug(
                        "[evaluator] failed to normalize evaluator exception "
                        "backend=%s candidate=%s",
                        name,
                        candidate.candidate_id,
                        exc_info=True,
                    )
        return events


def _exception_result(
    ctx: EvalContext,
    *,
    backend: str,
    exc: Exception,
) -> EvaluationResult:
    metric_cfg = _evaluator_config_value(ctx.cfg, "metric", None)
    metric_name = str(_config_value(metric_cfg, "name", "metric") or "metric")
    lower = _config_value(metric_cfg, "lower_is_better", None)
    if not isinstance(lower, bool):
        lower = None
    return EvaluationResult(
        backend=backend,
        ok=False,
        status="evaluator_exception",
        metric_value=None,
        metric_name=metric_name,
        lower_is_better=lower,
        validation_ok=False,
        candidate_ready=False,
        selection_eligible=False,
        metric_validity="low",
        reason_code="evaluator_exception",
        message=f"{type(exc).__name__}: {exc}",
    )


def _config_value(cfg: object, key: str, default: object = None) -> object:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _evaluator_config_value(cfg: object, key: str, default: object = None) -> object:
    evaluator = _config_value(cfg, "evaluator", None)
    if evaluator is None:
        return default
    return _config_value(evaluator, key, default)
