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

"""Command-based evaluator backend for generic artifact tasks."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from scienceflow.gates.evaluator.cache import EvaluatorCache
from scienceflow.gates.evaluator.models import (
    CandidateRef,
    EvalContext,
    EvaluationResult,
    MetricEvent,
)
from scienceflow.gates.evaluator.backends.command_helpers import (
    as_bool_value,
    as_float,
    bool_cfg,
    cache_key,
    cache_path,
    candidate_artifact,
    clean_relative_path,
    path_sha256,
    parse_metric,
    run_command,
    str_cfg,
)


class ArtifactCommandBackend:
    name = "artifact_command"

    def prepare_workspace(self, ctx: EvalContext) -> None:
        _ = ctx

    def build_prompt_contract(self, ctx: EvalContext) -> str:
        artifact = candidate_artifact(ctx)
        return (
            f"Save the candidate artifact at `{artifact}`. The framework evaluator "
            "will score that artifact; do not treat self-reported scores as authoritative."
        )

    def detect_candidates(self, ctx: EvalContext) -> list[CandidateRef]:
        rel = candidate_artifact(ctx)
        path = ctx.workspace / rel
        if not path.exists():
            if not bool_cfg(ctx, ("candidate", "emit_missing_artifact_event"), default=False):
                return []
            return [
                CandidateRef(
                    candidate_id=_candidate_id(ctx, ""),
                    backend=self.name,
                    workspace=ctx.workspace,
                    artifact_path=rel,
                    artifact_kind=str_cfg(ctx, ("candidate", "artifact_kind"), "artifact"),
                    stage_id=ctx.stage_id,
                    worker_id=ctx.worker_id,
                    created_at=time.time(),
                    metadata={"missing_artifact": True},
                )
            ]
        sha = path_sha256(path)
        return [
            CandidateRef(
                candidate_id=_candidate_id(ctx, sha),
                backend=self.name,
                workspace=ctx.workspace,
                artifact_path=rel,
                artifact_sha=sha,
                artifact_kind=str_cfg(ctx, ("candidate", "artifact_kind"), "artifact"),
                stage_id=ctx.stage_id,
                worker_id=ctx.worker_id,
                created_at=time.time(),
            )
        ]

    def evaluate(self, ctx: EvalContext, candidate: CandidateRef) -> EvaluationResult:
        artifact = ctx.workspace / candidate.artifact_path
        metric_name = str_cfg(ctx, ("metric", "name"), "metric")
        lower = bool_cfg(ctx, ("metric", "lower_is_better"), default=None)
        if not artifact.exists():
            return _result(
                status="missing_artifact",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="low",
                reason="missing_artifact",
                message=f"candidate artifact not found: {candidate.artifact_path}",
            )
        sha_ok, sha_message = _validate_commit_sha(ctx, candidate)
        if not sha_ok:
            return _result(
                status="artifact_sha_mismatch",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="low",
                reason="artifact_sha_mismatch",
                message=sha_message,
            )
        command = str_cfg(ctx, ("command", "evaluator_command"), "")
        if not command:
            return _result(
                status="missing_evaluator_command",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="low",
                reason="missing_evaluator_command",
                message="evaluator.command.evaluator_command is empty",
            )
        cache = EvaluatorCache(cache_path(ctx))
        key = cache_key(ctx, candidate)
        cached = cache.get(key) if candidate.artifact_sha else None
        if cached:
            return _result_from_dict(cached)
        result = self._evaluate_uncached(ctx, candidate, command, metric_name, lower)
        if candidate.artifact_sha:
            cache.put(key, result.to_dict())
        return result

    def _evaluate_uncached(
        self,
        ctx: EvalContext,
        candidate: CandidateRef,
        command: str,
        metric_name: str,
        lower: bool | None,
    ) -> EvaluationResult:
        started = time.monotonic()
        completed = run_command(ctx, candidate, command)
        run_time = time.monotonic() - started
        metadata = {"run_time_sec": run_time, **completed}
        if completed.get("timeout"):
            return _result(
                status="evaluator_timeout",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="low",
                reason="evaluator_timeout",
                message=f"evaluator command timed out after {completed.get('timeout_sec')}s",
                stdout_tail=str(completed.get("stdout_tail") or ""),
                stderr_tail=str(completed.get("stderr_tail") or ""),
                metadata=metadata,
            )
        if int(completed.get("returncode") or 0) != 0:
            return _result(
                status="evaluator_error",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="low",
                reason="evaluator_nonzero_exit",
                message=f"evaluator command exited with {completed.get('returncode')}",
                stdout_tail=str(completed.get("stdout_tail") or ""),
                stderr_tail=str(completed.get("stderr_tail") or ""),
                metadata=metadata,
            )
        metric_value, parse_message = parse_metric(ctx, str(completed.get("stdout") or ""))
        if metric_value is None:
            return _result(
                status="no_metric",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="medium",
                reason="metric_parse_failed",
                message=parse_message,
                stdout_tail=str(completed.get("stdout_tail") or ""),
                stderr_tail=str(completed.get("stderr_tail") or ""),
                metadata=metadata,
            )
        eligible = lower is not None
        return EvaluationResult(
            backend=self.name,
            ok=True,
            status="ok",
            metric_value=metric_value,
            metric_name=metric_name,
            lower_is_better=lower,
            validation_ok=True,
            candidate_ready=True,
            selection_eligible=eligible,
            metric_validity="high" if eligible else "medium",
            reason_code="artifact_command_metric" if eligible else "metric_direction_missing",
            message=parse_message,
            stdout_tail=str(completed.get("stdout_tail") or ""),
            stderr_tail=str(completed.get("stderr_tail") or ""),
            metadata=metadata,
        )

    def to_metric_event(
        self,
        ctx: EvalContext,
        candidate: CandidateRef,
        result: EvaluationResult,
    ) -> MetricEvent:
        run_time = as_float(result.metadata.get("run_time_sec") if isinstance(result.metadata, Mapping) else None)
        return MetricEvent(
            candidate_id=candidate.candidate_id,
            worker_id=candidate.worker_id or ctx.worker_id,
            stage_id=candidate.stage_id or ctx.stage_id,
            metric_value=result.metric_value,
            metric_name=result.metric_name,
            lower_is_better=result.lower_is_better,
            validation_ok=result.validation_ok,
            candidate_ready=result.candidate_ready,
            selection_eligible=result.selection_eligible,
            metric_validity=result.metric_validity,
            metric_validity_reason_code=result.reason_code,
            artifact_path=candidate.artifact_path,
            artifact_sha=candidate.artifact_sha,
            evaluator_backend=self.name,
            evaluator_status=result.status,
            metric_type=str_cfg(ctx, ("metric", "type"), "benchmark"),
            task_profile=ctx.task_profile,
            metric_note=result.message,
            run_time_sec=run_time,
            extra={
                "artifact_kind": candidate.artifact_kind,
                "stdout_tail": result.stdout_tail,
                "stderr_tail": result.stderr_tail,
                "cache_key": cache_key(ctx, candidate),
            },
        )


def _candidate_id(ctx: EvalContext, sha: str) -> str:
    worker = str(ctx.worker_id or "W00").strip()
    stage = str(ctx.stage_id or "").strip()
    if stage:
        return f"{worker}:{stage}"
    if sha:
        return f"{worker}:candidate:{sha[:12]}"
    return f"{worker}:candidate:missing"


def _result(
    *,
    status: str,
    metric_name: str,
    lower_is_better: bool | None,
    validity: str,
    reason: str,
    message: str,
    stdout_tail: str = "",
    stderr_tail: str = "",
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        backend="artifact_command",
        ok=False,
        status=status,
        metric_value=None,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        validation_ok=False,
        candidate_ready=False,
        selection_eligible=False,
        metric_validity=validity,
        reason_code=reason,
        message=message,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        metadata=metadata or {},
    )


def _result_from_dict(data: Mapping[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        backend=str(data.get("backend") or "artifact_command"),
        ok=bool(data.get("ok")),
        status=str(data.get("status") or "cached"),
        metric_value=as_float(data.get("metric_value")),
        metric_name=str(data.get("metric_name") or "metric"),
        lower_is_better=as_bool_value(data.get("lower_is_better")),
        validation_ok=bool(data.get("validation_ok")),
        candidate_ready=bool(data.get("candidate_ready")),
        selection_eligible=bool(data.get("selection_eligible")),
        metric_validity=str(data.get("metric_validity") or "medium"),
        reason_code=str(data.get("reason_code") or "cached_evaluation_result"),
        message=str(data.get("message") or ""),
        stdout_tail=str(data.get("stdout_tail") or ""),
        stderr_tail=str(data.get("stderr_tail") or ""),
        metadata=dict(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}),
    )


def _validate_commit_sha(ctx: EvalContext, candidate: CandidateRef) -> tuple[bool, str]:
    commit_file = str_cfg(ctx, ("candidate", "commit_file"), "")
    require_sha = bool_cfg(ctx, ("candidate", "require_sha"), default=False) is True
    if not commit_file:
        return (not require_sha or bool(candidate.artifact_sha), "")
    path = ctx.workspace / clean_relative_path(commit_file)
    if not path.is_file():
        return (not require_sha, f"candidate commit file not found: {commit_file}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"candidate commit file is invalid: {exc}"
    if not isinstance(data, dict):
        return False, "candidate commit file must contain a JSON object"
    declared_path = str(data.get("artifact_path") or "").strip()
    if declared_path and clean_relative_path(declared_path) != candidate.artifact_path:
        return False, "candidate commit artifact_path does not match detected artifact"
    declared_sha = str(data.get("artifact_sha") or "").strip()
    if require_sha and not declared_sha:
        return False, "candidate commit file does not declare artifact_sha"
    if declared_sha and declared_sha != candidate.artifact_sha:
        return False, "candidate commit artifact_sha does not match detected artifact"
    return True, ""
