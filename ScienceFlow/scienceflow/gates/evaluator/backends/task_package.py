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

"""Evaluator backend for registered task packages."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from scienceflow.core.task_package import (
    TaskPackageSpec,
    file_sha256,
    find_task_package,
    repo_root,
    prepare_task_runtime,
)
from scienceflow.gates.evaluator.backends.command_env import (
    task_command_env,
    task_python_executable,
)
from scienceflow.gates.evaluator.models import (
    CandidateRef,
    EvalContext,
    EvaluationResult,
    MetricEvent,
)


class TaskPackageBackend:
    name = "task_package"

    def prepare_workspace(self, ctx: EvalContext) -> None:
        _ = ctx

    def build_prompt_contract(self, ctx: EvalContext) -> str:
        spec = _spec(ctx)
        artifact = _artifact_path(ctx, spec)
        if spec is None:
            return f"Save the candidate artifact at `{artifact}` for the configured task evaluator."
        return (
            f"Save the candidate artifact at `{artifact}`. The task-package evaluator "
            "will score it outside the writable worker workspace."
        )

    def detect_candidates(self, ctx: EvalContext) -> list[CandidateRef]:
        spec = _spec(ctx)
        rel = _artifact_path(ctx, spec)
        path = ctx.workspace / rel
        if not path.exists():
            if not _bool_cfg(ctx, ("candidate", "emit_missing_artifact_event"), False):
                return []
            return [
                CandidateRef(
                    candidate_id=_candidate_id(ctx, ""),
                    backend=self.name,
                    workspace=ctx.workspace,
                    artifact_path=rel,
                    artifact_kind=_artifact_kind(ctx, spec),
                    stage_id=ctx.stage_id,
                    worker_id=ctx.worker_id,
                    created_at=time.time(),
                    metadata={"missing_artifact": True},
                )
            ]
        sha = _path_sha256(path)
        return [
            CandidateRef(
                candidate_id=_candidate_id(ctx, sha),
                backend=self.name,
                workspace=ctx.workspace,
                artifact_path=rel,
                artifact_sha=sha,
                artifact_kind=_artifact_kind(ctx, spec),
                stage_id=ctx.stage_id,
                worker_id=ctx.worker_id,
                created_at=time.time(),
            )
        ]

    def evaluate(self, ctx: EvalContext, candidate: CandidateRef) -> EvaluationResult:
        spec = _spec(ctx)
        metric_name = _metric_name(ctx, spec)
        lower = _lower_is_better(ctx, spec)
        artifact = ctx.workspace / candidate.artifact_path
        if not artifact.exists():
            return _result(
                status="missing_artifact",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="low",
                reason="missing_artifact",
                message=f"candidate artifact not found: {candidate.artifact_path}",
            )
        if spec is None:
            return _result(
                status="missing_task_package",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="low",
                reason="missing_task_package",
                message=f"task package not found for {ctx.task_id!r}",
            )
        started = time.monotonic()
        completed = self._run_task_evaluator(ctx, candidate, spec)
        run_time = time.monotonic() - started
        metadata = {"run_time_sec": run_time, **completed}
        if completed.get("hash_mismatch"):
            return _result(
                status="evaluator_hash_mismatch",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="low",
                reason="evaluator_hash_mismatch",
                message=str(completed.get("message") or "task evaluator hash changed before execution"),
                metadata=metadata,
            )
        if completed.get("timeout"):
            return _result(
                status="evaluator_timeout",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="low",
                reason="evaluator_timeout",
                message=f"task evaluator timed out after {completed.get('timeout_sec')}s",
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
                message=f"task evaluator exited with {completed.get('returncode')}",
                stdout_tail=str(completed.get("stdout_tail") or ""),
                stderr_tail=str(completed.get("stderr_tail") or ""),
                metadata=metadata,
            )
        payload = _parse_payload(str(completed.get("stdout") or ""))
        if payload is None:
            return _result(
                status="no_metric",
                metric_name=metric_name,
                lower_is_better=lower,
                validity="medium",
                reason="metric_parse_failed",
                message="task evaluator output did not contain a JSON object",
                stdout_tail=str(completed.get("stdout_tail") or ""),
                stderr_tail=str(completed.get("stderr_tail") or ""),
                metadata=metadata,
            )
        return _result_from_payload(payload, metric_name=metric_name, lower_is_better=lower, metadata=metadata)

    def _run_task_evaluator(
        self,
        ctx: EvalContext,
        candidate: CandidateRef,
        spec: TaskPackageSpec,
    ) -> dict[str, Any]:
        runtime = prepare_task_runtime(spec.task_id, task_root=ctx.task_root)
        current_sha = file_sha256(runtime.entrypoint_path)
        if current_sha != runtime.entrypoint_sha256:
            return {"hash_mismatch": True, "message": f"{runtime.entrypoint_path} changed after registration"}
        python = task_python_executable(ctx) or Path(sys.executable)
        config = _runner_config(ctx, spec, runtime.package_sha256)
        cmd = [
            str(python),
            "-m",
            "scienceflow.gates.evaluator.entrypoint_runner",
            "--entrypoint",
            str(runtime.entrypoint_path),
            "--function",
            runtime.entrypoint_function,
            "--artifact",
            str((ctx.workspace / candidate.artifact_path).resolve()),
            "--workspace",
            str(ctx.workspace.resolve()),
            "--task-dir",
            str(runtime.runtime_task_dir.resolve()),
            "--dataset-dir",
            str((ctx.workspace / "dataset").resolve()),
            "--config-json",
            json.dumps(config, sort_keys=True),
        ]
        env = task_command_env(ctx, python=python)
        env["PYTHONPATH"] = _prepend_env_path(str(repo_root()), env.get("PYTHONPATH", ""))
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ctx.workspace),
                env=env,
                text=True,
                capture_output=True,
                timeout=spec.evaluator_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return {
                "timeout": True,
                "timeout_sec": spec.evaluator_timeout_sec,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_tail": _tail(stdout),
                "stderr_tail": _tail(stderr),
            }
        return {
            "timeout": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
        }

    def to_metric_event(
        self,
        ctx: EvalContext,
        candidate: CandidateRef,
        result: EvaluationResult,
    ) -> MetricEvent:
        spec = _spec(ctx)
        payload_extra = (
            result.metadata.get("payload_extra") if isinstance(result.metadata, Mapping) else None
        )
        extra = dict(payload_extra) if isinstance(payload_extra, Mapping) else {}
        run_time = _as_float(result.metadata.get("run_time_sec") if isinstance(result.metadata, Mapping) else None)
        extra.update(
            {
                "artifact_kind": candidate.artifact_kind,
                "stdout_tail": result.stdout_tail,
                "stderr_tail": result.stderr_tail,
            }
        )
        if spec is not None:
            extra.update(
                {
                    "task_category": spec.category,
                    "task_provider": spec.provider,
                    "metric_authoritative": spec.metric_authoritative,
                }
            )
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
            metric_type=_metric_type(ctx, spec),
            task_profile=ctx.task_profile or (spec.profile if spec else ""),
            metric_note=result.message,
            run_time_sec=run_time,
            extra=extra,
        )


def _runner_config(ctx: EvalContext, spec: TaskPackageSpec, package_sha: str) -> dict[str, Any]:
    metric_event = ctx.metadata.get("metric_event") if isinstance(ctx.metadata, Mapping) else {}
    if not isinstance(metric_event, Mapping):
        metric_event = {}
    return {
        "task": dict(spec.config),
        "task_id": spec.task_id,
        "task_profile": ctx.task_profile or spec.profile,
        "metric_event": dict(metric_event),
        "evaluation_context": {
            "worker_id": str(ctx.worker_id or ""),
            "stage_id": str(ctx.stage_id or ""),
            "query_budget_scope": _query_budget_scope(ctx),
        },
        "mlebench_data_root_dir": str(_cfg(ctx.cfg, "mlebench_data_root_dir", "") or ""),
        "package_sha256": package_sha,
    }


def _query_budget_scope(ctx: EvalContext) -> str:
    evaluator = _cfg(ctx.cfg, "evaluator", None)
    scope = str(_cfg(evaluator, "query_budget_scope", "task") or "task").strip().lower()
    if scope not in {"task", "worker"}:
        raise ValueError(
            "evaluator.query_budget_scope must be either 'task' or 'worker'"
        )
    if scope == "worker" and not str(ctx.worker_id or "").strip():
        raise ValueError(
            "evaluator.query_budget_scope=worker requires a trusted worker_id"
        )
    return scope


def _result_from_payload(
    payload: Mapping[str, Any],
    *,
    metric_name: str,
    lower_is_better: bool | None,
    metadata: dict[str, Any],
) -> EvaluationResult:
    metric = payload.get("metric") if isinstance(payload.get("metric"), Mapping) else {}
    value = _as_float(metric.get("value"))
    name = str(metric.get("name") or metric_name or "metric")
    lower = _as_bool(metric.get("lower_is_better"), default=lower_is_better)
    valid = bool(payload.get("valid"))
    ready = bool(payload.get("candidate_ready", valid))
    status = str(payload.get("status") or ("ok" if valid else "invalid_candidate"))
    reason = str(payload.get("reason_code") or ("task_package_evaluator_ok" if valid else "invalid_candidate"))
    message = str(payload.get("feedback") or payload.get("message") or "")
    selection = bool(payload.get("selection_eligible", valid and value is not None))
    validity = str(payload.get("metric_validity") or ("high" if selection else "low" if not valid else "medium"))
    payload_extra = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}
    metadata = {**metadata, "payload_extra": dict(payload_extra)}
    return EvaluationResult(
        backend="task_package",
        ok=valid,
        status=status,
        metric_value=value,
        metric_name=name,
        lower_is_better=lower,
        validation_ok=valid,
        candidate_ready=ready,
        selection_eligible=selection,
        metric_validity=validity,
        reason_code=reason,
        message=message,
        metadata=metadata,
    )


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
        backend="task_package",
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


def _spec(ctx: EvalContext) -> TaskPackageSpec | None:
    try:
        return find_task_package(ctx.task_id)
    except Exception:
        return None


def _artifact_path(ctx: EvalContext, spec: TaskPackageSpec | None) -> str:
    configured = _nested_cfg(ctx, ("candidate", "artifact"), None)
    if spec is not None and str(configured or "") in {"", "submission.csv"}:
        return spec.artifact_path
    return _clean_relpath(configured or (spec.artifact_path if spec else None), "submission.csv")


def _artifact_kind(ctx: EvalContext, spec: TaskPackageSpec | None) -> str:
    configured = _nested_cfg(ctx, ("candidate", "artifact_kind"), None)
    if spec is not None and str(configured or "") in {"", "submission_csv"}:
        return spec.artifact_kind
    return str(configured or (spec.artifact_kind if spec else "artifact"))


def _metric_name(ctx: EvalContext, spec: TaskPackageSpec | None) -> str:
    configured = _nested_cfg(ctx, ("metric", "name"), None)
    if spec is not None and str(configured or "") in {"", "metric", "Final Validation Score"}:
        return spec.metric_name
    return str(configured or (spec.metric_name if spec else "metric"))


def _metric_type(ctx: EvalContext, spec: TaskPackageSpec | None) -> str:
    configured = _nested_cfg(ctx, ("metric", "type"), None)
    if spec is not None and str(configured or "") in {"", "holdout", "benchmark"}:
        return spec.metric_type
    return str(configured or (spec.metric_type if spec else "benchmark"))


def _lower_is_better(ctx: EvalContext, spec: TaskPackageSpec | None) -> bool | None:
    return _as_bool(_nested_cfg(ctx, ("metric", "lower_is_better"), None), default=spec.lower_is_better if spec else None)


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    return ""


def _candidate_id(ctx: EvalContext, sha: str) -> str:
    worker = str(ctx.worker_id or "W00").strip()
    stage = str(ctx.stage_id or "").strip()
    if stage:
        return f"{worker}:{stage}"
    return f"{worker}:candidate:{sha[:12] if sha else 'missing'}"


def _parse_payload(stdout: str) -> Mapping[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, Mapping):
            return data
    return None


def _nested_cfg(ctx: EvalContext, path: tuple[str, str], default: Any) -> Any:
    evaluator = _cfg(ctx.cfg, "evaluator", None)
    block = _cfg(evaluator, path[0], None)
    value = _cfg(block, path[1], None)
    return default if value is None else value


def _cfg(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _bool_cfg(ctx: EvalContext, path: tuple[str, str], default: bool) -> bool:
    return bool(_as_bool(_nested_cfg(ctx, path, default), default=default))


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _as_bool(value: Any, *, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "lower"}:
        return True
    if text in {"0", "false", "no", "n", "off", "higher"}:
        return False
    return default


def _clean_relpath(value: Any, default: str) -> str:
    text = str(value or default).replace("\\", "/").strip().lstrip("/")
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return default
    return "/".join(parts)


def _tail(text: str, max_chars: int = 4000) -> str:
    return text[-max_chars:] if max_chars > 0 else ""


def _prepend_env_path(path: str, existing: str) -> str:
    return path if not existing else path + ":" + existing
