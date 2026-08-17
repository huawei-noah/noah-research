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

"""Shared MLE-bench task-package evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scienceflow.gates.evaluator.providers.mlebench import (
    resolve_mlebench_exp_id,
    validate_submission_light,
    validate_submission_local,
)


def evaluate(
    *,
    artifact_path: Path,
    workspace_dir: Path,
    task_dir: Path,
    dataset_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    _ = task_dir, dataset_dir
    metric_event = config.get("metric_event") if isinstance(config.get("metric_event"), Mapping) else {}
    task_cfg = config.get("task") if isinstance(config.get("task"), Mapping) else {}
    metric_cfg = task_cfg.get("metric") if isinstance(task_cfg.get("metric"), Mapping) else {}
    ok, result = _validate_submission(
        workspace_dir=workspace_dir,
        artifact_path=artifact_path,
        task_id=str(config.get("task_id") or task_cfg.get("id") or ""),
        data_root=str(config.get("mlebench_data_root_dir") or ""),
    )
    is_valid = bool(result.get("is_valid")) if isinstance(result, Mapping) else False
    message = str(result.get("result") or "") if isinstance(result, Mapping) else ""
    metric_value = _as_float(metric_event.get("metric_value"))
    metric_name = str(
        metric_event.get("metric_name")
        or metric_cfg.get("name")
        or "Final Validation Score"
    )
    lower = _as_bool(metric_event.get("lower_is_better"), default=_as_bool(metric_cfg.get("lower_is_better")))
    if not ok:
        status = "validator_error"
        validity = "low"
        reason = "mlebench_validator_error"
    elif not is_valid:
        status = "invalid_submission"
        validity = "low"
        reason = "invalid_submission"
    elif metric_value is None:
        status = "validated_no_metric"
        validity = "medium"
        reason = "candidate_validated_without_metric"
    else:
        status = "ok"
        validity = "high"
        reason = "mlebench_submission_valid"
    return {
        "valid": bool(ok and is_valid),
        "candidate_ready": bool(is_valid),
        "selection_eligible": bool(is_valid and metric_value is not None),
        "status": status,
        "metric": {
            "name": metric_name,
            "value": metric_value,
            "lower_is_better": lower,
        },
        "metric_validity": validity,
        "reason_code": reason,
        "feedback": message,
        "extra": {
            "deliverable_role": "submission_csv",
            "validator": dict(result or {}),
            "artifact_kind": "submission_csv",
        },
    }


def _validate_submission(
    *,
    workspace_dir: Path,
    artifact_path: Path,
    task_id: str,
    data_root: str,
) -> tuple[bool, dict[str, Any]]:
    if data_root:
        resolved = resolve_mlebench_exp_id(workspace_dir, cfg_exp_id=task_id)
        if resolved:
            return validate_submission_local(resolved, artifact_path, data_root)
    is_valid, message = validate_submission_light(workspace_dir, submission_name=artifact_path.name or "submission.csv")
    return True, {"is_valid": is_valid, "result": message}


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
