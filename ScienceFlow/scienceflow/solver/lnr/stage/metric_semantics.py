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
import shlex
from typing import Any

from scienceflow.solver.lnr.stage.metric_validity import infer_metric_validity
from scienceflow.solver.lnr.resource_runtime.review.research_cadence import (
    normalize_execution_scale,
    normalize_route_key,
)


METRIC_SEMANTIC_TYPES = {"holdout", "cv", "post_fulltrain_replay", "training_loss", "unknown"}
TRAIN_DATA_USED_TYPES = {"train_only", "train_plus_validation", "cv_folds", "unknown"}
METRIC_EVAL_DATA_TYPES = {"validation", "oof", "train", "unknown"}

_METRIC_DECL_RE = re.compile(
    r"^\s*(?P<key>metric[_\s-]*protocol|train[_\s-]*data[_\s-]*used|"
    r"metric[_\s-]*eval[_\s-]*data|artifacts[_\s-]*reused[_\s-]*from|"
    r"training[_\s-]*rows|validation[_\s-]*rows|route[_\s-]*id|"
    r"execution[_\s-]*scale)\s*[:=]\s*(?P<value>.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def normalize_metric_semantics_value(value: Any, allowed: set[str], default: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in allowed:
        return text
    return default


def extract_metric_semantics_declarations(data: dict[str, Any], text: str) -> dict[str, Any]:
    declarations: dict[str, Any] = {}
    key_map = {
        "metric_protocol": "metric_protocol",
        "train_data_used": "train_data_used",
        "metric_eval_data": "metric_eval_data",
        "artifacts_reused_from": "artifacts_reused_from",
        "training_rows": "training_rows",
        "validation_rows": "validation_rows",
        "route_id": "route_id",
        "execution_scale": "execution_scale",
    }
    for raw_key, dst in key_map.items():
        value = data.get(raw_key)
        if value not in (None, ""):
            declarations[dst] = value
    for match in _METRIC_DECL_RE.finditer(str(text or "")):
        key = match.group("key").lower().replace("-", "_").replace(" ", "_")
        value = match.group("value").strip()
        if key in key_map and key_map[key] not in declarations:
            declarations[key_map[key]] = value
    if "metric_protocol" in declarations:
        declarations["metric_protocol"] = normalize_metric_semantics_value(
            declarations["metric_protocol"], METRIC_SEMANTIC_TYPES
        )
    if "train_data_used" in declarations:
        declarations["train_data_used"] = normalize_metric_semantics_value(
            declarations["train_data_used"], TRAIN_DATA_USED_TYPES
        )
    if "metric_eval_data" in declarations:
        declarations["metric_eval_data"] = normalize_metric_semantics_value(
            declarations["metric_eval_data"], METRIC_EVAL_DATA_TYPES
        )
    for key in ("training_rows", "validation_rows"):
        if key in declarations:
            try:
                declarations[key] = int(float(str(declarations[key]).strip()))
            except (TypeError, ValueError):
                declarations[key] = ""
    hint = _resource_value_hint_from_command(str(data.get("bash_cmd") or ""))
    if "route_id" not in declarations and hint.get("route_id"):
        declarations["route_id"] = hint["route_id"]
    if "execution_scale" not in declarations and hint.get("execution_scale"):
        declarations["execution_scale"] = hint["execution_scale"]
    if "route_id" in declarations:
        declarations["route_id"] = normalize_route_key(declarations["route_id"])
    if "execution_scale" in declarations:
        declarations["execution_scale"] = normalize_execution_scale(declarations["execution_scale"])
    return declarations


def _resource_value_hint_from_command(command: str) -> dict[str, Any]:
    try:
        tokens = shlex.split(str(command or ""), posix=True)
    except ValueError:
        return {}
    for token in tokens[:12]:
        if not token.startswith(
            (
                "SCIENCEFLOW_RESOURCE_VALUE_HINT=",
                "_SCIENCEFLOW_RESOURCE_VALUE_HINT=",
            )
        ):
            continue
        try:
            value = json.loads(token.split("=", 1)[1])
        except (json.JSONDecodeError, IndexError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def infer_execution_mode(command: str, text: str, *, source_changed: Any = None) -> str:
    haystack = f"{command}\n{text}".lower()
    runs_train = bool(re.search(r"\b(?:python[0-9.]*|uv\s+run\s+python)\b[^\n;&|]*\btrain[\w.-]*\.py\b", haystack))
    runs_predict = bool(re.search(r"\b(?:python[0-9.]*|uv\s+run\s+python)\b[^\n;&|]*\bpredict[\w.-]*\.py\b", haystack))
    loading_artifacts = (
        "loading artifacts" in haystack
        or "loaded model" in haystack
        or "pickle.load" in haystack
        or "model trained on train+val" in haystack
    )
    if runs_train and runs_predict:
        return "train_and_predict"
    if runs_train:
        return "train_only_or_training_metric"
    if runs_predict and loading_artifacts:
        return "predict_only_reuse_artifacts"
    if runs_predict:
        return "predict_only"
    if source_changed is False and loading_artifacts:
        return "predict_only_reuse_artifacts"
    return "unknown"


def looks_like_fulltrain_replay(text: str, declarations: dict[str, Any]) -> bool:
    lowered = str(text or "").lower()
    declared_fulltrain = (
        declarations.get("train_data_used") == "train_plus_validation"
        and declarations.get("metric_eval_data") == "validation"
    )
    fulltrain_artifact_phrase = bool(
        re.search(
            r"(trained on train\+val|trained on train\s*\+\s*val|train\+validation|"
            r"train\s*\+\s*validation|model trained on train\+val|n_full)",
            lowered,
        )
    )
    replay_eval_phrase = bool(
        re.search(
            r"(loading validation features|evaluating on validation|validation features for score computation|"
            r"predict validation|predicting validation)",
            lowered,
        )
    )
    validation_metric = "final validation score" in lowered or "validation rmsle" in lowered
    return bool(declared_fulltrain or (fulltrain_artifact_phrase and replay_eval_phrase and validation_metric))


def looks_like_cv_metric(text: str, declarations: dict[str, Any]) -> bool:
    if declarations.get("metric_protocol") == "cv" or declarations.get("metric_eval_data") == "oof":
        return True
    lowered = str(text or "").lower()
    return bool(re.search(r"\b(oof|kfold|k-fold|cross[- ]validation|cv rmsle|fold\s+\d+/\d+)\b", lowered))


def looks_like_training_loss(text: str, declarations: dict[str, Any]) -> bool:
    if declarations.get("metric_protocol") == "training_loss" or declarations.get("metric_eval_data") == "train":
        return True
    lowered = str(text or "").lower()
    return "training loss" in lowered and "validation" not in lowered


def classify_metric_semantics(
    *,
    metric_value: float | None,
    data: dict[str, Any],
    workspace_source_text: str,
    source_changed: Any = None,
) -> dict[str, Any]:
    command = str(data.get("bash_cmd") or "")
    stdout_tail = str(data.get("stdout_tail") or "")
    stderr_tail = str(data.get("stderr_tail") or "")
    audit_text = "\n".join([command, stdout_tail, stderr_tail, workspace_source_text])
    declarations = extract_metric_semantics_declarations(data, audit_text)
    declared_protocol = declarations.get("metric_protocol") or "unknown"
    execution_mode = infer_execution_mode(command, audit_text, source_changed=source_changed)

    if looks_like_fulltrain_replay(audit_text, declarations):
        val_score_type = "post_fulltrain_replay"
    elif looks_like_cv_metric(audit_text, declarations):
        val_score_type = "cv"
    elif looks_like_training_loss(audit_text, declarations):
        val_score_type = "training_loss"
    elif declared_protocol in {"holdout", "cv", "training_loss"}:
        val_score_type = declared_protocol
    elif "final validation score" in audit_text.lower() or "validation rmsle" in audit_text.lower():
        val_score_type = "holdout"
    else:
        val_score_type = "unknown"

    selection_eligible = val_score_type in {"holdout", "cv"} and metric_value is not None
    selection_score = metric_value if selection_eligible else ""
    note = ""
    if val_score_type == "post_fulltrain_replay":
        note = "fulltrain_replay_metric_kept_but_not_used_for_selection"
        if declared_protocol == "holdout":
            note = "agent_declared_holdout_but_system_detected_fulltrain_replay"
    elif val_score_type == "training_loss":
        note = "training_loss_metric_kept_but_not_used_for_selection"
    elif val_score_type == "unknown":
        note = "insufficient_metric_semantics_evidence"

    metric_validity, metric_validity_note = infer_metric_validity(
        {
            **data,
            "val_score_type": val_score_type,
            "selection_eligible": selection_eligible,
            "selection_note": note,
            "metric_protocol": declared_protocol,
            "metric_eval_data": declarations.get("metric_eval_data") or "unknown",
            "train_data_used": declarations.get("train_data_used") or "unknown",
            "execution_mode": execution_mode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    )

    return {
        "reported_val_score": metric_value,
        "val_score_type": val_score_type,
        "selection_eligible": selection_eligible,
        "selection_score": selection_score,
        "selection_note": note,
        "metric_validity": metric_validity,
        "metric_validity_note": metric_validity_note,
        "metric_protocol": declared_protocol,
        "train_data_used": declarations.get("train_data_used") or "unknown",
        "metric_eval_data": declarations.get("metric_eval_data") or "unknown",
        "artifacts_reused_from": declarations.get("artifacts_reused_from") or "",
        "training_rows": declarations.get("training_rows") or "",
        "validation_rows": declarations.get("validation_rows") or "",
        "execution_mode": execution_mode,
        "route_id": declarations.get("route_id") or "",
        "execution_scale": declarations.get("execution_scale") or "unknown",
    }
