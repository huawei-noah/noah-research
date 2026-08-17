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

"""Trusted adapter for the official SciModelingBench TFBind8 task."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


TASK_ID = "sci-modeling-bench-tfbind8-v1"
METRIC_NAME = "top_1_normalized_e_score"
PACKAGE_VERSION = "0.2.0"
REPO_ID = "sci-modeling-bench/design-bench"
CONFIG_NAME = "tfbind8"
SPLIT = "six6_ref_r1"
REVISION = "2ee2856f4255bb6a64c11b6c2660a6f41418e654"
PROTOCOL_ID = "design-bench/tfbind8-bottom-percentile-v1"


def evaluate(
    *,
    artifact_path: Path,
    workspace_dir: Path,
    task_dir: Path,
    dataset_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    _ = workspace_dir, task_dir, config
    try:
        manifest = _validate_public_dataset(dataset_dir)
        candidates = _load_candidates(artifact_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _invalid_payload("tfbind8_input_invalid", str(exc))

    try:
        result = _load_task().evaluate(candidates)
        official = json.loads(result.model_dump_json())
    except (ImportError, ModuleNotFoundError) as exc:
        return _invalid_payload(
            "tfbind8_dependency_unavailable",
            f"install sci-modeling-bench=={PACKAGE_VERSION}: {exc}",
        )
    except Exception as exc:
        return _invalid_payload("tfbind8_official_evaluator_failed", str(exc))

    metrics = official.get("metrics")
    score = metrics.get(METRIC_NAME) if isinstance(metrics, dict) else None
    submission_valid = bool(official.get("submission_valid"))
    valid_candidates = int(official.get("valid_candidates") or 0)
    invalid_candidates = int(official.get("invalid_candidates") or 0)
    submitted_candidates = int(official.get("submitted_candidates") or 0)
    candidate_ready = submission_valid and valid_candidates > 0

    if not submission_valid:
        status = "invalid_submission"
        reason_code = "tfbind8_submission_invalid"
    elif not candidate_ready:
        status = "all_candidates_invalid"
        reason_code = "tfbind8_all_candidates_invalid"
    else:
        status = "ok"
        reason_code = "tfbind8_official_objective_scored"

    return {
        "valid": candidate_ready,
        "candidate_ready": candidate_ready,
        "selection_eligible": candidate_ready,
        "status": status,
        "metric": {
            "name": METRIC_NAME,
            "value": score if candidate_ready else None,
            "lower_is_better": False,
        },
        "metric_validity": "high" if candidate_ready else "low",
        "reason_code": reason_code,
        "feedback": (
            f"official TFBind8 evaluation: submitted={submitted_candidates}, "
            f"valid={valid_candidates}, invalid={invalid_candidates}, score={score}"
        ),
        "extra": {
            "official_task_id": official.get("task_id"),
            "official_score": score,
            "expected_candidates": int(official.get("expected_candidates") or 128),
            "submitted_candidates": submitted_candidates,
            "valid_candidates": valid_candidates,
            "invalid_candidates": invalid_candidates,
            "all_candidates_valid": bool(official.get("all_candidates_valid")),
            "best_candidate_index": official.get("best_candidate_index"),
            "best_objective_output": official.get("best_objective_output"),
            "aggregation": official.get("aggregation"),
            "dataset_revision": manifest["revision"],
            "dataset_repo_id": manifest["repo_id"],
            "protocol_id": manifest["protocol_id"],
            "evaluator": "sci_modeling_bench_official_task",
            "package_version": PACKAGE_VERSION,
        },
    }


@lru_cache(maxsize=1)
def _load_task() -> Any:
    from sci_modeling_bench.suites.design_bench import (
        TFBind8BlackBoxOptimizationTask,
    )

    return TFBind8BlackBoxOptimizationTask.from_hub(
        repo_id=REPO_ID,
        config_name=CONFIG_NAME,
        revision=REVISION,
    )


def _load_candidates(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list):
        raise ValueError("submission must be a JSON array or an object with a candidates array")
    return candidates


def _validate_public_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    data_path = dataset_dir / "offline_data.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("dataset_manifest.json must contain a JSON object")

    expected = {
        "task_id": TASK_ID,
        "repo_id": REPO_ID,
        "config_name": CONFIG_NAME,
        "split": SPLIT,
        "revision": REVISION,
        "protocol_id": PROTOCOL_ID,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"dataset manifest {field} mismatch: expected {value!r}, "
                f"got {manifest.get(field)!r}"
            )
    expected_sha = str(manifest.get("offline_data_sha256") or "")
    if len(expected_sha) != 64:
        raise ValueError("dataset manifest has no valid offline_data_sha256")
    actual_sha = _sha256(data_path)
    if actual_sha != expected_sha:
        raise ValueError("offline_data.csv does not match the pinned dataset manifest")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _invalid_payload(reason_code: str, feedback: str) -> dict[str, Any]:
    return {
        "valid": False,
        "candidate_ready": False,
        "selection_eligible": False,
        "status": "invalid",
        "metric": {
            "name": METRIC_NAME,
            "value": None,
            "lower_is_better": False,
        },
        "metric_validity": "low",
        "reason_code": reason_code,
        "feedback": feedback,
    }
