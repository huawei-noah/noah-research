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

"""MLEBench-specific submission checks."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import yaml

from scienceflow.core.artifact_io import wait_for_stable_file
from scienceflow.utils.node_paths import find_node_context_path

logger = logging.getLogger("scienceflow")

_SAMPLE_SUBMISSION_PATTERNS = (
    "sample_submission.csv",
    "*sample_submission*.csv",
)


def resolve_mlebench_exp_id(workspace_dir: Path, *, cfg_exp_id: str = "") -> str | None:
    """Resolve the MLEBench competition id from explicit or legacy runtime context."""
    explicit = (cfg_exp_id or "").strip()
    if explicit:
        return explicit
    workspace = Path(workspace_dir).resolve()
    context_path = find_node_context_path(workspace)
    if context_path.is_file():
        try:
            context = json.loads(context_path.read_text(encoding="utf-8"))
            exp_id = context.get("exp_id")
            if isinstance(exp_id, str) and exp_id.strip():
                return exp_id.strip()
        except (OSError, json.JSONDecodeError):
            pass
    if workspace.parent.name == "wsp":
        return workspace.parent.parent.name
    for parent in (workspace, *workspace.parents):
        config_path = parent / "resolved_config.yaml"
        if not config_path.is_file():
            continue
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            continue
        if isinstance(raw, dict):
            exp_id = raw.get("exp_id") or raw.get("task_id")
            if isinstance(exp_id, str) and exp_id.strip():
                return exp_id.strip()
    return None


def validate_submission_local(
    exp_id: str,
    submission_path: Path,
    mlebench_data_dir: str | Path,
) -> tuple[bool, dict]:
    """Validate submission format through the local MLEBench SDK."""
    submission_path = Path(submission_path)
    data_dir = Path(mlebench_data_dir).resolve()
    try:
        from mlebench.grade import validate_submission as mlebench_validate
        from mlebench.registry import registry
    except ImportError as exc:
        logger.info(
            "mlebench import failed, skipping submission validation (executable=%s): %s",
            sys.executable,
            exc,
        )
        detail = str(exc).strip() or exc.__class__.__name__
        return True, {
            "is_valid": True,
            "result": f"mlebench not installed, skipped: {detail}",
        }
    if not submission_path.is_file():
        return True, {
            "is_valid": False,
            "result": f"Submission file does not exist: {submission_path}",
        }
    stable, stable_message = wait_for_stable_file(submission_path)
    if not stable:
        return True, {
            "is_valid": False,
            "result": stable_message,
            "transient": True,
        }
    try:
        competition = registry.set_data_dir(data_dir).get_competition(exp_id)
        is_valid, message = mlebench_validate(submission_path, competition)
        return True, {"is_valid": is_valid, "result": message}
    except ValueError as exc:
        logger.warning("mlebench validation failed (ValueError): %s", exc)
        return False, {"is_valid": False, "result": str(exc)}
    except Exception as exc:
        logger.exception("mlebench validation error: %s", exc)
        return False, {"is_valid": False, "result": str(exc)}


def validate_submission_light(
    workspace_dir: Path,
    *,
    submission_name: str = "submission.csv",
) -> tuple[bool, str]:
    """Apply the provider's conservative sample-submission CSV fallback."""
    workspace = Path(workspace_dir).resolve()
    submission = workspace / submission_name
    if not submission.is_file():
        return False, f"{submission_name} missing"
    stable, stable_message = wait_for_stable_file(submission)
    if not stable:
        return False, stable_message
    try:
        raw = submission.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return False, f"cannot read {submission_name}: {exc}"
    if not raw.strip():
        return False, f"{submission_name} empty"
    sample = _find_sample_csv(workspace / "dataset")
    try:
        submission_rows = _read_csv_rows(submission)
        sample_rows = _read_csv_rows(sample) if sample is not None else None
    except OSError as exc:
        return False, f"csv read error: {exc}"
    ok, message = _check_csv_shape_and_required_cells(
        submission_rows,
        sample_rows=sample_rows,
        submission_name=submission_name,
    )
    if not ok:
        return False, message
    if sample_rows is None:
        return True, "no sample csv found; csv structure and required cells ok"
    return True, f"matches {sample.name}; required cells ok"


def _find_sample_csv(dataset_dir: Path) -> Path | None:
    if not dataset_dir.is_dir():
        return None
    for pattern in _SAMPLE_SUBMISSION_PATTERNS:
        matches = sorted(path for path in dataset_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.reader(handle))


def _check_csv_shape_and_required_cells(
    rows: list[list[str]],
    *,
    sample_rows: list[list[str]] | None,
    submission_name: str,
) -> tuple[bool, str]:
    if not rows:
        return False, f"{submission_name} empty csv"
    header = rows[0]
    if not header or any(_is_blank_cell(cell) for cell in header):
        return False, "submission header has blank column name"
    if len(set(header)) != len(header):
        return False, "submission header has duplicate columns"
    if sample_rows is not None:
        if not sample_rows:
            return False, "empty sample submission csv"
        if header != sample_rows[0]:
            return False, "submission header does not match sample submission"
        if len(rows) != len(sample_rows):
            return False, f"submission rows {len(rows)} != sample rows {len(sample_rows)}"
    if len(rows) <= 1:
        return False, f"{submission_name} has no data rows"
    expected_cols = len(header)
    for row_num, row in enumerate(rows[1:], start=2):
        if len(row) != expected_cols:
            return False, f"row {row_num} has {len(row)} columns, expected {expected_cols}"
        for col_name, cell in zip(header, row, strict=True):
            if _is_blank_cell(cell):
                return False, f"blank required cell at row {row_num}, column {col_name}"
    return True, "csv structure ok"


def _is_blank_cell(value: str) -> bool:
    return value.strip() == ""
