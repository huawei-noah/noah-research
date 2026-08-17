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

import csv
import os
import time
from pathlib import Path
from typing import Any

_SAMPLE_SUBMISSION_CANDIDATES = (
    "sample_submission.csv",
    "dataset/sample_submission.csv",
    "data/sample_submission.csv",
    "input/sample_submission.csv",
    "public/sample_submission.csv",
)
_EXCLUDED_DIRS = {
    ".git",
    ".scienceflow_checkpoints",
    "__pycache__",
    "data",
    "dataset",
    "input",
    "logs",
    "public",
    "submission_snapshots",
}
_DELIVERABLE_DIR_HINTS = {
    "artifact",
    "artifacts",
    "checkpoint",
    "checkpoints",
    "model",
    "models",
    "output",
    "outputs",
    "result",
    "results",
    "solution",
    "solutions",
}
_TRAIN_SUFFIXES = {".bin", ".ckpt", ".joblib", ".onnx", ".pkl", ".pt", ".pth", ".safetensors"}
_SOLVER_SUFFIXES = {".csv", ".json", ".jsonl", ".npy", ".npz", ".parquet", ".txt"}
_TRAIN_NAME_HINTS = ("best", "checkpoint", "ckpt", "model", "weights")
_SOLVER_NAME_HINTS = ("best", "output", "result", "route", "solution", "tour")


def submission_completion_state(workspace_dir: str | Path, *, settle_sec: float = 120.0) -> dict[str, Any]:
    ws = Path(workspace_dir)
    submission_path = ws / "submission.csv"
    if not submission_path.is_file():
        return {"complete": False, "mode": "submission", "reason": "missing_submission", "deliverable_validity": "none"}
    sample_path = find_sample_submission(ws)
    if sample_path is None:
        return {
            "complete": False,
            "mode": "submission",
            "reason": "missing_sample_submission",
            "submission_path": "submission.csv",
            "deliverable_validity": "produced_unknown",
        }
    try:
        submission_stat = submission_path.stat()
        sample_stat = sample_path.stat()
    except OSError:
        return {"complete": False, "mode": "submission", "reason": "stat_failed"}
    now = time.time()
    submission_rows = count_lines(submission_path)
    expected_rows = count_lines(sample_path)
    schema = submission_schema_state(submission_path, sample_path)
    age_sec = max(0.0, now - submission_stat.st_mtime)
    settled = age_sec >= max(0.0, float(settle_sec or 0.0))
    rows_complete = expected_rows > 1 and submission_rows >= expected_rows
    schema_valid = bool(schema.get("valid"))
    complete = rows_complete and schema_valid and settled
    if schema.get("invalid"):
        reason = str(schema.get("reason") or "invalid_submission_schema")
        validity = "produced_invalid"
    elif complete:
        reason = "complete"
        validity = "produced_valid"
    elif rows_complete and not schema_valid:
        reason = str(schema.get("reason") or "invalid_submission_schema")
        validity = "produced_invalid"
    else:
        reason = "incomplete_or_unsettled"
        validity = "none"
    return {
        "complete": bool(complete),
        "mode": "submission",
        "reason": reason,
        "submission_path": "submission.csv",
        "sample_submission_path": safe_rel(sample_path, ws),
        "submission_rows": int(submission_rows),
        "expected_rows": int(expected_rows),
        "schema": schema,
        "deliverable_validity": validity,
        "age_sec": age_sec,
        "settle_sec": max(0.0, float(settle_sec or 0.0)),
        "submission_size_bytes": int(submission_stat.st_size),
        "sample_size_bytes": int(sample_stat.st_size),
    }


def submission_schema_state(submission_path: str | Path, sample_path: str | Path) -> dict[str, Any]:
    submission = Path(submission_path)
    sample = Path(sample_path)
    sub_cols = csv_header(submission)
    sample_cols = csv_header(sample)
    if not sample_cols:
        return {"valid": False, "invalid": False, "reason": "missing_sample_header"}
    if not sub_cols:
        return {"valid": False, "invalid": True, "reason": "invalid_missing_submission_header", "sample_columns": sample_cols}
    missing = [col for col in sample_cols if col not in sub_cols]
    extra = [col for col in sub_cols if col not in sample_cols]
    order_matches = sub_cols == sample_cols
    valid = not missing and order_matches
    reason = "schema_ok" if valid else "invalid_submission_columns"
    return {
        "valid": bool(valid),
        "invalid": not valid,
        "reason": reason,
        "submission_columns": sub_cols,
        "sample_columns": sample_cols,
        "missing_columns": missing,
        "extra_columns": extra,
        "order_matches": order_matches,
    }


def deliverable_artifact_state(workspace_dir: str | Path, *, settle_sec: float = 120.0, max_seen: int = 1000) -> dict[str, Any]:
    ws = Path(workspace_dir)
    candidates = list(iter_deliverable_artifacts(ws, max_seen=max_seen))
    if not candidates:
        return {"complete": False, "mode": "artifact", "reason": "missing_deliverable_artifact"}
    latest = max(candidates, key=lambda item: float(item.get("mtime") or 0.0))
    age_sec = max(0.0, time.time() - float(latest.get("mtime") or 0.0))
    settled = age_sec >= max(0.0, float(settle_sec or 0.0))
    return {
        "complete": bool(settled),
        "mode": str(latest.get("mode") or "artifact"),
        "reason": "complete" if settled else "artifact_unsettled",
        "artifact_path": str(latest.get("path") or ""),
        "artifact_size_bytes": int(latest.get("size_bytes") or 0),
        "artifact_age_sec": age_sec,
        "settle_sec": max(0.0, float(settle_sec or 0.0)),
        "candidate_count": len(candidates),
        "latest_candidates": candidates[:10],
    }


def configured_artifact_state(workspace_dir: str | Path, candidate_artifact: str, *, settle_sec: float = 120.0) -> dict[str, Any]:
    ws = Path(workspace_dir)
    rel = safe_candidate_artifact(candidate_artifact)
    if not rel:
        return {"complete": False, "mode": "artifact", "reason": "missing_candidate_artifact_config"}
    path = ws / rel
    if not path.is_file():
        return {
            "complete": False,
            "mode": "artifact",
            "reason": "missing_candidate_artifact",
            "artifact_path": rel,
            "deliverable_validity": "none",
        }
    try:
        stat = path.stat()
    except OSError:
        return {
            "complete": False,
            "mode": "artifact",
            "reason": "candidate_artifact_stat_failed",
            "artifact_path": rel,
            "deliverable_validity": "produced_unknown",
        }
    if stat.st_size <= 0:
        return {
            "complete": False,
            "mode": "artifact",
            "reason": "candidate_artifact_empty",
            "artifact_path": rel,
            "artifact_size_bytes": int(stat.st_size),
            "deliverable_validity": "produced_invalid",
        }
    age_sec = max(0.0, time.time() - float(stat.st_mtime))
    settled = age_sec >= max(0.0, float(settle_sec or 0.0))
    return {
        "complete": bool(settled),
        "mode": "artifact",
        "reason": "complete" if settled else "candidate_artifact_unsettled",
        "artifact_path": rel,
        "artifact_size_bytes": int(stat.st_size),
        "artifact_age_sec": age_sec,
        "settle_sec": max(0.0, float(settle_sec or 0.0)),
        "deliverable_validity": "produced_valid" if settled else "produced_unknown",
    }


def safe_candidate_artifact(candidate_artifact: str) -> str:
    raw = str(candidate_artifact or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        return ""
    return path.as_posix()


def iter_deliverable_artifacts(workspace_dir: Path, *, max_seen: int = 1000):
    seen = 0
    try:
        walker = os.walk(workspace_dir)
    except OSError:
        return
    for root, dirnames, filenames in walker:
        root_path = Path(root)
        rel_parts = root_path.relative_to(workspace_dir).parts if root_path != workspace_dir else ()
        if set(rel_parts) & _EXCLUDED_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in _EXCLUDED_DIRS and not name.startswith(".")]
        dir_hint = bool(set(rel_parts) & _DELIVERABLE_DIR_HINTS)
        for filename in filenames:
            if filename.startswith("."):
                continue
            path = root_path / filename
            mode = classify_deliverable(path, dir_hint=dir_hint)
            if not mode:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size <= 0:
                continue
            yield {
                "path": safe_rel(path, workspace_dir),
                "mode": mode,
                "size_bytes": int(stat.st_size),
                "mtime": float(stat.st_mtime),
            }
            seen += 1
            if seen >= max_seen:
                return


def classify_deliverable(path: Path, *, dir_hint: bool = False) -> str:
    name = path.stem.lower()
    suffix = path.suffix.lower()
    if suffix in _TRAIN_SUFFIXES and (dir_hint or any(hint in name for hint in _TRAIN_NAME_HINTS)):
        return "train_artifact"
    if suffix in _SOLVER_SUFFIXES and (dir_hint or any(hint in name for hint in _SOLVER_NAME_HINTS)):
        return "solver_solution"
    return ""


def find_sample_submission(workspace_dir: Path) -> Path | None:
    for rel in _SAMPLE_SUBMISSION_CANDIDATES:
        path = workspace_dir / rel
        if path.is_file():
            return path
    try:
        children = list(workspace_dir.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir() or child.name.startswith(".") or child.name in _EXCLUDED_DIRS:
            continue
        path = child / "sample_submission.csv"
        if path.is_file():
            return path
    return None


def count_lines(path: Path) -> int:
    count = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                count += chunk.count(b"\n")
    except OSError:
        return 0
    try:
        if path.stat().st_size > 0:
            with path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    count += 1
    except OSError:
        pass
    return count


def csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                return [str(cell).strip() for cell in row]
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    return []


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name
