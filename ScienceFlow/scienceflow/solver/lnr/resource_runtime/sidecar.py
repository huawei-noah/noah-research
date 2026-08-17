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
import os
import time
import uuid
from pathlib import Path
from typing import Any


CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth", ".pkl", ".joblib", ".npy", ".npz"}
CODE_NAMES = {"train.py", "predict.py", "inference.py", "submit.py", "solution.py"}


def run_cpu_sidecar_backfill(
    *,
    resource_dir: Path,
    workspace_dir: Path,
    parent_job_id: str,
    parent_state: str,
    elapsed_sec: float,
    parent_worker_id: str = "",
    task_type: str = "submission_checker",
    budget_sec: float = 900.0,
) -> dict[str, Any]:
    """Create a small CPU-only sidecar report without touching parent files."""
    resource_root = Path(resource_dir)
    workspace = Path(workspace_dir)
    sidecar_id = f"sc_{uuid.uuid4().hex[:10]}"
    sidecar_dir = resource_root / "sidecars" / sidecar_id
    sidecar_workspace = sidecar_dir / "workspace"
    sidecar_workspace.mkdir(parents=True, exist_ok=True)

    candidate_artifact = _candidate_artifact_path()
    candidate_rel = _safe_rel(workspace / candidate_artifact, workspace)
    sample = _safe_rel(workspace / "dataset" / "sample_submission.csv", workspace)
    root_submission = _safe_rel(workspace / "submission.csv", workspace)
    checkpoints = _find_recent_artifacts(workspace, suffixes=CHECKPOINT_SUFFIXES, limit=12)
    code_files = [name for name in sorted(CODE_NAMES) if (workspace / name).is_file()]
    checker_path = ""
    if sample:
        checker_path = _write_submission_checker(sidecar_workspace)

    observations = []
    if sample and candidate_artifact == "submission.csv" and not root_submission:
        observations.append("sample_submission exists but root submission.csv is not present")
    if candidate_artifact != "submission.csv" and not candidate_rel:
        observations.append(f"configured candidate artifact is not present: {candidate_artifact}")
    if checkpoints:
        observations.append("checkpoint or model artifacts are present")
    if "predict.py" not in code_files and "inference.py" not in code_files:
        observations.append("no obvious prediction script is present")
    if not observations:
        observations.append("no immediate submission gap detected")

    useful_artifacts: list[dict[str, str]] = []
    if checker_path:
        useful_artifacts.append({"path": checker_path, "type": "submission_checker"})

    quality_gate = "review" if observations else "discard"
    sanity_checks = {
        "sidecar_writes_isolated": True,
        "sample_submission_found": bool(sample),
        "root_submission_found": bool(root_submission),
        "candidate_artifact_found": bool(candidate_rel),
        "checker_created": bool(checker_path),
    }
    report = {
        "report_version": 1,
        "sidecar_id": sidecar_id,
        "status": "complete",
        "task_type": str(task_type or "submission_checker"),
        "parent_worker_id": str(parent_worker_id or ""),
        "parent_job_id": str(parent_job_id or ""),
        "parent_state": str(parent_state or ""),
        "elapsed_sec": float(elapsed_sec or 0.0),
        "budget_sec": float(budget_sec or 0.0),
        "quality_gate": quality_gate,
        "sanity_checks": sanity_checks,
        "observations": observations,
        "recommended_next_action": (
            "review submission path and prediction readiness"
            if quality_gate == "review"
            else "discard sidecar report"
        ),
        "useful_artifacts": useful_artifacts,
        "workspace_facts": {
            "sample_submission": sample,
            "root_submission": root_submission,
            "candidate_artifact": candidate_rel,
            "code_files": code_files,
            "checkpoint_artifacts": checkpoints,
        },
        "created_at": time.time(),
    }
    report_path = sidecar_dir / "sidecar_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = report_path.relative_to(resource_root).as_posix()
    return report


def checkpoint_submission_gap(
    *,
    workspace_dir: Path,
    started_at: float,
    candidate_artifact: str = "",
) -> dict[str, Any] | None:
    workspace = Path(workspace_dir)
    artifact_rel = _candidate_artifact_path(candidate_artifact)
    checkpoints = _find_recent_artifacts(
        workspace,
        suffixes=CHECKPOINT_SUFFIXES,
        limit=8,
        min_mtime=float(started_at or 0.0) - 1.0,
    )
    if not checkpoints:
        return None
    artifact = workspace / artifact_rel
    artifact_updated = False
    if artifact.is_file():
        try:
            artifact_updated = artifact.stat().st_mtime + 1.0 >= float(started_at or 0.0)
        except OSError:
            artifact_updated = False
    if artifact_updated:
        return None
    reason = "checkpoint_without_submission" if artifact_rel == "submission.csv" else "checkpoint_without_candidate_artifact"
    return {
        "reason": reason,
        "checkpoint_artifacts": checkpoints,
        "artifact_path": artifact_rel,
        "artifact_updated": False,
        "submission_path": "submission.csv" if artifact_rel == "submission.csv" else "",
        "submission_updated": False if artifact_rel == "submission.csv" else None,
    }


def _candidate_artifact_path(raw: str = "") -> str:
    candidate = str(raw or os.environ.get("SCIENCEFLOW_CANDIDATE_ARTIFACT") or "submission.csv").strip()
    path = Path(candidate)
    if not candidate or path.is_absolute() or ".." in path.parts:
        return "submission.csv"
    return path.as_posix()


def _write_submission_checker(sidecar_workspace: Path) -> str:
    path = sidecar_workspace / "submission_checker.py"
    path.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "import pandas as pd\n\n"
        "sub = Path(sys.argv[1] if len(sys.argv) > 1 else 'submission.csv')\n"
        "sample = Path(sys.argv[2] if len(sys.argv) > 2 else 'dataset/sample_submission.csv')\n"
        "s = pd.read_csv(sub)\n"
        "t = pd.read_csv(sample)\n"
        "assert list(s.columns) == list(t.columns), (s.columns.tolist(), t.columns.tolist())\n"
        "assert len(s) == len(t), (len(s), len(t))\n"
        "print('submission schema ok')\n",
        encoding="utf-8",
    )
    return path.name


def _find_recent_artifacts(
    workspace: Path,
    *,
    suffixes: set[str],
    limit: int,
    min_mtime: float = 0.0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    excluded = {".git", ".venv", "__pycache__", "dataset", "data", "input", "logs"}
    for root, dirnames, filenames in os.walk(workspace):
        root_path = Path(root)
        parts = set(root_path.relative_to(workspace).parts) if root_path != workspace else set()
        if parts & excluded:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in excluded]
        for name in filenames:
            path = root_path / name
            if path.suffix.lower() not in suffixes:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime < min_mtime:
                continue
            out.append({
                "path": _safe_rel(path, workspace),
                "mtime": stat.st_mtime,
                "size_bytes": int(stat.st_size),
            })
    out.sort(key=lambda item: float(item.get("mtime") or 0.0), reverse=True)
    return out[: max(0, int(limit or 0))]


def _safe_rel(path: Path, root: Path) -> str:
    try:
        if not path.exists():
            return ""
        return path.relative_to(root).as_posix()
    except (OSError, ValueError):
        return ""
