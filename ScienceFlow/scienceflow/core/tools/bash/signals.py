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
import re
import shlex
import time
from pathlib import Path
from typing import Any

from .progress_signals import parse_progress_signals


def _parse_progress_signals(text: str) -> dict[str, Any]:
    return parse_progress_signals(text)


_RESOURCE_CONTEXT_VERSION_RE = re.compile(
    r"(?:SCIENCEFLOW_RESOURCE_CONTEXT_VERSION|resource_context_version)\s*[:=]\s*(\d+)",
    re.IGNORECASE,
)


_RESOURCE_PRESSURE_GENERATION_RE = re.compile(
    r"(?:SCIENCEFLOW_RESOURCE_PRESSURE_GENERATION|pressure_generation)\s*[:=]\s*(\d+)",
    re.IGNORECASE,
)


_ARTIFACT_WATCH_EXCLUDED_DIRS = frozenset({
    ".git",
    ".venv",
    ".logs",
    ".agent_memory",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "dataset",
    "datasets",
    "data",
    "input",
    "logs",
    "task_logs",
    "parent_workspace",
})


_ARTIFACT_WATCH_INCLUDED_HIDDEN_DIRS = frozenset({
    ".test_features",
    ".features",
    ".predictions",
    ".submissions",
    ".outputs",
    ".checkpoints",
    ".scienceflow_runs",
})


_ARTIFACT_WATCH_SUFFIXES = frozenset({
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".log",
    ".pkl",
    ".joblib",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
})


_ARTIFACT_WATCH_NAMES = frozenset({"submission", "submissions", "result", "results", "score", "scores"})


def _latest_artifact_snapshot(
    workspace_dir: Path,
    *,
    started_at: float,
    max_seen: int = 1000,
    run_artifact_dir: Path | None = None,
    run_state_path: Path | None = None,
    stability_cache: dict[str, Any] | None = None,
    settle_sec: float = 5.0,
    stable_poll_count: int = 1,
) -> dict[str, Any]:
    ws = Path(workspace_dir).resolve(strict=False)
    latest: dict[str, Any] = {}
    seen = 0
    now = time.time()
    run_dir = Path(run_artifact_dir).resolve(strict=False) if run_artifact_dir is not None else None
    state_path = Path(run_state_path).resolve(strict=False) if run_state_path is not None else None
    run_state = _read_run_state_json(state_path, started_at=started_at) if state_path is not None else {}
    run_state_rel = _safe_rel(state_path, ws) if state_path is not None else ""
    try:
        walker = os.walk(ws)
    except OSError:
        return latest
    for root, dirnames, filenames in walker:
        root_path = Path(root)
        parts = set(root_path.relative_to(ws).parts) if root_path != ws else set()
        if parts & _ARTIFACT_WATCH_EXCLUDED_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _ARTIFACT_WATCH_EXCLUDED_DIRS
            and (not name.startswith(".") or name in _ARTIFACT_WATCH_INCLUDED_HIDDEN_DIRS)
        ]
        for name in filenames:
            if name.startswith("."):
                continue
            path = root_path / name
            rel_path = _safe_rel(path, ws)
            if rel_path == run_state_rel or name.endswith(".done"):
                continue
            stem = Path(name).stem.lower()
            suffix = Path(name).suffix.lower()
            if suffix not in _ARTIFACT_WATCH_SUFFIXES and stem not in _ARTIFACT_WATCH_NAMES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size <= 0:
                continue
            current_run_scope = bool(run_dir is not None and _path_is_relative_to(path.resolve(strict=False), run_dir))
            workspace_recent = bool(stat.st_mtime + 1.0 >= started_at)
            if not current_run_scope and not workspace_recent:
                continue
            seen += 1
            if seen > max_seen:
                return latest
            if latest and stat.st_mtime <= float(latest.get("mtime") or 0.0):
                continue
            done_marker = _artifact_done_marker_exists(path, started_at=started_at)
            stable_info = _artifact_stability(
                rel_path,
                stat_size=int(stat.st_size),
                stat_mtime=float(stat.st_mtime),
                now=now,
                cache=stability_cache,
                settle_sec=settle_sec,
                stable_poll_count=stable_poll_count,
            )
            run_state_confirmed = _run_state_references_artifact(run_state, rel_path=rel_path, path=path, workspace_dir=ws)
            run_state_status = str(run_state.get("status") or "") if run_state else ""
            if done_marker:
                stability = "done_marker"
            elif run_state_confirmed and run_state_status in {"running", "interrupted", "completed"}:
                stability = "run_state_confirmed"
            elif stable_info["stable"]:
                stability = "stable"
            else:
                stability = "candidate"
            artifact_scope = "current_run" if current_run_scope else "workspace_recent"
            recoverable = bool(
                stability in {"done_marker", "run_state_confirmed"}
                and (current_run_scope or workspace_recent)
            )
            # A stable file under the run artifact dir is also recoverable even
            # without a marker. Stable workspace files outside the run dir remain
            # candidates unless a marker/run_state ties them to this job.
            if not recoverable and current_run_scope and stability == "stable":
                recoverable = True
            latest = {
                "path": rel_path,
                "mtime": stat.st_mtime,
                "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                "age_sec": max(0.0, now - stat.st_mtime),
                "size_bytes": int(stat.st_size),
                "artifact_scope": artifact_scope,
                "stability": stability,
                "stable_polls": int(stable_info["stable_polls"]),
                "settle_sec": max(0.0, float(settle_sec or 0.0)),
                "done_marker": bool(done_marker),
                "run_state_confirmed": bool(run_state_confirmed),
                "run_state_status": run_state_status,
                "safe_to_resume": run_state.get("safe_to_resume") if run_state else "unknown",
                "recoverable_artifact_on_disk": bool(recoverable),
                "candidate_artifact": not bool(recoverable),
            }
    return latest


def _safe_rel(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).name


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _artifact_done_marker_exists(path: Path, *, started_at: float) -> bool:
    marker = path.with_name(path.name + ".done")
    try:
        stat = marker.stat()
    except OSError:
        return False
    return bool(stat.st_mtime + 1.0 >= started_at)


def _read_run_state_json(path: Path | None, *, started_at: float) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        stat = path.stat()
    except OSError:
        return {}
    if stat.st_mtime + 1.0 < started_at or stat.st_size <= 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _run_state_references_artifact(
    run_state: dict[str, Any],
    *,
    rel_path: str,
    path: Path,
    workspace_dir: Path,
) -> bool:
    if not run_state:
        return False
    candidates: list[Any] = []
    for key in (
        "checkpoint_path",
        "submission_path",
        "artifact_path",
        "latest_checkpoint_path",
        "best_checkpoint_path",
        "partial_submission_path",
    ):
        value = run_state.get(key)
        if value:
            candidates.append(value)
    artifacts = run_state.get("artifacts")
    if isinstance(artifacts, list):
        candidates.extend(artifacts)
    elif isinstance(artifacts, dict):
        candidates.extend(artifacts.values())
    rel_norm = str(rel_path or "").strip().lstrip("./")
    abs_norm = str(path.resolve(strict=False))
    for raw in candidates:
        if isinstance(raw, dict):
            raw = raw.get("path") or raw.get("artifact_path") or raw.get("checkpoint_path")
        candidate = str(raw or "").strip()
        if not candidate:
            continue
        candidate_rel = candidate.lstrip("./")
        if candidate_rel == rel_norm:
            return True
        candidate_abs = Path(candidate)
        if not candidate_abs.is_absolute():
            candidate_abs = workspace_dir / candidate_abs
        if str(candidate_abs.resolve(strict=False)) == abs_norm:
            return True
    return False


def _artifact_stability(
    rel_path: str,
    *,
    stat_size: int,
    stat_mtime: float,
    now: float,
    cache: dict[str, Any] | None,
    settle_sec: float,
    stable_poll_count: int,
) -> dict[str, Any]:
    previous = cache.get(rel_path) if isinstance(cache, dict) else None
    same = bool(
        isinstance(previous, dict)
        and int(previous.get("size_bytes") or -1) == int(stat_size)
        and abs(float(previous.get("mtime") or 0.0) - float(stat_mtime)) < 1e-6
    )
    stable_polls = int(previous.get("stable_polls") or 0) + 1 if same and isinstance(previous, dict) else 0
    if isinstance(cache, dict):
        cache[rel_path] = {
            "size_bytes": int(stat_size),
            "mtime": float(stat_mtime),
            "seen_at": float(now),
            "stable_polls": stable_polls,
        }
    age_sec = max(0.0, float(now) - float(stat_mtime))
    stable = bool(stable_polls >= max(1, int(stable_poll_count or 1)) and age_sec >= max(0.0, float(settle_sec or 0.0)))
    return {"stable": stable, "stable_polls": stable_polls, "age_sec": age_sec}


def _parse_resource_context_version(command: str, env: dict[str, str] | None = None) -> int | None:
    for key in (
        "SCIENCEFLOW_RESOURCE_CONTEXT_VERSION",
        "_SCIENCEFLOW_RESOURCE_CONTEXT_VERSION",
    ):
        raw = (env or {}).get(key)
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None
    match = _RESOURCE_CONTEXT_VERSION_RE.search(str(command or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _parse_resource_pressure_generation(command: str, env: dict[str, str] | None = None) -> int | None:
    for key in (
        "SCIENCEFLOW_RESOURCE_PRESSURE_GENERATION",
        "_SCIENCEFLOW_RESOURCE_PRESSURE_GENERATION",
    ):
        raw = (env or {}).get(key)
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None
    match = _RESOURCE_PRESSURE_GENERATION_RE.search(str(command or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _parse_resource_value_hint(command: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    raw = ""
    value_hint_keys = (
        "SCIENCEFLOW_RESOURCE_VALUE_HINT",
        "_SCIENCEFLOW_RESOURCE_VALUE_HINT",
    )
    for key in value_hint_keys:
        value = (env or {}).get(key)
        if value:
            raw = str(value)
            break
    if not raw:
        try:
            tokens = shlex.split(str(command or ""), posix=True)
        except ValueError:
            tokens = str(command or "").split()
        for token in tokens[:12]:
            if any(token.startswith(f"{key}=") for key in value_hint_keys):
                raw = token.split("=", 1)[1]
                break
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


_LEADING_SLEEP_RE = re.compile(
    r"^\s*sleep\s+([^\s;&|]+)\s*(?:(?:&&|;)\s*(.+))?\s*$",
    re.DOTALL,
)


def _parse_sleep_token_seconds(token: str) -> float | None:
    raw = str(token or "").strip().lower()
    if not raw:
        return None
    multiplier = 1.0
    if raw[-1:] in {"s", "m", "h", "d"}:
        suffix = raw[-1]
        raw = raw[:-1]
        multiplier = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[suffix]
    try:
        seconds = float(raw) * multiplier
    except ValueError:
        return None
    if seconds <= 0 or seconds != seconds:
        return None
    return seconds


def _parse_leading_sleep_command(command: str) -> tuple[float, str] | None:
    raw = str(command or "")
    match = _LEADING_SLEEP_RE.match(raw)
    if not match:
        return None
    seconds = _parse_sleep_token_seconds(match.group(1))
    if seconds is None:
        return None
    remainder = str(match.group(2) or "").strip()
    return seconds, remainder


def _parse_simple_sleep_seconds(command: str) -> float | None:
    parsed = _parse_leading_sleep_command(command)
    if parsed is None:
        return None
    seconds, remainder = parsed
    if remainder:
        return None
    return seconds


def _is_gpu_visibility_probe(command: str) -> bool:
    cmd = str(command or "")
    lowered = cmd.lower()
    if re.match(r"^\s*(?:cuda_visible_devices=\S+\s+)?nvidia-smi(?:\s|$)", lowered):
        return True
    if "torch.cuda" not in lowered or not re.search(r"\bpython(?:3|\d+(?:\.\d+)*)?\b", lowered):
        return False
    if " -c " not in lowered and " - <<" not in lowered:
        return False
    probe_tokens = (
        "torch.cuda.is_available",
        "torch.cuda.device_count",
        "torch.cuda.get_device_name",
        "torch.version.cuda",
    )
    if not any(token in lowered for token in probe_tokens):
        return False
    heavy_tokens = (
        " fit(",
        ".fit(",
        " backward(",
        ".backward(",
        "optimizer.step",
        "dataloader",
        "--epochs",
        "train.py",
        "model.train(",
    )
    return not any(token in lowered for token in heavy_tokens)
