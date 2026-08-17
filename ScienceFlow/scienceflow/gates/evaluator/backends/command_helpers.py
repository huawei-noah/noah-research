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

"""Helpers for command-style evaluator backends."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping

from scienceflow.gates.evaluator.cache import digest_mapping, evaluator_cache_key
from scienceflow.gates.evaluator.models import CandidateRef, EvalContext
from scienceflow.gates.evaluator.backends.command_env import (
    task_command_env,
    task_python_executable,
)


def candidate_artifact(ctx: EvalContext) -> str:
    return clean_relative_path(str_cfg(ctx, ("candidate", "artifact"), "artifacts/best_solution.json"))


def cache_key(ctx: EvalContext, candidate: CandidateRef) -> str:
    command = str_cfg(ctx, ("command", "evaluator_command"), "")
    metric_cfg = {
        "name": str_cfg(ctx, ("metric", "name"), ""),
        "lower_is_better": bool_cfg(ctx, ("metric", "lower_is_better"), default=None),
        "regex": str_cfg(ctx, ("metric", "regex"), ""),
        "json_path": str_cfg(ctx, ("metric", "json_path"), ""),
    }
    return evaluator_cache_key(
        backend="artifact_command",
        artifact_sha=candidate.artifact_sha,
        command_digest=hashlib.sha256(command.encode("utf-8")).hexdigest(),
        metric_digest=digest_mapping(metric_cfg),
    )


def cache_path(ctx: EvalContext) -> Path:
    evaluator = get_cfg(ctx.cfg, "evaluator", None)
    name = str(get_cfg(evaluator, "cache_log", "evaluator_cache.jsonl") or "evaluator_cache.jsonl").strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        name = "evaluator_cache.jsonl"
    return ctx.task_root / "task_logs" / name


def run_command(ctx: EvalContext, candidate: CandidateRef, command: str) -> dict[str, Any]:
    timeout = float(number_cfg(ctx, ("command", "timeout_sec"), 300.0) or 300.0)
    cwd = command_cwd(ctx)
    env = task_command_env(ctx)
    extra_env = mapping_cfg(ctx, ("command", "env"))
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    rendered = render_command(ctx, candidate, command)
    try:
        proc = subprocess.run(
            rendered,
            cwd=str(cwd),
            env=env,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "timeout": True,
            "timeout_sec": timeout,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_tail": tail(stdout, tail_chars(ctx, "stdout_tail_chars")),
            "stderr_tail": tail(stderr, tail_chars(ctx, "stderr_tail_chars")),
        }
    return {
        "timeout": False,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_tail": tail(proc.stdout, tail_chars(ctx, "stdout_tail_chars")),
        "stderr_tail": tail(proc.stderr, tail_chars(ctx, "stderr_tail_chars")),
    }


def render_command(ctx: EvalContext, candidate: CandidateRef, command: str) -> str:
    artifact_abs = ctx.workspace / candidate.artifact_path
    python = task_python_executable(ctx)
    values = {
        "artifact_path": shlex.quote(candidate.artifact_path),
        "artifact_abs_path": shlex.quote(str(artifact_abs)),
        "workspace": shlex.quote(str(ctx.workspace)),
        "task_root": shlex.quote(str(ctx.task_root)),
        "stage_id": shlex.quote(str(ctx.stage_id or "")),
        "worker_id": shlex.quote(str(ctx.worker_id or "")),
        "python": shlex.quote(str(python)) if python else "python",
        "python_executable": shlex.quote(str(python)) if python else "python",
    }
    rendered = command
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def command_cwd(ctx: EvalContext) -> Path:
    raw = str_cfg(ctx, ("command", "cwd"), "workspace").strip().lower()
    if raw in {"", "workspace"}:
        return ctx.workspace
    if raw == "task_root":
        return ctx.task_root
    return ctx.workspace / clean_relative_path(raw)


def parse_metric(ctx: EvalContext, stdout: str) -> tuple[float | None, str]:
    json_path = str_cfg(ctx, ("metric", "json_path"), "")
    if json_path:
        value = metric_from_json_path(stdout, json_path)
        if value is not None:
            return value, f"metric parsed from json path {json_path}"
    regex = str_cfg(ctx, ("metric", "regex"), "")
    if regex:
        try:
            match = re.search(regex, stdout, re.MULTILINE)
        except re.error as exc:
            return None, f"invalid metric regex: {exc}"
        if match:
            group = match.group(1) if match.groups() else match.group(0)
            value = as_float(group)
            if value is not None:
                return value, "metric parsed from regex"
    return None, "evaluator output did not contain a parseable metric"


def metric_from_json_path(text: str, path: str) -> float | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    cur: Any = data
    for part in [p for p in path.split(".") if p]:
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
        else:
            return None
    return as_float(cur)


def path_sha256(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = child.relative_to(path).as_posix()
            h.update(rel.encode("utf-8"))
            with child.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        return h.hexdigest()
    return ""


def clean_relative_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip().lstrip("/")
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return "artifact"
    return "/".join(parts)


def mapping_cfg(ctx: EvalContext, path: tuple[str, str]) -> Mapping[str, Any]:
    value = nested_cfg(ctx, path, {})
    return value if isinstance(value, Mapping) else {}


def str_cfg(ctx: EvalContext, path: tuple[str, str], default: str) -> str:
    value = nested_cfg(ctx, path, default)
    return str(value if value is not None else default)


def number_cfg(ctx: EvalContext, path: tuple[str, str], default: float) -> float | None:
    return as_float(nested_cfg(ctx, path, default))


def bool_cfg(ctx: EvalContext, path: tuple[str, str], *, default: bool | None) -> bool | None:
    value = nested_cfg(ctx, path, default)
    return as_bool_value(value, default=default)


def nested_cfg(ctx: EvalContext, path: tuple[str, str], default: Any) -> Any:
    evaluator = get_cfg(ctx.cfg, "evaluator", None)
    block = get_cfg(evaluator, path[0], None)
    value = get_cfg(block, path[1], None)
    if value is not None:
        return value
    legacy_key = path[1] if path[0] != "command" else "evaluator_command"
    return get_cfg(ctx.cfg, legacy_key, default)


def get_cfg(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def as_bool_value(value: Any, *, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def tail(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return text[-max_chars:]


def tail_chars(ctx: EvalContext, key: str) -> int:
    return int(number_cfg(ctx, ("command", key), 4000) or 4000)
