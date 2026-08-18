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

"""Trusted, task-local query ledger for SciModelingBench evaluators."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


@dataclass(frozen=True)
class QueryBudgetResult:
    payload: dict[str, Any] | None
    queries_used: int
    queries_remaining: int
    query_limit: int
    cache_hit: bool
    exhausted: bool


def evaluate_with_query_budget(
    *,
    task_dir: Path,
    config: Mapping[str, Any],
    task_id: str,
    query: Any,
    query_limit: int,
    evaluate: Callable[[], Mapping[str, Any]],
) -> QueryBudgetResult:
    """Evaluate one canonical query under a persistent trusted budget."""

    limit = int(query_limit)
    if limit <= 0:
        raise ValueError("query_limit must be positive")
    scope, owner = query_budget_identity(config)
    state_path = query_state_path(
        task_dir,
        task_id=task_id,
        scope=scope,
        owner=owner,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    query_sha = _stable_json_sha(query)
    identity = {
        "schema_version": 1,
        "task_id": str(task_id),
        "query_budget_scope": scope,
        "query_owner": owner,
        "query_limit": limit,
    }

    with _exclusive_file_lock(state_path.with_suffix(".lock")):
        state = _read_state(state_path)
        if state:
            existing_identity = state.get("identity")
            if existing_identity != identity:
                raise ValueError(
                    "query state belongs to a different task or budget identity"
                )
        results = state.get("results")
        if not isinstance(results, dict):
            results = {}
        cached = results.get(query_sha)
        if isinstance(cached, dict):
            used = len(results)
            return QueryBudgetResult(
                payload=dict(cached),
                queries_used=used,
                queries_remaining=max(0, limit - used),
                query_limit=limit,
                cache_hit=True,
                exhausted=False,
            )
        used = len(results)
        if used >= limit:
            return QueryBudgetResult(
                payload=None,
                queries_used=used,
                queries_remaining=0,
                query_limit=limit,
                cache_hit=False,
                exhausted=True,
            )
        payload = dict(evaluate())
        results[query_sha] = payload
        used = len(results)
        _write_state(
            state_path,
            {
                "identity": identity,
                "results": results,
            },
        )
        return QueryBudgetResult(
            payload=payload,
            queries_used=used,
            queries_remaining=max(0, limit - used),
            query_limit=limit,
            cache_hit=False,
            exhausted=False,
        )


def query_budget_identity(config: Mapping[str, Any]) -> tuple[str, str]:
    context = config.get("evaluation_context")
    context = context if isinstance(context, Mapping) else {}
    scope = str(context.get("query_budget_scope") or "task").strip().lower()
    if scope not in {"task", "worker"}:
        raise ValueError("query_budget_scope must be either 'task' or 'worker'")
    if scope == "task":
        return scope, "task"
    worker_id = str(context.get("worker_id") or "").strip()
    if not worker_id:
        raise ValueError("worker query budget scope requires a trusted worker_id")
    return scope, _safe_name(worker_id)


def query_state_path(
    task_dir: Path,
    *,
    task_id: str,
    scope: str,
    owner: str,
) -> Path:
    resolved = Path(task_dir).resolve()
    filename = f"queries.{_safe_name(owner)}.json" if scope == "worker" else "queries.json"
    for parent in (resolved, *resolved.parents):
        if parent.name == "task_runtime":
            return parent.parent / "evaluator_state" / _safe_name(task_id) / filename
    raise ValueError(f"task runtime root not found above evaluator task directory: {task_dir}")


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("query state must be a JSON object")
    return data


def _write_state(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stable_json_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip()).strip("-._")
    if not cleaned:
        raise ValueError("query budget identity has no filesystem-safe characters")
    return cleaned
