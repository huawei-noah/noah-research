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

import importlib.util
import sys
from pathlib import Path


def _load_query_budget_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tasks"
        / "sci_modeling_bench"
        / "_shared"
        / "query_budget.py"
    )
    spec = importlib.util.spec_from_file_location("_test_sci_query_budget", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_worker_query_budgets_and_duplicate_caches_are_independent(tmp_path: Path) -> None:
    module = _load_query_budget_module()
    task_dir = tmp_path / "task_runtime" / "sci_modeling_bench" / "demo"
    task_dir.mkdir(parents=True)
    calls: list[tuple[str, int]] = []

    def run(worker: str, candidate: int):
        return module.evaluate_with_query_budget(
            task_dir=task_dir,
            config={
                "evaluation_context": {
                    "query_budget_scope": "worker",
                    "worker_id": worker,
                }
            },
            task_id="demo-task",
            query={"candidates": [candidate]},
            query_limit=2,
            evaluate=lambda: calls.append((worker, candidate)) or {"score": candidate},
        )

    first = run("W00", 1)
    duplicate = run("W00", 1)
    second = run("W00", 2)
    exhausted = run("W00", 3)
    peer_first = run("W01", 1)

    assert first.queries_used == 1 and first.queries_remaining == 1
    assert duplicate.cache_hit is True and duplicate.queries_used == 1
    assert second.queries_used == 2 and second.queries_remaining == 0
    assert exhausted.exhausted is True and exhausted.payload is None
    assert peer_first.queries_used == 1 and peer_first.cache_hit is False
    assert calls == [("W00", 1), ("W00", 2), ("W01", 1)]


def test_default_task_query_budget_is_shared_across_workers(tmp_path: Path) -> None:
    module = _load_query_budget_module()
    task_dir = tmp_path / "task_runtime" / "sci_modeling_bench" / "demo"
    task_dir.mkdir(parents=True)

    first = module.evaluate_with_query_budget(
        task_dir=task_dir,
        config={"evaluation_context": {"worker_id": "W00"}},
        task_id="demo-task",
        query=["a"],
        query_limit=1,
        evaluate=lambda: {"score": 1},
    )
    second = module.evaluate_with_query_budget(
        task_dir=task_dir,
        config={"evaluation_context": {"worker_id": "W01"}},
        task_id="demo-task",
        query=["b"],
        query_limit=1,
        evaluate=lambda: {"score": 2},
    )

    assert first.exhausted is False
    assert second.exhausted is True
