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
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_kttsp() -> Any:
    path = _REPO_ROOT / "tasks" / "opt_solver" / "kttsp" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("_test_kttsp_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_KTTSP = _load_kttsp()
KTTSPInstance = _KTTSP.KTTSPInstance
evaluate_vector = _KTTSP.evaluate_vector
load_instance = _KTTSP.load_instance
load_solution_vector = _KTTSP.load_solution_vector


def _instance(n: int = 2, *, n_exceptions: int = 1) -> KTTSPInstance:
    rows = tuple((1.0 + i * 0.1, 0.0, 0.0, 0.0, 0.0, 0.0) for i in range(n))
    return KTTSPInstance(
        t0=0.0,
        min_tof=0.001,
        max_time=10.0,
        dv_threshold=100.0,
        dv_exception=600.0,
        n_exceptions=n_exceptions,
        orbital_rows=rows,
    )


def test_load_instance_parses_text_kttsp(tmp_path: Path) -> None:
    problem = tmp_path / "easy.kttsp"
    problem.write_text(
        "\n".join(
            [
                "c comment",
                "p kttsp 0.0 0.001 200.0 100.0 600.0 5",
                "1.0 0.0 0.0 0.0 0.0 0.0",
                "1.1 0.0 0.0 0.0 0.0 0.0",
            ]
        ),
        encoding="utf-8",
    )

    instance = load_instance(problem, max_revs=20)

    assert instance.n_tomatoes == 2
    assert instance.max_time == 200.0
    assert instance.n_exceptions == 5


def test_load_solution_vector_supports_official_submission(tmp_path: Path) -> None:
    solution = tmp_path / "submission.json"
    solution.write_text(
        json.dumps([{"decisionVector": [0.0, 1.0, 0, 1], "problem": "small"}]),
        encoding="utf-8",
    )

    assert load_solution_vector(solution) == [0.0, 1.0, 0.0, 1.0]


def test_evaluate_vector_accepts_platform_feasible_solution() -> None:
    result = evaluate_vector(_instance(), [0.0, 1.0, 0, 1], transfer_fn=lambda *_: 80.0)

    assert result["valid"] is True
    assert result["n_tomatoes"] == 2
    assert result["metric"]["value"] == 1.0
    assert result["exceptions_used"] == 0


def test_evaluate_vector_rejects_too_many_delta_v_exceptions() -> None:
    instance = _instance(n=3, n_exceptions=1)
    vector = [0.0, 1.0, 1.0, 1.0, 0, 1, 2]

    with pytest.raises(ValueError, match="exception count exceeds limit"):
        evaluate_vector(instance, vector, transfer_fn=lambda *_: 200.0)


def test_evaluate_vector_rejects_platform_bounds() -> None:
    with pytest.raises(ValueError, match="tof 0 value 0.0 is below lower bound"):
        evaluate_vector(_instance(), [0.0, 0.0, 0, 1], transfer_fn=lambda *_: 0.0)


def test_evaluate_vector_rejects_time_chain_violation() -> None:
    instance = _instance(n=3)
    vector = [0.0, 0.5, 1.0, 1.0, 0, 1, 2]

    with pytest.raises(ValueError, match="arrives at"):
        evaluate_vector(instance, vector, transfer_fn=lambda *_: 0.0)
