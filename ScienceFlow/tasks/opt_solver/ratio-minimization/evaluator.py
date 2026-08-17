#!/usr/bin/env python3
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

"""Evaluate maximum/minimum distance ratio JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON: {path}: {exc}") from exc


def _load_problem(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"problem file not found: {path}")
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError("problem JSON must be an object")
    return data


def _load_points(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"solution file not found: {path}")
    data = _load_json(path)
    raw = data.get("points") if isinstance(data, dict) else data
    if raw is None:
        raise ValueError("solution JSON must be a list of points or contain key 'points'")
    points = np.asarray(raw, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"points must have shape (n, d), got {points.shape}")
    return points


def evaluate(
    *,
    artifact_path: Path,
    workspace_dir: Path,
    task_dir: Path,
    dataset_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    _ = workspace_dir, task_dir, config
    return _evaluate_problem_solution(dataset_dir / "problem.json", artifact_path)


def _evaluate_problem_solution(problem_path: Path, solution_path: Path) -> dict[str, Any]:
    problem = _load_problem(problem_path)
    num_points = int(problem.get("num_points", 16))
    dimension = int(problem.get("dimension", 2))
    benchmark = problem.get("benchmark_inv_ratio_squared")

    points = _load_points(solution_path)
    if points.shape != (num_points, dimension):
        raise ValueError(f"points must have shape ({num_points}, {dimension}), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("point coordinates must be finite")

    distances = pdist(points)
    if distances.size == 0:
        raise ValueError("at least two points are required")
    min_distance = float(np.min(distances))
    max_distance = float(np.max(distances))
    if max_distance <= 0.0:
        raise ValueError("maximum pairwise distance must be positive")
    inv_ratio_squared = float((min_distance / max_distance) ** 2)

    payload: dict[str, Any] = {
        "metric": {"name": "inv_ratio_squared", "value": inv_ratio_squared},
        "valid": True,
        "num_points": num_points,
        "dimension": dimension,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "max_min_ratio": float(max_distance / min_distance) if min_distance > 0 else float("inf"),
    }
    if benchmark is not None:
        payload["benchmark_ratio"] = inv_ratio_squared / float(benchmark)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--solution", required=True)
    args = parser.parse_args()
    try:
        result = _evaluate_problem_solution(Path(args.problem), Path(args.solution))
    except Exception as exc:
        print(f"ratio_minimization_eval error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
