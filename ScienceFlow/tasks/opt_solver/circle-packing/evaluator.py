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

"""Evaluate circle-packing JSON artifacts for opt_solver tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TOL = 1e-6


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


def _load_circles(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"solution file not found: {path}")
    data = _load_json(path)
    if isinstance(data, dict):
        raw = data.get("circles")
        if raw is None and "centers" in data and "radii" in data:
            centers = np.asarray(data["centers"], dtype=float)
            radii = np.asarray(data["radii"], dtype=float)
            if centers.ndim != 2 or centers.shape[1] != 2:
                raise ValueError("centers must have shape (n, 2)")
            if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
                raise ValueError("radii must have shape (n,) and match centers")
            return np.column_stack([centers, radii])
    else:
        raw = data
    if raw is None:
        raise ValueError("solution JSON must be a list of circles or contain key 'circles'")
    circles = np.asarray(raw, dtype=float)
    if circles.ndim != 2 or circles.shape[1] != 3:
        raise ValueError(f"circles must have shape (n, 3), got {circles.shape}")
    return circles


def _validate_shape(circles: np.ndarray, expected_n: int) -> None:
    if circles.shape != (expected_n, 3):
        raise ValueError(f"circles must have shape ({expected_n}, 3), got {circles.shape}")
    if not np.isfinite(circles).all():
        raise ValueError("circle coordinates and radii must be finite")
    if np.any(circles[:, 2] < 0):
        idx = int(np.where(circles[:, 2] < 0)[0][0])
        raise ValueError(f"circle {idx} has negative radius {circles[idx, 2]}")


def _validate_overlaps(circles: np.ndarray, tol: float) -> None:
    centers = circles[:, :2]
    radii = circles[:, 2]
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            required = float(radii[i] + radii[j])
            if dist < required - tol:
                raise ValueError(
                    f"circles {i} and {j} overlap: dist={dist:.12g}, radii_sum={required:.12g}"
                )


def _validate_unit_square(circles: np.ndarray, tol: float) -> None:
    for idx, (x, y, r) in enumerate(circles):
        if x - r < -tol or x + r > 1.0 + tol or y - r < -tol or y + r > 1.0 + tol:
            raise ValueError(f"circle {idx} is outside the unit square")


def _validate_rectangle_perimeter4(circles: np.ndarray, tol: float) -> tuple[float, float]:
    min_x = float(np.min(circles[:, 0] - circles[:, 2]))
    max_x = float(np.max(circles[:, 0] + circles[:, 2]))
    min_y = float(np.min(circles[:, 1] - circles[:, 2]))
    max_y = float(np.max(circles[:, 1] + circles[:, 2]))
    width = max_x - min_x
    height = max_y - min_y
    if width + height > 2.0 + tol:
        raise ValueError(
            f"minimum enclosing rectangle has perimeter {2 * (width + height):.12g}, exceeding 4"
        )
    return width, height


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
    expected_n = int(problem.get("num_circles", 26))
    container = str(problem.get("container", "unit_square"))
    tol = float(problem.get("tolerance", DEFAULT_TOL))
    benchmark = problem.get("benchmark_radii_sum")

    circles = _load_circles(solution_path)
    _validate_shape(circles, expected_n)
    _validate_overlaps(circles, tol)

    metadata: dict[str, Any] = {"container": container, "num_circles": expected_n}
    if container == "unit_square":
        _validate_unit_square(circles, tol)
    elif container == "rectangle_perimeter4":
        width, height = _validate_rectangle_perimeter4(circles, tol)
        metadata.update({"rectangle_width": width, "rectangle_height": height})
    else:
        raise ValueError(f"unsupported circle packing container: {container}")

    radii_sum = float(np.sum(circles[:, 2]))
    metadata["radii_sum"] = radii_sum
    if benchmark is not None:
        metadata["benchmark_ratio"] = radii_sum / float(benchmark)
    return {
        "metric": {"name": "radii_sum", "value": radii_sum},
        "valid": True,
        **metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--solution", required=True)
    args = parser.parse_args()
    try:
        result = _evaluate_problem_solution(Path(args.problem), Path(args.solution))
    except Exception as exc:
        print(f"circle_packing_eval error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
