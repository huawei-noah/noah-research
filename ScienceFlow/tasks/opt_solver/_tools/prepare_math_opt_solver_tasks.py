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

"""Prepare portable math opt-solver task packages.

The three math tasks do not need a large dataset, but ScienceFlow still expects a
small task package under an opt_solver data root. This script writes those
packages with problem metadata, a valid baseline solution, and source notes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SOURCE_REFS = """# Source References

These task definitions are adapted for ScienceFlow opt_solver from public mathematical optimization examples:

- OpenEvolve repository: https://github.com/algorithmicsuperintelligence/openevolve
- OpenEvolve alphaevolve_math_problems: https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/alphaevolve_math_problems
- DeepMind AlphaEvolve results: https://github.com/google-deepmind/alphaevolve_results

OpenEvolve and AlphaEvolve result software are published under Apache-2.0. The ScienceFlow evaluator wrappers use the public problem definitions and implement the artifact JSON contract locally.
"""


def _circle_baseline() -> list[list[float]]:
    circles: list[list[float]] = []
    for row in range(5):
        for col in range(6):
            if len(circles) >= 26:
                return circles
            circles.append([(col + 0.5) / 6.0, (row + 0.5) / 5.0, 0.07])
    return circles


def _ratio_baseline() -> list[list[float]]:
    return [[math.cos(2 * math.pi * i / 16), math.sin(2 * math.pi * i / 16)] for i in range(16)]


def _write_json(path: Path, data: Any, *, force: bool) -> None:
    _write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n", force=force)


def _write_text(path: Path, text: str, *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare(target: Path, *, force: bool = False) -> None:
    target = target.expanduser().resolve()
    circle_problem = {
        "artifact_schema": {"circles": "list of [x, y, radius] with length 26"},
        "benchmark_radii_sum": 2.635,
        "container": "unit_square",
        "lower_is_better": False,
        "metric_name": "radii_sum",
        "num_circles": 26,
        "objective": "maximize_sum_of_radii",
        "task": "circle_packing",
        "tolerance": 1e-6,
        "variant": "unit_square_26",
    }
    ratio_problem = {
        "also_supported": {"benchmark_inv_ratio_squared": 1.0 / 4.165849767, "dimension": 3, "num_points": 14},
        "artifact_schema": {"points": "list of 16 points, each [x, y]"},
        "benchmark_inv_ratio_squared": 1.0 / 12.889266112,
        "dimension": 2,
        "lower_is_better": False,
        "metric_name": "inv_ratio_squared",
        "num_points": 16,
        "objective": "maximize_inv_ratio_squared_equivalent_to_minimize_max_min_ratio",
        "task": "ratio_minimization",
        "variant": "minimizing_max_min_dist_dim2_16",
    }
    uncertainty_problem = {
        "artifact_schema": {
            "c4_bound": "optional reported value",
            "coeffs": "non-empty numeric coefficient list",
            "r_max": "optional reported value",
        },
        "benchmark_c4": 0.3215872333529007,
        "check_atol": 1e-9,
        "check_rtol": 1e-9,
        "lower_is_better": False,
        "max_coeff_count": 8,
        "metric_name": "c4_score",
        "objective": "maximize_benchmark_c4_over_computed_c4_bound",
        "task": "uncertainty_ineq",
        "variant": "alphaevolve_appendix_b4",
    }

    packages = [
        (
            "circle_packing",
            "# Circle Packing\n\nPlace 26 disjoint circles inside the unit square and maximize the sum of radii. The default artifact is `artifacts/best_solution.json` with `{\"circles\": [[x, y, r], ...]}`.\n\nThis is a small mathematical optimization task. There is no train/test dataset; `problem.json` is the task instance and `initial_solution.json` is only a valid baseline.\n",
            circle_problem,
            {"circles": _circle_baseline(), "note": "simple valid grid baseline"},
        ),
        (
            "ratio_minimization",
            "# Ratio Minimization\n\nPlace 16 points in 2D and maximize `(min_pairwise_distance / max_pairwise_distance)^2`, equivalent to minimizing the maximum/minimum distance ratio. The default artifact is `artifacts/best_solution.json` with `{\"points\": [[x, y], ...]}`.\n\nThere is no large dataset; `problem.json` defines the point count and dimension.\n",
            ratio_problem,
            {"points": _ratio_baseline(), "note": "regular polygon baseline"},
        ),
        (
            "uncertainty_ineq",
            "# An Uncertainty Inequality\n\nConstruct coefficients for an even Hermite-polynomial combination used to bound a Fourier-analysis constant. The default artifact is `artifacts/best_solution.json` with `{\"coeffs\": [...]}` and optional `c4_bound` / `r_max`; the evaluator recomputes both values.\n\nThere is no data table. The task instance is the symbolic/numeric verification rule in `problem.json`.\n",
            uncertainty_problem,
            {"coeffs": [1.0], "note": "simple valid one-coefficient baseline"},
        ),
    ]

    for name, readme, problem, initial in packages:
        directory = target / name
        _write_text(directory / "README.md", readme, force=force)
        _write_json(directory / "problem.json", problem, force=force)
        _write_json(directory / "initial_solution.json", initial, force=force)
        _write_text(directory / "source_refs.md", SOURCE_REFS, force=force)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="./data/opt_solver",
        help="Opt-solver data root to populate. Defaults to the local development path.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    target = Path(args.target)
    prepare(target, force=args.force)
    print(f"math opt-solver task packages are ready under {target.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
