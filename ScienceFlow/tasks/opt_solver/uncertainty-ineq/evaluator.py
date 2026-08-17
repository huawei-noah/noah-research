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

"""Evaluate uncertainty-inequality coefficient artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

X = sp.symbols("x")


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


def _load_solution(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"solution file not found: {path}")
    data = _load_json(path)
    if isinstance(data, list):
        return {"coeffs": data}
    if not isinstance(data, dict):
        raise ValueError("solution JSON must be an object or a coefficient list")
    return data


def _coefficients(data: dict[str, Any], *, max_count: int) -> np.ndarray:
    raw = data.get("coeffs", data.get("coefficients"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("solution must contain a non-empty 'coeffs' list")
    if len(raw) > max_count:
        raise ValueError(f"too many coefficients: {len(raw)} > {max_count}")
    coeffs = np.asarray(raw, dtype=float)
    if coeffs.ndim != 1 or not np.isfinite(coeffs).all():
        raise ValueError("coefficients must be a finite one-dimensional numeric list")
    if np.allclose(coeffs, 0.0):
        raise ValueError("all coefficients are zero")
    return coeffs


def _rational(value: float) -> sp.Rational:
    return sp.Rational(str(float(value)))


def _hermite_4k_polys(count: int) -> list[sp.Expr]:
    return [sp.polys.orthopolys.hermite_poly(n=4 * k, x=X, polys=False) for k in range(count)]


def _construct_polynomial(coeffs: np.ndarray) -> sp.Expr:
    # Build c0*H0 + ... + c_{m-1}*H_{4(m-1)} + c_last*H_{4m},
    # choosing c_last so P(0)=0. Degrees are multiples of 4, so P is even.
    m = len(coeffs)
    hermites = _hermite_4k_polys(m + 1)
    partial = sum(_rational(float(coeffs[i])) * hermites[i] for i in range(m))
    divisor = hermites[m].subs(X, 0)
    if divisor == 0:
        raise ValueError("cannot force P(0)=0 because H_{4m}(0)=0")
    c_last = -partial.subs(X, 0) / divisor
    polynomial = sp.expand(partial + c_last * hermites[m])
    leading = sp.LC(sp.Poly(polynomial, X), X)
    if leading < 0:
        polynomial = -polynomial
    if polynomial.subs(X, 0) != 0:
        raise ValueError("internal error: constructed polynomial does not satisfy P(0)=0")
    return sp.expand(polynomial)


def _largest_positive_root_after_dividing_x2(polynomial: sp.Expr) -> float:
    quotient, remainder = sp.div(sp.Poly(polynomial, X), sp.Poly(X**2, X))
    if remainder.as_expr() != 0:
        raise ValueError("constructed polynomial is not divisible by x^2")
    roots = sp.real_roots(quotient.as_expr(), X)
    positives: list[float] = []
    for root in roots:
        value = float(root.evalf(60))
        if math.isfinite(value) and value > 0.0:
            positives.append(value)
    if not positives:
        raise ValueError("P(x)/x^2 has no positive real root")
    return max(positives)


def compute_c4_and_rmax(coeffs: np.ndarray) -> tuple[float, float]:
    polynomial = _construct_polynomial(coeffs)
    r_max = _largest_positive_root_after_dividing_x2(polynomial)
    c4_bound = float((r_max**2) / (2.0 * math.pi))
    if not math.isfinite(c4_bound) or c4_bound <= 0.0:
        raise ValueError("computed c4_bound is not positive and finite")
    return c4_bound, r_max


def _check_reported(data: dict[str, Any], c4_bound: float, r_max: float, atol: float, rtol: float) -> None:
    if "c4_bound" in data and not math.isclose(float(data["c4_bound"]), c4_bound, rel_tol=rtol, abs_tol=atol):
        raise ValueError(f"reported c4_bound {float(data['c4_bound']):.12g} does not match recomputed {c4_bound:.12g}")
    if "r_max" in data and not math.isclose(float(data["r_max"]), r_max, rel_tol=rtol, abs_tol=atol):
        raise ValueError(f"reported r_max {float(data['r_max']):.12g} does not match recomputed {r_max:.12g}")


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
    data = _load_solution(solution_path)
    max_coeff_count = int(problem.get("max_coeff_count", 8))
    benchmark_c4 = float(problem.get("benchmark_c4", 0.3215872333529007))
    atol = float(problem.get("check_atol", 1e-9))
    rtol = float(problem.get("check_rtol", 1e-9))

    coeffs = _coefficients(data, max_count=max_coeff_count)
    c4_bound, r_max = compute_c4_and_rmax(coeffs)
    _check_reported(data, c4_bound, r_max, atol, rtol)
    score = float(benchmark_c4 / c4_bound)
    return {
        "metric": {"name": "c4_score", "value": score},
        "valid": True,
        "c4_bound": c4_bound,
        "r_max": r_max,
        "benchmark_c4": benchmark_c4,
        "coeff_count": int(len(coeffs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--solution", required=True)
    args = parser.parse_args()
    try:
        result = _evaluate_problem_solution(Path(args.problem), Path(args.solution))
    except Exception as exc:
        print(f"uncertainty_ineq_eval error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
