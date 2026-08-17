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

"""Platform-compatible evaluator for Keplerian Tomato TSP artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

MU_MOON = 4.904869500000000e12
EPS = 1e-6

TransferFn = Callable[[int, int, float, float], float]


@dataclass(frozen=True)
class KTTSPInstance:
    """Parsed KTTSP instance parameters used by the Optimize UDP."""

    t0: float
    min_tof: float
    max_time: float
    dv_threshold: float
    dv_exception: float
    n_exceptions: int
    orbital_rows: tuple[tuple[float, float, float, float, float, float], ...]
    max_revs: int = 20

    @property
    def n_tomatoes(self) -> int:
        return len(self.orbital_rows)


class KTTSPRuntime:
    """Runtime Lambert adapter matching the public Optimize UDP."""

    def __init__(self, instance: KTTSPInstance):
        self.instance = instance
        self._pk = _load_pykep()
        self._tomatoes = []
        epoch = self._pk.epoch(instance.t0)
        for row in instance.orbital_rows:
            self._tomatoes.append(self._pk.planet.keplerian(epoch, list(row), MU_MOON, 0.0, 0.0, 0.0))

    def compute_transfer(self, i_from: int, i_to: int, t_start: float, tof: float) -> float:
        """Compute the minimum two-impulse Lambert delta-v for one transfer."""

        tof_sec = float(tof) * self._pk.DAY2SEC
        r_i, v_i = self._tomatoes[i_from].eph(float(t_start))
        r_j, v_j = self._tomatoes[i_to].eph(float(t_start) + float(tof))

        best_dv = float("inf")
        for cw in (False, True):
            try:
                lambert = self._pk.lambert_problem(r_i, r_j, tof_sec, MU_MOON, cw, self.instance.max_revs)
            except Exception:
                continue
            for v1, v2 in zip(lambert.get_v1(), lambert.get_v2()):
                delta_v_i = np.linalg.norm(np.asarray(v1) - np.asarray(v_i))
                delta_v_j = np.linalg.norm(np.asarray(v2) - np.asarray(v_j))
                best_dv = min(best_dv, float(delta_v_i + delta_v_j))
        return best_dv


def evaluate(
    *,
    artifact_path: Path,
    workspace_dir: Path,
    task_dir: Path,
    dataset_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    _ = workspace_dir, task_dir
    task_cfg = config.get("task") if isinstance(config.get("task"), dict) else {}
    evaluator_cfg = task_cfg.get("evaluator") if isinstance(task_cfg.get("evaluator"), dict) else {}
    problem_name = str(evaluator_cfg.get("problem") or "problem.kttsp")
    max_revs = int(evaluator_cfg.get("max_revs") or 20)
    return _evaluate_problem_solution(dataset_dir / problem_name, artifact_path, max_revs=max_revs)


def load_instance(path: Path, *, max_revs: int = 20) -> KTTSPInstance:
    """Load a text ``.kttsp`` instance file."""

    if not path.is_file():
        raise FileNotFoundError(f"KTTSP instance file not found: {path}")
    header: list[str] | None = None
    rows: list[tuple[float, float, float, float, float, float]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        parts = line.split()
        if parts[:2] == ["p", "kttsp"]:
            header = parts
            continue
        if len(parts) != 6:
            raise ValueError(f"{path}:{line_no}: expected 6 orbital parameters, got {len(parts)}")
        try:
            rows.append(_six_floats(parts))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid orbital row: {line}") from exc
    if header is None:
        raise ValueError(f"missing kttsp header line in {path}")
    if len(header) != 8:
        raise ValueError(f"kttsp header must have 8 fields like the Optimize UDP, got: {header}")
    if not rows:
        raise ValueError(f"kttsp file has no tomato orbital rows: {path}")
    return KTTSPInstance(
        t0=float(header[2]),
        min_tof=float(header[3]),
        max_time=float(header[4]),
        dv_threshold=float(header[5]),
        dv_exception=float(header[6]),
        n_exceptions=int(float(header[7])),
        orbital_rows=tuple(rows),
        max_revs=int(max_revs),
    )


def load_solution_vector(path: Path) -> list[float]:
    """Load local ``{"x": ...}`` or official Optimize ``decisionVector`` JSON."""

    data = json.loads(path.read_text(encoding="utf-8"))
    raw: Any
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and "decisionVector" in data[0]:
            raw = data[0].get("decisionVector")
        else:
            raw = data
    elif isinstance(data, dict):
        raw = data.get("decisionVector")
        if raw is None:
            raw = data.get("x") or data.get("chromosome") or data.get("solution")
    else:
        raw = None
    if not isinstance(raw, list):
        raise ValueError("solution JSON must be a vector list or contain key 'decisionVector'/'x'")
    try:
        vector = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("all chromosome entries must be numeric") from exc
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("chromosome entries must be finite")
    return vector


def evaluate_solution(problem_path: Path, solution_path: Path, *, max_revs: int = 20) -> dict[str, Any]:
    """Evaluate a solution file using the platform-compatible KTTSP contract."""

    instance = load_instance(problem_path, max_revs=max_revs)
    runtime = KTTSPRuntime(instance)
    vector = load_solution_vector(solution_path)
    return evaluate_vector(instance, vector, transfer_fn=runtime.compute_transfer)


def _evaluate_problem_solution(problem_path: Path, solution_path: Path, *, max_revs: int = 20) -> dict[str, Any]:
    return evaluate_solution(problem_path, solution_path, max_revs=max_revs)


def evaluate_vector(
    instance: KTTSPInstance,
    vector: list[float],
    *,
    transfer_fn: TransferFn,
) -> dict[str, Any]:
    """Validate and score a KTTSP chromosome.

    The checks mirror the public Optimize UDP: variable bounds are enforced,
    permutation/delta-v/time-order are hard gates, and high-delta-v exception
    usage must be at most ``n_exceptions``.
    """

    n = instance.n_tomatoes
    expected = 3 * n - 2
    if len(vector) != expected:
        raise ValueError(f"chromosome length must be {expected} for {n} tomatoes, got {len(vector)}")
    legs = n - 1
    departures = vector[:legs]
    tofs = vector[legs : 2 * legs]
    route = _as_route(vector[2 * legs :], n)

    _validate_time_bounds(departures, 0.0, instance.max_time, "departure")
    _validate_time_bounds(tofs, instance.min_tof, instance.max_time, "tof")
    _validate_time_chain(departures, tofs)

    high_delta_v: list[tuple[int, int, int, float]] = []
    for idx in range(legs):
        dv = float(transfer_fn(route[idx], route[idx + 1], float(departures[idx]), float(tofs[idx])))
        if not math.isfinite(dv):
            raise ValueError(f"leg {idx} has no finite Lambert transfer")
        if dv > instance.dv_exception + EPS:
            raise ValueError(
                f"leg {idx} delta-v {dv:.6g} exceeds exception threshold {instance.dv_exception:.6g}"
            )
        if dv > instance.dv_threshold:
            high_delta_v.append((idx, route[idx], route[idx + 1], dv))
            if len(high_delta_v) > instance.n_exceptions:
                last = high_delta_v[-1]
                raise ValueError(
                    "delta-v exception count exceeds limit "
                    f"{instance.n_exceptions}; leg {last[0]} {last[1]}->{last[2]} has {last[3]:.6g}"
                )

    objective = float(departures[-1]) + float(tofs[-1])
    return {
        "metric": {"name": "mission_duration_days", "value": objective},
        "valid": True,
        "n_tomatoes": n,
        "exceptions_used": len(high_delta_v),
        "constraint_values": {
            "permutation": 0,
            "dv": 0,
            "time": 0,
            "dv_exception": len(high_delta_v) - instance.n_exceptions,
        },
    }


def _six_floats(values: list[str]) -> tuple[float, float, float, float, float, float]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 6:
        raise ValueError("expected six values")
    return parsed


def _as_route(values: list[float], n: int) -> list[int]:
    route: list[int] = []
    for offset, value in enumerate(values):
        idx = int(value)
        if float(idx) != float(value):
            raise ValueError(f"route entry {offset} must be an integer tomato id, got {value!r}")
        if idx < 0 or idx >= n:
            raise ValueError(f"route entry {offset} must be in 0..{n - 1}, got {idx}")
        route.append(idx)
    if sorted(route) != list(range(n)):
        raise ValueError(f"route must be a permutation of 0..{n - 1}")
    return route


def _validate_time_bounds(values: list[float], lower: float, upper: float, label: str) -> None:
    for idx, value in enumerate(values):
        if value < lower - EPS:
            raise ValueError(f"{label} {idx} value {value} is below lower bound {lower}")
        if value > upper + EPS:
            raise ValueError(f"{label} {idx} value {value} exceeds upper bound {upper}")


def _validate_time_chain(departures: list[float], tofs: list[float]) -> None:
    for idx in range(len(departures) - 1):
        arrival = float(departures[idx]) + float(tofs[idx])
        if arrival > float(departures[idx + 1]) + EPS:
            raise ValueError(
                f"leg {idx} arrives at {arrival:.12g} after next departure {departures[idx + 1]:.12g}"
            )


def _load_pykep() -> Any:
    try:
        import pykep as pk
    except ImportError as exc:
        raise RuntimeError("KTTSP evaluator requires pykep in the configured task environment") from exc
    return pk
