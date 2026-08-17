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

from typing import Any


def _float_hint(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def _clip_score(value: Any) -> float:
    return max(0.0, min(1.0, _float_hint(value, 0.0)))


def calculate_admission_priority(value_hint: dict[str, Any] | None) -> dict[str, float]:
    hint = value_hint if isinstance(value_hint, dict) else {}
    expected = _clip_score(hint.get("expected_value_score"))
    diversity = _clip_score(hint.get("lineage_diversity_score"))
    near_submission = _clip_score(hint.get("near_submission_score"))
    starvation = _clip_score(hint.get("worker_starvation_score"))
    duplicate = _clip_score(hint.get("duplicate_penalty"))
    timeout = _clip_score(hint.get("timeout_history_penalty"))
    runtime = _clip_score(hint.get("long_runtime_penalty"))

    # Long jobs should not crowd out final submission work just because inference has a high timeout.
    effective_runtime = runtime * max(0.25, 1.0 - 0.6 * near_submission)
    score = (
        2.0 * expected
        + 1.2 * diversity
        + 1.0 * near_submission
        + 0.8 * starvation
        - 1.5 * duplicate
        - 1.2 * timeout
        - 0.8 * effective_runtime
    )
    return {
        "admission_priority_score": round(score, 4),
        "expected_value_score": expected,
        "lineage_diversity_score": diversity,
        "near_submission_score": near_submission,
        "worker_starvation_score": starvation,
        "duplicate_penalty": duplicate,
        "timeout_history_penalty": timeout,
        "long_runtime_penalty": runtime,
        "effective_long_runtime_penalty": round(effective_runtime, 4),
        "priority_schema_version": 2.0,
    }
