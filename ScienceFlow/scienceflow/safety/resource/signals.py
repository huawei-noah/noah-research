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

import re
from typing import Any

_INVALID_METRIC_RE = re.compile(r"\b(?:nan|[+-]?inf|infinite|diverg(?:ed|ence))\b", re.IGNORECASE)
_METRIC_CONTEXT_RE = re.compile(
    r"\b(?:loss|pred|prediction|metric|auc|score|val|valid|train|epoch|batch)\b",
    re.IGNORECASE,
)
_HIGHER_BETTER_ZERO_RE = re.compile(
    r"\b(?:val(?:idation)?[_ -]?)?(?:auc|accuracy|acc|f1|mcc|map|ndcg|r2)\s*[:=]\s*0(?:\.0+)?(?:\s|$|\|)",
    re.IGNORECASE,
)
_TERMINAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "final_metric",
        re.compile(
            r"\bfinal\s+(?:validation\s+)?(?:score|metric|auc|accuracy|loss|rmse|mae|cost|objective|distance)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "best_metric",
        re.compile(
            r"\bbest\s+(?:validation\s+)?(?:score|metric|auc|accuracy|loss|rmse|mae|cost|objective|distance)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "training_complete",
        re.compile(r"\b(?:training\s+(?:complete|completed|finished)|finished\s+training)\b", re.IGNORECASE),
    ),
    ("early_stopping", re.compile(r"\bearly\s+stopp(?:ing|ed)\b", re.IGNORECASE)),
    (
        "artifact_saved",
        re.compile(
            r"\b(?:best\s+model\s+saved|model\s+saved|checkpoint\s+saved|solution\s+saved|result\s+saved)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "solver_terminal",
        re.compile(
            r"\b(?:converged|optimal|incumbent|best\s+objective|final\s+objective|final\s+cost|best\s+distance|final\s+distance)\b",
            re.IGNORECASE,
        ),
    ),
)


def scan_output_health(text: str) -> dict[str, Any]:
    raw = str(text or "")
    if not raw:
        return {}
    out: dict[str, Any] = {}
    if _INVALID_METRIC_RE.search(raw) and _METRIC_CONTEXT_RE.search(raw):
        out["invalid_metric_events"] = 1
        out["last_invalid_metric_text"] = _snippet(raw)
    if _HIGHER_BETTER_ZERO_RE.search(raw):
        out["zero_score_events"] = 1
        out["last_zero_score_text"] = _snippet(raw)
    terminal = terminal_signal(raw)
    if terminal:
        out["terminal_signal_events"] = 1
        out["terminal_signal_kind"] = terminal["kind"]
        out["last_terminal_signal_text"] = terminal["text"]
    return out


def terminal_signal(text: str) -> dict[str, str]:
    raw = str(text or "")
    if not raw:
        return {}
    for kind, pattern in _TERMINAL_PATTERNS:
        if pattern.search(raw):
            return {"kind": kind, "text": _snippet(raw)}
    return {}


def _snippet(text: str, limit: int = 240) -> str:
    one_line = " ".join(str(text or "").split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."
