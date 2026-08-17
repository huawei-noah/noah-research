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

from dataclasses import dataclass
from typing import Any

KILL = "KILL"
NO_ACTION = "NO_ACTION"
TIMEBOX = "TIMEBOX"

_VALUE_REVIEW_TYPES = {
    "kill_proposal",
    "periodic_efficiency_review",
    "quick_probe_review",
    "resource_contention_review",
}
_KILL_ACTIONS = {"KILL_AND_REPLAN", "STOP_BOUNDARY_VIOLATION"}
_NO_ACTIONS = {"CONTINUE", "DENY_KILL", "OBSERVE_MORE", "MARK_STALLED_NO_KILL"}


@dataclass(frozen=True)
class ValueReviewOutcome:
    outcome: str
    suppress_main_agent_feedback: bool = False
    clear_on: str = ""

    def to_json(self) -> dict[str, Any]:
        out = {
            "outcome": self.outcome,
            "suppress_main_agent_feedback": self.suppress_main_agent_feedback,
        }
        if self.clear_on:
            out["clear_on"] = self.clear_on
        return out


def is_value_review_proposal(proposal: dict[str, Any] | None) -> bool:
    proposal_type = str((proposal or {}).get("proposal_type") or "").strip().lower()
    return proposal_type in _VALUE_REVIEW_TYPES


def normalize_value_review_outcome(action: str, proposal: dict[str, Any] | None = None, *, canonical_outcome: str = "", clear_on: str = "") -> ValueReviewOutcome:
    canonical = str(canonical_outcome or "").strip().upper()
    if canonical == TIMEBOX:
        return ValueReviewOutcome(TIMEBOX, suppress_main_agent_feedback=True, clear_on=str(clear_on or ""))
    if canonical == KILL:
        return ValueReviewOutcome(KILL, suppress_main_agent_feedback=False)
    if canonical == NO_ACTION and is_value_review_proposal(proposal):
        return ValueReviewOutcome(NO_ACTION, suppress_main_agent_feedback=True)
    upper = str(action or "").strip().upper()
    if upper in _KILL_ACTIONS:
        return ValueReviewOutcome(KILL, suppress_main_agent_feedback=False)
    if is_value_review_proposal(proposal) and upper in _NO_ACTIONS:
        return ValueReviewOutcome(NO_ACTION, suppress_main_agent_feedback=True)
    return ValueReviewOutcome(NO_ACTION, suppress_main_agent_feedback=False)
