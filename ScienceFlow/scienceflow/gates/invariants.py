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

"""Non-configurable safety checks applied before every gate plugin."""

from __future__ import annotations

import math

from scienceflow.gates.evaluator.models import GateDecision, MetricEvent


FINAL_TRIGGERS = frozenset({"final", "finalize", "global_merge"})


def failure_decision(
    event: MetricEvent,
    *,
    trigger: str,
    reason_code: str,
    message: str,
) -> GateDecision:
    """Build the shared fail-closed retry/reject decision."""

    normalized_trigger = str(trigger or "stage_end").strip().lower()
    return GateDecision(
        action="reject" if normalized_trigger in FINAL_TRIGGERS else "retry",
        accepted=False,
        candidate_ready=bool(event.candidate_ready and event.validation_ok),
        selection_eligible=False,
        reason_code=reason_code,
        message=message,
    )


def check_core_invariants(
    event: MetricEvent,
    *,
    trigger: str,
) -> GateDecision | None:
    """Return a failure for unsafe evaluator facts, otherwise ``None``.

    Task and run configuration intentionally cannot disable these checks.
    Policy plugins only run after the normalized event passes this boundary.
    """

    if not event.validation_ok:
        return failure_decision(
            event,
            trigger=trigger,
            reason_code="validation_failed",
            message=event.metric_note or "candidate validation failed",
        )
    if not event.candidate_ready:
        return failure_decision(
            event,
            trigger=trigger,
            reason_code="candidate_not_ready",
            message=event.metric_note or "candidate artifact is not ready",
        )
    if event.metric_value is not None:
        try:
            finite = math.isfinite(float(event.metric_value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            return failure_decision(
                event,
                trigger=trigger,
                reason_code="metric_not_finite",
                message="evaluator primary metric is not finite",
            )
    return None
