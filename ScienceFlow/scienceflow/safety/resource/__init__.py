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

"""Resource-safety helpers for process health and deliverable completion."""

from .completion import deliverable_completion_state
from .signals import scan_output_health

__all__ = [
    "deliverable_completion_state",
    "scan_output_health",
]
from .review_boundary import PROGRESS_WINDOW, RESOURCE_PRESSURE, ROUTE_VALUE, STALL, TIMEBOX_EXPIRED, ReviewBoundary
from .review_outcome import KILL, NO_ACTION, TIMEBOX, ValueReviewOutcome, normalize_value_review_outcome
from .review_signal import ResourceReviewSignal, build_review_signal
from .review_state import (
    ACTIONED,
    RUNNING_HEALTHY,
    RUNNING_NO_PROGRESS,
    STALL_SUSPECT,
    TIMEBOX_ACTIVE,
    WARMUP,
    ResourceReviewConfig,
    ResourceReviewState,
    advance_review_state,
    apply_kill,
    apply_no_action,
    apply_timebox,
    compute_timebox_sec,
    normalize_clear_on,
    mark_review_emitted,
    new_review_state,
    next_review_boundary,
)

__all__ += [
    "ACTIONED",
    "KILL",
    "NO_ACTION",
    "PROGRESS_WINDOW",
    "RESOURCE_PRESSURE",
    "ROUTE_VALUE",
    "RUNNING_HEALTHY",
    "RUNNING_NO_PROGRESS",
    "STALL",
    "STALL_SUSPECT",
    "TIMEBOX",
    "TIMEBOX_ACTIVE",
    "TIMEBOX_EXPIRED",
    "WARMUP",
    "ResourceReviewConfig",
    "ResourceReviewSignal",
    "ResourceReviewState",
    "ReviewBoundary",
    "ValueReviewOutcome",
    "advance_review_state",
    "apply_kill",
    "apply_no_action",
    "apply_timebox",
    "build_review_signal",
    "compute_timebox_sec",
    "mark_review_emitted",
    "new_review_state",
    "next_review_boundary",
    "normalize_clear_on",
    "normalize_value_review_outcome",
]
