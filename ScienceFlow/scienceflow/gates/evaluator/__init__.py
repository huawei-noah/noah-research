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

"""Evaluator plugins owned and invoked by the top-level Gate transaction.

Evaluators normalize task-specific artifacts into :class:`MetricEvent` facts.
They do not decide Stage admission; callers should enter through
``scienceflow.gates.GateService`` for an authoritative decision.
"""

from scienceflow.gates.evaluator.manager import EvaluatorManager
from scienceflow.gates.evaluator.models import (
    CandidateRef,
    EvalContext,
    EvaluationOutcome,
    EvaluationRequest,
    EvaluationResult,
    GateDecision,
    MetricEvent,
)

__all__ = [
    "CandidateRef",
    "EvalContext",
    "EvaluationOutcome",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluatorManager",
    "GateDecision",
    "MetricEvent",
]
