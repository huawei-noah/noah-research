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

"""Top-level candidate admission interface.

Gate owns evaluator execution and returns the authoritative accept/retry/reject
outcome consumed by search and finalization layers.
"""

from scienceflow.gates.feedback import format_invalid_evaluator_feedback
from scienceflow.gates.policy import (
    DefaultGatePolicy,
    GateManager,
    GatePolicy,
    OptimizationFeasibilityGatePolicy,
    decide_metric_event,
    legacy_score_contract_enabled,
    normalize_gate_name,
    resolve_gate_config,
)
from scienceflow.gates.service import GateService

__all__ = [
    "DefaultGatePolicy",
    "GateManager",
    "GatePolicy",
    "GateService",
    "OptimizationFeasibilityGatePolicy",
    "decide_metric_event",
    "format_invalid_evaluator_feedback",
    "legacy_score_contract_enabled",
    "normalize_gate_name",
    "resolve_gate_config",
]
