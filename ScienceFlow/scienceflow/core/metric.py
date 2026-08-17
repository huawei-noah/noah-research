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

"""MetricValue / WorstMetricValue — unified comparison for optimization metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any


@dataclass
@total_ordering
class MetricValue:
    """Represents an optimization metric. Comparisons reflect *better*, not *larger*."""

    value: float | None
    maximize: bool | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.value is not None:
            self.value = float(self.value)

    def __gt__(self, other: Any) -> bool:
        if self.value is None:
            return False
        if other is None or getattr(other, "value", None) is None:
            return True
        if type(self) is not type(other) or self.maximize != other.maximize:
            return False
        if self.value == other.value:
            return False
        comp = self.value > other.value
        return comp if self.maximize else not comp

    def __eq__(self, other: Any) -> bool:
        if other is None:
            return False
        if type(self) is not type(other) or self.maximize != getattr(other, "maximize", None):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash((self.value, self.maximize))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        arrow = {"↑": True, "↓": False, "?": None}.get(None)
        for sym, val in (("↑", True), ("↓", False), ("?", None)):
            if self.maximize is val:
                arrow = sym
                break
        v = f"{self.value:.4f}" if self.value is not None else "nan"
        return f"Metric{arrow}({v})"

    @property
    def is_worst(self) -> bool:
        return self.value is None

    def to_dict(self) -> dict:
        return {"value": self.value, "maximize": self.maximize}

    @classmethod
    def from_dict(cls, d: dict) -> MetricValue:
        if d is None or d.get("value") is None:
            return WorstMetricValue()
        return cls(value=d["value"], maximize=d.get("maximize"))


@dataclass
class WorstMetricValue(MetricValue):
    """Always compares worse than any valid MetricValue."""

    value: None = None
