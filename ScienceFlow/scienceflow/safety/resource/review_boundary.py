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

STALL = "stall"
ROUTE_VALUE = "route_value"
PROGRESS_WINDOW = "progress_window"
TIMEBOX_EXPIRED = "timebox_expired"
RESOURCE_PRESSURE = "resource_pressure"


@dataclass(frozen=True)
class ReviewBoundary:
    kind: str
    reason: str

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "reason": self.reason}
