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


@dataclass(frozen=True)
class ESTRAMagentConfig:
    enabled: bool = False
    sidecar_enabled: bool = False
    min_parent_runtime_sec: float = 300.0
    sidecar_mode: str = "cpu_only"
    budget_sec: float = 900.0
    join_inject_parent: bool = True
    join_inject_estra: bool = True
    join_inject_resource_context: bool = True

    @classmethod
    def from_values(
        cls,
        *,
        enabled: bool,
        sidecar_enabled: bool,
        min_parent_runtime_sec: float,
        sidecar_mode: str,
        budget_sec: float,
        join_inject_parent: bool,
        join_inject_estra: bool,
        join_inject_resource_context: bool,
    ) -> "ESTRAMagentConfig":
        mode = str(sidecar_mode or "cpu_only").strip().lower()
        if mode != "cpu_only":
            mode = "cpu_only"
        return cls(
            enabled=bool(enabled),
            sidecar_enabled=bool(sidecar_enabled),
            min_parent_runtime_sec=max(1.0, float(min_parent_runtime_sec or 300.0)),
            sidecar_mode=mode,
            budget_sec=max(1.0, float(budget_sec or 900.0)),
            join_inject_parent=bool(join_inject_parent),
            join_inject_estra=bool(join_inject_estra),
            join_inject_resource_context=bool(join_inject_resource_context),
        )

    def sidecar_allowed(self) -> bool:
        return bool(self.enabled and self.sidecar_enabled and self.sidecar_mode == "cpu_only")
