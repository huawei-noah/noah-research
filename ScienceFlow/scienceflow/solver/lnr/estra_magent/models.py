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

from dataclasses import dataclass, field
from typing import Any


ACCEPTED_QUALITY_GATES = {"use", "review"}


@dataclass(frozen=True)
class SidecarTicket:
    ticket_version: int = 1
    sidecar_id: str = ""
    parent_worker_id: str = ""
    parent_job_id: str = ""
    parent_state: str = "healthy_running"
    task_type: str = "submission_checker"
    resource_mode: str = "cpu_only"
    budget_sec: float = 900.0
    allowed_paths: list[str] = field(default_factory=list)
    readonly_paths: list[str] = field(default_factory=lambda: ["workspace", "logs", "dataset"])
    forbidden_actions: list[str] = field(
        default_factory=lambda: ["heavy_gpu_train", "overwrite_parent_files", "submit_as_stage"]
    )
    expected_outputs: list[str] = field(default_factory=lambda: ["sidecar_report.json"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_version": int(self.ticket_version),
            "sidecar_id": self.sidecar_id,
            "parent_worker_id": self.parent_worker_id,
            "parent_job_id": self.parent_job_id,
            "parent_state": self.parent_state,
            "task_type": self.task_type,
            "resource_mode": self.resource_mode,
            "budget_sec": float(self.budget_sec),
            "allowed_paths": list(self.allowed_paths),
            "readonly_paths": list(self.readonly_paths),
            "forbidden_actions": list(self.forbidden_actions),
            "expected_outputs": list(self.expected_outputs),
        }


@dataclass(frozen=True)
class JoinPacket:
    join_version: int = 1
    sidecar_id: str = ""
    parent_worker_id: str = ""
    parent_job_id: str = ""
    quality_gate: str = "review"
    summary_for_parent: str = ""
    summary_for_estra: str = ""
    artifact_refs: list[dict[str, str]] = field(default_factory=list)
    inject: dict[str, bool] = field(default_factory=dict)

    def is_accepted(self) -> bool:
        return self.quality_gate in ACCEPTED_QUALITY_GATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "join_version": int(self.join_version),
            "sidecar_id": self.sidecar_id,
            "parent_worker_id": self.parent_worker_id,
            "parent_job_id": self.parent_job_id,
            "quality_gate": self.quality_gate,
            "summary_for_parent": self.summary_for_parent,
            "summary_for_estra": self.summary_for_estra,
            "artifact_refs": [dict(x) for x in self.artifact_refs],
            "inject": dict(self.inject),
        }
