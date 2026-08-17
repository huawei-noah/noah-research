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

"""Typed records used by the monitor trace HTML view."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TracePoint:
    row_order: int
    x_value: float
    x_label: str
    metric: float
    metric_name: str
    worker_id: str
    stage_id: str
    candidate_id: str
    validity: str
    metric_validity: str
    valid_comparable: bool
    candidate_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BestPoint:
    x_value: float
    x_label: str
    metric: float
    stage_id: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    x_value: float | None
    x_label: str
    kind: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeaderboardEntry:
    rank: int
    user: str
    score: float
    submitted_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Leaderboard:
    title: str
    metric_name: str
    source_url: str
    lower_is_better: bool | None
    display_limit: int
    entries: list[LeaderboardEntry] = field(default_factory=list)
    our_score: float | None = None
    our_score_source: str = ""
    estimated_rank: int | None = None
    gap_to_rank1: float | None = None
    gap_to_next: float | None = None
    rank_basis: str = "known_public_entries"
    score_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "metric_name": self.metric_name,
            "source_url": self.source_url,
            "lower_is_better": self.lower_is_better,
            "display_limit": self.display_limit,
            "entries": [e.to_dict() for e in self.entries],
            "our_score": self.our_score,
            "our_score_source": self.our_score_source,
            "estimated_rank": self.estimated_rank,
            "gap_to_rank1": self.gap_to_rank1,
            "gap_to_next": self.gap_to_next,
            "rank_basis": self.rank_basis,
            "score_source": self.score_source,
        }


@dataclass(frozen=True)
class TaskTrace:
    run_id: str
    exp_id: str
    gpu_list: str
    task_root: str
    status: str
    metric_name: str
    lower_is_better: bool
    x_mode: str
    summary: dict[str, Any] = field(default_factory=dict)
    points: list[TracePoint] = field(default_factory=list)
    best_points: list[BestPoint] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    leaderboard: Leaderboard | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "exp_id": self.exp_id,
            "gpu_list": self.gpu_list,
            "task_root": self.task_root,
            "status": self.status,
            "metric_name": self.metric_name,
            "lower_is_better": self.lower_is_better,
            "x_mode": self.x_mode,
            "summary": self.summary,
            "points": [p.to_dict() for p in self.points],
            "best_points": [p.to_dict() for p in self.best_points],
            "events": [e.to_dict() for e in self.events],
            "leaderboard": self.leaderboard.to_dict() if self.leaderboard else None,
        }


@dataclass(frozen=True)
class MonitorTraceReport:
    schema_version: int
    generated_at_utc: str
    manifest_path: str
    tasks: list[TaskTrace]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "manifest_path": self.manifest_path,
            "tasks": [t.to_dict() for t in self.tasks],
        }
