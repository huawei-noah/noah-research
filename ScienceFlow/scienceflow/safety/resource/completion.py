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

from pathlib import Path
from typing import Any

from .artifacts import configured_artifact_state, deliverable_artifact_state, safe_candidate_artifact, submission_completion_state


def deliverable_completion_state(
    workspace_dir: str | Path,
    *,
    terminal_signal_seen: bool = False,
    terminal_signal_kind: str = "",
    settle_sec: float = 120.0,
    candidate_artifact: str = "",
) -> dict[str, Any]:
    configured = safe_candidate_artifact(candidate_artifact)
    if configured and configured != "submission.csv":
        artifact = configured_artifact_state(workspace_dir, configured, settle_sec=settle_sec)
        if artifact.get("complete"):
            return {
                **artifact,
                "reason": "candidate_artifact_complete",
                "terminal_signal_seen": bool(terminal_signal_seen),
                "terminal_signal_kind": str(terminal_signal_kind or ""),
            }
        return {
            **artifact,
            "terminal_signal_seen": bool(terminal_signal_seen),
            "terminal_signal_kind": str(terminal_signal_kind or ""),
        }

    submission = submission_completion_state(workspace_dir, settle_sec=settle_sec)
    if submission.get("complete"):
        return {
            **submission,
            "reason": "submission_complete",
            "terminal_signal_seen": bool(terminal_signal_seen),
            "terminal_signal_kind": str(terminal_signal_kind or ""),
        }
    if submission.get("deliverable_validity") == "produced_invalid":
        return {
            "complete": False,
            "mode": "submission",
            "reason": str(submission.get("reason") or "invalid_submission_schema"),
            "deliverable_validity": "produced_invalid",
            "terminal_signal_seen": bool(terminal_signal_seen),
            "terminal_signal_kind": str(terminal_signal_kind or ""),
            "submission_state": submission,
        }

    artifact = deliverable_artifact_state(workspace_dir, settle_sec=settle_sec)
    if terminal_signal_seen and artifact.get("complete"):
        return {
            **artifact,
            "reason": str(artifact.get("mode") or "deliverable") + "_complete",
            "terminal_signal_seen": True,
            "terminal_signal_kind": str(terminal_signal_kind or ""),
            "submission_state": submission,
        }

    reason = "terminal_signal_missing" if artifact.get("complete") else str(artifact.get("reason") or submission.get("reason") or "incomplete")
    return {
        "complete": False,
        "mode": "deliverable",
        "reason": reason,
        "terminal_signal_seen": bool(terminal_signal_seen),
        "terminal_signal_kind": str(terminal_signal_kind or ""),
        "submission_state": submission,
        "artifact_state": artifact,
    }
