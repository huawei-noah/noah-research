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


def build_command_timeout_feedback(
    *,
    timeout_sec: float,
    resource_class: str = "",
    gpu_ids: list[str] | None = None,
    saw_progress: bool = False,
    saw_artifact: bool = False,
    current_phase: str = "",
) -> str:
    timeout = max(0.0, float(timeout_sec or 0.0))
    cls = str(resource_class or "unknown")
    ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
    phase = str(current_phase or "unknown")
    progress = "yes" if saw_progress else "no"
    artifact = "yes" if saw_artifact else "no"
    location = f" on GPU {','.join(ids)}" if ids else ""
    if saw_progress or saw_artifact:
        reason = "it was still running when the bash budget ended, so the run was too large for one command budget"
    else:
        reason = "no useful progress finished before the bash budget ended, so the run was too large or too quiet for one command budget"
    return (
        f"RESOURCE_FEEDBACK: command timed out because the {timeout:.0f}s bash budget was exhausted "
        f"while {cls}{location} was in phase {phase}; progress_seen={progress}, "
        f"artifact_seen={artifact}; {reason}.\n"
    )
