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

"""Model-visible feedback for bash process creation failures."""

from __future__ import annotations

from deepcraft_core.tool import ToolResult


def build_spawn_failure_tool_result(
    exc: Exception,
    *,
    elapsed_sec: float,
) -> ToolResult:
    """Return a tool error instead of letting transient spawn failures crash a worker."""

    reason = f"{type(exc).__name__}: {exc}".strip()
    block = "\n".join(
        [
            f"[spawn-failed, {max(0.0, elapsed_sec):.1f}s]",
            (
                "[infra-error] Bash could not start a subprocess. "
                "This is an infrastructure/resource pressure signal, not evidence "
                "about the solution quality."
            ),
            f"reason: {reason}",
            (
                "Continue the search until the wall-clock budget expires; wait, retry, "
                "or reduce local concurrency before running another heavy command."
            ),
        ],
    )
    return ToolResult(output=block, error=f"bash spawn failed: {type(exc).__name__}")
