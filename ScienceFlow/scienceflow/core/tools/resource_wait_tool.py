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

"""Managed resource wait tool for LNR resource feedback."""

from __future__ import annotations

from typing import Any

from deepcraft_core.tool import BaseTool, ToolResult
from pydantic import ConfigDict, Field


class ResourceWaitTool(BaseTool):
    name: str = "resource_wait"
    description: str = (
        "Wait for a resource-state change only when recent RESOURCE_FEEDBACK "
        "provided a wait_token. This is not a general sleep tool and never "
        "launches GPU work or chooses a research route."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "wait_token": {
                "type": "string",
                "description": "Token copied exactly from recent RESOURCE_FEEDBACK wait_token.",
            },
            "max_wait_sec": {
                "type": "number",
                "description": "Optional cap; the system wakes earlier on material resource events.",
            },
            "reason": {
                "type": "string",
                "description": "Short reason for waiting, e.g. resource_busy.",
            },
        },
        "required": ["wait_token"],
    }

    resource_observer: Any | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(
        self,
        wait_token: str | None = None,
        max_wait_sec: float | None = None,
        reason: str = "resource_busy",
        **kwargs: Any,
    ) -> ToolResult:
        kwargs.pop("config", None)
        token = str(wait_token or "").strip()
        if not token:
            return ToolResult(
                output=(
                    "RESOURCE_FEEDBACK: RESOURCE_WAIT_NOT_AVAILABLE because no active resource wait option; "
                    "retry_allowed=false.\n"
                ),
            )
        observer = self.resource_observer
        wait_fn = getattr(observer, "managed_resource_wait", None) if observer is not None else None
        if wait_fn is None:
            return ToolResult(
                output=(
                    "RESOURCE_FEEDBACK: RESOURCE_WAIT_NOT_AVAILABLE because resource observer is unavailable; "
                    "retry_allowed=false.\n"
                ),
            )
        result = wait_fn(wait_token=token, max_wait_sec=max_wait_sec, reason=reason)
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, dict):
            return ToolResult(output=str(result.get("feedback") or ""))
        return ToolResult(output=str(result or ""))
