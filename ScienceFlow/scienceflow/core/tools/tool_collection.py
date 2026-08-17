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

"""ToolCollection that forwards extra kwargs (e.g. on_output) to tools."""

from __future__ import annotations

from typing import Any, Dict

from deepcraft_core.tool import ToolCollection as _DeepcraftToolCollection
from deepcraft_core.tool.base import ToolFailure, ToolError, ToolResult


class ToolCollection(_DeepcraftToolCollection):
    async def execute(
        self,
        *,
        name: str,
        tool_input: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        ti = tool_input or {}
        if name == "bash":
            cmd = ti.get("command")
            if cmd is None or not str(cmd).strip():
                return ToolFailure(
                    error="bash tool requires non-empty 'command' argument",
                )
        try:
            return await tool(**ti, **kwargs)
        except ToolError as e:
            return ToolFailure(error=e.message)
