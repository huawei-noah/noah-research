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

"""Lazy public exports for tool primitives and optional integrations."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ToolCall": ("deepcraft_core.tool.base", "ToolCall"),
    "BaseTool": ("deepcraft_core.tool.base", "BaseTool"),
    "ToolResult": ("deepcraft_core.tool.base", "ToolResult"),
    "ToolError": ("deepcraft_core.tool.base", "ToolError"),
    "CLIResult": ("deepcraft_core.tool.base", "CLIResult"),
    "TOOL_CHOICE_VALUES": ("deepcraft_core.tool.base", "TOOL_CHOICE_VALUES"),
    "TOOL_CHOICE_TYPE": ("deepcraft_core.tool.base", "TOOL_CHOICE_TYPE"),
    "ToolChoice": ("deepcraft_core.tool.base", "ToolChoice"),
    "ToolCollection": ("deepcraft_core.tool.tool_collection", "ToolCollection"),
    "MCPClient": ("deepcraft_core.tool.mcp", "MCPClient"),
    "MCPClientManager": ("deepcraft_core.tool.mcp", "MCPClientManager"),
    "func_to_tool_instance": ("deepcraft_core.tool.convert", "func_to_tool_instance"),
    "func_to_openai_format": ("deepcraft_core.tool.convert", "func_to_openai_format"),
    "dict_to_tool_instance": ("deepcraft_core.tool.convert", "dict_to_tool_instance"),
    "dict_to_tool_fun": ("deepcraft_core.tool.convert", "dict_to_tool_fun"),
}

__all__ = [
    "ToolCall",
    "BaseTool",
    "ToolResult",
    "ToolError",
    "ToolCollection",
    "CLIResult",
    "TOOL_CHOICE_VALUES",
    "TOOL_CHOICE_TYPE",
    "ToolChoice",
    "MCPClient",
    "MCPClientManager",
    "func_to_tool_instance",
    "func_to_openai_format",
    "dict_to_tool_instance",
    "dict_to_tool_fun",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
