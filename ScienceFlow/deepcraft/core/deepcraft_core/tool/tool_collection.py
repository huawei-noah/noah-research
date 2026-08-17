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

"""Collection classes for managing multiple tools."""
from typing import Any, Dict, List

from .base import BaseTool, ToolFailure, ToolResult, ToolError


class ToolCollection:
    """A collection of defined tools."""

    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        """Add tools parameters to the collection."""
        return [tool.to_param() for tool in self.tools]

    async def execute(
        self,
        *,
        name: str,
        tool_input: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a single tool.
        Args:
            name (str): The name of the tool to execute.
            tool_input (Dict[str, Any], optional): The input parameters for the tool. Defaults to None.
            **kwargs: Extra arguments forwarded to the tool (e.g. ``on_output`` for bash streaming).

        Returns:
            ToolResult: The result of the tool execution, which can be a ToolSuccess or ToolFailure object.
        """
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            result = await tool(**(tool_input or {}), **kwargs)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)

    async def execute_all(self) -> List[ToolResult]:
        """Execute all tools in the collection sequentially."""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
        return results

    def get_tool(self, name: str) -> BaseTool:
        """Retrieves a tool from the tool map by its name.
        Attributes:
            name (str): The name of the tool.
        """
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool):
        """Adds a new tool to the collection.
        Attributes:
            tool (BaseTool): The new tool.
        """
        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        """Adds multiple tools to the collection.
        Attributes:
            tools (BaseTool): The list of tools to add.
            """
        for tool in tools:
            self.add_tool(tool)
        return self
