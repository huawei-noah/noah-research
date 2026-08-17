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

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import json

from pydantic import Field

from deepcraft_core  import Message
from deepcraft_core.tool import MCPClientManager, ToolCall
from deepcraft_core.message import Role

from ..base import AgentState
from .react import ReActAgent

logger = logging.getLogger(__name__)


class MCPReActAgent(ReActAgent):
    """Agent for interacting with MCP (Model Context Protocol) servers.

    This agent connects to an MCP server using either SSE or stdio transport
    and makes the server's tools available through the agent's tool interface.
    Attributes:
        name (str): The name of the agent, set to "mcp_agent".
        description (str): A description of the agent.
        mcpClients (MCPClientManager): Manages connections to MCP servers.
        availableTools (MCPClientManager): Provides access to tools available on the MCP server.
        toolSchemas (Dict[str, Dict[str, Any]]): Tracks schemas of tools to detect changes.

    """

    name: str = "mcp_agent"
    description: str = "An agent that connects to an MCP server and uses its tools."

    # Initialize MCP tool collection
    mcpClients: MCPClientManager = None
    availableTools: MCPClientManager = None  # Will be set in initialize()

    # Track tool schemas to detect changes
    toolSchemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    async def initialize(self,
            mcp_servers: Optional[Union[Dict, MCPClientManager]] = None,
            modify_system_prompt: bool = True) -> None:
        """Initialize the MCP connection.

        Args:
            mcp_servers: configuration of mcp servers or connected mcp servers
            modify_system_prompt: modify system prompt with available mcp tools info
        """
        if not mcp_servers and isinstance(mcp_servers, MCPClientManager):
            # Connect to the MCP server based on configuration
            self.mcpClients = MCPClientManager.from_existed(
                name=mcp_servers.name,
                description=mcp_servers.description,
                exit_stack=mcp_servers.exitStack,
                clients=mcp_servers.clients
                )

        else:
            self.mcpClients = MCPClientManager()
            # Connect to the MCP server based on configuration
            await self.mcpClients.connect(mcp_servers)

        # Set availableTools to our MCP instance
        self.availableTools = self.mcpClients

        # Store initial tool schemas
        await self._refresh_tools(modify_system_prompt)

        # Add system message about available tools
        self.tool_names = list(self.mcpClients.tool_map.keys())
        self.tools_info = ", ".join(self.tool_names)

        if modify_system_prompt:
            # Add system prompt and available tools information
            self.memory.add_message(
                Message.system_message(
                    f"{self.systemPrompt}\n\nAvailable MCP tools: {self.tools_info}"
                )
            )

    async def _refresh_tools(self, modify_system_prompt: bool = True) -> Tuple[List[str], List[str]]:
        """Refresh the list of available tools from the MCP servers.

        Args:
            modify_system_prompt: modify system prompt with available mcp tools info

        Returns:
            A tuple of (added_tools, removed_tools)
        """
        if not self.mcpClients.clients:
            return [], []

        # Get current tool schemas directly from the server
        current_tools = {}
        for client in self.mcpClients.clients:
            response = await client.session.list_tools()
            current_tools.update({tool.name: tool.inputSchema for tool in response.tools})

        # Determine added, removed, and changed tools
        current_names = set(current_tools.keys())
        previous_names = set(self.toolSchemas.keys())

        added_tools = list(current_names - previous_names)
        removed_tools = list(previous_names - current_names)

        # Check for schema changes in existing tools
        changed_tools = []
        for name in current_names.intersection(previous_names):
            if current_tools[name] != self.toolSchemas.get(name):
                changed_tools.append(name)

        # Update stored schemas
        self.toolSchemas = current_tools

        # Log and notify about changes
        if added_tools:
            logger.info(f"Added {len(added_tools)} MCP tools: {added_tools}")
            if modify_system_prompt:
                self.memory.add_message(
                    Message.system_message(f"New tools available: {', '.join(added_tools)}")
                )
        if removed_tools:
            logger.info(f"Removed {len(removed_tools)} MCP tools: {removed_tools}")
            if modify_system_prompt:
                self.memory.add_message(
                    Message.system_message(
                        f"Tools no longer available: {', '.join(removed_tools)}"
                    )
                )
        if changed_tools:
            logger.info(f"Changed {len(changed_tools)} MCP tools: {changed_tools}")

        return added_tools, removed_tools

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling.
        Args:
            command (ToolCall): The tool call to execute.
        Returns:
            str: A formatted result or error message from the tool execution.
        """
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.availableTools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.availableTools.execute(name=name, tool_input=args)

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            result = await self._run(request)
            return result
        except Exception as e:
            await self.cleanup()
            error_msg = f"⚠️ Agent {self.name} encountered a problem: {str(e)}"
            logger.exception(error_msg)

    async def _run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.updateMemory(Role.USER, request)

        results: List[str] = []
        # asyncio cancel scope error with mcp clients
        # async with self.stateContext(AgentState.RUNNING):
        while (self.currentStep < self.maxSteps and self.state != AgentState.FINISHED):
            self.currentStep += 1
            logger.info(f"Executing step {self.currentStep}/{self.maxSteps}")
            stepResult = await self.step()

            # Check for stuck state
            if super().isStuck():
                super().handleStuckState()

            results.append(f"Step {self.currentStep}: {stepResult}")

        if self.currentStep >= self.maxSteps:
            results.append(f"Terminated: Reached max steps ({self.maxSteps})")

        return "\n".join(results) if results else "No steps executed"

    def _should_finish_execution(self, name: str, **kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        # Terminate if the tool name is 'terminate'
        return name.lower() == "terminate"

    async def cleanup(self) -> None:
        """Clean up MCP connection when done."""
        if self.mcpClients:
            await self.mcpClients.disconnect()
            logger.info("MCP connection closed")
