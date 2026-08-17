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

from contextlib import AsyncExitStack
from typing import List, Dict, Optional
import logging
import asyncio
import shutil

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from .base import BaseTool, ToolResult
from .tool_collection import ToolCollection

logger = logging.getLogger(__name__)

class MCPClientTool(BaseTool):
    """Represents a tool proxy that can be called on the MCP server from the client side."""

    session: Optional[ClientSession] = None

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool by making a remote call to the MCP server."""
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        try:
            result = await self.session.call_tool(self.name, kwargs)
            logger.info(f'mcp tool result: {result}')
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "No output returned.")
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")


class MCPClient(ToolCollection):
    """
    A collection of tools that connects to an MCP server and manages available tools through the Model Context Protocol.
    """

    session: Optional[ClientSession] = None
    description: str = "MCP client tools for server interaction"

    def __init__(self, name='mcp'):
        super().__init__()  # Initialize with empty tools list
        self.name = name

    async def connect_sse(self, exit_stack: AsyncExitStack, server_url: str) -> None:
        """Connect to an MCP server using SSE transport."""
        if not server_url:
            raise ValueError("Server URL is required.")
        if self.session:
            await self.disconnect()

        streams_context = sse_client(url=server_url)
        streams = await exit_stack.enter_async_context(streams_context)
        self.session = await exit_stack.enter_async_context(
            ClientSession(*streams)
        )

        await self._initialize_and_list_tools()

    async def connect_stdio(self, exit_stack: AsyncExitStack, 
                            command: str, args: List[str], env: Dict[str,str] = None) -> None:
        """Connect to an MCP server using stdio transport."""
        command = shutil.which("npx") if command == "npx" else command
        if not command:
            raise ValueError("Server command is required.")
        if self.session:
            await self.disconnect()

        server_params = StdioServerParameters(command=command, args=args, env=env)
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        self.session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self._initialize_and_list_tools()

    async def _initialize_and_list_tools(self) -> None:
        """Initialize session and populate tool map."""
        if not self.session:
            raise RuntimeError("Session not initialized.")

        await self.session.initialize()
        response = await self.session.list_tools()

        # Clear existing tools
        self.tools = tuple()
        self.tool_map = {}

        # Create proper tool objects for each server tool
        for tool in response.tools:
            server_tool = MCPClientTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
                session=self.session,
            )
            self.tool_map[tool.name] = server_tool

        self.tools = tuple(self.tool_map.values())
        msg = f"Connected to mcp server '{self.name}' with {len(response.tools)} tools: {[tool.name for tool in response.tools]}"
        logger.info(msg)

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        try:
            if self.session:
                self.session = None
                self.tools = tuple()
                self.tool_map = {}
        except Exception as e:
            logger.error(f"Error during closing {self.name}: {e}")


class MCPClientManager(ToolCollection):
    name: str = "mcp clients manager"
    clients: List[MCPClient] = []
    description: str = "Manage a list of MCP clients providing tools for server interaction"
    exitStack: AsyncExitStack = None
    fromExisted: bool = False

    @classmethod
    def from_existed(cls,
            name: str, 
            description: str,
            exit_stack : AsyncExitStack,
            clients: List[MCPClient]):
        new_manager = cls(
            name=name,
            description=description,
            exitStack=exit_stack,
            clients=clients,
            fromExisted=True,
            )
        for mcp_client in new_manager.clients:
            new_manager.tools += mcp_client.tools
            new_manager.tool_map.update(mcp_client.tool_map)
        return new_manager

    async def connect(self, mcp_servers: List[Dict]):
        if not self.exitStack:
            self.exitStack = AsyncExitStack()

        self.clients = []
        for server in mcp_servers:
            mcp_client = MCPClient(server.get('name'))
            connection_type = server.get('connection_type')
            if connection_type == 'sse':
                await mcp_client.connect_sse(self.exitStack, server_url=server.get('server_url',None))
            elif connection_type == 'stdio':
                await mcp_client.connect_stdio(self.exitStack, 
                    command=server.get('command'), args=server.get('args',[]), env=server.get('env',None))
            else:
                raise ValueError(
                    f"Failed to connect with MCP server {mcp_client.name}: unsupported connection type {connection_type}")

            self.clients.append(mcp_client)
            self.tools += mcp_client.tools
            self.tool_map.update(mcp_client.tool_map)

    async def disconnect(self) -> None:
        try:
            if not self.fromExisted:
                await self.exitStack.aclose()
                for client in self.clients:
                    await client.disconnect()
        except Exception as e:
            logging.error(f"Error closing mcp servers: {e}")
