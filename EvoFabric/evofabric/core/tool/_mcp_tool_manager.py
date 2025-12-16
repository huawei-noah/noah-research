# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import json
from json import JSONDecodeError
from typing import cast, Dict, List, Optional

from pydantic import Field, PrivateAttr

from ._controller import ToolController
from ._mcp_session_controller import McpSessionController
from ._tool_manager_base import ToolManagerBase
from ._tool_utils import parse_mcp_tool_function
from ..typing import McpServerLink, PromptRequest, ResourceRequest, ToolCall, ToolCallResult
from ...logger import get_logger

logger = get_logger()


class McpToolManager(ToolManagerBase):
    """MCP (Model Context Protocol) Tool Manager for managing multiple MCP servers and their tools.

    Important Note:
        The actual tool names received by LLMs are generated as "server_name" + "_" + "tool_name".
        This naming convention supports using tools with the same name across different servers.
        For example, if server "math_server" has a tool "calculate", its full tool name will be "math_server_calculate".
    """

    server_links: Dict[str, McpServerLink] = Field(
        default_factory=dict,
        description="Dict of mcp server connection params. Keys are names of mcp servers, values are connection params.")

    timeout: int = Field(
        default=300,
        description="Timeout in seconds of tool execution")

    tool_controller: Optional[ToolController] = Field(
        default=None,
        description="Use ToolController to manage tool activity status"
    )

    persistent_link: bool = Field(
        default=False,
        description="Indicates whether to reuse the MCP connection. "
                    "True means the same connection is used for each interaction; "
                    "False means a new connection is created each time."
    )

    _server_tool_map: Dict[str, Dict[str, dict]] = PrivateAttr(default_factory=dict)
    """
    Maps each server to its available tools and their corresponding schemas.

    Structure example:
    {
        "server_a": {
            "tool_a": {...},  # tool a schema
            "tool_b": {...}   # tool b schema
        }, 
        "server_b": {
            ...
        }
    }

    Keys:
        Server name (str)
    Values:
        A dictionary where keys are tool names (str) and values are the tool schema dictionaries (dict)

    Purpose:
        Provides quick lookup of the tools available on a given server and their associated schema definitions.
    """

    _tool_server_lookup: Dict[str, str] = PrivateAttr(default_factory=dict)
    """A lookup table from tool name to server name."""

    _mcp_sessions: Dict[str, McpSessionController] = PrivateAttr(default_factory=dict)

    def _get_mcp_session(
            self,
            server_name: str,
            server_link: Optional[McpServerLink] = None,
            create_new: bool = True,
    ):
        """Establish or retrieve an MCP session connection to a server.

        Args:
            server_name: Name of the MCP server to connect to
            server_link: Optional server link configuration
            create_new: Flag to create new session if one doesn't exist

        Returns:
            McpSessionController instance for the specified server
        """
        if server_name not in self._mcp_sessions and create_new:
            self._mcp_sessions[server_name] = McpSessionController(
                server_link=server_link,
                server_name=server_name,
                persistent_link=self.persistent_link,
            )
        session = self._mcp_sessions.get(server_name, None)
        return session

    async def _update_tool_server_map(self, server_name: str):
        """Update the tool mapping for a specific server by fetching available tools from MCP.

        Args:
            server_name: Name of the server to update

        Returns:
            Dictionary of available tools for the server
        """
        if tools := self._server_tool_map.get(server_name, {}):
            return tools

        self._server_tool_map[server_name] = {}
        async with self._get_mcp_session(server_name, self.server_links[server_name]) as mcp_session:
            mcp_tool_lst = await mcp_session.session.list_tools()
        for tool in mcp_tool_lst.tools:
            mcp_tool_schema = parse_mcp_tool_function(tool, server_name)
            self._server_tool_map[server_name][tool.name] = mcp_tool_schema
            self._tool_server_lookup[mcp_tool_schema["function"]["name"]] = server_name
        return self._server_tool_map[server_name]

    async def _split_server_tool_name(self, tool_name: str) -> list[str]:
        """Split a fully qualified tool name into server name and tool name components.

        Args:
            tool_name: Fully qualified tool name (format: "server_name_tool_name")

        Returns:
            List [server_name, tool_name]
        """
        if self._tool_server_lookup.get(tool_name, None):
            server_name = self._tool_server_lookup[tool_name]
            raw_tool_name = tool_name.removeprefix(server_name + "_")

            return [server_name, raw_tool_name]
        else:
            for server_name, _ in self.server_links.items():
                await self._update_tool_server_map(server_name=server_name)

            if self._tool_server_lookup.get(tool_name, None):
                server_name = self._tool_server_lookup[tool_name]
                raw_tool_name = tool_name.removeprefix(server_name + "_")
                return [server_name, raw_tool_name]
            else:
                raise KeyError(f"Got an unknown tool name: {tool_name}")

    async def add_mcp_servers(self, mcp_server_links: Dict[str, McpServerLink]):
        """Add new MCP servers to the manager.

        Args:
            mcp_server_links: Dictionary of server name to McpServerLink configurations
        """
        # solve a situation that 2 mcp server have same names.
        for server_name in mcp_server_links:
            if server_name in self.server_links:
                await self.delete_mcp_servers(mcp_server_names=[server_name])

        self.server_links.update(mcp_server_links)

        for server_name, server_link in self.server_links.items():
            await self._update_tool_server_map(server_name)

    async def delete_mcp_servers(self, mcp_server_names: List[str]):
        """Remove MCP servers from the manager and clean up connections.

        Args:
            mcp_server_names: List of server names to remove
        """
        for server_name in mcp_server_names:
            self.server_links.pop(server_name)

            link = self._get_mcp_session(
                server_name=server_name,
                create_new=False,
            )
            if link:
                await link.disconnect()

            self._server_tool_map.pop(server_name, {})

            # delete corresponding item in self._tool_server_lookup
            deleted_tool_names = []
            for tool_name, tool_server in self._tool_server_lookup.items():
                if tool_server == server_name:
                    deleted_tool_names.append(tool_name)

            for tool_name in deleted_tool_names:
                self._tool_server_lookup.pop(tool_name)

    async def list_tools(self, server_name: str = None):
        """List available tools, optionally filtered by server name and tool controller.

        Args:
            server_name: Optional server name to filter results

        Returns:
            List of tool schemas (with tool controller status filtered if enabled)
        """
        if server_name:
            schemas = await self._update_tool_server_map(server_name)
            schemas = schemas.values()
        else:
            schemas = []
            for server_name in self.server_links.keys():
                _schemas = await self._update_tool_server_map(server_name)
                _schemas = _schemas.values()
                schemas.extend(_schemas)

        if self.tool_controller:
            schemas = [x for x in schemas if self.tool_controller.check_tool_status(x["function"]["name"])]

        return schemas

    async def call_tools(
            self,
            tasks: List[ToolCall],
    ) -> List[ToolCallResult]:
        """Execute tool calls for the provided tasks.

        Args:
            tasks: List of ToolCall objects to execute

        Returns:
            List of ToolCallResult objects containing execution outcomes
        """

        async def _run_tool(mcp_tool_call: ToolCall):
            if self.tool_controller and not self.tool_controller.check_tool_status(mcp_tool_call.function.name):
                raise ValueError(f"Cannot call {mcp_tool_call.function.name} because this tool is deactivate now.")

            server_name, tool_name = await self._split_server_tool_name(mcp_tool_call.function.name)
            if server_name not in self.server_links:
                return ToolCallResult(
                    tool_call_id=mcp_tool_call.id,
                    success=False,
                    content=f"Cannot find tool named {mcp_tool_call.function.name}, unknown mcp server: {server_name}"
                )
            try:
                arguments = json.loads(mcp_tool_call.function.arguments)
            except JSONDecodeError as e:
                return ToolCallResult(
                    tool_call_id=mcp_tool_call.id,
                    success=False,
                    content=f"Tool call argument parse failed, not a valid JSON: {mcp_tool_call.function.arguments}"
                )

            async with self._get_mcp_session(server_name, self.server_links[server_name]) as mcp_session:
                tool_res = await mcp_session.session.call_tool(tool_name, arguments)

            if tool_res.isError:
                return ToolCallResult(
                    tool_call_id=mcp_tool_call.id,
                    success=False,
                    content=tool_res.content[0].text
                )
            else:
                return ToolCallResult(
                    tool_call_id=mcp_tool_call.id,
                    success=True,
                    content=tool_res.content[0].text
                )

        coros = [_run_tool(t) for t in tasks]

        if self.timeout:
            results = await asyncio.wait_for(asyncio.gather(*coros), timeout=self.timeout)
        else:
            results = await asyncio.gather(*coros)
        return cast(List[ToolCallResult], results)

    async def list_prompts(self, server_name: str = None):
        """List available prompts, optionally filtered by server name.

        Args:
            server_name: Optional server name to filter results

        Returns:
            List of available prompts
        """
        server_name_list = server_name or list(self.server_links.keys())
        server_name_list = [server_name_list] if not isinstance(
            server_name_list, list) else server_name_list

        result = []
        for server_name in server_name_list:
            if server_name not in self.server_links:
                raise KeyError(
                    f"Got an unknown server name, exist servers: {self.server_links.keys()}")
            async with self._get_mcp_session(server_name, self.server_links[server_name]) as session:
                res = await session.session.list_prompts()
            result.extend(res.prompts)

        return result

    async def list_resources(self, server_name: str = None):
        """List available resources, optionally filtered by server name.

        Args:
            server_name: Optional server name to filter results

        Returns:
            List of available resources
        """
        server_name_list = server_name or list(self.server_links.keys())
        server_name_list = [server_name_list] if not isinstance(
            server_name_list, list) else server_name_list

        result = []
        for server_name in server_name_list:
            if server_name not in self.server_links:
                raise KeyError(
                    f"Got an unknown server name, exist servers: {self.server_links.keys()}")
            async with self._get_mcp_session(server_name, self.server_links[server_name]) as session:
                res = await session.session.list_resources()
            result.extend(res.resources)
        return result

    async def list_resource_templates(self, server_name: str = None):
        """List available resource templates, optionally filtered by server name.

        Args:
            server_name: Optional server name to filter results

        Returns:
            List of available resource templates
        """
        server_name_list = server_name or list(self.server_links.keys())
        server_name_list = [server_name_list] if not isinstance(
            server_name_list, list) else server_name_list

        result = []
        for server_name in server_name_list:
            if server_name not in self.server_links:
                raise KeyError(
                    f"Got an unknown server name, exist servers: {self.server_links.keys()}")
            async with self._get_mcp_session(server_name, self.server_links[server_name]) as session:
                res = await session.session.list_resource_templates()
            result.extend(res.resourceTemplates)
        return result

    async def read_resource(self, resources: List[ResourceRequest]):
        """Read specified resources from MCP servers.

        Args:
            resources: List of ResourceRequest objects

        Returns:
            List of resource responses
        """
        results = []
        for resource in resources:
            async with self._get_mcp_session(resource.server_name, self.server_links[resource.server_name]) as session:
                res = await session.session.read_resource(resource.url)
                results.append(res)
        return results

    async def get_prompt(self, prompt_requests: List[PromptRequest]):
        """Get specified prompts from MCP servers.

        Args:
            prompt_requests: List of PromptRequest objects

        Returns:
            List of prompt responses
        """
        results = []
        for prompt_request in prompt_requests:
            async with self._get_mcp_session(
                    prompt_request.server_name,
                    self.server_links[prompt_request.server_name]
            ) as session:
                res = await session.session.get_prompt(
                    name=prompt_request.prompt_name,
                    arguments=prompt_request.arguments
                )
            results.append(res)
        return results

    async def save_state(self, save_path: str):
        logger.warning(
            "MCP tool manager does not support state management. "
            "This operation will not cause any changes.")

    async def load_state(self, load_path: str):
        logger.warning(
            "MCP tool manager does not support state management. "
            "This operation will not cause any changes.")

    def set_tool_controller(self, controller: 'ToolController') -> None:
        """Set the tool controller that manages tool activation rules

        Args:
            controller: ToolController instance containing activation/deactivation rules
        """
        self.tool_controller = controller

    def get_tool_controller(self) -> Optional['ToolController']:
        """Get the current tool controller instance

        Returns:
            The current ToolController instance, or None if not configured
        """
        return self.tool_controller

    async def connect(self, server_name: str = None):
        """Establish connections to MCP servers.

        Args:
            server_name: Optional server name to connect to. If None, connects to all servers.

        Raises:
            KeyError: If specified server name is not found
        """
        if server_name and server_name not in self.server_links:
            raise KeyError(f"Cannot connect to unknown server name: {server_name}")

        for _server_name, _server_link in self.server_links.items():
            if server_name is not None and server_name != _server_name:
                continue
            session = self._get_mcp_session(server_name=_server_name, server_link=_server_link)
            if session:
                await session.connect()

    async def disconnect(self, server_name: str = None):
        """Disconnect from MCP servers.

        Args:
            server_name: Optional server name to disconnect from. If None, disconnects from all servers.

        Raises:
            KeyError: If specified server name is not found
        """
        if server_name and server_name not in self.server_links:
            raise KeyError(f"Cannot disconnect from unknown server name: {server_name}")

        for _server_name, _server_link in self.server_links.items():
            if server_name is not None and server_name != _server_name:
                continue
            session = self._get_mcp_session(
                server_name=_server_name,
                server_link=_server_link
            )
            await session.disconnect()

    async def get_mcp_status(self):
        """Get connection status of all managed MCP servers.

        Returns:
            Dictionary mapping server names to connection status (True/False)
        """
        status = {}
        for _server_name, _server_link in self.server_links.items():
            session = self._get_mcp_session(
                server_name=_server_name,
                server_link=_server_link,
                create_new=False,
            )
            status[_server_name] = False if session is None else session.is_connect
        return status

    async def __aenter__(self):
        """Enter the runtime context for this McpToolManager.

        Establishes connections to all configured MCP servers when entering the context.

        Returns:
            The McpToolManager instance itself for use within the context.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context for this McpToolManager.

        Automatically disconnects from all MCP servers when exiting the context.

        Args:
            exc_type: Exception type if an exception occurred, otherwise None
            exc_val: Exception instance if an exception occurred, otherwise None
            exc_tb: Traceback if an exception occurred, otherwise None
        """
        await self.disconnect()

    async def start(self):
        ...

    async def stop(self):
        ...

    async def reset(self):
        ...
