# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from mcp import StdioServerParameters
from pydantic import BaseModel, Field
from typing_extensions import Annotated

TOOL_EXCLUDE_PRESERVED_PARAMS = ["inner_state", "stream_writer"]


class CodeExecDockerConfig(BaseModel):
    """Code sandbox initialize config"""

    image: str = "python:3-slim"

    auto_remove: bool = True

    working_dir: str = "/tmp"

    tty: bool = True

    detach: bool = True

    mem_limit: str = "4096m"

    cpu_quota: int = 50000

    entrypoint: str = "/bin/sh"

    command: Optional[Union[str, List[str]]] = None

    name: str = "evofabric_sandbox"

    network: str = "host"

    volumes: Optional[dict] = None


class PromptRequest(BaseModel):
    """PromptRequest is a class for prompt request."""
    server_name: str
    """server name"""

    prompt_name: str
    """prompt name"""

    arguments: Dict[str, str]


class ResourceRequest(BaseModel):
    """ResourceRequest is a class for resource request."""
    server_name: str
    """server name"""

    url: str
    """url"""


class StdioLink(StdioServerParameters):
    """Stdio link type config for MCP sever"""
    type: Literal["StdioLink"] = "StdioLink"

    read_time_out: float = 10.0


class SseLink(BaseModel):
    """SseLink type config for MCP sever"""
    type: Literal["SseLink"] = "SseLink"

    url: str

    headers: Optional[Dict[str, Any]] = None

    timeout: float = 30.0

    sse_read_timeout: float = 300.0


class StreamableHttpLink(BaseModel):
    """StreamableHttpLink type config for MCP sever"""
    type: Literal["StreamableHttpLink"] = "StreamableHttpLink"

    url: str

    headers: Optional[Dict[str, Any]] = None

    timeout: float = 30.0

    sse_read_timeout: float = 300.0

    terminate_on_close: bool = True


McpServerLink = Annotated[
    Union[StdioLink, SseLink, StreamableHttpLink],
    Field(discriminator="type")
]


class MCPConfig(BaseModel):
    """
    MCPConfig is a configuration class for MCP.
    """
    url: str
    """url for sse/http transportation. When stdio transportation used, it can be filled 
    with absolute path of mcp server."""


class ToolInnerState(BaseModel):
    """
    ToolInnerState is a class for tool inner state.
    """
    type: Literal["ToolInnerState"] = "ToolInnerState"
    """tool inner state type """

    state: Optional[Dict[str, Any]] = None
    """tool state. {state_name: state_content}"""

    meta_state: Optional[Dict[str, Any]] = None
    """tool meta state. {state_name: state_content}"""


class ToolManagerState(BaseModel):
    """
    ToolManagerState is a class for tool manager state.
    """
    type: Literal["ToolManagerState"] = "ToolManagerState"
    """state type """

    state: Optional[Dict[str, ToolInnerState]] = None
    """tool manager state. key is tool name, value is its ToolInnerState."""


class ActiveToolPattern(BaseModel):
    mode: Literal['activate'] = 'activate'

    pattern: str


class DeactivateToolPattern(BaseModel):
    mode: Literal['deactivate'] = 'deactivate'

    pattern: str


ToolControlPattern = Annotated[
    Union[
        ActiveToolPattern,
        DeactivateToolPattern
    ],
    Field(discriminator="mode")
]

