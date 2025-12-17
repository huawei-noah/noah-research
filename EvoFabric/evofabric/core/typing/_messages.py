# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import uuid
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import Annotated


class ChatUsage(BaseModel):
    """This class defines the usage information of LLM chat client"""

    completion_tokens: Optional[int] = None
    """Number of tokens in the generated completion."""

    prompt_tokens: Optional[int] = None
    """Number of tokens in the prompt."""

    total_tokens: Optional[int] = None
    """Total number of tokens used in the request (prompt + completion)."""

    generation_time: Optional[float] = Field(default=None)
    """Generation time in seconds."""


class EmbedUsage(BaseModel):
    """This class defines the usage information of embedding"""

    generation_time: int
    """Generation time in seconds."""


class RerankUsage(BaseModel):
    """This class defines the usage information of reranking"""

    generation_time: int
    """Generation time in seconds."""


class Function(BaseModel):
    arguments: str
    """function arguments in JSON format"""

    name: str
    """function name"""


class ToolCall(BaseModel):
    """Tool call definition"""

    id: str
    """function call id"""

    function: Function
    """The function that the model called."""

    type: Literal['function'] = 'function'


class ChatStreamChunk(BaseModel):
    """This class defines the stream chunk of an LLM chat client"""

    reasoning_content: Optional[str] = Field(default=None)
    """reasoning content"""

    content: Optional[str] = Field(default=None)
    """content"""


class LLMChatResponse(BaseModel):
    """This class defines the response format of LLMClient."""

    content: str
    """response content"""

    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    """tool calls"""

    reasoning_content: Optional[str] = Field(default=None)
    """The reasoning content of the response."""

    usage: Optional[ChatUsage] = Field(default=None)
    """Usage information"""

    id: str = Field(default_factory=str)
    """unique id of a chat response"""

    meta: dict = Field(default_factory=dict)
    """meta information"""


class EmbedResponse(BaseModel):
    """This class defines the response format of embedding client."""

    embeddings: List[float]
    """The embedding vector, which is a list of floats."""

    usage: Optional[EmbedUsage] = Field(default=None)
    """Usage information"""


class RerankResponse(BaseModel):
    """This class defines the response format of reranking client."""

    scores: List[float]
    """The scores of the reranking"""

    texts: List[str]
    """The texts of the reranking"""

    usage: Optional[RerankUsage] = Field(default=None)
    """Usage information"""


class StateBaseMessage(BaseModel):
    """Messages in State"""
    content: Any

    node_name: Optional[str] = None
    """Node name of this message"""

    msg_id: Optional[str] = None
    """Will be automatically added by append_message strategy"""


class SystemMessage(StateBaseMessage):
    role: Literal['system'] = 'system'


class UserMessage(StateBaseMessage):
    role: Literal['user'] = 'user'


class AssistantMessage(StateBaseMessage):
    role: Literal['assistant'] = 'assistant'

    reasoning_content: Optional[str] = None

    tool_calls: Optional[List[ToolCall]] = None

    usage: Optional[ChatUsage] = None


class ToolMessage(StateBaseMessage):
    tool_call_id: str
    """tool call id"""

    role: Literal['tool'] = 'tool'


class ToolCallResult(BaseModel):
    tool_call_id: str
    """tool call id"""

    success: bool
    """flag of tool returning result as expected."""

    content: Any
    """tool returning specific information. expected result if success, or short error information if failed."""

    traceback: Optional[str] = None
    """complete error traceback"""


StateMessage = Union[
    UserMessage,
    ToolMessage,
    AssistantMessage,
    SystemMessage,
    StateBaseMessage
]


def cast_state_message(msg) -> StateMessage:
    msg = TypeAdapter(
        Annotated[
            ToolMessage |
            AssistantMessage |
            UserMessage |
            SystemMessage,
            Field(discriminator="role")
        ]
    ).validate_python(msg)

    if not msg.msg_id:
        msg.msg_id = str(uuid.uuid4())
    return msg
