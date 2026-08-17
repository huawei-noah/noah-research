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

from enum import Enum
from typing import Any, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from ..tool import ToolCall

class Role(str, Enum):
    """Message role options"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore

class Message(BaseModel):
    """Represents a chat message in the conversation.
    Attributes:
        role (ROLE_TYPE): The role of the message sender (e.g., user, agent).
        content (Optional[Union[str, List[dict]]]): The text content of the message (or image path).
        tool_calls (Optional[List[ToolCall]]): A list of tool calls associated with the message.
        name (Optional[str]): The name associated with the message.
        tool_call_id (Optional[str]): The ID of the tool call.
        reasoning_content (Optional[str]): Assistant-only; chain-of-thought for APIs that require
            round-tripping (e.g. DeepSeek reasoning / thinking mode).
    """

    role: ROLE_TYPE = Field(...)
    content: Optional[Union[str, List[dict]]] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    reasoning_content: Optional[str] = Field(default=None)

    def __add__(self, other) -> List["Message"]:
        "Supports operations between Message and list or between two Message objects."
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        "Supports operations between a list and a Message object."
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.model_dump() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        return message

    @classmethod
    def user_message(cls, content: Union[str, List[dict]]) -> "Message":
        """
        Creates a user message with the specified content.

        Args: content (str, List[dict]): The content of the user message, string for unimodal Model (text-based) and
        List[dict] for Multimodal Model.

        Returns:
            Message: A Message object with the role set to "user" and the specified content.
        """
        return cls(role="user", content=content)

    @classmethod
    def system_message(cls, content: Union[str, List[dict]]) -> "Message":
        """
        Creates a system message with the specified content.

        Args:
            content (str): The content of the system message.

        Returns:
            Message: A Message object with the role set to "system" and the specified content.
        """
        return cls(role="system", content=content)

    @classmethod
    def assistant_message(
        cls,
        content: Optional[Union[str, List[dict]]] = None,
        *,
        reasoning_content: Optional[str] = None,
    ) -> "Message":
        """
        Creates a assistant message with the specified content.

        Args:
            content (str): The content of the assistant message.
            reasoning_content: Optional reasoning/thinking text for providers that require echo-back.

        Returns:
            Message: A Message object with the role set to "assistant" and the specified content.
        """
        return cls(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
        )

    @classmethod
    def tool_message(cls, content: str, name, tool_call_id: str) -> "Message":
        """
        Creates a tool message with the specified content.

        Args:
            content (str): The content of the tool message.

        Returns:
            Message: A Message object with the role set to "tool" and the specified content.
        """
        return cls(role="tool", content=content, name=name, tool_call_id=tool_call_id)

    @classmethod
    def from_tool_calls(
        cls, tool_calls: List[Any], content: Union[str, List[str]] = "", **kwargs
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role="assistant", content=content, tool_calls=formatted_calls, **kwargs
        )
