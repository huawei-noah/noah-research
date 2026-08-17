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

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Dict
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

class ToolChoice(str, Enum):
    """Tool choice options.

    Attributes:
        NONE (str): No tool will be used.
        AUTO (str): The model will automatically choose which tools to use.
        REQUIRED (str): Tools must be used.
        """
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE   = Literal[TOOL_CHOICE_VALUES]  # type: ignore

class Function(BaseModel):
    """Tool function.
    Attributes:
        name (str): The name of the function.
        arguments (str): The arguments to pass to the function.
        """
    name: str
    arguments: str

class ToolCall(BaseModel):
    """Represents a tool/function call in a message.
    Attributes:
        id (str): A unique identifier for the tool call.
        type (str): The type of the tool call, defaults to "function".
        function (Function): The function being called.
        """

    id: str
    type: str = "function"
    function: Function

class BaseTool(ABC, BaseModel):
    """
    Abstract base class representing a tool.

    Attributes:
        name (str): The name of the tool.
        description (str): A description of the tool's functionality.
        parameters (Optional[dict]): Optional parameters for the tool. Defaults to None.
    """
    name: str
    description: str
    parameters: Optional[dict] = {}
    func_signature: Optional[str] = "()"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @staticmethod
    def codeact_func(**kwargs) -> Any:
        ...

class ToolResult(BaseModel):
    """Represents the result of a tool execution.
    Attributes:
        output (Any): The output of the tool execution. Defaults to None.
        error (Optional[str]): An error message if the tool execution failed. Defaults to None.
        system (Optional[str]): System-related information from the tool execution. Defaults to None.
        """

    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __bool__(self):
        return any(getattr(self, field) for field in self.model_fields)

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        if self.error:
            if self.output:
                return f"Error: {self.error}\n{self.output}"
            return f"Error: {self.error}"
        return self.output

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        return self.model_copy(update=kwargs)


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""

class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message