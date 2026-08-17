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

import logging
import json
from abc import ABC
from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator
import asyncio

from deepcraft_core import Memory, Message, ToolCall
from deepcraft_core.message import ROLE_TYPE, Role
from deepcraft_core.tool import ToolCollection, TOOL_CHOICE_TYPE, ToolChoice
from deepcraft_core.memory.memory_retrieval import MemoryRetrievalMixin

from .state import AgentState

logger = logging.getLogger(__name__)

class BaseAgent(BaseModel, MemoryRetrievalMixin, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a one step execution.
    Attributes:
        name (Optional[str]): Unique name of the agent.
        description (Optional[str]): Optional description of the agent.
        systemPrompt (Optional[Union[str, List[dict]]]): System-level instruction prompt.
        llm (Optional[BaseLLM]): Language model instance.
        memory (Memory): Agent's memory store.
        state (AgentState): Current state of the agent.
        availableTools (Optional[ToolCollection]): List of available tools.
        toolChoices (Optional[TOOL_CHOICE_TYPE]): Tool choice strategy.
        toolCalls (Optional[List[ToolCall]]): List of tool calls.
        ToolConfig (Optional[dict]): Other tool configurations or parameters.

    """
    # Core attributes
    name: Optional[str] = Field(None, description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    systemPrompt: Optional[Union[str, List[dict]]] = Field(
        None, description="System-level instruction prompt"
    )

    # Dependencies (Any: allow BaseLLM plus navieflow PooledLLM and other ask()-compatible wrappers)
    llm: Optional[Any] = Field(None, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(default=AgentState.IDLE, description="Current agent state")

    # Tools
    availableTools: Optional[ToolCollection] = Field(None, description="List of available tools")
    toolChoices: Optional[TOOL_CHOICE_TYPE] = Field(ToolChoice.AUTO, description="Tool choice strategy")
    toolCalls: Optional[List[ToolCall]] = Field(None, description="List of tool calls")
    ToolConfig: Optional[dict] = Field(None, description="Other tool's configuration or parameters")
    exec_mode: Optional[str] = Field(None, description="execution mode:single turn or multiturn")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",  # Allow extra fields for flexibility in subclasses
    )

    @model_validator(mode="after")
    def initializeAgent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.llm is None:
            raise ValueError("llm is none!")

        if not isinstance(self.memory, Memory):
            self.memory = Memory()

        return self

    async def run(self, request: Optional[Union[str, List[dict]]] = None) -> str:
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

        result = await self.step()

        # Reset LLM's chunk queue after interrupts
        if self.llm:
            self.llm.chunk_queue = asyncio.Queue()

        return result

    async def step(self) -> str:
        """Execute a single step in the agent's workflow.
        Must be implemented by subclasses to define specific behavior.
        """
        response = await self.llm.ask(
            messages=self.memory.messages,
            system_msgs=[Message.system_message(self.systemPrompt)]
        )
        self.updateMemory(Role.ASSISTANT, response)
        return response

    def updateMemory(
        self,
        role: ROLE_TYPE,
        content: Union[str, Message],  
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content (str) or a pre-constructed Message object.
            extra_info: The extra information.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        if isinstance(content, Message):
            msg = content
        else:
            messageMap = {
                Role.USER: Message.user_message,
                Role.SYSTEM: Message.system_message,
                Role.ASSISTANT: Message.assistant_message,
                Role.TOOL: lambda content, **kw: Message.tool_message(content, **kw),
            }

            if role not in messageMap:
                raise ValueError(f"Unsupported message role: {role}")

            msgFactory = messageMap[role]
            msg = msgFactory(content, **kwargs) if role == Role.TOOL else msgFactory(content)
        
        if extra_info:
            self.memory.add_message(msg, extra_info=extra_info)
        else:
            self.memory.add_message(msg)

    async def executeTool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling.
        Args:
        command (ToolCall): The tool call to execute.

        Returns:
            str: A formatted result or error message from the tool execution.
        """
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.availableTools.tool_map: #todo
            return f"Error: Unknown tool '{name}'"
        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            # Extra config for tools
            args["config"] = self.ToolConfig

            # Execute the tool
            logger.info(f"�� Activating tool: '{name}'...")
            result = await self.availableTools.execute(name=name, tool_input=args)

            # Format result for display
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            # Handle special tools like `finish`
            await self._handleSpecialTool(name=name, result=result)
            return observation

        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"�� Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"

        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    async def _handleSpecialTool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes.
             Args:
                name (str): The name of the tool that was executed.
                result (Any): The result of the tool execution.
                **kwargs: Additional keyword arguments from the tool execution.
        """
        if not self._isSpecialTool(name):
            return

        if self._shouldFinishExecution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"�� Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _shouldFinishExecution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _isSpecialTool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory.
        Returns:
            List[Message]: A list of messages stored in the agent's memory.
        """
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value