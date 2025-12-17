# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from ..factory import BaseComponent

if TYPE_CHECKING:
    from ._controller import ToolController
    from ..typing import ToolCall


class ToolManagerBase(ABC, BaseComponent):
    @abstractmethod
    def list_tools(self, **kwargs):
        """Return a list of tool schemas in OpenAI function format

        Returns:
            List[Dict]: List of tool definitions compatible with OpenAI's function calling format
        """
        ...

    @abstractmethod
    async def call_tools(self, tasks: List['ToolCall']):
        """Execute tool calls with provided tasks

        Args:
            tasks: List of tool call tasks containing tool name and arguments

        Returns:
            Async iterator yielding tool call results
        """
        ...

    @abstractmethod
    def set_tool_controller(self, controller: 'ToolController') -> None:
        """Set the tool controller that manages tool activation rules

        Args:
            controller: ToolController instance containing activation/deactivation rules
        """
        ...

    @abstractmethod
    def get_tool_controller(self) -> Optional['ToolController']:
        """Get the current tool controller instance

        Returns:
            The current ToolController instance, or None if not configured
        """
        ...

    @abstractmethod
    async def start(self):
        ...

    @abstractmethod
    async def stop(self):
        ...

    @abstractmethod
    async def reset(self):
        ...

    @abstractmethod
    async def save_state(self, save_path: str):
        ...

    @abstractmethod
    async def load_state(self, load_path: str):
        ...
