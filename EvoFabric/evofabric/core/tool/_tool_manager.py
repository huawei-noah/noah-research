# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import json
import traceback
from json import JSONDecodeError
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from pydantic import Field, field_serializer, field_validator, PrivateAttr

from ._base_tool import BaseTool
from ._controller import ToolController
from ._tool_manager_base import ToolManagerBase
from ._tool_utils import load_function
from ..graph import stream_writer_env, StreamCtx
from ..typing import ToolCall, ToolCallResult, ToolInnerState
from ...logger import get_logger

logger = get_logger()


class ToolManager(ToolManagerBase):
    tools: List[Union[Callable, Tuple[Callable, ToolInnerState], BaseTool]] = Field(
        default_factory=list,
        description="List of tool sources"
    )

    timeout: Optional[int] = Field(
        default=None,
        description="Timeout in seconds of tool execution"
    )

    tool_controller: Optional[ToolController] = Field(
        default=None,
        description="Use ToolController to manage tool activity status"
    )

    _tool_map: Dict[str, BaseTool] = PrivateAttr(default_factory=dict)

    @field_serializer('tools')
    def serialize_tools(self, tools_value: Any, _info: Any) -> List[BaseTool]:
        return list(self._tool_map.values())

    @field_validator('tools', mode='before')
    @classmethod
    def deserialize_stream_parser(cls, tools: Any) -> List[Union[Callable, Tuple[Callable, ToolInnerState], BaseTool]]:
        deserialized = []
        for tool in tools:
            if isinstance(tool, dict):
                deserialized.append(BaseTool.model_validate(tool))
            elif isinstance(tool, BaseTool):
                deserialized.append(tool)
            elif isinstance(tool, Callable):
                deserialized.append(BaseTool.from_callable(tool))
            elif isinstance(tool, tuple):
                deserialized.append(BaseTool.from_callable(tool[0], inner_state=tool[1]))
            else:
                deserialized.append(tool)
        return deserialized

    def _add_tool_from_python_file(
            self,
            file_path: str,
            include_patterns: List[str] = None,
            exclude_patterns: List[str] = None
    ):
        functions_dict = load_function(
            file_path=file_path, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        for name, fn in functions_dict.items():
            converted_tool = BaseTool.from_callable(fn)
            self._tool_map[name] = converted_tool

    def _add_tool_from_function(
            self,
            function: Callable,
            inner_state: ToolInnerState = None
    ):
        converted_tool = BaseTool.from_callable(
            function,
            inner_state=inner_state
        )
        self._tool_map[converted_tool.name] = converted_tool

    def _add_tool_from_base_tool(
            self,
            base_tool: BaseTool
    ):
        self._tool_map[base_tool.name] = base_tool

    def model_post_init(self, context: Any, /):
        self.add_callable_tools(self.tools)

    async def list_tools(self):
        schemas = []
        for name, v in self._tool_map.items():
            if self.tool_controller and not self.tool_controller.check_tool_status(name):
                continue
            schemas.append(v.get_tool_schema())
        return schemas

    async def call_tools(
            self,
            tasks: List[ToolCall],
    ) -> List[ToolCallResult]:
        async def _run_tool(tool_call: ToolCall):
            if self.tool_controller and not self.tool_controller.check_tool_status(tool_call.function.name):
                raise ValueError(f"Cannot call {tool_call.function.name} because this tool is deactivate now.")
            if tool_call.function.name not in self._tool_map:
                raise ValueError(f"Tool not found: {tool_call.function.name}")

            tool = self._tool_map[tool_call.function.name]

            try:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except JSONDecodeError as e:
                    raise ValueError(
                        f"Tool call arguments parse failed, not a valid JSON: {tool_call.function.arguments}")

                with stream_writer_env(StreamCtx(tool_name=tool_call.function.name, tool_call_id=tool_call.id)):
                    tool_res = await tool(**arguments)

                # output all results from different tool in a unified structure
                return ToolCallResult(
                    tool_call_id=tool_call.id,
                    success=True,
                    content=tool_res
                )

            except Exception as e:
                # Error message when tool call fail
                return ToolCallResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    content=str(e),
                    traceback=traceback.format_exc()
                )

        # parallel run all tools
        coros = [_run_tool(t) for t in tasks]

        if self.timeout:
            results = await asyncio.wait_for(asyncio.gather(*coros), timeout=self.timeout)
        else:
            results = await asyncio.gather(*coros)
        return cast(List[ToolCallResult], results)

    def add_callable_tools(
            self,
            tools: List[Union[Union[Callable, BaseTool], Tuple[Callable, ToolInnerState]]],
    ):
        for tool_pair in tools:
            if not isinstance(tool_pair, (tuple, list)):
                tool_pair = (tool_pair, None)

            tool, inner_state = tool_pair
            if isinstance(tool, Callable) and not isinstance(tool, BaseTool):
                tool = BaseTool.from_callable(
                    tool,
                    inner_state=inner_state
                )
            if not isinstance(tool, BaseTool):
                raise TypeError(
                    f"Tool can only be BaseTool or Callable, got invalid type: {type(tool)}")
            self._tool_map[tool.name] = tool

    def add_python_file_tools(
            self,
            file_paths: List[str],
            include_pattern_list: List[List[str]] = None,
            exclude_pattern_list: List[List[str]] = None
    ):
        for i, file_path in enumerate(file_paths):
            self._add_tool_from_python_file(
                file_path=file_path,
                include_patterns=include_pattern_list[i] if isinstance(
                    include_pattern_list, list) else None,
                exclude_patterns=exclude_pattern_list[i] if isinstance(
                    exclude_pattern_list, list) else None
            )

    def delete_tools(self, tool_names: List[str]):
        for tool_name in tool_names:
            try:
                self._tool_map.pop(tool_name)
            except KeyError as e:
                raise KeyError(
                    f"Tool not found in tool manager {tool_name}") from e

    def update_tools(self, tools: List[BaseTool]):
        """Update the tool instances which already in tool manager."""
        for tool in tools:
            try:
                self._tool_map[tool.name] = tool
            except KeyError as e:
                raise KeyError(
                    f"Tool not found in tool manager {tool.name}") from e

    def find_tools(self, tool_names: List[str]):
        tools_info_list = []
        for tool_name in tool_names:
            try:
                tools_info_list.append(
                    self._tool_map[tool_name].get_tool_schema()
                )
            except KeyError as e:
                raise KeyError(
                    f"Tool not found in tool manager {tool_name}") from e

        return tools_info_list

    async def dump_state(self, tool_name: str = None):
        tool_manager_state = {}
        if tool_name is None:
            for tool_name, tool in self._tool_map.items():
                tool_manager_state[tool_name] = tool.inner_state.model_dump()
        else:
            tool_manager_state[tool_name] = self._tool_map[tool_name].inner_state.model_dump(
            )

        return tool_manager_state

    async def save_state(self, save_path: str):
        tool_manager_state = await self.dump_state()
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(tool_manager_state, f, ensure_ascii=False, indent=4)
        logger.info(f"Save state of tool manager at {save_path}")

    async def load_state(self, load_path: str):
        with open(load_path, 'r', encoding='utf-8') as f:
            loaded_state_dict = json.load(f)

        for tool_name, tool_inner_state in loaded_state_dict.items():
            await self._tool_map[tool_name].load_state(
                input_state=ToolInnerState(
                    state=tool_inner_state.get('state', {}),
                    meta_state=tool_inner_state.get('meta_state', {}) if tool_inner_state.get(
                        'meta_state', {}) is not None else {}
                )
            )
        logger.info("Load Tool Manager state successfully!")

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

    async def start(self):
        pass

    async def stop(self):
        pass

    async def reset(self):
        pass
