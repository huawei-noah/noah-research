# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import inspect
from typing import Any, Callable, List, Optional

from pydantic import Field, field_serializer, field_validator, PrivateAttr

from ._tool_utils import parse_callable_schema
from ..factory import BaseComponent, get_func_serializer
from ..graph import get_stream_writer
from ..typing import TOOL_EXCLUDE_PRESERVED_PARAMS, ToolInnerState


class BaseTool(BaseComponent):
    func: Callable = Field(
        description="Function to be called, can be sync or async function"
    )

    exclude_params: List[str] = Field(
        default_factory=list,
        description="Parameters name will not be listed in tool schema"
    )

    inner_state: Optional[ToolInnerState] = Field(
        default=None,
        description=(
            "The internal state of the tool. If the tool's input parameters define "
            "`inner_state: ToolInnerState`, this parameter will be passed to the tool as an input argument."
            " The tool can read and modify this internal state to maintain its own internal status."
        )
    )

    name: Optional[str] = Field(
        default=None,
        description="Name of the tool, if none, will use function name"
    )

    description: Optional[str] = Field(
        default=None,
        description="Description of the tool, if none, will use function comment"
    )

    include_long_description: bool = Field(
        default=True,
        description="Include long description of the tool"
    )

    include_var_positional: bool = Field(
        default=True,
        description="Include positional arguments of the tool like *args"
    )

    include_var_keyword: bool = Field(
        default=True,
        description="Include keyword arguments of the tool like **kwargs"
    )

    _tool_schema: dict = PrivateAttr(default_factory=dict)

    _exist_exclude_params: List[str] = PrivateAttr(default_factory=list)
    """The actual excluded parameters of the function."""

    @field_serializer("func")
    def serialize_state_filter(self, _value: Callable) -> str:
        return get_func_serializer().serialize(_value)

    @field_validator('func', mode='before')
    @classmethod
    def deserialize_state_filter(cls, v: Any) -> Callable:
        if callable(v):
            return v
        return get_func_serializer().deserialize(v)

    @classmethod
    def from_callable(
            cls,
            func: Callable,
            *,
            name: str = None,
            description: str = None,
            inner_state: Optional[ToolInnerState] = None,
            exclude_params: List[str] = None,
    ):
        exclude_params = exclude_params or []
        if isinstance(func, BaseTool):
            return func

        return BaseTool(
            func=func,
            name=name,
            description=description,
            inner_state=inner_state,
            exclude_params=exclude_params
        )

    async def __call__(self, **kwargs) -> Any:
        """Call this tool using given kwargs."""
        kw = kwargs
        if "inner_state" in self._exist_exclude_params:
            kw["inner_state"] = self.inner_state
        if "stream_writer" in self._exist_exclude_params:
            kw["stream_writer"] = get_stream_writer()

        # handle positional-only param
        pos_args = []
        sig = inspect.signature(self.func)
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                if name in kw:
                    pos_args.append(kw.pop(name))
                elif param.default is inspect.Parameter.empty:
                    raise TypeError(f"Missing parameter {name} for function {self.name}")

        if inspect.iscoroutinefunction(self.func):
            return await self.func(*pos_args, **kw)

        return self.func(*pos_args, **kw)

    def model_post_init(self, context: Any, /) -> None:
        self.name = self.name or self.func.__name__
        self.description = self.description or self.func.__doc__
        self.exclude_params.extend(TOOL_EXCLUDE_PRESERVED_PARAMS)

        self._tool_schema, self._exist_exclude_params = parse_callable_schema(
            function=self.func,
            exclude_params=self.exclude_params,
            name=self.name,
            description=self.description,
            include_long_description=self.include_long_description,
            include_var_positional=self.include_var_positional,
            include_var_keyword=self.include_var_keyword
        )

    def get_tool_schema(self) -> dict:
        return self._tool_schema

    async def dump_state(self) -> ToolInnerState:
        """Dump tool inner states."""
        return self.inner_state

    async def load_state(self, input_state: ToolInnerState):
        """Load tool inner states."""
        self.inner_state = input_state
