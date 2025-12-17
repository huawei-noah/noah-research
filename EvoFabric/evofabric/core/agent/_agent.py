# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import asyncio
import json
from collections import defaultdict
from itertools import chain
from typing import Annotated, Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, model_validator, PrivateAttr

from ..clients import ChatClientBase
from ..factory import FactoryTypeAdapter, safe_get_attr, safe_set_attr
from ..graph import AsyncStreamNode, StreamWriter
from ..mem import MemBase
from ..tool import ToolManagerBase
from ..typing import (
    cast_state_message, LLMChatResponse, State, StateDelta, StateMessage, SystemMessage, ToolCall, ToolCallResult,
    ToolMessage,
    UserMessage
)

ToolManagerType = Union[
    Annotated[ToolManagerBase, FactoryTypeAdapter],
    List[Annotated[ToolManagerBase, FactoryTypeAdapter]]
]

MemoryType = Union[
    Annotated[MemBase, FactoryTypeAdapter],
    List[Annotated[MemBase, FactoryTypeAdapter]]
]


def msg_formatter(format_string: str, **kwargs) -> str:
    """Format msg using jinja2 template, value is extracted from kwargs"""
    from jinja2 import Template

    return Template(format_string).render(**kwargs)


class AgentNode(AsyncStreamNode):
    """
    An agent node.
    """
    client: Annotated[ChatClientBase, FactoryTypeAdapter, Field(description="LLM backend client")]

    system_prompt: str = Field(default="You are a helpful assistant.", description="System prompt of this agent")

    inference_kwargs: Dict = Field(
        default_factory=dict,
        description="Inference kwargs when using client.create_on_stream()")

    tool_manager: ToolManagerType = Field(
        default_factory=list,
        description="Agent tool manager. Can be a ToolManager instance or a list of tools."
    )

    memory: MemoryType = Field(
        default_factory=list,
        description="Memory component. Can be a single instance or a list."
    )

    output_schema: Optional[Type[BaseModel]] = Field(
        default=None,
        description="A LLM response follows the output schema will be a json string "
                    "that can be loaded by `json.loads(msg.content)`"
    )

    output_msg_format: Optional[str] = Field(
        default=None,
        description="""Output message format string
    
    Example:
    Class OutputSchema:
        name: str = "Alice"
        age: int = 32
        
    String: Hello, my name is {{name}}, age is {{age}}.
    
    Result: Hello, my name is Alice, age is 32.
    """)

    input_msg_format: Optional[str] = Field(
        default=None,
        description=""""
        Extract a value from the input state and format it as a string.
        
        Example:
        state = {'messages': [
                {'role': 'user', 'content': 'Hello'}, 
                {'role': 'assistant', 'content': 'Hello, what can I do for you?'}, 
                {'role': 'user', 'content': 'Apply a mailbox for me.'}, 
            ], 
            'user_name': "Alice"
        }
        input_msg_format_string = "Apply a mailbox for {{state['user_name']}}."
        format_result will be:
             Apply a mailbox for Alice.
        
        LLM input message will be:
        [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': 'Apply a mailbox for Alice.'},
        ]
        """
    )

    _stream_writer: StreamWriter = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        if isinstance(self.tool_manager, ToolManagerBase):
            self.tool_manager = [self.tool_manager]
        if isinstance(self.memory, MemBase):
            self.memory = [self.memory]

    async def __call__(self, state: State, stream_writer: StreamWriter) -> StateDelta:
        """Run this node agent."""
        self._stream_writer = stream_writer

        # format input messages by memory modules
        safe_set_attr(
            state,
            "messages",
            await self._update_model_context(safe_get_attr(state, "messages"))
        )

        # format input message by input_msg_format
        safe_set_attr(
            state,
            "messages",
            self._format_input_messages(state)
        )

        # get all tool list
        tool_list = await asyncio.gather(*(tm.list_tools() for tm in self.tool_manager))
        tool_list = list(chain.from_iterable(tool_list))

        # get LLM response
        llm_chat_response = await self._call_llm(
            messages=safe_get_attr(state, "messages"),
            tools=tool_list,
            output_schema=self.output_schema,
            **self.inference_kwargs
        )
        # format output when need
        llm_chat_response = self._format_output_schema(llm_chat_response)

        # convert llm response into assistant message
        assist_msg = cast_state_message(
            dict(
                content=llm_chat_response.content,
                reasoning_content=llm_chat_response.reasoning_content,
                tool_calls=llm_chat_response.tool_calls,
                usage=llm_chat_response.usage,
                role='assistant'
            )
        )

        # execute tools when need
        tool_msgs = await self._execute_tools(llm_chat_response) if llm_chat_response.tool_calls else []

        # concat msgs
        messages = [
            assist_msg,
            *tool_msgs
        ]

        total_messages = safe_get_attr(state, "messages") + messages
        await self._update_model_memory(total_messages)
        return dict(messages=messages)

    @model_validator(mode="after")
    def _check_output_schema_and_format(self):
        """Output_msg_format cannot be none when output_schema is none"""
        if self.output_schema is None and self.output_msg_format is not None:
            raise ValueError(
                "Cannot use `output_msg_format` when `output_schema` is None."
            )
        return self

    def _format_input_messages(self, state: State) -> List[StateMessage]:
        """Format input messages if input_msg_format exist"""
        if self.input_msg_format:
            format_input = msg_formatter(format_string=self.input_msg_format, state=state)
            msgs = [
                UserMessage(content=format_input),
            ]
        else:
            msgs = safe_get_attr(state, "messages")

        msgs = [SystemMessage(content=self.system_prompt)] + msgs
        return msgs

    async def _call_llm(
            self,
            messages: List[StateMessage],
            tools: List,
            output_schema: Type[BaseModel],
            **inference_kwargs
    ) -> LLMChatResponse:
        """Generate LLM response"""
        async for msg in self.client.create_on_stream(  # type: ignore
                messages,
                tools=tools,
                response_format=output_schema,
                **inference_kwargs):
            self._stream_writer.put(msg)
            if isinstance(msg, LLMChatResponse):
                return msg

    def _format_output_schema(self, chat_response: LLMChatResponse) -> LLMChatResponse:
        """Format output msg using output_msg_format"""
        if not self.output_msg_format:
            return chat_response

        chat_response.content = msg_formatter(self.output_msg_format, **json.loads(chat_response.content))
        return chat_response

    async def _execute_tools(self, chat_response: LLMChatResponse) -> List[ToolMessage]:
        """Execute tools using tool_manager"""
        if not chat_response.tool_calls:
            return []
        call_id_to_original_index: dict[str, int] = {
            call.id: i for i, call in enumerate(chat_response.tool_calls)
        }

        all_tool_lists = await asyncio.gather(*[tm.list_tools() for tm in self.tool_manager])

        tool_name_to_tm_idx_map: dict[str, int] = {}
        for idx, tool_list in enumerate(all_tool_lists):
            for tool_spec in tool_list:
                tool_name = tool_spec["function"]["name"]
                tool_name_to_tm_idx_map[tool_name] = idx

        tm_to_tool_calls: dict[int, list] = defaultdict(list)
        not_found_calls: list = []

        for tool_call in chat_response.tool_calls:
            tm_idx = tool_name_to_tm_idx_map.get(tool_call.function.name)
            if tm_idx is not None:
                tm_to_tool_calls[tm_idx].append(tool_call)
            else:
                not_found_calls.append(tool_call)

        tm_to_tool_calls: dict[int, list[ToolCall]] = defaultdict(list)
        not_found_calls: list[ToolCall] = []

        for tool_call in chat_response.tool_calls:
            tm_idx = tool_name_to_tm_idx_map.get(tool_call.function.name)
            if tm_idx is not None:
                tm_to_tool_calls[tm_idx].append(tool_call)
            else:
                not_found_calls.append(tool_call)

        tasks = [
            self.tool_manager[idx].call_tools(calls)
            for idx, calls in tm_to_tool_calls.items()
        ]
        list_of_tool_results = await asyncio.gather(*tasks) if tasks else []

        ordered_results: list[Union[ToolCallResult, None]] = [None] * len(chat_response.tool_calls)

        successful_results = [
            res
            for single_tm_result in list_of_tool_results
            if single_tm_result
            for res in single_tm_result
        ]

        for result in successful_results:
            if result.tool_call_id in call_id_to_original_index:
                original_index = call_id_to_original_index[result.tool_call_id]
                ordered_results[original_index] = result

        for call in not_found_calls:
            original_index = call_id_to_original_index[call.id]
            ordered_results[original_index] = ToolCallResult(
                content="Tool not found", tool_call_id=call.id, success=False
            )

        return [
            ToolMessage(content=r.content, tool_call_id=r.tool_call_id)
            for r in ordered_results if r is not None
        ]

    async def _update_model_context(self, messages: List[StateMessage]) -> List[StateMessage]:
        """Update model context using memory"""
        for mem in self.memory:
            messages = await mem.retrieval_update(messages)
        return messages

    async def _update_model_memory(self, messages: List[StateMessage]):
        for mem in self.memory:
            await mem.add_messages(messages)
        return
