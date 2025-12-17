# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import time
from typing import Any, AsyncGenerator, Callable, cast, Dict, List, Optional, Sequence

import httpx
import openai
from openai import AsyncStream
from openai.lib._pydantic import to_strict_json_schema
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import Field, field_serializer, field_validator, TypeAdapter, ValidationError

from ._base import ChatClientBase
from ..factory import get_func_serializer, is_basemodel, is_dataclass
from ..trace import trace_chat
from ..typing import (
    AssistantMessage, cast_state_message, ChatStreamChunk, ChatUsage, Function, LLMChatResponse, StateMessage,
    SystemMessage, ToolCall,
    ToolMessage, UserMessage
)
from ...logger import get_logger

logger = get_logger()


def type_to_response_format_param(
        response_format: type,
) -> ResponseFormat:
    """Convert response_format into openai format"""
    if isinstance(response_format, dict):
        return response_format
    response_format = cast(type, response_format)

    if is_basemodel(response_format):
        name = response_format.__name__
        json_schema_type = response_format
    elif is_dataclass(response_format):
        name = response_format.__name__
        json_schema_type = TypeAdapter(response_format)
    else:
        raise TypeError(f"Unsupported response_format type - {response_format}")

    return {
        "type": "json_schema",
        "json_schema": {
            "schema": to_strict_json_schema(json_schema_type),
            "name": name,
            "strict": True,
        },
    }


def _format_tool_call(tool_calls: Optional[List]) -> Optional[List[ToolCall]]:
    """Convert tool_calls in openai response into ToolCall"""
    if not tool_calls:
        return None
    formatted_tool_calls = []
    for tool_call in tool_calls:
        formatted_tool_calls.append(
            ToolCall(
                id=tool_call["id"],
                function=Function(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
            )
        )
    return formatted_tool_calls


async def _parse_openai_stream_response(
        response: AsyncStream[ChatCompletionChunk]
) -> AsyncGenerator[ChatStreamChunk, LLMChatResponse]:
    """Parse openai stream response"""
    content = ""
    reasoning_content = ""
    function_call = {}
    usage = None
    async for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if _content := getattr(delta, "content", None):
                content += _content
            if _reasoning_content := getattr(delta, "reasoning_content", None):
                reasoning_content += _reasoning_content
            if _tool_call := getattr(delta, "tool_calls", None):
                for tc in _tool_call:
                    idx = tc.index
                    call_id = tc.id or function_call.get(idx, {}).get("id", "")
                    function_call.setdefault(idx, {"id": call_id, "name": "", "arguments": ""})
                    if tc.function:
                        if tc.function.name:
                            function_call[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            function_call[idx]["arguments"] += tc.function.arguments
            if _reasoning_content or _content:
                # yield stream only when there is something to show...
                yield ChatStreamChunk(
                    reasoning_content=_reasoning_content if _reasoning_content else None,
                    content=_content if _content else None,
                )
        if _usage := getattr(chunk, "usage", None):
            usage = ChatUsage(
                completion_tokens=_usage.completion_tokens,
                prompt_tokens=_usage.prompt_tokens,
                total_tokens=_usage.total_tokens,
            )

    tool_calls = [
        ToolCall(
            id=v["id"],
            function=Function(
                name=v["name"],
                arguments=v["arguments"],
            )
        )
        for v in function_call.values()
        if v["id"]
    ]

    yield LLMChatResponse(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls if tool_calls else None,
        usage=usage,
    )


class OpenAIChatClient(ChatClientBase):
    """
    An Openai chat client.
    """
    model: str = Field(description="LLM model name")

    stream: bool = Field(default=True, description="Using stream mode")

    client_kwargs: Dict = Field(
        default_factory=dict,
        description="Fields for initialize openai async client")

    http_client_kwargs: Dict = Field(
        default_factory=dict,
        description="Fields for initialize http_client in openai async client")

    inference_kwargs: Dict = Field(
        default_factory=dict,
        description="Fields for chat.completions.create(), such as temperature, top_p")

    stream_parser: Callable = Field(
        default=_parse_openai_stream_response,
        description="An async function that can iterate process streaming chunks returned by openai"
    )

    @field_serializer("stream_parser")
    def serialize_stream_parser(self, _value: Callable) -> str:
        return get_func_serializer().serialize(_value)

    @field_validator('stream_parser', mode='before')
    @classmethod
    def deserialize_stream_parser(cls, v: Any) -> Callable:
        if callable(v):
            return v
        return get_func_serializer().deserialize(v)

    @staticmethod
    def _state_msgs_to_input_msgs(messages: List[StateMessage]) -> List:
        """Convert state msg list into openai format"""
        transformed = []
        for msg in messages:
            try:
                msg = cast_state_message(msg) if isinstance(msg, dict) else msg
            except ValidationError as e:
                logger.warning(f"Cannot cast message to StateMessage, skipping. "
                               f"Type: {type(msg)}, value: {msg}, error: {e}")
                continue

            if isinstance(msg, UserMessage):
                transformed.append({
                    'role': 'user',
                    'content': msg.content
                })
            elif isinstance(msg, SystemMessage):
                transformed.append({
                    'role': 'system',
                    'content': msg.content
                })
            elif isinstance(msg, AssistantMessage):
                _msg = {
                    'role': 'assistant',
                    'content': msg.content,
                }
                if msg.reasoning_content:
                    _msg['reasoning_content'] = msg.reasoning_content
                if msg.tool_calls:
                    _msg['tool_calls'] = [x.model_dump() for x in msg.tool_calls]
                transformed.append(_msg)
            elif isinstance(msg, ToolMessage):
                transformed.append({
                    'role': 'tool',
                    'tool_call_id': msg.tool_call_id,
                    'content': repr(msg.content)
                })
            else:
                logger.warning(f"Skip unrecognized message type: {type(msg)}, value: {msg}")
        return transformed

    @trace_chat
    async def create_on_stream(
            self,
            messages: List[StateMessage],
            tools: Optional[Sequence] = None,
            response_format: Optional = None,
            **kwargs
    ) -> AsyncGenerator[ChatStreamChunk, LLMChatResponse]:
        """Generate LLM response using streaming mode"""
        tools = tools or []
        client = self._get_client()
        infer_kwargs = self._format_create_kwargs(
            messages,
            tools=tools,
            response_format=response_format,
            **kwargs
        )
        time_st = time.time()
        response = await client.chat.completions.create(**infer_kwargs)

        if self.stream:
            # stream mode: yield stream chunk and the final chat response
            async for msg in self.stream_parser(response):
                if isinstance(msg, LLMChatResponse):
                    msg: LLMChatResponse
                    time_ed = time.time()
                    if msg.usage:
                        msg.usage.generation_time = time_ed - time_st
                yield msg
            return

        # non-stream mode: only yield final chat response
        time_ed = time.time()
        yield LLMChatResponse(
            content=response.choices[0].message.content,
            reasoning_content=getattr(response.choices[0].message, "reasoning_content", None),
            tool_calls=_format_tool_call(response.choices[0].message.tool_calls),
            usage=ChatUsage(
                completion_tokens=response.usage.completion_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
                generation_time=time_ed - time_st,
            )
        )

    def _get_client(self):
        """Initialize openai client"""
        return openai.AsyncClient(
            http_client=httpx.AsyncClient(**self.http_client_kwargs),
            **self.client_kwargs
        )

    def _format_create_kwargs(
            self,
            messages: List[StateMessage],
            tools: Optional[Sequence] = None,
            response_format: Optional = None,
            **kwargs) -> Dict:
        """Format input kwargs of openai_client.chat.completions.create()"""
        input_kwargs = {
            "model": self.model,
            "messages": self._state_msgs_to_input_msgs(messages),
            "stream": self.stream,
        }
        if tools:
            if self.inference_kwargs.get("tools", None):
                logger.warning("This client already has tools, which will be overwritten.")
            input_kwargs["tools"] = tools

        if response_format:
            input_kwargs["response_format"] = type_to_response_format_param(response_format)

        if self.stream:
            input_kwargs["stream_options"] = {"include_usage": True}
        input_kwargs.update(self.inference_kwargs)
        input_kwargs.update(kwargs)
        return input_kwargs
