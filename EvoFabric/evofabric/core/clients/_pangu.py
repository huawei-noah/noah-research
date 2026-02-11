# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from functools import partial
from typing import Any, AsyncIterator, Callable, List, Union

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from pydantic import Field, ValidationError

from ._openai import OpenAIChatClient
from ..typing import (
    AssistantMessage, cast_state_message, ChatStreamChunk, ChatUsage, Function, LLMChatResponse,
    StateMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
)
from ...logger import get_logger

logger = get_logger()

THINK_START_PATTERN = "[unused16]"
THINK_END_PATTERN = "[unused17]"


async def _parse_pangu_stream_response(
        response: AsyncStream[ChatCompletionChunk],
        enable_think: bool = True,
) -> AsyncGenerator[ChatStreamChunk, LLMChatResponse]:
    """Parse openai stream response"""
    content = ""
    reasoning_content = ""
    function_call = {}
    end_think = not enable_think
    usage = None
    async for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if _content := getattr(delta, "content", None):
                if THINK_END_PATTERN in _content:
                    end_think = True
                    _reasoning_content, _content = _content.split(THINK_END_PATTERN)

                    reasoning_content += _reasoning_content
                    content += _content

                if not end_think:
                    if THINK_START_PATTERN in _content:
                        _, _tmp_reasoning_content = _content.split(THINK_START_PATTERN)
                        reasoning_content += _tmp_reasoning_content
                    reasoning_content += _content
                else:
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


class PanguClient(OpenAIChatClient):
    stream_parser: Callable = Field(
        default=_parse_pangu_stream_response,
        description="An async function that can iterate process streaming chunks returned by openai"
    )

    enable_think: bool = Field(
        default=True,
        description="Whether to enable think mode"
    )

    def model_post_init(self, context: Any, /) -> None:
        self.stream_parser = partial(self.stream_parser, enable_think=self.enable_think)

    def _state_msgs_to_input_msgs(self, messages: List[StateMessage]) -> List:
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
                content = msg.content
                if not self.enable_think:
                    content += "/no_think"
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
