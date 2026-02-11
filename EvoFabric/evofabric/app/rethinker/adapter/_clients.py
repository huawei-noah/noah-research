# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import re
from functools import partial
from typing import AsyncIterator, Callable, List, Union

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from pydantic import ValidationError

from evofabric.app.rethinker._config import config, LLMConfig
from evofabric.core.clients import OpenAIChatClient
from evofabric.core.typing import (
    AssistantMessage, cast_state_message, ChatStreamChunk, ChatUsage, LLMChatResponse, StateMessage,
    SystemMessage, ToolMessage, UserMessage
)
from evofabric.logger import get_logger

logger = get_logger()


async def _parse_openai_stream_response_with_stop_condition(
    response: AsyncStream[ChatCompletionChunk],
    stop_condition: Callable[[str], bool]
) -> AsyncIterator[Union[ChatStreamChunk, LLMChatResponse]]:
    """
    Parse OpenAI chat completion streaming responses with an explicit
    stop condition.

    This coroutine consumes ChatCompletionChunk objects from an async
    stream, incrementally aggregates content and reasoning_content,
    yields intermediate ChatStreamChunk updates, and produces a final
    LLMChatResponse either when the stop condition is triggered or when
    the stream is exhausted.
    """
    content = ""
    reasoning_content = ""
    usage = None
    cumulative_log_prob = 0.0
    token_logprobs_data = []

    async for chunk in response:
        if chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta
            if _content := getattr(delta, "content", None):
                content += _content

            if _reasoning_content := getattr(delta, "reasoning_content", None):
                reasoning_content += _reasoning_content

            if _reasoning_content or _content:
                yield ChatStreamChunk(
                    reasoning_content=_reasoning_content if _reasoning_content else None,
                    content=_content if _content else None,
                )

            # get logprob values
            if hasattr(choice, "logprobs") and choice.logprobs:
                if choice.logprobs.content:
                    for lp in choice.logprobs.content:
                        cumulative_log_prob += lp.logprob

                        token_logprobs_data.append({
                            "token": lp.token,
                            "logprob": lp.logprob,
                        })

        if _usage := getattr(chunk, "usage", None):
            usage = ChatUsage(
                completion_tokens=_usage.completion_tokens,
                prompt_tokens=_usage.prompt_tokens,
                total_tokens=_usage.total_tokens,
            )

        if stop_condition(content):
            yield LLMChatResponse(
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=None,
                usage=usage,
                meta={"stop_reason": "code"},
            )
            return

    yield LLMChatResponse(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=None,
        usage=usage,
        meta={
            "stop_reason": "finished",
            "answer_log_prob": cumulative_log_prob,
            "token_logprobs": token_logprobs_data
        }
    )


def generate_stop_condition(pattern: str):
    def streaming_code_stop_condition(content: str):
        matches = list(re.finditer(pattern, content, re.DOTALL))
        return len(matches) > 0

    return streaming_code_stop_condition


class FastSlowThinkOpenAIChatClient(OpenAIChatClient):

    fast_think: bool = False

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

        if self.fast_think:
            transformed[-1]["content"] += "/no_think"
        return transformed


def get_client(model: str):
    """Get an OpenAI chat client for the specified model."""
    llm_configs = config.llm_resources

    if model not in llm_configs:
        raise ValueError(
            f"Unknown model '{model}'. Available models are: {list(llm_configs.keys())}"
        )

    llm_config: LLMConfig = llm_configs[model]

    # Prepare common keyword arguments for the client
    client_init_kwargs = {
        "model": llm_config.model_name,
        "stream": llm_config.stream,
        "client_kwargs": llm_config.create_client_kwargs(),
        "inference_kwargs": llm_config.create_inference_kwargs(),
        "http_client_kwargs": llm_config.http_client_kwargs,
        "fast_think": llm_config.fast_think,
    }

    # Add stream_parser only if a stop condition is defined
    if llm_config.stop_condition:
        client_init_kwargs["stream_parser"] = partial(
            _parse_openai_stream_response_with_stop_condition,
            stop_condition=generate_stop_condition(llm_config.stop_condition)
        )

    return FastSlowThinkOpenAIChatClient(**client_init_kwargs)
