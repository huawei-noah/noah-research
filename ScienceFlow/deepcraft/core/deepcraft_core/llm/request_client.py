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
from typing import List, Optional, Union, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pydantic import Field, model_validator

import requests
import json

from .base import BaseLLM
from .utils import AsyncLineIterator, openapi2pangu, response2tool_call, ASK_TOOL_PROMPT, count_chars
from ..message import Message
from ..tool import TOOL_CHOICE_VALUES, TOOL_CHOICE_TYPE

logger = logging.getLogger(__name__)


class RequestClient(BaseLLM):
    """
    Enhanced Request client with SSL verification configuration support
    ssl_verify： SSL verification
    """
    ssl_verify: bool = Field(default=False)

    @model_validator(mode="after")
    def init_requestclient(self) -> "RequestClient":
        """Initialize the RequestClient instance.
        Returns:
            RequestClient: The initialized RequestClient instance.
        """
        self = super().initialize()

        return self

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Execute an LLM request with optional streaming response.

        Args:
            messages: List of messages, supporting dictionaries or objects with a to_dict() method.
            additional_headers: Additional headers to override {'Content-Type': 'application/json'}
            system_msgs: Optional system messages to prepend
            stream: Whether to use streaming response mode. Defaults to True.
            temperature (float): Sampling temperature for the response.
            top_p (float): The top-p parameter for controlling the diversity of the output.

        Returns:
            str:  The generated response

        Raises:
            Exception: If the LLM request fails.
        """
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            return await self._stream_request(messages, stream, temperature, top_p)

        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise ValueError("Error in calling LLM")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 60,
        tools: Optional[List[Dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = "auto",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: Optional[bool] = True,
        parallel_tool_calls: bool = False,
        **kwargs
    ) -> Any:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of messages, supporting dictionaries or objects with a to_dict() method.
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds, defaults to 60.
            tools: Optional list of tools.
            tool_choice: Tool selection strategy, defaults to "auto".
            temperature: Sampling temperature for the response
            top_p (float): The top-p parameter for controlling the diversity of the output.
            stream (bool): The flag for streamlize, which by default is False
            parallel_tool_calls: Whether to call tools in parallel, defaults to False.

        Returns:
            Any: Response from the tool call.

        Raises:
            Exception: If the tool call fails.
        """

        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError(
                            "Each tool must be a dict with 'type' field")

            # Construct prompt words to guide LLM to decide whether to call the tool
            tool_descriptions = openapi2pangu(tools)

            tool_message = ASK_TOOL_PROMPT.format(
                tool_descriptions=tool_descriptions,
            )
            tool_system_message = Message.system_message(tool_message)

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(
                    system_msgs) + self.format_messages([tool_system_message])
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(
                    [tool_system_message]) + self.format_messages(messages)

            content = await self._stream_request(
                messages, stream, temperature, top_p)
            response = response2tool_call(
                content, parallel_tool_calls=parallel_tool_calls)

            return response

        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise

    async def _stream_request(self, messages, stream, temperature, top_p):
        self.interrupted = False # Reset interrupt status
        response = requests.post(url=self.base_url,
                                 stream=stream and self.stream,
                                 json={
                                     "model": self.model,
                                     "messages": messages,
                                     "temperature": temperature or self.temperature,
                                     "stream": stream and self.stream,
                                     "top_p": top_p or self.top_p},
                                 headers=self.headers,
                                 proxies={"http": None, "https": None},
                                 verify=False,
                                 )

        code = response.status_code
        ret = ''
        if code != 200:
            logger.error(f"code is :{code}")
            return ret

        if stream and self.stream:  # Process streaming response
            self.token_tracker(len(messages), 0)
            async for chunk in AsyncLineIterator(response):
                if self.interrupted:
                    logger.info("Generation interrupted by user")
                    break
                line = chunk.strip()
                if line:
                    line = line.removeprefix("data:").strip()

                    if line != '[DONE]':
                        data = json.loads(line)

                        content = data['choices'][0]['delta']['content']
                        ret += content
                        await self.chunk_queue.put(content)
                        num_chars_dict = count_chars(content)
                        num_tokens_estimated = num_chars_dict["chinese"] * self.chinese_token_ratio + \
                            num_chars_dict["english"] * self.english_token_ratio \
                            + num_chars_dict["symbol"]
                        self.token_tracker(0, num_tokens_estimated)
            # Signal the end of streaming
            if not self.interrupted:
                await self.chunk_queue.put(None)

        else:
            response_json = response.json()
            if "choices" not in response_json or not response_json['choices'][0]['message']['content']:
                raise ValueError("Empty or invalid response from LLM")
            ret = response.json()["choices"][0]["message"]["content"]

            self.token_tracker(
                # Extract token usage from response
                response.json()["usage"]["prompt_tokens"],
                response.json()["usage"]["completion_tokens"])

        return ret
