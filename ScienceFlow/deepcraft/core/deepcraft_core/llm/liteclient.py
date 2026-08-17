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
import asyncio
import aiohttp
import atexit
import gc

from .utils import lazy_import,count_chars
litellm = lazy_import('litellm')
acompletion = lazy_import('litellm', 'acompletion')


from .base import BaseLLM
from ..message import Message
from ..tool import TOOL_CHOICE_VALUES, TOOL_CHOICE_TYPE

logger = logging.getLogger(__name__)

class LiteLLMClient(BaseLLM):
    """
    Enhanced LiteLLM client with SSL verification configuration support
    ssl_verify： SSL verification
    """
    ssl_verify: bool = Field(default=False)

    @model_validator(mode="after")
    def init_liteclient(self) -> "LiteLLMClient":
        litellm.ssl_verify = self.ssl_verify

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
        **kwargs,
    ) -> str:
        """
        Execute an LLM request with optional streaming response.

        Args:
            messages: List of messages, supporting dictionaries or objects with a to_dict() method.
            system_msgs: Optional system messages to prepend
            stream: Whether to use streaming response mode. Defaults to True.
            temperature (float): Sampling temperature for the response

        Returns:
            str:  The generated response

        Raises:
            Exception: If the LLM request fails.
            ValueError: If messages are invalid or response is empty
        """
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            return await self._stream_request(messages, stream, temperature)

        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise
        finally:
            atexit.register(self.close_aiohttp_sessions)

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
            parallel_tool_calls: Whether to call tools in parallel, defaults to False.

        Returns:
            Any: Response from the tool call.

        Raises:
            ValueError: If tools, tool_choice, or messages are invalid
            Exception: If the tool call fails.
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            litellm.drop_params = True

            # Execute the tool call request
            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                temperature=temperature or self.temperature,
                parallel_tool_calls=parallel_tool_calls,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs
            )

            # Check if the response is valid
            if not response.choices or not response.choices[0].message:
                raise ValueError("No valid response received")
            return response.choices[0].message

        except Exception as e:
            logger.error(f"Tool call failed: {str(e)}")
            raise
    async def _stream_request(self, messages, stream, temperature):
        """
        Asynchronously sends a request to the language model and handles the response.

        Args:
            messages: The messages to send to the model.
            stream: Flag indicating whether to stream the response.
            temperature: The temperature parameter for response generation.

        Returns:
            The generated response from the model.
        """
        # Execute the request
        self.interrupted = False # Reset interrupt status
        response = await acompletion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=temperature or self.temperature,
            stream=stream and self.stream,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        # Process the response based on whether it's streaming
        if stream and self.stream:
            # Process streaming response
            collected = []
            self.token_tracker(len(messages), 0)  # Track input tokens
            try:
                async for chunk in response:
                    choice = chunk.choices[0]
                    if self.interrupted:
                        logger.info("Generation interrupted by user")
                        break
                    if not choice:
                        continue

                    if choice.delta.content is None:
                        break
                    content = chunk.choices[0].delta.content or ""
                    collected.append(content)
                    num_chars_dict = count_chars(content)
                    num_tokens_estimated = num_chars_dict["chinese"] * self.chinese_token_ratio + \
                                           num_chars_dict["english"] * self.english_token_ratio \
                                           + num_chars_dict["symbol"]
                    self.token_tracker(0, num_tokens_estimated)   #Track output tokens
                    # Ensure chunk_queue is initialized
                    if not hasattr(self, 'chunk_queue'):
                        self.chunk_queue = asyncio.Queue()

                    # Put the chunk into the shared queue
                    await self.chunk_queue.put(content)

                # Send termination signal
                if not self.interrupted:
                    await self.chunk_queue.put(None)
                full_response = "".join(collected).strip()

                if not full_response:
                    raise ValueError("Empty response from streaming LLM")
                return full_response
            except Exception as e:
                logger.error(f"Streaming processing failed: {str(e)}")
                raise
        else:
            # Track token usage
            num_chars_dict = count_chars(response.choices[0].message.content.strip())
            num_tokens_estimated = num_chars_dict["chinese"] * self.chinese_token_ratio + \
                                   num_chars_dict["english"] * self.english_token_ratio \
                                   + num_chars_dict["symbol"]
            self.token_tracker(0, num_tokens_estimated)   # Record output tokens
            # Process non-streaming response
            if not response.choices:
                raise ValueError("No valid response received")
            return response.choices[0].message.content.strip()

    def close_aiohttp_sessions(self):
        """
        Close all unclosed aiohttp ClientSession objects to fix session warning errors.
        """
        #Fix session warning error not closed
        async def close():
            for obj in gc.get_objects():
                # Check if the object is an instance of aiohttp.ClientSession and is not closed
                if isinstance(obj, aiohttp.ClientSession) and not obj.closed:
                    await obj.close()
        asyncio.run(close())    # Run the asynchronous close function using asyncio.run