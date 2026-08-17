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

from __future__ import annotations
from typing import Any, List, Optional, Union
import asyncio
from pydantic import ConfigDict, Field, model_validator

# Updated: lazy load <SentenceTransformer> since not all projects need this.
# from sentence_transformers import SentenceTransformer
from .utils import lazy_import
sentence_transformers = lazy_import('sentence_transformers')
SentenceTransformer = lazy_import('sentence_transformers', 'SentenceTransformer')

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

from .base import BaseLLM
from ..message import Message
from ..tool import TOOL_CHOICE_TYPE


class SentenceTransformerEncoder(BaseLLM):
    """Sentence Transformers embedding model compatible with BaseLLM."""
    transformer_model: Any = Field(default=None, exclude=True)  #: :meta private:

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",  # Allow extra fields for flexibility in subclasses
    )

    @model_validator(mode="after")
    def initializeAgent(self) -> "SentenceTransformerEncoder":
        self.transformer_model = SentenceTransformer(model_name_or_path=self.model)

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
            **kwargs: Any,
    ) -> str:
        """
           Asynchronously generate a response based on input messages and parameters.
        Args:
            messages (List[Union[dict, Message]]): List of input messages, each can be a dict or Message object.
            system_msgs (Optional[List[Union[dict, Message]]]): Optional list of system messages, each can be a dict or Message object.
            stream (bool): Flag for streaming the response, default is True.
            temperature (Optional[float]): Optional temperature parameter for model generation.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            str: Generated response, though currently not implemented.

        Raises:
            NotImplementedError: Always raised as text generation is not supported by this embedding model.
    """
        raise NotImplementedError("Text generation is not supported by this embedding model.")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
            self,
            messages: List[Union[dict, Message]],
            system_msgs: Optional[List[Union[dict, Message]]] = None,
            timeout: int = 60,
            tools: Optional[List[dict]] = None,
            tool_choice: TOOL_CHOICE_TYPE = "auto",
            temperature: Optional[float] = None,
            parallel_tool_calls: Optional[bool] = None,
            **kwargs: Any,
    ) -> str:
        """
           Asynchronously handle tool calls based on input messages and parameters.
        Args:
            messages (List[Union[dict, Message]]): List of input messages, each can be a dict or Message object.
            system_msgs (Optional[List[Union[dict, Message]]]): Optional list of system messages, each can be a dict or Message object.
            timeout (int): Timeout in seconds for the tool call, default is 60.
            tools (Optional[List[dict]]): Optional list of tools, each represented as a dict.
            tool_choice (TOOL_CHOICE_TYPE): Tool choice strategy, default is "auto".
            temperature (Optional[float]): Optional temperature parameter for model generation.
            parallel_tool_calls (Optional[bool]): Optional flag for parallel tool calls.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            str: Result of the tool call, though currently not implemented.

        Raises:
            NotImplementedError: Always raised as tool calls are not supported by this embedding model."""
        raise NotImplementedError("Tool calls are not supported by this embedding model.")


    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_embedding(
            self,
            input: str,
            encoding_format: str = "float",
    ) -> List[float]:  # Return embedding vector as a list of floats
        """
        Use the transformer model to encode the input text and return the embedding vector.

        Args:
            input (str): Input text.
            encoding_format (str): Encoding format, default is "float".

        Returns:
            List[float]: Embedding vector as a list of floats.

        Raises:
            ValueError: If the encoding_format is not supported.
        """
        loop = asyncio.get_event_loop()  # Get the current event loop
        embeddings = await loop.run_in_executor(  # Asynchronously run model encoding in a thread pool
            None,
            lambda: self.transformer_model.encode([input], )  # Encode the input text
        )
        if encoding_format == "float":  # If encoding format is float
            return embeddings[0].tolist()  # Return the embedding vector as a list
        else:
            raise ValueError(
                f"Unsupported encoding format: {encoding_format}")  # Raise an exception for unsupported encoding format

    def get_output_dim(self) -> int:
        """
        Get the output dimension of the transformer model.

        Returns:
            int: Output dimension of the model.
        """
        return self.transformer_model.get_sentence_embedding_dimension()  # Return the output dimension of the model