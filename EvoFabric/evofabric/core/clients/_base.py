# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from typing import AsyncIterator, List, Sequence, Union

from ..factory import BaseComponent
from ..trace import trace_chat
from ..typing import ChatStreamChunk, LLMChatResponse, StateMessage


class ChatClientBase(BaseComponent):
    """
    This Client will serve as the foundational type for interacting with any backend LLM models.
    """

    @trace_chat
    async def create_on_stream(
            self, messages: Sequence[StateMessage], **kwargs
    ) -> AsyncIterator[Union[ChatStreamChunk, LLMChatResponse]]:
        """Streaming mode to generate LLM response"""
        raise NotImplementedError

    @trace_chat
    async def create(
            self, messages: Sequence[StateMessage], **kwargs
    ) -> LLMChatResponse:
        """Generate LLM response"""
        final_result = None

        async for msg in self.create_on_stream(messages, **kwargs):  # type: ignore
            if isinstance(msg, LLMChatResponse):
                final_result = msg
        if not final_result:
            raise ValueError("Client do not return final chat response.")
        return final_result


class EmbedClientBase(BaseComponent):
    """
    This client will serve as the foundational type for interacting with any backend embedding models.
    It is in langchain openai embedding format, thus can be used by langchain chromadb.
    """

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_documents(
            self, texts: list[str], **kwargs
    ) -> list[list[float]]:
        raise NotImplementedError

    async def aembed_query(self, text: str, **kwargs) -> list[float]:
        raise NotImplementedError

    async def aembed_documents(
            self, texts: list[str], **kwargs
    ) -> list[list[float]]:
        raise NotImplementedError


class RerankClientBase(BaseComponent):
    """This client will serve as the foundational type for interacting with any backend reranking models."""

    async def rank(self, query: str, texts: List[str], **kwargs) -> List[int]:
        """rank query texts pairs and return top n indices"""
        raise NotImplementedError
