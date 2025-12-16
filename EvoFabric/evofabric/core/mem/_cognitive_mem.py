# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import abstractmethod
from typing import Annotated, List, Optional, Union

from pydantic import Field

from ._base_mem import MemBase
from ..factory import FactoryTypeAdapter
from ..typing import StateMessage
from ..vectorstore import DBBase


class CognitiveMem(MemBase):
    """A basic class of cognitive mem, which knows how to understand information, store memory and use memory"""
    zh_mode: Optional[bool] = Field(default=True, description="Control the frame in Chinese mode or English mode")
    message_rounds: Optional[int] = Field(default=100, description="Max considered message rounds")
    vectorstore: Annotated[DBBase, FactoryTypeAdapter, Field(description="vector store")]

    @abstractmethod
    async def _mem_feat_extract(self, context: str) -> List[Union[str, dict]]:
        """Extract features from contexts"""
        ...

    @abstractmethod
    async def _mem_update(self, context: str, feats: List[Union[str, dict]]) -> List[dict[str, str]]:
        """Merge current info with existing memories"""
        ...

    @abstractmethod
    async def _summary(self, feats: List[Union[str, dict]]) -> str:
        """Summary the recalled memories"""
        ...

    @abstractmethod
    async def _context_update(self, summary: str, messages: List[StateMessage]) -> List[StateMessage]:
        """Based on mem summary info to update basic messages"""
        ...

    def _select_messages(self, messages: List[StateMessage]) -> List[StateMessage]:
        """Select the saved messeages. In default inplementation, we save the last "message_rounds" messages. Default inplementation."""
        return messages[-self.message_rounds:]

    def _message_to_text(self, messages: List[StateMessage]) -> str:
        """Transfer contexts to str. Default inplementation"""
        translate = {"user": "用户", "assistant": "助手", "tool": "工具", "system": "系统"}
        if self.zh_mode:
            text = "上下文信息："
        else:
            text = "Context info: "
        for msg in messages:
            if self.zh_mode:
                delta = f"\n{translate[msg.role]}: {msg.content}"
            else:
                delta = f"\n{msg.role}: {msg.content}"
            text += delta
        return text

    async def _retrival_text(self, question: str) -> List[str]:
        """retrieval text basic function"""
        items = await self.vectorstore.similarity_search(question)
        return [f"content: {d.text}" for d in items]

    async def add_messages(self, messages: List[StateMessage], **kwargs) -> None:
        """
        Agent node interface: update memory based on messages, default implementation
        :param messages: context
        :param kwargs:
        :return:
        """
        save_messages = self._select_messages(messages)
        context = self._message_to_text(save_messages)
        feat_lists = await self._mem_feat_extract(context)
        await self._mem_update(context, feat_lists)

    async def retrieval_update(self, messages: List[StateMessage], **kwargs) -> List[StateMessage]:
        """Agent node interface: update messages based on memory, default implementation"""
        query = self._message_to_text(messages=messages[-self.message_rounds:])
        retrival_chunks = await self._retrival_text(query)
        content = await self._summary(retrival_chunks)
        update_messages = await self._context_update(content, messages)
        return update_messages

    async def clear(self):
        """clear the memory"""
        await self.vectorstore.clear_db()
