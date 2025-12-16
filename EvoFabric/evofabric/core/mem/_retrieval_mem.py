# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import uuid
from typing import Annotated, List, Optional

from pydantic import Field

from ._base_mem import MemBase
from ..typing import UserMessage
from ..clients import RerankClientBase
from ..factory import FactoryTypeAdapter
from ..typing import DBItem, StateMessage
from ..vectorstore import DBBase


class RetrievalMem(MemBase):
    """Implementation of basic retrieval memory"""
    vectorstore: Annotated[DBBase, FactoryTypeAdapter, Field(description="vector store")]
    reranker: Annotated[RerankClientBase, FactoryTypeAdapter, Field(description="rerank function")]
    use_rerank: Optional[bool] = Field(default=True, description="whether adopt reranker")
    message_rounds: Optional[int] = Field(default=1, description="considering message rounds")

    async def _retrival_text(self, question: str) -> List[str]:
        """retrieval text basic function"""
        items = await self.vectorstore.similarity_search(question)
        if len(items) == 0:
            return []
        if not self.use_rerank:
            return [d.text for d in items]

        # rerank
        if self.use_rerank:
            contents = [d.text for d in items]
            rerank_ids = await self.reranker.rank(question, contents)

            results = [contents[ids] for ids in rerank_ids]
            return results

    def _message_to_query(self, messages: List[StateMessage]) -> str:
        """transfer the last N round contexts to retrieval query"""
        query = ""
        consider_messages = messages[-self.message_rounds:]
        for msg in consider_messages:
            if not hasattr(msg, "role") or not hasattr(msg, "content"):
                continue
            query += f"{msg.role}: {msg.content}\n"
        return query.strip()

    async def clear(self):
        """clear the memory"""
        await self.vectorstore.clear_db()

    async def retrieval_update(self, messages: List[StateMessage], **kwargs) -> List[StateMessage]:
        """retrieval the memory based on contexts and update to contexts"""
        query = self._message_to_query(messages)
        retrival_chunks = await self._retrival_text(query)
        if len(retrival_chunks) == 0:
            return messages
        messages = [UserMessage(content=str(retrival_chunks))] + messages
        return messages

    async def add_messages(self, messages: List[StateMessage], **kwargs) -> None:
        """add messages to memory"""
        # Retrieval memory cannot be updated based on agent contexts.
        return

    async def add_texts(self, texts: List[str]) -> None:
        """batch add texts to vector store"""
        items = [DBItem(text=text, metadata=dict(), ids=str(uuid.uuid4())) for text in texts]
        await self.vectorstore.add_texts(items)
        await self.vectorstore.persist()

    async def save(self):
        """save the memory"""
        await self.vectorstore.persist()
