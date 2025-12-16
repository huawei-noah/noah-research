# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Callable, List, Optional, Awaitable, Union

import uuid
from pydantic import Field
from loguru import logger

from ._cognitive_mem import CognitiveMem
from ..clients import ChatClientBase
from ..typing import DBItem, StateMessage, SystemMessage, UserMessage, LLMChatResponse
from ..vectorstore import DBBase


class TaskMem(CognitiveMem):
    """
    Features：
    1. Memory metadata include：task, task_id, context，correctness，score
    2. Can retrieve different memories in each round: context
    3. Evaluation entrance：score_fuc, correctness_fuc
    4. Meta prompt generation in each execution round
    """

    chat_client: ChatClientBase = Field(description="Call llm api")
    vectorstore: DBBase = Field(description="Long term memory DB")
    user_summary_prompt: Optional[str] = Field(default=None,
                                               description="Generate the experience based on recalled context")

    eval_fuc: Callable[[List[SystemMessage]], Awaitable[tuple[bool, float, str]]] = Field(
        description="Evaluate the current score of the execution correctness, score and critics")

    async def _mem_feat_extract(self, context: str) -> List[Union[str, dict]]:
        """It is replaced by _task_mem_feat_extract."""
        return

    async def add_messages(self, messages: List[StateMessage], **kwargs) -> None:
        """
        Agent node interface: update memory based on messages, default implementation
        :param messages: context
        :param kwargs:
        :return:
        """
        save_messages = self._select_messages(messages)
        context = self._message_to_text(save_messages)
        feat_lists = await self._task_mem_feat_extract(save_messages)
        await self._mem_update(context, feat_lists)

    async def _task_mem_feat_extract(self, messages: List[StateMessage]) -> List[dict]:
        """
        Generate the memory structure
        :param messages: execution contexts
        :return: List of task memory struct
        """

        if not hasattr(messages[-1], 'content'):
            return []

        # create user message list:
        # Get task and task id
        task_id = None
        task = None
        for msg in messages:  # use the first effective information as task instruction.
            if msg.content is not None:
                task_id = msg.msg_id
                task = msg.content
                break
        if task_id is None:
            task_id = str(uuid.uuid4())

        # reward marker
        [correctness, score, critic] = await self.eval_fuc(messages)

        context = self._message_to_text(messages)

        task_mem_item = {
            "context": context,  # search item
            "task": task,
            "task_id": task_id,
            "score": score,
            "correctness": correctness,
            "critic": critic
        }

        return [task_mem_item]

    async def _mem_update(self, context: str, feats: List[dict]) -> List[dict[str, str]]:
        """update the long-term memory strategy"""
        add_items = []
        for mem_item in feats:  # Often, there is only one item in feats
            retrieval_memory_items = await self.vectorstore.similarity_search(mem_item["context"])
            ignore = False

            # Simple implementation of update strategy
            for mem_db_item in retrieval_memory_items:
                task_id = mem_db_item.metadata["task_id"]

                if task_id == mem_item["task_id"] and not mem_item["correctness"]:  # This task has failed
                    ignore = True
                    break

            if not ignore:
                add_items.append(mem_item)

        for item in add_items:
            mem = DBItem(text=item["context"], metadata=item, ids=None)
            await self.vectorstore.add_texts([mem])

        return add_items

    async def _summary(self, feats: List[str]) -> str:
        """Summary the recalled memories"""

        # todo: experience extraction
        if self.zh_mode:
            conclude_prompt = self.user_summary_prompt + str(feats) + "输出："
        else:
            conclude_prompt = self.user_summary_prompt + str(feats) + "Output:"

        conclude_gen_context = [UserMessage(role="user", content=conclude_prompt)]
        try:
            async for msg in self.chat_client.create_on_stream(
                    messages=conclude_gen_context):
                if isinstance(msg, LLMChatResponse):
                    res = msg.content
                    return res
        except Exception as e:
            logger.error(str(e))
            return ""

    async def _context_update(self, summary: str, messages: List[StateMessage]) -> List[StateMessage]:
        """Based on mem summary info to update basic messages"""
        messages = [UserMessage(content=summary)] + messages
        return messages

    async def _retrival_text(self, question: str) -> List[str]:
        """retrieval text, return all message"""
        items = await self.vectorstore.similarity_search(question)
        return [
            f"content: {d.text}, correctness: {d.metadata['correctness']}, score: {d.metadata['score']}, critic: {d.metadata['critic']}"
            for d in items]
