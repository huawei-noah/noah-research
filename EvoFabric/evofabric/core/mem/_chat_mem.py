# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from typing import Annotated, Optional, Any, List
import ast

import json
from loguru import logger
from pydantic import Field

from ..typing import StateMessage, UserMessage, LLMChatResponse, DBItem
from ..clients import ChatClientBase
from ..vectorstore import DBBase
from ._cognitive_mem import CognitiveMem
from ._default_mem_prompt_zh import FACT_RETRIEVAL_PROMPT_ZH, DEFAULT_UPDATE_MEMORY_PROMPT_ZH
from ._default_mem_prompt_en import FACT_RETRIEVAL_PROMPT_EN, DEFAULT_UPDATE_MEMORY_PROMPT_EN
from ..factory import FactoryTypeAdapter


class ChatMem(CognitiveMem):
    """Multi round conversation, prompt-driven implementation"""
    user_extract_prompt: Optional[str] = Field(default=None, description="Mem features extract prompt")
    user_update_prompt: Optional[str] = Field(default=None, description="Merge current info with existing memories")
    feat_define_prompt: Optional[str] = Field(default=None, description="Merge current info with existing memories")
    chat_client: ChatClientBase = Field(description="Call llm api")
    vectorstore: Annotated[DBBase, FactoryTypeAdapter, Field(description="Long term memory DB")]

    def model_post_init(self, context: Any, /) -> None:
        """format the memory system prompt"""
        if self.user_extract_prompt is None:
            if self.zh_mode:
                self.user_extract_prompt = FACT_RETRIEVAL_PROMPT_ZH
            else:
                self.user_extract_prompt = FACT_RETRIEVAL_PROMPT_EN

        if self.user_update_prompt is None:
            if self.zh_mode:
                self.user_update_prompt = DEFAULT_UPDATE_MEMORY_PROMPT_ZH
            else:
                self.user_update_prompt = DEFAULT_UPDATE_MEMORY_PROMPT_EN

        if self.feat_define_prompt is not None:  # additional memory feats define
            if self.zh_mode:
                self.user_extract_prompt = self.user_extract_prompt.replace("[记忆信息]", self.feat_define_prompt)
            else:
                self.user_extract_prompt = self.user_extract_prompt.replace("[Memory Information]",
                                                                            self.feat_define_prompt)

    async def _update_db_mem(self, old_memory: list[DBItem], new_mem: list[dict[str, str]]) -> None:
        """Long term mem db operation"""
        for mem_item in new_mem:
            if mem_item["event"] == "NONE":
                continue
            elif mem_item["event"] == "ADD":
                await self.vectorstore.add_texts([DBItem(text=mem_item["text"])])
            elif mem_item["event"] == "DELETE":
                oid = int(mem_item["id"])
                mem_id = old_memory[oid].ids
                await self.vectorstore.delete_by_ids([mem_id])
            elif mem_item["event"] == "UPDATE":
                oid = int(mem_item["id"])
                mem_id = old_memory[oid].ids
                await self.vectorstore.delete_by_ids([mem_id])
                await self.vectorstore.add_texts([DBItem(text=mem_item["text"])])
            else:
                logger.error(f"Unknown mem operation {mem_item['event']}, ignore.")
        return

    async def _mem_feat_extract(self, context: str) -> List[str]:
        """
        Extract memory feats from message lists
        :param messages: contexts
        :return: mem features
        """

        extract_prompt = self.user_extract_prompt + context

        # create user message list:
        mem_gen_context = [UserMessage(role="user", content=extract_prompt)]
        try:
            raw_fact = '[]'
            async for msg in self.chat_client.create_on_stream(
                    messages=mem_gen_context):
                if isinstance(msg, LLMChatResponse):
                    raw_fact = msg.content
            lid, rid = raw_fact.find('['), raw_fact.rfind(']')
            feats = ast.literal_eval(raw_fact[lid:rid + 1])  # transform llm response to feat list
            if not isinstance(feats, list):
                return []
            return feats

        except Exception as e:
            logger.error(str(e))
            return []

    async def _mem_update(self, context: str, feats: List[str]) -> List[dict[str, str]]:
        """
        based on given features to merge and update memory
        :param messages: context
        :param feats: extracted mem features
        :return: memory operations
        """
        update_prompt = self.user_update_prompt + context
        retrieval_memory_items = await self.vectorstore.similarity_search(str(feats))

        if len(retrieval_memory_items) == 0:  # No related memory
            res = [{"id": i, "text": feats[i], "event": "ADD"} for i in range(len(feats))]
            await self._update_db_mem([x.to_db_item() for x in retrieval_memory_items], res)
            return res

        # merge existing mem and new mem
        old_memory = [{"id": i, "text": retrieval_memory_items[i].text} for i in
                      range(len(retrieval_memory_items))]

        if self.zh_mode:
            update_instance = f"旧记忆：\n{str(old_memory)}\n提取的新记忆：{feats}\n输出：\n"
        else:
            update_instance = f"Old Memory: \n{str(old_memory)}\nNew facts: {str(feats)}\nOutput:\n"
        update_prompt += update_instance

        # Analyze by LLM.
        mem_update_context = [UserMessage(role="user", content=update_prompt)]
        try:
            base_memory = []
            raw_mem = json.dumps(base_memory, ensure_ascii=False, indent=4)
            async for msg in self.chat_client.create_on_stream(  # type: ignore
                    messages=mem_update_context):
                if isinstance(msg, LLMChatResponse):
                    raw_mem = msg.content
            lid, rid = raw_mem.find('['), raw_mem.rfind(']')
            update_mem = json.loads(raw_mem[lid:rid + 1])
            if not isinstance(update_mem, list):
                return []
            mem_operations = update_mem
            await self._update_db_mem([x.to_db_item() for x in retrieval_memory_items], mem_operations)
            return mem_operations

        except Exception as e:
            logger.error(str(e))
            return []

    async def _summary(self, feats: List[str]) -> str:
        """Basically merge the extracted"""
        if self.zh_mode:
            content = "已知的记忆信息为：" + str(feats)
        else:
            content = "Known memory info: " + str(feats)
        return content

    async def _context_update(self, summary: str, messages: List[StateMessage]) -> List[StateMessage]:
        """Based on mem summary info to update basic messages"""
        messages = [UserMessage(content=summary)] + messages
        return messages
