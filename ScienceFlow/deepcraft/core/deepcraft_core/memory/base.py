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

from typing import List, Optional, Union, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
import logging
from datetime import timezone, datetime
from uuid import UUID, uuid4

from ..message import Message, Role
from ..storage.kv_storage import BaseKeyValueStorage, InMemoryKeyValueStorage
from .record import MemoryRecord, ContextRecord
from .context_creators import BaseContextCreator
import json

logger = logging.getLogger(__name__)


class BaseChatHistoryMemory:
    r"""Base class for chat history memory.

    Args:
        storage (BaseKeyValueStorage, optional): A storage backend for storing
            chat history. If `None`, an :obj:`InMemoryKeyValueStorage`
            will be used. (default: :obj:`None`)
        max_messages (int, optional): The maximum number of stored messages.
        keep_rate (float, optional): The decay rate when scoring the messages.
    """

    def __init__(
        self,
        storage: Optional[BaseKeyValueStorage] = None,
        max_messages: Optional[int] = None,
        keep_rate: Optional[float] = 0.9,
    ) -> None:
        if keep_rate > 1.0 or keep_rate < 0.0:
            raise ValueError("`keep_rate` should be in [0,1]")
        self.storage = storage or InMemoryKeyValueStorage()
        self.keep_rate = keep_rate
        self._max_records = max_messages
        self._num_records = 0

    def set_memory_limit(
        self,
        max_messages: Optional[int] = None,
        **kwargs,
        ) -> None:
        self._max_records = max_messages
    
    def exceed_memory_limit(self) -> bool:
        r"""Checks whether exceed the memory limit or not."""
        if not self._max_records:
            return False

        return (
            self._num_records >= self._max_records
        )

    def add_record(self, record: MemoryRecord) -> None:
        r"""Writes one memory record to the memory."""
        self.storage.save(records=[record.to_dict()])
        self._num_records += 1

    def clear(self) -> None:
        r"""Clean up the storage"""
        self.storage.clear()
        self._num_records = 0

    def compress(self) -> None:
        r"""compress chat history when exceeding the memory limit.

        The default strategy will discard the old memory records directly.
        """
        if not self.exceed_memory_limit():
            return

        saved_records = self.retrieve(window_size=self._max_records-1)
        self.clear()

        for record in saved_records:
            self.add_record(record.memory_record)

    def retrieve(
        self,
        query: Optional[Message] = None,
        window_size: Optional[int] = None,
    ) -> List[ContextRecord]:
        r"""Retrieves records with a proper size for the agent from the memory
        based on the window size or fetches the entire chat history if no
        window size is specified.

        Args:
            query (Message): the query message (not used)
            window_size (int, optional): Specifies the number of recent chat
                messages to retrieve. If not provided, the entire chat history
                will be retrieved. (default: :obj:`None`)

        Returns:
            List[ContextRecord]: A list of retrieved records.
        """
        record_dicts = self.storage.load()
        if len(record_dicts) == 0:
            logger.info("The `BaseChatHistoryMemory` is empty.")
            return list()

        chat_records: List[MemoryRecord] = []
        if window_size is not None and window_size >= 0:
            # Initial preserved index: Keep first message
            # if it's SYSTEM/DEVELOPER (index 0)
            start_index = (
                1
                if (
                    record_dicts
                    and record_dicts[0]['role']
                    in {Role.SYSTEM}
                )
                else 0
            )

            """
            Message Processing Logic:
            1. Preserve first system/developer message (if needed)
            2. Keep latest window_size messages from the rest
            """
            preserved_messages = record_dicts[
                :start_index
            ]  # Preserve system message (if exists)
            sliding_messages = record_dicts[
                start_index:
            ]  # Messages to be truncated

            # Take last window_size messages (if exceeds limit)
            truncated_messages = sliding_messages[-window_size:]

            # Combine preserved messages with truncated window messages
            final_records = preserved_messages + truncated_messages
        else:
            # Return full records when no window restriction
            final_records = record_dicts

        chat_records = [
            MemoryRecord.from_dict(record) for record in final_records
        ]

        # We assume that, in the chat history memory, the closer the record is
        # to the current message, the more score it will be.
        output_records = []
        score = 1.0
        for record in reversed(chat_records):
            if record.role == Role.SYSTEM:
                # System messages are always kept.
                output_records.append(
                    ContextRecord(
                        memory_record=record,
                        score=1.0,
                        timestamp=record.timestamp,
                    )
                )
            else:
                # Other messages' score drops down gradually
                score *= self.keep_rate
                output_records.append(
                    ContextRecord(
                        memory_record=record,
                        score=score,
                        timestamp=record.timestamp,
                    )
                )

        output_records.reverse()
        return output_records


class Memory(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",  # Allow extra fields for flexibility in subclasses
    )

    max_messages: int = Field(default=100)

    long_term_log: Optional[str] = Field(default=None)
    # Path to the long-term memory storage file (JSONL format)

    chat_history_memory: BaseChatHistoryMemory = Field(default_factory=BaseChatHistoryMemory)
    # store the convesation history

    # knowledge_base: Optional[BaseKnowledgeMemory] = Field(default=None)
    # TODO: interface to the external knowledges

    context_creator: BaseContextCreator = Field(default_factory=BaseContextCreator)
    # strategy to generate the conversation context

    window_size: Optional[int] = Field(default=None)
    # The number of recent chat messages to retrieve.
    # If not provided, the entire chat history will be retrieved.

    def model_post_init(self, __context: Any) -> None:
        if self.window_size is not None and not isinstance(self.window_size, int):
            raise TypeError("`window_size` must be an integer or None.")
        if self.window_size is not None and self.window_size < 0:
            raise ValueError("`window_size` must be non-negative.")

        self.chat_history_memory.set_memory_limit(max_messages=self.max_messages)

    def add_message(
        self,
        message: Message,
        extra_info: Optional[Dict[str,Any]] = None,
        uuid: Optional[UUID] = None,
        timestamp: Optional[float] = None,
        agent_id: Optional[str] = "",
        **kwargs,
    ) -> None:
        """Add a message to chat history memory and optionally to long-term logging"""
        
        record = MemoryRecord(
            message=message,
            role=message.role,
            extra_info=extra_info or dict(),
            uuid=uuid or uuid4(),
            timestamp=timestamp or datetime.now(timezone.utc).timestamp(),
            agent_id=agent_id,
        )
        
        # Add to chat history
        self.chat_history_memory.add_record(record)
        
        if self.chat_history_memory.exceed_memory_limit():
            self.chat_history_memory.compress()
        
        # save to long-term logging (when configured)
        if self.long_term_log is not None:
            record_dict = record.to_dict()
            with open(self.long_term_log, "a", encoding='utf-8') as f:
                f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

    def add_messages(
        self,
        messages: List[Message],
        **kwargs,
        ) -> None:
        """Add multiple messages to memory
        Args:
            messages:  conversation messages
            """
        for message in messages:
            self.add_message(message, **kwargs)

    def clear(self) -> None:
        """Clear chat history"""
        self.chat_history_memory.clear()

    def get_context(
        self,
        query: Union[str, Message],
        **kwargs,
        ) -> List[Message]:
        r"""Gets chat context relating to the query"""
        if query is not None:
            # TODO: retrieve chat history and knowledges related to the given query
            ...
        else:
            return self.get_recent_records()

    def get_recent_records(
        self,
        n: int = None,
        **kwargs,
        ) -> List[Message]:
        r"""Gets chat context constructed with the recent memory records.
        
        Args:
            n (int): The number of retrieved memory records.
        
        Returns:
            List[Message]: A list of Message objects.
        """
        n = n or self.window_size
        retrieved_chat_records = self.chat_history_memory.retrieve(window_size=n)

        return self.context_creator.create_context(
            chat_history_records=retrieved_chat_records,
            **kwargs)

    @property
    def messages(self) ->List[Message]:
        return self.get_recent_records(n=self.window_size)
    
    @messages.setter
    def messages(self, value: List[Message]):
        self.clear()
        self.add_messages(value)

    @property
    def context_messages(self) -> str:
        r"""Concatenates the retrived context messages into one string"""
        out_str = ""
        context_messages = self.messages
        for msg in context_messages:
            out_str += f"{msg.content}; "
        return out_str
