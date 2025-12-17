# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


from abc import ABC, abstractmethod
from typing import List

from ..factory import BaseComponent
from ..typing import StateMessage


class MemBase(ABC, BaseComponent):
    """The basic memory interface define, can be used for opensource adaption"""

    @abstractmethod
    async def retrieval_update(self, messages: List[StateMessage], **kwargs) -> List[StateMessage]:
        """
        retrival memory and update context messages.
        :param messages: context messages
        :return: updated context messages based on memory
        """
        ...

    @abstractmethod
    async def add_messages(self, messages: List[StateMessage], **kwargs) -> None:
        """
        add context messages to memory vectorstore
        :param messages: context messages
        :return:
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """
        clear all memories
        :return:
        """
        ...
