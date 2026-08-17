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

from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

from .record import ContextRecord
from ..message import Role, Message


class _ContextUnit(BaseModel):
    idx: int
    record: ContextRecord
    num_tokens: int = 0


class BaseContextCreator:
    r"""A default implementation of context creation strategy.

    This class provides a strategy to generate a conversational context from
    a list of chat history records and knowledges.

    NOTE:
        This default strategy will **IGNORE** any given external knowledges.

    """
    def create_context(
        self,
        chat_history_records: List[ContextRecord],
        **kwargs,
    ) -> List[Message]:
        r"""Constructs conversation context from chat history solely.

        Key strategies:
        1. System message is always prioritized and preserved
        2. Truncation removes low-score messages first
        3. Final output maintains chronological order and in history memory,
           the score of each message decreases according to keep_rate. The
           newer the message, the higher the score.

        Args:
            chat_history_records (List[ContextRecord]): List of context records with scores
                and timestamps.

        Returns:
            List[Message]: Ordered list of messages
        """
        # ======================
        # 1. System Message Handling
        # ======================
        system_unit, regular_units = self._extract_system_message(chat_history_records)

        # ======================
        # 2. Deduplication & Initial Processing
        # ======================
        seen_uuids = set()
        if system_unit:
            seen_uuids.add(system_unit.record.memory_record.uuid)

        # Process non-system messages with deduplication
        for idx, record in enumerate(chat_history_records):
            if record.memory_record.uuid in seen_uuids:
                continue
            seen_uuids.add(record.memory_record.uuid)

            regular_units.append(
                _ContextUnit(
                    idx=idx,
                    record=record,
                    num_tokens=0,
                )
            )

        # ======================
        # 3. Return
        # ======================
        sorted_units = sorted(
            regular_units, key=self._conversation_sort_key ## FORSECURE
        )
        return self._assemble_output(sorted_units, system_unit)

    def _extract_system_message(
        self, records: List[ContextRecord]
    ) -> Tuple[Optional[_ContextUnit], List[_ContextUnit]]:
        r"""Extracts the system message from records and validates it.

        Args:
            records (List[ContextRecord]): List of context records
                representing conversation history.

        Returns:
            Tuple[Optional[_ContextUnit], List[_ContextUnit]]: containing:
            - The system message as a `_ContextUnit`, if valid; otherwise,
                `None`.
            - An empty list, serving as the initial container for regular
                messages.
        """
        if not records:
            return None, []

        first_record = records[0]
        if (first_record.memory_record.role != Role.SYSTEM):
            return None, []

        system_message_unit = _ContextUnit(
            idx=0,
            record=first_record,
            num_tokens=0,
        )
        return system_message_unit, []

    def _conversation_sort_key(
        self, unit: _ContextUnit
    ) -> Tuple[float, float]:
        r"""Defines the sorting key for assembling the final output.

        Sorting priority:
        - Primary: Sort by timestamp in ascending order (chronological order).
        - Secondary: Sort by score in descending order (higher scores first
            when timestamps are equal).

        Args:
            unit (_ContextUnit): A `_ContextUnit` representing a conversation
                record.

        Returns:
            Tuple[float, float]:
            - Timestamp for chronological sorting.
            - Negative score for descending order sorting.
        """
        return (unit.record.timestamp, -unit.record.score)

    def _assemble_output(
        self,
        context_units: List[_ContextUnit],
        system_unit: Optional[_ContextUnit],
    ) -> List[Message]:
        r"""Assembles final message list with proper ordering.

        Args:
            context_units (List[_ContextUnit]): Sorted list of regular message units.
            system_unit (Optional[_ContextUnit]): System message unit (if present).

        Returns:
            List[Message]: ordered messages
        """
        messages = []

        # Add system message first if present
        if system_unit:
            messages.append(
                system_unit.record.memory_record.message
            )

        # Add sorted regular messages
        for unit in context_units:
            messages.append(unit.record.memory_record.message)

        return messages
