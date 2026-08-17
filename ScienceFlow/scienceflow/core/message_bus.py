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

"""Async event-driven message bus for inter-agent communication."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger("scienceflow")


class AsyncMessageBus:
    """Publish/subscribe message bus. All handlers must be async callables."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable[..., Coroutine]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[..., Coroutine]) -> None:
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[..., Coroutine]) -> None:
        handlers = self._subscribers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, topic: str, message: Any) -> None:
        handlers = self._subscribers.get(topic, [])
        if not handlers:
            return
        results = await asyncio.gather(
            *(h(message) for h in handlers), return_exceptions=True
        )
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning(f"Handler error on topic '{topic}': {r}")

    @property
    def topics(self) -> list[str]:
        return list(self._subscribers.keys())
