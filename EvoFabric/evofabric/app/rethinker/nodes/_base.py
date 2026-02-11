# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import json
from pathlib import Path
from typing import Optional

from pydantic import Field

from evofabric.core.factory import safe_get_attr
from evofabric.core.graph import AsyncNode, current_ctx
from evofabric.core.typing import MISSING, State, StateDelta
from evofabric.logger import get_logger
from .._config import config
from ..agent import CodingAgent

logger = get_logger()


class AsyncNodeWithCacheAndConcurrencyLimit(AsyncNode):
    """
    An asynchronous node with built-in caching and concurrency control.

    This class wraps node execution with:
    1) A cache mechanism to avoid redundant computation by reusing previously
       stored results.
    2) A semaphore-based concurrency limit to control the maximum number of
       concurrent executions.

    Subclasses should implement the `_run` method to define the actual node logic.
    """
    agent: Optional[CodingAgent] = Field(default=None)
    """A coding agent that can iteratively exec python code to solve the problem."""

    cache_dir_key: Optional[str] = Field(default="cache_dir")
    """Key value in state, whose value stores the cache dir path."""

    async def _run(self, state: State) -> StateDelta:
        """
        Execute the core logic of the node.

        This method should be implemented by subclasses. It contains the actual
        computation or model invocation logic, without any concern for caching
        or concurrency control.

        Args:
            state: The current execution state.

        Returns:
            A StateDelta representing the changes produced by this node.
        """
        raise NotImplementedError

    async def __call__(self, state: State) -> StateDelta:
        """
        Execute the node with caching and concurrency control.

        The execution follows these steps:
        1) Check whether a cached result exists for this node and state.
           If so, load and return it directly.
        2) Acquire a semaphore to enforce the global concurrency limit.
        3) Invoke the core logic via `_run`.
        4) Persist the result to disk for future reuse.

        Args:
            state: The current execution state.

        Returns:
            A StateDelta produced by this node.
        """
        semaphore = config.get_semaphore()
        node_name = current_ctx().node_name
        query_id = current_ctx().meta['query_id']

        async with semaphore:
            if self.cache_dir_key and (cache_dir := safe_get_attr(state, self.cache_dir_key)) is not MISSING:
                cache_dir = Path(cache_dir)
                node_cache_path = cache_dir / (node_name + ".json")
                if node_cache_path.exists():
                    logger.info(f"Loading cached result | Node name: {node_name} | Query id: {query_id}")
                    with open(node_cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                logger.info(f"Running node. | Node name: {node_name} | Query id: {query_id}")
                response = await self._run(state)
                cache_dir.parent.mkdir(parents=True, exist_ok=True)
                with open(node_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(response, f, ensure_ascii=False, indent=4)
                return response
            else:
                logger.info(f"Running node. | Node name: {node_name} | Query id: {query_id}")
                response = await self._run(state)
                return response
