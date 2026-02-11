# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from typing import Optional

from evofabric.core.factory import safe_get_attr
from evofabric.core.graph import current_ctx
from evofabric.core.typing import State, StateDelta
from evofabric.logger import get_logger
from ._base import AsyncNodeWithCacheAndConcurrencyLimit
from ._utils import parse_answer
from .._config import config


class SolutionWithReThinkNode(AsyncNodeWithCacheAndConcurrencyLimit):
    # Key indicating from which previous round the solution is retrieved.
    # If None, this is the first round and no rethinking is performed.
    last_round: Optional[str] = None

    # Key under which the final (or rethought) solution is stored
    output_key: str = "solution"

    async def _run(self, state: State) -> StateDelta:
        query = safe_get_attr(state, "query")
        index = int(safe_get_attr(state, "index"))

        if not self.last_round:
            prompt = str(config.prompts.solver_user_prompt)
            prompt = prompt.format(query=query)
        else:
            last_round_answer = safe_get_attr(state, self.last_round)[index]["response"]
            prompt = str(config.prompts.solver_twice_user_prompt)
            prompt = prompt.format(query=query, last_round_answer=parse_answer(last_round_answer))

        response = await self.agent(prompt)

        solution: list[Optional[dict]] = [None] * config.structure.num_parallel
        solution[index] = response.model_dump()
        return {self.output_key: solution}
