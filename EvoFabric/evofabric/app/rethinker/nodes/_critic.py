# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


from typing import Optional

from evofabric.core.factory import safe_get_attr
from evofabric.core.graph import current_ctx
from evofabric.core.typing import State, StateDelta
from evofabric.logger import get_logger
from ._base import AsyncNodeWithCacheAndConcurrencyLimit
from ._utils import clip_str, parse_answer, strip_think_and_exec
from .._config import config


class CriticWithRethinkNode(AsyncNodeWithCacheAndConcurrencyLimit):
    # Key from which the content to be reflected (criticized) is retrieved
    input_key: Optional[str] = None

    # Index of the previous round.
    # If None, this is the first round and no rethink is performed.
    # Otherwise, the answer from the specified round is retrieved and rethought.
    last_round: Optional[str] = None

    # Key under which the reflection (rethinking result) is stored
    output_key: Optional[str] = None

    async def _run(self, state: State) -> StateDelta:
        query = safe_get_attr(state, "query")
        index = int(safe_get_attr(state, "index"))

        solution_need_critic = safe_get_attr(state, self.input_key)[index]["response"]
        solution_need_critic = strip_think_and_exec(solution_need_critic)

        if self.last_round is None:
            prompt = str(config.prompts.critic_user_prompt)
            prompt = prompt.format(
                query=query,
                s_solution=solution_need_critic
            )
        else:
            last_round_critic = parse_answer(safe_get_attr(state, self.last_round)[index]["response"])

            prompt = str(config.prompts.critic_twice_user_prompt)
            prompt = prompt.format(
                query=query,
                s_solution=solution_need_critic,
                last_round_answer=clip_str(last_round_critic, 300),
            )

        response = await self.agent(prompt)
        critic: list[Optional[dict]] = [None] * config.structure.num_parallel
        critic[index] = response.model_dump()
        return {self.output_key: critic}
