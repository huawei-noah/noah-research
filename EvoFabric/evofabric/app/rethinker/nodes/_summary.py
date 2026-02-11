# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from typing import Optional

from evofabric.core.clients import ChatClientBase
from evofabric.core.factory import safe_get_attr
from evofabric.core.graph import current_ctx
from evofabric.core.typing import State, StateDelta
from evofabric.logger import get_logger
from ._base import AsyncNodeWithCacheAndConcurrencyLimit
from ._utils import strip_response
from .. import config


class GuidedSummaryNode(AsyncNodeWithCacheAndConcurrencyLimit):
    client: ChatClientBase
    """LLM chat client used to generate the summary."""

    input_key: str = None
    """Key from which the content to be summarized is retrieved"""

    output_key: str = None
    """Key under which the generated summary is stored"""

    async def _run(self, state: State) -> StateDelta:
        query = safe_get_attr(state, "query")
        index = int(safe_get_attr(state, "index"))

        solution_need_summary = safe_get_attr(state, self.input_key)[index]["response"]
        solution_need_summary = strip_response(solution_need_summary)

        prompt = str(config.prompts.guided_summary_prompt)
        prompt = prompt.format(
            problem=query,
            student_solution=solution_need_summary,
        )
        response = await self.client.create(messages=[{"role": "user", "content": prompt}])  # type: ignore
        response = response.model_dump()
        response["response"] = response["content"]

        solution_summary: list[Optional[dict]] = [None] * config.structure.num_parallel
        solution_summary[index] = response
        return {self.output_key: solution_summary}
