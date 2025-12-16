# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import re
from dataclasses import dataclass
from typing import Annotated, List

from loguru import logger
from pydantic import BaseModel

from evofabric.core.agent import AgentNode
from evofabric.core.factory import ComponentFactory
from evofabric.core.graph import GraphBuilder, GraphEngine
from evofabric.core.typing import AssistantMessage, State, StateMessage, UserMessage
from .llm_config import LLMConfig
from ..evaluator import BaseEvaluator
from ..prompt import PromptSampler


@dataclass
class KernelEvolve:
    agent: AgentNode
    initial_code: str
    graph: GraphEngine

    def __init__(
            self, initial_code: str,
            llm_config: LLMConfig,
            evaluator: BaseEvaluator):
        self.initial_code = initial_code
        self.__build_prompt_sampler(initial_code)
        self.__initial_evaluator(evaluator)
        self.__build_agent_node(llm_config)
        self.__build_graph_node()

    def evolve(self):
        input_msg = {
            "messages": [UserMessage(content=self.prompt.user_prompt)]
        }
        result = asyncio.run(self.graph.run(input_msg))
        return self.__resolve_result(result.messages[-1].content)

    def __resolve_result(self, result):
        pattern = r"```.*?\n(.*?)```"
        matches = re.findall(pattern, result, re.DOTALL)
        if len(matches) == 0:
            reason = f"""
            Sorry, unable to evaluate your code: 
            {self.initial_code}
            You can try again later.
            The detail reason is : \n
            """
            return False, reason + result
        else:
            return True, matches[0]

    def __build_prompt_sampler(self, initial_code: str):
        logger.info(f"building prompt sampler, evaluate code: {initial_code}")
        self.prompt = PromptSampler(initial_code)

    def __initial_evaluator(self, evaluator: BaseEvaluator):
        def validate_model(candidate_code: str):
            """
            Get metrics of optimized torch code.

            Args:
                candidate_code (str):
                    The optimized PyTorch model code to be evaluated.

            Returns:
                Metrics: An object containing performance evaluation results.
                    - speedup (float): Ratio of original runtime to optimized runtime.
                      Values > 1.0 indicate improved performance.
                    - original_time (float): Execution time (in seconds) of the original model.
                    - optimized_time (float): Execution time (in seconds) of the optimized model.
                    - error (Optional[str]): Error message if validation fails, otherwise None.
            """
            logger.info(f"start evaluate code: {candidate_code}")
            metrics = evaluator.evaluate(self.initial_code, candidate_code)
            logger.info(f"evaluate results: {metrics.to_dict()}")
            return metrics.to_dict()

        self.evaluator = validate_model

    def __build_agent_node(self, llm_config: LLMConfig):
        client_config = ComponentFactory.create(
            llm_config.model_class,
            model=llm_config.model_name,
            stream=True,
            client_kwargs={
                "api_key": llm_config.api_key,
                "base_url": llm_config.base_url,
                **llm_config.extra_params
            },
            http_client_kwargs={"verify": False},
        )
        tool_manager_config = ComponentFactory.create(
            "ToolManager",
            tools=[self.evaluator]
        )
        self.agent = AgentNode(
            client=client_config,
            tool_manager=tool_manager_config,
            system_prompt=self.prompt.system_prompt
        )

    def __build_graph_node(self):
        class StateSchema(BaseModel):
            messages: Annotated[List[StateMessage], "append_messages"]

        def repeat_fc_router(state: State):
            if isinstance(state.messages[-1], AssistantMessage) and not state.messages[-1].tool_calls:
                return "end"
            return "agent"

        graph = GraphBuilder(state_schema=StateSchema)
        graph.add_node("agent", self.agent)
        graph.add_condition_edge(source="agent", router=repeat_fc_router,
            possible_targets={"end", "agent"}, group="self-evolve")
        graph.set_entry_point("agent")
        self.graph = graph.build()
