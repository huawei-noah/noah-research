# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import json
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Type

from pydantic import BaseModel, Field, model_validator

from ..agent import AgentNode
from ..factory import BaseComponent
from ..graph import GraphBuilder
from ..tool import ToolManager
from ..typing import AssistantMessage, State, StateMessage
from ...logger import get_logger


logger = get_logger()


class Swarm(BaseComponent):
    """
    A swarm component that automatically builds and compiles a "Swarm" style multi-agent collaboration graph.

    Core functionalities:
    1. Receives a mapping from agent names to agent configurations or instances.
    2. Supports dynamically adding or removing agents before building the graph.
    3. Supports defining the communication topology between agents via the `edges` parameter (which agents can handsoff to which agents).
    4. Dynamically creates customized 'handoff' tools and injects them into each relevant agent.
    5. Uses GraphBuilder to build a graph where each agent can 'handoff' to other agents according to the defined topology.
    6. Sets up general routing logic to interpret 'handoff' calls and guide the flow.
    7. Provides a `build()` method that returns a compiled, runnable Graph instance.
    """

    agents: Dict[str, AgentNode] = Field(
        description="A dictionary where keys are unique agent names (str) and values are AgentNode instances or their configurations."
    )

    state_schema: Type[BaseModel] = Field(
        description="A Pydantic model for the graph state."
    )

    entry_point_agent: str = Field(
        description="The name of the entry point agent for the Swarm. This name must exist among the keys of the agents dictionary."
    )

    edges: Optional[List[Tuple[str, str]]] = Field(
        default=None,
        description="A list of directed edges defining allowed 'handoff' paths between agents. Each tuple is formatted as (source_agent_name, target_agent_name). If None, any agent is allowed to 'handoff' to any other by default."
    )

    max_turns: int = Field(
        default=20,
        description="Maximum number of turns for graph execution, used to prevent infinite loops."
    )

    termination_pattern: str = Field(
        default="FINISHED",
        description="Output keywords if any nodes need to end the conversation,"
    )

    _agents_instances: Dict[str, AgentNode] = {}
    _agent_names: List[str] = []

    @model_validator(mode="after")
    def _validate_initial_config(self) -> 'Swarm':
        """
        During Pydantic model instantiation, only validate that `entry_point_agent` is in the initial `agents` list.
        """
        initial_agent_keys = list(self.agents.keys())
        if self.entry_point_agent not in initial_agent_keys:
            raise ValueError(
                f"Entry point '{self.entry_point_agent}' not found in the initially provided agents list. "
                f"Available Agents: {initial_agent_keys}"
            )
        return self

    def add_agent(self, name: str, agent: AgentNode):
        """
        Dynamically add an agent. This change will take effect the next time the `build()` method is called.

        Args:
            name: The unique name of the agent.
            agent: An instance of AgentNode or its configuration.
        """
        if name in self.agents:
            raise ValueError(f"Agent named '{name}' already exists.")
        self.agents[name] = agent
        logger.info(f"Agent '{name}' has been added. Please re-call the build() method to apply the changes.")

    def remove_agent(self, name: str):
        """
        Dynamically remove an agent. This change will take effect the next time the `build()` method is called.

        Args:
            name: The name of the agent to be removed.
        """
        if name not in self.agents:
            raise ValueError(f"Agent named '{name}' does not exist.")
        if name == self.entry_point_agent:
            raise ValueError(
                f"Cannot delete the entry point agent '{name}'. If you must delete it, please change the `entry_point_agent` attribute first.")
        del self.agents[name]
        logger.info(f"Agent '{name}' has been removed. Please re-call the build() method to apply the changes.")

    def _prepare_internal_state(self):
        """
        Prepare the internal state of the Swarm, including instantiating all agents and validating the configuration.
        This method is called at the beginning of `build()` to ensure the latest agent configurations are used.
        """
        validated_agents = {}
        for name, agent in self.agents.items():
            validated_agents[name] = agent

        self._agents_instances = validated_agents
        self._agent_names = list(self._agents_instances.keys())

        if self.entry_point_agent not in self._agent_names:
            raise ValueError(
                f"Entry point '{self.entry_point_agent}' not found in the current agents list. "
                f"Available agents: {self._agent_names}"
            )

        if self.edges:
            for source, target in self.edges:
                if source not in self._agent_names:
                    raise ValueError(f"In edges definition, source agent '{source}' does not exist in the agent list.")
                if target not in self._agent_names:
                    raise ValueError(f"In edges definition, target agent '{target}' does not exist in the agent list.")

    def build(self):
        """
        Build and compile the Swarm graph.
        This method uses the current `agents` and `edges` configuration to generate a new graph.
        """
        # Step 1: Prepare internal state to ensure configuration is up-to-date
        self._prepare_internal_state()

        # Step 2: Create and inject customized handoff tools for each agent
        self._inject_handoff_tools()

        # Step 3: Initialize GraphBuilder
        graph_builder = GraphBuilder(state_schema=self.state_schema)

        # Step 4: Add all agent nodes
        for name, node in self._agents_instances.items():
            graph_builder.add_node(name, node, action_mode="any")

        # Step 5: Add conditional routing edges for each node
        possible_targets: Set[str] = set(self._agent_names) | {"end"}
        for name in self._agent_names:
            router_func = self._create_router(name)
            graph_builder.add_condition_edge(
                source=name,
                router=router_func,
                possible_targets=possible_targets
            )

        # Step 6: Set entry point and build the graph
        graph_builder.set_entry_point(self.entry_point_agent)
        graph = graph_builder.build(max_turn=self.max_turns)

        logger.info("Swarm Graph built successfully.")
        logger.info(f"  - Agents: {self._agent_names}")
        logger.info(f"  - Entry point: {self.entry_point_agent}")
        logger.info(f"  - State schema: {self.state_schema.__name__}")
        if self.edges:
            logger.info(f"  - Connection topology (Edges): {self.edges}")
        else:
            logger.info("  - Connection topology: Fully connected (All-to-All)")

        return graph

    def _get_allowed_targets(self, agent_name: str) -> List[str]:
        """Get the list of allowed handoff targets for a specified agent based on the `edges` configuration."""
        if self.edges is None:
            # Default fully connected, excluding the agent itself
            return [name for name in self._agent_names if name != agent_name]

        # Filter targets for the current agent based on the `edges` list
        targets = [target for source, target in self.edges if source == agent_name]
        # Remove duplicates while preserving order
        return list(dict.fromkeys(targets))

    def _create_handoff_tool(self, agent_name: str) -> Optional[Callable]:
        """
        Dynamically create a 'handoff' tool function for the specified Agent based on the current connection topology.
        Returns None if the Agent has no valid handoff targets.
        """
        allowed_targets = self._get_allowed_targets(agent_name)
        if not allowed_targets:
            return None

        AgentNamesLiteral = Literal[tuple(allowed_targets)]

        def handoff(target_agent: AgentNamesLiteral, info: str):
            """
            Hand off the current task to another specified Agent for processing.

            Args:
                target_agent: The name of the target Agent to hand off to. Must be one of: {', '.join(allowed_targets)}
                info: A clear and specific description of the task to be handed off, including all necessary context information.
            """
            return f"Task is ready to be handed off to {target_agent} with information: {info}"

        handoff.__doc__ = f"""
            Hand off the current task to another specified Agent for processing.

            Args:
                target  Must be one of: {allowed_targets}
                info: A clear and specific description of the task to be handed off, including all necessary context information.
            """
        return handoff

    def _inject_handoff_tools(self):
        """Inject customized 'handoff' tools into each relevant Agent."""
        for agent_name, agent_node in self._agents_instances.items():
            # Create a dedicated handoff tool for the current agent
            handoff_tool = self._create_handoff_tool(agent_name)

            # If there are no valid handoff targets, do not inject the tool
            if not handoff_tool:
                continue

            tool_manager_list = agent_node.tool_manager
            injected = False

            # Iterate through the agent's tool_manager list for injection
            for tm_config in tool_manager_list:
                if isinstance(tm_config, ToolManager):
                    # Ensure idempotency: remove the old handoff tool first, then add the new one
                    if "handoff" in tm_config._tool_map:
                        tm_config.delete_tools(['handoff'])
                    tm_config.add_callable_tools([handoff_tool])
                    injected = True
                    break

            # If the agent doesn't have any ToolManager, create a new one for it
            if not injected:
                new_tm_lazy_instance = ToolManager(tools=[handoff_tool])
                agent_node.tool_manager.append(new_tm_lazy_instance)

    @staticmethod
    def _get_last_assistant_message(messages: List[StateMessage]) -> Optional[AssistantMessage]:
        for msg in reversed(messages):
            if isinstance(msg, AssistantMessage):
                return msg
        return None

    def _create_router(self, current_agent_name: str) -> Callable[[State], str]:
        def generic_router(state: State) -> str:
            last_msg = self._get_last_assistant_message(state.messages)
            if not last_msg:
                return "end"

            if self.termination_pattern in last_msg.content:
                return "end"

            if last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    if tool_call.function.name == "handoff":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            target = args.get("target_agent")
                            if target in self._agent_names:
                                logger.info(f"Router: Detected handoff from '{current_agent_name}' to '{target}'.")
                                return target
                        except (json.JSONDecodeError, AttributeError):
                            pass

            return current_agent_name

        return generic_router
