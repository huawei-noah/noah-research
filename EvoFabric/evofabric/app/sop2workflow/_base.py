# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os.path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from ...core.factory import BaseComponent, safe_get_attr
from ...core.graph import GraphEngine
from ...logger import get_logger

logger = get_logger()


ROUTE_PATTERN = "::TO::{target}::"


def extract_text_between(text: str, start: str, end: str) -> Optional[str]:
    """Extract sub-string from [start] to [end]"""
    try:
        start_index = text.index(start) + len(start)
        end_index = text.index(end, start_index)
        return text[start_index:end_index]
    except ValueError:
        return None


def generate_condition_router_function_call(
        source: str,
        possible_targets: list,
        fallback_target: str = "end",
        exit_function_name: str = None):
    """
    Generate a condition-router function-call snippet that can be injected into
    the system prompt of a *decision* node.

    Parameters
    ----------
    source : str
        Name of the current (decision) node.
    possible_targets : list[str]
        Node names that are legal successors of `source`.
    fallback_target : str, optional (default="end")
        Node to jump to when the model returns an unknown or empty choice.
    exit_function_name : str | None, optional
        If provided, the model may alternatively call this function to
        trigger an immediate transition to the **end** node (emergency exit).
        The function must be registered in the tool pool and its name
        should match the value given to the workflow generator.

    Returns
    -------
    callable
        a router function
    """
    if fallback_target is None:
        raise ValueError(
            "fallback_target cannot be None. "
            "A fallback target is required when LLM routing fails to locate the next node, "
            "otherwise the system will unable to determine the subsequent processing path."
        )
    def router_func(state):
        last_msg = None
        last_assistant_msg = None
        for msg in reversed(state.messages):
            if last_msg is None:
                last_msg = msg
            if msg.role == "assistant":
                last_assistant_msg = msg
                break

        if last_assistant_msg.tool_calls:
            for tool_call in last_assistant_msg.tool_calls:
                if exit_function_name and tool_call.function.name == exit_function_name:
                    return "end"

        route_to = None
        if last_msg.role == "tool":
            route_to = source
        else:
            content = last_msg.content
            for target in possible_targets:
                if target == source:
                    continue
                if f"::TO::{target}::" in content:
                    route_to = target
                    break
        # if no matched route pattern, route to [fallback_target]
        route_to = route_to or fallback_target
        return route_to

    return router_func


def user_feedback_router(state):
    # find the last assistant message and route next target back to it
    messages = safe_get_attr(state, "messages")
    for msg in reversed(messages):
        if msg.role == "assistant":
            return msg.node_name
    logger.warning("[User feedback router] Cannot find user feedback target, route to end")
    return "end"


class GraphDespNode(BaseModel):
    """Description of a node"""

    name: str
    """node name"""

    tools: List[str]
    """list of tool names"""

    memories: List[str]
    """list of memory names"""

    instruction: str
    """Instruction of this node"""

    sop: Optional[str] = None
    """Sop chunk of building this node"""


class GraphDespEdge(BaseModel):
    """Description of an edge"""

    source: str
    """Start node name of this graph"""

    possible_targets: List[str]
    """list of possible target names (next nodes of this node in workflow)"""

    type: Literal["condition"] = "condition"


class GraphDescription(BaseModel):
    nodes: List[GraphDespNode]
    """List of nodes"""

    edges: List[GraphDespEdge]
    """List of edges"""

    entry_point: str
    """Entry point of this graph"""

    global_instruction: str
    """Instructions for all nodes"""


class WorkflowGeneratorBase(BaseComponent):
    sop: str = Field(
        description="The SOP for generating a workflow"
    )

    @staticmethod
    def load_yaml(file_path):
        if not file_path or not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def dump_yaml(data, file_path):
        if not file_path:
            return

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True)

    def generate(self) -> GraphEngine:
        """Generate a runnable `evofabric.core.graph.GraphEngine` using SOP"""
        raise NotImplementedError("A workflow generator must implement this generate() method.")
