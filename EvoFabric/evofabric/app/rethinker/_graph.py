# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import asyncio
import json
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type
from contextlib import nullcontext

from evofabric.app.rethinker.adapter import get_client
from evofabric.core.clients import ChatClientBase
from evofabric.core.graph import (
    generate_state_schema, GraphBuilder, NodeBase, StateUpdater,
    stream_writer_env, StreamCtx
)
from evofabric.core.typing import MISSING

from ._config import config
from .agent import CodingAgent
from .nodes import (
    ConfidenceGuideSelectNode, CriticWithRethinkNode, DispatchNode, get_dispatch_filter, GuidedSummaryNode,
    SolutionWithReThinkNode
)
from .prompts import FORCE_FINISH_PROMPTS
from .tools import execute_python_code

DISPATCH_NODE_NAME = "dispatch"
SELECTOR_NODE_NAME = "selector"
SOLUTION_LAYER_PREFIX = "solution_layer"
SOLUTION_SUMMARY_PREFIX = "solution_summary"
CRITIC_LAYER_PREFIX = "critic_layer"
SOLUTION_SUMMARY_KEY = "solution_summary"
RESULT_FILE_NAME = "result.json"


@StateUpdater.register(f"list_ele_overwrite_max_parallel")
def list_ele_overwrite(old: list = MISSING, new: list = MISSING):
    """
    Element-wise overwrite of a fixed-length list with a maximum parallel size.

    This function merges two lists (`old` and `new`) into a result list whose
    length is defined by `config.structure.num_parallel`. The merge follows an
    element-wise overwrite rule:
    - Elements from `old` are copied to the result first (if provided).
    - Elements from `new` overwrite the corresponding positions in the result
      only when the new element is not None.

    Args:
        old: The previous list state. If MISSING, it is ignored.
        new: The new list state. If MISSING, it is ignored.

    Returns:
        A list of length `config.structure.num_parallel` after element-wise overwrite.
    """
    result = [None] * config.structure.num_parallel

    if old is not MISSING:
        for index, old_ele in enumerate(old):
            result[index] = old_ele

    if new is not MISSING:
        for index, new_ele in enumerate(new):
            if new_ele is not None:
                result[index] = new_ele
    return result


def get_agent(client: ChatClientBase, finish_type: Literal['answer', 'select']):
    """Build coding agent due to configs"""
    return CodingAgent(
        client=client,
        max_agent_step=config.solution.max_agent_step,
        max_empty_response=config.solution.max_empty_response,
        tool_timeout=config.solution.tool_timeout,
        force_finish_prompt_candidates=FORCE_FINISH_PROMPTS[finish_type],
        tool_content_pattern=config.solution.stop_condition,
        answer_pattern=config.solution.answer_condition if finish_type == 'answer' else config.solution.selection_condition,
        chat_template=config.solution.chat_template,
        py_exec_handler=execute_python_code
    )


def _add_sequential_nodes(
    model_name: str,
    num_layers: int,
    node_prefix: str,
    node_class: Type[NodeBase],
    branch_index: int,
    start_node: str,
    node_init_kwargs: Dict,
    nodes_to_register: Dict[str, NodeBase],
    edges_to_register: List[Tuple[str, str]],
    branch_schemas: List[Tuple[str, str, str]],
) -> str:
    """
    Helper function to add a sequential layer of nodes within a branch.

    Args:
        num_layers (int): The number of node layers to add.
        node_prefix (str): The name prefix for the nodes.
        node_class (Type[NodeBase]): The node class to instantiate.
        branch_index (int): The index of the current parallel branch.
        start_node (str): The starting node to connect this sequence from.
        node_init_kwargs (Dict): Keyword arguments required for node class instantiation.
        nodes_to_register (Dict): The dictionary to register new nodes.
        edges_to_register (List): The list to register new edges.
        branch_schemas (List): The list to add branch state schema definitions.

    Returns:
        str: The name of the last node in this sequential layer.
    """
    last_node_name = start_node
    for i in range(num_layers):
        node_name = f"{node_prefix}{i}_{branch_index}"
        output_key = f"{node_prefix}{i}"

        # Dynamically update node initialization arguments
        current_kwargs = node_init_kwargs.copy()
        current_kwargs["output_key"] = output_key
        if i > 0:
            current_kwargs["last_round"] = f"{node_prefix}{i - 1}"
        else:
            # Ensure the first layer doesn't have a 'last_round' key
            current_kwargs.pop("last_round", None)

        current_kwargs["agent"] = get_agent(get_client(model_name), "answer")
        node = node_class(**current_kwargs)

        nodes_to_register[node_name] = node
        edges_to_register.append((last_node_name, node_name))
        branch_schemas.append((output_key, "list", "list_ele_overwrite_max_parallel"))
        last_node_name = node_name

    return last_node_name


def build_rethinker_graph():
    """
    Builds and configures a computation graph with a dynamic, parallel structure.

    Graph Structure Overview:
    --------------------------
    The graph is designed to process a query in multiple parallel branches to generate,
    refine, and select the best solution. The structure can be visualized as follows:

    1. Entry Point (Dispatch):
       - A single `dispatch` node serves as the entry point. It directs the initial
         state to all parallel branches.

    2. Parallel Processing Branches:
       - The graph splits into `N` identical, parallel branches (`N` is configured by
         `config.structure.num_parallel`).
       - Each branch is a sequential pipeline of nodes designed to iteratively
         improve a solution. The sequence within each branch is:

         a. Solution Stage: A chain of `SolutionWithReThinkNode`s. Each node
            generates or refines a set of solutions.
            (e.g., dispatch -> solution_0 -> solution_1 -> ...)

         b. Summary Stage: A single `GuidedSummaryNode` that summarizes the outputs
            from the Solution Stage.
            (e.g., ... -> solution_N -> summary)

         c. Critic Stage: A chain of `CriticWithRethinkNode`s that evaluate and
            critique the summary, providing feedback for refinement.
            (e.g., summary -> critic_0 -> critic_1 -> ...)

    3. Aggregation Point (Selector):
       - A single `selector` node (`ConfidenceGuideSelectNode`) acts as the
         aggregation point.
       - The final node from *each* parallel branch (the last critic node) is
         connected to this selector node.

    4. Final Output:
       - The `selector` node evaluates the outputs from all branches and chooses
         the best final result, completing the graph's execution.
    """
    if config.structure.num_parallel < 1:
        raise ValueError("Parallel must be at least 1.")

    state_schema_defines = [
        ("query", "str", "overwrite"),
        ("index", "int", "overwrite"),
    ]
    if config.exp.output_root is not None:
        state_schema_defines.append(("cache_dir", "str", "overwrite"))

    nodes_to_register: Dict[str, NodeBase] = {
        DISPATCH_NODE_NAME: DispatchNode(),
    }
    edges_to_register: List[Tuple[str, str]] = []

    final_nodes_before_selector = []

    for para_idx in range(config.structure.num_parallel):
        branch_schemas = []
        last_node = DISPATCH_NODE_NAME

        last_node = _add_sequential_nodes(
            model_name=config.solution.solver_model,
            num_layers=config.structure.num_solution_iteration,
            node_prefix=SOLUTION_LAYER_PREFIX,
            node_class=SolutionWithReThinkNode,
            branch_index=para_idx,
            start_node=last_node,
            node_init_kwargs={"last_round": None},
            nodes_to_register=nodes_to_register,
            edges_to_register=edges_to_register,
            branch_schemas=branch_schemas,
        )

        summary_node_name = f"{SOLUTION_SUMMARY_PREFIX}_{para_idx}"
        solution_input_key, _ = last_node.rsplit("_", maxsplit=1)
        node = GuidedSummaryNode(
            agent=None,  # summary node do not need coding agent
            client=get_client(config.solution.summary_model),
            input_key=solution_input_key,
            output_key=SOLUTION_SUMMARY_KEY
        )
        nodes_to_register[summary_node_name] = node
        edges_to_register.append((last_node, summary_node_name))
        branch_schemas.append((SOLUTION_SUMMARY_KEY, "list", "list_ele_overwrite_max_parallel"))
        last_node = summary_node_name

        last_node = _add_sequential_nodes(
            model_name=config.solution.critic_model,
            num_layers=config.structure.num_critic_iteration,
            node_prefix=CRITIC_LAYER_PREFIX,
            node_class=CriticWithRethinkNode,
            branch_index=para_idx,
            start_node=last_node,
            node_init_kwargs={
                "input_key": SOLUTION_SUMMARY_KEY,
                "last_round": None,
            },
            nodes_to_register=nodes_to_register,
            edges_to_register=edges_to_register,
            branch_schemas=branch_schemas,
        )

        final_nodes_before_selector.append(last_node)

        # Only add the branch's schema definitions to the main schema on the first iteration
        if para_idx == 0:
            state_schema_defines.extend(branch_schemas)

    critic_output_key, _ = final_nodes_before_selector[0].rsplit("_", maxsplit=1)
    selector_node = ConfidenceGuideSelectNode(
        agent=get_agent(get_client(config.solution.selector_model), "select"),
        input_key=critic_output_key,
        output_key=SELECTOR_NODE_NAME,
    )
    nodes_to_register[SELECTOR_NODE_NAME] = selector_node
    state_schema_defines.append((SELECTOR_NODE_NAME, "dict", "overwrite"))

    # Connect the final node of each branch to the Selector
    for node_name in final_nodes_before_selector:
        edges_to_register.append((node_name, SELECTOR_NODE_NAME))

    graph = GraphBuilder(state_schema=generate_state_schema(state_schema_defines))
    for node_name, node in nodes_to_register.items():
        graph.add_node(node_name, node, action_mode="all")

    first_nodes_in_branches = [f"{SOLUTION_LAYER_PREFIX}0_{i}" for i in range(config.structure.num_parallel)]

    for src, dst in edges_to_register:
        state_filter = None
        if src == DISPATCH_NODE_NAME and dst in first_nodes_in_branches:
            _, dst_para_idx = dst.rsplit("_", maxsplit=1)
            state_filter = get_dispatch_filter(int(dst_para_idx))
        graph.add_edge(src, dst, state_filter=state_filter)

    graph.set_entry_point(DISPATCH_NODE_NAME)
    return graph.build()


async def run_rethinker_graph(
    query: str,
    query_id: Optional[str] = None,
    semaphore: Optional[asyncio.Semaphore] = None
):
    """
    Run the rethinker execution graph for a single query.

    This function builds the rethinker graph, initializes per-query logging
    directories, injects query metadata into the stream context, and executes
    the graph asynchronously.

    If `config.exp.output_root` is set, the output files will be organized as:
        output_root/
            qid00001/
                node1.json
                node2.json
                ...
                result.json
            qid00002/
            ...
    Args:
        query (str): Input query to be processed by the rethinker graph.
        query_id (Optional[str]): Unique identifier for the query. If not
            provided, a UUID will be generated automatically.
        semaphore (asyncio.Semaphore): Semaphore to synchronize execution

    Returns:
        Any: The execution result returned by the rethinker graph.
    """
    async with semaphore if semaphore else nullcontext():
        query_id = query_id or str(uuid.uuid4())
        graph = build_rethinker_graph()

        run_kwargs = {"query": query}

        if config.exp.output_root:
            cache_dir = Path(config.exp.output_root) / config.exp.exp_name / query_id
            cache_dir.mkdir(parents=True, exist_ok=True)
            run_kwargs["cache_dir"] = str(cache_dir)
        else:
            cache_dir = None

        with stream_writer_env(StreamCtx(meta={"query_id": query_id})):
            result = await graph.run(run_kwargs)

        if cache_dir:
            with open(cache_dir / RESULT_FILE_NAME, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=4, ensure_ascii=False)
