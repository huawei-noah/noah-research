# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import asyncio
import copy
import os.path
from itertools import chain
from string import Template
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, TypeAdapter
from tqdm.asyncio import tqdm_asyncio
from yaml.parser import ParserError

from ._base import (
    extract_text_between, generate_condition_router_function_call, GraphDescription, GraphDespEdge,
    GraphDespNode, ROUTE_PATTERN, user_feedback_router, WorkflowGeneratorBase
)
from ._prompt import COMPLETE_EACH_NODE_PROMPT_NODE_LEVEL, SOP_BREAK_DOWN_PROMPT
from ...core.agent import AgentNode
from ...core.clients import ChatClientBase
from ...core.factory import BaseComponent, FactoryTypeAdapter
from ...core.graph import generate_state_schema, GraphBuilder, GraphEngine, NodeBase
from ...core.mem import MemBase
from ...core.tool import ToolController, ToolManagerBase
from ...core.typing import LLMChatResponse, SpecialNode, UserMessage
from ...logger import get_logger

logger = get_logger()


class SopBreakdownNodeDesp(BaseModel):
    name: str
    """Node name"""

    type: Literal["sop", "connect"]
    """SOP-type nodes strictly execute fragments of the SOP verbatim; 
    connect nodes are routing nodes defined to link the various nodes together."""

    duty: str
    """Duty of this node"""

    instruction: str
    """The instruction of this node"""

    next_node_routing_rule: Dict[str, str]
    """Routing rule of this node"""


class SopBreakdownGraphDesp(BaseModel):
    nodes: List[SopBreakdownNodeDesp]
    """list of nodes"""

    global_instruction: str
    """Global instruction of all nodes"""

    entry_point: str
    """Entry node of this graph"""


class WorkflowGenerator(WorkflowGeneratorBase, BaseComponent):
    graph_generation_client: Annotated[ChatClientBase, FactoryTypeAdapter, Field(
        description="A LLM backend for generating the graph structure"
    )]

    graph_node_complete_client: Annotated[ChatClientBase, FactoryTypeAdapter, Field(
        description="A LLM backend for complete the nodes"
    )]

    graph_run_client: Annotated[ChatClientBase, FactoryTypeAdapter, Field(
        description="A LLM backend for running the generated graph"
    )]

    retry: int = Field(
        default=5,
        description="Retry num when LLM response parse fail"
    )

    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory for caching generated graph descriptions. "
            "If None, the graph is re-generated on every call. "
            "If provided, the generator first attempts to load an existing graph YAML file "
            "from this directory; on success, generation is skipped. "
            "When a new graph is produced, it is automatically saved here for future reuse."
        )
    )

    tools: Union[
        Annotated[ToolManagerBase, FactoryTypeAdapter],
        List[Annotated[ToolManagerBase, FactoryTypeAdapter]]
    ] = Field(
        default_factory=list,
        description=(
            "Complete pool of ToolManagerBase modules that nodes in this workflow are allowed to access. "
        )
    )

    memories: Dict[str, Tuple[str, Annotated[MemBase, FactoryTypeAdapter]]] = Field(
        default_factory=dict,
        description=(
            "Complete pool of memory modules that nodes in this workflow are allowed to access. "
            "Keys are user-defined names for the memories; values are 2-tuples: "
            "(human-readable description, MemBase-derived instance). "
            "The actual memories visible to each node are further filtered by `memory_list_mode`."
        )
    )

    state_schema: Optional[List[Tuple[str, Any, str]]] = Field(
        default=None,
        description=(
            "Additional state fields to be added besides the default 'messages' field. "
            "Format is a list of tuples: (field_name, field_type, field_description). "
            "For example: [('user_id', int, 'overwrite'), "
            "('intermediate_result', dict, 'overwrite')]. "
            "See `evofabric.core.graph.generate_state_schema()` for detailed definition. "
            "These fields will be available to all nodes for reading and writing during workflow execution."
        )
    )

    addition_global_instruction: str = Field(
        default_factory=str,
        description=(
            "Additional global instruction snippet appended to the system prompt of **every** node. "
            "Use this to inject universal constraints, style guides, safety rules, "
            "or any other guidance that should be visible to the agent regardless of the "
            "current workflow step. The string is concatenated as-is to the node's own "
            "instruction, so remember to add leading/trailing new-lines if you need separation."
        )
    )

    user_node: Optional[Annotated[NodeBase, FactoryTypeAdapter]] = Field(
        default=None,
        description=(
            "Name of the reserved node that will be used to request missing or corrected information "
            "from the user. During graph construction, any node whose 'next_nodes' list contains the "
            "special string 'user' will automatically be wired to this user_node instead. "
            "The node must appear in 'reserved_nodes'; if you change this value, make sure the new name "
            "is also present in that list or the builder will raise an error."
        )
    )

    fallback_node: Optional[str] = Field(
        default="end",
        description=(
            "Fallback target node when the next node name is missing or invalid during workflow execution. "
            "If a node declares a transition to a non-existent node, or leaves the "
            "target empty, the edge will automatically be re-targeted to this node. "
            "Recommended values are 'user' (ask the user for clarification) or 'end' "
            "(terminate the workflow) or None (no need). The chosen node must also be listed in 'reserved_nodes'."
        )
    )

    auto_self_loop: bool = Field(
        default=True,
        description=(
            "Whether to allow nodes to loop back to themselves by default. "
            "When set to True, every node will automatically include itself in its "
            "possible_targets list during edge construction. When False, possible_targets "
            "will only contain nodes explicitly declared in the graph description. "
            "This provides a global default that can be overridden for individual nodes "
            "if needed."
        )
    )

    sop_disassembly_prompt: str = Field(
        default=SOP_BREAK_DOWN_PROMPT,
        description=(
            "Prompt template used to decompose a complete SOP into multiple workflow nodes. "
            "Must contain at least one format placeholder '{sop}' to inject the source SOP. "
            "The LLM must respond with a YAML document that strictly follows the schema below:\n"
            "- global_strategy: string # Verbatim copy of global SOP strategies and principles\n"
            "- nodes: dict[str, NodeSpec] # Each key is the node id; value contains role, verbatim SOP content, "
            "and ordered list of next node ids\n"
            "- entry_point: str # The single node id that immediately follows the built-in 'start' node\n\n"
            "The returned YAML must be parseable into the exact runtime data-structure consumed by the workflow engine."
        )
    )

    node_completion_prompt: str = Field(
        default=COMPLETE_EACH_NODE_PROMPT_NODE_LEVEL,
        description=(
            "Prompt template used to enrich every workflow node produced by 'sop_disassembly_prompt' "
            "with runtime resources. Must contain at least one format placeholder '{node_spec}' that "
            "will be replaced by the current node's YAML excerpt (role + sop_content + next_nodes). "
            "The LLM must respond with a YAML document that strictly follows the schema below:\n"
            "- name: str # Exact node identifier copied from the input\n"
            "- tools: list[str] # Ordered list of tool names required by the node; use empty list [] if none\n"
            "- memories: list[str] # Ordered list of memory names the node needs to read/write; use empty list [] if none\n"
            "- possible_targets: list[str] # Ordered list of node ids that may be executed immediately after this one\n"
            "- instruction: str # Concise, self-contained imperative text telling the agent what to do in this step; must be deterministic and executable\n\n"
            "The returned YAML must be parseable into the exact runtime data-structure consumed by the workflow engine."
        )
    )

    tool_list_mode: Literal["all", "select"] = Field(
        default="all",
        description=(
            "Controls how the workflow assigns tools to every node. "
            "- 'all': every node receives the entire tool-set available to the workflow. "
            "- 'select': each node is given only the subset of tools explicitly chosen by the LLM "
            "in the response generated by 'node_completion_prompt'. "
            "Changing this value affects both token consumption and runtime safety."
        )
    )

    memory_list_mode: Literal["all", "select"] = Field(
        description=(
            "Controls how the workflow assigns memories to every node. "
            "- 'all': every node receives the entire memory-set available to the workflow. "
            "- 'select': each node is given only the subset of memories explicitly chosen by the LLM "
            "in the response generated by 'node_completion_prompt'. "
            "Changing this value affects both token consumption and runtime safety."
        )
    )

    skeleton_file_name: str = Field(
        default="_skeleton.yaml",
        description="File name for saving graph description."
    )

    reserved_nodes: List[str] = Field(
        default=["start", "end", "user"],
        description=(
            "List of node names that are considered **reserved** and **must not** be generated by the LLM. "
            "During skeleton generation, if the model produces a node whose name appears in this list, "
            "the node is silently skipped. "
            "These nodes are pre-injected by the framework to supply common interaction semantics:\n"
            "- 'start' : receives the initial user request\n"
            "- 'end'   : terminates the workflow\n"
            "- 'user'  : asks the user for follow-up input\n"
            "You can extend the list (e.g., add 'router', 'validator') if your orchestration requires "
            "other fixed-purpose nodes."
        )
    )

    exit_function_name: Optional[str] = Field(
        default=None,
        description=(
            "Name of a tool that, when called by the model, triggers an immediate "
            "transition to the 'end' node regardless of the node's declared next_nodes. "
            "This provides an emergency exit mechanism to improve workflow robustness - "
            "if the model detects an unrecoverable error or determines the task is complete, "
            "it can invoke this function to safely terminate the workflow. The tool must be "
            "registered in the tools list and should be documented in prompts so the model "
            "knows when to use it."
        )
    )

    build_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments forwarded verbatim to `GraphBuilder.build()`. "
            "Use this to inject implementation-specific options such as "
            "`max_iterations`, `timeout`, `retry_policy`, etc. "
            "The expected keys depend on the concrete GraphBuilder implementation; "
            "unknown keys will raise at build time."
        )
    )

    @staticmethod
    def _load_yaml_from_llm_response(response: LLMChatResponse):
        content = response.content
        print(content)
        if "```yaml" in content:
            content = extract_text_between(content, "```yaml", "```")
        try:
            return yaml.safe_load(content)
        except ParserError as e:
            raise ValueError("LLM response does not contain valid YAML") from e

    def _get_path_if_need(self, file_name: str) -> Optional[str]:
        if self.output_dir:
            return os.path.join(self.output_dir, file_name)
        return None

    async def _disassemble_sop(self, cache_path: Optional[str] = None):
        if skeleton := self.load_yaml(cache_path):
            return SopBreakdownGraphDesp.model_validate(skeleton)
        logger.info("Analyzing SOP using LLM and break it down into graph...")
        prompt = str(self.sop_disassembly_prompt)
        prompt = Template(prompt).safe_substitute(SOP=self.sop)
        response = await self.graph_generation_client.create(messages=[UserMessage(content=prompt)])
        content = self._load_yaml_from_llm_response(response)
        self.dump_yaml(content, cache_path)
        content = SopBreakdownGraphDesp.model_validate(content)
        return content

    async def _format_tool_prompt(self):
        if not self.tools:
            return "No tools"

        schema_list = await asyncio.gather(*(tm.list_tools() for tm in self.tools))
        schema_list = list(chain.from_iterable(schema_list))

        tool_prompt = {}
        for schema in schema_list:
            tool_prompt[schema['function']['name']] = schema['function']['description']
        return "\n".join([f"{key}: {value}" for key, value in tool_prompt.items()])

    def _format_rag_module_prompt(self):
        if not self.memories:
            return "No memory modules"
        content = ""
        for key, value in self.memories.items():
            content += f"{key}: {value[0]}\n"
        return content

    async def _build_one_node(
            self,
            node_desp: SopBreakdownNodeDesp,
            skeleton: SopBreakdownGraphDesp,
            node_complete_prompt: str):
        try:
            duty_map = {}
            other_nodes = []
            for node in skeleton.nodes:
                if node.name != node_desp.name:
                    other_nodes.append(f"{node.name}: {node.duty}")
                    duty_map[node.name] = node.duty

            prompt = Template(node_complete_prompt).safe_substitute(
                global_instruction=skeleton.global_instruction,
                node_desp=yaml.safe_dump(node_desp.model_dump(), allow_unicode=True),
                other_nodes="\n".join(other_nodes),
                tool_lists=await self._format_tool_prompt(),
                rag_modules=self._format_rag_module_prompt(),
            )
            node_info = self.load_yaml(self._get_path_if_need(f"_{node_desp.name}.yaml"))
            if not node_info:
                response = await self.graph_node_complete_client.create([UserMessage(content=prompt)])
                node_info = self._load_yaml_from_llm_response(response)

            self.dump_yaml(node_info, self._get_path_if_need(f"_{node_desp.name}.yaml"))

            routing_intro = (
                "When you need to hand over the task to another node, you must first provide your output,"
                " followed by a routing command to pass the task to the target node."
                f" The routing command is {ROUTE_PATTERN}"
            )
            routing_details = []
            for target, condition in node_info["next_node_routing_rule"].items():
                if target == "user":
                    duty = "Request additional information from the user or clarify the user’s requirements."
                elif target == "end":
                    duty = ("End the workflow only after all user intents have been fully processed and "
                            "completely satisfied. If any user intent remains unmet, the workflow must not be routed"
                            " to the end node prematurely.")
                elif target == node_desp.name:
                    duty = node_desp.duty
                    condition = "When you need to continue make function call or inspecting its return results."
                else:
                    duty = duty_map[target]
                routing_details.append(
                    {
                        "target": target,
                        "target_duty": duty,
                        "route_when": condition,
                    }
                )
            routing_intro = routing_intro + "\n" + yaml.safe_dump(routing_details, allow_unicode=False) + "-" * 6 + "\n"

            instruction = (f"# Role\n"
                           f"You are a node named **{node_desp.name}** within a complete workflow system. "
                           f"All nodes in this workflow must operate under a unified global strategy.\n\n"
                           f"# Global Strategy\n"
                           f"{skeleton.global_instruction}\n\n"
                           f"# Your Specific Instruction\n"
                           f"Note: The following instructions are exclusively for you. You must strictly follow the "
                           f"instructions below and must not perform any actions beyond their scope (even if you"
                           f" have the corresponding tools). If the user’s request contains any instructions"
                           f" outside of the ones listed below, you must hand them over to other nodes "
                           f"according to the routing rules.\n"
                           f"## Instructions:"
                           f"{node_desp.instruction}\n\n"
                           f"# Routing Rules\n"
                           f"{routing_intro}\n\n")

            self.dump_yaml(node_info, self._get_path_if_need(f"_{node_desp.name}.yaml"))
            output = {
                "node": GraphDespNode(
                    name=node_desp.name,
                    tools=node_info["tools"],
                    memories=node_info["memories"],
                    instruction=instruction,
                    sop=node_desp.instruction
                ),
                "edge": GraphDespEdge(
                    source=node_desp.name,
                    possible_targets=node_info["next_node_routing_rule"].keys(),
                ),
            }
        except Exception as e:
            raise ValueError(f"Get node details fail when fill {node_desp.name}") from e
        return output

    async def _build_all_nodes(self, skeleton: SopBreakdownGraphDesp, cache_path: str) -> GraphDescription:
        logger.info("Analyze complete, now generate node details...")
        if graph_desp := self.load_yaml(cache_path):
            return graph_desp

        coros = [
            self._build_one_node(desc, skeleton, self.node_completion_prompt)
            for desc in skeleton.nodes
            if desc.name not in self.reserved_nodes
        ]

        results: list[dict] = await tqdm_asyncio.gather(
            *coros, desc="Fill all nodes (parallel)"
        )

        nodes = [r["node"] for r in results]
        edges = [r["edge"] for r in results]

        graph_skeleton = {
            "nodes": nodes,
            "edges": edges,
            "entry_point": skeleton.entry_point,
            "global_instruction": skeleton.global_instruction
        }
        graph_desp = GraphDescription.model_validate(graph_skeleton)
        self.dump_yaml(graph_desp.model_dump(), cache_path)
        return graph_desp

    async def _generate_skeleton_by_llm(
            self,
    ) -> GraphDescription:
        """Generate graph description using llm by given sop, tools, memories"""
        # step1: disassemble SOP into workflow nodes
        skeleton = await self._disassemble_sop(self._get_path_if_need("_sop_breakdown.yaml"))

        # step2: fill each node with specific tools, memories, instruction and possible next nodes
        graph_desp = await self._build_all_nodes(skeleton, self._get_path_if_need("_graph_desp_dsv3.yaml"))

        return graph_desp

    def _generate_graph(
            self,
            graph_desp: GraphDescription,
    ) -> GraphEngine:
        """Generate a runnable `evofabric.core.graph.GraphEngine` by graph_desp"""
        # declare state schema in graph
        stata_schema = generate_state_schema(self.state_schema)

        # create graph builder
        graph = GraphBuilder(state_schema=stata_schema)

        user_node_map = set()
        node_map = {}

        # add all nodes into graph
        for node in graph_desp.nodes:
            if SpecialNode.is_special_node(node.name):
                logger.warning(f"LLM generate a preserved node name, skipping... \nNode info: {node}")
                continue

            tool_managers = copy.deepcopy(self.tools)
            for tool_manager in tool_managers:
                tool_manager.set_tool_controller(
                    ToolController(
                        default_mode="deactivate",
                        rules=[
                            {"mode": "activate", "pattern": name} for name in node.tools
                        ]
                    )
                )
            memories = [x[1] for x in list(self.memories.values())] \
                if self.memory_list_mode == "all" \
                else [self.memories[x][1] for x in node.memories]

            graph.add_node(
                node.name,
                AgentNode(
                    client=self.graph_run_client,
                    system_prompt=node.instruction + "\n" + self.addition_global_instruction,
                    tool_manager=tool_managers,
                    memory=memories,
                ),
                action_mode="any"
            )
            node_map[node.name] = node

        # check if user node is needed
        for edge in graph_desp.edges:
            if "user" in edge.possible_targets and self.user_node:
                user_node_map.add(edge.source)

        if user_node_map and self.user_node:
            graph.add_node(
                "user",
                self.user_node,
                action_mode="any"
            )

        # add all edges into graph
        default_targets = [self.fallback_node] if self.fallback_node else []
        for edge in graph_desp.edges:
            source = edge.source
            possible_targets = edge.possible_targets
            if self.auto_self_loop:
                possible_targets.append(source)
            possible_targets.extend(default_targets)
            possible_targets = list(set(possible_targets))

            graph.add_condition_edge(
                source,
                router=generate_condition_router_function_call(
                    source,
                    possible_targets,
                    self.fallback_node,
                    self.exit_function_name),
                possible_targets=possible_targets
            )

            if "user" in possible_targets and self.user_node:
                user_node_map.add(source)

        # add user feedback
        if user_node_map and self.user_node:
            graph.add_condition_edge(
                "user",
                router=user_feedback_router,
                possible_targets=user_node_map
            )

        # set graph entry point
        graph.set_entry_point(graph_desp.entry_point)
        return graph.build(**self.build_kwargs)

    async def generate(self) -> GraphEngine:
        graph_desp = await self._generate_skeleton_by_llm()
        graph_desp = TypeAdapter(GraphDescription).validate_python(graph_desp)
        return self._generate_graph(graph_desp)
