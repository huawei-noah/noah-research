# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr, SkipValidation

from ._edge import EdgeSpecBase, EdgeSpec
from ._node import GraphNodeSpec
from ._plot import graph_to_mermaid, render_mermaid_as_html
from ._state import StateCkpt
from ._streaming import StreamWriter
from ..factory import BaseComponent
from ..typing import DEFAULT_EDGE_GROUP, NodeActionMode, SpecialNode, State, StateDelta, StateSchema
from ...logger import get_logger

logger = get_logger()


class RunTimeTask(BaseModel):
    node_name: str
    """target node name"""

    state_ckpt: Union[StateCkpt, List[StateCkpt]]
    """State ckpt"""

    edge_group: str
    """Which group of current edge"""

    predecessor: Optional[Union[str, List[str]]] = None
    """Predecessor node name"""

    state_filter: Optional[Callable] = None
    """State filter function in edge"""

    trace_route: List[str] = Field(default_factory=list)
    """Node execution trace"""


# for RuntimeTask.state_filter ser/deser
RunTimeTask.serialize_state_filter = EdgeSpec.serialize_state_filter
RunTimeTask.deserialize_state_filter = EdgeSpec.deserialize_state_filter


class GraphEngine(BaseComponent):
    nodes: Dict[str, GraphNodeSpec] = Field(
        default_factory=dict,
        description="Graph node map")

    edges: Dict[str, List[EdgeSpecBase]] = Field(
        default_factory=dict,
        description="Edge node map")

    state_schema: Optional[SkipValidation[type[StateSchema]]] = Field(
        default=None,
        description="State schema definition"
    )

    max_turn: Optional[int] = Field(
        default=None,
        description="max_turn: maximum number of node invocations allowed for this graph."
                    "If the count exceeds the number of nodes, the run will terminate immediately."
                    "Outputs that have already reached the END node remain accessible."
                    "If none, no limits are applied."
    )

    timeout: Optional[int] = Field(
        default=None,
        description="Sets the timeout duration for node execution.")

    _is_running: bool = PrivateAttr(default=False)
    _queue: asyncio.Queue = PrivateAttr(default_factory=asyncio.Queue)
    _output_channels: asyncio.Queue = PrivateAttr(default_factory=asyncio.Queue)
    _state_root: StateCkpt = PrivateAttr(init=False)
    _waiting_inputs: Dict[str, Dict[str, Dict[str, RunTimeTask]]] = PrivateAttr(init=False)
    _node_exec_cnt: int = PrivateAttr(default=0)

    def model_post_init(self, __context: Any):
        self.reset()

    def _check_can_running(self):
        if self._is_running:
            raise RuntimeError("This graph is still running, cannot reset.")
        if self.state_schema is None:
            raise RuntimeError("A schema of state must be assigned.")

    def _analyze_graph(self):
        config = dict()

        for name, edges in self.edges.items():
            for edge in edges:
                edge: EdgeSpecBase
                for target in edge.get_possible_targets():
                    if SpecialNode.is_special_node(target) or self.nodes[target].action_mode == NodeActionMode.ANY:
                        continue
                    config.setdefault(target, {}).setdefault(edge.group, {})

                    config[target][edge.group][name] = None

        for target, edge_group in config.items():
            for edge_group_name, sources in edge_group.items():
                for source in sources:
                    if target == source and len(config[target][edge_group_name]) > 1:
                        raise RuntimeError(f"When a node has a self-loop and its action mode is set to “all,” "
                                           f"you must either assign a distinct edge_group to the self-loop edge or "
                                           f"change the node’s behavior mode to “any.” Otherwise, "
                                           f"the node will never be triggered. Self-loop node is {target}!")

        return config

    def _is_predecessor_all_ready(self, runtime_task: RunTimeTask):
        if runtime_task.node_name not in self._waiting_inputs:
            return True

        self._waiting_inputs[runtime_task.node_name][runtime_task.edge_group][runtime_task.predecessor] = runtime_task

        if all([x is not None for x in self._waiting_inputs[runtime_task.node_name][runtime_task.edge_group].values()]):
            return True
        return False

    def _get_node_inputs(self, runtime_task: RunTimeTask) -> Union[List[RunTimeTask], RunTimeTask]:
        if runtime_task.node_name not in self._waiting_inputs:
            return runtime_task

        return list(self._waiting_inputs[runtime_task.node_name][runtime_task.edge_group].values())

    def _init_state_ckpt(self, inputs: Dict):
        return StateCkpt(delta=inputs, parent=self._state_root, state_schema=self.state_schema)

    def _filter_state(self, runtime: RunTimeTask):
        if runtime.state_filter:
            runtime.state_ckpt = StateCkpt.filter(runtime.state_ckpt, runtime.state_filter)
        return runtime

    def _get_merge_strategy(self, node_name: str, edge_group: str):
        mims = self.nodes[node_name].multi_input_merge_strategy
        if mims is None:
            return mims
        return mims[edge_group]

    def _merge_state(self, runtimes: List[RunTimeTask]):
        runtimes = [self._filter_state(runtime) for runtime in runtimes]

        ckpt = StateCkpt.merge(
            [x.state_ckpt for x in runtimes],
            self._get_merge_strategy(runtimes[0].node_name, runtimes[0].edge_group)
        )
        return RunTimeTask(
            node_name=runtimes[0].node_name,
            edge_group=runtimes[0].edge_group,
            predecessor=[x.predecessor for x in runtimes],
            state_filter=None,
            state_ckpt=ckpt
        )

    async def _process_node(
            self, runtime_task: RunTimeTask
    ) -> List[RunTimeTask]:
        """Process current node and return successor nodes"""
        node_name = runtime_task.node_name
        try:
            if self.max_turn and self._node_exec_cnt >= self.max_turn:
                await self._output_channels.put(runtime_task)
                logger.info(f"Maximum invocation count exceeded: {self._node_exec_cnt}")
                return []
            self._node_exec_cnt += 1
            node = self.nodes[node_name]
            # get all runtime value from all predecessors and restore full state
            input_runtimes = self._get_node_inputs(runtime_task)
            if isinstance(input_runtimes, list):
                runtime = self._merge_state(input_runtimes)
            else:
                runtime = self._filter_state(input_runtimes)

            if SpecialNode.is_end_node(node_name):
                await self._output_channels.put(runtime)
                return []

            full_state = runtime.state_ckpt.materialize()
            state_delta = await node(full_state)
            new_state_ckpt = StateCkpt(delta=state_delta, parent=runtime.state_ckpt, state_schema=self.state_schema)
            next_tasks = []
            for edge in self.edges[node_name]:
                targets = edge.get_targets(new_state_ckpt.materialize())
                for target, state_filter in targets:
                    next_tasks.append(
                        RunTimeTask(
                            node_name=target,
                            state_ckpt=new_state_ckpt,
                            edge_group=edge.group,
                            state_filter=state_filter,
                            predecessor=node_name
                        )
                    )

            return next_tasks
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"GraphEngine encountered an error during processing node: {node_name}") from e

    def _update_waiting_inputs(self, runtime_task: RunTimeTask):
        if runtime_task.node_name not in self._waiting_inputs:
            return
        self._waiting_inputs[runtime_task.node_name][runtime_task.edge_group][
            runtime_task.predecessor] = runtime_task

    def _get_node_action_mode(self, node_name):
        return self.nodes[node_name].action_mode

    async def run(self, inputs: Dict):
        try:
            self._check_can_running()
            self.reset()
            self._is_running = True
            running_queue: List[RunTimeTask] = [RunTimeTask(
                node_name=SpecialNode.START_NODE.value,
                state_ckpt=self._init_state_ckpt(inputs),
                edge_group=DEFAULT_EDGE_GROUP,
                state_filter=None
            )]
            batch_cnt = 0
            while running_queue:
                batch_cnt += 1
                # This is where we will collect tasks for the parallel run
                results_from_gather = await asyncio.wait_for(
                    asyncio.gather(
                        *[self._process_node(runtime_task)
                          for runtime_task in running_queue],
                    ),
                    timeout=self.timeout
                )

                next_batch_candidates = []
                for runtime_task_list in results_from_gather:
                    for runtime_task in runtime_task_list:
                        if self._get_node_action_mode(runtime_task.node_name) == NodeActionMode.ALL:
                            self._update_waiting_inputs(runtime_task)
                        if self._is_predecessor_all_ready(runtime_task):
                            next_batch_candidates.append(runtime_task)
                running_queue = next_batch_candidates

            self._is_running = False
            return await self.get_output()
        except asyncio.CancelledError:
            logger.error(f"Task was cancelled!")
            raise
        except Exception as e:
            raise e
        finally:
            self._is_running = False

    async def __call__(self, state: State, stream_writer: StreamWriter) -> StateDelta:
        return await self.run(state)

    def reset(self):
        self._check_can_running()
        self._state_root = StateCkpt(delta=None, parent=None, state_schema=self.state_schema)
        self._queue = asyncio.Queue()
        self._waiting_inputs = self._analyze_graph()
        self._output_channels: asyncio.Queue[RunTimeTask] = asyncio.Queue()
        self._node_exec_cnt = 0

    async def get_output(self):
        if self._output_channels.empty():
            return None

        ckpts = []
        while not self._output_channels.empty():
            ckpts.append(self._output_channels.get_nowait().state_ckpt)

        return StateCkpt.merge(ckpts).materialize()

    def draw_graph(
            self,
            save_path: str = None,
            auto_open: bool = True,
    ):
        """
        Generating graphs in HTML formats.
        When auto_open is True, the generated HTML page will be opened automatically.
        If save_path is None, no file will be saved.

        Args:
            save_path (str): Path to save the file. If None, the file will not be saved.
            auto_open (bool): Whether to automatically open the generated HTML file (only effective when render='html'). Default is True.

        """
        mermaid_code = graph_to_mermaid(
            self.nodes,
            self.edges
        )

        render_mermaid_as_html(mermaid_code, save_path, auto_open)
