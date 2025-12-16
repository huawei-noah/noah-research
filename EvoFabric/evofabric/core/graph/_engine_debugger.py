# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import sqlite3
import uuid
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, PrivateAttr

from ._edge import ConditionEdgeSpec, EdgeSpec, EdgeSpecBase
from ._engine import GraphEngine, RunTimeTask
from ._state import StateCkpt
from ..typing import DEFAULT_EDGE_GROUP, NodeActionMode, SpecialNode
from ...logger import get_logger

logger = get_logger()

BranchFinished = "branch_finished"


class TreeNode(BaseModel):
    """Represents a node in the RuntimeTaskTree"""

    node_name: str
    """node name"""

    task: Optional[RunTimeTask] = None
    """task of this node"""

    children: Dict[str, 'TreeNode'] = Field(default_factory=dict)
    """children of this node"""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """unique id for tree node"""

    _parent: Dict[str, 'TreeNode'] = PrivateAttr(default_factory=dict)
    """parent of this node"""

    def __repr__(self):
        parent_name = self._parent.keys() if self._parent else "None"
        return (f"TreeNode(uuid='{self.uuid[:8]}...', name='{self.node_name}', "
                f"parent='{parent_name}', children={list(self.children.keys())})")

    def __hash__(self) -> int:
        return hash(self.uuid)

    def __eq__(self, other):
        return self.uuid == other.uuid

    @property
    def parent(self):
        return self._parent

    def add_child(self, node: 'TreeNode'):
        self.children[node.node_name] = node
        node._parent = self


class RuntimeTaskTree(BaseModel):
    """
    A doubly-linked tree/graph structure for managing task execution flow.
    Supports backward traversal and branch pruning.
    """

    db_path: str = Field(default=".db_storage", exclude=True)
    """Persistence database save path"""

    _root: Optional[TreeNode] = PrivateAttr(default=None)
    """Root node of this tree"""

    _leaf_nodes: Dict[str, TreeNode] = PrivateAttr(default_factory=dict)
    """Stores all nodes that currently have no children"""

    _uuid_map: Dict[str, TreeNode] = PrivateAttr(default_factory=dict)
    """A quick lookup map from UUID to Node object"""

    _conn: Optional[sqlite3.Connection] = PrivateAttr(default=None)
    """DB Connection object"""

    def model_post_init(self, __context: Any) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def clear_tree(self):
        self._root = None
        self._leaf_nodes = {}
        self._uuid_map = {}

    def _create_tables(self):
        cur = self._conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trees (
            root_uuid TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
        """)
        self._conn.commit()

    def save_tree(self):
        """Save entire tree to database"""
        cur = self._conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO trees (root_uuid, data)
            VALUES (?, ?)
        """, ("HEAD", self._root.model_dump_json()))
        self._conn.commit()

    def load_tree(self) -> Optional[object]:
        """Load entire tree from database"""

        cur = self._conn.cursor()
        cur.execute("SELECT data FROM trees WHERE root_uuid = ?", ("HEAD",))
        row = cur.fetchone()
        if not row:
            return None
        data = row[0]
        self._root = TreeNode.model_validate_json(data)
        self._rebuild_tree_params()

    def _rebuild_tree_params(self):
        # rebuild leaf_nodes and _uuid_map
        def dfs(node: TreeNode):
            self._uuid_map[node.uuid] = node
            if len(node.children) == 0:
                self._leaf_nodes[node.uuid] = node
            else:
                for child in node.children.values():
                    dfs(child)

        dfs(self._root)

    def clear(self):
        """Clear database"""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM trees")
        self._conn.commit()

    def close(self):
        """Close database connection"""
        self._conn.close()

    def get_node_by_uuid(self, node_uuid: str) -> Optional[TreeNode]:
        """Helper to find a node by its UUID."""
        return self._uuid_map.get(node_uuid)

    def get_leaf_node_uuid_by_node_name(self, node_name: str) -> str:
        """Helper to find a leaf node uuid by its node name."""
        for node_uuid, node in self._leaf_nodes.items():
            if node.node_name == node_name:
                return node_uuid
        raise ValueError(f"No leaf node uuid found for node name {node_name}")

    def _traverse_depth_first(self, node: TreeNode) -> List[TreeNode]:
        """Internal DFS for name lookup sorting."""
        if not node:
            return []

        # Simple DFS traversal for guaranteed root-to-leaf order
        # This is a robust way to generate a list that respects tree order
        # and can be searched.
        result = [node]
        for child in node.children.values():
            result.extend(self._traverse_depth_first(child))
        return result

    def add_node(self, node_name: str, task: Any, parent_uuid: Optional[str] = None) -> TreeNode:
        """
        Adds a new leaf node to the tree.
        If parent_uuid is None, it sets the root node (only on first call).
        """
        new_node = TreeNode(node_name=node_name, task=task)
        self._uuid_map[new_node.uuid] = new_node

        if self._root is None:
            # First node becomes the root
            self._root = new_node
            self._leaf_nodes[new_node.uuid] = new_node
            return new_node

        parent_node = self.get_node_by_uuid(parent_uuid)
        if not parent_node:
            raise ValueError(f"Parent node with UUID '{parent_uuid}' not found.")

        # Establish bidirectional link
        new_node.parent[parent_node.uuid] = parent_node
        parent_node.children[new_node.uuid] = new_node

        # Update leaf nodes: parent is no longer a leaf
        if parent_node.uuid in self._leaf_nodes:
            self._leaf_nodes.pop(parent_node.uuid)

        # The new node is now the new leaf
        self._leaf_nodes[new_node.uuid] = new_node

        return new_node

    def merge_nodes(self, from_nodes: List[TreeNode], to_node: TreeNode):
        """
        merge nodes，focus on all mode node, when compute process is a -> b -> c and a -> d -> c, we must ensure c is
        the same runtime node, but runtime node c will only save 1 trace route, but this process will not influence backward
        """
        for from_node in from_nodes:
            if from_node.uuid != to_node.uuid:
                from_node_parents = from_node.parent.values()
                # source node parent add and merge -> rm origin node
                for from_node_parent in from_node_parents:
                    from_node_parent.children[to_node.uuid] = to_node
                    from_node_parent.children.pop(from_node.uuid)
                    # add to node parent
                    to_node.parent[from_node_parent.uuid] = from_node_parent
                #  rm leaf nodes
                if from_node.uuid in self._leaf_nodes:
                    self._leaf_nodes.pop(from_node.uuid)

    def backtrack_and_prune(self, node_uuid: str) -> Optional[TreeNode]:
        """
        Backtracks from the specified node (inclusive),
        effectively pruning all subsequent nodes and resetting the leaf/path.
        Returns the new leaf node (which is the specified node's parent).
        """
        target_node = self.get_node_by_uuid(node_uuid)
        if not target_node:
            logger.warning(f"Node with UUID '{node_uuid}' not found for backtracking.")

        # Prune logic:
        # 1. Recursively find and delete all subsequent nodes (children and their descendants)
        nodes_to_remove: List[TreeNode] = []

        def _collect_and_remove(_cl_node: TreeNode):

            # Collect children first
            for cld in _cl_node.children.values():
                _collect_and_remove(cld)

            # Now remove the node itself
            nodes_to_remove.append(_cl_node)

        # Start collection from the children of the target node
        for child in target_node.children.values():
            _collect_and_remove(child)

        # 2. Perform the actual removal from maps and sets
        for node in nodes_to_remove:
            self._uuid_map.pop(node.uuid, None)
            self._leaf_nodes.pop(node.uuid, None)

        # 3. Clear the children list of the target node (the new leaf)
        target_node.children.clear()

        # 4. Update leaf nodes: the target node is now a leaf
        self._leaf_nodes[target_node.uuid] = target_node

        return target_node

    def find_nodes_by_trace_route(self, trace_route: List[str]) -> TreeNode:
        """
         according to runtime task route for node seeking, runtime task route update to uuid
        """
        if trace_route[0] != self._root.uuid:
            raise ValueError(f"trace_route[0] is wrong!")
        cur_node = self._root
        trace_route = deque(trace_route)
        trace_route.popleft()
        while trace_route:
            next_node_uuid = trace_route.popleft()
            if next_node_uuid not in cur_node.children:
                raise ValueError(f"Node with UUID '{next_node_uuid=}' not found in {cur_node.children=}.")
            cur_node = cur_node.children[next_node_uuid]

        return cur_node

    def in_leaf(self, node_uuid: str) -> bool:
        return node_uuid in self._leaf_nodes

    # Helper to print the tree structure
    def print_tree(self, node: Optional[TreeNode] = None, level: int = 0):
        if node is None:
            node = self._root
            if not node:
                logger.info("Tree is empty.")
                return

        logger.info('  ' * level + f"- {node.node_name} ({node.uuid[:4]}...)")
        for child in node.children.values():
            self.print_tree(child, level + 1)


class GraphEngineDebugger(GraphEngine):
    db_file_path: str = Field(default=".state_storage.db")

    _trace_tree: RuntimeTaskTree = PrivateAttr(default=None)
    _bp_set: set = PrivateAttr(default=None)
    _inverse_graph: Dict[str, Set[str]] = PrivateAttr(default=None)
    _waiting_inputs_backup: Dict[str, Any] = PrivateAttr(default=None)
    _true_waiting_inputs: Dict[str, Any] = PrivateAttr(default=None)
    _merged_mapping: Dict[str, Any] = PrivateAttr(default=None)

    def _build_inverse_graph(self):
        self._inverse_graph = {}
        for source, edges in self.edges.items():
            for edge in edges:
                if isinstance(edge, ConditionEdgeSpec):
                    for pos_tar in edge.possible_targets:
                        if pos_tar not in self._inverse_graph:
                            self._inverse_graph[pos_tar] = set()
                        self._inverse_graph[pos_tar].add(source)
                elif isinstance(edge, EdgeSpec):
                    if edge.target not in self._inverse_graph:
                        self._inverse_graph[edge.target] = set()
                    self._inverse_graph[edge.target].add(source)

    def model_post_init(self, context: Any, /) -> None:
        self._trace_tree = RuntimeTaskTree()
        self._bp_set = set()
        self._waiting_inputs_backup = {}
        self._merged_mapping = {}
        self.reset()
        self._build_inverse_graph()

    def save_status_to_db(self):
        self._trace_tree.save_tree()

    def load_status_from_db(self):
        self._trace_tree.load_tree()

    def reset(self):
        self._check_can_running()
        self._state_root = StateCkpt(delta=None, parent=None, state_schema=self.state_schema)
        self._queue = asyncio.Queue()
        self._waiting_inputs = self._analyze_graph()
        self._true_waiting_inputs = self._analyze_graph_true_waiting()
        self._output_channels: asyncio.Queue[RunTimeTask] = asyncio.Queue()
        self._node_exec_cnt = 0
        self._trace_tree.clear_tree()
        self._waiting_inputs_backup.clear()
        self._merged_mapping.clear()

    def _check_can_running(self):
        if self._is_running:
            raise RuntimeError("This graph is still running, cannot reset.")
        if self.state_schema is None:
            raise RuntimeError("A schema of state must be assigned.")

    def _analyze_graph_true_waiting(self):
        config = dict()

        for name, edges in self.edges.items():
            for edge in edges:
                edge: EdgeSpecBase
                for target in edge.get_possible_targets():
                    config.setdefault(target, {}).setdefault(edge.group, {})

                    config[target][edge.group][name] = None
        return config

    def set_breakpoint(self, /, node_name_bp=None, condition_bp=None, condition=None):
        """
        set breakpoint：
            1. break on node
            2. feature: conditional breakpoint, split to input state and output state breakpoint
        """
        if node_name_bp is None:
            raise RuntimeError("condition_bp is not supported.")
        else:
            self._bp_set.add(node_name_bp)

    def clear_breakpoint(self, /, node_name_bp=None, condition_bp=None, condition=None):
        """
        clear breakpoint
        """
        if node_name_bp is None:
            raise RuntimeError("condition_bp is not supported.")
        else:
            self._bp_set.remove(node_name_bp)

    def clear_all_breakpoint(self):
        self._bp_set.clear()

    def _all_task_finished(self):
        for task in self._trace_tree._leaf_nodes.values():
            if task.node_name != BranchFinished:
                return False
        return True

    async def resume(self, running_queue=None):
        """
        resume program
        """
        if self._is_running:
            raise RuntimeError("This graph is still running, cannot run again.")
        if running_queue is None:
            running_queue = [node.task for node in self._trace_tree._leaf_nodes.values()
                if node.node_name not in self._bp_set]

        one_step_result = None
        while running_queue:
            one_step_result, running_queue = await self.run_one_step(running_queue)

        self._is_running = False

        return one_step_result

    def _executable_node(self, node_name_from: str, node_name_to: str):
        if node_name_from == node_name_to:
            return True

        bfs_queue = deque([node_name_from])
        visited = set()
        while bfs_queue:
            node_name = bfs_queue.popleft()
            visited.add(node_name)
            for next_step_edge in self.edges[node_name]:
                if isinstance(next_step_edge, EdgeSpec):
                    next_step_node_name = next_step_edge.target
                    if next_step_node_name == node_name_to:
                        return True
                    if next_step_node_name not in visited:
                        bfs_queue.append(next_step_node_name)
                elif isinstance(next_step_edge, ConditionEdgeSpec):
                    for next_step_node_name in next_step_edge.possible_targets:
                        if next_step_node_name == node_name_to:
                            return True
                        if next_step_node_name not in visited:
                            bfs_queue.append(next_step_node_name)
        return False

    def _get_run_nodes(self, node_name: str):
        all_ava_nodes = [node for node in self._trace_tree._leaf_nodes.values() if node.node_name != node_name]
        run_tasks = []
        for node in all_ava_nodes:
            if self._executable_node(node.node_name, node_name):
                run_tasks.append(node.task)
        return run_tasks

    def _get_step_over_nodes(self, runtime_node: TreeNode, candidate_nodes: List[TreeNode]):
        node = self.nodes[runtime_node.node_name]
        if node.action_mode == NodeActionMode.ANY or len(self._inverse_graph.get(node.node_name, [])) <= 1:
            return [runtime_node]
        else:
            run_nodes = [runtime_node]
            for cand_node in candidate_nodes:
                if cand_node.node_name != node.node_name and self._executable_node(cand_node.node_name, node.node_name):
                    run_nodes.append(cand_node)
            return run_nodes

    async def step_over(self, node_uuid=None):
        """
        step over current breakpoint, program will just execute uuid_node when node_uuid is not None
        else will execute current next step
        """
        if self._trace_tree.in_leaf(node_uuid):
            leaf_node = self._trace_tree._leaf_nodes[node_uuid]
        else:
            leaf_node = None
        run_nodes = []
        pre_nodes = {}
        new_breakpoints = set()

        # set step over nodes and previous nodes node_name list
        if leaf_node is not None:
            run_nodes = [leaf_node]
        else:
            logger.warning(f"node_name not specified, so will step over all current nodes")
            for node in self._trace_tree._leaf_nodes.values():
                if node.node_name == BranchFinished:
                    continue
                elif node.node_name == SpecialNode.END_NODE.value:
                    run_nodes.append(node)
                    continue
                run_nodes.append(node)

        for node in run_nodes:
            if self.nodes[node.node_name].action_mode == NodeActionMode.ALL:
                step_over_nodes = self._get_step_over_nodes(
                    node,
                    candidate_nodes=[cand_node for cand_node in self._trace_tree._leaf_nodes.values()]
                )
                if len(step_over_nodes) > 1:
                    # only all mode will collect pre nodes
                    if step_over_nodes[0] not in pre_nodes:
                        pre_nodes[step_over_nodes[0]] = []
                    pre_nodes[step_over_nodes[0]].extend(step_over_nodes[1:])

        # merge all dependency
        head_nodes = list(pre_nodes.keys())
        for idx, node in enumerate(head_nodes):
            step_over_nodes = self._get_step_over_nodes(
                node,
                candidate_nodes=head_nodes[:idx] + head_nodes[idx + 1:]
            )
            if len(step_over_nodes) > 1:
                for need_merge_node in step_over_nodes[1:]:
                    pre_nodes[step_over_nodes[0]].extend(pre_nodes[need_merge_node])

        # pre_nodes remove duplicates
        for head_node, cands in pre_nodes.items():
            new_cands = []
            has_uuid = set()
            for cand in cands:
                if cand.uuid not in has_uuid:
                    new_cands.append(cand)
                    has_uuid.add(cand.uuid)
            pre_nodes[head_node] = new_cands

        # if previous nodes not empty, will run the previous nodes to breakpoint and clear new breakpoints
        if len(pre_nodes) > 0:
            resume_queue = []
            for bk_node, nodes in pre_nodes.items():
                if bk_node.node_name not in self._bp_set:
                    new_breakpoints.add(bk_node.node_name)
                self.set_breakpoint(bk_node.node_name)
                resume_queue.extend(nodes)
            await self.resume([
                node.task for node in resume_queue
            ])

            # clear new breakpoints
            for bk in new_breakpoints:
                self.clear_breakpoint(bk)

        # if current node have been merged, the node will replace to merged node
        for idx, node in enumerate(run_nodes):
            if node.uuid in self._merged_mapping:
                run_nodes[idx] = self._merged_mapping[node.uuid]

        result, _ = await self.run_one_step([
            node.task for node in run_nodes
        ])

        return result

    def _restore_waiting_input(self, end_nodes: List[TreeNode]):
        """
        given restore end node list, restore waiting_input
        """
        start_nodes: List[TreeNode] = []

        def _dfs(node: TreeNode):
            if len(node.children) == 0:
                start_nodes.append(node)
            else:
                for child in node.children.values():
                    _dfs(child)

        for node in end_nodes:
            _dfs(node)

        end_uuids = set([node.uuid for node in end_nodes])
        visited = set()
        while start_nodes:
            node = start_nodes.pop()
            if node.uuid not in end_uuids and node.uuid not in visited:
                for parent in node.parent.values():
                    self._waiting_inputs[node.node_name][node.task.edge_group][parent.node_name] = (
                        self._waiting_inputs_backup["__".join([parent.uuid, node.uuid])])
                    start_nodes.append(parent)
                visited.add(node.uuid)

    def restore_step(self, node_uuid: str = None):
        """
        restore last step, program will restore last step when node_name is None
        """
        restore_nodes = []
        restore_waiting_input = []
        if node_uuid is None:
            for node in deepcopy(list(self._trace_tree._leaf_nodes.values())):
                for node_parent in node.parent.values():
                    restore_nodes.append(node_parent)
                    if self.nodes[node.node_name].action_mode == NodeActionMode.ALL:
                        # ANY Mode Node will not restore, because the node runtime must through parent node computation
                        restore_waiting_input.append(node_parent)
        else:
            restore_nodes.append(self._trace_tree.get_node_by_uuid(node_uuid))

        self._restore_waiting_input(restore_waiting_input)

        for node_parent in restore_nodes:
            self._trace_tree.backtrack_and_prune(node_parent.uuid)

    async def _process_node(
            self, runtime_task: RunTimeTask
    ) -> List[RunTimeTask]:

        """Process current node and return successor nodes"""
        node_name = runtime_task.node_name

        if self.max_turn and self._node_exec_cnt >= self.max_turn:
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

        parent_node = self._trace_tree.find_nodes_by_trace_route(runtime_task.trace_route)

        if SpecialNode.is_end_node(node_name):
            await self._output_channels.put(runtime)
            self._trace_tree.add_node(node_name=BranchFinished, task=None, parent_uuid=parent_node.uuid)
            return []

        full_state = runtime.state_ckpt.materialize()
        state_delta = await node(full_state)
        new_state_ckpt = StateCkpt(delta=state_delta, parent=runtime.state_ckpt, state_schema=self.state_schema)

        next_tasks = []
        for edge in self.edges[node_name]:
            targets = edge.get_targets(new_state_ckpt.materialize())
            for target, state_filter in targets:
                next_task = RunTimeTask(
                    node_name=target,
                    state_ckpt=new_state_ckpt,
                    edge_group=edge.group,
                    state_filter=state_filter,
                    predecessor=node_name,
                    trace_route=runtime_task.trace_route
                )
                new_node = self._trace_tree.add_node(
                    node_name=target,
                    task=next_task,
                    parent_uuid=parent_node.uuid
                )
                next_task.trace_route += [new_node.uuid]
                next_tasks.append(next_task)

        return next_tasks

    def _backup_waiting_input(self, runtime_task: RunTimeTask, value):
        if len(runtime_task.trace_route) < 2:
            backup_key = "__" + runtime_task.trace_route[-1]
        else:
            backup_key = "__".join(runtime_task.trace_route[-2:])

        self._waiting_inputs_backup[backup_key] = value

    def _update_waiting_inputs(self, runtime_task: RunTimeTask):
        self._true_waiting_inputs[runtime_task.node_name][runtime_task.edge_group][
            runtime_task.predecessor] = runtime_task
        if runtime_task.node_name not in self._waiting_inputs:
            return
        self._waiting_inputs[runtime_task.node_name][runtime_task.edge_group][
            runtime_task.predecessor] = runtime_task

    def _is_predecessor_all_ready(self, runtime_task: RunTimeTask):
        self._true_waiting_inputs[runtime_task.node_name][runtime_task.edge_group][
            runtime_task.predecessor] = runtime_task

        if runtime_task.node_name not in self._waiting_inputs:
            return True

        self._waiting_inputs[runtime_task.node_name][runtime_task.edge_group][runtime_task.predecessor] = runtime_task

        if all([x is not None for x in self._waiting_inputs[runtime_task.node_name][runtime_task.edge_group].values()]):
            # add merge nodes process
            all_need_merge_nodes_uuid = [
                rt.trace_route[-1] for rt in
                self._waiting_inputs[runtime_task.node_name][runtime_task.edge_group].values()
            ]
            target_uuid = runtime_task.trace_route[-1]
            from_nodes = [self._trace_tree.get_node_by_uuid(uid) for uid in all_need_merge_nodes_uuid]
            to_node = self._trace_tree.get_node_by_uuid(target_uuid)
            self._trace_tree.merge_nodes(
                from_nodes=from_nodes,
                to_node=to_node,
            )
            # updating _waiting_inputs_backup
            for from_node in from_nodes:
                from_parent_node_uuid = from_node.task.trace_route[-2]
                to_node_uuid = to_node.uuid
                if from_node.uuid != to_node_uuid:
                    self._merged_mapping[from_node.uuid] = to_node
                    self._waiting_inputs_backup["__".join([from_parent_node_uuid, to_node_uuid])] = (
                        self._waiting_inputs_backup)["__".join(from_node.task.trace_route[-2:])]
            return True
        return False

    async def run_one_step(self, running_queue):
        self._check_can_running()
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
                if not SpecialNode.is_end_node(runtime_task.node_name):
                    self._backup_waiting_input(
                        runtime_task,
                        self._true_waiting_inputs[runtime_task.node_name][runtime_task.edge_group][
                            runtime_task.predecessor]
                    )
                if self._get_node_action_mode(runtime_task.node_name) == NodeActionMode.ALL:
                    self._update_waiting_inputs(runtime_task)
                if self._is_predecessor_all_ready(runtime_task):
                    if runtime_task.node_name not in self._bp_set:
                        next_batch_candidates.append(runtime_task)
        self._is_running = False
        if self._all_task_finished():
            output = await self.get_output()
        else:
            output = None
        return output, next_batch_candidates

    async def debug(self, inputs: Dict):
        self._check_can_running()
        self.reset()
        running_queue: List[RunTimeTask] = [RunTimeTask(
            node_name=SpecialNode.START_NODE.value,
            state_ckpt=self._init_state_ckpt(inputs),
            edge_group=DEFAULT_EDGE_GROUP,
            state_filter=None,
            trace_route=[SpecialNode.START_NODE.value],
        )]
        new_node = self._trace_tree.add_node(
            node_name=SpecialNode.START_NODE.value,
            task=running_queue[-1],
        )
        running_queue[-1].trace_route = [new_node.uuid]

        one_step_result = None
        while running_queue:
            one_step_result, running_queue = await self.run_one_step(running_queue)

        self._is_running = False
        return one_step_result

    def change_output(self, from_node_uuid: str, to_node_uuid: str, change_key: str, change_value: Any):
        if to_node_uuid not in self._trace_tree._leaf_nodes:
            raise RuntimeError(f"change output only support current node!")
        target_node_name = self._trace_tree.get_node_by_uuid(to_node_uuid).node_name
        source_node_name = self._trace_tree.get_node_by_uuid(from_node_uuid).node_name
        source_runtime_node = self._trace_tree.get_node_by_uuid(from_node_uuid)
        if self.nodes[target_node_name].action_mode != NodeActionMode.ANY:
            state_ckpt = self._waiting_inputs[target_node_name][source_runtime_node.task.edge_group][
                source_node_name].state_ckpt
            setattr(state_ckpt.delta['messages'][-1], change_key, change_value)
            setattr(state_ckpt.materialized_state_cache.messages[-1], change_key, change_value)
