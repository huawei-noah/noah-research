# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import json
from typing import Callable, Dict, List, Optional, Set, Union

from pydantic import Field, PrivateAttr

from ..factory import BaseComponent, dump_schema_annotated_info, StateSchemaSerializable
from ._edge import cast_edge, ConditionEdgeSpec, EdgeSpec
from ._engine import GraphEngine
from ._engine_debugger import GraphEngineDebugger
from ._node import EndNode, GraphNodeSpec, NodeBase, StartNode
from ._streaming import get_stream_writer
from ..typing import DEFAULT_EDGE_GROUP, NodeActionMode, SpecialNode, State, StateSchema, GraphMode


class GraphBuilder(BaseComponent, StateSchemaSerializable):
    state_schema: type[StateSchema] = Field(description="State schema of this graph")
    """State schema of this graph"""

    _nodes_: Dict[str, GraphNodeSpec] = PrivateAttr(default_factory=dict)
    """Record all nodes in this graph"""

    _edges_: Dict[str, list] = PrivateAttr(default_factory=dict)
    """Record edges that fan-out from one node"""

    _multi_input_merge_: Dict[str, Dict[str, Callable[[list[State]], State]]] = PrivateAttr(default_factory=dict)
    """Record state merge strategy when a node have multiple predecessors"""

    _entry_set_: bool = PrivateAttr(default=False)
    """Mark if this graph has a start node."""

    def _add_start_node(self):
        self._nodes_[SpecialNode.START_NODE.value] = GraphNodeSpec(
            node_name=SpecialNode.START_NODE.value,
            node=StartNode(),
            action_mode=NodeActionMode("any"),
            stream_writer=get_stream_writer(),
        )

    def _add_end_node(self):
        self._nodes_[SpecialNode.END_NODE.value] = GraphNodeSpec(
            node_name=SpecialNode.END_NODE.value,
            node=EndNode(),
            action_mode=NodeActionMode("any"),
            stream_writer=get_stream_writer()
        )

    def _has_end_node(self) -> bool:
        return SpecialNode.END_NODE.value in self._nodes_

    def _has_node(self, node_name):
        return SpecialNode.is_special_node(node_name) or node_name in self._nodes_

    def add_node(
            self,
            name: str,
            node: Union[Callable, NodeBase],
            action_mode: Union[NodeActionMode, str] = "any",
            multi_input_merge_strategy: dict[str, Callable[[List[State]], State]] = None,
    ):
        """
        Add a node to the graph.
        Args:
            name: node name
            node: callable node, can be NodeBase or a simple python function
            action_mode: 'all' means this node should only be executed when all predecessors are executed in one channel.
                          'any' means this node will be executed instantly when one predecessor is executed.
            multi_input_merge_strategy: dict[channel_name(str), merge_strategy for each channel], this is useful for merge state
                when a node has multiple predecessors.

        Returns:

        """
        if SpecialNode.is_special_node(name):
            raise ValueError(f"Node '{name}' is preserved by GraphBuilder.")

        if self._has_node(name):
            raise ValueError(f"Node '{name}' already exists.")

        action_mode = NodeActionMode(action_mode)

        self._nodes_[name] = GraphNodeSpec(
            node_name=name,
            node=node,
            action_mode=NodeActionMode(action_mode),
            stream_writer=get_stream_writer(),
            multi_input_merge_strategy=multi_input_merge_strategy
        )

    def add_edge(
            self,
            source: str,
            target: str,
            group: str = DEFAULT_EDGE_GROUP,
            state_filter: Optional[Callable[[State], State]] = None,
    ):
        """
        Connect two nodes.
        Args:
            source: source node name
            target: target node name
            group: edge group name
            state_filter: filter state function if needed

        Returns:

        """
        if not self._has_node(source):
            raise ValueError(f"Node '{source}' not exists.")

        if not self._has_node(target):
            raise ValueError(f"Node '{target}' not exists.")

        self._edges_.setdefault(source, []).append(
            EdgeSpec(
                source=source,
                target=target,
                group=group,
                state_filter=state_filter,
            )
        )

    def add_condition_edge(
            self,
            source: str,
            router: Callable,
            possible_targets: Union[List[str], Set[str]],
            group: str = DEFAULT_EDGE_GROUP,
    ):
        """
        Route next node by condition

        We support four modes of router function:
        * next target node name: return a single str
        * multiple next target node name: return a list of str
        * next target node name with specific state_filter: return a tuple[str, Callable]
        * multiple next target node name with specific state_filter: return a list of tuple[str, Callable]

        Args:
            source:
            router:
            possible_targets:
            group:

        Returns:

        """
        if not self._has_node(source):
            raise ValueError(f"Node '{source}' not exists.")

        for target in possible_targets:
            if not self._has_node(target):
                raise ValueError(f"Node '{target}' not exists.")

        self._edges_.setdefault(source, []).append(
            ConditionEdgeSpec(
                source=source,
                router=router,
                group=group,
                possible_targets=list(set(possible_targets))
            )
        )

    def set_entry_point(self, entry_name: str):
        self._add_start_node()
        self._edges_[SpecialNode.START_NODE.value] = [EdgeSpec(
            source=SpecialNode.START_NODE.value,
            target=entry_name)]
        self._entry_set_ = True

    def _auto_connect_end(self):
        """Use this method, if u wish to connect all nodes without any fan-out to end"""
        self._add_end_node()

        wait_conn_nodes = []
        for node_name, _ in self._nodes_.items():
            if SpecialNode.is_special_node(node_name):
                continue
            if not self._edges_.get(node_name, None):
                wait_conn_nodes.append(node_name)

        for node_name in wait_conn_nodes:
            self.add_edge(node_name, SpecialNode.END_NODE.value)

    def build(
            self,
            auto_conn_end: bool = True,
            max_turn: int = None,
            timeout: int = None,
            graph_mode: Union[GraphMode, str] = "run",
            db_file_path: str = "./.storage.db",
            db_name: str = "evofabric"
    ):
        assert self._entry_set_, "Graph must have an entry point, using set_entry_point('node_name') to assign one."

        if auto_conn_end:
            self._auto_connect_end()
        assert self._has_end_node(), "Graph must have at least one end node."

        graph_mode = GraphMode(graph_mode)

        if graph_mode == GraphMode.RUN:
            return GraphEngine(
                nodes=self._nodes_,
                edges=self._edges_,
                state_schema=self.state_schema,
                max_turn=max_turn,
                timeout=timeout
            )
        else:
            return GraphEngineDebugger(
                nodes=self._nodes_,
                edges=self._edges_,
                state_schema=self.state_schema,
                max_turn=max_turn,
                timeout=timeout,
                db_file_path=db_file_path,
            )

    def dumps(self, graph_name: str = "graph", version: str = "1.0") -> dict:
        """dump the existing components of this graph builder into config dict"""
        nodes_json = {}
        for k, v in self._nodes_.items():
            try:
                v = v.model_dump()
                # pre-check if v can dump to json
                json.dumps(v)
            except Exception as e:
                raise ValueError(f"Cannot serialize node {k}, value: {v}") from e

            nodes_json[k] = v

        edges_json = {}
        for k, vs in self._edges_.items():
            try:
                vs = [v.model_dump() for v in vs]
                # pre-check if v can dump to json
                json.dumps(vs)
            except Exception as e:
                raise ValueError(f"Cannot serialize edge of node {k}, value: {vs}") from e
            edges_json[k] = vs

        return {
            "graph_name": graph_name,
            "version": version,
            "state_schema": dump_schema_annotated_info(self.state_schema),
            "nodes": nodes_json,
            "edges": edges_json,
            "entry_set": self._entry_set_
        }

    def dump(self, save_path: str, graph_name: str = "graph", version: str = "1.0"):
        """dump the existing components of this graph builder into file"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.dumps(graph_name, version), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> 'GraphBuilder':
        """load a built graph_builder from file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.loads(data)

    @classmethod
    def loads(cls, data: dict) -> 'GraphBuilder':
        """load a built graph_builder from a config dict"""
        graph_builder = cls(state_schema=data["state_schema"])
        graph_builder._edges_ = {k: [cast_edge(v) for v in vs] for k, vs in data.get("edges", {}).items()}
        graph_builder._nodes_ = {k: GraphNodeSpec(**v) for k, v in data.get("nodes", {}).items()}
        graph_builder._entry_set_ = data.get("entry_set", False)
        return graph_builder
