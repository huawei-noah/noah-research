# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import inspect
import traceback
import uuid
from typing import Any, Callable, cast, Dict, List, Optional, Type, Union

from pydantic import Field, field_serializer, field_validator, PrivateAttr

from ._streaming import stream_writer_env, StreamCtx, StreamWriter
from ._utils import _effective_func, _has_stream_writer, _make_class_name
from ..factory import BaseComponent, ComponentFactory, get_func_serializer, safe_get_attr, safe_set_attr
from ..typing import NodeActionMode, State, StateDelta
from ...logger import get_logger

logger = get_logger()


class NodeBase(BaseComponent):
    ...


class SyncNode(NodeBase):

    def __call__(self, state: State) -> StateDelta:
        ...


class AsyncNode(NodeBase):
    async def __call__(self, state: State) -> StateDelta:
        ...


class SyncStreamNode(NodeBase):
    def __call__(self, state: State, stream_writer: StreamWriter) -> StateDelta:
        """A sync stream node.
        Stream messages can be collected by stream.put(data)
        For example:
        ```python
        class SyncStreamNode(Protocol):
            def __call__(self, state: State, stream_writer: StreamWriter) -> StateDelta:
                msg = []
                for stream in llm.chat():
                    msg.append(stream)
                    stream_writer.put(stream)
                return {"msg": "".join(msg)}
        ```
        """
        ...


class AsyncStreamNode(NodeBase):
    async def __call__(self, state: State, stream_writer: StreamWriter) -> StateDelta:
        """
        A async stream node.
        Stream messages can be collected by stream.put(data)
        For example:
        ```python
        class SyncStreamNode(Protocol):
            async def __call__(self, state: State, stream_writer: StreamWriter) -> StateDelta:
                msg = []
                for stream in llm.chat():
                    msg.append(stream)
                    stream_writer.put(stream)
                return {"msg": "".join(msg)}
        ```
        """
        ...


class _SyncPlainFactory:
    def __init__(self, func: Callable[..., StateDelta]) -> None:
        self.func = func

    def build(self) -> NodeBase:
        cls = cast(Type[SyncNode], type(
            _make_class_name("_SyncNode"),
            (SyncNode,),
            {"__call__": self._call})
        )
        return cls()

    def _call(self, state: State) -> StateDelta:
        return self.func(state)


class _SyncStreamFactory:
    def __init__(self, func: Callable[..., StateDelta]) -> None:
        self.func = func

    def build(self) -> NodeBase:
        cls = cast(Type[SyncStreamNode], type(
            _make_class_name("_SyncStreamNode"),
            (SyncStreamNode,),
            {"__call__": self._call})
        )
        return cls()

    def _call(self, state: State, stream_writer: StreamWriter) -> StateDelta:
        return self.func(state, stream_writer)


class _AsyncPlainFactory:
    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func

    def build(self) -> NodeBase:
        cls = cast(Type[AsyncNode], type(
            _make_class_name("_AsyncNode"),
            (AsyncNode,),
            {"__call__": self._call})
        )
        return cls()

    async def _call(self, state: State) -> StateDelta:
        return await self.func(state)


class _AsyncStreamFactory:
    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func

    def build(self) -> NodeBase:
        cls = cast(Type[AsyncStreamNode], type(
            _make_class_name("_AsyncStreamNode"),
            (AsyncStreamNode,),
            {"__call__": self._call})
        )
        return cls()

    async def _call(self, state: State, stream_writer: StreamWriter) -> StateDelta:
        return await self.func(state, stream_writer)


def callable_to_node(callable_obj: Callable[..., Any]) -> NodeBase:
    """Convert any callable object to four subclass of node base."""
    func = _effective_func(callable_obj)
    is_async = inspect.iscoroutinefunction(func)
    need_stream = _has_stream_writer(func)

    if is_async and need_stream:
        factory_cls = _AsyncStreamFactory
    elif is_async and not need_stream:
        factory_cls = _AsyncPlainFactory
    elif not is_async and need_stream:
        factory_cls = _SyncStreamFactory
    else:
        factory_cls = _SyncPlainFactory
    return factory_cls(func).build()


class StartNode(SyncNode):
    def __call__(self, state: State) -> StateDelta:
        logger.info(f"[Node start] Input: {state}")
        return {}


class EndNode(SyncNode):
    def __call__(self, state: State) -> StateDelta:
        return {}


class GraphNodeSpec(BaseComponent):
    node: Union[Callable, NodeBase]
    node_name: str
    action_mode: NodeActionMode = Field(default=NodeActionMode.ALL)
    stream_writer: Optional[StreamWriter] = Field(default=None)
    multi_input_merge_strategy: Optional[Dict[str, Callable[[List[State]], State]]] = Field(default=None)
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _active: bool = PrivateAttr(default=False)
    _original_node_callable: Optional[Callable] = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        if callable(self.node) and not isinstance(self.node, NodeBase):
            self._original_node_callable = self.node
            self.node = callable_to_node(self.node)

    @field_serializer('node')
    def serialize_node(self, _value: NodeBase, _info) -> Dict[str, Any]:
        if self._original_node_callable:
            serialized_func = get_func_serializer().serialize(self._original_node_callable)
            return {'type': 'callable', 'data': serialized_func}
        else:
            return {
                'type': 'node_instance', 'data': self.node.model_dump(), "node_class_name": self.node.__class__.__name__
            }

    @field_validator('node', mode='before')
    @classmethod
    def deserialize_node(cls, v: Any) -> Any:
        if isinstance(v, dict) and 'type' in v and 'data' in v:
            if v['type'] == 'callable':
                return get_func_serializer().deserialize(v['data'])
            elif v['type'] == 'node_instance':
                return ComponentFactory.create(v["node_class_name"], **v['data'])
        return v

    @field_serializer('multi_input_merge_strategy')
    def serialize_merge_strategy(self, strategy: Optional[Dict[str, Callable]]) -> Optional[Dict[str, str]]:
        if strategy is None:
            return strategy
        return {key: get_func_serializer().serialize(func) for key, func in strategy.items()}

    @field_validator('multi_input_merge_strategy', mode='before')
    @classmethod
    def deserialize_merge_strategy(cls, v: Any) -> Optional[Dict[str, Callable]]:
        if v is None:
            return None

        if isinstance(v, dict):
            return {key: v if callable(v) else get_func_serializer().deserialize(v) for key, v in v.items()}
        return v

    @property
    def is_active(self) -> bool:
        """Is this node active"""
        return self._active

    def _inject_node_name(self, delta):
        if _msg := safe_get_attr(delta, "messages", []):
            for msg in _msg:
                safe_set_attr(msg, "node_name", self.node_name)
        return delta

    async def __call__(self, state: State, **kwargs) -> StateDelta:
        """Run node with asyncio locker and set activate status"""
        async with self._lock:
            self._active = True
            try:
                delta = await self._run(state)

                try:
                    delta = self._inject_node_name(delta)
                except Exception as e:
                    logger.warning(
                        f"Injecting node name in `messages` failed, reason: {e}\ntraceback: {traceback.format_exc()}")

                logger.info(f"[Node {self.node_name}] Output: {delta}")
                return delta
            finally:
                self._active = False

    async def _run(self, state: State):
        """Run node with different mode and set stream writer environment"""
        with stream_writer_env(StreamCtx(call_id=str(uuid.uuid4()), node_name=self.node_name)):
            if isinstance(self.node, SyncNode):
                return self.node(state)
            elif isinstance(self.node, AsyncNode):
                return await self.node(state)
            elif isinstance(self.node, SyncStreamNode):
                return self.node(state, stream_writer=self.stream_writer)
            elif isinstance(self.node, AsyncStreamNode):
                return await self.node(state, stream_writer=self.stream_writer)

        raise TypeError("Node type not recognized")
