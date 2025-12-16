# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Annotated, Any, Callable, List, Literal, Optional, Set, Tuple, Union

from pydantic import Field, field_serializer, field_validator, TypeAdapter

from ..factory import BaseComponent, get_func_serializer
from ..typing import DEFAULT_EDGE_GROUP, State

StateFilterLike = Callable[[State], State]


def _always_list(target: Union[str, List, Set]) -> List[str]:
    """Ensure target is always a list"""
    if isinstance(target, str):
        return [target]
    elif isinstance(target, List):
        return target
    elif isinstance(target, set):
        return list(target)

    raise TypeError(f'Edge target must be either a string or a list, '
                    f'not {type(target)}, please check your router setting')


class EdgeSpecBase(ABC, BaseComponent):
    source: str = Field()
    """Source node"""

    group: str = Field(default=DEFAULT_EDGE_GROUP)
    """Edge group"""

    edge_type: Literal['base'] = "base"
    """type of this edge"""

    @abstractmethod
    def get_targets(self, state: State) -> List[Tuple[str, Optional[StateFilterLike]]]:
        """Return next nodes due to current state."""
        ...

    @abstractmethod
    def get_possible_targets(self) -> List[str]:
        """Return all possible targets.
        This is used to static analyze the graph structure."""
        ...


class EdgeSpec(EdgeSpecBase):
    """An ordinary edge, which only have one target node."""
    target: str = Field()
    """Target node"""

    state_filter: Optional[StateFilterLike] = Field(default=None)
    """State filter"""

    edge_type: Literal['edge'] = "edge"
    """type of this edge"""

    @field_serializer("state_filter")
    def serialize_state_filter(self, _value: Optional[StateFilterLike]) -> Optional[str]:
        if not _value:
            return None
        return get_func_serializer().serialize(self.state_filter)

    @field_validator('state_filter', mode='before')
    @classmethod
    def deserialize_state_filter(cls, v: Any) -> Optional[StateFilterLike]:
        if v is None or callable(v):
            return v
        return get_func_serializer().deserialize(v)

    def get_targets(self, state: State) -> List[str]:
        """Get target and state_filter"""
        targets = _always_list(self.target)
        return list(zip(targets, [self.state_filter] * len(targets)))

    def get_possible_targets(self) -> List[str]:
        """Get target node"""
        return _always_list(self.target)


class ConditionEdgeSpec(EdgeSpecBase):
    """A conditional edge, user can specific target node and corresponding state filter."""

    router: Callable[[State], Union[
        str,
        List[str],
        Tuple[str, Callable[[State], State]],
        List[Tuple[str, Callable[[State], State]]]
    ]]
    """Edge router function, receiving State as input, support four types of output:
    next target node name: return a single str
    multiple next target node name: return a list of str
    next target node name with specific state_filter: return a tuple[str, Callable]
    multiple next target node name with specific state_filter: return a list of tuple[str, Callable]"""

    possible_targets: List[str]
    """Possible targets of this edge. 
    Note that a target returned by the router which is not in `possible_targets` is not allowed."""

    edge_type: Literal['conditional'] = "conditional"
    """type of this edge"""

    @field_serializer("router")
    def serialize_router(self, _value) -> str:
        return get_func_serializer().serialize(self.router)

    @field_validator('router', mode='before')
    @classmethod
    def deserialize_router(cls, v: Any) -> Optional[StateFilterLike]:
        if callable(v):
            return v

        return get_func_serializer().deserialize(v)

    @staticmethod
    def _is_tuple_str_callable(element) -> bool:
        """if element is tuple[str, callable]"""
        return len(element) == 2 and isinstance(element[0], str) and callable(element[1])

    def get_targets(self, state: State) -> List[Tuple[str, Optional[StateFilterLike]]]:
        """Check router return and format all types of return into list[tuple[str, Callable]]"""
        route_res = self.router(state)
        if route_res is None:
            raise ValueError(f"Conditional router result return none, edge source: {self.source}")

        if isinstance(route_res, str):
            formatted_route_res = [(route_res, None)]
        elif self._is_tuple_str_callable(route_res):
            formatted_route_res = [route_res]
        elif isinstance(route_res, Iterable):
            formatted_route_res = []
            for route in route_res:
                if isinstance(route, str):
                    formatted_route_res.append((route, None))
                elif self._is_tuple_str_callable(route):
                    formatted_route_res.append(route)
                else:
                    raise ValueError(f"Unrecognized route type: {route!r}, edge source: {self.source}")
        else:
            raise ValueError(f"Unrecognized route result: {route_res!r}, edge source: {self.source}")
        if not all([x[0] in self.possible_targets for x in formatted_route_res]):
            raise ValueError(f"Router return an target not in possible_ends, edge source: {self.source}")
        return formatted_route_res

    def get_possible_targets(self) -> List[str]:
        """Get all possible targets."""
        return _always_list(self.possible_targets)


def cast_edge(v: dict) -> EdgeSpecBase:
    return TypeAdapter(
        Annotated[
            EdgeSpec |
            ConditionEdgeSpec,
            Field(discriminator="edge_type")
        ]
    ).validate_python(v)
