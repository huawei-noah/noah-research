# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from enum import Enum
from typing import Any, Dict, Union

from pydantic import BaseModel


class NodeActionMode(str, Enum):
    """When a node has multiple predecessors, it must have an action mode."""
    ANY = "any"
    """This node will execute when receiving a message from any predecessor"""

    ALL = "all"
    """This node will only execute when receiving all message from predecessors"""


class SpecialNode(Enum):
    START_NODE = "start"
    """start node in graph"""

    END_NODE = "end"
    """end node in graph"""

    @classmethod
    def is_special_node(cls, name: str) -> bool:
        return any(name == item.value or name == item for item in cls)

    @classmethod
    def is_end_node(cls, name: Union[str, 'SpecialNode']) -> bool:
        return cls.END_NODE == name or cls.END_NODE.value == name

    @classmethod
    def is_start_node(cls, name: Union[str, 'SpecialNode']) -> bool:
        return cls.START_NODE == name or cls.START_NODE.value == name


class GraphMode(Enum):
    RUN = "run"
    """graph run mode"""

    DEBUG = "debug"
    """graph debug mode"""


DEFAULT_EDGE_GROUP = "all"

STREAM_CHUNK = Any

StateDelta = Dict

State = Union[BaseModel, Dict]

StateSchema = Union[dict, BaseModel]
