# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import uuid
from typing import Any, Callable, Dict, List

from ..factory import safe_get_attr, safe_set_attr
from ..typing import (
    cast_state_message, MISSING, StateMessage
)


class StateUpdater:
    """
    register and manage state update methods
    usage:
        @StateUpdateStrategy.register("overwrite")
        def overwrite(old: Any, new: Any) -> Any:
            return new

        strategy = StateUpdateStrategy.get("overwrite")
        merged = strategy(old_state, new_state)
    """
    _strategies: Dict[str, Callable[[Any, Any], Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable], Callable]:
        def decorator(func: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
            if name in cls._strategies:
                raise KeyError(f"Strategy '{name}' already registered")
            cls._strategies[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable[[Any, Any], Any]:
        try:
            return cls._strategies[name]
        except KeyError:
            raise KeyError(f"Unknown strategy '{name}'. "
                           f"Available: {list(cls._strategies)}") from None

    @classmethod
    def list_strategies(cls) -> List[str]:
        return list(cls._strategies.keys())

    @classmethod
    def registered(cls, name: str) -> bool:
        return name in cls._strategies


@StateUpdater.register('overwrite')
def _overwrite_state_update_strategy(old: Any = MISSING, new: Any = MISSING) -> Any:
    if old is MISSING and new is MISSING:
        return MISSING

    if new is MISSING:
        return old

    return new


@StateUpdater.register('append_messages')
def _append_messages(old: List[StateMessage] = MISSING, new: List[StateMessage] = MISSING) -> List[StateMessage]:
    """
    Append messages
    """
    if old is MISSING:
        old = []

    if new is MISSING:
        new = []

    id_map = set()
    for msg in old:
        if not isinstance(msg, StateMessage):
            msg = cast_state_message(msg)
        if not safe_get_attr(msg, "msg_id"):
            safe_set_attr(msg, "msg_id", str(uuid.uuid4()))
        id_map.add(safe_get_attr(msg, "msg_id"))

    for msg in new:
        if not isinstance(msg, StateMessage):
            msg = cast_state_message(msg)
        if not safe_get_attr(msg, "msg_id"):
            safe_set_attr(msg, "msg_id", str(uuid.uuid4()))

    result = list([x for x in old] + [x for x in new if safe_get_attr(x, "msg_id") not in id_map])
    result = [cast_state_message(x) for x in result]
    return result
