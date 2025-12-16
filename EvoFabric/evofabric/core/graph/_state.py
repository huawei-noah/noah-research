# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import copy
from functools import lru_cache
from typing import (
    Any, Callable, ClassVar, Dict, get_args, get_origin, List, Optional, Tuple, Union
)

from pydantic import create_model, Field, SkipValidation, TypeAdapter
from typing_extensions import Annotated

from ._state_update import StateUpdater
from ._utils import _make_class_name
from ..factory import (
    BaseComponent, fill_defaults, is_basemodel, is_typeddict, safe_convert_to_schema, safe_get_attr,
    safe_set_attr, StateSchemaSerializable
)
from ..typing import MISSING, State, StateDelta, StateMessage, StateSchema


@lru_cache
def get_update_function(state_schema):
    def _walk_type(tp, factory, prefix: str, out: dict[str, [type, Callable]]):
        origin = get_origin(tp)
        if origin is Annotated:
            typ, method_name, *_ = get_args(tp)
            if isinstance(method_name, str):
                out[prefix] = [typ, factory(method_name)]
            return
        if origin is dict or origin is Dict:
            args = get_args(tp)
            if len(args) == 2:  # Dict[K, V]
                _walk_type(args[1], factory, prefix + ".*", out)
            return
        if is_typeddict(tp):
            for k, v in tp.__annotations__.items():
                _walk_type(v, factory, f"{prefix}.{k}", out)
            return

    updaters: dict[str, [type, Callable]] = {}
    for base in reversed(state_schema.__mro__):
        if not hasattr(base, "__annotations__"):
            continue
        for name, anno in base.__annotations__.items():
            if get_origin(anno) is ClassVar or name.endswith("_"):
                continue
            _walk_type(anno, StateUpdater.get, name, updaters)
    return updaters


def generate_state_schema(
        variables: Optional[List[Tuple[str, Any, str]]] = None
):
    """
    Declare the variable names and types for the state information transmitted in the workflow.

    Notes:
        * Variable names must follow Python's variable naming conventions.
        * Variable types must be one of: str, int, float, list, tuple, dict.
        * A constant variable named `messages` exists to record agent context;
          avoid assigning a variable with the same name.

    Args:
        variables: A list of tuples, each containing a variable name, its type and update strategy.

    Examples:
        declare_state_variables([("msg_id", "str", "overwrite"), ("user_id", bool, "overwrite")])

    Raises:
        ValueError: If a variable type is invalid.
        ValueError: If a variable name conflicts with a reserved field or duplicates an existing name.
        ValueError: If an update strategy is not registered in StateUpdater
    """
    variables = variables or []

    fields = {
        "messages": (Annotated[list[StateMessage], "append_messages"], ...)
    }

    for name, type_str, update_strategy in variables:
        if name in fields:
            raise ValueError(f"Got an repeat name: {name}")
        if not StateUpdater.registered(update_strategy):
            raise ValueError(f"Got an unregistered update strategy: {update_strategy}")
        try:
            if isinstance(type_str, str):
                type_str = eval(type_str)
            annotated_type = Annotated[type_str, update_strategy]
        except Exception as e:
            raise ValueError(f"Cannot convert input to a valid annotated, got {type_str}") from e
        fields[name] = (annotated_type, ...)

    state_schema = create_model(
        _make_class_name("StateSchema"),
        **fields
    )
    return state_schema


class StateCkpt(BaseComponent, StateSchemaSerializable):
    delta: Optional[SkipValidation[StateDelta]] = None
    parent: Optional['StateCkpt'] = None
    state_schema: Optional[type[StateSchema]] = None
    materialized_state_cache: Optional[SkipValidation[State]] = Field(default=None, init=False, repr=False)

    @staticmethod
    def merge_state(state, delta, state_schema) -> Union[Dict, StateSchema]:
        state = copy.deepcopy(state)
        updater = get_update_function(state_schema)

        for attr, (typ, update_func) in updater.items():
            safe_set_attr(
                state,
                attr,
                TypeAdapter(typ).validate_python(update_func(
                    safe_get_attr(state, attr, MISSING),
                    safe_get_attr(delta, attr, MISSING))),
            )

        if is_basemodel(state_schema) and not isinstance(state, state_schema):
            state = safe_convert_to_schema(state, state_schema)
        elif is_typeddict(state_schema):
            state = state_schema(**state)
        return state

    def materialize(self) -> Union[State, StateSchema]:
        if self.materialized_state_cache:
            return copy.deepcopy(self.materialized_state_cache)
        state = self.parent.materialize() if self.parent else fill_defaults(self.state_schema)
        updated = self.merge_state(state, self.delta, self.state_schema)
        self.materialized_state_cache = copy.deepcopy(updated)
        return updated

    @classmethod
    def merge(
            cls,
            checkpoints: List['StateCkpt'],
            strategy: Callable[[List[State]], State] = None
    ):
        if not checkpoints:
            raise ValueError("Cannot merge an empty list of checkpoints.")
        parent_states = [ckpt.materialize() for ckpt in checkpoints]

        if strategy:
            merged_state = strategy(parent_states)
        else:
            # use update strategy to merge
            merged_state = parent_states[0]
            for state in parent_states[1:]:
                merged_state = cls.merge_state(merged_state, state, checkpoints[0].state_schema)

        return cls(
            delta=merged_state,
            parent=None,
            state_schema=checkpoints[0].state_schema,
        )

    @classmethod
    def filter(
            cls,
            checkpoint: 'StateCkpt',
            strategy: Callable[[State], State]
    ):
        state = checkpoint.materialize()
        state = strategy(state)
        return cls(
            delta=state,
            parent=None,
            state_schema=checkpoint.state_schema,
        )
