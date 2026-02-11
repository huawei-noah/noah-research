# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from evofabric.core.graph import SyncNode
from evofabric.core.typing import State, StateDelta


class DispatchNode(SyncNode):
    def __call__(self, state: State) -> StateDelta:
        return {}


def get_dispatch_filter(index: int):
    def dispatch_filter(state: State) -> State:
        state.index = index
        return state

    return dispatch_filter
