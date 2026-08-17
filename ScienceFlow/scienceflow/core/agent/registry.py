# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

import logging
from typing import Type

logger = logging.getLogger("scienceflow")

_AGENT_REGISTRY: dict[str, Type] = {}


def register_agent(name: str):
    """Decorator to register an agent class."""

    def decorator(cls):
        _AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def get_agent_class(name: str) -> Type:
    if name not in _AGENT_REGISTRY:
        raise KeyError(
            f"Agent '{name}' not registered. Available: {list(_AGENT_REGISTRY.keys())}"
        )
    return _AGENT_REGISTRY[name]


def list_agents() -> list[str]:
    return list(_AGENT_REGISTRY.keys())
