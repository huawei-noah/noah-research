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

"""Public exports for ``deepcraft_core``.

The package keeps the short import API, e.g. ``from deepcraft_core import Message``,
but resolves symbols lazily so importing one lightweight class does not load optional
vector DB, embedding, MCP, or LLM provider dependencies.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseLLM": ("deepcraft_core.llm", "BaseLLM"),
    "CallStats": ("deepcraft_core.llm", "CallStats"),
    "OpenAI": ("deepcraft_core.llm", "OpenAI"),
    "OnlineLLM": ("deepcraft_core.llm", "OnlineLLM"),
    "LiteLLMClient": ("deepcraft_core.llm", "LiteLLMClient"),
    "RequestClient": ("deepcraft_core.llm", "RequestClient"),
    "StreamHandle": ("deepcraft_core.llm", "StreamHandle"),
    "SentenceTransformerEncoder": ("deepcraft_core.llm", "SentenceTransformerEncoder"),
    "Message": ("deepcraft_core.message", "Message"),
    "Memory": ("deepcraft_core.memory", "Memory"),
    "ToolCall": ("deepcraft_core.tool", "ToolCall"),
    "BaseTool": ("deepcraft_core.tool", "BaseTool"),
    "ToolCollection": ("deepcraft_core.tool", "ToolCollection"),
    "func_to_tool_instance": ("deepcraft_core.tool", "func_to_tool_instance"),
    "BaseKeyValueStorage": ("deepcraft_core.storage", "BaseKeyValueStorage"),
    "InMemoryKeyValueStorage": ("deepcraft_core.storage", "InMemoryKeyValueStorage"),
    "JsonKeyValueStorage": ("deepcraft_core.storage", "JsonKeyValueStorage"),
    "BaseVectorStorage": ("deepcraft_core.storage", "BaseVectorStorage"),
    "ChromaVectorStorage": ("deepcraft_core.storage", "ChromaVectorStorage"),
}

__all__ = [
    "BaseLLM",
    "CallStats",
    "OpenAI",
    "OnlineLLM",
    "LiteLLMClient",
    "RequestClient",
    "StreamHandle",
    "Message",
    "Memory",
    "ToolCall",
    "BaseTool",
    "ToolCollection",
    "func_to_tool_instance",
    "SentenceTransformerEncoder",
    "BaseKeyValueStorage",
    "InMemoryKeyValueStorage",
    "JsonKeyValueStorage",
    "BaseVectorStorage",
    "ChromaVectorStorage",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
