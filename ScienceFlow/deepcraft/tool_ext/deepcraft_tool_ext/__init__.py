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


"""Lazy public exports for optional tool extensions."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "PythonExecute": ("deepcraft_tool_ext.code_execute", "PythonExecute"),
    "FileSaver": ("deepcraft_tool_ext.file_operate", "FileSaver"),
    "GoogleSearch": ("deepcraft_tool_ext.web_search", "GoogleSearch"),
}

__all__ = [
    "PythonExecute",
    "FileSaver",
    "GoogleSearch",
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
