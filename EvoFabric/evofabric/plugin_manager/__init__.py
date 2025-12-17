# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import importlib.metadata

from ..core.clients import ChatClientBase, EmbedClientBase, RerankClientBase
from ..core.graph import AsyncNode, AsyncStreamNode, SyncNode, SyncStreamNode
from ..core.mem import MemBase
from ..core.tool import McpToolManager, ToolManager, CodeSandbox
from ..core.vectorstore import DBBase
from ..logger import get_logger

logger = get_logger()

PluginTypeDict = {
    ChatClientBase: "ChatClientBase",
    EmbedClientBase: "EmbedClientBase",
    RerankClientBase: "RerankClientBase",
    SyncNode: "SyncNode",
    AsyncNode: "AsyncNode",
    SyncStreamNode: "SyncStreamNode",
    AsyncStreamNode: "AsyncStreamNode",
    MemBase: "MemBase",
    ToolManager: "ToolManager",
    McpToolManager: "McpToolManager",
    CodeSandbox: "CodeSandbox",
    DBBase: "DBBase"
}


def load_plugins(parent_cls, tool_type):
    entry_points = importlib.metadata.entry_points(group=tool_type)
    plugins = {}
    for entry_point in entry_points:
        plugin_register_func = entry_point.load()
        class_name = plugin_register_func()
        module_path, class_name = class_name.rsplit(".", 1)
        plugin_class = getattr(importlib.import_module(module_path), class_name)
        new_class = type(entry_point.name, (parent_cls, plugin_class), {})
        plugins[entry_point.name] = new_class
    return plugins


def init_plugins():
    for plugin_parent_cls, tool_type in PluginTypeDict.items():
        load_plugins(plugin_parent_cls, tool_type)


__all__ = ['init_plugins', 'load_plugins', 'PluginTypeDict']
