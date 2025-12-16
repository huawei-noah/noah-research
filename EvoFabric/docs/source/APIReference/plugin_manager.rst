.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

Plugin_manager
======================

.. py:module:: evofabric.plugin_manager

.. py:data:: PluginTypeDict

    插件类型字典，映射基础类到其对应的字符串标识。用于动态加载和注册插件。

    :type: Dict[type, str]
    :example:
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


.. py:method:: load_plugins(parent_cls, tool_type)

    动态加载指定类型的插件，并返回插件类字典。

    :param parent_cls: 插件基类，加载的插件类会继承此类。
    :type parent_cls: type

    :param tool_type: 插件类型字符串，对应 entry points 的 group 名称。
    :type tool_type: str

    :returns: 加载的插件字典，键为 entry point 名称，值为动态生成的插件类。
    :rtype: Dict[str, type]

    .. note::
        1. 使用 importlib.metadata.entry_points 获取指定 group 下的所有 entry points。
        2. 调用每个 entry point 的注册函数获取插件的完整类路径。
        3. 动态导入模块并获取插件类对象。
        4. 使用 type 动态创建一个新类，继承自 parent_cls 和插件类。
        5. 将新类添加到返回字典中。


.. py:method:: init_plugins()

    初始化所有插件，依据 :py:data:`PluginTypeDict` 自动加载对应的插件。

    .. note::
        1. 遍历 PluginTypeDict 中的每个 (插件基类, 工具类型) 对。
        2. 对每个基类和工具类型调用 :py:func:`load_plugins` 进行插件加载。
        3. 该函数主要用于在系统启动时统一初始化所有插件。
