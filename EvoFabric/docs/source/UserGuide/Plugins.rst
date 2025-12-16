.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

===================
插件
===================

EvoFabric支持用户自定义插件，目前已经预定义的插件类型有：

- :py:class:`~evofabric.core.graph.SyncNode`: 同步节点
- :py:class:`~evofabric.core.graph.AsyncNode`: 异步节点
- :py:class:`~evofabric.core.graph.SyncStreamNode`: 同步流节点
- :py:class:`~evofabric.core.graph.AsyncStreamNode`: 异步流节点
- :py:class:`~evofabric.core.clients.ChatClientBase`: 自定义大模型
- :py:class:`~evofabric.core.clients.EmbedClientBase`: 自定义文本Embedding模块
- :py:class:`~evofabric.core.clients.RerankClientBase`: 自定义ReRank模块
- :py:class:`~evofabric.core.mem.MemBase`: 自定义记忆模块
- :py:class:`~evofabric.core.tool.ToolManager`: 自定义工具管理器
- :py:class:`~evofabric.core.tool.McpToolManager`: MCP工具管理器
- :py:class:`~evofabric.core.tool.CodeSandbox`: 代码沙箱
- :py:class:`~evofabric.core.vectorstore.DBBase`: 数据库

.. note::

     也可在 :class:`evofabric.plugin_manager.PluginTypeDict` 查找支持的插件类型

插件接入原理
------------------

插件接入 EvoFabric 框架需使用 Python 的 ``entry-points`` 机制: 当用户在自己的插件包中配置文件定义了 ``entry-points``，EvoFabric会自动识别到插件。

示例如下：

.. code-block:: toml

    # ChatClientBase type plugin
    [project.entry-points."ChatClientBase"]
    # 'demo_tool1' is plugin name
    demo_tool1 = "demo_tool1:create"

以上配置 :meth:`demo_tool1:create` 对应 ``demo_tool1.create`` 函数，EvoFabric将在注册插件阶段调用该函数以获取插件类。

插件初始化逻辑可见： :func:`evofabric.plugin_manager.load_plugins`

在被EvoFabric框架加载后，插件会自动继承于对应类型的父类，用户即可根据需要使用对应插件。

------------------
插件示例
------------------

接下来我们可以按照此指引自定义一个名为demo_tool1的插件：

1. 首先我们要新建一个python包，目录结构如下：

.. code-block:: bash

    demo_tool1
    │  pyproject.toml
    │  README.md
    │
    ├─demo_tool1
    │  │ invoke_tool.py
    │  │ __init__.py

其中 ``pyproject.toml`` 中包含了entry-points注册块，具体的配置如下：

.. code-block:: toml

    # pyproject.toml

    [build-system]
    requires = ["setuptools>=61.0"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "demo_tool1"
    version = "0.1.2"
    authors = [
      { name="author", email="author@example.com" },
    ]
    description = "An Agent System plugin demo."
    readme = "README.md"
    requires-python = ">=3.11"
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]

    dependencies = []

    [project.entry-points."ChatClientBase"]
    # 'demo_tool1' is plugin name.
    demo_tool1 = "demo_tool1:register"

2. 编写插件逻辑，如下是一个LLM模块的流输出示例，位于 ``demo_tool1/invoke_tool.py`` ：

.. code-block:: python

    from pydantic import Field


    class LLMRunner:
    state: str = Field(description="zhuan")

    async def create_on_stream(
            self, messages
    ):
        for token in "This is the implementation of create_on_stream.":
            yield token


3. 增加注册函数，位于 ``demo_tool1/invoke_tool.py`` ：

.. code-block:: python

    def register():
        return "demo_tool1.LLMRunner"

插件已完成，仅需在当前目录中执行 ``pip install -e .`` 即可安装该插件。