.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

McpToolManager
=====================================

概述
------------------

:py:class:`~evofabric.core.tool.McpToolManager` 继承自 :py:class:`~evofabric.core.tool.ToolManagerBase` ，
用于统一管理**多个外部 MCP 服务器**及其暴露的工具、资源、提示词。

与原生 Python 工具管理器不同，MCP 工具管理器通过 `Model Context Protocol` 与任意语言实现的 MCP 服务器通信，
天然支持跨语言、跨进程、甚至跨主机的工具调用。

在 Agent 场景下，它负责向 :py:class:`~evofabric.core.agent.AgentNode` 提供工具列表，
根据大语言模型（LLM）输出的工具调用指令执行对应工具，并将执行结果反馈给 LLM。
同时支持查看连接状态、动态增删服务器、工具级权限控制等高级功能。


建立MCP连接
----------------

在使用 :py:class:`~evofabric.core.tool.McpToolManager` 前，你需要为每个 MCP 服务配置对应的连接参数 :py:class:`~evofabric.core.typing.McpServerLink`。
随后，通过在 ``async with`` 语句中使用 ``McpToolManager``，即可自动建立与所有已注册 MCP 服务器的连接，并在代码块结束后自动断开连接。
在连接建立后，你可以调用如 ``list_tools``、``call_tools`` 等方法来获取工具信息或执行工具调用。

:py:class:`~evofabric.core.tool.McpToolManager` 提供两种连接管理模式，请务必根据场景选择：

1. **异步上下文管理器（推荐）**

   .. code-block:: python

        from evofabric.core.tool import McpToolManager
        from evofabric.core.typing import StreamableHttpLink

        # create mcp tool manager
        manager = McpToolManager(
            server_links={
                "math_server": StreamableHttpLink(url="http://127.0.0.1:8000/mcp"),
                "file_server": StreamableHttpLink(url="http://127.0.0.1:8001/mcp"),
            },
            timeout=300,
            persistent_link=False
        )
        async with manager:
            await manager.list_tools()
            ...
        # After leaving the `with` block, **regardless of whether `persistent_link` is True**,
        # all MCP connections will be forcibly disconnected to prevent resource leaks
        # when the event loop shuts down.


2. **手动 connect / disconnect**

   .. code-block:: python

       await manager.connect()     # establish connection

       await manager.list_tools()
       ...

       await manager.disconnect()   # must disconnect manually, otherwise the connection will remain active!

   若忘记调用 ``disconnect()``，可能造成 socket、子进程等资源泄露。

.. note::

    即使 ``persistent_link=True``，使用 ``with`` 方式也会在退出时 **主动断连所有服务器** ；
    如需要长连接保持，请采用手动 ``connect/disconnect`` 并自行确保在合适时机释放。


添加、删除MCP服务器
--------------------------

:py:class:`~evofabric.core.tool.McpToolManager` 支持在运行期间动态增删 MCP 服务器。
你可以随时添加新的服务器连接配置，也可以删除已有服务器。删除服务器时，工具管理器会自动断开与该服务器的现有连接。
同时，你也可以根据需要单独重连某个服务器或重连全部服务器，并通过 :py:meth:`~evofabric.core.tool.McpToolManagerget_mcp_status()` 查看当前所有服务器的连接状态。

.. code-block:: python

    # dynamically add servers
    await manager.add_mcp_servers({
        "new_server": StreamableHttpLink(url="http://127.0.0.1:8003/mcp")
    })

    # delete previously added servers (will auto-disconnect first)
    await manager.delete_mcp_servers(["file_server"])

    # reconnect on demand
    await manager.connect("math_server")   # connect a single server
    await manager.connect()                # reconnect all servers

    # check connection status
    status = await manager.get_mcp_status()
    # example return: {"math_server": True, "file_server": False}


MCP服务器使用
-----------------

:py:class:`~evofabric.core.tool.McpToolManager` 支持工具查看、工具调用、获取提示词、获取资源等功能。

示例：

.. code-block:: python

    manager = McpToolManager(
        server_links={
            "math_server": StreamableHttpLink(url="http://127.0.0.1:8000/mcp"),
            "file_server": StreamableHttpLink(url="http://127.0.0.1:8001/mcp"),
        },
        timeout=300,
        persistent_link=False
    )
    async with manager:
        await manager.list_tools()
        res = await manager.call_tools([ToolCall(
            id="0",
            function=Function(
                name="math_server_add",
                arguments=json.dumps(
                    {
                        "a": 2,
                        "b": 1
                    }
                )
            )
        )])
        resources = await manager.list_resources()
        ...

工具控制器（可选）
----------------------

在 :py:class:`~evofabric.core.tool.McpToolManager` 中，你可以为工具调用配置一个可选的
:py:class:`~evofabric.core.tool.ToolController`。工具控制器允许你通过规则动态启用或禁用特定工具。
当控制器被注册后：

1. :py:meth:`~evofabric.core.tool.McpToolManager.list_tools` 会自动过滤掉被禁用的工具，不再返回这些工具的 schema。
2. :py:meth:`~evofabric.core.tool.McpToolManager.call_tools` 若尝试调用被禁用的工具，将返回工具被禁用的异常信息。

通过这种方式，你可以灵活地管理不同场景下允许调用的工具集。

.. code-block:: python

    from evofabric.core.tool import ToolController

    # create and register a tool controller
    controller = ToolController(
        rules={
            "math_server_calculate": True,   # enabled
            "file_server_write": False       # disabled
        }
    )
    manager.set_tool_controller(controller)

    # after registering the controller:
    # - disabled tools will no longer appear in list_tools()
    # - calling a disabled tool will raise an error or return a failure result