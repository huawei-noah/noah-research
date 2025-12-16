.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

MCP 会话管理
================================

概述
----------------
:py:class:`~evofabric.core.tool.McpSessionController` 类用于管理与 MCP（Model Context Protocol）服务器的完整连接生命周期。它负责连接建立、会话管理、异常安全断开，以及通过异步上下文管理器确保资源正确释放。该组件能够在需要时保持持久连接状态。

核心功能
--------
1. **连接生命周期管理** - 通过异步上下文器自动处理MCP服务器的连接/断开
2. **会话状态监控** - 提供 `is_connect` 属性实时监控连接状态
3. **持久连接支持** - 通过 `persistent_link` 参数控制退出时是否断开连接
4. **后台任务管理** - 维护异步会话线程实现非阻塞通信
5. **异常安全断开** - 处理通信错误时自动执行清理操作

创建会话控制器
---------------
以下示例展示了创建McpSession控制器的基础用法：

.. code-block:: python

    from mcp import McpServerLink
    from evofabric.core.tool import McpSessionController
    from evofabric.core.typing import StdioLink

    # Create MCP server link
    server_link = StdioLink(
        command="python",
        args=[str(current_dir / "your_mcp_server.py")],
    )

    # Initialize the session controller
    session_controller = McpSessionController(
        server_link=server_link,
        server_name="math_service",
        persistent_link=False  # Disconnect when exiting context
    )

    # Establish connection using async context manager
    async with session_controller:
        print(f"Connected: {session_controller.is_connect}")
        await session_controller.session.list_tools()

    # Connection automatically closed when persistent_link=False

属性说明
--------
- ``session`` (只读): 返回当前活动的客户端会话对象
- ``is_connect`` (只读): 布尔值指示当前是否处于连接状态
- ``persistent_link``: 布尔标志，控制是否在上下文退出时保持连接
- ``server_name``: 只读字符串，标识连接的MCP服务器名称

异步方法详解
-------------
connect()
~~~~~~~~~
建立MCP服务器的异步连接：
- 安全可重复调用（重复调用会被忽略）
- 创建后台维护任务等待连接就绪
- 完全建立连接后设置 `_connected` 标志

.. code-block:: python

    await session_controller.connect()
    assert session_controller.is_connect == True

disconnect()
~~~~~~~~~~~
断开 MCP 连接并重置内部状态。包含：

- 取消后台维护任务

- 清除连接相关属性

- 自动处理取消和关闭异常

.. code-block:: python

    await session_controller.disconnect()
    assert session_controller.is_connect == False


异步上下文管理器
----------------

控制器实现了Python的异步上下文管理器协议，确保资源自动管理：

.. code-block:: python

    # Standard usage: auto connect & auto disconnect
    async with McpSessionController(...) as controller:
        print(f"Connected: {controller.is_connect}")
        # Perform operations...

    # Persistent mode: connection stays open after exiting context
    controller = McpSessionController(..., persistent_link=True)
    async with controller:
        # Perform operations...
    # Connection remains active here

.. note::

    当设置 persistent_link=True 时，会话在退出异步上下文管理器后不会自动断开。
    在这种模式下，用户必须显式调用 disconnect() 来释放网络连接与后台任务。
    否则可能导致 资源泄露（例如挂起的会话协程、未释放的 socket 连接等）。