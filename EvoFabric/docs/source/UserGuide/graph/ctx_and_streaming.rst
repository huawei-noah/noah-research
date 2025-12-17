.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

上下文和流式消息管理
===============================

获取上下文环境
~~~~~~~~~~~~~~~~~~~~~~~~~

图引擎运行的过程中，每个节点以及 :py:class:`~evofabric.core.tool.ToolManager` 调用的工具，都会有独立的上下文环境 :py:class:`~evofabric.core.graph.StreamCtx`，可以通过调用 :py:meth:`~evofabric.core.graph.current_ctx` 来获取。

上下文环境包括：

* node_name: 当前节点名字。

* call_id: 节点运行时生成的唯一 UUID。同一节点不同运行轮次时，ID不同。

* tool_name: 当前执行的工具名。

* tool_call_id: 工具执行ID。对应大语言模型输出的 :py:class:`~evofabric.core.typing.ToolCall` 中 ``id`` 。

在节点中获取上下文信息：

.. code-block:: python

    from evofabric.core.graph import current_ctx
    from evofabric.core.typing import State


    def node_a(state: State):
        ctx = current_ctx()
        print(ctx.node_name)


管理流式消息
~~~~~~~~~~~~~~~~~~~

我们支持通过 :py:meth:`~evofabric.core.graph.set_streaming_handler` 注册自定义的流式消息 Handler 来处理节点输出的流式消息。

流式消息输出是 ``dict`` 格式，包含上下文环境和一个 ``payload`` 字段，``payload`` 的值即为节点通过 :py:meth:`~evofabric.core.graph.StreamWriter.put` 方法传出流式消息。

注册示例：

.. code-block:: python

    from evofabric.core.graph import set_streaming_handler

    def print_streaming_msg(x):
        """
        Example input:
        inside node_a
        {
            "node_name": "node_a",
            "call_id": "xxxx-xxx-xxx-xxxx",
            "payload": "xxxxx"
        }

        inside tool_b of agent node b
        {
            "node_name": "node_b",
            "call_id": "xxxx-xxx-xxx-xxxx",
            "tool_name": "tool_b",
            "tool_call_id": "tool-call-id-xxx-xxx-xxx",
            "payload": "xxxxx"
        }
        """
        print(x)

    set_streaming_handler(print_streaming_msg)


