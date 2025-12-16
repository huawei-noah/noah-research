.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

构建图并可视化
==============================

在完成 :doc:`添加节点 <node>` 和 :doc:`添加边 <edge>` 后，就可以继续完成图引擎的生成。

指定入口节点
~~~~~~~~~~~~~~~~~

首先，调用 :py:meth:`~evofabric.core.graph.GraphBuilder.set_entry_point` 方法，指定图中唯一的入口节点（entry point）。入口节点是图运行时的起始节点，必须已通过 :py:meth:`~evofabric.core.graph.GraphBuilder.add_node` 添加。

.. code-block:: python

    builder = GraphBuilder(state_schema=MyStateSchema)
    builder.add_node("start_node", MyNode(), action_mode="any")
    builder.set_entry_point("start_node")

.. note::

    入口节点可以有前驱节点。但运行时会从入口节点向后运行，无法触发前驱节点。

构建图实例
~~~~~~~~~~~~~~~~~

通过 :py:meth:`~evofabric.core.graph.GraphBuilder.build` 方法构建图实例。构建完成可得到 ``run`` 模式的 :py:class:`~evofabric.core.graph.GraphEngine` 或 ``debug`` 模式的 :py:class:`~evofabric.core.graph.GraphEngineDebugger`。

.. code-block:: python

    graph = builder.build(
        auto_conn_end=True,
        max_turn=100,
        timeout=60,
        graph_mode="run",
    )

主要参数说明：

* ``auto_conn_end`` : bool，是否自动将所有无后继节点连接到内置 ``END`` 节点，默认 ``True``。
* ``max_turn`` : int，图最大运行轮数，用于防止无限循环。
* ``timeout`` : int，单个节点运行的超时时间（秒）。
* ``graph_mode`` : str，图的运行模式，``run`` 表示执行模式，``debug`` 表示调试模式。
* ``db_file_path`` : str，持久化数据库文件路径，用于在调试模式下存储状态快照。
* ``db_name`` : str，数据库名称。


可视化图结构
~~~~~~~~~~~~~~~~~

EvoFabric 支持在构建完成后可视化图结构，便于调试和理解节点之间的执行关系。使用 :py:meth:`~evofabric.core.graph.GraphEngine.draw_graph` 方法可以生成图的可视化表示。

.. code-block:: python

    from typing import Annotated, List

    from pydantic import BaseModel

    from evofabric.core.graph import GraphBuilder
    from evofabric.core.typing import AssistantMessage


    class State(BaseModel):
        messages: Annotated[List, "append_messages"]
        node_output: Annotated[str, "overwrite"]


    def node_a(state):
        return {"messages": [AssistantMessage(content="node a output")], 'node_output': 'a'}


    def node_b(state):
        return {"messages": [AssistantMessage(content="node b output")], 'node_output': 'b'}


    def node_c(state):
        return {"messages": [AssistantMessage(content="node c output")], 'node_output': 'c'}


    def node_d(state):
        return {"messages": [AssistantMessage(content="node d output")], 'node_output': 'd'}


    graph = GraphBuilder(state_schema=State)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_node("c", node_c)
    graph.add_node("d", node_d)

    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("a", "d")
    graph.add_edge("d", "c")
    graph.set_entry_point("a")
    graph = graph.build()

    graph.draw_graph()


画图结果：

.. image:: graph_plot_result.png
   :alt: 画图结果
   :width: 600px
   :align: center
