.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

图引擎的调试模式
===================

EvoFabric支持用户像调试Python一样对图进行调试，目前已支持的调试操作有：

- 设置断点、取消断点：程序会在断点对应节点停止运行
- Resume：从当前状态运行到结束/下一个断点
- 单步调试：单步步过当前节点/当前运行到的所有节点
- 返回上一步：从当前断点返回运行的上一步
- 修改从节点A -> 节点B的输出结果

------------------
使用方式
------------------

1. 构建图：

.. code-block:: python

    from typing import Annotated, List, TypedDict

    from evofabric.core.graph import GraphBuilder
    from evofabric.core.typing import AssistantMessage, GraphMode


    class State(TypedDict):
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


    graph_builder = GraphBuilder(state_schema=State)
    graph_builder.add_node("a", node_a)
    graph_builder.add_node("b", node_b)
    graph_builder.add_node("c", node_c)
    graph_builder.add_node("d", node_d)

    graph_builder.add_edge("a", "b")
    graph_builder.add_edge("b", "c")
    graph_builder.add_edge("c", "d")
    graph_builder.set_entry_point("a")


2. build图阶段使用 ``graph_mode=GraphMode.DEBUG`` 开启图Debug模式

.. code-block:: python

    graph = graph_builder.build(graph_mode=GraphMode.DEBUG)

3. 设置断点

.. code-block:: python

    graph.set_breakpoint(node_name_bp="b")

4. 开始以debug模式执行

.. code-block:: python

    await graph.debug({"messages": [UserMessage(content='hello')]})

5. 此时图会在 ``b`` 节点停止执行，可以使用单步调试 ``step_over`` 执行节点 ``b``

.. code-block:: python

    await graph.step_over(node_uuid=graph._trace_tree.get_leaf_node_uuid_by_node_name(node_name="b"))

6. 此时已执行完成节点 ``b`` ，如果想恢复到执行节点 ``b`` 之前，则可以使用 ``restore_step``

.. code-block:: python

    graph.restore_step()

7. 此时节点执行状态又恢复至最开始节点 ``b`` 状态，可以使用 ``clear_breakpoint`` 清除断点，接着执行到程序结束

.. code-block:: python

    graph.clear_breakpoint(node_name_bp="b")
    result = await graph.resume()

8. 程序执行结束后，可获得执行结果 ``result`` ，此结果与常规运行结果一致