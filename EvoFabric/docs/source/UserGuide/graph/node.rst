.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

添加节点
==============================

节点类型
~~~~~~~~~~~~~~~~~

EvoFabric 的图引擎提供四种类型的节点，分别支持同步、异步及其对应的流式处理模式。不同节点通过继承相应的基类并实现 ``__call__`` 方法来定义处理逻辑，需要流式输出的节点可利用 StreamWriter 实现流式输出。

.. note::

    EvoFabric 支持直接添加普通的 Python 函数作为节点。
    系统会根据函数签名自动识别其类型并封装为合适的节点实例。
    详情参考：:func:`~evofabric.core.graph.callable_to_node`


.. list-table:: 节点类型
   :header-rows: 1
   :widths: 15 15 25 15 10

   * - 节点类型
     - 继承类
     - 方法定义
     - 处理逻辑
     - 流式输出
   * - 同步节点
     - :class:`~evofabric.core.graph.SyncNode`
     - ``def __call__(state: State) -> StateDelta``
     - 在方法中实现节点的同步处理逻辑
     - 不支持
   * - 异步节点
     - :class:`~evofabric.core.graph.AsyncNode`
     - ``async def __call__(state: State) -> StateDelta``
     - 在方法中实现节点的异步处理逻辑
     - 不支持
   * - 同步流式节点
     - :class:`~evofabric.core.graph.SyncStreamNode`
     - ``def __call__(state: State, stream_writer: StreamWriter) -> StateDelta``
     - 在方法中实现节点的同步处理逻辑，并通过 :meth:`~evofabric.core.graph.StreamWriter.put` 输出流式消息
     - 支持
   * - 异步流式节点
     - :class:`~evofabric.core.graph.AsyncStreamNode`
     - ``async def __call__(state: State, stream_writer: StreamWriter) -> StateDelta``
     - 在方法中实现节点的异步处理逻辑，并通过 :meth:`~evofabric.core.graph.StreamWriter.put` 输出流式消息
     - 支持

添加节点
~~~~~~~~~~~~~~~~~~~~~~~

通过 :py:meth:`~evofabric.core.graph.GraphBuilder.add_node` 方法在图中添加节点，同时可以指定节点名、行为模式以及多输入的合并策略（可选）。

每个节点代表一个独立的计算单元或逻辑步骤，可为同步、异步或流式节点。

.. code-block:: python

    from evofabric.core.graph import GraphBuilder, AsyncNode
    from evofabric.core.typing import State, StateDelta

    class AnalyzeNode(AsyncNode):
        async def __call__(self, state: State) -> StateDelta:
            # add your code here
            return {"result": f"analyzed: {state.text}"}

    graph = GraphBuilder(state_schema=State)
    graph.add_node(
        name="analyze",
        node=AnalyzeNode(),
        action_mode="any",
        multi_input_merge_strategy={"default": lambda states: states[0]}
    )

**参数说明：**

* ``name`` ：节点名称，必须唯一。
  不可使用系统保留名称 ``start`` 与 ``end``。

* ``node`` ：节点实例，可为四种节点类型之一（同步、异步、同步流式、异步流式），
  或普通 Python 函数（系统会自动包装为对应节点类型）。

* ``action_mode`` ：节点的触发模式，控制前驱节点完成后何时执行。

  * ``"any"`` ：任意一个前驱节点执行完成即触发。
  * ``"all"`` ：同组别的所有前驱节点执行完成后触发。


* ``multi_input_merge_strategy`` ：多前驱状态合并策略。
  当节点存在来自多个组别的输入时，可为不同组别指定合并函数。
  该参数为一个字典，``key`` 为边的 ``group`` 名，``value`` 为 ``Callable[[List[State]], State]`` 类型的函数。

  若未指定此参数，则系统使用默认的状态更新机制依次合并输入状态。

  .. seealso::
     组别的定义和使用方式详见 :doc:`edge`

  **使用示例：**

  .. code-block:: python

      # When node has multiple predecessor
      graph.add_node(
          name="merge_result",
          node=lambda s: {"final": s["a"] + s["b"]},
          action_mode="all",
          multi_input_merge_strategy={
              "group_a": lambda states: states[0],
              "group_b": lambda states: states[-1]
          }
      )

**运行机制：**

每个节点接收一个完整的 ``State`` 输入，执行计算逻辑后返回字典类型的状态增量（``StateDelta``）。
引擎会将增量自动合并到全局状态中，并按定义的边结构继续向下传递。


