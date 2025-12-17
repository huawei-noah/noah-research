.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

添加边
==============================

边类型
~~~~~~~~~~~~~~~~~~~~~~

EvoFabric 提供两类边：**普通边** 与 **条件边**，用于描述节点之间的状态传递逻辑。

.. list-table:: 边类型
   :header-rows: 1
   :widths: 20 25 40

   * - 边类型
     - 继承类
     - 功能说明
   * - 普通边
     - :class:`~evofabric.core.graph.EdgeSpec`
     - 固定从源节点到单一目标节点，可选地附带 ``state_filter`` 对状态进行过滤或改写。
   * - 条件边
     - :class:`~evofabric.core.graph.ConditionEdgeSpec`
     - 通过 ``router`` 函数在运行时动态决定下一跳目标节点，可返回一个或多个目标节点，以及每个节点的 ``state_filter`` 函数。


添加普通边
~~~~~~~~~~~~~~~~~~~~~~

调用 :py:meth:`~evofabric.core.graph.GraphBuilder.add_edge` 方法可在两个节点间建立一条普通边。

该方法的典型使用场景是固定地连接两个节点，常用于线性或分支逻辑较少的工作流。

.. code-block:: python

    from pydantic import BaseModel
    from typing import Annotated

    class MyState(BaseModel):
        text: Annotated[str, "overwrite"]
        validated: Annotated[bool, "overwrite"] = False

    graph.add_edge(
        source="preprocess",
        target="analyze",
        group="all",
        state_filter=lambda s: s.copy(update={"validated": True})
    )

**参数说明：**

* ``source`` : 源节点名称，必须是已经添加到图中的节点。
* ``target`` : 目标节点名称，必须是已存在的节点。
* ``group`` : 边所属组名。多个组可用于区分逻辑通道或执行路径，默认值为 ``all``。
* ``state_filter`` : 可选状态过滤函数，用于在状态传递时对 ``State`` 进行改写。

**运行机制：**

当源节点执行完毕后，其输出状态将沿此边流向目标节点。
若提供了 ``state_filter``，状态将在到达目标节点前经过该函数处理。


添加条件边
~~~~~~~~~~~~~~~~~~~~~~

当下一跳节点需要根据运行时状态动态决定时，可使用 :py:meth:`~evofabric.core.graph.GraphBuilder.add_condition_edge` 添加条件边。

此类边的行为由 ``router`` 函数决定，可灵活实现多路径分支逻辑。

.. code-block:: python

    from pydantic import BaseModel
    from typing import Annotated

    class MyState(BaseModel):
        score: Annotated[float, "overwrite"]

    def route_next(state):
        if state.score > 0.8:
            return "success"
        else:
            return "retry"

    graph.add_condition_edge(
        source="evaluate",
        router=route_next,
        possible_targets=["success", "retry"],
        group="decision"
    )

**参数说明：**

* ``source`` : 源节点名称。
* ``router`` : 路由函数，输入为当前节点输出后的完整 ``State``，返回下一跳节点或节点列表。

  支持以下返回形式：

  * ``str``：单一目标节点。
  * ``List[str]``：多个目标节点。
  * ``Tuple[str, Callable]``：目标节点 + 该边专用的状态过滤函数。
  * ``List[Tuple[str, Callable]]``：多个（目标节点, 状态过滤函数）对。

  .. note::
     ``router`` 函数接收到的是 **完整状态** （即输入状态与源节点输出的增量状态合并后的结果）。

* ``possible_targets`` : 所有允许路由到的目标节点集合，用于校验与优化。
* ``group`` : 边所属组名，默认为 ``all``。

**运行机制：**

条件边会在运行时调用 ``router`` 函数，根据状态决定目标节点列表。
系统会自动校验返回的目标节点是否包含在 ``possible_targets`` 中，并统一封装为 ``(目标节点, 状态过滤器)`` 列表格式传递。

.. note::

    条件边通常用于需要根据上下文状态分支、重试或动态路由的场景，
    例如根据Agent的输出分流到不同的处理节点、根据模型置信度判断是否进入 ``retry`` 流程等。


边的分组机制
~~~~~~~~~~~~~~~~~~~~~~

EvoFabric 支持为边设置 ``group`` 参数，用于对不同通道的状态传递进行分组管理。

同一组内的边在目标节点触发逻辑中将被统一处理，可配合节点的 ``action_mode`` 与 ``multi_input_merge_strategy`` 实现更复杂的执行控制。

.. note::

    当一个节点的 ``action_mode`` 设置为 ``all`` 时，通过条件边添加的自循环和退出逻辑应当指定单独的分组。否则，该节点可能永远无法被执行。
    因为 ``all`` 模式下，其自循环逻辑所在的分组始终无法满足“所有前驱节点均已执行完毕”的条件。
