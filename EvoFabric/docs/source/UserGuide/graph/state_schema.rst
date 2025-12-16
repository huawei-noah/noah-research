.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

声明状态
====================================

EvoFabric 框架通过在节点之间传递的是状态，有如下特点：

* 在图运行过程中，每个节点的输入是一个完整的 ``State`` 状态变量，节点输出是状态的增量 ``StateDelta`` 。

* 增量状态会通过 **注册过的更新机制** 和状态合并在一起，新的状态会通过边传递给下一个节点。

在构建图引擎时，需要您首先对图中的状态需要的字段 , 数据类型和更新机制进行声明，如果您需要用到自定义的更新机制，还需要将自定义的更新进行注册。

状态声明和默认更新机制
~~~~~~~~~~~~~~~~~~~~~~~~~

状态的声明支持 ``pydantic.BaseModel`` 和 ``TypedDict`` 两种类型，通过Annotated声明数据类型和更新机制。

Annotated中第一个变量用来声明数据类型，第二个变量来声明更新机制（字符串格式，代表注册到 :py:class:`~evofabric.core.graph.StateUpdater` 中的更新机制名字）

我们已经在 :py:class:`~evofabric.core.graph.StateUpdater` 中预置了两种更新机制：

* :py:func:`append_messages <evofabric.core.graph._state_update._append_messages>` 用来记录大模型的上下文（只支持 list[ :py:class:`~evofabric.core.typing.StateMessage` ] 类型），会将大模型的输出消息添加对应的 ``messages`` 列表中，并自动去重。

* :py:func:`overwrite <evofabric.core.graph._state_update._overwrite_state_update_strategy>` 代表节点输出的状态增量会直接覆盖原有值。


.. note::

    如果期望使用 :py:class:`~evofabric.core.agent.AgentNode` 和 :py:class:`~evofabric.core.agent.UserNode` 等依赖LLM类型的节点，必须声明 ``messages`` 字段。

状态声明示例：

.. code-block:: python

    from pydantic import BaseModel
    from typing import Annotated, TypedDict
    from evofabric.core.typing import StateMessage

    class StateSchemaBaseModelType(BaseModel):
        messages: Annotated[list[StateMessage], "append_messages"]
        user_name: Annotated[str, "overwrite"] = "Unknown"

    class StateSchemaTypedDictType(TypedDict):
        messages: Annotated[list[StateMessage], "append_messages"]
        user_name: Annotated[str, "overwrite"]

.. note::

    状态的类型和节点的输入类型是一致的。

    即：状态声明为 ``BaseModel`` 类型，节点的输入也是 ``BaseModel`` 类型，需要通过 ``state.messages`` 取值。
    反之，需要通过 ``state["messages"]`` 取值。
    如果期望对取值方法不做区分，我们也提供 :py:func:`~evofabric.core.factory.safe_get_attr()` 来自适应取值逻辑。


注册自定义更新机制
~~~~~~~~~~~~~~~~~~~

:py:class:`~evofabric.core.graph.StateUpdater` 负责记录状态的更新机制，在您需要自定义状态更新机制时，需要通过 :py:meth:`@StateUpdater.register("update_name") <evofabric.core.graph.StateUpdater.register>` 装饰器注册更新函数。

被注册的更新函数需要实现给定一个旧值 ( ``old`` )和增量值 ( ``new`` )，返回合并后的新值的逻辑。

下面提供了一个实现了列表扩展的更新机制：

.. code-block:: python

    from typing import Annotated
    from pydantic import BaseModel
    from evofabric.core.graph import StateUpdater
    from evofabric.core.typing import MISSING


    @StateUpdater.register("append_list")
    def _append_list(old: list = MISSING, new: list = MISSING):
        result = [] if old is MISSING else old

        if new is MISSING:
            return result
        result.extend(new)
        return result


    class StateSchemaWithListAppend(BaseModel):
        messages: Annotated[list, "append_message"]
        trajectories: Annotated[list, "append_list"]


动态创建状态声明
~~~~~~~~~~~~~~~~~~~~~~~

通过调用 :py:func:`~evofabric.core.graph.generate_state_schema` 方法可以动态创建状态声明。该方法会默认创建一个 ``messages`` 字段，并将输入的其他字段添加到声明中。

.. note::

    变量名必须符合 Python 的变量命名规范。

    变量类型必须是以下之一： ``str`` , ``int`` , ``float`` , ``list`` , ``tuple`` , ``dict``。

示例：

.. code-block:: python

    from evofabric.core.graph import generate_state_schema

    DynamicStateSchema = generate_state_schema()


状态维护与重构机制
==========================

在图引擎的运行过程中，系统通过树形结构维护状态信息，以实现状态的可追溯与高效重构。

1. **状态树结构**

   * 每个节点的输入是一个完整的 ``State`` 对象。
   * 节点的输出不是完整状态，而是该节点的 **状态增量（delta）**。
   * 在运行过程中，所有状态以树的形式存储：

     - 根节点表示输入状态。
     - 每个子节点包含其父节点引用和当前节点的 ``delta``。
     - 状态恢复时，会递归地从父节点向上合并，逐级重构完整状态。

2. **状态重构逻辑**

   当节点或边需要使用完整状态时，图引擎会执行如下步骤：

   * 从当前节点向上递归合并所有 ``delta``。
   * 按声明的 **更新策略（update strategy）** 逐层融合状态。
   * 将得到的完整状态进行深拷贝后，传递给节点或边执行。

   .. note::

    节点和边对输入状态的修改不会影响引擎内部维护的状态树，因为传入的是深拷贝的状态对象。

3. **特殊策略的影响**

   图引擎中存在两类特殊策略，可能导致状态树的根被重置：

   * 节点的 ``multi_input_merge_strategy``
   * 边的 ``state_filter``

   由于这两者都是用户可自定义的合并逻辑，引擎无法保证其结果与原状态完全兼容。因此：

   - 一旦执行上述策略，根节点会被重置为策略输出的状态。
   - 如果这些策略删除了部分状态信息，这些信息将无法在后续节点中恢复。

4. **最终状态输出**

   图执行结束时， ``end`` 节点会收集所有流入的状态，并构建一个 **状态队列**：

   * 队列中的状态按路由顺序排列。
   * 引擎将根据定义的更新策略，依次合并队列中的状态。
   * 最终输出即为所有状态合并后的完整 ``State`` 对象。

   通过这种机制，图引擎能够在保证状态一致性的同时，实现灵活的状态重构与可追溯性。
