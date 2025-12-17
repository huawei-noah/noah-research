.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.core.multi_agent
=================================

.. py:module:: evofabric.core.multi_agent

Swarm
~~~~~~~~~~


.. py:class:: Swarm(BaseComponent)

   用于创建“Swarm风格”多智能体协作图的自动化构建器。

   Swarm 组件接收一个智能体字典、一个入口点以及一个可选的拓扑定义，用于构建一个可立即运行的图结构。它会动态地为每个智能体注入一个特殊的 ``handoff`` 工具，使其能够在定义的通信路径中将任务委托给其他智能体。

   :param agents: 唯一智能体名称到 ``AgentNode`` 实例或其配置的映射。
                  内部接受的类型为 ``AgentNodeOrConfig = InstanceOrConfig[AgentNode]``，并且可以包含 ``LazyInstance`` 条目。
   :type agents: Dict[str, Union[AgentNode, Dict, LazyInstance]]

   :param state_schema: 定义图共享状态结构的 Pydantic 模型。
   :type state_schema: Type[BaseModel]

   :param entry_point_agent: 入口智能体名称。必须是 ``agents`` 中的一个键。
   :type entry_point_agent: str

   :param edges: 可选的有向边，用于指定允许的任务交接候选，格式为
                 ``[(source_agent_name, target_agent_name), ...]``。如果为 ``None``，则默认每个智能体都可以将任务交接给其他任意智能体。
                 注意：这是一个软约束，用于调整 handoff 工具的函数签名和文档说明。
   :type edges: Optional[List[Tuple[str, str]]]

   :param max_turns: 编译后图的最大轮次限制，用于防止无限循环。
   :type max_turns: int, default 20



   .. py:method:: add_agent(name: str, agent: AgentNodeOrConfig) -> None

      动态添加一个智能体。将在下次调用 :py:meth:`build` 时生效。

      :param name: 要注册的唯一智能体名称。
      :type name: str
      :param agent: 一个 ``AgentNode`` 实例或配置（例如 ``LazyInstance``）。
      :type agent: AgentNodeOrConfig
      :raises ValueError: 如果 ``name`` 已存在。

   .. py:method:: remove_agent(name: str) -> None

      动态移除一个智能体。将在下次调用 :py:meth:`build` 时生效。

      :param name: 要移除的智能体名称。
      :type name: str
      :raises ValueError: 当智能体不存在，或尝试移除入口智能体时抛出。

   .. py:method:: build()

      根据当前配置构建并编译 Swarm 图。

      当 Swarm 的配置（例如添加/删除智能体）发生变化时，应调用此方法。

      :return: 一个已编译的 :py:class:`GraphEngine` 或 :py:class:`GraphEngineDebugger` 实例，准备运行。


   **示例代码：**

   .. code-block:: python

    from typing import List, Tuple, Dict
    from pydantic import BaseModel
    from evofabric.core.multi_agent import Swarm
    from evofabric.core.agent import AgentNode
    from evofabric.core.factory import LazyInstance
    from evofabric.core.typing import StateMessage

    # Define the state schema with a messages field
    class MyState(BaseModel):
        messages: List[StateMessage] = []

    # Define agents (instances or LazyInstance configs)
    # You can supply real AgentNode instances...
    planner = AgentNode(...)     # configure per your framework
    writer = AgentNode(...)

    # ...or defer construction using LazyInstance
    # planner = LazyInstance(class_name="AgentNode", kwargs={...})
    # writer = LazyInstance(class_name="AgentNode", kwargs={...})

    swarm = Swarm(
        agents={
            "planner": planner,
            "writer": writer,
        },
        state_schema=MyState,
        entry_point_agent="planner",
        # If omitted, defaults to all-to-all (excluding self) suggestions
        edges=[("planner", "writer")],
        max_turns=15,
    )

    graph = swarm.build()
    # Use the returned graph as per your Graph runtime API.

   **通过边进行拓扑定制：**

   .. code-block:: python

        # Allow planner to hand off to researcher or writer; researcher can hand off to writer
        edges = [
            ("planner", "researcher"),
            ("planner", "writer"),
            ("researcher", "writer"),
            # writer has no outgoing edges -> no handoff tool injected for writer
        ]

        swarm = Swarm(
            agents={"planner": planner, "researcher": AgentNode(...), "writer": writer},
            state_schema=MyState,
            entry_point_agent="planner",
            edges=edges,
        )

        graph = swarm.build()
        # The "handoff" tool on each agent will expose only the allowed target names.