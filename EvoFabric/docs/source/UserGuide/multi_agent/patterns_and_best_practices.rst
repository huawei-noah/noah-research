.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

模式与最佳实践
===================

常见拓扑模式
------------

1. Star（中心化 Planner）
   - 边只在 Planner 与其他 Agent 之间往返
   - 便于集中调度与约束
   - 示例：

   .. code-block:: python

      edges = [
          ("planner", "a"), ("a", "planner"),
          ("planner", "b"), ("b", "planner"),
          ("planner", "c"), ("c", "planner"),
      ]

2. Pipeline（流水线）
   - 线性顺序：A -> B -> C -> Sink
   - 适合固定流程与严格阶段划分
   - 示例：

   .. code-block:: python

      edges = [("a", "b"), ("b", "c")]  

3. All-to-All（全连接）
   - 快速试验，最高自由度
   - 易出现循环，建议设置较小 ``max_turns`` 并在 Agent 提示中强调“完成即 FINISHED”

4. Hybrid（混合）
   - 例如 Planner 统筹主线，但某些专家 Agent 之间允许直接协作
   - 示例：

   .. code-block:: python

        edges = [
            ("planner", "retriever"), ("retriever", "planner"),
            ("planner", "writer"), ("writer", "planner"),
            ("retriever", "writer"),  # Allow direct handoff from retriever to writer
        ]

编写 Agent 提示的要点
---------------------------

- 明确角色分工：能力边界与工具限制，避免 Agent 主动向用户追问

- 移交流程清晰：说明何时使用 ``handoff``，应携带的 ``info``（上下文/需求/结果）

- 终止协议统一：当可以直接回答用户时，回复末尾必须包含 ``FINISHED``

- 一次只移交一个 Agent：避免并行造成的竞争与混乱（可在 Planner 的提示中强调）


handoff 工具的语义
------------------

- Swarm 为每个 Agent 注入定制化的 ``handoff`` 工具

- ``target_agent`` 的可选值来源于 ``edges``（若无 edges 则为“所有其他 Agent”）

- 工具文档字符串包含明确的目标列表，有助于 LLM 正确调用

严格约束拓扑（高级）
--------------------

``edges`` 默认作为“软约束”：路由器仍允许跳转到任何已注册 Agent（前提是消息中的工具调用指定了该目标）。若需“硬约束”，可通过继承 Swarm 并重写路由器来实现：

.. code-block:: python

    from evofabric.core.multi_agent import Swarm

    class StrictSwarm(Swarm):
        def _create_router(self, current_agent_name: str):
            base_router = super()._create_router(current_agent_name)
            allowed = set(self._get_allowed_targets(current_agent_name))

            def router(state):
                nxt = base_router(state)
                # Intercept jumps that are not in allowed (except for "end")
                if nxt in self._agent_names and nxt != "end" and nxt not in allowed:
                    return current_agent_name
                return nxt

            return router

其他建议
--------

- 设置合理的 ``max_turns``，防止 LLM 在边界条件下进入循环
- 通过 ``print`` 路由日志或捕获标准输出，分析真实的跳转路径
- 若 Agent 无出边，它不会注入 handoff 工具，可作为“汇聚/终点”角色