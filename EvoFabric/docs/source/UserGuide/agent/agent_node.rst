.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

AgentNode
===================

概述
~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~evofabric.core.agent.AgentNode` 是 EvoFabric 中的智能体节点，继承自 :py:class:`~evofabric.core.graph.AsyncStreamNode`，集成了 **记忆管理**、**工具调用**、**LLM 调用** 和 **输入输出格式化** 功能。
它适用于需要自然语言交互、状态追踪以及工具操作的复杂工作流场景。

**工作流程：**

1. 调用记忆模块的 :py:meth:`~evofabric.core.mem.MemBase.retrieval_update` ，按顺序更新上下文状态。
2. 使用 ``input_msg_format`` 模板，将输入状态格式化为 LLM 可识别的消息。
3. 调用 :py:class:`~evofabric.core.clients.ChatClientBase` 执行推理，获取初步回复。
4. 根据需要调用工具管理器执行相关操作。
5. 使用 ``output_msg_format`` 模板将 LLM 回复渲染为最终输出。
6. 调用记忆模块的 :py:meth:`~evofabric.core.mem.MemBase.add_messages` 按顺序更新记忆。

特性
~~~~~~~~~~~~~~~~~~~~~~

* **集成记忆管理**：可同时管理多个记忆组件，实现上下文追踪。
* **工具调用支持**：可在推理过程中动态调用注册的工具。
* **LLM 推理封装**：统一管理输入格式化、推理参数和输出渲染。
* **流式输出**：通过 :py:class:`~evofabric.core.graph.StreamWriter` 实现逐步输出 LLM 回复。
* **Pydantic 输出验证**：支持 ``output_schema`` 对 LLM 输出进行结构化校验。

使用方法
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pydantic import BaseModel
    from evofabric.core.agent import AgentNode

    class OutputSchema(BaseModel):
        name: str = "Alice"
        age: int = 32

    agent_node = AgentNode(
        client=my_llm_client,
        system_prompt="You are an assistant.",
        tool_manager=[my_tool_manager],
        memory=[my_memory_component],
        output_schema=OutputSchema,
        input_msg_format="{{ state.query }}",
        output_msg_format="My name is {{ name }}, my age in {{ age }}"
    )

    ...
    graph.add_node("agent", agent_node)


