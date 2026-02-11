.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


Welcome to EvoFabric
========================

EvoFabric 是一款基于异步图引擎构建的轻量级多智能体（Multi-Agent）框架，具备高度灵活性。
我们致力于为开发者提供更直观、更自由的方式，助力其设计、构建和调试复杂的人工智能应用。

该框架以 “图即编排（Graph as Orchestration）” 为核心设计理念，将智能体系统抽象为 状态（State）、节点（Node）、边（Edge） 三个基本组成单元。

整体介绍
------------------------------------

与传统的串行或管道式编排不同，EvoFabric 采用状态驱动的异步图执行引擎，每个节点只需专注于输入的处理与输出的状态变化，而图引擎会自动完成调度、分组、执行与同步工作。

借助高度模块化的设计以及便捷的注册机制，开发者不仅能使用系统提供的标准模块，还可自由定义新的节点、工具、记忆单元、状态更新逻辑等。自定义模块完成注册后，便能像内置组件一样无缝接入图引擎。

EvoFabric 的核心理念是：
**“状态驱动、图式编排、可演化的智能体生态。”**

它既是一套多智能体运行架构，也是一个支持工作流生成与演化的平台。


特性
------------------------------------

EvoFabric 的独特优势体现在对底层图引擎和组件可扩展性的设计上：

1. **直观的状态图驱动引擎**

   - 图由 **状态**、**节点**、 **边** 构成，节点接收完整状态后，输出状态增量。

   - 边不仅负责连接各节点，还承担着状态传递、过滤与改写的功能。同时，其支持的 **条件边** 可动态控制数据流向。

   - 节点能够从多个上游 “边组” 接收数据，并且支持为每个 “边组” 指定不同的状态合并策略，轻松应对复杂的数据汇聚场景。

2. **灵活、可定制的状态管理**

   - 所有状态字段都需明确声明类型与更新规则。

   - 自定义更新逻辑只需定义 ``def function(old, new) -> updated`` 并完成注册，即可无缝接入图引擎。

3. **流式消息处理与上下文感知**

   - 提供上下文访问接口，使节点与工具在运行中能感知在图中的节点和工具调用信息。

   - 支持注册自定义 ``stream_handler``，自由控制流式消息的处理与展示。

4. **严格的 Pydantic 序列化体系**

   - 所有组件遵循 Pydantic 规范，以实现序列化与反序列化功能。

   - 图可导出为配置文件并重新加载运行，保障应用的可迁移性。

   - 原生/自定义组件通过继承基类自动注册到工厂，随图一同导出。

主要功能
----------

EvoFabric 提供了一套开箱即用的功能，帮助您快速启动项目。

*   **核心引擎**:

    - 自动识别并执行四种类型的节点：**同步/异步 + 流式/非流式**。

    - 支持 逐步调试（Debug Mode），让您能够调试复杂的长链工作流。

    - 支持图的可视化功能，方便您查看有向图的连接情况。

*   **内置节点与组件**:

    - **AgentNode**: 集成了记忆、工具调用和 大语言模型（LLM） 的标准化智能体节点。

    - **UserNode**: 暂停图运行并接收用户消息的交互型节点。

    - **多种记忆模块**: 包括 **ChatMem** (对话记忆)、**RetrievalMem** (检索记忆) 等。

*   **强大的工具系统**:

    - 强大的工具管理系统，支持 Python 工具和 MCP 工具。支持工具内在状态的管理。

    - 内置 **Docker 沙箱**，安全执行第三方代码。

*   **模型与数据**:

    - 提供多种 **Client** ，支持与 **chat**, **embed**, **rerank** 等模型交互。

    - 配备向量数据库管理模块，支持文档、记忆等持久性管理。

*   **多智能体协作**:

    - 内置 **Swarm** 模式，支持构建多个智能体（Agent）协同工作的复杂系统。

*   **高级工作流生成**:

    - **SOP to Workflow**: 可将标准作业流程（SOP）文档直接转换为可执行的 EvoFabric 工作流图。

    - **Kernel Evolve**: 具备动态演化和优化 **kernel代码** 的能力。


为什么选择 EvoFabric
------------------------------------

- **逻辑可视，调试可控**：图结构呈现智能体逻辑，结合 Debug 与可视化功能，让系统运行路径与状态变化清晰可见，告别 “黑盒开发”；

- **高度可扩展**：模块化注册机制与 Pydantic 规范，支持自定义节点、工具、记忆模块快速接入，适配各类业务场景；

- **异步原生，性能优异**：基于 Python asyncio 构建，完美支持高并发与流式响应，保障大规模多智能体系统稳定运行；

- **全流程支持**：从图构建、执行、调试，到导出、重载、部署，提供全流程工具链，降低开发与运维成本；

- **兼顾多场景需求**：无论是快速验证原型的研究场景，还是大规模部署的工程化场景，都能提供稳固基础与灵活扩展点。

EvoFabric 让“智能体编排”不再是黑盒，而是一个可视、可调、可演化的系统。
它是你构建下一代智能体系统的结构化基石。


开始
~~~~~~~~~~

:doc:`构建您的第一个智能体应用 </UserGuide/BuildFirstAgent>`

:doc:`了解图引擎 </UserGuide/graph/index>`

:doc:`了解图引擎Debug模式 </UserGuide/GraphDebugger>`

:doc:`了解工具管理模块 </UserGuide/tool/index>`

:doc:`了解记忆管理模块 </UserGuide/mem/index>`

:doc:`了解向量数据存储 </UserGuide/vectorstore/index>`

:doc:`了解图的导出和重载 </UserGuide/Factory>`

.. toctree::
   :hidden:
   :caption: User Guide:
   :maxdepth: 1

   UserGuide/BuildFirstAgent
   UserGuide/graph/index
   UserGuide/vectorstore/index
   UserGuide/agent/index
   UserGuide/multi_agent/index
   UserGuide/clients/index
   UserGuide/mem/index
   UserGuide/tool/index
   UserGuide/GraphDebugger
   UserGuide/Factory
   UserGuide/Plugins

.. toctree::
   :hidden:
   :caption: Developer Guide:
   :maxdepth: 1

   DeveloperGuide/guide

.. toctree::
   :hidden:
   :caption: Applications:
   :maxdepth: 1

   Applications/rethinker
   Applications/sop2workflow
   Applications/kernel_evolve

.. toctree::
   :hidden:
   :caption: API Reference:
   :maxdepth: 1

   APIReference/logger
   APIReference/core/index
   APIReference/app/index
   APIReference/plugin_manager

.. toctree::
   :hidden:
   :caption: Community:
   :maxdepth: 1

   Community/introduction