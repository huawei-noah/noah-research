[**英文主页**](README.md)

<h2 align="center">EvoFabric: 创造智能系统的开放、可进化 Agent 框架</h2>


<p align="center">
    <a href="https://pypi.org/project/evofabric/">
        <img
            src="https://img.shields.io/badge/python-3.11+-blue?logo=python"
            alt="pypi"
        />
    </a>
    <a href="https://pypi.org/project/evofabric/">
        <img
            src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fevofabric%2Fjson&query=%24.info.version&prefix=v&logo=pypi&label=version"
            alt="pypi"
        />
    </a>
    <a href="https://evofabric.readthedocs.io/zh_CN/latest/">
        <img
            src="https://img.shields.io/badge/Docs-English%7C%E4%B8%AD%E6%96%87-blue?logo=markdown"
            alt="docs"
        />
    </a>
    <a href="./LICENSE">
        <img
            src="https://img.shields.io/badge/license-MIT-black"
            alt="license"
        />
    </a>
</p>


## 📢 新闻
- **[2025-11]** EvoFabric 0.1.3 现已发布！请查看我们的[指导文档](https://evofabric.readthedocs.io/zh_CN/latest/)以获取详细说明和最佳实践。

## ✨ 为什么选择 EvoFabric？
* **逻辑可视，调试可控：** 图结构呈现智能体逻辑，结合 Debug 与可视化功能，让系统运行路径与状态变化清晰可见，告别 “黑盒开发”；

* **高度可扩展：** 模块化注册机制与 Pydantic 规范，支持自定义节点、工具、记忆模块快速接入，适配各类业务场景；

* **异步原生，性能优异：** 基于 Python asyncio 构建，完美支持高并发与流式响应，保障大规模多智能体系统稳定运行；

* **全流程支持：** 从图构建、执行、调试，到导出、重载、部署，提供全流程工具链，降低开发与运维成本；

* **兼顾多场景需求：** 无论是快速验证原型的研究场景，还是大规模部署的工程化场景，都能提供稳固基础与灵活扩展点。

* **行业亲和：** 专注于行业Agent能力构建及研究，高效利用行业知识及专家经验，提升行业效率。

## 🚀 快速入门

### 安装

> EvoFabric 要求 **Python>=3.11**

#### 使用 PIP 安装

```bash
pip install evofabric
```

### 构建你的第一个应用

```python
import asyncio
from typing import Annotated

from pydantic import BaseModel

from evofabric.core.agent import AgentNode, UserNode
from evofabric.core.clients import OpenAIChatClient
from evofabric.core.graph import GraphBuilder
from evofabric.core.tool import ToolManager
from evofabric.core.typing import AssistantMessage, State, StateMessage, ToolMessage, UserMessage


class StateSchema(BaseModel):
    messages: Annotated[list[StateMessage], "append_messages"]


def check_weather(city: str):
    """Check city weather"""
    return f"Weather of {city} if good"


async def main():
    llm_chat_client = OpenAIChatClient(
        model="your-model-name",
        client_kwargs={"api_key": "<your-api-key>"}
    )
    agent_node = AgentNode(
        client=llm_chat_client,
        system_prompt="You are a helpful assistant. You can make tool calls to solve user's query."
                      "If you need more information from user, output ::TO::user:"
                      "If you wish to end the conversation, output ::TO::end:",
        tool_manager=ToolManager(tools=[check_weather]),
    )

    user_node = UserNode()

    def fc_router(state: State):
        last_message = state.messages[-1]
        if isinstance(last_message, AssistantMessage):
            reply = last_message.content
            if "::TO::user:" in reply:
                return "user"
            elif "::TO::end:" in reply:
                return "end"
        elif isinstance(last_message, ToolMessage):
            return "agent"
        return "end"

    graph_builder = GraphBuilder(state_schema=StateSchema)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("user", user_node)
    graph_builder.set_entry_point("agent")
    graph_builder.add_condition_edge(
        "agent",
        router=fc_router,
        possible_targets={"user", "end", "agent"}
    )
    graph_builder.add_edge("user", "agent")
    graph = graph_builder.build()

    response = await graph.run({
        "messages": [UserMessage(content="What's the weather of my city?")]
    })
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🌍 应用场景

EvoFabric 的设计目标不仅是构建通用的智能体系统，还旨在支持**领域特定的智能工作流**与**科学推理流水线**。

---

### 🏭 工业应用

#### 1. SOP2Workflow：从 SOP 文档到可执行的智能体工作流

工业流程通常以 **标准操作规程（Standard Operating Procedures, SOPs）** 的形式编写——
这类结构化文档描述了任务应如何一步步执行。

然而，SOP 通常是 **静态文本** ，这意味着：

* 它们无法直接执行，
* 需要人工进行工作流工程设计，
* 并且难以扩展为基于智能体的自动化系统。

**SOP2Workflow** 能够自动将文本形式的 SOP 转换为可运行的多智能体工作流图。

它支持：

* **SOP转换为工作流**  
     将业务文档转换为可执行的智能体流水线。

* **节点采风**  
  将复杂流程拆分为连贯的智能体节点。

* **按节点分配工具与记忆模块**  
  每个工作流节点都可以配置特定工具与记忆模块。

* **可人工参与编辑**  
  生成的工作流定义会保存到磁盘，并可手动进一步优化。

* **端到端执行**  
  从文档解析 → 图生成 → 智能体执行 → 可视化展示。

##### SOP2Workflow：快速开始

完整可运行示例位于 [sop2workflow example](examples/sop2workflow/README.md)，其中包含详细的配置与执行说明。

### 🔬 Rethinker：基于 OpenPangu 的科学推理框架

EvoFabric 还提供了一个名为 Rethinker 的科学推理框架，该框架基于 Rethink 范式构建。它将上一轮迭代的答案融入下一步推理过程，从而帮助消除推理误差的累积。

此外，我们还提出了引导式反思机制与置信度驱动的选择策略，以进一步提升可靠性。因此，EvoFabric 在 HLE 基准排行榜上取得了顶尖表现。

我们的论文地址为：[https://arxiv.org/abs/2602.04496](https://arxiv.org/abs/2602.04496)

**Rethinker 在专家级推理基准上的主要结果。**

| Category                    | Model / Framework                                  | HLE  | GAIA | XBench |
|-----------------------------|----------------------------------------------------|------|------|--------|
| Foundation Model w. tools   | Kimi K2 (Kimi et al., 2025)                        | 18.1 | 57.7 | 50.0   |
| Foundation Model w. tools   | Claude-4.5-Sonnet (Anthropic, 2025)                | 24.5 | 71.2 | 66.0   |
| Foundation Model   w. tools | DeepSeek-V3.2 (Liu et al., 2025a)                  | 27.2 | 63.5 | 71.0   |
| Foundation Model  w. tools  | GLM-4.6 (Zhipu, 2025)                              | 30.4 | 71.9 | 70.0   |
| Foundation Model  w. tools  | GPT-5-high (OpenAI, 2025b)                         | 35.2 | 76.4 | 77.8   |
| Foundation Model   w. tools | Gemini-3-Pro (Google, 2025)                        | 38.3 | 79.0 | 87.0   |
| Inference Framework         | WebExplorer (Liu et al., 2025b)                    | 17.3 | 50.0 | 53.7   |
| Inference Framework         | OpenAI DeepResearch (OpenAI, 2025a)                | 26.6 | 67.4 | –      |
| Inference Framework         | Kimi Researcher (Kimi, 2025)                       | 26.9 | –    | 69.0   |
| Inference Framework         | Tongyi DeepResearch (30BA3B) (Tongyi et al., 2025) | 32.9 | 70.9 | 75.0   |
| Inference Framework         | MiroThinker-v1.0 (30B) (MiroMind et al., 2025)     | 33.4 | 73.5 | 70.6   |
| Inference Framework         | **ReThinker (OpenPangu-72B) (Ours)**               | 33.1 | 72.8 | 78.0   |
| Inference Framework         | **ReThinker (Gemini-3-pro) (Ours)**                        | 52.2 | 81.6 | 90.0   |

## 🤝 贡献者

EvoFabric 是一个开放且持续演进的项目，离不开贡献者们的努力。
我们由衷感谢每一位帮助改进框架的人，无论是通过代码、文档、测试还是想法。

### 核心贡献者

按姓氏字母顺序排列：

* Yuqi Cui
* Da Chen 
* Guojin Chen 
* Zihao Chen 
* Wenyi Fang 
* Jiaquan Guo 
* Hailin Hu 
* Shoubo Hu 
* Shixiong Kai 
* Kaichao Liang 
* Xinduo Liu 
* Ke Ye 
* Lihao Yin
* Mingxuan Yuan

## ⚖️ 许可证

EvoFabric 在 MIT License 许可下发布。