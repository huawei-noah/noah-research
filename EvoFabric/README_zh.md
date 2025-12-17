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


## ⚖️ 许可证

EvoFabric 在 MIT License 许可下发布。