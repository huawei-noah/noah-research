[**中文主页**](README_zh.md)

<h2 align="center">EvoFabric: An Open, Evolving Fabric for Creative Intelligence</h2>


<p align="center">
    <a href="https://pypi.org/project/evofabric/">
        <img
            src="https://img.shields.io/badge/python-3.11+-blue?logo=python"
            alt="pypi"
        />
    </a>
    <a href="https://pypi.org/project/evofabric/">
        <img
            src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fagentscope%2Fjson&query=%24.info.version&prefix=v&logo=pypi&label=version"
            alt="pypi"
        />
    </a>
    <a href="https://evofabric.readthedocs.io/en/latest/">
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


## 📢 News
- **[2025-11]** EvoFabric 0.1.2 is released now! Checkout our [guidance documentation](https://evofabric.readthedocs.io/en/latest/) for detailed instructions and best practices.

## ✨Why EvoFabric?
* **Logical Visibility, Controllable Debugging**: The graph structure visualizes agent logic, combined with Debug and visualization features, making the system’s operational paths and state changes clearly visible, bidding farewell to “black-box development”;

* **Highly Scalable**: Modular registration mechanism and Pydantic specifications support rapid integration of custom nodes, tools, and memory modules, adapting to various business scenarios;

* **Natively Asynchronous, Excellent Performance**: Built on Python asyncio, perfectly supports high concurrency and streaming responses, ensuring stable operation of large-scale multi-agent systems;

* **End-to-End Support**: from graph construction, execution, debugging, to export, reload, deployment, providing an end-to-end toolchain to reduce development and operations costs;

* **Versatile Across Scenarios**: Whether it’s the research scenario for rapid prototype validation or the engineering scenario for large-scale deployment, it can provide a solid foundation and flexible expansion points.

## 🚀 QuickStart

### Installation

> EvoFabric requires **Python>=3.11**

#### Using pip

```bash
pip install evofabric
```

### Build you first application

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


## ⚖️ License

EvoFabric is released under MIT License.