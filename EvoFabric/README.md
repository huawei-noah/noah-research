[**中文主页**](README_zh.md)

<h2 align="center">EvoFabric: An Open, Evolvable Agent Framework for Creative Intelligence System</h2>

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
- **[2025-11]** EvoFabric 0.1.3 is released now! Checkout our [guidance documentation](https://evofabric.readthedocs.io/en/latest/) for detailed instructions and best practices.

## ✨Why EvoFabric?
* **Logical Visibility, Controllable Debugging**: The graph structure visualizes agent logic, combined with Debug and visualization features, making the system’s operational paths and state changes clearly visible, bidding farewell to “black-box development”;

* **Highly Scalable**: Modular registration mechanism and Pydantic specifications support rapid integration of custom nodes, tools, and memory modules, adapting to various business scenarios;

* **Natively Asynchronous, Excellent Performance**: Built on Python asyncio, perfectly supports high concurrency and streaming responses, ensuring stable operation of large-scale multi-agent systems;

* **End-to-End Support**: from graph construction, execution, debugging, to export, reload, deployment, providing an end-to-end toolchain to reduce development and operations costs;

* **Versatile Across Scenarios**: Whether it’s the research scenario for rapid prototype validation or the engineering scenario for large-scale deployment, it can provide a solid foundation and flexible expansion points.

* **Industry Compatibility**: Focus on the construction and research of industry-specific Agent capabilities, leverage industry knowledge and expert experience efficiently, and enhance industry operational efficiency.
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


## 🌍 Applications

EvoFabric is designed not only for building general-purpose agent systems, but also for enabling **domain-specific intelligent workflows** and **scientific reasoning pipelines**.  
Below are two representative application directions.

---

### 🏭 Industry Applications


#### 1. SOP2Workflow: From SOP Documents to Executable Agent Workflows

Industrial processes are often written as **Standard Operating Procedures (SOPs)** —  
structured documents describing how tasks should be executed step by step.

However, SOPs are usually **static text**, meaning:

- they cannot be directly executed,
- they require manual workflow engineering,
- and they are difficult to scale into agent-based automation systems.

**SOP2Workflow** automatically transforms a textual SOP into a runnable multi-agent workflow graph.

It enables:

- **SOP → Workflow automation**  
  Convert business documents into executable agent pipelines.

- **Node-level modularization**  
  Break complex procedures into coherent agent nodes.

- **Tool & Memory allocation per node**  
  Each workflow node can be equipped with specific tools and memory modules.

- **Editable and evolvable workflows**  
  Generated workflow definitions are saved to disk and can be refined manually.

- **End-to-end execution support**  
  From document parsing → graph generation → agent execution → visualization.

##### SOP2Workflow: Getting Started

A full runnable example is provided under [sop2workflow example](examples/sop2workflow/README.md) for detailed setup and execution instructions.

#### 2. Another application

### 🔬 Rethinker: A Scientific Reasoning Framework with OpenPangu

EvoFabric also provides a scientific reasoning framework called Rethinker, which is built upon the Rethink paradigm. This framework incorporates the answer from the previous iteration into the next reasoning step, helping to eliminate accumulated reasoning errors.

In addition, we also proposed guided reflection and a confidence-driven selection mechanism to further improve reliability. As a result, EvoFabric achieves top-tier performance on the HLE benchmark leaderboard.

Our paper can be found at [https://arxiv.org/abs/2602.04496](https://arxiv.org/abs/2602.04496)

**Main Results of Rethinker on Expert-Level Reasoning Benchmarks.**

| Category            | Model / Framework                                  | HLE  | GAIA | XBench |
| ------------------- | -------------------------------------------------- | ---- | ---- | ------ |
| Foundation Model    | Kimi K2 (Kimi et al., 2025)                        | 18.1 | 57.7 | 50.0   |
| Foundation Model    | Claude-4.5-Sonnet (Anthropic, 2025)                | 24.5 | 71.2 | 66.0   |
| Foundation Model    | DeepSeek-V3.2 (Liu et al., 2025a)                  | 27.2 | 63.5 | 71.0   |
| Foundation Model    | GLM-4.6 (Zhipu, 2025)                              | 30.4 | 71.9 | 70.0   |
| Foundation Model    | GPT-5-high (OpenAI, 2025b)                         | 35.2 | 76.4 | 77.8   |
| Foundation Model    | Gemini-3-Pro (Google, 2025)                        | 38.3 | 79.0 | 87.0   |
| Inference Framework | WebExplorer (Liu et al., 2025b)                    | 17.3 | 50.0 | 53.7   |
| Inference Framework | OpenAI DeepResearch (OpenAI, 2025a)                | 26.6 | 67.4 | –      |
| Inference Framework | Kimi Researcher (Kimi, 2025)                       | 26.9 | –    | 69.0   |
| Inference Framework | Tongyi DeepResearch (30BA3B) (Tongyi et al., 2025) | 32.9 | 70.9 | 75.0   |
| Inference Framework | MiroThinker-v1.0 (30B) (MiroMind et al., 2025)     | 33.4 | 73.5 | 70.6   |
| Inference Framework | ReThinker (OpenPangu-72B)                          | 33.1 | 72.8 | 78.0   |


## 🤝 Contributors

EvoFabric is an open and evolving project made possible by the efforts of our contributors.  
We sincerely appreciate everyone who helps improve the framework, whether through code, documentation, testing, or ideas.

### Core Contributors

Listed in alphabetical order by first name:

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

## ⚖️ License

EvoFabric is released under MIT License.