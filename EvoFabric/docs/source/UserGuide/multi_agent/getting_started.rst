.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

快速开始
===================

本节将带你用最少的代码跑通一个 Swarm 多智能体协作示例，并了解核心概念与常见配置。


最小可用示例（All-to-All）
--------------------------

以下示例创建 4 个 Agent：planner（规划）、user_name（查姓名）、user_conf（查城市）、weather（查天气）。不提供 ``edges``，默认全连接（除自身），便于快速验证。

.. code-block:: python

    import asyncio
    from typing import Annotated, List
    from pydantic import BaseModel

    from evofabric.core.multi_agent import Swarm
    from evofabric.core.agent import AgentNode
    from evofabric.core.factory import ComponentFactory
    from evofabric.core.tool import ToolManager
    from evofabric.core.typing import StateMessage, UserMessage, AssistantMessage

    # 1) Prepare the LLM client (example, replace with your own client)
    client = ComponentFactory.create(
        "OpenAIChatClient",
        api_key="YOUR_API_KEY",
        model="gpt-4o-mini",
        stream=False
    )

    # 2) Define the state model: must include messages
    class MyState(BaseModel):
        # It is recommended to use Annotated + "append_messages" as an aggregation strategy (if this semantic is enabled)
        messages: Annotated[List[StateMessage], "append_messages"] = []

    # 3) Define example tools
    def check_user_name():
        return "Zhang San"

    def check_user_location(name: str):
        return "Fuxin" if name == "Zhang San" else "Hong Kong"

    def check_weather(city: str):
        return "Light rain" if city == "Fuxin" else "Cloudy"

    # 4) Define Agents
    planner = AgentNode(
        client=client,
        system_prompt=(
            "You are the Planner. Analyze the user's request and delegate subtasks to other agents. "
            "Only delegate to one agent at a time. After completion, output 'FINISHED' at the end of the reply. "
            "The handoff tool parameters include target_agent and info (context/requirement)."
        ),
        tool_manager=[ToolManager(tools=[])]
    )

    user_name = AgentNode(
        client=client,
        system_prompt="You can query the user's name. After completion, hand off the information to the planner.",
        tool_manager=[ToolManager(tools=[check_user_name])]
    )

    user_conf = AgentNode(
        client=client,
        system_prompt="You can query the user's city based on their name. After completion, hand off to the planner.",
        tool_manager=[ToolManager(tools=[check_user_location])]
    )

    weather = AgentNode(
        client=client,
        system_prompt="You can query the weather based on the city. After completion, hand off to the planner. "
                      "If you can already answer, output 'FINISHED' at the end.",
        tool_manager=[ToolManager(tools=[check_weather])]
    )

    # 5) Assemble the Swarm and build the graph
    swarm = Swarm(
        agents={
            "planner": planner,
            "user_name": user_name,
            "user_conf": user_conf,
            "weather": weather,
        },
        state_schema=MyState,
        entry_point_agent="planner",
        # No edges provided => fully connected by default (excluding self)
        max_turns=20,
    )

    graph = swarm.build()

    # 6) Run (Graph is asynchronous)
    async def main():
        state_in = {"messages": [UserMessage(content="What’s the weather like in my city today?")]}
        state_out = await graph.run(state_in)

        last = state_out.messages[-1]
        assert isinstance(last, AssistantMessage)
        print("Final reply:", last.content)

    asyncio.run(main())
