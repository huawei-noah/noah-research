.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

使用示例
===================

示例一：问天气（Star 拓扑）
-------------------------------

Planner 统筹，其他 Agent 各司其职。用户仅输入自然语言，不需要明确知道系统内部如何分工。

.. code-block:: python

    import asyncio
    from typing import Annotated, List
    from pydantic import BaseModel
    from evofabric.core.multi_agent import Swarm
    from evofabric.core.agent import AgentNode
    from evofabric.core.factory import ComponentFactory
    from evofabric.core.tool import ToolManager
    from evofabric.core.typing import StateMessage, UserMessage, AssistantMessage

    client = ComponentFactory.create("OpenAIChatClient", api_key="...", model="gpt-4o-mini")

    class S(BaseModel):
        messages: Annotated[List[StateMessage], "append_messages"] = []

    def check_user_name(): return "Zhang San"
    def check_user_location(name: str): return "Fuxin" if name == "Zhang San" else "Hong Kong"
    def check_weather(city: str): return "Light rain" if city == "Fuxin" else "Cloudy"

    planner = AgentNode(
        client=client,
        system_prompt=(
            "Plan the order of agent calls based on the user's question, and output 'FINISHED' when done. "
            "Available agents: user_name, user_conf, weather. Use 'handoff' to delegate tasks."
        ),
        tool_manager=[ToolManager(tools=[])]
    )
    user_name = AgentNode(
        client=client,
        system_prompt="Query the user's name, then hand off to the planner.",
        tool_manager=[ToolManager(tools=[check_user_name])]
    )
    user_conf = AgentNode(
        client=client,
        system_prompt="Query the user's city based on their name, then hand off to the planner.",
        tool_manager=[ToolManager(tools=[check_user_location])]
    )
    weather = AgentNode(
        client=client,
        system_prompt="Query the weather based on the city; you may answer directly and output 'FINISHED' at the end.",
        tool_manager=[ToolManager(tools=[check_weather])]
    )

    edges = [
        ("planner", "user_name"), ("user_name", "planner"),
        ("planner", "user_conf"), ("user_conf", "planner"),
        ("planner", "weather"), ("weather", "planner"),
    ]

    swarm = Swarm(
        agents={"planner": planner, "user_name": user_name, "user_conf": user_conf, "weather": weather},
        state_schema=S,
        entry_point_agent="planner",
        edges=edges,
        max_turns=15,
    )
    graph = swarm.build()

    async def main():
        state_in = {"messages": [UserMessage(content="What’s the weather like in my city today?")]}
        state_out = await graph.run(state_in)
        last = state_out.messages[-1]
        assert isinstance(last, AssistantMessage)
        print(last.content)  # Expected to contain weather information + FINISHED

    asyncio.run(main())

示例二：运行时扩展能力（动态添加 Agent）
-------------------------------------------

先以全连接启动，运行时引入新的 ``time_agent``，然后重新 ``build()`` 使生效。

.. code-block:: python

    from evofabric.core.agent import AgentNode
    from evofabric.core.tool import ToolManager

    def time_checker(): return "23:50"

    time_agent = AgentNode(
        client=client,
        system_prompt="Query the current time and hand off to the planner.",
        tool_manager=[ToolManager(tools=[time_checker])]
    )

    swarm.add_agent("time_agent", time_agent)
    graph = swarm.build()  # Important: rebuild after adding the new agent

    # Now the planner's handoff can point to time_agent (under the default fully-connected mode)
    # If edges were configured earlier, make sure to extend the edges before rebuilding


示例三：捕获路由日志进行调试
----------------------------

运行时 Swarm 会打印路由跳转日志。可以捕获并分析，验证是否遵循你的拓扑约束与预期规划路径。

.. code-block:: python

    import io, contextlib

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        state_out = await graph.run({"messages": [UserMessage(content="Ask about the weather")]})
    logs = f.getvalue()
    print(logs)
    # Example output (may vary depending on model behavior):
    # Router: Detected handoff from 'user_name' to 'planner'.
    # Router: Detected handoff from 'planner' to 'weather'.
