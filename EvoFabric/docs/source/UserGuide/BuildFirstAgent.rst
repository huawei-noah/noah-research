.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

构建您的第一个智能体应用
============================

安装 EvoFabric
~~~~~~~~~~~~~~~~

您可以使用 pip 命令安装 EvoFabric:

.. code-block:: bash

    pip install evofabric

定义StateSchema
~~~~~~~~~~~~~~~~~~~~

图的构建需要首先声明图中传递的状态和更新机制。目的是为了清楚定义节点之间传递的消息是什么、消息应该如何更新。

其中，

* StateSchema支持 ``BaseModel`` 和 ``TypedDict`` 两种类型。

* 所有参数都必须使用Annotated声明变量类型和更新机制。

* 更新机制需提前使用 :py:meth:`@StateUpdater.register("update_name") <evofabric.core.graph.StateUpdater.register>` 注册，框架已经提供两种更新机制 :py:func:`append_messages <evofabric.core.graph._state_update._append_messages>` 和 :py:func:`overwrite <evofabric.core.graph._state_update._overwrite_state_update_strategy>`。
* 需要在图中引入AgentNode时，必须声明 ``messages: Annotated[list[StateMessage], "append_messages"]`` ，它维护了AgentNode所需的大模型上下文。(:py:func:`append_messages <evofabric.core.graph._state_update._append_messages>` 更新会将每个节点输出的消息加入消息列表中。)

.. seealso::

    :doc:`更多了解关于状态的声明和自定义更新机制注册？ </UserGuide/graph/state_schema>`


示例：

.. code-block:: python

    class StateSchema(BaseModel):
        messages: Annotated[list[StateMessage], "append_messages"]

定义节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们目前提供 :py:class:`~evofabric.core.agent.AgentNode` 和 :py:class:`~evofabric.core.agent.UserNode` 两种预定义Node类型，并支持用户定义通过多种方式自定义节点。

* :py:class:`~evofabric.core.agent.AgentNode` 通过Chat模式获取大模型输出，并具有工具调用、记忆管理、输入格式、输出格式化功能。

* :py:class:`~evofabric.core.agent.UserNode` 允许用户在控制台输入信息作为 :py:class:`~evofabric.core.typing.UserMessage` 补充进 ``messages`` 列表。

.. seealso::
    :doc:`如何声明自定义节点？ </UserGuide/graph/node>`

示例：

.. code-block:: python

    def check_weather(city: str):
        """Check city weather"""
        return f"Weather of {city} if good"

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


在图中添加节点和边
~~~~~~~~~~~~~~~~~~~~~~~~~~~


图的构建需要需要先后添加节点、边，并设置图的起始节点。随后调用 :py:meth:`~evofabric.core.graph.GraphBuilder.build()` 方法获取一个可运行的图引擎。

* 在构建节点时，需要指定节点名、节点实例、节点的动作模式( ``any`` 和 ``all`` )，以及多输入节点需要的状态合并策略（可选）。

* 添加普通边时，需要指定源节点和目标节点。(参考 :py:meth:`~evofabric.core.graph.GraphBuilder.add_edge`)

* 添加条件边时，需要指定源节点、一个路由函数以及路由函数会返回的目标节点全集。(参考 :py:meth:`~evofabric.core.graph.GraphBuilder.add_condition_edge`)

.. note::

    条件边执行时，如果路由函数返回了一个不在目标节点全集里的节点名，会抛出异常。

示例：

.. code-block:: python

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


运行图
~~~~~~~~~~~~~~~~~~~~

图运行可以使用 :py:meth:`~evofabric.core.graph.GraphEngine.run()` 方法，输入是一个和 ``StateSchema`` 的声明严格对应的字典。

.. note::

    - 没有在 ``StateSchema`` 中声明过的字段会被丢弃。

    - 不需要全量声明 ``StateSchema`` 中的所有字段，所有未赋值的字段都会根据类型给出默认值。


.. code-block:: python

    response = await graph.run({
        "messages": [UserMessage(content="What's the weather of my city?")],
        "user": "xxx"
    })


完整Demo
~~~~~~~~~~~~~~~


.. code-block:: python

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
