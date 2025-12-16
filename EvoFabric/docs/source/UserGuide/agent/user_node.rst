.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

UserNode
===========================

.. currentmodule:: evofabric.core.agent

UserNode 是一个从终端获取用户输入并为图提供交互功能的节点。

------------------
概述
------------------

UserNode 继承自 :class:`~evofabric.core.graph.AsyncNode`，提供一个简单的用户交互接口。当图执行到 UserNode 时，会在终端显示提示信息并等待用户输入，然后将输入内容传递给图的后续流程。

------------------
特性
------------------

* **异步处理**: 避免阻塞事件循环，支持高并发场景
* **可配置提示**: 支持自定义用户提示信息
* **灵活存储**: 可指定输入在状态中的存储键名
* **异常安全**: 优雅处理各种异常情况
* **简单易用**: 配置简单，开箱即用

基本使用
--------------------

.. code-block:: python

   from evofabric.core.agent._user_node import UserNode

   # Create a UserNode with default configuration
   user_node = UserNode()

   # Custom prompt message
   custom_node = UserNode(prompt_message="Please enter your command: ")

   # Custom storage key name
   keyed_node = UserNode(input_key="custom_input")

在图中使用
--------------------

.. code-block:: python

   # Define a graph containing UserNode
   from evofabric.core.graph import GraphBuilder

   graph = GraphBuilder()

   # Add nodes and edges
   graph.add_node("user_input", User  Node(prompt_message="Please enter task description: "))
   graph.add_node("process", YourProcessingNode())
   graph.add_edge("user_input", "process")

   # Execute the graph
   final_state = await graph.run({"messages": []})

   # Assume user input "Complete data analysis"
   # Result: final_state["messages"] contains the user input message

最佳实践
------------------

**1. 输入验证**

如果需要对用户输入进行验证，可以在后续节点中添加处理逻辑：

.. code-block:: python

   async def validate_input(state: State) -> StateDelta:
       user_input = state.get("user_input", "")
       if not user_input.strip():
           return {"error": "Input cannot be empty"}
       # Other validation ...

**2. 错误处理**

UserNode 已经内置完善的异常处理，但建议在图的后续节点中添加业务逻辑错误处理：

.. code-block:: python

   async def handle_error(state: State) -> StateDelta:
       if "error" in state:
           print(f"Error: {state['error']}")
           return {"messages": [UserMessage(content="Please re-enter")]}
       return {}

**3. 多轮对话**

可以实现简单的多轮对话模式：

.. code-block:: python

   class ConversationNode(AsyncNode):
       async def __call__(self, state: State) -> StateDelta:
           messages = state.get("  "messages", [])
           if not messages:
               # First round: ask for requirements
               return {"messages": [UserMessage(content="What help do you need?")]}
           else:
               # Subsequent rounds: process user replies
               last_message = messages[-1].content
               if last_message.lower() in ["exit", "quit"]:
                   return {"messages": [UserMessage(content="Goodbye!")]}
               # Process other replies...
               return {"messages": [UserMessage(content="Your reply has been received")]}

参数说明
------------------

.. py:attribute:: prompt_message

   显示的用户输入提示信息。

   :type: str
   :default: "Please enter your input: "

.. py:attribute:: input_key

   用户输入在状态中存储的键名。

   :type: str
   :default: "user_input"

异常处理
------------------

UserNode 自动处理以下异常情况：

- **EOFError**: 用户输入流结束（如文件结束符）
- **KeyboardInterrupt**: 用户中断输入（Ctrl+C）
- **其他异常**: 捕获所有未预期的错误

所有异常情况下都会返回空的状态增量，确保图执行不会中断。