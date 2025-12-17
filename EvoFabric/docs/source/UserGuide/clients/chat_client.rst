.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

===================
ChatClient
===================

.. currentmodule:: evofabric.core.clients

ChatClient 是用于与大语言模型进行对话交互的客户端模块，支持流式和非流式响应获取，适用于构建智能对话系统、Agent 节点等场景。

------------------
概述
------------------

ChatClient 模块提供了统一的接口用于调用不同后端的大语言模型（LLM）进行对话生成。实现对 OpenAI、盘古等模型的统一调用方式，并支持配置模型参数、流式解析、HTTP 客户端设置等功能。

------------------
特性
------------------

* **统一接口**: 所有客户端继承自 :py:class:`~evofabric.core.clients.ChatClientBase`，提供一致的调用方式。
* **流式支持**: 支持流式响应获取，适用于实时交互场景。
* **灵活配置**: 支持模型参数、HTTP 客户端参数、推理参数的灵活配置。
* **可扩展性**: 易于扩展新的模型客户端。
* **异步支持**: 基于异步接口实现，适用于高并发场景。

基本使用
--------------------

.. code-block:: python

    from evofabric.core.clients import OpenAIChatClient
    from evofabric.core.typing import ChatStreamChunk, LLMChatResponse

    # init client
    client = OpenAIChatClient(
        model="gpt-3.5-turbo",
        client_kwargs={"api_key": "your-api-key"},
        inference_kwargs={"temperature": 0.7}
    )

    # non-stream create
    response = await client.create(messages=[{"role": "user", "content": "hello"}])
    print(response.content)

    # streaming create
    async for chunk in client.create_on_stream(messages=[{"role": "user", "content": "hello"}]):
        if isinstance(chunk, ChatStreamChunk):
            print(f"delta: {chunk.content}")
        elif isinstance(chunk, LLMChatResponse):
            print(f"final reply: {chunk.content}")

在 Agent 中使用
--------------------

.. code-block:: python

    from evofabric.core.clients import OpenAIChatClient
    from evofabric.core.agent import AgentNode

    client = OpenAIChatClient(
        model="gpt-3.5-turbo",
        client_kwargs={"api_key": "your-api-key"},
        inference_kwargs={"temperature": 0.7}
    )

    agent = AgentNode(
        client=client
    )

------------------
最佳实践
------------------

**1. 流式响应处理**

适用于实时展示流式消息等场景：

.. code-block:: python

   async def stream_response(client, messages):
       full_content = ""
       async for chunk in client.create_on_stream(messages=messages):
           if hasattr(chunk, 'delta'):
               full_content += chunk.delta
               print(chunk.delta, end="", flush=True)
       return full_content

**2. 参数覆盖机制**

调用时传入的 ``kwargs`` 会覆盖初始化时的 ``inference_kwargs``：

.. code-block:: python

   client = OpenAIChatClient(
       model="gpt-4",
       inference_kwargs={"temperature": 0.7}
   )

   # Temporarily increase temperature
    response = await client.create(
        messages=[{"role": "user", "content": "Write a poem"}],
        temperature=0.9
    )


**3. 错误处理与重试**

建议在调用层加入异常处理逻辑：

.. code-block:: python

   import asyncio

   async def robust_call(client, messages, retries=3):
       for attempt in range(retries):
           try:
               return await client.create(messages=messages)
           except Exception as e:
               if attempt == retries - 1:
                   raise
               await asyncio.sleep(2 ** attempt)

