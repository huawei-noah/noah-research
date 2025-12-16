.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

===================
ChatMem
===================

------------------
概述
------------------

ChatMem 是Agent框架的提供的一项高阶记忆能力，可直接接入Agent节点，作为对话记忆，你可以编写定制化提示词来引导记忆关注内容

------------------
特性
------------------

* **作为记忆模块**: ChatMem 继承自 :py:class:`~evofabric.core.mem.MemBase`，是一个高阶记忆模块，可按用户提示词从特定角度理解上下文信息
* **高自由度**: 系统提供了直接可运行的ChatMem实施，其中记忆角度、记忆的抽取、记忆的更新策略，均可通过传入提示词自定义
* **上下文Messages拼接方案**：你可以继承 :py:class:`~evofabric.core.mem.CognitiveMem._select_messages` 以及 :py:class:`~evofabric.core.mem.CognitiveMem._message_to_text` 实现自定义的从智能体消息列表转为待处理储存内容的方法。

最佳实践
--------------------
构建一个ChatMem，及其使用

.. code-block:: python

    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.mem import ChatMem, FEAT_DEFINE_PROMPT_ZH
    from evofabric.core.typing import UserMessage, LLMChatResponse
    from evofabric.core.clients import OpenAIChatClient, SentenceTransformerEmbed
    import os

    # Define a Chat client
    chat_client = OpenAIChatClient(
        model=os.getenv("MODEL_NAME"),
        stream=False,
        client_kwargs={
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
        },
    )

    # Define a vector database
    embed_client = SentenceTransformerEmbed(
        device="cpu",
        model="your-model-path",
    )

    vectorstore = ChromaDB(
        collection_name="chroma_db",
        persist_directory="./chroma_test",
        embedding=embed_client,
        top_k=2
    )

    # Define your ChatMem
    chat_mem = ChatMem(
        zh_mode=False,
        vectorstore=vectorstore,
        chat_client=chat_client,
        feat_define_prompt=FEAT_DEFINE_PROMPT_ZH,
        user_extract_prompt="Your memory extraction prompt, or use default",
        user_update_prompt="Your memory update prompt, or use default",
    )

    # Add messages to memory
    state_messages = [
        UserMessage(content="Ate a cake, feeling very happy")
    ]
    await chat_mem.add_messages(state_messages)

    state_messages = [
        UserMessage(content="On the way home, fell down, feeling sad")
    ]
    await chat_mem.add_messages(state_messages)

    # Use ChatMem for retrieval
    retrieval_messages = [
        UserMessage(content="How am I feeling today?")
    ]
    retrieved_messages = await chat_mem.retrieval_update(retrieval_messages)
