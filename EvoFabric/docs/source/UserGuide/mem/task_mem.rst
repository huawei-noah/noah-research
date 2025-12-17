.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

===================
TaskMem
===================

------------------
概述
------------------

TaskMem 是Agent框架的提供的一项基于记忆的能力提升功能。其可在每一步智能体推理将操作以及评估模块的结论储存。再次执行时检索相似场景，动态总结经验提升任务成功率。

------------------
特性
------------------

* **基于记忆的演进**: 多轮执行时，TaskMem利用轨迹记忆以及当时的评估结果（可选） 形成经验优化上下文。
* **可接入的评估函数**: TaskMem包含一个即时评估模块入口，可以产生正误、评分及评价。也可以空缺部分字段，直接储存，在规则生成时统一分析。
* **经验生成引导接口**: 支持用户以Prompt形式传入经验总结引导描述，指引新执行任务的经验生成。

最佳实践
--------------------
构建TaskMem并直接使用（也可接入Agent）

.. code-block:: python
    
    import os
    from loguru import logger
    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.mem import TaskMem, TASK_SUMMARY_PROMPT_EN
    from evofabric.core.typing import UserMessage, AssistantMessage, LLMChatResponse, StateMessage
    from evofabric.core.clients import OpenAIChatClient, SentenceTransformerEmbed


    # 1. create a chat client
    chat_client = OpenAIChatClient(
        model=os.getenv("MODEL_NAME"),
        stream=False,
        client_kwargs={
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL")
        }
        )

    # 2. create a vectorstore
    embed_client = SentenceTransformerEmbed(
        device="cpu",
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

    vectorstore = ChromaDB(
        collection_name="chroma_db",
        persist_directory="your db path",
        embedding=embed_client,
        top_k=2
    )

    # 3. define your critic function – demo implementation
    async def demo_critic(messages: list[StateMessage]) -> tuple[bool, float, str]:
        try:
            system_prompt = f"""
            The agent is helping a user troubleshoot mobile-phone malfunctions.
            The correct actions for some common faults are:
            - When the phone loses network: check the wireless-network configuration.
            - When the phone overheats: inspect and clean up unnecessary background apps.
            ...

            Message history:
            {messages}

            Judge whether the agent's action is reasonable in the current context.
            Reply with the following fields:
            <correctness>True/False</correctness>
            <comment>Analyse the agent's policy: explain why it is right or wrong.</comment>
            """
            analysis_messages = [UserMessage(content=system_prompt)]
            analysis = ""
            async for msg in chat_client.create_on_stream(analysis_messages):
                if isinstance(msg, LLMChatResponse):
                    analysis = msg.content
            logger.info(f"Evaluation result: {analysis}")

            correctness_str = analysis.split("<correctness>")[1].split("</correctness>")[0]
            correctness = "True" in correctness_str
            score = 1.0 if correctness else 0.0
            comment = analysis.split("<comment>")[1].split("</comment>")[0]
            return correctness, score, comment
        except Exception as e:
            return True, 0.5, str(e)


    # Create an English-task memory instance
    task_mem = TaskMem(
        zh_mode=False,  # English mode
        vectorstore=vectorstore,
        chat_client=chat_client,
        user_summary_prompt=TASK_SUMMARY_PROMPT_EN,  # English prompt constant example
        eval_func=demo_critic  
    )

    # Example conversation turns
    state_messages = [
        UserMessage(content="My phone lost network connection."),
        AssistantMessage(content="I comforted you."),
    ]
    await task_mem.add_messages(state_messages)

    state_messages = [
        UserMessage(content="My phone lost network connection."),
        AssistantMessage(content="Let me help you check the wireless network configuration."),
    ]
    await task_mem.add_messages(state_messages)

    # System prompt generation via retrieval
    retrieval_messages = [UserMessage(content="My phone lost network again.")]
    retrieved_messages = await task_mem.retrieval_update(retrieval_messages)
    logger.info(f"Memory retrieved: {retrieved_messages[0].content}")   