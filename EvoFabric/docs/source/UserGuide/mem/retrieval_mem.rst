.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

===================
RetrievalMem
===================

------------------
概述
------------------

RetrievalMem 是Agent框架的提供的一项记忆功能，可直接接入Agent节点，作为RAG功能使用

------------------
特性
------------------

* **作为记忆模块**: RetrievalMem 继承自 :py:class:`~evofabric.core.mem.MemBase`，是一个记忆模块，可接入Agent系统
* **RAG功能**: RetrievalMem 本质上实现的是RAG功能，可以基于RetrievalMem直接实现RAG
* **记忆储存**: RetrievalMem中的记忆内容并不会在Agent执行过程中根据上下文刷新，而需要用户通过add_texts方法传入文本列表，如文档的切片列表。

最佳实践
--------------------
基于RetrievalMem构建一个含Retrival和Rerank的RAG模块

.. code-block:: python

    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.mem import RetrievalMem
    from evofabric.core.clients import OpenAIEmbedClient, FlagRerankModel, RerankClientBase

    embed_client = OpenAIEmbedClient(
            api_key="your-api-key",
            base_url="your-base-url",
            model="qwen3_0.6B:latest",
        )

           
    vectorstore = ChromaDB(
        collection_name="chroma_db",
        persist_directory="./chroma_test",
        embedding=embed_client,
    )

    rerank_model = FlagRerankModel(
        model="your-model-path",
        top_n=3,
        devices="cpu"
    )

    retrieval_mem = RetrievalMem(
        vectorstore=vectorstore,
        reranker=rerank_model,
        use_rerank=True
    )
    
    await retrieval_mem.clear()
    await retrieval_mem.add_texts(["The Tianjin Sports Games will open on May 26, 2024."])
    await retrieval_mem.add_texts(["The Jinan Sports Games will open on June 26, 2024."])
    await retrieval_mem.add_texts(["The Changsha Sports Games will open on July 26, 2024."])
    # You can also add all texts together.

    result_messages = await retrieval_mem.retrieval_update([UserMessage(content="What's the date of Tianjin Sports Games")])

