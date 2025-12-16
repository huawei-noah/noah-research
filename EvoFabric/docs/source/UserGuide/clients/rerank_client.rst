.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

===================
RerankClient
===================

概述
------------------

RerankClient 提供了统一的Reranker基础接口，对齐该接口可实现自定义的Rerank方案。提供了FlagRerank的实现供用户调用和参考

特性
------------------

* **统一接口**: 所有客户端继承自 :py:class:`~evofabric.core.clients.RerankClientBase`，提供一致的调用方式。
* **本地模式**: 支持使用 :py:class:`~evofabric.core.clients.FlagRerankModel` 对接本地部署的Rerank模型

基本使用
--------------------

.. code-block:: python

    import asyncio
    from evofabric.core.clients import FlagRerankModel

    # FlagRerankModel
    rerank_model = FlagRerankModel(
        model="your-model",
        top_n=1,
        devices="cpu"
    )
    
    asyncio.run(rerank_model.rank(query="The tallest mountain in the world", texts=["The tallest mountain in the world is Mount Everest", "The deepest ocean is the Mariana Trench"]))




在 RetrievalMem 中使用
---------------------------

.. code-block:: python

    from evofabric.core.clients import OpenAIEmbedClient, FlagRerankModel
    from evofabric.core.vectorstore import ChromaDB

    embed_client = OpenAIEmbedClient(
        api_key="your-api-key",
        base_url="your-base-url",
        model="qwen3_0.6B:latest",
    )

    # Embed_client acts as the component for the vectorstore, invoked implicitly
    vectorstore = ChromaDB(
        collection_name="demo_collection",
        persist_directory="./demo_collection",
        embedding=embed_client,
    )

    # Example of using FlagRerankModel
    rerank_model = FlagRerankModel(
        model="your-model",
        top_n=1,
        devices="cpu"
    )

    retrieval_mem = RetrievalMem(
        vectorstore=vectorstore,
        reranker=rerank_model,
        use_rerank=True
    )