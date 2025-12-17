.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

===================
EmbedClient
===================

EmbedClient 核心功能是将文本或文本列表通过Embedding模型转化为特征向量

------------------
概述
------------------

EmbedClient 模块提供了统一的接口用于调用本地SentenceTransformer Embedding模型，或OpenAI Embedding模型

------------------
特性
------------------

* **统一接口**: 所有客户端继承自 :py:class:`~evofabric.core.clients.EmbedClientBase`，提供一致的调用方式。
* **多文本支持**: 可单次并发创建多条文本的Embedding特征。
* **本地模式**: 支持使用 :py:class:`~evofabric.core.clients.SentenceTransformerEmbed` 对接本地部署的Embedding模型

基本使用
--------------------

.. code-block:: python

    from evofabric.core.clients import OpenAIEmbedClient, SentenceTransformerEmbed

    # OpenAIEmbedClient
    embed_client = OpenAIEmbedClient(
        api_key="your-api-key",
        base_url="your-base-url",
        model="qwen3_0.6B:latest",
    )
    res = embed_client.embed_query("hello")

    # SentenceTransformerEmbed
    embed_client = SentenceTransformerEmbed(
            device="cpu",
            model="hf_models/sentence-transformers/all-MiniLM-L6-v2",
        )
    res = embed_client.embed_query("hello")


在 vectorstore 中使用
----------------------------

.. code-block:: python

    from evofabric.core.clients import OpenAIEmbedClient
    from evofabric.core.vectorstore import ChromaDB

    embed_client = OpenAIEmbedClient(
        api_key="your-api-key",
        base_url="your-base-url",
        model="qwen3_0.6B:latest",
    )

    # embed_client is implicitly called as a component of vectorstore
    vectorstore = ChromaDB(
        collection_name="demo_collection",
        persist_directory="./demo_collection",
        embedding=embed_client,
    )

