.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

快速开始
===================

本指南将帮助您开始使用 EvoFabric 中的 :py:class:`~evofabric.core.vectorstore` 模块。您将学习如何设置、配置和使用向量数据库来满足您的应用需求。

先决条件
--------

在使用 VectorStore 模块之前，请确保您拥有：

- Python 3.11 或更高版本

- 必需的依赖项：

  - ``chromadb>=1.1.0`` （用于 ChromaDB 后端）

安装
--------

安装必需的包：

.. code-block:: bash

    pip install chromadb>=1.1.0

如果您计划使用特定的嵌入模型，请安装相应的客户端：

.. code-block:: bash

    # OpenAI embedding
    pip install openai

    # HuggingFace embedding
    pip install transformers torch

基本设置
-----------

以下是设置基本向量存储的方法：

.. code-block:: python

    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.typing import DBItem
    from evofabric.core.clients import OpenAIEmbedClient, SentenceTransformerEmbed

    # Option 1: Use OpenAI embedding client
    embed_client = OpenAIEmbedClient(
        api_key="your-api-key",
        model="text-embedding-ada-002"
    )

    # Option 2: Use local SentenceTransformer embedding (recommended for local development)
    embed_client = SentenceTransformerEmbed(
        model="/path/to/sentence-transformer-model",  # e.g., "all-MiniLM-L6-v2"
        device="cpu"  # or "cuda" for GPU
    )

    # Option 3: Use a local model path
    embed_client = SentenceTransformerEmbed(
        model="/path/to/sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    # Initialize the vector store
    vector_store = ChromaDB(
        collection_name="my_documents",
        persist_directory="./chroma_db",
        embedding=embed_client,
        top_k=5
    )

添加文档
--------

向向量存储中添加文档：

.. code-block:: python

    # Creating documents
    documents = [
        DBItem(
            text="EvoFabric is an agent framework.",
            metadata={"category": "AI", "source": "evofabric_docs"}
        ),
        DBItem(
            text="Vector storage enables efficient similarity search",
            metadata={"category": "database", "source": "tech_docs"}
        ),
        DBItem(
            text="ChromaDB offers vector database solutions",
            metadata={"category": "database", "source": "chroma_docs"}
        )
    ]

    # Add documents to vector storage
    ids = await vector_store.add_texts(documents)
    print(f"Documents updated: ID: {ids}")

搜索文档
--------

执行相似性搜索：

.. code-block:: python

    # Search for similar documents
    results = await vector_store.similarity_search("What is a vector database?")

    for result in results:
        print(f"Score: {result.score}")
        print(f"Text: {result.text}")
        print(f"Metadata: {result.metadata}")
        print("-" * 50)

完整示例
--------

以下是使用 SentenceTransformer 嵌入的完整工作示例：

.. code-block:: python

    import asyncio
    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.typing import DBItem
    from evofabric.core.clients import SentenceTransformerEmbed

    async def main():
        # Initialize vector store with local SentenceTransformer embedding
        embed_client = SentenceTransformerEmbed(
            model="all-MiniLM-L6-v2",  # Can use model name or local path
            device="cpu"
        )

        vector_store = ChromaDB(
            collection_name="example_collection",
            persist_directory="./example_database",
            embedding=embed_client,
            top_k=3
        )

        # Add sample documents
        sample_docs = [
            DBItem(
                text="Python is a popular programming language",
                metadata={"language": "Python", "type": "programming"}
            ),
            DBItem(
                text="Machine learning enables computers to learn from data",
                metadata={"field": "ML", "type": "technology"}
            ),
            DBItem(
                text="Vector databases store data as high-dimensional vectors",
                metadata={"database": "vector", "type": "storage"}
            )
        ]

        # Add documents
        doc_ids = await vector_store.add_texts(sample_docs)
        print(f"Added {len(doc_ids)} documents")

        # Search for similar documents
        search_results = await vector_store.similarity_search("machine learning")
        print(f"\nFound {len(search_results)} similar documents:")

        for result in search_results:
            print(f"- {result.text}")
            print(f"  Metadata: {result.metadata}")

        # Get collection information
        info = vector_store.get_collection_info()
        print(f"\nCollection information: {info}")

    # Run example
    asyncio.run(main())

数据库管理
----------

清理数据库
~~~~~~~~~~~

清理向量数据库有两种方式：

1. 清除所有文档（推荐）：保留集合结构，只删除文档内容。调用方法 :py:meth:`evofabric.core.vectorstore.VectorDB.clear_db`

示例用法：

.. code-block:: python

    # clear all documents and return deleted document number
    deleted_count = await vector_store.clear_db()
    print(f"Delete {deleted_count} documents")


配置选项
--------

VectorStore 支持各种配置选项：

- **collection_name**: 集合名称（必需）
- **persist_directory**: 持久化存储目录（可选）。未设置时使用内存模式，设置后启用持久化存储
- **embedding**: 用于文本向量化的嵌入客户端（必需）
- **top_k**: 相似性搜索的默认结果数量

嵌入客户端选项
~~~~~~~~~~~~~~~~~~~~~~~~~

嵌入参数接受任何继承自 :py:class:`~evofabric.core.clients.EmbedClientBase` 的客户端：

嵌入适配器
~~~~~~~~~~~

:py:class:`~evofabric.core.vectorstore.ChromaDB` 自动提供了嵌入适配器功能，可以将自定义嵌入客户端转换为 ChromaDB 期望的接口格式。适配器支持：

- **单文本嵌入**：处理单个文本的向量化
- **批量嵌入**：同时处理多个文本的向量化
- **兼容性**：与 ChromaDB 的标准接口完全兼容

1. :py:class:`~evofabric.core.clients.SentenceTransformerEmbed` 本地句子转换器模型

   - ``model``: 模型名称（如 "all-MiniLM-L6-v2"）或本地路径
   - ``device``: "cpu" 或 "cuda" 用于 GPU 加速

2. :py:class:`~evofabric.core.clients.OpenAIEmbedClient` 基于 OpenAI API 的嵌入

   - ``model``: OpenAI 嵌入模型名称
   - ``api_key``: OpenAI API 密钥
   - ``base_url``: 自定义 API 端点（可选）

示例配置：

.. code-block:: python

    # Local SentenceTransformer with GPU
    embed_client = SentenceTransformerEmbed(
        model="all-MiniLM-L6-v2",
        device="cuda"
    )

    # OpenAI with custom endpoint
    embed_client = OpenAIEmbedClient(
        model="text-embedding-3-small",
        api_key="your-api-key",
        base_url="https://your-custom-endpoint/v1"
    )

你可以根据具体的使用场景、性能需求和可用资源来自定义这些选项。