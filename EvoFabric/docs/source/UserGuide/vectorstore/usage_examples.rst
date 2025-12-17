.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

使用示例
===================

本节提供了各种场景中如何使用 VectorStore 模块的实用示例。

基本文本搜索
-------------------

简单的文本搜索功能：

.. code-block:: python

    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.typing import DBItem
    from evofabric.core.clients import SentenceTransformerEmbed

    async def main():
        # Initialize embedding client
        embed_client = SentenceTransformerEmbed(
            model="all-MiniLM-L6-v2",
            device="cpu"
        )

        # Initialize vector store
        vector_store = ChromaDB(
            collection_name="documents",
            persist_directory="./db",
            embedding=embed_client,
            top_k=5
        )

        # Add documents
        documents = [
            DBItem(text="The quick brown fox jumps over the lazy dog"),
            DBItem(text="A journey of a thousand miles begins with a single step"),
            DBItem(text="To be or not to be, that is the question")
        ]

        await vector_store.add_texts(documents)

        # Perform similarity search
        results = await vector_store.similarity_search("animal jumping")

        for result in results:
            print(f"Match: {result.text}")

    # Run the function
    if __name__ == "__main__":
        import asyncio
        asyncio.run(main())


基于元数据的搜索
--------------------

使用元数据进行过滤搜索：

.. code-block:: python

    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.typing import DBItem
    from evofabric.core.clients import SentenceTransformerEmbed

    # Initialize embedding client
    embed_client = SentenceTransformerEmbed(
        model="all-MiniLM-L6-v2",
        device="cpu"
    )

    # Initialize vector store
    vector_store = ChromaDB(
        collection_name="technical_docs",
        persist_directory="./tech_db",
        embedding=embed_client,
        top_k=5
    )

    # Add documents with metadata
    technical_docs = [
        DBItem(
            text="Introduction to machine learning algorithms",
            metadata={"category": "ML", "difficulty": "beginner", "length": "short"}
        ),
        DBItem(
            text="Advanced deep learning architectures",
            metadata={"category": "DL", "difficulty": "advanced", "length": "long"}
        ),
        DBItem(
            text="Python programming basics",
            metadata={"category": "programming", "difficulty": "beginner", "length": "medium"}
        )
    ]

    await vector_store.add_texts(technical_docs)

    # Search with metadata filter
    filter_results = await vector_store.similarity_search(
        query="learning algorithms",
        filter={"category": "ML", "difficulty": "beginner"}
    )

    for result in filter_results:
        print(f"Filtered result: {result.text} (score: {result.score})")

批处理
--------

高效处理大型文档集合：

.. code-block:: python

    import asyncio
    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.typing import DBItem

    async def process_large_dataset():
        vector_store = ChromaDB(
            collection_name="large_dataset",
            persist_directory="./large_db",
            embedding=your_embed_client,
            top_k=10
        )

        # Generate sample documents
        batch_size = 100
        all_documents = []

        for i in range(1000):
            doc = DBItem(
                text=f"Document {i}: This is sample content for document number {i}",
                metadata={"batch": i // batch_size, "doc_id": i}
            )
            all_documents.append(doc)

        # Process in batches
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i + batch_size]
            doc_ids = await vector_store.add_texts(batch)
            print(f"Processed batch {i//batch_size + 1}: {len(doc_ids)} documents")

        # Search entire dataset
        results = await vector_store.similarity_search("Find documents containing content")
        print(f"Found {len(results)} matching documents")

    asyncio.run(process_large_dataset())


文档管理
--------

添加、更新和删除文档：

.. code-block:: python

    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.typing import DBItem

    async def main():
        vector_store = ChromaDB(
            collection_name="doc_management",
            persist_directory="./doc_db",
            embedding=your_embed_client
        )

        # Add documents with IDs
        docs_with_ids = [
            DBItem(
                text="Initial document content",
                ids="doc_001",
                metadata={"status": "published"}
            ),
            DBItem(
                text="Second document content",
                ids="doc_002",
                metadata={"status": "draft"}
            )
        ]

        await vector_store.add_texts(docs_with_ids)

        # Update document by re-adding with same ID
        updated_doc = DBItem(
            text="Updated document content",
            ids="doc_001",
            metadata={"status": "published", "version": "2"}
        )

        await vector_store.add_texts([updated_doc])

        # Delete document by ID
        await vector_store.delete_by_ids(["doc_002"])

        # Verify deletion
        remaining_docs = await vector_store.similarity_search("document")
        print(f"Remaining documents: {len(remaining_docs)}")

    # Run the function
    if __name__ == "__main__":
        import asyncio
        asyncio.run(main())

RAG应用集成
-----------

向量化存储在检索增强生成系统中的应用：

.. code-block:: python

    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.typing import DBItem
    from evofabric.core.clients import SentenceTransformerEmbed, OpenAIChatClient

    class RAGSystem:
        def __init__(self):
            # Use local SentenceTransformer embedding for RAG
            self.embed_client = SentenceTransformerEmbed(
                model="all-MiniLM-L6-v2",
                device="cpu"
            )

            self.vector_store = ChromaDB(
                collection_name="knowledge_base",
                persist_directory="./rag_db",
                embedding=self.embed_client,
                top_k=3
            )

            self.llm = OpenAIChatClient(api_key="your-key")

        async def add_knowledge(self, documents):
            """Add knowledge documents to vector store"""
            db_items = [DBItem(text=doc) for doc in documents]
            await self.vector_store.add_texts(db_items)

        async def ask_question(self, question):
            """Ask a question and get RAG-enhanced answer"""
            # Retrieve relevant documents
            relevant_docs = await self.vector_store.similarity_search(question)

            if not relevant_docs:
                return "I don't have enough context to answer this question."

            # Format context
            context = "\n\n".join([doc.text for doc in relevant_docs])

            # Create prompt
            prompt = f"""
            Context:
            {context}

            Question: {question}

            Please answer the question based on the provided context.
            """

            # Get LLM response
            response = await self.llm.generate(prompt)
            return response

    # Usage example
    async def main():
        rag_system = RAGSystem()

        # Add knowledge
        knowledge_docs = [
            "EvoFabric is a distributed framework for building AI applications",
            "It supports multiple backends and provides high-performance processing",
            "The framework is designed for scalability and reliability"
        ]

        await rag_system.add_knowledge(knowledge_docs)

        # Ask question
        answer = await rag_system.ask_question("What is EvoFabric?")
        print(answer)

    # Run the example
    if __name__ == "__main__":
        import asyncio
        asyncio.run(main())


错误处理和恢复
--------------

错误处理模式：

.. code-block:: python

    import asyncio
    from evofabric.core.vectorstore import ChromaDB
    from evofabric.core.typing import DBItem

    async def robust_vector_store_operations():
        vector_store = ChromaDB(
            collection_name="robust_example",
            persist_directory="./robust_db",
            embedding=your_embed_client
        )

        try:
            # Add documents with error handling
            try:
                documents = [
                    DBItem(text="Document 1"),
                    DBItem(text="Document 2")
                ]
                doc_ids = await vector_store.add_texts(documents)
                print(f"Successfully added {len(doc_ids)} documents")
            except Exception as e:
                print(f"Error adding documents: {e}")
                # Retry logic or fallback strategy
                return

            # Search with error handling
            try:
                results = await vector_store.similarity_search("test query")
                print(f"Found {len(results)} results")
            except Exception as e:
                print(f"Error during search: {e}")
                # Implement fallback search or error response
                return

            # Get collection information with error handling
            try:
                info = vector_store.get_collection_info()
                print(f"Collection information: {info}")
            except Exception as e:
                print(f"Error getting collection information: {e}")

        except Exception as e:
            print(f"Unexpected error: {e}")
            # Global error handling and recovery

    asyncio.run(robust_vector_store_operations())
