.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

介绍
===================

EvoFabric 中的 :py:class:`~evofabric.core.vectorstore` 模块提供了一个全面的向量数据库解决方案，专为高效的文本存储、检索和相似性搜索操作而设计。采用模块化架构构建，它支持各种向量数据库后端，并提供同步和异步 API。

概述
--------

:py:class:`~evofabric.core.vectorstore` 模块提供以下功能：

- **多后端支持**：目前包含 :py:class:`~evofabric.core.vectorstore.ChromaDB` 实现，架构可扩展以支持其他向量数据库
- **灵活集成**：与嵌入客户端的无缝集成，支持自动文本向量化
- **丰富的 API 套件**：全面的文本添加、相似性搜索、数据库管理和元数据处理方法
- **异步支持**：为高性能应用程序提供完整的异步 API 支持
- **元数据管理**：过滤和基于元数据的搜索功能

核心组件
--------

模块包含几个关键组件：

1. :py:class:`~evofabric.core.vectorstore.DBBase`: 定义基本向量数据库接口的抽象基类
2. :py:class:`~evofabric.core.vectorstore.VectorDB`：具有高级向量操作的增强抽象类
3. :py:class:`~evofabric.core.vectorstore.ChromaDB`：生产就绪的 ChromaDB 实现
4. **数据类型**: 用于结构化数据处理的 :py:class:`~evofabric.core.typing.DBItem` 和 :py:class:`~evofabric.core.typing.SearchResult`

使用场景
--------

:py:class:`~evofabric.core.vectorstore` 模块适用于：

- **检索增强生成 (RAG)**：存储和检索相关文档以提供 LLM 上下文
- **语义搜索**：在文本语料库上实现基于相似性的搜索
- **文档管理**：存储、索引和检索带有元数据的文档
- **知识库**：构建和管理知识检索系统
- **内容推荐**：基于语义相似性查找相似内容

架构优势
--------

- **模块化设计**：易于使用新的向量数据库后端进行扩展
- **类型安全**：全面的类型注解和 Pydantic 验证
- **错误处理**：错误处理和恢复机制
- **性能**：针对速度和内存效率进行优化
- **灵活性**：支持自定义嵌入函数和元数据过滤