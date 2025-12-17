.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.core.vectorstore
=================================

.. py:module:: evofabric.core.vectorstore


.. py:class:: DBBase(BaseComponent, ABC)

    定义基本向量数据库接口的抽象基类。

    :param collection_name: 数据库集合名称
    :type collection_name: str
    :param persist_directory: 持久化存储目录
    :type persist_directory: str
    :param embedding: 用于文本向量化的嵌入客户端
    :type embedding: EmbedClientBase
    :param top_k: 相似性搜索的默认结果数量
    :type top_k: int

    .. py:method:: persist()
        :async:

        将向量存储持久化到磁盘。

        :raises: NotImplementedError

    .. py:method:: clear_db() -> int
        :async:

        清除整个向量存储。

        :return: 删除的文档数量
        :rtype: int
        :raises: NotImplementedError

    .. py:method:: similarity_search(query: str, k: int = None, filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]
        :async:

        在向量存储上执行相似性搜索。

        :param query: 要搜索的查询文本
        :type query: str
        :param k: 返回的结果数量（默认：top_k）
        :type k: int, optional
        :param filter: 搜索的元数据过滤器
        :type filter: Optional[Dict[str, Any]], optional
        :return: 匹配查询的数据库项目列表
        :rtype: List[SearchResult]
        :raises: NotImplementedError

    .. py:method:: add_texts(items: Union[Sequence[DBItem], Sequence[str]], metadatas: Optional[Sequence[dict]] = None, ids: Optional[Sequence[str]] = None) -> List[str]
        :async:

        向向量存储添加新的数据库项目。

        :param items: 要添加的 DBItem 或文本列表
        :type items: Union[Sequence[DBItem], Sequence[str]]
        :param metadatas: 元数据列表（可选）
        :type metadatas: Optional[Sequence[dict]], optional
        :param ids: 文档 ID 列表（可选）
        :type ids: Optional[Sequence[str]], optional
        :return: 添加项目的 ID 列表
        :rtype: List[str]
        :raises: NotImplementedError

    .. py:method:: delete_by_ids(ids: List[str]) -> None
        :async:

        根据其原始 ID 删除项目。

        :param ids: 要删除的 ID 列表
        :type ids: List[str]
        :raises: NotImplementedError


.. py:class:: VectorDB(DBBase, ABC)

    具有高级功能的增强型向量数据库操作的抽象类。

    :param collection_name: 集合名称
    :type collection_name: str
    :param persist_directory: 持久化存储目录（可选）。如果未设置，使用内存模式。
    :type persist_directory: str, optional
    :param embedding: 嵌入函数（可选）
    :type embedding: EmbedClientBase, optional
    :param top_k: 默认搜索 top_k
    :type top_k: int

    .. py:method:: get_vector_count() -> int

        获取数据库中的向量数量。

        :return: 向量数量
        :rtype: int
        :raises: NotImplementedError

    .. py:method:: get_collection_info() -> Dict[str, Any]

        获取当前集合的信息。

        :return: 集合信息字典
        :rtype: Dict[str, Any]
        :raises: NotImplementedError


.. py:class:: ChromaDB(VectorDB)

    基于原生 ChromaDB 的向量数据库实现。

    继承自 :py:class:`VectorDB` 并实现所有抽象方法。

    :param collection_name: ChromaDB 集合名称
    :type collection_name: str
    :param persist_directory: 持久化存储目录（可选）。如果未设置，使用内存模式。
    :type persist_directory: str, optional
    :param embedding: 用于文本向量化的嵌入客户端（必需）
    :type embedding: EmbedClientBase
    :param top_k: 相似性搜索的默认结果数量
    :type top_k: int

    **嵌入客户端要求：**

    embedding 参数必须是 ``EmbedClientBase`` 的实例。框架提供两个主要实现：

    1. :py:class:`~evofabric.core.clients.SentenceTransformerEmbed` 用于本地句子转换器模型
    2. :py:class:`~evofabric.core.clients.OpenAIEmbedClient` 基于 OpenAI API 的嵌入

    .. py:method:: model_post_init(context: Any, /) -> None

        在 pydantic 验证后初始化 ChromaDB。

        :param context: Pydantic 验证上下文
        :type context: Any

    .. py:method:: persist()
        :async:

        持久化数据。在 ChromaDB 中，数据会自动持久化。

        :return: None

    .. py:method:: clear_db() -> int
        :async:

        通过删除所有文档同时保留集合来清除向量存储。

        :return: 删除的文档数量
        :rtype: int

    .. py:method:: similarity_search(query: str, k: int = None, filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]
        :async:

        执行相似性搜索并返回 :py:class:`SearchResult` 对象列表。

        :param query: 查询文本
        :type query: str
        :param k: 返回的结果数量（默认 top_k）
        :type k: int, optional
        :param filter: 元数据过滤器
        :type filter: Optional[Dict[str, Any]], optional
        :return: 搜索结果列表
        :rtype: List[SearchResult]

    .. py:method:: add_texts(items: Union[Sequence[DBItem], Sequence[str]], metadatas: Optional[Sequence[dict]] = None, ids: Optional[Sequence[str]] = None) -> List[str]
        :async:

        向向量存储添加新的数据库项目。

        :param items: DBItem 或文本列表
        :type items: Union[Sequence[DBItem], Sequence[str]]
        :param metadatas: 元数据列表（可选）
        :type metadatas: Optional[Sequence[dict]], optional
        :param ids: 文档 ID 列表（可选）
        :type ids: Optional[Sequence[str]], optional
        :return: 添加后的 ID 列表
        :rtype: List[str]

    .. py:method:: delete_by_ids(ids: List[str]) -> bool
        :async:

        根据其 ID 删除向量。

        :param ids: 要删除的 ID 列表
        :type ids: List[str]
        :return: 删除是否成功
        :rtype: bool

    .. py:method:: get_vector_count() -> int

        获取集合中存储的向量数量。

        :return: 向量数量
        :rtype: int

    .. py:method:: get_collection_info() -> Dict[str, Any]

        获取当前集合的信息。

        :return: 集合信息字典
        :rtype: Dict[str, Any]
