.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.core.clients
============================

.. py:module:: evofabric.core.clients

Chat Clients
~~~~~~~~~~~~~~

.. py:class:: ChatClientBase

    获取大模型Chat模式响应的基类

    .. py:method:: create_on_stream(self, messages: Sequence[StateMessage], **kwargs) -> AsyncGenerator[ChatStreamChunk, LLMChatResponse]
        :async:

        流式获取大模型的响应。

        :param messages: 表示多轮历史对话消息序列。
        :type messages: Sequence[StateMessage]
        :param \**kwargs: 其他模型推理时需要设置的配置参数
        :returns: 一个异步生成器，流式过程中会持续返回 :py:class:`ChatStreamChunk` 对象记录流式消息，最后返回大模型的响应结果 :py:class:`LLMChatResponse`
        :rtype: AsyncGenerator[ChatStreamChunk, LLMChatResponse]

    .. py:method:: create(self, messages: Sequence[StateMessage], **kwargs) -> LLMChatResponse
        :async:

        非流式获取大模型的响应

        :param messages: 表示多轮历史对话消息序列。
        :type messages: Sequence[StateMessage]
        :param \**kwargs: 其他模型推理时需要设置的配置参数
        :returns: 大模型的响应结果 :py:class:`LLMChatResponse`


.. py:class:: OpenAIChatClient(ChatClientBase)

    基于 OpenAI 接口的 Chat 客户端实现，继承自 :py:class:`ChatClientBase` 。

    :param model: 要使用的 OpenAI 模型名称，如 "gpt-3.5-turbo"、"gpt-4" 等。
    :type model: str
    :param stream: 是否默认以流式方式请求模型；为 ``True`` 时，调用 :py:meth:`create_on_stream`；为 ``False`` 时，调用 :py:meth:`create`。
    :type stream: bool
    :param client_kwargs: 用于初始化 ``openai.AsyncOpenAI`` 客户端的额外关键字参数，例如 ``base_url``、``api_key``、``timeout`` 等。
    :type client_kwargs: Dict
    :param http_client_kwargs: 用于初始化底层 ``httpx.AsyncClient`` 的关键字参数，例如 ``proxy``、``limits``、``verify`` 等。
    :type http_client_kwargs: Dict
    :param inference_kwargs: 每次调用 ``chat.completions.create()`` 时传递的推理参数，如 ``temperature``、``top_p``、``max_tokens`` 等。
    :type inference_kwargs: Dict
    :param stream_parser: 一个异步可调用对象，用于逐块解析 OpenAI 返回的流式数据包，需满足 ``AsyncIterator[str] -> ChatStreamChunk`` 的协议。
    :type stream_parser: Callable

    .. py:method:: create_on_stream(self, messages: Sequence[StateMessage], **kwargs) -> AsyncGenerator[ChatStreamChunk, LLMChatResponse]
        :async:

        流式获取大模型的响应。

        :param messages: 表示多轮历史对话消息序列。
        :type messages: Sequence[StateMessage]
        :param kwargs: 其他模型推理时需要设置的配置参数（会覆盖类属性 inference_kwargs 中的同名参数）
        :returns: 一个异步生成器，流式过程中会持续返回 :py:class:`ChatStreamChunk` 对象记录流式消息，最后返回大模型的响应结果 :py:class:`LLMChatResponse`
        :rtype: AsyncGenerator[ChatStreamChunk, LLMChatResponse]

    .. py:method:: create(self, messages: Sequence[StateMessage], **kwargs) -> LLMChatResponse
        :async:

        非流式获取大模型的响应。

        :param messages: 表示多轮历史对话消息序列。
        :type messages: Sequence[StateMessage]
        :param \**kwargs: 其他模型推理时需要设置的配置参数（会覆盖类属性 inference_kwargs 中的同名参数）
        :returns: 大模型的响应结果 :py:class:`LLMChatResponse`
        :rtype: LLMChatResponse


.. py:class:: PanguClient(OpenAIChatClient)

    基于盘古大模型接口的 Chat 客户端实现，继承自 :py:class:`OpenAIChatClient` 。

    :param model: 要使用的模型名称。
    :type model: str

    :param stream: 是否默认以流式方式请求模型；为 ``True`` 时，调用 :py:meth:`create_on_stream`；为 ``False`` 时，调用 :py:meth:`create`。
    :type stream: bool

    :param client_kwargs: 用于初始化 ``openai.AsyncOpenAI`` 客户端的额外关键字参数，例如 ``base_url``、``api_key``、``timeout`` 等。
    :type client_kwargs: Dict

    :param http_client_kwargs: 用于初始化底层 ``httpx.AsyncClient`` 的关键字参数，例如 ``proxy``、``limits``、``verify`` 等。
    :type http_client_kwargs: Dict

    :param inference_kwargs: 每次调用 ``chat.completions.create()`` 时传递的推理参数，如 ``temperature``、``top_p``、``max_tokens`` 等。
    :type inference_kwargs: Dict

    :param stream_parser: 一个异步可调用对象，用于逐块解析盘古返回的流式数据包，需满足 ``AsyncIterator[str] -> ChatStreamChunk`` 的协议。
    :type stream_parser: Callable

    :param enable_think: 是否启用“思考”模式。
    :type enable_think: bool

    .. py:method:: create_on_stream(self, messages: Sequence[StateMessage], **kwargs) -> AsyncGenerator[ChatStreamChunk, LLMChatResponse]
        :async:

        流式获取大模型的响应。

        :param messages: 表示多轮历史对话消息序列。
        :type messages: Sequence[StateMessage]

        :param kwargs: 其他模型推理时需要设置的配置参数（会覆盖类属性 inference_kwargs 中的同名参数）

        :returns: 一个异步生成器，流式过程中会持续返回 :py:class:`ChatStreamChunk` 对象记录流式消息，最后返回大模型的响应结果 :py:class:`LLMChatResponse`

        :rtype: AsyncGenerator[ChatStreamChunk, LLMChatResponse]

    .. py:method:: create(self, messages: Sequence[StateMessage], **kwargs) -> LLMChatResponse
        :async:

        非流式获取大模型的响应。

        :param messages: 表示多轮历史对话消息序列。
        :type messages: Sequence[StateMessage]

        :param \**kwargs: 其他模型推理时需要设置的配置参数（会覆盖类属性 inference_kwargs 中的同名参数）

        :returns: 大模型的响应结果 :py:class:`LLMChatResponse`

        :rtype: LLMChatResponse


Embedding Clients
~~~~~~~~~~~~~~~~~~~~~~~


.. py:class:: EmbedClientBase

    与任意后端嵌入模型交互的基类客户端。
    兼容 LangChain OpenAI 嵌入格式，可直接用于 LangChain、ChromaDB 等生态。

    .. py:method:: embed_query(self, text: str) -> list[float]

        同步生成单段文本的嵌入向量。

        :param text: 待嵌入的文本字符串。
        :type text: str

        :returns: 长度为 `embedding_dim` 的浮点向量。
        :rtype: list[float]

    .. py:method:: embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]

        同步生成多段文本的嵌入向量列表。

        :param texts: 文本字符串列表。
        :type texts: list[str]

        :param kwargs: 额外推理参数，如 `chunk_size`、`retry` 等。
        :type kwargs: Any

        :returns: 与 `texts` 顺序对应的向量列表，每条向量长度为 `embedding_dim`。
        :rtype: list[list[float]]

    .. py:method:: aembed_query(self, text: str, **kwargs) -> list[float]
        :async:

        异步生成单段文本的嵌入向量。

        :param text: 待嵌入的文本字符串。
        :type text: str

        :param kwargs: 额外推理参数。
        :type kwargs: Any

        :returns: 长度为 `embedding_dim` 的浮点向量。
        :rtype: list[float]

    .. py:method:: aembed_documents(self, texts: list[str], **kwargs) -> AsyncGenerator[list[list[float]], None]
        :async:

        异步生成多段文本的嵌入向量列表。

        :param texts: 文本字符串列表。
        :type texts: list[str]

        :param kwargs: 额外推理参数，如 `chunk_size`、`retry` 等。
        :type kwargs: Any

        :returns: 与 `texts` 顺序对应的向量列表，每条向量长度为 `embedding_dim`。
        :rtype: list[list[float]]


.. py:class:: OpenAIEmbedClient(EmbedClientBase)

    基于 OpenAI 接口规范的嵌入模型客户端，支持同步与异步批量嵌入，兼容 Ollama 等 OpenAI-Format 后端。

    :param base_url: 请求端点 URL，默认空字符串时使用官方地址。
    :type base_url: Optional[str]
    :param api_key: 服务访问密钥，默认空字符串时尝试读取环境变量或本地配置。
    :type api_key: Optional[str]
    :param model: 要调用的嵌入模型名称，例如 ``text-embedding-3-small``。
    :type model: str
    :param dimensions: 指定返回向量的维度；模型支持降维时生效，留空则使用模型默认维度。
    :type dimensions: Optional[int]
    :param max_retries: 请求失败时的最大重试次数，默认 2。
    :type max_retries: Optional[int]
    :param request_timeout: 单次请求最长等待时间（秒），支持浮点数或 (connect, read) 元组。
    :type request_timeout: Optional[Union[float, tuple, Any]]

    .. py:method:: model_post_init(self, context: Any, /) -> None

        实例化完成后自动初始化底层 OpenAI 客户端。

    .. py:method:: embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]

        批量生成多段文本的嵌入向量。

        :param texts: 待嵌入文本列表。
        :type texts: List[str]

        :param kwargs: 额外推理参数，如 ``dimensions``、``user`` 等，将透传至底层 API。
        :type kwargs: Any

        :returns: 与输入顺序对应的向量矩阵，每行维度由模型或 ``dimensions`` 字段决定。
        :rtype: List[List[float]]

    .. py:method:: embed_query(self, text: str, **kwargs) -> List[float]

        生成单段文本的嵌入向量，内部调用 ``embed_documents`` 并返回第一条结果。

        :param text: 待嵌入文本。
        :type text: str

        :param kwargs: 额外推理参数，用法同 ``embed_documents``。
        :type kwargs: Any

        :returns: 长度为 ``dimensions``（或模型默认）的浮点向量。
        :rtype: List[float]

    .. py:method:: aembed_query(self, text: str, **kwargs) -> list[float]
        :async:

        异步生成单段文本的嵌入向量。

        :param text: 待嵌入的文本字符串。
        :type text: str

        :param kwargs: 额外推理参数。
        :type kwargs: Any

        :returns: 长度为 `embedding_dim` 的浮点向量。
        :rtype: list[float]

    .. py:method:: aembed_documents(self, texts: list[str], **kwargs) -> AsyncGenerator[list[list[float]], None]
        :async:

        异步生成多段文本的嵌入向量列表。

        :param texts: 文本字符串列表。
        :type texts: list[str]

        :param kwargs: 额外推理参数，如 `chunk_size`、`retry` 等。
        :type kwargs: Any

        :returns: 与 `texts` 顺序对应的向量列表，每条向量长度为 `embedding_dim`。
        :rtype: list[list[float]]

.. py:class:: SentenceTransformerEmbed(EmbedClientBase)

    基于本地 Sentence-Transformer 模型的轻量级嵌入客户端，无需外部 API 即可生成高质量向量，适合离线、私有化及边缘部署场景。

    :param model: 本地 Sentence-Transformer 模型名称或 HuggingFace Hub ID，例如 ``all-MiniLM-L6-v2``。
    :type model: str
    :param device: 模型运行设备，默认为 ``"cpu"``；可指定 ``"cuda"``、``"mps"`` 等以启用 GPU 加速。
    :type device: str

    .. py:method:: model_post_init(self, context: Any, /) -> None

        实例化完成后自动加载本地模型，设备与模型名由字段值注入。

    .. py:method:: embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]

        批量生成多段文本的嵌入向量。

        :param texts: 待嵌入文本列表。
        :type texts: List[str]

        :param kwargs: 额外推理参数，如 ``batch_size``、``normalize_embeddings`` 等，将透传至底层模型。
        :type kwargs: Any

        :returns: 与输入顺序对应的向量矩阵，每行维度由模型决定。
        :rtype: List[List[float]]

    .. py:method:: embed_query(self, text: str, **kwargs) -> List[float]

        生成单段文本的嵌入向量，内部调用 ``embed_documents`` 并返回第一条结果。

        :param text: 待嵌入文本。
        :type text: str

        :param kwargs: 额外推理参数，用法同 ``embed_documents``。
        :type kwargs: Any

        :returns: 长度为模型输出维度的浮点向量。
        :rtype: List[float]

    .. py:method:: aembed_query(self, text: str, **kwargs) -> list[float]
        :async:

        异步生成单段文本的嵌入向量。

        :param text: 待嵌入的文本字符串。
        :type text: str

        :param kwargs: 额外推理参数。
        :type kwargs: Any

        :returns: 长度为 `embedding_dim` 的浮点向量。
        :rtype: list[float]

    .. py:method:: aembed_documents(self, texts: list[str], **kwargs) -> AsyncGenerator[list[list[float]], None]
        :async:

        异步生成多段文本的嵌入向量列表。

        :param texts: 文本字符串列表。
        :type texts: list[str]

        :param kwargs: 额外推理参数，如 `chunk_size`、`retry` 等。
        :type kwargs: Any

        :returns: 与 `texts` 顺序对应的向量列表，每条向量长度为 `embedding_dim`。
        :rtype: list[list[float]]

Rerank Clients
~~~~~~~~~~~~~~~~~

.. py:class:: RerankClientBase(BaseComponent)

    与任意后端重排序模型交互的基类客户端，用于对“查询-文本”对进行相关性重排并返回排序后的索引序列。

    .. py:method:: rank(self, query: str, texts: List[str], **kwargs) -> List[int]
        :async:

        异步对查询与多条文本进行相关性重排序，返回按相关性从高到低排列的原始索引列表。

        :param query: 查询字符串。
        :type query: str

        :param texts: 待重排序的文本列表。
        :type texts: List[str]

        :param kwargs: 其他推理参数，如 `top_n`、`truncate`、`temperature` 等，将透传至底层模型。
        :type kwargs: Any

        :returns: 按相关性降序排列的原始文本索引列表，长度默认为 `len(texts)` 或 `top_n`（若指定）。
        :rtype: List[int]


.. py:class:: FlagRerankModel(RerankClientBase)

    基于 FlagEmbedding 的本地重排序模型实现，无需外部 API 即可对“查询-文本”对进行相关性打分与重排，适用于私有化及离线场景。

    :param model: 本地 FlagRerank 模型名称或 HuggingFace Hub ID，例如 ``BAAI/bge-reranker-base``。
    :type model: str
    :param top_n: 返回相关性最高的前 N 条索引，默认 1。
    :type top_n: int
    :param device: 模型运行设备，默认为 ``"cpu"``；可指定 ``"cuda"``、``"cuda:0"`` 等以启用 GPU 加速。
    :type device: str

    .. py:method:: model_post_init(self, context: Any, /) -> None

        实例化完成后自动加载本地重排序模型，设备与模型名由字段值注入。

    .. py:method:: rank(self, query: str, texts: List[str], **kwargs) -> List[int]
        :async:

        异步对查询与多条文本进行相关性重排序，返回按相关性降序排列的原始索引列表。

        :param query: 查询字符串。
        :type query: str

        :param texts: 待重排序的文本列表。
        :type texts: List[str]

        :param kwargs: 额外推理参数，如 `truncate`、`batch_size` 等，将透传至底层模型。
        :type kwargs: Any

        :returns: 长度不超过 `top_n` 的索引列表，按相关性从高到低排列。
        :rtype: List[int]
