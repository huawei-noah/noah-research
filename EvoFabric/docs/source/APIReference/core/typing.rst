.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.core.typing
===========================

.. py:module:: evofabric.core.typing

General
~~~~~~~~~~~~

.. py:data:: MISSING = object()

    空参数描述符。

Graph Related
~~~~~~~~~~~~~~~~~


.. py:class:: SpecialNode(Enum)

    图中的特殊节点枚举，用于标识图的起始节点和结束节点。

    .. py:method:: is_special_node(name: str) -> bool

        判断给定名称是否为特殊节点。

        :param name: 节点名称。
        :type name: str

        :returns: 如果是特殊节点返回 ``True``，否则返回 ``False``。
        :rtype: bool


    .. py:method:: is_end_node(name: Union[str, SpecialNode]) -> bool

        判断给定名称或节点是否为结束节点。

        :param name: 节点名称或 SpecialNode 实例。
        :type name: Union[str, SpecialNode]

        :returns: 如果是结束节点返回 ``True``，否则返回 ``False``。
        :rtype: bool


    .. py:method:: is_start_node(name: Union[str, SpecialNode]) -> bool

        判断给定名称或节点是否为起始节点。

        :param name: 节点名称或 SpecialNode 实例。
        :type name: Union[str, SpecialNode]

        :returns: 如果是起始节点返回 ``True``，否则返回 ``False``。
        :rtype: bool



.. py:class:: GraphMode(Enum)

    图执行模式的枚举，表示图的运行模式。

    .. py:attribute:: RUN = "run"

        图的正常运行模式。

    .. py:attribute:: DEBUG = "debug"

        图的调试运行模式。



.. py:class:: NodeActionMode(str, Enum)

   节点动作模式枚举类，用于定义节点在拥有多个前驱节点时的执行条件。

   .. py:attribute:: ANY = "any"

      任意模式：当收到任意前驱节点的消息时，该节点即执行。

   .. py:attribute:: ALL = "all"

      全部模式：仅当收到所有前驱节点的消息后，该节点才执行。



.. py:data:: DEFAULT_EDGE_GROUP = "all"

   默认边分组标识。

.. py:data:: STREAM_CHUNK = Any

   流式数据块类型别名，表示任意类型的流式片段。

.. py:data:: StateDelta = Dict

   状态增量类型别名，表示状态变更的字典结构。

.. py:data:: State = Union[BaseModel, Dict]

   状态类型别名，可为 ``BaseModel`` 实例或普通字典。

.. py:data:: StateSchema = Union[dict, BaseModel]

   状态 Schema 的数据类型。

Messages
~~~~~~~~~~~~~~~

.. py:class:: ChatUsage(BaseModel)

    定义 LLM 聊天客户端的使用信息。

    :param completion_tokens: 生成补全内容所使用的 token 数。
    :type completion_tokens: Optional[int]

    :param prompt_tokens: 提示词中使用的 token 数。
    :type prompt_tokens: Optional[int]

    :param total_tokens: 请求中使用的总 token 数（提示词 + 补全内容）。
    :type total_tokens: Optional[int]

    :param generation_time: 生成所耗时间（秒）。
    :type generation_time: Optional[float]



.. py:class:: EmbedUsage(BaseModel)

    定义嵌入（embedding）的使用信息。

    :param generation_time: 生成所耗时间（秒）。
    :type generation_time: int


.. py:class:: Reranking(BaseModel)

    定义重排序（reranking）的使用信息。

    :param generation_time: 生成所耗时间（秒）。
    :type generation_time: int



.. py:class:: Function(BaseModel)

    定义函数调用的信息。

    :param arguments: JSON 格式的函数参数。
    :type arguments: str

    :param name: 函数名称。
    :type name: str



.. py:class:: ToolCall(BaseModel)

    定义工具调用信息。

    :param id: 函数调用的唯一标识符。
    :type id: str

    :param function: 模型调用的函数对象。
    :type function: Function

    :param type: 调用类型。
    :type type: Literal['function']



.. py:class:: ChatStreamChunk(BaseModel)

    定义 LLM 聊天客户端的流式分块内容。

    :param reasoning_content: 推理内容。
    :type reasoning_content: Optional[str]

    :param content: 输出内容。
    :type content: Optional[str]



.. py:class:: LLMChatResponse(BaseModel)

    定义 LLM 客户端的响应格式。

    :param content: 响应内容。
    :type content: str

    :param tool_calls: 工具调用列表。
    :type tool_calls: Optional[List[ToolCall]]

    :param reasoning_content: 响应的推理内容。
    :type reasoning_content: Optional[str]

    :param usage: 使用信息。
    :type usage: Optional[ChatUsage]

    :param id: 响应的唯一标识符。
    :type id: str

    :param meta: 元信息。
    :type meta: dict



.. py:class:: EmbedResponse(BaseModel)

    定义嵌入客户端的响应格式。

    :param embeddings: 嵌入向量（浮点数列表）。
    :type embeddings: List[float]

    :param usage: 使用信息。
    :type usage: Optional[EmbedUsage]



.. py:class:: RerankResponse(BaseModel)

    定义重排序客户端的响应格式。

    :param scores: 重排序得分。
    :type scores: List[float]

    :param texts: 重排序文本。
    :type texts: List[str]

    :param usage: 使用信息。
    :type usage: Optional[RerankUsage]



.. py:class:: StateBaseMessage(BaseModel)

    表示状态消息的基类。

    :param content: 消息内容。
    :type content: Any

    :param msg_id: 消息唯一 ID（由 append_message 策略自动添加）。
    :type msg_id: Optional[str]

    :param node_name: 消息由哪个Node发出（由 :py:class:`GraphNodeSpec` 在运行时注入）。
    :type node_name: Optional[str]


.. py:class:: SystemMessage(StateBaseMessage)

    系统消息。

    **继承自** :py:class:`StateBaseMessage`

    :param content: 消息内容。
    :type content: Any

    :param msg_id: 消息唯一 ID。
    :type msg_id: Optional[str]

    :param role: 消息角色，固定为 ``system``。
    :type role: Literal['system']


.. py:class:: UserMessage(StateBaseMessage)

    用户消息。

    **继承自** :py:class:`StateBaseMessage`

    :param content: 消息内容。
    :type content: Any

    :param msg_id: 消息唯一 ID。
    :type msg_id: Optional[str]

    :param role: 消息角色，固定为 ``user``。
    :type role: Literal['user']




.. py:class:: AssistantMessage(StateBaseMessage)

    助手消息。

    **继承自** :py:class:`StateBaseMessage`

    :param content: 消息内容。
    :type content: Any

    :param msg_id: 消息唯一 ID。
    :type msg_id: Optional[str]

    :param role: 消息角色，固定为 ``assistant``。
    :type role: Literal['assistant']

    :param reasoning_content: 推理内容。
    :type reasoning_content: Optional[str]

    :param tool_calls: 工具调用列表。
    :type tool_calls: Optional[List[ToolCall]]

    :param usage: 使用信息。
    :type usage: Optional[ChatUsage]


.. py:class:: ToolMessage(StateBaseMessage)

    工具消息。

    **继承自** :py:class:`StateBaseMessage`

    :param content: 消息内容。
    :type content: Any

    :param msg_id: 消息唯一 ID。
    :type msg_id: Optional[str]

    :param tool_call_id: 工具调用 ID。
    :type tool_call_id: str

    :param role: 消息角色，固定为 ``tool``。
    :type role: Literal['tool']



.. py:class:: ToolCallResult(BaseModel)

    定义工具调用的执行结果。

    :param tool_call_id: 工具调用 ID。
    :type tool_call_id: str

    :param success: 工具是否成功返回结果。
    :type success: bool

    :param content: 工具返回的具体信息。若成功，为预期结果；若失败，为错误简述。
    :type content: Any

    :param traceback: 完整错误回溯。
    :type traceback: str


.. py:method:: cast_state_message(msg) -> StateMessage

    将输入消息转换为验证后的 :py:class:`StateMessage` 实例。

    :returns: 一个经过验证的 StateMessage 实例。
    :rtype: StateMessage


.. py:class:: StateMessage

    聊天消息类型别名，表示对话历史中可能出现的任意消息类型。
    包括 :class:`UserMessage`、:class:`ToolMessage`、:class:`AssistantMessage`、:class:`SystemMessage` 或 :class:`StateBaseMessage`。

    定义为：
    ::

        StateMessage = Union[UserMessage, ToolMessage, AssistantMessage, SystemMessage, StateBaseMessage]



Tool
~~~~~~~


.. py:class:: CodeExecDockerConfig(BaseModel)

    代码沙盒初始化配置类。

    :param image: Docker 镜像名称，默认为 ``"python:3-slim"``
    :type image: str

    :param auto_remove: 容器运行结束后是否自动删除，默认为 ``True``
    :type auto_remove: bool

    :param working_dir: 容器内的工作目录，默认为 ``"/tmp"``
    :type working_dir: str

    :param tty: 是否分配伪终端，默认为 ``True``
    :type tty: bool

    :param detach: 是否在后台运行容器，默认为 ``True``
    :type detach: bool

    :param mem_limit: 内存限制，默认为 ``"4096m"``
    :type mem_limit: str

    :param cpu_quota: CPU 配额，默认为 ``50000``
    :type cpu_quota: int

    :param entrypoint: 容器入口点，默认为 ``"/bin/sh"``
    :type entrypoint: str

    :param command: 容器启动时执行的命令，默认为 ``None``
    :type command: Union[str, List[str]]

    :param name: 容器名称，默认为 ``"evofabric_sandbox"``
    :type name: str

    :param network: 容器使用的网络模式，默认为 ``"host"``
    :type network: str

    :param volumes: 挂载卷映射，默认为 ``None``
    :type volumes: dict



.. py:class:: PromptRequest(BaseModel)

    PromptRequest 用于定义提示请求。

    :param server_name: 服务器名称。
    :type server_name: str

    :param prompt_name: 提示模板名称。
    :type prompt_name: str

    :param arguments: 参数字典。
    :type arguments: Dict[str, str]



.. py:class:: ResourceRequest(BaseModel)

    ResourceRequest 用于定义资源请求。

    :param server_name: 服务器名称。
    :type server_name: str

    :param url: 资源 URL 地址。
    :type url: str


.. py:class:: StdioLink(StdioServerParameters)

    定义 MCP 服务器的标准输入输出（Stdio）链接类型配置。

    **继承自** :py:class:`StdioServerParameters`

    :param type: 类型标识符，固定为 ``"StdioLink"``。
    :type type: Literal["StdioLink"]

    :param read_time_out: 读取超时时间，默认为 ``10.0`` 秒。
    :type read_time_out: float

    :param command: 启动服务器的执行方式（继承自父类）。
    :type command: str

    :param args: 启动服务器的指令参数（继承自父类）。
    :type args: List[str]



.. py:class:: SseLink(BaseModel)

    定义 MCP 服务器的 SSE（Server-Sent Events）链接类型配置。

    :param type: 类型标识符，固定为 ``"SseLink"``。
    :type type: Literal["SseLink"]

    :param url: SSE 服务器地址。
    :type url: str

    :param headers: 请求头，默认为 ``None``。
    :type headers: Dict[str, Any]

    :param timeout: 请求超时时间，默认为 ``30.0`` 秒。
    :type timeout: float

    :param sse_read_timeout: SSE 流读取超时时间，默认为 ``300.0`` 秒。
    :type sse_read_timeout: float



.. py:class:: StreamableHttpLink(BaseModel)

    定义 MCP 服务器的可流式传输 HTTP 链接配置。

    :param type: 类型标识符，固定为 ``"StreamableHttpLink"``。
    :type type: Literal["StreamableHttpLink"]

    :param url: HTTP 服务器地址。
    :type url: str

    :param headers: 请求头，默认为 ``None``。
    :type headers: Dict[str, Any]

    :param timeout: 请求超时时间，默认为 ``30.0`` 秒。
    :type timeout: float

    :param sse_read_timeout: SSE 流读取超时时间，默认为 ``300.0`` 秒。
    :type sse_read_timeout: float

    :param terminate_on_close: 连接关闭时是否终止，默认为 ``True``。
    :type terminate_on_close: bool



.. py:class:: MCPConfig(BaseModel)

    MCPConfig 是 MCP 的配置类。

    :param url: SSE/HTTP 传输的 URL。当使用 stdio 传输时，可以填写 MCP 服务器的绝对路径。
    :type url: str


.. py:class:: ToolInnerState(BaseModel)

    定义单个工具的内部状态。

    :param type: 工具内部状态类型，固定为 ``"ToolInnerState"``。
    :type type: Literal["ToolInnerState"]

    :param state: 工具状态内容，格式为 ``{state_name: state_content}``。
    :type state: Dict[str, Any]

    :param meta_state: 工具元状态内容，格式为 ``{state_name: state_content}``。
    :type meta_state: Dict[str, Any]



.. py:class:: ToolManagerState(BaseModel)

    定义工具管理器的整体状态。

    :param type: 状态类型，固定为 ``"ToolManagerState"``。
    :type type: Literal["ToolManagerState"]

    :param state: 工具管理状态。键为工具名称，值为对应的 :py:class:`ToolInnerState`。
    :type state: Dict[str, ToolInnerState]


.. py:class:: McpServerLink

    MCP 服务器连接类型别名，用于表示不同类型的 MCP 服务器通信方式。

    支持的连接类型：

    - :class:`StdioLink`
    - :class:`SseLink`
    - :class:`StreamableHttpLink`

    使用 `Annotated` 和 Pydantic 的 `Field` 设置了类型区分：
    ::

        McpServerLink = Annotated[
            Union[StdioLink, SseLink, StreamableHttpLink],
            Field(discriminator="type")
        ]

    说明：
        该类型用于 MCP 工具管理系统中，便于在不同通信方式的服务器间进行统一管理和调用。


VectorStore
~~~~~~~~~~~~~~

.. py:class:: DBItem

    用于向量存储和检索的基础数据库项目。

    :param text: 需要向量化的文本内容
    :type text: str
    :param ids: 数据库中的唯一标识符
    :type ids: Optional[str]
    :param metadata: 附加的元数据字典
    :type metadata: Optional[dict]

    **使用示例：**

    .. code-block:: python

        from evofabric.core.typing import DBItem

        # Create a database item
        item = DBItem(
            text="This is a sample document text",
            ids="doc_001",
            metadata={"category": "article", "author": "John Doe"}
        )

        # Access item attributes
        print(item.text)      # "This is a sample document text"
        print(item.ids)       # "doc_001"
        print(item.metadata)  # {"category": "article", "author": "John Doe"}


.. py:class:: SearchResult

    相似性搜索返回的结果数据结构。

    :param text: 文档文本
    :type text: str
    :param metadata: 关联的元数据
    :type metadata: Optional[Dict[str, Any]]
    :param score: 相似性分数（0-1，值越高表示越相似）
    :type score: Optional[float]
    :param id: 文档唯一标识符
    :type id: str

    **使用示例：**

    .. code-block:: python

        from evofabric.core.typing import SearchResult

        # Create a search result
        result = SearchResult(
            text="This is a matching document",
            metadata={"category": "article", "source": "web"},
            score=0.95,
            id="result_001"
        )

        # Access result attributes
        print(result.text)      # "This is a matching document"
        print(result.score)     # 0.95
        print(result.metadata)  # {"category": "article", "source": "web"}
