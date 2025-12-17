.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.core.agent
==========================

.. py:module:: evofabric.core.agent

AgentNode
~~~~~~~~~~~


.. py:class:: AgentNode(AsyncStreamNode)

    集成记忆管理、工具调用、LLM 调用与输入输出格式化的智能体节点。继承自 :py:class:`AsyncStreamNode`

    **工作流程：**

    1. 按顺序调用记忆模块的 ``retrieval_update()`` 更新模型上下文。
    2. 按 ``input_msg_format`` 模板格式化输入。
    3. 调用 LLM Client 获取回复。
    4. 按需执行被调用的工具。
    5. 按 ``output_msg_format`` 模板格式化回复。
    6. 按顺序调用记忆模块的 ``add_messages()`` 更新记忆。

    :param client: LLM 后端客户端。
    :type client: ChatClientBase

    :param system_prompt: 系统提示词。
    :type system_prompt: str, optional

    :param inference_kwargs: 调用 LLM 时的推理参数。
    :type inference_kwargs: Dict, optional

    :param tool_manager: 工具管理器或管理器列表。
    :type tool_manager: Union[ToolManagerBase, List[ToolManagerBase]], optional

    :param memory: 记忆组件或组件列表。
    :type memory: Union[MemBase, List[MemBase]], optional

    :param output_schema: 期望 LLM 返回的 Pydantic 模型。
    :type output_schema: Optional[Type[BaseModel]], optional

    :param output_msg_format: `jinja2` 模板字符串，用于渲染 LLM 输出；需同时提供 ``output_schema``。
    :type output_msg_format: Optional[str], optional

    :param input_msg_format: `jinja2` 模板字符串，用于将输入状态格式化为单条用户消息。
    :type input_msg_format: Optional[str], optional

    .. py:method:: __call__(self, state: State, stream_writer: StreamWriter) -> StateDelta

        执行智能体推理。

        :param state: 输入状态，须含 ``messages`` 字段。
        :type state: State

        :param stream_writer: 流式输出写入器。
        :type stream_writer: StreamWriter

        :returns: 增量状态，含更新后的消息。
        :rtype: StateDelta

UserNode
~~~~~~~~~~~


.. py:class:: UserNode(AsyncNode)


    一个从终端获取用户输入并为图提供异步输入的节点。

    继承自 :class:`~evofabric.core.graph.AsyncNode`。

    UserNode 自动处理以下异常情况：

    - **EOFError**: 用户输入流结束（如文件结束符）
    - **KeyboardInterrupt**: 用户中断输入（Ctrl+C）
    - **其他异常**: 捕获所有未预期的错误

    所有异常情况下都会返回空的状态增量：``{"messages": []}``

    :param prompt_message: 显示的用户输入提示信息。默认为 ``Please enter your input: ``
    :type prompt_message: str

    :param input_key: 用户输入在状态中存储的键名。默认为 ``user_input``
    :type input_key: str


    .. py:method:: __call__(state: State) -> StateDelta

        获取用户输入并返回状态增量。

        :param state: 当前状态
        :type state: Dict
        :returns: 包含用户输入的状态增量
        :rtype: StateDelta