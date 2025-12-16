.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

========================
evofabric.core.mem
========================


.. py:module:: evofabric.core.mem

Base Memory
~~~~~~~~~~~~~~~

.. py:class:: MemBase

    基础记忆接口定义，可用于开源适配

    .. py:method:: retrieval_update(self, messages: List[StateMessage], **kwargs) -> List[StateMessage]

        检索记忆并更新上下文消息

        :param messages: 当前的上下文消息序列
        :type messages: List[StateMessage]

        :param kwargs: 其他检索或更新时所需的配置参数

        :returns: 基于记忆内容更新后的上下文消息序列
        :rtype: List[StateMessage]

    .. py:method:: add_messages(self, messages: List[StateMessage], **kwargs) -> None

        将上下文消息写入记忆向量库

        :param messages: 待写入的上下文消息序列
        :type messages: List[StateMessage]

        :param kwargs: 其他写入时所需的配置参数

        :returns: None

    .. py:method:: clear(self) -> None
        :async:

        清空全部记忆

        :returns: None

Retrieval Memory
~~~~~~~~~~~~~~~~~~~~

.. py:class:: RetrievalMem

    基础检索记忆，用于实现RAG功能

    :param vectorstore: 向量数据库实例，用于存储与检索记忆文本
    :type vectorstore: DBBase

    :param reranker: 重排序模型客户端，用于对召回结果二次排序
    :type reranker: RerankClientBase

    :param use_rerank: 是否启用重排序，默认 True
    :type use_rerank: Optional[bool]

    :param message_rounds: 保留的对话轮数，默认 1
    :type message_rounds: Optional[int]

    .. py:method:: retrieval_update(self, messages: List[StateMessage], **kwargs) -> List[StateMessage]
        :async:

        根据上下文检索记忆并更新上下文消息

        :param messages: 当前的上下文消息序列
        :type messages: List[StateMessage]

        :param kwargs: 检索或更新时所需的配置参数（可选）

        :returns: 在消息序列前插入检索结果后的新消息序列
        :rtype: List[StateMessage]

    .. py:method:: add_messages(self, messages: List[StateMessage], **kwargs) -> None
        :async:

        智能体中更新Memory的接口，在RetrievalMem中该方法不产生新增记忆

        :param messages: 待写入的上下文消息序列
        :type messages: List[StateMessage]

        :param kwargs: 写入时所需的配置参数（可选）

        :returns: None
    
    .. py:method:: add_texts(self, texts: List[str]) -> None:
        :async:

        用户自行添加记忆的接口，需要自行调用更新记忆库

        :param texts: 待写入的文本列表
        :type messages: List[str]

        :returns: None

    .. py:method:: clear(self) -> None
        :async:

        清空全部记忆

        :returns: None

Chat Memory
~~~~~~~~~~~~~~~

.. py:class:: ChatMem

    基于提示词驱动的大模型多轮对话认知记忆实现。

    :param vectorstore: 长期记忆向量数据库实例，用于存储与召回记忆文本。
    :type vectorstore: DBBase

    :param chat_client: 大模型客户端，负责记忆抽取、合并与总结的全部 LLM 调用。
    :type chat_client: ChatClientBase

    :param zh_mode: 是否启用中文提示模式；True 为中文，False 为英文，默认 True。
    :type zh_mode: Optional[bool]

    :param message_rounds: 构建检索或认知 query 时最多参考的历史对话轮数，默认 100。
    :type message_rounds: Optional[int]

    :param user_extract_prompt: 自定义“记忆特征抽取”提示词；若留空则根据 zh_mode 自动使用内置中英文模板。
    :type user_extract_prompt: Optional[str]

    :param user_update_prompt: 自定义“记忆合并更新”提示词；若留空则根据 zh_mode 自动使用内置中英文模板。
    :type user_update_prompt: Optional[str]

    :param feat_define_prompt: 额外注入的“记忆信息”提取引导提示词
    :type feat_define_prompt: Optional[str]

    .. py:method:: retrieval_update(self, messages: List[StateMessage], **kwargs) -> List[StateMessage]
        :async:

        智能体中检索接口：基于长期记忆内容生成总结，并插入到消息序列最前端返回，在智能体执行流程自动调用

        :param messages: 当前对话历史
        :type messages: List[StateMessage]

        :param kwargs: 预留扩展参数（可选）

        :returns: 新增记忆总结后的新消息序列
        :rtype: List[StateMessage]

    .. py:method:: add_messages(self, messages: List[StateMessage], **kwargs) -> None
        :async:

        智能体中储存接口：在智能体推理、工具执行结束后，自动储存记忆。

        :param messages: 当前对话历史
        :type messages: List[StateMessage]

        :param kwargs: 预留扩展参数（可选）

        :returns: None

    .. py:method:: clear(self) -> None
        :async:

        清空全部长期记忆

        :returns: None


Task Memory
~~~~~~~~~~~~

.. py:class:: TaskMem(CognitiveMem)

   基于任务执行上下文的认知记忆系统，支持任务逐步的记忆的存储、检索与经验总结。

   :param vectorstore: 长期记忆向量数据库实例，用于存储与召回任务记忆。
   :type vectorstore: DBBase

   :param chat_client: 大模型客户端，负责经验记忆总结的 LLM 调用。
   :type chat_client: ChatClientBase

   :param user_summary_prompt: 自定义"经验总结生成"提示词，用于基于召回用例生成执行经验。
   :type user_summary_prompt: Optional[str]

   :param eval_fuc: 异步评估函数，用于评估当前执行结果的正确性、得分和评价意见。
   :type eval_fuc: Callable[[List[SystemMessage]], Awaitable[tuple[bool, float, str]]]

   .. py:method:: retrieval_update(messages: List[StateMessage], **kwargs) -> List[StateMessage]
      :async:

      智能体中检索接口：基于长期记忆内容生成总结，并插入到消息序列最前端返回，在智能体执行流程自动调用

      :param messages: 当前任务执行上下文的历史消息
      :type messages: List[StateMessage]

      :param kwargs: 预留扩展参数

      :returns: 新增经验总结后的新消息序列
      :rtype: List[StateMessage]

   .. py:method:: add_messages(messages: List[StateMessage], **kwargs) -> None
      :async:

      智能体中储存接口：在智能体推理、工具执行结束后，自动储存记忆。

      :param messages: 任务执行历史消息（包含任务指令和上下文）
      :type messages: List[StateMessage]

      :param kwargs: 预留扩展参数

      :returns: None

   .. py:method:: clear() -> None
      :async:

      清空全部任务长期记忆。

      :returns: None