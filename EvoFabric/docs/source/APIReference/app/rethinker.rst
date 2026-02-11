.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.app.rethinker
==========================

.. py:module:: evofabric.app.rethinker


.. py:function:: list_ele_overwrite(old: list = MISSING, new: list = MISSING)

    对固定长度列表执行逐元素覆盖合并，并遵循最大并行规模限制。

    该函数将 ``old`` 与 ``new`` 合并为一个结果列表，其长度由 ``config.structure.num_parallel`` 决定。合并规则为逐元素覆盖：

    - 若提供 ``old``，先将其元素拷贝到结果中。
    - 若提供 ``new``，当 ``new`` 中对应位置的元素不为 ``None`` 时，覆盖结果中相同位置的元素。

    :param old: 旧的列表状态；若为 ``MISSING`` 则忽略。
    :type old: list

    :param new: 新的列表状态；若为 ``MISSING`` 则忽略。
    :type new: list

    :returns: 逐元素覆盖后的列表，长度为 ``config.structure.num_parallel``。
    :rtype: list


.. py:function:: get_agent(client: ChatClientBase, finish_type: Literal['answer', 'select'])

    根据配置构建用于编码/推理的智能体（Agent）。

    :param client: 聊天客户端基类实例，用于与模型进行交互。
    :type client: ChatClientBase

    :param finish_type: 结束类型，用于指定智能体的完成方式（例如 ``'answer'`` 或 ``'select'``）。
    :type finish_type: Literal['answer', 'select']

    :returns: 构建好的智能体对象。
    :rtype: Any


.. py:function:: build_rethinker_graph()

    构建并配置一个支持动态并行结构的计算图（Graph）。

    该图用于将单个查询拆分为多个并行分支进行生成、总结、批判与筛选，以得到最佳结果。整体结构包含：

    - 入口节点 ``dispatch``：将初始状态分发到所有并行分支。
    - 并行分支（数量为 ``config.structure.num_parallel``）：每个分支内部为串行流水线：
        - Solution 阶段：若干个 ``SolutionWithReThinkNode`` 级联，用于生成/改进候选方案。
        - Summary 阶段：一个 ``GuidedSummaryNode``，对 Solution 输出进行引导式总结。
        - Critic 阶段：若干个 ``CriticWithRethinkNode`` 级联，对总结结果进行评估与批判并提供反馈。
    - 汇聚节点 ``selector``（``ConfidenceGuideSelectNode``）：接收每个分支最后一个 critic 节点输出，并选择最佳最终结果。

    :returns: 构建完成的计算图对象。
    :rtype: Any


.. py:function:: run_rethinker_graph(query: str, query_id: Optional[str] = None, semaphore: Optional[asyncio.Semaphore] = None)
    :async:

    异步运行 rethinker 执行图以处理单条查询。

    该函数会构建 rethinker 计算图，为每个查询初始化日志/输出目录，将查询元数据注入到流式上下文（stream context）中，并以异步方式执行整个图。

    当配置 ``config.exp.output_root`` 被设置时，输出文件将按如下结构组织：

    - ``output_root/``
        - ``qid00001/``
            - ``node1.json``
            - ``node2.json``
            - ...
            - ``result.json``
        - ``qid00002/``
        - ...

    :param query: 需要由 rethinker 图处理的输入查询。
    :type query: str

    :param query_id: 查询的唯一标识；若未提供则自动生成 UUID。
    :type query_id: Optional[str]

    :param semaphore: 用于同步/限制并发执行的信号量。
    :type semaphore: Optional[asyncio.Semaphore]

    :returns: rethinker 执行图返回的最终结果。
    :rtype: Any





.. py:class:: LLMConfig

    大语言模型（LLM）调用相关的配置模型，用于描述模型名称、鉴权信息、请求参数与推理参数等。

    .. py:attribute:: model_name

        使用的 LLM 模型名称。

        :type: str

    .. py:attribute:: api_key

        OpenAI 兼容后端通常需要的 API Key，用于鉴权。

        :type: str

    .. py:attribute:: base_url

        LLM API 的基础地址（Base URL）。

        :type: str

    .. py:attribute:: csb_token

        CSB 鉴权令牌，用于基于 CSB Token 的授权方式。

        :type: Optional[str]

    .. py:attribute:: max_retries

        针对瞬时失败请求的最大重试次数。

        :type: int

    .. py:attribute:: timeout

        请求超时时间（秒）。若为 ``None`` 则使用后端默认值。

        :type: Optional[int]

    .. py:attribute:: top_p

        核采样（Nucleus Sampling）参数，用于控制累计概率质量。

        :type: Optional[float]

    .. py:attribute:: temperature

        采样温度，用于控制生成结果的随机性。

        :type: Optional[float]

    .. py:attribute:: max_tokens

        最大生成 token 数；若为 ``None`` 则使用后端默认值。

        :type: Optional[int]

    .. py:attribute:: extra_body

        额外的、后端特定的请求体参数。

        :type: dict

    .. py:attribute:: fast_think

        是否启用快速思考模式：通过在最后一条消息追加 ``/no_think`` 来触发。

        :type: bool

    .. py:attribute:: output_logp

        是否在响应中返回 token 级别的对数概率（log probabilities）。

        :type: bool

    .. py:attribute:: stop_condition

        流式输出的停止条件字符串；当匹配到该字符串时终止生成。

        :type: Optional[str]

    .. py:attribute:: http_client_kwargs

        传递给底层 OpenAI HTTP Client 的关键字参数。

        :type: Optional[dict]

    .. py:attribute:: stream

        是否启用流式输出；当设置了 ``stop_condition`` 时必须为 ``True``。

        :type: bool

    .. py:method:: create_client_kwargs(self) -> dict

        生成用于初始化底层客户端（Client）的参数字典。

        该方法会包含 ``api_key``、``base_url``、``max_retries``、``timeout`` 等基础参数；
        当设置了 ``csb_token`` 时，会在 ``default_headers`` 中追加 ``csb-token`` 头。

        :returns: 客户端初始化参数字典。
        :rtype: dict

    .. py:method:: create_inference_kwargs(self) -> dict

        生成用于推理调用（Inference）的参数字典。

        该方法会包含 ``temperature``、``top_p`` 等采样参数；当 ``extra_body`` 非空时会注入 ``extra_body``；
        当 ``output_logp`` 为 ``True`` 时，会开启 ``logprobs`` 并设置 ``top_logprobs``；
        当 ``max_tokens`` 非 ``None`` 时，会设置 ``max_tokens``。

        :returns: 推理调用参数字典。
        :rtype: dict



.. py:class:: WebParserConfig

    网页解析相关配置，用于控制抓取、解析与内容截断策略。

    .. py:attribute:: llm_input_max_char

        网页解析后，单个文档允许输入到模型的最大字符预算。

        :type: int

    .. py:attribute:: model

        用于网页内容解析或下游推理的后端模型名称。

        :type: str

    .. py:attribute:: use_jina

        是否启用基于 Jina 的网页内容解析。

        :type: bool

    .. py:attribute:: jina_api_key

        Jina 服务的 API Key；当 ``use_jina`` 启用时通常需要。

        :type: Optional[str]

    .. py:attribute:: ssl_verify

        发起 HTTP 请求时是否校验 SSL 证书。

        :type: bool

    .. py:attribute:: timeout

        拉取网页内容的网络请求超时（秒）。

        :type: int

    .. py:attribute:: show_url_content_max_char

        每个 URL 允许保留的最大字符数；小于 0 表示不包含 URL 内容。

        :type: int



.. py:class:: WebSearchConfig

    Web 搜索相关配置，用于控制检索服务鉴权与请求行为。

    .. py:attribute:: serper_api_key

        Serper 服务的 API Key；用于启用 web search 请求。

        :type: Optional[str]

    .. py:attribute:: retries

        Web 搜索请求失败时的最大重试次数。

        :type: int

    .. py:attribute:: ssl_verify

        Web 搜索 HTTP 请求时是否校验 SSL 证书。

        :type: bool

    .. py:attribute:: timeout

        单次 Web 搜索调用的请求超时（秒）。

        :type: int



.. py:class:: SolutionConfig

    解题/推理阶段相关配置，包含求解、反思、总结、选择等阶段的模型与规则设置。

    .. py:attribute:: solver_model

        解题（solution generation）阶段使用的 LLM 后端模型。

        :type: str

    .. py:attribute:: critic_model

        反思/反思（critic evaluation）阶段使用的 LLM 后端模型。

        :type: str

    .. py:attribute:: summary_model

        引导式总结（guided summaries）阶段使用的 LLM 后端模型。

        :type: str

    .. py:attribute:: selector_model

        选择（selection）阶段使用的 LLM 后端模型。

        :type: str

    .. py:attribute:: selector_iteration

        选择阶段的迭代次数，用于逐步收敛与优化选择结果。

        :type: int

    .. py:attribute:: stop_condition

        用于从 LLM 输出中抽取代码块的正则表达式模式。

        :type: str

    .. py:attribute:: selection_condition

        用于从 LLM 输出中抽取被选择响应序号的正则表达式模式。

        :type: str

    .. py:attribute:: answer_condition

        用于从 LLM 输出中抽取最终答案内容的正则表达式模式。

        :type: str

    .. py:attribute:: max_agent_step

        单次 agent 执行允许的最大推理/动作步数。

        :type: int

    .. py:attribute:: tool_timeout

        调用单个工具或外部服务的超时时间（秒）。

        :type: int

    .. py:attribute:: max_empty_response

        允许的连续空响应最大次数，超过后将中止 agent 执行。

        :type: int

    .. py:attribute:: chat_template

        多轮对话序列化模板字符串；仅用于序列化，不作为 LLM 提示词。

        :type: Optional[str]



.. py:class:: ExpConfig

    实验级配置，包含数据集输入、输出目录与并发限制等。

    .. py:attribute:: input_file_path

        实验使用的输入数据集文件路径。

        :type: str

    .. py:attribute:: exp_name

        实验名称，通常作为输出根目录下的子目录名。

        :type: str

    .. py:attribute:: output_root

        输出根目录；若为 ``None`` 则不写入缓存或输出。

        :type: Optional[str]

    .. py:attribute:: max_question_thread_limit

        允许并发执行的问题最大数量。

        :type: int

    .. py:attribute:: max_node_thread_limit

        执行过程中允许并发运行的节点最大数量。

        :type: int



.. py:class:: StructureConfig

    推理结构相关配置，用于控制并行候选数量与各阶段迭代次数。

    .. py:attribute:: num_parallel

        每个任务并行生成的 solution 候选数量。

        :type: int

    .. py:attribute:: num_solution_iteration

        每个候选 solution 的生成迭代次数。

        :type: int

    .. py:attribute:: num_critic_iteration

        反思阶段的迭代次数。

        :type: int



.. py:class:: PromptConfig

    提示词模板配置，集中管理解析、求解、反思、总结与选择等阶段的 prompt。

    .. py:attribute:: web_parser_pdf

        用于解析 PDF 网页内容的提示词模板。

        :type: str

    .. py:attribute:: web_parser_html

        用于解析 HTML 网页内容的提示词模板。

        :type: str

    .. py:attribute:: solver_user_prompt

        初次解题阶段的用户提示词模板。

        :type: str

    .. py:attribute:: solver_twice_user_prompt

        二次解题（second-pass solution generation）的用户提示词模板。

        :type: str

    .. py:attribute:: critic_user_prompt

        初次反思阶段的用户提示词模板。

        :type: str

    .. py:attribute:: critic_twice_user_prompt

        二次反思（second-pass critique/evaluation）的用户提示词模板。

        :type: str

    .. py:attribute:: guided_summary_prompt

        引导式总结（guided summaries）的提示词模板。

        :type: str

    .. py:attribute:: selector_user_prompt

        选择阶段的用户提示词模板。

        :type: str

    .. py:attribute:: selector_iteration_user_prompt

        迭代选择（iterative selection refinement）的用户提示词模板。

        :type: str



.. py:class:: GraphConfig

    运行图（Graph）整体配置，聚合 LLM 资源、网页解析/搜索、解题阶段、结构参数、实验参数与提示词模板。

    .. py:attribute:: llm_resources

        可用的 LLM 后端资源字典；键为模型名，值为对应的 ``LLMConfig`` 配置对象。

        :type: dict[str, LLMConfig]

    .. py:attribute:: web_parser

        网页内容解析配置。

        :type: WebParserConfig

    .. py:attribute:: web_search

        Web 搜索配置。

        :type: WebSearchConfig

    .. py:attribute:: solution

        解题、反思、总结与选择阶段的配置集合。

        :type: SolutionConfig

    .. py:attribute:: structure

        迭代推理图结构配置，包括并行度与选择器行为等。

        :type: StructureConfig

    .. py:attribute:: exp

        实验级配置，包括 I/O 路径、输出目录与线程限制等。

        :type: ExpConfig

    .. py:attribute:: prompts

        实验全流程使用的提示词模板集合（解析、求解、反思、选择等）。

        :type: PromptConfig



.. py:class:: Config

    实验配置管理器。

    该类负责加载并访问完整实验配置：既支持从 YAML 文件加载，也支持直接加载一个 ``GraphConfig`` 实例。
    它提供了对各子配置（LLM 资源、网页解析/搜索、解题设置、结构、实验 I/O、提示词等）的便捷访问，
    并维护一个全局 semaphore 用于控制执行过程中的节点级并发。

    .. py:property:: graph

        获取完整的 ``GraphConfig`` 配置对象（需先加载配置）。

        :returns: 当前已加载的图配置。
        :rtype: GraphConfig

    .. py:property:: llm_resources

        获取 LLM 后端资源配置（需先加载配置）。

        :returns: LLM 资源字典。
        :rtype: dict[str, LLMConfig]

    .. py:property:: web_parser

        获取网页解析配置（需先加载配置）。

        :returns: 网页解析配置对象。
        :rtype: WebParserConfig

    .. py:property:: web_search

        获取 web 搜索配置（需先加载配置）。

        :returns: web 搜索配置对象。
        :rtype: WebSearchConfig

    .. py:property:: solution

        获取解题阶段配置（需先加载配置）。

        :returns: 解题阶段配置对象。
        :rtype: SolutionConfig

    .. py:property:: structure

        获取结构配置（需先加载配置）。

        :returns: 结构配置对象。
        :rtype: StructureConfig

    .. py:property:: exp

        获取实验级配置（需先加载配置）。

        :returns: 实验级配置对象。
        :rtype: ExpConfig

    .. py:property:: prompts

        获取提示词模板配置（需先加载配置）。

        :returns: 提示词模板配置对象。
        :rtype: PromptConfig

    .. py:method:: load(self, config_path: Union[str, Path]) -> None

        从 YAML 文件加载实验配置。

        :param config_path: YAML 配置文件路径。
        :type config_path: Union[str, Path]

        :raises FileNotFoundError: 当配置文件不存在时抛出。

    .. py:method:: loads(self, config: GraphConfig) -> None

        从已有的 ``GraphConfig`` 实例加载实验配置。

        :param config: 预先构造好的 ``GraphConfig`` 实例。
        :type config: GraphConfig

    .. py:method:: get_semaphore(self) -> asyncio.Semaphore

        获取用于节点级并发控制的全局 semaphore。

        若尚未初始化，会根据 ``exp.max_node_thread_limit`` 创建并缓存一个 ``asyncio.Semaphore``。

        :returns: 限制并发节点执行数量的 ``asyncio.Semaphore`` 对象。
        :rtype: asyncio.Semaphore



.. py:data:: config

    全局单例配置管理器实例，用于管理实验配置的加载与访问。

    :type: Config


.. py:function:: repeat_prompt(prompt: str, repeat_time: int = 3) -> str

    将同一段提示词按指定次数重复拼接，并在重复段之间插入固定的英文提示语作为过渡。

    :param prompt: 需要被重复的提示词文本。
    :type prompt: str

    :param repeat_time: 重复次数，必须大于等于 1，默认为 ``3``。
    :type repeat_time: int, optional

    :raises ValueError: 当 ``repeat_time < 1`` 时抛出异常。

    :returns: 按重复次数拼接后的完整提示词字符串，段落之间使用两个换行分隔。
    :rtype: str



.. py:function:: generate_stop_condition(pattern: str)

    根据给定的正则表达式模式生成一个“流式停止条件”函数。

    生成的停止条件函数会在流式内容中使用 ``re.finditer``（带 ``re.DOTALL``）进行匹配；
    一旦检测到至少一个匹配项，即返回 ``True``，用于指示应当停止继续生成/解析。

    :param pattern: 用于匹配流式内容的正则表达式模式。
    :type pattern: str

    :returns: 一个停止条件函数。该函数签名为 ``(content: str) -> bool``，当检测到匹配时返回 ``True``。
    :rtype: callable



.. py:class:: FastSlowThinkOpenAIChatClient

    支持“快速思考”模式的 OpenAI Chat Client 扩展实现。

    当启用 ``fast_think`` 时，会在最后一条输入消息的 ``content`` 末尾追加 ``/no_think``，
    用于触发后端的快速模式（跳过或减少推理输出等）。

    .. py:attribute:: fast_think

        是否启用快速思考模式。

        :type: bool


.. py:function:: get_client(model: str)

    获取指定模型名称对应的 OpenAI Chat Client 实例。

    该函数会从全局配置中的 ``llm_resources`` 查找对应的 ``LLMConfig``，
    并据此构造 ``FastSlowThinkOpenAIChatClient`` 的初始化参数（模型名、流式开关、客户端参数、推理参数等）。
    若配置中存在 ``stop_condition``，会额外注入 ``stream_parser``，使其在流式解析时能够基于停止条件提前终止。

    :param model: 模型标识名称（用于在配置的 LLM 资源字典中检索）。
    :type model: str

    :raises ValueError: 当 ``model`` 不存在于配置的 LLM 资源中时抛出异常。

    :returns: 对应模型的 Chat Client 实例。
    :rtype: FastSlowThinkOpenAIChatClient


.. py:class:: CodingAgentResult

    编码/推理代理的执行结果数据模型。

    该结果对象用于汇总一次 agent 执行过程中的上下文信息、步数统计、耗时、
    完整对话日志、每轮客户端原始响应，以及（可选的）最终渲染输出文本。

    .. py:attribute:: ctx

        当前图节点执行过程中捕获的执行上下文。

        :type: StreamCtx

    .. py:attribute:: total_steps

        agent 在终止前完成的推理/动作总步数。

        :type: int

    .. py:attribute:: generation_time

        agent 端到端执行耗时（秒），包含 LLM 调用与工具调用。

        :type: float

    .. py:attribute:: agent_logs

        agent 维护的完整对话轨迹日志。

        :type: list

    .. py:attribute:: client_responses

        每次迭代由 LLM 客户端返回的原始响应列表。

        :type: list

    .. py:attribute:: response

        基于 agent 日志并使用配置的聊天模板渲染得到的最终输出文本。

        如果未提供聊天模板，则该字段为 ``None``。

        :type: Optional[str]


.. py:class:: CodingAgent

    一个由大语言模型驱动的自治编码代理，支持多步迭代推理并可原生执行 Python 代码。

    该代理会维护完整的对话轨迹，并重复执行以下流程，直至满足终止条件：

    1. 调用 LLM 后端生成响应；
    2. 从模型输出中解析结构化信号（例如代码块或最终答案）；
    3. 通过用户提供的执行器运行生成的 Python 代码；
    4. 将执行结果回注入对话上下文。

    主要能力包括：

    - 可配置的多步迭代推理（上限步数可控）
    - 结构化“工具内容（Python 代码）”提取与执行
    - 当模型输出为空或无效时自动使用兜底提示词推动结束
    - 基于正则模式检测最终答案
    - 详细的逐步日志记录与执行追踪

    该代理被设计为可在更大的执行图/工作流系统中，以异步可调用节点的形式使用。

    .. py:attribute:: client

        用于在每个 agent 步骤中发起生成请求的 LLM 对话客户端。

        :type: ChatClientBase

    .. py:attribute:: max_agent_step

        agent 允许的最大推理/动作步数。

        :type: int

    .. py:attribute:: max_empty_response

        允许的连续空响应最大次数。

        “空响应”指输出既不匹配代码模式，也不匹配答案模式的情况。

        :type: int

    .. py:attribute:: tool_timeout

        工具（Python 代码）执行超时时间（秒）。

        :type: int

    .. py:attribute:: force_finish_prompt_candidates

        当 agent 产生空响应时使用的兜底提示词候选列表，用于促使模型给出最终答案并结束对话。

        :type: list

    .. py:attribute:: tool_content_pattern

        用于从 agent 响应中提取工具内容（代码）的正则表达式模式。

        :type: str

    .. py:attribute:: answer_pattern

        用于从 agent 响应中提取最终答案内容的正则表达式模式。

        :type: str

    .. py:attribute:: chat_template

        用于将 agent 交互历史格式化为单一字符串的聊天模板。

        若为 ``None``，则不会输出最终渲染的 ``response`` 字段。

        :type: str

    .. py:attribute:: py_exec_handler

        用于执行生成的 Python 代码的处理函数，通常以 ``(code: str, timeout: int)`` 作为输入并异步返回执行结果对象。

        :type: Callable[[str, int], Awaitable[BaseModel]]

    .. py:method:: __call__(self, prompt: str) -> CodingAgentResult
        :async:

        使用单个输入提示词运行编码代理。

        该方法会将 ``prompt`` 作为对话中的第一条用户消息写入内部历史记录，
        并开始进行后续的推理、工具提取与执行、结果回注入，直到满足终止条件为止。

        :param prompt: 提供给 agent 的初始用户提示词，作为整个对话与推理流程的起点。
        :type prompt: str

        :returns: 结构化的 agent 执行结果对象，包含执行上下文、步数统计、耗时、完整日志、每轮原始响应以及可选的最终渲染输出。
        :rtype: CodingAgentResult



.. py:class:: AsyncNodeWithCacheAndConcurrencyLimit

    带有内置缓存与并发控制能力的异步节点基类。

    该类在节点执行外层包装了两类通用能力：

    1. **缓存机制**：通过复用此前已持久化的结果，避免重复计算。
    2. **并发限制**：使用全局 semaphore 控制同时执行的最大节点数量。

    子类应实现 ``_run`` 方法来定义实际的节点逻辑；该逻辑本身无需关心缓存与并发控制。

    .. py:attribute:: agent

        编码代理实例，可用于通过多轮迭代执行 Python 代码来完成任务（可选）。

        :type: Optional[CodingAgent]

    .. py:attribute:: cache_dir_key

        在 ``state`` 中用于获取缓存目录路径的字段名（key）。

        若能从 ``state`` 中取到该字段对应的缓存目录，则会启用磁盘缓存；
        默认值为 ``"cache_dir"``。

        :type: Optional[str]

    .. py:method:: __call__(self, state: State) -> StateDelta
        :async:

        在缓存与并发控制的保护下执行节点。

        执行流程如下：

        1. 获取全局 semaphore 并进入异步上下文，以施加并发上限。
        2. 若 ``cache_dir_key`` 存在且能够从 ``state`` 中取到缓存目录，则：
           - 根据当前节点名构造缓存文件路径（``<node_name>.json``）；
           - 若缓存文件已存在，则直接加载并返回缓存结果；
           - 否则执行 ``_run``，并将结果写入缓存文件后返回。
        3. 若无法启用缓存（未配置或无法从 ``state`` 获取缓存目录），则直接执行 ``_run`` 并返回结果。

        :param state: 当前执行状态对象。
        :type state: State

        :returns: 该节点产生的状态增量（或等价的变更结果）。
        :rtype: StateDelta


    .. py:method:: _run(self, state: State) -> StateDelta
        :async:

        执行节点的核心逻辑（由子类实现）。

        该方法仅包含实际的计算或模型调用逻辑，不需要处理缓存与并发控制。
        基类实现中会直接抛出 ``NotImplementedError``，用于提示必须由子类覆盖实现。

        :param state: 当前执行状态对象。
        :type state: State

        :raises NotImplementedError: 当子类未实现该方法时抛出异常。

        :returns: 该节点产生的状态增量（或等价的变更结果）。
        :rtype: StateDelta



.. py:class:: SolutionWithReThinkNode

    支持“复盘/再思考（ReThink）”的解题节点实现，继承自 ``AsyncNodeWithCacheAndConcurrencyLimit``。

    该节点会根据是否提供 ``last_round`` 来决定生成提示词的方式：

    - 当 ``last_round`` 为空时，表示第一轮解题，使用 ``config.prompts.solver_user_prompt`` 构造提示词；
    - 当 ``last_round`` 非空时，从上一轮结果中取出指定 ``index`` 的响应内容，解析后注入到
      ``config.prompts.solver_twice_user_prompt`` 中，用于二次解题/再思考。

    最终会调用 ``agent`` 执行提示词，并将结果以并行槽位列表的形式写入 ``output_key`` 对应字段。

    .. py:attribute:: last_round

        指示从哪一轮的结果中获取上一轮答案的 key。

        - 若为 ``None``：表示第一轮，不进行再思考；
        - 若非 ``None``：表示从 ``state[last_round]`` 中读取上一轮答案并进行再思考。

        :type: Optional[str]

    .. py:attribute:: output_key

        输出结果在状态增量中的存储 key，默认值为 ``"solution"``。

        :type: str

    .. py:method:: _run(self, state: State) -> StateDelta
        :async:

        执行解题（或再思考）逻辑并返回状态增量。

        执行要点：

        - 从 ``state`` 读取 ``query`` 与 ``index``；
        - 若为第一轮（``last_round`` 为空），构造初次求解提示词；
        - 若为再思考轮次（``last_round`` 非空），从上一轮结果中取出当前 ``index`` 的 ``response``，
          经解析后注入到二次求解提示词模板；
        - 调用 ``agent`` 异步生成结果；
        - 构造长度为 ``config.structure.num_parallel`` 的列表，仅在当前位置 ``index`` 放入本次结果，
          并以 ``{output_key: solution}`` 的形式返回。

        :param state: 当前执行状态对象。
        :type state: State

        :returns: 包含本节点输出的状态增量，键为 ``output_key``，值为并行槽位列表。
        :rtype: StateDelta



.. py:class:: CriticWithRethinkNode

    支持“复盘/再思考（ReThink）”的反思（Critic）节点实现，继承自 ``AsyncNodeWithCacheAndConcurrencyLimit``。

    该节点用于对某一轮的解题结果进行批判性反思/反思，并可选地基于上一轮反思结果进行二次反思（再思考）：

    - 从 ``input_key`` 指定的状态字段中取出待反思的解题内容；
    - 对内容进行清洗（移除思考/执行片段等）后注入反思提示词模板；
    - 若 ``last_round`` 为空则使用初次反思提示词 ``config.prompts.critic_user_prompt``；
      否则读取上一轮反思结果，裁剪后注入二次反思提示词 ``config.prompts.critic_twice_user_prompt``；
    - 调用 ``agent`` 获取反思输出，并以并行槽位列表写入 ``output_key``。

    .. py:attribute:: input_key

        从状态中读取“需要被反思/批判”的内容的 key。

        :type: Optional[str]

    .. py:attribute:: last_round

        上一轮反思结果所在的 key。

        - 若为 ``None``：表示第一轮反思，不进行再思考；
        - 若非 ``None``：表示读取指定轮次的反思结果并进行二次反思/再思考。

        :type: Optional[str]

    .. py:attribute:: output_key

        反思（再思考）结果在状态增量中的存储 key。

        :type: Optional[str]

    .. py:method:: _run(self, state: State) -> StateDelta
        :async:

        执行反思（或二次反思/再思考）逻辑并返回状态增量。

        执行要点：

        - 从 ``state`` 读取 ``query`` 与 ``index``；
        - 从 ``state[input_key][index]["response"]`` 读取待反思解题输出，并进行清洗；
        - 若 ``last_round`` 为 ``None``，构造初次反思提示词；
        - 否则读取 ``state[last_round][index]["response"]`` 作为上一轮反思结果，解析并裁剪后注入二次反思提示词；
        - 调用 ``agent`` 异步生成反思结果；
        - 构造长度为 ``config.structure.num_parallel`` 的列表，仅在 ``index`` 槽位放入本次反思结果，
          并以 ``{output_key: critic}`` 形式返回。

        :param state: 当前执行状态对象。
        :type state: State

        :returns: 包含本节点反思输出的状态增量，键为 ``output_key``，值为并行槽位列表。
        :rtype: StateDelta



.. py:class:: ConfidenceGuideSelectNode

    基于“置信度引导”的选择节点实现，继承自 ``AsyncNodeWithCacheAndConcurrencyLimit``。

    该节点用于在多个候选响应中进行选择，并维护完整的选择历史。整体流程通常包括：

    - 从 ``input_key`` 读取候选结果列表，并对每个候选进行答案解析；
    - 基于候选数量构造拉丁方（latin square）用于组织比较/选择顺序；
    - 进行初始化选择流程（``_init_select``）与迭代选择流程（``_iterative_select``）；
    - 汇总历史选择结果，提取得票/排名靠前的候选集合；
    - 若存在多个并列候选，则进入最终选择流程（``_final_select``）以决出唯一结果；
    - 将选择历史、最终选中结果与候选列表写入 ``output_key``。

    .. py:attribute:: input_key

        从状态中读取候选解（或中间结果）列表的 key，用于后续选择。

        :type: Optional[str]

    .. py:attribute:: output_key

        选择结果在状态增量中的存储 key，默认值为 ``"selector"``。

        :type: str

    .. py:attribute:: repeat_prompt

        提示词重复次数配置（用于控制选择阶段提示词的重复/强调程度）。

        :type: int

    .. py:method:: _run(self, state: State) -> StateDelta
        :async:

        执行候选结果的选择流程并返回状态增量。

        执行要点：

        - 从 ``state`` 读取 ``query``；
        - 从 ``state[input_key]`` 读取候选列表，并将每个候选的 ``response`` 解析为纯答案文本；
        - 依据候选数量构造拉丁方，用于组织比较/选择顺序；
        - 依次执行初始化选择与迭代选择，累积 ``_response_history``；
        - 计算当前选择历史中排名靠前的候选集合：
          若候选集合大小大于 1，则执行最终选择以获得唯一结果；
        - 返回一个字典，键为 ``output_key``，其值包含：
          ``selection_history``（选择历史）、``selected_response``（最终选中响应）与 ``response_list``（候选答案列表）。

        :param state: 当前执行状态对象。
        :type state: State

        :returns: 包含选择结果的状态增量，键为 ``output_key``。
        :rtype: StateDelta


.. py:class:: GuidedSummaryNode

    引导式总结节点实现，继承自 ``AsyncNodeWithCacheAndConcurrencyLimit``。

    该节点用于对候选解答进行“引导式总结”生成：

    - 从 ``input_key`` 指定的状态字段中读取待总结的解题输出；
    - 对输出内容进行清洗（去除不需要的响应结构/标记等）；
    - 使用 ``config.prompts.guided_summary_prompt`` 构造总结提示词；
    - 通过 ``client`` 发起一次对话生成请求得到总结内容；
    - 将总结结果以并行槽位列表形式写入 ``output_key``。

    .. py:attribute:: client

        用于生成总结内容的 LLM 对话客户端。

        :type: ChatClientBase

    .. py:attribute:: input_key

        从状态中读取待总结内容的 key。

        :type: str

    .. py:attribute:: output_key

        生成的总结结果在状态增量中的存储 key。

        :type: str


    .. py:method:: _run(self, state: State) -> StateDelta
        :async:

        生成引导式总结并返回状态增量。

        执行要点：

        - 从 ``state`` 读取 ``query`` 与 ``index``；
        - 从 ``state[input_key][index]["response"]`` 读取待总结内容并进行清洗；
        - 将问题与解答注入 ``guided_summary_prompt`` 模板构造提示词；
        - 调用 ``client.create`` 生成总结，并将返回内容整理为字典结构：
          其中 ``response["response"]`` 会被设置为 ``response["content"]`` 以统一字段命名；
        - 构造长度为 ``config.structure.num_parallel`` 的列表，仅在 ``index`` 槽位写入该总结结果；
        - 以 ``{output_key: solution_summary}`` 的形式返回。

        :param state: 当前执行状态对象。
        :type state: State

        :returns: 包含总结结果的状态增量，键为 ``output_key``，值为并行槽位列表。
        :rtype: StateDelta


.. py:class:: DispatchNode

    同步分发节点，用于在执行图中充当占位或分发入口节点。

    当前实现的 ``__call__`` 不对状态做任何修改，直接返回空的状态增量。

    .. py:method:: DispatchNode.__call__(self, state: State) -> StateDelta

        执行分发节点逻辑并返回状态增量。

        :param state: 当前执行状态对象。
        :type state: State

        :returns: 空的状态增量字典。
        :rtype: StateDelta


.. py:function:: get_dispatch_filter(index: int)

    生成一个用于设置 ``state.index`` 的分发过滤函数。

    返回的过滤函数会将传入的 ``state`` 的 ``index`` 字段设置为指定值，
    并返回更新后的 ``state``，常用于将并行任务分发到不同的槽位索引。

    :param index: 要写入到 ``state.index`` 的索引值。
    :type index: int

    :returns: 一个过滤函数，签名为 ``(state: State) -> State``。
    :rtype: callable



.. py:class:: BaseBenchmarkEvaluator

    基准评测的抽象基类。

    该类提供一套通用的评测流水线，覆盖数据与结果的匹配、并行评测、异常兜底、
    以及统计汇总与落盘保存等能力。子类必须实现 ``evaluate_item`` 方法来定义
    具体基准（benchmark）的评测逻辑。

    .. py:attribute:: data_file

        源数据集文件路径（JSON）。

        :type: Path

    .. py:attribute:: result_root

        模型生成结果所在的根目录。

        :type: Path

    .. py:attribute:: eval_llm

        作为裁判（judge）的 LLM 配置。

        :type: LLMConfig

    .. py:attribute:: max_workers

        并行评测的最大线程数。

        :type: int

    .. py:attribute:: max_char

        注入到裁判提示词中的响应文本最大保留字符数（用于截断）。

        :type: int

    .. py:attribute:: max_completion_tokens

        裁判模型输出允许的最大 token 数。

        :type: int

    .. py:attribute:: save_path

        评测结果保存路径。

        :type: Path

    .. py:attribute:: result_extractor

        可选的结果抽取函数，用于从单个问题目录中提取模型预测。

        若为 ``None``，则使用默认抽取逻辑；该字段被排除在序列化之外（函数不可序列化）。

        :type: Optional[Callable[[str], Optional[str]]]


    .. py:method:: model_post_init(self, __context: Any) -> None

        模型初始化后的钩子方法，用于完成运行期初始化与校验。

        该方法会：

        - 创建并缓存裁判用的 OpenAI 客户端；
        - 校验数据文件是否存在（不存在则抛出异常）；
        - 检查结果根目录是否存在（不存在则记录告警）；
        - 若未提供 ``result_extractor``，则设置为默认抽取逻辑。

        :param __context: Pydantic 传入的上下文对象。
        :type __context: Any

        :raises FileNotFoundError: 当 ``data_file`` 不存在时抛出。


    .. py:method:: evaluate_item(self, data: Dict) -> Dict

        评测单条样本（由子类实现）。

        子类必须覆盖该方法以实现特定基准的数据判分规则与元信息组织方式。

        :param data: 单条数据样本，通常包含问题、标准答案与模型预测等字段。
        :type data: Dict

        :returns: 单条样本的评测结果字典，通常包含 ``score`` 与 ``details`` 等字段。
        :rtype: Dict


    .. py:method:: run(self)

        运行完整的基准评测流程。

        该方法会先将源数据与生成结果进行匹配，随后使用线程池并行评测全部样本，
        最后对结果进行统计汇总并写入 ``save_path``。

        :returns: 无返回值。
        :rtype: None



.. py:class:: HLEEvaluator

    HLE 基准评测器，实现了 HLE 场景下的单条样本评测逻辑。

    .. py:method:: HLEEvaluator.evaluate_item(self, data: Dict) -> Dict

        对单条 HLE 样本进行评测。

        该实现会构造裁判提示词并使用裁判模型进行结构化解析，
        根据解析结果中的 ``correct`` 字段计算得分。

        :param data: 单条数据样本，包含 ``question``、``answer`` 与 ``prediction`` 等字段。
        :type data: Dict

        :returns: 评测结果字典，包含 ``score``、``error`` 与 ``details``。
        :rtype: Dict

        :raises ValueError: 当裁判模型返回的结构化解析结果为空时抛出异常。



.. py:class:: XBenchEvaluator

    XBench 基准评测器，实现了 XBench 场景下的单条样本评测逻辑。

    .. py:method:: XBenchEvaluator.evaluate_item(self, data: Dict) -> Dict

        对单条 XBench 样本进行评测。

        该实现会调用裁判模型生成判断文本，并通过正则提取“结论（正确/错误）”字段，
        进而计算得分。

        :param data: 单条数据样本，包含 ``question``、``answer`` 与 ``prediction`` 等字段。
        :type data: Dict

        :returns: 评测结果字典，包含 ``score``、``error`` 与 ``details``。
        :rtype: Dict

        :raises ValueError: 当无法从裁判输出中解析出结论时抛出异常。



.. py:class:: GaiaEvaluator

    GAIA 基准评测器，实现了 GAIA 场景下的单条样本评测逻辑。

    .. py:method:: GaiaEvaluator.evaluate_item(self, data: Dict) -> Dict

        对单条 GAIA 样本进行评测。

        该实现会构造裁判提示词并调用裁判模型生成判断结果，
        将输出内容规整后以 ``correct`` / 非 ``correct`` 作为判分依据。

        :param data: 单条数据样本，包含 ``question``、``answer`` 与 ``prediction`` 等字段。
        :type data: Dict

        :returns: 评测结果字典，包含 ``score``、``error`` 与 ``details``。
        :rtype: Dict


.. py:function:: execute_python_code(code: str, timeout: int = 300) -> CodeResponse
    :async:

    异步执行一段 Python 代码，并返回结构化的执行结果。

    该函数会将实际的代码执行委托到线程池执行器中运行（通过事件循环的 ``run_in_executor``），
    并统计执行耗时；若执行过程中发生异常，会记录错误日志并在结果中返回错误信息。

    :param code: 需要执行的 Python 代码字符串。
    :type code: str

    :param timeout: 执行超时时间（秒），默认 ``300``。
    :type timeout: int, optional

    :returns: 代码执行结果对象，包含标准输出、错误信息与执行耗时。
    :rtype: CodeResponse


.. py:function:: web_parse(link: str, query: str) -> Dict[str, Any]
    :async:

    统一的网页内容解析入口。

    该函数会根据传入的链接动态选择合适的解析器，并执行对应的解析逻辑，返回解析后的结构化结果。
    若解析过程中出现未处理异常，会记录错误日志并返回标准化的错误结果字典。

    :param link: 需要解析的网页 URL。
    :type link: str

    :param query: 解析上下文中的检索/问题文本，用于从页面中抽取与该上下文相关的信息。
    :type query: str

    :returns: 解析结果字典，通常包含解析内容（content）、相关 URL 列表（urls）以及评分（score）等字段；
              若发生未处理异常则返回形如 ``{"content": "System error during parsing"}`` 的错误字典。
    :rtype: Dict[str, Any]



.. py:function:: web_search(query: str, top_k: int = 10) -> Union[Dict[str, Any], List[Any]]
    :async:

    使用搜索引擎在互联网上检索信息并返回结果。

    该函数会构造请求所需的 payload 与 headers，调用 Serper 接口进行检索，并使用重试机制提升稳定性。
    成功时通常返回包含 ``organic`` 字段的结果字典；若重试处理最终返回 ``None``，则返回空列表。

    :param query: 需要提交的搜索查询语句。
    :type query: str

    :param top_k: 返回的最大结果数量，默认 ``10``。
    :type top_k: int, optional

    :returns: 成功时返回包含搜索结果的字典（通常结果位于 ``organic`` 键下）；
              若请求失败且重试处理未得到结果，则返回空列表。
    :rtype: Union[Dict[str, Any], List[Any]]



.. py:function:: download_and_read_pdf(url: str) -> str

    同步下载指定 URL 的 PDF 并提取其文本内容。

    该函数既支持普通 PDF 直链，也支持 arXiv 的摘要页链接（会转换为可直接下载的 PDF 链接）。
    下载完成后使用 PyMuPDF（fitz）解析 PDF 并提取全部文本。

    :param url: PDF 文件的 URL，或 arXiv 的摘要页 URL（例如 ``https://arxiv.org/abs/2106.07682``）。
    :type url: str

    :returns: 成功时返回提取出的 PDF 文本内容；失败时返回描述性错误字符串（例如“Failed to read the PDF”等）。
    :rtype: str