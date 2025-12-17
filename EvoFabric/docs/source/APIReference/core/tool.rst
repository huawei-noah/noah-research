.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.core.tool
=========================

.. py:module:: evofabric.core.tool


Base Tool
~~~~~~~~~~~~~~~~
.. py:class:: BaseTool(BaseComponent)

   基础工具类，用于封装可调用对象并管理其内部状态与工具模式。

   继承自 :py:class:`~evofabric.core.factory.BaseComponent`

   :param func: 要调用的函数，可以是同步或异步函数
   :type func: Callable
   :param exclude_params: 在工具模式中排除显示的参数名称列表
   :type exclude_params: List[str]
   :param inner_state: 工具的内部状态。如果工具的输入参数定义了 `inner_state: ToolInnerState`，此参数会被传递给工具，工具可读取、修改并实时更新该状态
   :type inner_state: Optional[ToolInnerState]
   :param name: 工具的名称，未提供则使用函数名称
   :type name: Optional[str]
   :param description: 工具的描述，未提供则使用函数文档字符串
   :type description: Optional[str]
   :param tool_schema: OpenAI 风格的工具模式
   :type tool_schema: Optional[dict]

   使用示例：

   .. code-block:: python

        from evofabric.core.tool._tool_manager import ToolManager

        tool_manager = ToolManager(tools=[])

        async def add(a: float, b: float):
            """add two numbers"""
            return a + b

        new_tool = BaseTool(
            name='add',
            description='add two float numbers.',
            func=add
        )

        tool_manager.add_callable_tools([new_tool])


   .. py:method:: __call__(**kwargs) -> Any

      调用该工具的方法。

      :param kwargs: 工具调用时的参数，如果工具在 `exclude_params` 中排除了 `inner_state` 或 `stream_writer`，会自动传入内部状态或流写入器。
      :return: 工具执行结果，如果工具为异步函数，则返回 awaitable 对象

   .. py:method:: model_post_init(context: Any, /)

      在 Pydantic 验证完成后初始化 BaseTool。

      :param context: Pydantic 验证的上下文信息
      :type context: Any

   .. py:method:: from_callable(func: Callable, name: str = None, description: str = None, tool_schema: Optional[dict] = None, inner_state: Optional[ToolInnerState] = None, exclude_params: List[str] = None)

      将可调用对象封装为 BaseTool 实例。

      :param func: 要调用的函数，可以是同步或异步函数
      :type func: Callable
      :param name: 工具的名称，未提供则使用函数名称
      :type name: str
      :param description: 工具的描述，未提供则使用函数文档字符串
      :type description: str
      :param tool_schema: OpenAI 风格的工具模式
      :type tool_schema: Optional[dict]
      :param inner_state: 工具的内部状态。如果工具的输入参数定义了 `inner_state: ToolInnerState`，此参数会被传递给工具
      :type inner_state: Optional[ToolInnerState]
      :param exclude_params: 在工具模式中排除显示的参数名称列表
      :type exclude_params: List[str]
      :return: 封装后的 BaseTool 实例
      :rtype: BaseTool

   .. py:method:: get_tool_schema()

      获取 BaseTool 实例的 OpenAI 风格工具 Schema 描述。

      :return: 工具 Schema 描述
      :rtype: dict

   .. py:method:: dump_state()

      获取 BaseTool 实例的当前内部状态。

      :return: 工具的内部状态
      :rtype: ToolInnerState

   .. py:method:: load_state(input_state: ToolInnerState)

      加载指定状态以更新 BaseTool 实例的内部状态。

      :param input_state: 要加载的内部状态
      :type input_state: ToolInnerState



Tool Manager
~~~~~~~~~~~~~~~


.. py:class:: ToolManagerBase(ABC, BaseComponent)

    工具管理系统的基础抽象类，定义工具管理相关接口。
    继承自 :py:class:`abc.ABC` 和 :py:class:`~evofabric.core.factory.BaseComponent`。

    .. py:method:: list_tools(**kwargs)

        以 OpenAI 风格返回工具模式列表。

        :raises: NotImplementedError  # 子类必须实现此方法

    .. py:method:: call_tools(tasks: List[ToolCall])

        根据提供的工具调用任务列表执行工具调用。

        :param tasks: 工具调用任务列表
        :type tasks: List[ToolCall]
        :raises: NotImplementedError  # 子类必须实现此方法

    .. py:method:: start()

        启动工具管理系统所依赖的资源。

        :raises: NotImplementedError  # 子类必须实现此方法

    .. py:method:: stop()

        停止工具管理系统所依赖的资源。

        :raises: NotImplementedError  # 子类必须实现此方法

    .. py:method:: reset()

        重置工具管理系统所依赖的资源。

        :raises: NotImplementedError  # 子类必须实现此方法

    .. py:method:: save_state(save_path: str)

        将工具管理系统中工具的内部状态保存到指定路径。

        :param save_path: 保存工具内部状态的路径
        :type save_path: str
        :raises: NotImplementedError  # 子类必须实现此方法

    .. py:method:: load_state(load_path: str)

        从指定路径加载工具内部状态到工具管理系统中对应的工具。

        :param load_path: 包含工具内部状态的文件路径
        :type load_path: str
        :raises: NotImplementedError  # 子类必须实现此方法


.. py:class:: ToolManager(ToolManagerBase)

    常规工具管理系统的实现类。

    继承自 :py:class:`ToolManagerBase`，实现其所有抽象方法，提供工具管理、调用、导入、删除、更新及状态管理功能。

    :param tools: 初始工具列表，可包含 Python 函数、BaseTool 实例或带内部状态的函数元组
    :type tools: List[Union[Callable, Tuple[Callable, ToolInnerState], BaseTool]]
    :param timeout: 工具执行的超时时间（秒）
    :type timeout: int, 可选
    :param tool_controller: 工具控制器，用来控制工具的激活状态，未激活的工具无法交互。
    :type tool_controller: :py:class:`ToolController` ，可选

    使用示例：

    .. code-block:: python

        from evofabric.core.tool._tool_manager import ToolManager

        async def MULTIPLY(a: int, b: int):
            """MULTIPLY two numbers."""
            return a * b

        tool_manager = ToolManager(
            tools=[MULTIPLY],
        )

    .. py:method:: model_post_init(context: Any, /)

        在 Pydantic 验证后初始化 :py:class:`ToolManager`，并将工具添加到系统中。

        :param context: Pydantic 验证上下文
        :type context: Any

    .. py:method:: list_tools()

        列出 :py:class:`ToolManager` 中所有工具的工具模式。

        :return: 工具模式列表
        :rtype: List[dict]

    .. py:method:: call_tools(tasks: List[ToolCall])

        执行工具调用以完成指定任务。

        :param tasks: 工具调用任务列表，包括工具名称及参数
        :type tasks: List[ToolCall]
        :return: 工具调用结果列表，每个元素包含调用 ID、结果及成功标志等信息
        :rtype: List[ToolCallResult]

    .. py:method:: add_callable_tools(tools: List[Union[Callable, BaseTool, Tuple[Callable, ToolInnerState]]])

        向系统中添加 Python 函数工具或 :py:class:`BaseTool` 实例。

        :param tools: 待添加的工具列表
        :type tools: List[Union[Callable, BaseTool, Tuple[Callable, ToolInnerState]]]

    .. py:method:: add_python_file_tools(file_paths: List[str], include_pattern_list: List[List[str]] = None, exclude_pattern_list: List[List[str]] = None)

        从 Python 文件批量导入符合条件的函数工具到系统中。

        :param file_paths: 包含 Python 函数工具的文件路径列表
        :type file_paths: List[str]
        :param include_pattern_list: 每个文件中函数名必须包含的字符串模式列表（长度应与 file_paths 一致）
        :type include_pattern_list: List[List[str]]
        :param exclude_pattern_list: 每个文件中函数名不能包含的字符串模式列表（长度应与 file_paths 一致）
        :type exclude_pattern_list: List[List[str]]

    .. py:method:: delete_tools(tool_names: List[str])

        从系统中删除指定工具。

        :param tool_names: 待删除的工具名称列表
        :type tool_names: List[str]

    .. py:method:: update_tools(tools: List[BaseTool])

        更新系统中已有的工具实例。

        :param tools: 待更新的 BaseTool 实例列表
        :type tools: List[BaseTool]

    .. py:method:: find_tools(tool_names: List[str])

        查找系统中的工具。

        :param tool_names: 待查找的工具名称列表
        :type tool_names: List[str]
        :return: 找到的工具的工具模式列表
        :rtype: List[dict]

    .. py:method:: save_state(save_path: str)

        保存系统中所有工具的内部状态到指定路径。

        :param save_path: 保存路径
        :type save_path: str

    .. py:method:: load_state(load_path: str)

        从指定路径加载工具内部状态到系统中对应的工具。

        :param load_path: 包含工具内部状态的文件路径
        :type load_path: str

    .. py:method:: dump_state(tool_name: str = None)

        获取系统中工具的内部状态。

        :param tool_name: 指定工具名称，默认为 None，表示获取所有工具状态
        :type tool_name: str
        :return: 如果指定了 tool_name，则返回该工具的状态；否则返回所有工具的状态
        :rtype: dict


.. py:class:: McpToolManager(ToolManagerBase)

   MCP (Model Context Protocol) 工具管理器，用于管理多个 MCP 服务器及其工具。

   注意：

    - 实际传递给 LLM 的工具名称格式为 "server_name" + "_" + "tool_name"
    - 此命名约定支持在不同服务器中使用同名的工具
    - 例如，如果服务器 "math_server" 有工具 "calculate"，其完整工具名将为 "math_server_calculate"

   :param server_links: MCP 服务器连接参数字典，键为服务器名称，值为连接参数
   :type server_links: Dict[str, McpServerLink]

   :param timeout: 工具执行超时时间（秒），默认为300
   :type timeout: int

   :param tool_controller: 用于管理工作具激活状态的工具控制器，可选
   :type tool_controller: Optional[ToolController]

   :param persistent_link: 是否重用 MCP 连接标志。True 表示每次交互使用相同连接；False 表示每次创建新连接
   :type persistent_link: bool

   .. py:method:: add_mcp_servers(mcp_server_links: Dict[str, McpServerLink])
      :async:

      向管理器添加新的 MCP 服务器。

      :param mcp_server_links: 服务器名称到 McpServerLink 配置的字典
      :type mcp_server_links: Dict[str, McpServerLink]


   .. py:method:: delete_mcp_servers(mcp_server_names: List[str])
      :async:

      从管理器中移除 MCP 服务器并清理连接。

      :param mcp_server_names: 要移除的服务器名称列表
      :type mcp_server_names: List[str]


   .. py:method:: list_tools(server_name: str = None) -> list
      :async:

      列出可用工具（可选按服务器名筛选），通过工具控制器过滤状态（如果已启用）。

      :param server_name: 可选的服务器名称用于筛选结果
      :type server_name: str, optional
      :return: 工具模式列表（如果启用了工具控制器则会过滤状态）
      :rtype: list


   .. py:method:: call_tools(tasks: List[ToolCall]) -> List[ToolCallResult]
      :async:

      执行提供的工具调用任务列表。

      :param tasks: 要执行的 ToolCall 对象列表
      :type tasks: List[ToolCall]
      :return: 包含执行结果的 ToolCallResult 对象列表
      :rtype: List[ToolCallResult]


   .. py:method:: list_prompts(server_name: str = None) -> list
      :async:

      列出可用提示（可选按服务器名筛选）。

      :param server_name: 可选的服务器名称用于筛选结果
      :type server_name: str, optional
      :return: 可用提示的列表
      :rtype: list


   .. py:method:: list_resources(server_name: str = None) -> list
      :async:

      列出可用资源（可选按服务器名筛选）。

      :param server_name: 可选的服务器名称用于筛选结果
      :type server_name: str, optional
      :return: 可用资源的列表
      :rtype: list


   .. py:method:: list_resource_templates(server_name: str = None) -> list
      :async:

      列出可用资源模板（可选按服务器名筛选）。

      :param server_name: 可选的服务器名称用于筛选结果
      :type server_name: str, optional
      :return: 可用资源模板的列表
      :rtype: list


   .. py:method:: read_resource(resources: List[ResourceRequest]) -> list
      :async:

      从 MCP 服务器读取指定的资源。

      :param resources: ResourceRequest 对象列表
      :type resources: List[ResourceRequest]
      :return: 资源响应列表
      :rtype: list


   .. py:method:: get_prompt(prompt_requests: List[PromptRequest]) -> list
      :async:

      从 MCP 服务器获取指定的提示。

      :param prompt_requests: PromptRequest 对象列表
      :type prompt_requests: List[PromptRequest]
      :return: 提示响应列表
      :rtype: list


   .. py:method:: set_tool_controller(controller: 'ToolController')
      :async:

      设置管理工作具激活规则的工具控制器。

      :param controller: 包含激活/停用规则的工具控制器实例
      :type controller: 'ToolController'


   .. py:method:: get_tool_controller() -> Optional['ToolController']
      :async:

      获取当前工具控制器实例。

      :return: 当前的 ToolController 实例，如果未配置则返回 None
      :rtype: Optional['ToolController']


   .. py:method:: connect(server_name: str = None)
      :async:

      建立 MCP 服务器的连接。

      :param server_name: 可选的服务器名称。如果为 None，则连接所有服务器
      :type server_name: str, optional
      :raises KeyError: 如果指定的服务器名称不存在


   .. py:method:: disconnect(server_name: str = None)
      :async:

      断开 MCP 服务器的连接。

      :param server_name: 可选的服务器名称。如果为 None，则断开所有服务器
      :type server_name: str, optional
      :raises KeyError: 如果指定的服务器名称不存在


   .. py:method:: get_mcp_status() -> Dict[str, bool]
      :async:

      获取所有管理的 MCP 服务器的连接状态。

      :return: 映射服务器名称到连接状态（True/False）的字典
      :rtype: Dict[str, bool]


   .. py:method::  __aenter__() -> McpToolManager
      :async:

      进入 McpToolManager 的运行时上下文。
      建立所有配置的 MCP 服务器的连接。

      :return: 供上下文内使用的 McpToolManager 实例
      :rtype: McpToolManager


   .. py:method:: __aexit__(exc_type, exc_val, exc_tb)
      :async:

      退出 McpToolManager 的运行时上下文。
      自动断开所有 MCP 服务器的连接。

      :param exc_type: 发生异常时的异常类型（否则为 None）
      :param exc_val: 发生异常时的异常实例（否则为 None）
      :param exc_tb: 发生异常时的回溯（否则为 None）


.. py:class:: McpSessionController(BaseComponent)

   管理MCP服务器连接生命周期和会话操作的控制器类。
   负责建立连接、管理会话状态以及使用异步上下文管理器实现优雅断开连接。

   :param server_link: MCP服务器通信链路对象
   :type server_link: McpServerLink
   :param server_name: 连接的MCP服务器的名称标识符
   :type server_name: str
   :param persistent_link: 设为True时，退出上下文（aexit）不会断开MCP服务器连接。默认为False
   :type persistent_link: bool

   .. py:property:: session

      返回当前活动的客户端会话（只读属性）。

   .. py:property:: is_connect

      返回连接状态（True/False），用于监控目的。

   .. py:method:: connect() -> None
      :async:

      启动并等待MCP服务器连接建立。
      创建后台任务维护会话循环，阻塞直至连接完全建立。
      安全地支持多次调用，重复调用会提前返回。

   .. py:method:: disconnect() -> None
      :async:

      终止MCP连接。
      取消会话任务，等待清理完成，并重置所有连接状态。

   .. py:method:: _run_session_loop(ready_event: asyncio.Event) -> None
      :async:

      内部方法：运行会话维护的主循环。
      建立MCP会话后等待连接就绪事件，进入持久等待状态直至被取消。

      :param ready_event: 用于通知连接已就绪的Event对象

   .. py:method:: __aenter__() -> McpSessionController
      :async:

      异步上下文管理器入口。
      自动调用connect()方法建立连接，返回自身实例。

   .. py:method:: __aexit__(exc_type: Any, exc_val: Any, exc_tb: Any) -> None
      :async:

      异步上下文管理器出口。
      仅当persistent_link为False时调用disconnect()方法断开连接。

Code Sandbox
~~~~~~~~~~~~~~~~


.. py:class:: CodeSandbox(BaseComponent)

    安全的 Python 代码沙箱，基于 Docker 实现，用于在隔离环境中执行 Python 代码。

    继承自 :py:class:`~evofabric.core.factory.BaseComponent`。

    :param config: 沙箱配置，包括镜像、容器名称、工作目录等信息
    :type config: :py:class:`CodeExecDockerConfig`

    使用示例：

    .. code-block:: python

        from evofabric.core.typing._tool import CodeExecDockerConfig
        from evofabric.core.tool._code_sandbox import CodeSandbox

        config = CodeExecDockerConfig(
            name="python_code_sandbox"
        )
        sandbox = CodeSandbox(config=config)
        sandbox.start()

        code1 = """
        def fib(n):
            if n <= 2:
                return 1
            else:
                return fib(n-1) + fib(n-2)

        res = fib(10)
        print("10th fib number: ", res)
        """

        result1 = sandbox.run_python(code1)
        print("python result: ", result1)
        sandbox.stop()

    .. py:method:: start()

        根据配置创建并启动 Docker 容器。

    .. py:method:: run_python(code: str)

        在沙箱容器中执行 Python 代码。

        :param code: 待执行的 Python 代码
        :type code: str
        :return: 执行结果
        :rtype: :py:class:`ExecResult`

    .. py:method:: run_cmd(cmd: str)

        在沙箱容器中执行 shell 命令。

        :param cmd: 待执行的 shell 命令
        :type cmd: str
        :return: 执行结果
        :rtype: :py:class:`ExecResult`

    .. py:method:: stop()

        停止 Docker 容器。
        如果配置中的 ``auto_remove`` 为 ``True``，容器停止后将被自动移除。


Tool Controller
~~~~~~~~~~~~~~~~~

.. py:class:: ToolController(BaseComponent)

    工具控制器，用于管理工具的激活和禁用状态。
    继承自 :py:class`~evofabric.core.factory.BaseComponent`

    :param default_mode: 默认工具行为，当没有规则匹配时使用
    :type default_mode: Literal['activate', 'deactivate']
    :param rules: 用于控制工具激活/禁用的规则列表，按顺序应用规则，第一个匹配规则决定工具状态
    :type rules: list[Union[ToolControlPattern, dict]]

    类属性默认值说明：

    *default_mode* 的默认值为 "activate"。可选值：

    - "activate": 自动激活没有匹配规则的工具（默认值）
    - "deactivate": 自动禁用没有匹配规则的工具

    *rules* 的默认值为空字典。每个规则可以是：

    1. 带有以下属性的 `ToolControlPattern` 实例：
      - `mode`: 'activate' 或 'deactivate'
      - `pattern`: glob 通配符字符串
    2. 带有以下键的字典：
      - 'mode': 'activate' 或 'deactivate'
      - 'pattern': glob 通配符字符串

    .. note::

        使用 McpToolManager 时的注意事项：

        - 实际工具名称格式为 [server_name]_[tool_name]
        - 通配符模式必须以 server_name 作为前缀
        - 示例：为匹配来自 "math" 服务器的所有工具，使用模式 "math_*" （匹配 "math_calculator"、"math_grapher" 等）

        模式示例：

        - math_*  → 匹配来自 "math" 服务器的所有工具 （例如 "math_calculator"、"math_grapher"）
        - text_*  → 匹配来自 "text" 服务器的所有工具
        - math_calculator → 仅匹配此特定工具
        - *_calculator   → 匹配任何以 "*_calculator" 结尾的工具（来自任何服务器）

    .. py:method:: check_tool_status(tool_name: str) -> bool

       根据应用规则检查工具是否处于激活状态。

       :param tool_name: 要检查的工具名称
       :type tool_name: str
       :return: 如果工具激活返回 True，否则返回 False
       :rtype: bool

    .. py:method:: filter_tool_list(tool_list: List[Dict]) -> List[Dict]

       过滤工具列表，仅包含激活的工具。

       :param tool_list: 工具字典列表
       :type tool_list: List[Dict]
       :return: 激活工具的列表
       :rtype: List[Dict]

    .. py:method:: activate_tool(tool_name: str)

       激活特定工具，赋予最高优先级。
       移除此工具名称的任何现有规则。

       :param tool_name: 要激活的工具的确切名称
       :type tool_name: str

    .. py:method:: deactivate_tool(tool_name: str)

       禁用特定工具，赋予最高优先级。
       移除此工具名称的任何现有规则。

       :param tool_name: 要禁用的工具的确切名称
       :type tool_name: str


Utils
~~~~~~~~~~~~~~~~~~~

.. py:function:: parse_callable_schema(function, name=None, description=None, include_long_description=True, include_var_positional=True, include_var_keyword=True, exclude_params=None)

   将Python可调用对象转换为LLM工具使用的JSON模式。此函数解析函数签名、文档字符串和参数默认值，创建与LLM工具调用格式兼容的完整接口描述。

   支持的函数类型包括：普通函数、类方法、@classmethod、@staticmethod、函数/类方法的partial对象、lambda函数。

   :param function: 要转换为模式的可调用函数
   :type function: Callable
   :param name: 自定义函数名，未提供则使用函数的`__name__`属性
   :type name: Optional[str]
   :param description: 自定义描述，未提供则使用文档字符串摘要
   :type description: Optional[str]
   :param include_long_description: 是否包含文档字符串中的长描述
   :type include_long_description: bool
   :param include_var_positional: 是否包含位置可变参数(*args)
   :type include_var_positional: bool
   :param include_var_keyword: 是否包含关键字可变参数(**kwargs)
   :type include_var_keyword: bool
   :param exclude_params: 需要排除的参数名称列表
   :type exclude_params: Optional[List[str]]
   :return: 包含OpenAI兼容函数模式的元组
   :rtype: tuple
   :return1: 包含完整函数模式的字典
   :rtype1: dict
   :return2: 实际被排除的参数名称列表
   :rtype2: List[str]
