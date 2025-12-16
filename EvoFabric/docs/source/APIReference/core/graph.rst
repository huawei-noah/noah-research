.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.core.graph
==========================

.. py:module:: evofabric.core.graph

Graph Builder
~~~~~~~~~~~~~~~

.. py:class:: GraphBuilder(BaseComponent)

    基于状态驱动的图构建器，用于编排节点与边的执行逻辑，继承自 :py:class:`~evofabric.core.factory.BaseComponent`。

    :param state_schema: 本图所使用的状态模式，可为 ``dict`` 或 ``pydantic.BaseModel`` 子类。
    :type state_schema: type[StateSchema]

    .. py:method:: add_node(self, name: str, node: Union[Callable, NodeBase], action_mode: Union[NodeActionMode, str] = "any", multi_input_merge_strategy: dict[str, Callable[[List[State]], State]] = None)

        向图中添加一个节点。

        :param name: 节点名称，需在图内唯一。不可以使用 ``start``和``end``，这两个节点是图引擎预留的特殊节点。
        :type name: str

        :param node: 节点执行体，可以是普通 Python 函数或 ``NodeBase`` 子类实例。
        :type node: Union[Callable, NodeBase]

        :param action_mode: 节点触发策略。默认是 ``any``
                            ``"all"`` 表示所有前驱节点执行完毕后才触发；
                            ``"any"`` 表示任意前驱节点执行完毕即触发。
        :type action_mode: Union[NodeActionMode, str]

        :param multi_input_merge_strategy: 按通道名指定多前驱状态合并策略，
                                           key为通道名，值为 ``Callable[[List[State] -> State]]`` 的可调用对象。
        :type multi_input_merge_strategy: dict[str, Callable[[List[State]], State]]

    .. py:method:: add_edge(self, source: str, target: str, group: str = "all", state_filter: Optional[Callable[[State], State]] = None)

        在两个节点之间建立一条普通边。

        :param source: 源节点名称。
        :type source: str

        :param target: 目标节点名称。
        :type target: str

        :param group: 边所属组名，用于批量控制或可视化分组。
        :type group: str

        :param state_filter: 可选的状态过滤函数，对流经该边的状态进行拦截与改写。
        :type state_filter: Optional[Callable[[State], State]]

    .. py:method:: add_condition_edge(self, source: str, router: Callable, possible_targets: Union[List[str], Set[str]], group: str = "all")

        为指定源节点添加条件路由边。

        ``router`` 支持解析四种类型的返回结果：

        * 单个 ``str``：下一跳目标节点名；
        * ``List[str]``：多个下一跳目标节点名；
        * ``Tuple[str, Callable]``：目标节点名 + 该边专用状态过滤函数；
        * ``List[Tuple[str, Callable]]``：多组（目标节点名, 状态过滤函数）。

        :param source: 源节点名称。
        :type source: str

        :param router: 路由函数，按运行时的状态决定下一跳，输入是源节点输出的完整State状态。

            .. note::
                注意：每个节点的输出是state的增量信息。但router函数的输入是：该节点输出的增量信息和输入state合并后的完整state。

        :type router: Callable

        :param possible_targets: 所有可能被路由到的目标节点名，用于校验与优化。
        :type possible_targets: Union[List[str], Set[str]]

        :param group: 边所属组名。
        :type group: str

    .. py:method:: set_entry_point(self, entry_name: str)

        设置图的唯一入口节点。

        :param entry_name: 入口节点名称，需已通过 ``add_node`` 添加。
        :type entry_name: str

    .. py:method:: build(self, auto_conn_end: bool = True, max_turn: int = None, timeout: int = None, graph_mode: GraphMode = GraphMode.RUN, db_file_path: str = "./.storage.db", db_name: str = "evofabric")

        根据已添加的节点与边构建出可执行图实例。

        :param auto_conn_end: 若为 ``True``，自动为所有无后继的节点连接至内置 END 节点。
        :type auto_conn_end: bool

        :param max_turn: 图最大运行轮数，防止无限循环。
        :type max_turn: int

        :param timeout: 图运行总超时时间（秒）。
        :type timeout: int

        :param graph_mode: 图运行模式，默认 ``run`` 表示执行模式， ``debug`` 表示调试模式。
        :type graph_mode: GraphMode | str

        :param db_file_path: 持久化数据库文件路径，用于在debug模式下存储图状态快照。
        :type db_file_path: str

        :param db_name: 数据库名称。
        :type db_name: str

        :returns: 可执行的 :py:class:`GraphEngine` 或 :py:class:`GraphEngineDebugger` 图引擎实例


    .. py:method:: dumps(self, graph_name: str = "graph", version: str = "1.0") -> dict

        将当前图构建器中的节点与边序列化为字典配置。

        :param graph_name: 图名称，默认为 "graph"。
        :type graph_name: str

        :param version: 配置版本号，默认为 "1.0"。
        :type version: str

        :returns: 包含图名称、版本、状态模式、节点、边及是否设置入口点的字典。
        :rtype: dict

    .. py:method:: dump(self, save_path: str, graph_name: str = "graph", version: str = "1.0")

        将当前图构建器中的内容序列化并保存到指定文件。

        :param save_path: 保存路径。
        :type save_path: str

        :param graph_name: 图名称，默认为 "graph"。
        :type graph_name: str

        :param version: 配置版本号，默认为 "1.0"。
        :type version: str

    .. py:method:: load(cls, file_path: str) -> 'GraphBuilder'

        从文件中加载已构建的图构建器。

        :param file_path: 文件路径。
        :type file_path: str

        :returns: 加载后的 GraphBuilder 实例。
        :rtype: GraphBuilder

    .. py:method:: loads(cls, data: dict) -> 'GraphBuilder'

        从配置字典中加载已构建的图构建器。

        :param data: 包含图配置的字典。
        :type data: dict

        :returns: 加载后的 GraphBuilder 实例。
        :rtype: GraphBuilder


Graph Engine
~~~~~~~~~~~~~~~~~


.. py:class:: GraphEngine(BaseComponent)

    图执行引擎，负责解析并驱动整个图（节点+边）的运行，支持状态检查点、最大轮次与超时控制。

    :param nodes: 图中所有节点的规格映射，键为节点名，值为 :py:class:`GraphNodeSpec` 对象。
    :type nodes: Dict[str, GraphNodeSpec]

    :param edges: 图中所有边的规格映射，键为源节点名，值为该节点出发的 :py:class:`EdgeSpecBase` 列表。
    :type edges: Dict[str, List[EdgeSpecBase]]

    :param state_schema: 状态模式类型，用于运行时状态校验；为 ``None`` 时不做校验。
    :type state_schema: Optional[SkipValidation[type[StateSchema]]]

    :param max_turn: 允许的最大节点调用次数；超过即强制终止已运行的图，但已到达 END 节点的输出仍保留；为 ``None`` 表示无限制。
    :type max_turn: Optional[int]

    :param timeout: 单个节点执行的超时时间（秒）；为 ``None`` 表示不限制。
    :type timeout: Optional[int]

    .. py:method:: run(self, inputs: Dict)
        :async:

        异步启动图运行。

        :param inputs: 初始输入数据，将注入 ``start`` 节点的状态，key和value需要严格对应 ``StateSchema`` 。
        :type inputs: Dict

        :returns: 运行结束后的最终状态。
        :rtype: Dict

    .. py:method:: draw_graph(self, save_path: str = None, auto_open: bool = True)

        生成图的可视化 HTML 文件。

        :param save_path: 文件保存路径；为 ``None`` 时不落盘。
        :type save_path: str

        :param auto_open: 生成后是否自动用浏览器打开。
        :type auto_open: bool



.. py:class:: RunTimeTask(BaseModel)

   用于描述在图执行过程中需要被调度的节点任务及其相关状态与过滤条件。

   :param node_name: 目标节点名称。
   :type node_name: str

   :param state_ckpt: 当前任务对应的 :py:class:`StateCkpt` ；支持单个 :py:class:`StateCkpt` 或列表。
   :type state_ckpt: Union[StateCkpt, List[StateCkpt]]

   :param edge_group: 当前任务所属边的分组名称，用于在并行或条件分支中区分不同边集合。
   :type edge_group: str

   :param predecessor: 前驱节点名称；支持单个节点名或节点名列表，为 ``None`` 时表示无前驱节点。
   :type predecessor: Optional[Union[str, List[str]]]

   :param state_filter: 应用于边的 **状态过滤函数**，签名需满足 ``Callable[[Any], bool]``；为 ``None`` 时不进行额外过滤。
   :type state_filter: Optional[Callable]

   :param trace_route: 已执行的节点轨迹，按顺序记录沿途经过的节点名；默认空列表。
   :type trace_route: List[str]


Graph Engine Debugger
~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: GraphEngineDebugger(GraphEngine)

    可调试版本的 :py:class:`GraphEngine`，支持断点、步进执行、状态恢复及节点输出修改。

    :param db_file_path: 用于状态存储的数据库文件路径。默认值 ``".state_storage.db"``
    :type db_file_path: str

    :param db_name: 数据库名称。默认值 ``"graph state"``
    :type db_name: str

    .. py:method:: model_post_init(context: Any) -> None

        在模型创建后初始化内部结构。

    .. py:method:: reset() -> None

        重置所有内部状态，包括跟踪树和断点。

    .. py:method:: set_breakpoint(node_name_bp: Optional[str] = None, condition_bp: Any = None, condition: Any = None) -> None

        在指定节点上设置断点。

        :param node_name_bp: 要设置断点的节点名称。
        :type node_name_bp: Optional[str]

        :param condition_bp: 断点条件（暂不支持）。
        :type condition_bp: Any

        :param condition: 断点条件（暂不支持）。
        :type condition: Any

        :raises RuntimeError: 当 ``node_name_bp`` 为 ``None`` 时抛出。

    .. py:method:: clear_breakpoint(node_name_bp: Optional[str] = None, condition_bp: Any = None, condition: Any = None) -> None

        清除指定节点的断点。

        :param node_name_bp: 要清除断点的节点名称。
        :type node_name_bp: Optional[str]

        :param condition_bp: 断点条件（暂不支持）。
        :type condition_bp: Any

        :param condition: 断点条件（暂不支持）。
        :type condition: Any

        :raises RuntimeError: 当 ``node_name_bp`` 为 ``None`` 时抛出。

    .. py:method:: clear_all_breakpoint() -> None

        清除所有断点。

    .. py:method:: resume(running_queue: Optional[List[RunTimeTask]] = None) -> Awaitable[Any]
        :async:

        从当前断点或指定的 ``RuntimeTask`` 队列继续执行。

        :param running_queue: 待恢复任务队列；若为 ``None``，则从非断点的叶节点恢复。
        :type running_queue: Optional[List[RunTimeTask]]

        :returns: 一步执行结果。
        :rtype: Any

        :raises RuntimeError: 当图引擎已经启动时抛出。

    .. py:method:: step_over(node_uuid: Optional[str] = None) -> Awaitable[Any]
        :async:

        跳过当前断点或指定节点。

        :param node_uuid: 要跳过的节点UUID；若为 ``None``，则跳过所有当前节点。
        :type node_uuid: Optional[str]

        :returns: 一步执行结果。
        :rtype: Any

    .. py:method:: restore_step(node_uuid: Optional[str] = None) -> None

        恢复上一步执行操作。

        :param node_uuid: 要恢复的节点UUID；若为 ``None``，则恢复上一步操作。
        :type node_uuid: Optional[str]

    .. py:method:: run_one_step(running_queue: List[RunTimeTask]) -> Awaitable[Tuple[Any, List[RunTimeTask]]]
        :async:

        执行图中的一个步骤。

        :param running_queue: 待执行任务队列。
        :type running_queue: List[RunTimeTask]

        :returns: 输出结果与下一批候选任务节点。
        :rtype: Tuple[Any, List[RunTimeTask]]

    .. py:method:: debug(inputs: Dict) -> Awaitable[Any]
        :async:

        启动调试会话并注入初始输入。

        :param inputs: 初始状态字典。
        :type inputs: Dict

        :returns: 调试完成后的最终结果。
        :rtype: Any

    .. py:method:: change_output(from_node_uuid: str, to_node_uuid: str, change_key: str, change_value: Any) -> None

        在节点间修改输出值。

        :param from_node_uuid: 源节点UUID。
        :type from_node_uuid: str

        :param to_node_uuid: 目标节点UUID。
        :type to_node_uuid: str

        :param change_key: 要修改的键。
        :type change_key: str

        :param change_value: 新的键值。
        :type change_value: Any

        :raises RuntimeError: 当目标节点不是当前叶节点时抛出。

    **示例**

    .. code-block:: python

        from typing import Annotated, List, TypedDict
        from pydantic import BaseModel
        from evofabric.core.graph import GraphBuilder
        from evofabric.core.typing import *
        from evofabric.logger import get_logger

        class State(TypedDict):
            messages: Annotated[List, "append_messages"]
            node_output: Annotated[str, "overwrite"]

        def node_a(state):
            return {"messages": [AssistantMessage(content="node a output")], 'node_output': 'a'}

        def node_b(state):
            self.assertIsInstance(state, dict)
            self.assertIsInstance(state['messages'][0], StateMessage)
            return {"messages": [AssistantMessage(content="node b output")], 'node_output': 'b'}

        def node_c(state):
            self.assertIsInstance(state, dict)
            self.assertIsInstance(state['messages'][0], StateMessage)
            return {"messages": [AssistantMessage(content="node c output")], 'node_output': 'c'}

        def node_d(state):
            self.assertIsInstance(state, dict)
            self.assertIsInstance(state['messages'][0], StateMessage)
            return {"messages": [AssistantMessage(content="node d output")], 'node_output': 'd'}

        graph_builder = GraphBuilder(state_schema=State)
        graph_builder.add_node("a", node_a)
        graph_builder.add_node("b", node_b)
        graph_builder.add_node("c", node_c)
        graph_builder.add_node("d", node_d)
        graph_builder.add_edge("a", "b")
        graph_builder.add_edge("b", "c")
        graph_builder.add_edge("c", "d")
        graph_builder.set_entry_point("a")
        graph = graph_builder.build(graph_mode=GraphMode.DEBUG, db_file_path="D:/Download/.db_storage.db")

        # debug mode example
        graph.set_breakpoint(node_name_bp="b")
        await graph.debug({"messages": [UserMessage(content='hello')]})
        await graph.step_over(node_uuid=graph._trace_tree.get_leaf_node_uuid_by_node_name(node_name="b"))
        graph.restore_step()
        graph.clear_breakpoint(node_name_bp="b")
        result = await graph.resume()

Node
~~~~~~~~~~~~

.. py:class:: NodeBase(BaseComponent)

    所有节点的抽象基类，继承自 :py:class:`~evofabric.core.factory.BaseComponent`。
    子类需实现 ``__call__`` 协议，以定义节点在被调用时的具体行为。

.. py:class:: SyncNode(NodeBase)

    同步执行节点，继承自 :py:class:`NodeBase`。
    在调用时以同步方式接收当前状态并返回状态增量。

    .. py:method:: __call__(self, state: State) -> StateDelta

        同步执行节点逻辑。

        :param state: 当前工作流状态。
        :type state: State

        :returns: 相对于当前状态的变化量。
        :rtype: StateDelta

.. py:class:: AsyncNode(NodeBase)

    异步执行节点，继承自 :py:class:`NodeBase`。
    在调用时以异步方式接收当前状态并返回状态增量。

    .. py:method:: __call__(self, state: State) -> StateDelta
        :async:

        异步执行节点逻辑。

        :param state: 当前工作流状态。
        :type state: State

        :returns: 相对于当前状态的变化量。
        :rtype: StateDelta


.. py:class:: SyncStreamNode(NodeBase)

    支持同步流式输出的节点，继承自 :py:class:`NodeBase`。
    在调用过程中可通过 ``stream_writer`` 实时向外发送流式数据块。

    .. py:method:: __call__(self, state: State, stream_writer: StreamWriter) -> StateDelta

        同步流式执行节点逻辑。

        :param state: 当前工作流状态。
        :type state: State

        :param stream_writer: 用于实时输出流式数据的写入器。
        :type stream_writer: StreamWriter

        :returns: 相对于当前状态的变化量。
        :rtype: StateDelta


.. py:class:: AsyncStreamNode(NodeBase)

    支持异步流式输出的节点，继承自 :py:class:`NodeBase`。
    在调用过程中可通过 ``stream_writer`` 实时向外发送流式数据块。

    .. py:method:: __call__(self, state: State, stream_writer: StreamWriter) -> StateDelta
        :async:

        异步流式执行节点逻辑。

        :param state: 当前工作流状态。
        :type state: State

        :param stream_writer: 用于实时输出流式数据的写入器。
        :type stream_writer: StreamWriter

        :returns: 相对于当前状态的变化量。
        :rtype: StateDelta



.. py:class:: GraphNodeSpec(BaseComponent)

    该节点在图引擎中使用，会首先将图运行信息注入到上下文中，随后根据不同节点类型决定调用方式并分发不同入参。继承自 :py:class:`~evofabric.core.factory.BaseComponent`。

    :param node: 节点实例，必须继承自 :py:class:`NodeBase`。
    :type node: NodeBase

    :param node_name: 节点在图中的可读名称。
    :type node_name: str

    :param action_mode: 节点执行模式，默认 ``NodeActionMode.ALL``。
    :type action_mode: NodeActionMode

    :param stream_writer: 可选的流式写手指针，用于把节点中间结果实时推送出去。
    :type stream_writer: Optional[StreamWriter]

    :param multi_input_merge_strategy: 当节点存在多路输入时，按 key 指定自定义合并函数 ``Callable[[List[State]], State]``；若未提供则使用默认策略。
    :type multi_input_merge_strategy: Optional[Dict[str, Callable[[List[State]], State]]]

    :param node_id: 节点全局唯一标识，默认自动生成 UUID。
    :type node_id: str

    .. py:method:: is_active(self) -> bool

        ``@property`` 类型。当前节点是否处于激活状态。

        :returns: 激活返回 ``True``，否则 ``False``。
        :rtype: bool

    .. py:method:: __call__(self, state: State, **kwargs) -> StateDelta
        :async:

        通过 :py:func:`evofabric.core.graph.stream_writer_env` 注入节点执行信息，并在异步锁保护下执行节点逻辑。

        :param state: 输入状态对象。
        :type state: State

        :returns: 节点执行后产生的状态增量。
        :rtype: StateDelta


.. py:function:: callable_to_node(callable_obj: Callable[..., Any]) -> NodeBase

    将任意可调用对象（函数或实现了 ``__call__`` 方法的类实例）自动转换为对应的节点类型实例。

    函数会根据目标对象的定义特征自动判断其类型：

    * 若为 **异步函数** 且带有 ``stream_writer`` 参数 → 转换为异步流式节点。
    * 若为 **异步函数** 且不带 ``stream_writer`` 参数 → 转换为异步普通节点。
    * 若为 **同步函数** 且带有 ``stream_writer`` 参数 → 转换为同步流式节点。
    * 若为 **同步函数** 且不带 ``stream_writer`` 参数 → 转换为同步普通节点。

    :param callable_obj: 任意可调用对象，可以是函数、方法或实现了 ``__call__`` 的类实例。
    :type callable_obj: Callable[..., Any]

    :returns: 对应的节点类型实例，类型为 :py:class:`NodeBase` 的子类。
    :rtype: NodeBase


Edge
~~~~~~~~~~~~~~~

.. py:class:: EdgeSpecBase(ABC, BaseComponent)

    边规范的抽象基类，定义图中边的基本属性与接口。继承自 :py:class:`ABC` 与 :py:class:`~evofabric.core.factory.BaseComponent`。

    :param source: 源节点名称。
    :type source: str

    :param group: 边所属分组，默认为 ``DEFAULT_EDGE_GROUP``。
    :type group: str

    :param edge_type: 边的类型标识，默认为 ``"base"``。
    :type edge_type: Literal['base']

    .. py:method:: get_targets(self, state: State) -> List[Tuple[str, Optional[StateFilterLike]]]

        抽象方法。根据当前状态返回下一步的目标节点及其对应的状态过滤器。

        :param state: 当前图状态。
        :type state: State

        :returns: 目标节点与状态过滤器的列表。
        :rtype: List[Tuple[str, Optional[StateFilterLike]]]

    .. py:method:: get_possible_targets(self) -> List[str]

        抽象方法。返回该边在静态分析中可能到达的所有目标节点。

        :returns: 所有可能的目标节点名称。
        :rtype: List[str]


.. py:class:: EdgeSpec(EdgeSpecBase)

    普通边类型，仅连接一个目标节点。继承自 :py:class:`EdgeSpecBase`。

    :param source: 源节点名称。
    :type source: str

    :param group: 边所属分组，默认为 ``DEFAULT_EDGE_GROUP``。
    :type group: str

    :param edge_type: 边的类型标识，固定为 ``"edge"``。
    :type edge_type: Literal['edge']

    :param target: 目标节点名称。
    :type target: str

    :param state_filter: 状态过滤函数，可选，用于控制边的触发条件。
    :type state_filter: Optional[StateFilterLike]

    .. py:method:: get_targets(self, state: State) -> List[Tuple[str, Optional[StateFilterLike]]]

        返回目标节点及其状态过滤器，适配统一接口。

        :param state: 当前状态。
        :type state: State

        :returns: (目标节点, 状态过滤器) 的列表。
        :rtype: List[Tuple[str, Optional[StateFilterLike]]]

    .. py:method:: get_possible_targets(self) -> List[str]

        返回当前边的所有可能目标节点（对普通边为单一节点）。

        :returns: 目标节点名称列表。
        :rtype: List[str]


.. py:class:: ConditionEdgeSpec(EdgeSpecBase)

    条件边类型，支持根据状态动态决定目标节点。继承自 :py:class:`EdgeSpecBase`。

    :param source: 源节点名称。
    :type source: str

    :param group: 边所属分组，默认为 ``DEFAULT_EDGE_GROUP``。
    :type group: str

    :param edge_type: 边的类型标识，固定为 ``"conditional"``。
    :type edge_type: Literal['conditional']

    :param router: 路由函数，接受 ``State`` 作为输入，返回目标节点或节点列表，可附带状态过滤函数。
    :type router: Callable[[State], Union[str, List[str], Tuple[str, Callable], List[Tuple[str, Callable]]]]

    :param possible_targets: 所有允许的目标节点列表。
    :type possible_targets: List[str]

    .. py:method:: get_targets(self, state: State) -> List[Tuple[str, Optional[StateFilterLike]]]

        调用 ``router`` 函数，根据当前状态获取下一步目标节点，并统一格式化为 ``[(节点名, 状态过滤函数)]``。

        :param state: 当前状态。
        :type state: State

        :returns: 目标节点与状态过滤函数列表。
        :rtype: List[Tuple[str, Optional[StateFilterLike]]]

    .. py:method:: get_possible_targets(self) -> List[str]

        返回条件边在静态分析下的所有可能目标节点。

        :returns: 所有可能的目标节点名称。
        :rtype: List[str]


.. py:function:: cast_edge(v: dict) -> EdgeSpecBase

    根据输入字典中的 ``edge_type`` 字段自动判别边的类型，并反序列化为对应的 :py:class:`EdgeSpec` 或 :py:class:`ConditionEdgeSpec` 实例。

    :param v: 包含边定义的字典对象。
    :type v: dict

    :returns: 解析后的边对象。
    :rtype: EdgeSpecBase


State & Update
~~~~~~~~~~~~~~~~~

.. py:class:: StateUpdater

    用于注册和管理状态更新策略的类。

    通过该类，可以将自定义的状态合并函数注册为命名策略，并在需要时通过名称获取对应的策略函数。

    使用示例：

    .. code-block:: python

        @StateUpdater.register("overwrite")
        def overwrite(old: Any, new: Any) -> Any:
            return new

        strategy = StateUpdater.get("overwrite")
        merged = strategy(old_state, new_state)

    .. py:method:: register(cls, name: str) -> Callable[[Callable], Callable]

        类方法，用于注册一个新的状态更新策略。

        :param name: 策略名称，注册后可通过该名称获取对应的策略函数。
        :type name: str

        :returns: 一个装饰器函数，用于装饰实际的状态更新函数。
        :rtype: Callable[[Callable], Callable]

        :raises KeyError: 如果策略名称已存在，则抛出异常。

    .. py:method:: get(cls, name: str) -> Callable[[Any, Any], Any]

        类方法，根据名称获取已注册的状态更新策略函数。

        :param name: 策略名称。
        :type name: str

        :returns: 对应的状态更新函数，接受旧状态和新状态作为参数，返回合并后的状态。
        :rtype: Callable[[Any, Any], Any]

        :raises KeyError: 如果策略名称未注册，则抛出异常。

    .. py:method:: list_strategies(cls) -> List[str]

        类方法，返回所有已注册策略的名称列表。

        :returns: 包含所有策略名称的列表。
        :rtype: List[str]

    .. py:method:: registered(cls, name: str) -> bool

        类方法，判断指定名称的策略是否已注册。

        :param name: 策略名称。
        :type name: str

        :returns: 如果策略已注册返回 ``True``，否则返回 ``False``。
        :rtype: bool


.. py:class:: StateCkpt(BaseComponent)

    表示状态检查点，继承自 :py:class:`~evofabric.core.factory.BaseComponent`，用于管理状态及其变化。

    :param delta: 状态的变化量（增量），可选。
    :type delta: Optional[SkipValidation[StateDelta]]

    :param parent: 父节点，用于构建状态链，可选。
    :type parent: Optional[StateCkpt]

    :param state_schema: 状态结构的类型定义，可选。
    :type state_schema: Optional[type[StateSchema]]

    :param materialized_state_cache: 缓存的具体化状态，初始化时默认为 None，不参与初始化参数传递。
    :type materialized_state_cache: Optional[SkipValidation[State]]

    .. py:method:: materialize(self) -> Union[State, StateSchema]

        递归追溯父节点，合并状态增量，获得完整的 :py:class:`State` 状态。

        :returns: 具体化的状态对象，类型为 :py:class:`State` 或 :py:class:`StateSchema`。
        :rtype: Union[State, StateSchema]

    .. py:method:: merge(cls, checkpoints: List['StateCkpt'], strategy: Callable[[List[State]], State] = None)

        合并多个状态为一个新的状态。

        :param checkpoints: 需要合并的状态列表。
        :type checkpoints: List[StateCkpt]

        :param strategy: 合并策略函数，接收多个状态并返回合并后的状态。若未提供，则使用:py:class:`StateSchema`中声明的默认逻辑。
        :type strategy: Callable[[List[State]], State]

        :returns: 合并后的新状态。
        :rtype: StateCkpt

    .. py:method:: filter(cls, checkpoint: 'StateCkpt', strategy: Callable[[State], State])

        对给定的状态应用过滤策略，生成新的状态。

        :param checkpoint: 需要过滤的状态。
        :type checkpoint: StateCkpt

        :param strategy: 过滤策略函数，接收一个状态并返回处理后的状态。
        :type strategy: Callable[[State], State]

        :returns: 应用过滤策略后的新状态检查点。
        :rtype: StateCkpt

    .. py:method:: merge_state(state, delta, state_schema) -> Union[Dict, StateSchema]

        静态方法，用于将状态和增量合并为新的状态。

        :param state: 原始状态。
        :type state: State

        :param delta: 状态增量。
        :type delta: StateDelta

        :param state_schema: 状态结构定义，用于类型转换。
        :type state_schema: type[StateSchema]

        :returns: 合并后的新状态，类型为字典或 :py:class:`StateSchema`。
        :rtype: Union[Dict, StateSchema]


.. py:function:: generate_state_schema(variables: Optional[List[Tuple[str, Any, str]]] = None)

   声明图引擎中传递的状态信息的变量名称和类型。

   注意事项：
      * 变量名必须符合 Python 的变量命名规范。
      * 变量类型必须是以下之一：str、int、float、list、tuple、dict。
      * 存在一个名为 ``messages`` 的常量变量用于记录代理上下文；请避免使用相同名称定义其他变量。

   **示例：**

   .. code-block:: python

      generate_state_schema([
          ("msg_id", str, "overwrite"),
          ("user_id", bool, "overwrite")
      ])

   :param variables: 一个由元组组成的列表，每个元组包含变量名、变量类型和更新策略。
   :type variables: Optional[List[Tuple[str, Any, str]]]

   :returns: 根据提供的变量动态生成的 Pydantic 模型类，表示状态结构。
   :rtype: pydantic.BaseModel

   :raises ValueError: 如果变量类型无效。
   :raises ValueError: 如果变量名与保留字段冲突或重复定义。
   :raises ValueError: 如果更新策略未在 StateUpdater 中注册。


.. py:function:: _overwrite_state_update_strategy(old: Any = MISSING, new: Any = MISSING) -> Any

    更新策略 ``overwrite`` 的具体实现。

    使用新值覆盖旧值的策略函数。

    如果新值为MISSING，则返回旧值；

    :param old: 旧状态值，默认为MISSING。
    :type old: Any

    :param new: 新状态值，默认为MISSING。
    :type new: Any

    :returns: 更新后的状态值。
    :rtype: Any


.. py:function:: _append_messages(old: List[StateMessage] = MISSING, new: List[StateMessage] = MISSING) -> List[StateMessage]

    更新策略 ``append_messages`` 的具体实现。

    将新消息列表追加到旧消息列表末尾的策略函数，自动去重。

    如果旧值或新值为MISSING，则视为空列表。

    :param old: 旧消息列表，默认为MISSING。
    :type old: List[StateMessage]

    :param new: 新消息列表，默认为MISSING。
    :type new: List[StateMessage]

    :returns: 合并后的消息列表，已去重。
    :rtype: List[StateMessage]

Streaming & Context
~~~~~~~~~~~~~~~~~~~~~~


.. py:class:: StreamWriter(BaseComponent)

    流式消息写入器，用于在异步流式处理过程中发送消息块。该类继承自 :py:class:`~evofabric.core.factory.BaseComponent` 。

    .. py:method:: put(payload: Any) -> None

        将流式消息内容放入写入器，并触发已注册的消息回调函数。

        :param payload: 要发送的流式数据内容，可以是任意类型。
        :type payload: Any

        :returns: 无返回值
        :rtype: None



.. py:class:: StreamCtx(BaseModel)

    表示流式处理上下文信息的数据模型，继承自 :py:class:`BaseModel` 。

    :param node_name: 当前节点名称（可选）。
    :type node_name: Optional[str]

    :param call_id: 当前调用 ID（可选）。
    :type call_id: Optional[str]

    :param tool_name: 当前工具名称（可选）。
    :type tool_name: Optional[str]

    :param tool_call_id: 当前工具调用 ID（可选）。
    :type tool_call_id: Optional[str]

    .. py:method:: __bool__(self) -> bool

        判断当前上下文是否为空。若所有属性都为空，则返回 ``False``，否则返回 ``True``。

        :returns: 上下文是否存在有效信息
        :rtype: bool

    .. py:method:: __repr__(self) -> str

        返回当前上下文的字符串表示形式。

        :returns: 描述当前上下文的字符串
        :rtype: str



.. py:function:: set_streaming_handler(callback: Callable[[dict], Union[None, Awaitable[None]]]) -> None

    注册一个用于处理流式消息的回调函数。

    使用示例：

    .. code-block:: python

        def print_handler(payload):
            print(payload)

        set_streaming_handler(print_handler)

    :param callback: 设置处理流式消息的回调函数，支持同步、异步类型。
    :type callback: Callable[[dict], Union[None, Awaitable[None]]]

    :returns: 无返回值
    :rtype: None



.. py:function:: current_ctx() -> StreamCtx

    获取当前线程/协程中的流式处理上下文。

    :returns: 当前的流式处理上下文对象
    :rtype: StreamCtx


.. py:function:: stream_writer_env(ctx_updates: StreamCtx)

    创建一个临时的流式处理上下文环境，在该上下文中执行代码块。

    使用示例：

    .. code-block:: python

        def node(stream_writer):
            stream_writer.put("streaming msg 1")
            stream_writer.put("streaming msg 2")
            ...

        with stream_writer_env(StreamCtx(call_id=str(uuid.uuid4()), node_name=self.node_name)):
            node(get_stream_writer())

    :param ctx_updates: 用于更新当前上下文的新字段值。
    :type ctx_updates: StreamCtx

    :returns: 上下文管理器，可用于 with 语句中。
    :rtype: contextmanager


.. py:function:: get_stream_writer()

    获取全局共享的 StreamWriter 实例。

    :returns: 全局 StreamWriter 实例
    :rtype: StreamWriter