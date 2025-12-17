.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.app.sop2workflow
==========================

.. py:module:: evofabric.app.sop2workflow



.. py:function:: extract_text_between(text: str, start: str, end: str) -> Optional[str]

	从给定字符串中提取位于指定起始和结束标记之间的子字符串。

	:param text: 原始文本。
	:type text: str

	:param start: 起始标记字符串。
	:type start: str

	:param end: 结束标记字符串。
	:type end: str

	:returns: 如果成功找到起始和结束标记，则返回两者之间的子字符串；否则返回 ``None``。
	:rtype: Optional[str]



.. py:function:: generate_condition_router_function_call(source: str, possible_targets: list, fallback_target: str = "end", exit_function_name: str = None)

	生成一个条件路由函数，可以注入到决策节点的系统提示中，用于根据模型输出决定下一步执行的节点。

	:param source: 当前决策节点的名称。
	:type source: str

	:param possible_targets: 当前节点可能跳转到的目标节点名称列表。
	:type possible_targets: list[str]

	:param fallback_target: 当模型返回未知或空选择时，默认跳转到的节点名称。默认为 ``end``。
	:type fallback_target: str, optional

	:param exit_function_name: 可选参数。如果提供，模型可以通过调用该名称的工具函数立即跳转到 ``end`` 节点（紧急退出）。
	:type exit_function_name: str | None, optional

	:returns: 一个路由函数，接收状态对象 ``state`` 作为输入，返回目标节点名称。
	:rtype: callable




.. py:function:: user_feedback_router(state)

	用户反馈路由函数。查找最后一条助手消息，并将流程跳转回该消息对应的节点。

	:param state: 当前状态对象，需包含 ``messages`` 属性。
	:type state: object

	:returns: 目标节点名称，若未找到则返回 ``end``。
	:rtype: str




.. py:class:: GraphDespNode(BaseModel)

	节点的描述信息，继承自 :py:class:`BaseModel` 。

	:param name: 节点名称。
	:type name: str

	:param tools: 该节点使用的工具名称列表。
	:type tools: List[str]

	:param memories: 该节点使用的记忆名称列表。
	:type memories: List[str]

	:param instruction: 该节点的指令内容。
	:type instruction: str

	:param sop: 构建该节点时使用的标准操作流程（SOP）片段。
	:type sop: Optional[str], optional



.. py:class:: GraphDespEdge(BaseModel)

	图中边的描述信息，继承自 :py:class:`BaseModel` 。

	:param source: 边的起始节点名称。
	:type source: str

	:param possible_targets: 边可能指向的目标节点名称列表。
	:type possible_targets: List[str]

	:param type: 边的类型，默认为 ``condition``。
	:type type: Literal["condition"], optional




.. py:class:: GraphDescription(BaseModel)

	整个图结构的描述信息，继承自 :py:class:`BaseModel` 。

	:param nodes: 图中所有节点的列表。
	:type nodes: List[GraphDespNode]

	:param edges: 图中所有边的列表。
	:type edges: List[GraphDespEdge]

	:param entry_point: 图的入口节点名称。
	:type entry  _point: str

	:param global_instruction: 所有节点共享的全局指令。
	:type global_instruction: str



.. py:class:: WorkflowGeneratorBase(BaseComponent)

	工作流生成器的基类，继承自 :py:class:`BaseComponent` 。

	:param sop: 用于生成工作流的标准操作流程（SOP）。
	:type sop: str

	.. py:method:: generate(self) -> GraphEngine

		使用标准操作流程（SOP）生成一个可运行的图引擎。

		:returns: 生成的图引擎实例。
		:rtype: GraphEngine

	.. py:method:: load_yaml(file_path) -> Any

		静态方法。加载指定路径的 YAML 文件内容。

		:param file_path: YAML 文件路径。
		:type file_path: str

		:returns: 加载的 YAML 数据，如果文件不存在则返回 ``None``。
		:rtype: Any

	.. py:method:: dump_yaml(data, file_path) -> None

		静态方法。将数据写入指定路径的 YAML 文件。

		:param data: 要写入的数据。
		:type data: Any

		:param file_path: YAML 文件路径。
		:type file_path: str

		:returns: 无返回值。
		:rtype: None




.. py:class:: SopBreakdownNodeDesp(BaseModel)

	描述一个分解后的SOP工作流节点。

	:param name: 节点名称。
	:type name: str

	:param type: 节点类型，只能是 ``sop`` 或 ``connect``。其中 ``sop`` 类型的节点严格执行SOP片段；``connect`` 类型是用于连接各节点的路由节点。
	:type type: Literal["sop", "connect"]

	:param duty: 该节点的职责说明。
	:type duty: str

	:param instruction: 该节点的执行指令。
	:type instruction: str

	:param next_node_routing_rule: 该节点的路由规则，键为目标节点名，值为触发条件。
	:type next_node_routing_rule: Dict[str, str]

	.. py:method:: to_full_instruction(self, global_instruction) -> str

		根据全局指令和节点自身信息，生成完整的节点执行指令。

		:param global_instruction: 全局执行策略或说明。
		:type global_instruction: str

		:returns: 完整的节点执行指令文本。
		:rtype: str



.. py:class:: SopBreakdownGraphDesp(BaseModel)

	描述一个分解后的SOP工作流图结构。

	:param nodes: 节点列表。
	:type nodes: List[SopBreakdownNodeDesp]

	:param global_instruction: 所有节点共享的全局指令。
	:type global_instruction: str

	:param entry_point: 工作流的入口节点名称。
	:type entry_point: str



.. py:class:: WorkflowGenerator(WorkflowGeneratorBase, BaseComponent)

	基于SOP（标准操作流程）自动生成工作流图（Graph）的组件，继承自 :py:class:`WorkflowGeneratorBase` 和 :py:class:`BaseComponent`。

	:param graph_generation_client: 用于生成图结构的大模型客户端。
	:type graph_generation_client:     ChatClientBase

	:param graph_node_complete_client: 用于完善节点信息的大模型客户端。
	:type graph_node_complete_client:     ChatClientBase

	:param graph_run_client: 用于运行生成图的大模型客户端。
	:type graph_run_client:     ChatClientBase

	:param retry: 当大模型响应解析失败时的重试次数，默认为5次。
	:type retry: int

	:param output_dir: 用于缓存生成的图描述文件的目录。如果为 ``None``，每次都会重新生成；如果提供目录路径，将尝试加载已有文件以跳过生成步骤。
	:type output_dir: Optional[str]

	:param tools: 工作流中节点可使用的工具管理器列表。
	:type tools: List[ToolManagerBase]

	:param memories: 工作流中节点可访问的记忆模块字典，键为自定义名称，值为（描述，实例）元组。
	:type memories: Dict[str, Tuple[str, MemBase]]

	:param state_schema: 除默认的 ``messages`` 字段外，额外添加到状态中的字段定义列表。格式为：``[(字段名, 字段类型, 字段描述), ...]``。
	:type state_schema: Optional[List[Tuple[str, Any, str]]]

	:param addition_global_instruction: 添加到每个节点系统提示中的全局指令片段。
	:type addition_global_instruction: str

	:param user_node: 当节点需要与用户交互时使用的保留节点。
	:type user_node: Optional[AsyncNode]

	:param fallback_node: 当节点跳转目标无效或缺失时的默认跳转节点。
	:type fallback_node: Optional[str]

	:param auto_self_loop: 是否默认允许节点跳转到自身。
	:type auto_self_loop: bool

	:param sop_disassembly_prompt: 用于将完整SOP拆解为工作流节点的提示模板。
	:type sop_disassembly_prompt: str

	:param node_completion_prompt: 用于完善每个节点信息（如工具、记忆、指令等）的提示模板。
	:type node_completion_prompt: str

	:param tool_list_mode: 控制节点如何获取工具列表，可选 ``all``（全部）或 ``select``（按LLM选择）。
	:type tool_list_mode: Literal["all", "select"]

	:param memory_list_mode: 控制节点如何获取记忆模块列表，可选 ``all``（全部）或 ``select``（按LLM选择）。
	:type memory_list_mode: Literal["all", "select"]

	:param skeleton_file_name: 用于保存图描述的文件名，默认为 ``_skeleton.yaml``。
	:type skeleton_file_name: str

	:param reserved_nodes: 禁止由LLM生成的保留节点名称列表，默认为 ``["start", "end", "user"]``。
	:type reserved_nodes: List[str]

	:param exit_function_name: 当调用该工具时立即跳转到 ``end`` 节点的函数名称。
	:type exit_function_name: Optional[str]

	:param build_kwargs: 传递给 :py:meth:`evofabric.core.graph.GraphBuilder.build` 的额外参数。
	:type build_kwargs: Dict[str, Any]

	.. py:method:: generate(self) -> GraphEngine

		异步生成完整的工作流图引擎对象。

		:returns: 可运行的 :py:class:`~evofabric.core.graph.GraphEngine` 实例。
		:rtype: GraphEngine
