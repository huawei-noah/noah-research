.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.core.factory
=========================

.. py:module:: evofabric.core.factory


Factory
~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: ComponentFactory

    用于注册和创建组件实例的工厂类。

    .. py:method:: create(cls, name: str, /, **kwargs) -> BaseComponent

        根据给定的名称和参数创建一个类实例。

        :param name: 组件类的注册名称。
        :type name: str

        :param kwargs: 传递给组件类构造函数的关键字参数。
        :type kwargs: Any

        :returns: 创建的组件实例。
        :rtype: BaseComponent

        :raises ValueError: 如果未找到指定名称的组件。

    .. py:method:: register(cls, name: str, component_cls: Type[BaseComponent]) -> None

        将一个类注册到工厂中。

        :param name: 注册时使用的名称。
        :type name: str

        :param component_cls: 要注册的组件类。
        :type component_cls: Type[BaseComponent]

        :raises ValueError: 如果该名称已被注册。

    .. py:method:: is_registered(cls, name: str) -> bool

        检查指定名称是否已注册。

        :param name: 要检查的组件名称。
        :type name: str

        :returns: 如果名称已注册返回 ``True``，否则返回 ``False``。
        :rtype: bool



.. py:class:: BaseComponent(BaseModel)

    所有组件类的基类，继承自 :py:class:`pydantic.BaseModel`，提供文档生成、懒加载实例等功能。

    继承了该基类的子类，会自动注册到 :py:class:`ComponentFactory` 中，允许后续通过工厂创建该类的实例。




.. py:class:: FactoryTypeAdapter

    一个用于 Pydantic V2 的类型适配器，支持将字典序列化/反序列化为 :py:class:`BaseComponent` 及其子类实例。该类通过 `__class_name__` 字段识别具体组件类型，并借助 :py:class:`ComponentFactory` 进行实例创建。

    .. py:method:: __get_pydantic_core_schema__(cls, source_type, handler) -> core_schema.CoreSchema

        生成 Pydantic 核心 schema，用于支持从字典到 :py:class:`BaseComponent` 实例的验证和序列化。

        :param source_type: 被装饰的原始类型。
        :type source_type: Type

        :param handler: Pydantic 提供的 schema 处理函数。
        :type handler: GetCoreSchemaHandler

        :returns: 返回一个支持字典与 :py:class:`BaseComponent` 之间相互转换的 schema。
        :rtype: core_schema.CoreSchema


Function Serializer
~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: FunctionSerializerProto()

    定义了函数序列化与反序列化方法的协议类，用于在模块中存储和重新加载 DSL 文件时处理 Python 函数句柄。

    .. py:method:: serialize(self, obj: Any) -> str

        将给定对象序列化为字符串。

        :param obj: 需要序列化的对象。
        :type obj: Any

        :returns: 序列化后的字符串。
        :rtype: str

    .. py:method:: deserialize(self, s: str) -> Any

        从字符串中反序列化出原始对象。

        :param s: 序列化后的字符串。
        :type s: str

        :returns: 反序列化后的对象。
        :rtype: Any




.. py:class:: FunctionSerializerCloudPickle()

    使用 ``cloudpickle`` 实现的函数序列化器，支持对复杂 Python 函数对象（包括闭包、lambda 等）进行序列化与反序列化。

    .. py:method:: serialize(self, function: Callable) -> str

        将一个可调用对象序列化为 Base64 编码的字符串。

        :param function: 需要序列化的函数或可调用对象。
        :type function: Callable

        :returns: 序列化后的 Base64 字符串。
        :rtype: str

    .. py:method:: deserialize(self, string: str, required_modules: Optional[List[str]] = None) -> Callable

        从 Base64 编码的字符串中反序列化出函数对象。

        :param string: 已序列化的函数字符串。
        :type string: str

        :param required_modules: 反序列化前需要导入的模块列表，用于确保依赖类型正确加载。
        :type required_modules: Optional[List[str]]

        :returns: 反序列化后的函数对象。
        :rtype: Callable




.. py:function:: register_deserialize_modules(modules: List[str]) -> None

    注册反序列化过程中可能需要用到的模块列表。默认包含了：

    .. code-block:: python

        DESERIALIZER_MODULES = [
            "evofabric.logger",
            "evofabric.core.agent",
            "evofabric.core.clients",
            "evofabric.core.factory",
            "evofabric.core.graph",
            "evofabric.core.mem",
            "evofabric.core.multi_agent",
            "evofabric.core.tool",
            "evofabric.core.trace",
            "evofabric.core.typing",
            "evofabric.core.vectorstore"
        ]

    :param modules: 模块名称列表，每个名称应为合法的 Python 模块路径字符串。
    :type modules: List[str]

    :returns: 无返回值。
    :rtype: None

    使用示例：

    .. code-block:: python

        register_deserialize_modules([
            "evofabric.logger",
            "evofabric.core.agent",
            "evofabric.core.clients",
        ])



.. py:function:: set_func_serializer(impl: Optional[FunctionSerializerProto]) -> None

    设置全局使用的函数序列化器实现。

    :param impl: 一个实现了 :py:class:`FunctionSerializerProto` 协议的对象；若为 ``None``，则默认使用 :py:class:`FunctionSerializerCloudPickle`。
    :type impl: Optional[FunctionSerializerProto]

    :returns: 无返回值。
    :rtype: None



.. py:function:: get_func_serializer() -> FunctionSerializerProto

    获取当前全局使用的函数序列化器实例。

    :returns: 当前的函数序列化器实现。
    :rtype: FunctionSerializerProto


State Schema Serializer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: dump_schema_annotated_info(schema: Type[Union[BaseModel, Dict]]) -> Dict

    将 Pydantic 的 BaseModel 类型或 TypedDict 类型转换为包含注解信息的字典结构，便于序列化和传输。

    :param schema: 要转换的 BaseModel 或 TypedDict 类型。
    :type schema: Type[Union[BaseModel, Dict]]

    :returns: 包含类型名称、类型类别（BaseModel 或 TypedDict）及字段详细信息的字典。
    :rtype: Dict


.. py:function:: load_schema_annotated_info(schema_info: Dict) -> Type[Union[BaseModel, Dict]]

    根据包含注解信息的字典结构，还原出对应的 BaseModel 或 TypedDict 类型。

    :param schema_info: 由 :py:func:`dump_schema_annotated_info` 生成的类型描述信息。
    :type schema_info: Dict

    :returns: 还原后的 BaseModel 或 TypedDict 类型。
    :rtype: Type[Union[BaseModel, Dict]]


.. py:class:: StateSchemaSerializable

    提供对状态 schema 的序列化与反序列化功能，通过继承该类，可以自动为子类添加对 ``state_schema: type[Union[BaseModel, TypedDict]]`` 属性的序列化、反序列化方法。


Utils
~~~~~~~~~~~~~~~~

.. py:function:: is_typeddict(tp) -> bool

    判断给定类型是否为 ``TypedDict``。

    :param tp: 待判断的类型。
    :type tp: type

    :returns: 如果是 ``TypedDict`` 类型则返回 ``True``，否则返回 ``False``。
    :rtype: bool

.. py:function:: is_basemodel(typ) -> bool

    判断给定类型是否为 Pydantic 的 ``BaseModel`` 子类。

    :param typ: 待判断的类型。
    :type typ: type

    :returns: 如果是 ``BaseModel`` 的子类则返回 ``True``，否则返回 ``False``。
    :rtype: bool

.. py:function:: is_dataclass(typ) -> bool

    判断给定类型是否为 Pyd 3.7+ 标准库中的 ``dataclass`` 或 Pydantic 模型（通过 ``__pydantic_config__`` 判断）。

    :param typ: 待判断的类型。
    :type typ: type

    :returns: 如果是 ``dataclass`` 或 Pydantic 模型则返回 ``True``，否则返回 ``False``。
    :rtype: bool

.. py:function:: strip_annotated(tp)

    若类型被 ``typing.Annotated`` 包装，则返回其原始类型；否则原样返回。

    :param tp: 可能被 ``Annotated`` 包装的类型。
    :type tp: type

    :returns: 原始未包装的类型。
    :rtype: type

.. py:function:: deep_dump(obj: Any) -> Any

    递归地将对象中所有值转换为字典形式。支持 ``BaseModel``、``dict``、``list`` 和 ``tuple`` 类型。

    :param obj: 需要转换的对象。
    :type obj: Any

    :returns: 转换后的嵌套字典结构。
    :rtype: Any

.. py:function:: fill_defaults(model_or_cls: type[BaseModel] | type[TypedDict], *, extra: Dict[str, Any] | None = None) -> Dict[str, Any]

    为 ``BaseModel`` 或 ``TypedDict`` 填充默认字段值，并可选地合并额外字段。

    :param model_or_cls: 目标模型或类型。
    :type model_or_cls: type[BaseModel] | type[TypedDict]

    :param extra: 可选的额外字段值字典。
    :type extra: Dict[str, Any] | None

    :returns: 包含默认值和额外字段的完整字段字典。
    :rtype: Dict[str, Any]

.. py:function:: safe_get_attr(data, attr, default=MISSING)

    安全地从 ``BaseModel`` 等对象或字典中获取属性值。

    :param data: 数据源，可以是对象或字典。
    :type data: Any

    :param attr: 属性名称。
    :type attr: str

    :param default: 默认值，当属性不存在时返回该值。
    :type default: Any

    :returns: 获取到的属性值或默认值。
    :rtype: Any

.. py:function:: safe_set_attr(data, attr, value)

    安全地为 ``BaseModel`` 等对象或字典设置属性值。

    :param data: 数据源，可以是对象或字典。
    :type data: Any

    :param attr: 属性名称。
    :type attr: str

    :param value: 要设置的属性值。
    :type value: Any

.. py:function:: safe_convert_to_schema(data, schema)

    将数据安全地转换为目标 schema 类型（如 Pydantic 模型）。

    :param data: 输入数据，可以是字典、对象或 BaseModel 实例。
    :type data: Any

    :param schema: 目标 schema 类型，应为 Pydantic 模型类。
    :type schema: type

    :returns: 转换后的 schema 实例或字典。
    :rtype: BaseModel | dict
