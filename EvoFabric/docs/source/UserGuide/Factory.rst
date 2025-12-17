.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

图的导出和重载
===============

我们支持通过 :py:meth:`~evofabric.core.graph.GraphBuilder.dump` 将添加到 :py:class:`~evofabric.core.graph.GraphBuilder` 中的节点、边、入口节点等信息导出成 ``json`` 文件。随后，可以在其他场景下通过 :py:meth:`~evofabric.core.graph.GraphBuilder.load` 加载该配置文件以重构 :py:class:`~evofabric.core.graph.GraphBuilder` 。

.. note::

    图中节点和节点依赖的组件使用 ``pydantic`` 特性序列化和反序列化，需要保证图中所有组件都是 :py:class:`~evofabric.core.factory.BaseComponent` 的子类。

    图中的函数（如，条件边的 ``router`` ， 多输入节点的 ``multi_input_merge_strategy`` 等，依赖 ``cloudpickle`` 的序列化和反序列化功能。

组件的序列化/反序列化
~~~~~~~~~~~~~~~~~~~~~


EvoFabric 中的组件都会继承 :py:class:`~evofabric.core.factory.BaseComponent` 。同时，我们也对不可序列化的入参单独定义了序列化和反序列化的方法。

继承了 :py:class:`~evofabric.core.factory.BaseComponent` 类后会支持：

* 支持通过 :py:meth:`~evofabric.core.factory.ComponentFactory.create` 动态创建类（继承过程中会自动将该类注册进工厂）。

* 支持通过 :py:meth:`~pydantic.BaseModel.model_dump_json` 和 :py:meth:`~pydantic.BaseModel.model_validate` 对类进行序列化和重载。

另外，当一个组件的入参类型定义是一个虚基类，但期望实际的入参是这个虚基类的子类时， ``pydantic`` 不能自动支持实例化子类，而是用子类的参数实例化基类，导致错误。因此，对于这类参数，可以使用 ``Annotated`` 声明注解为 :py:class:`~evofabric.core.factory.FactoryTypeAdapter` ，它会在 ``model_dump`` 时将类名保存到 ``__class_name__`` 字段，并在重载时调用 :py:class:`~evofabric.core.factory.ComponentFactory` 创建对应类的实例（注意该子类同样需要继承 :py:class:`~evofabric.core.factory.BaseComponent` ）。

.. note::
    子类同样需要继承 :py:class:`~evofabric.core.factory.BaseComponent` 以支持工厂实例化。

函数的序列化/反序列化
~~~~~~~~~~~~~~~~~~~~~~

框架提供基于 ``cloudpickle`` 的序列化/反序列化器，可以通过  :py:func:`~evofabric.core.factory.get_func_serializer` 获取实例。

图的序列化/反序列化过程中，会使用该实例对函数类型的参数进行序列化和反序列化。如果期望使用图的保存和重载，需要保证所有自定义函数都支持 ``pickle`` 。

如果反序列化过程中缺失一些模块，可以首先通过 :func:`~evofabric.core.factory.register_deserialize_modules` 注册缺失模块。

.. code-block:: python

    from evofabric.core.factory import get_func_serializer

    def custom_function():
        ...


    serialized = get_func_serializer().serialize(custom_function)

    deserialized_function = get_func_serializer().deserialize(serialized)


.. note::

    如果您需要替换当前的函数序列化/反序列化方法，可通过 :py:class:`~evofabric.core.factory.register_deserialize_modules` 注册一个符合 :py:class:`~evofabric.core.factory.FunctionSerializerProto` 接口的实例。


在图中使用自定义组件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

自定义组件放入图中时，需要遵循下面的要求才能支持图的保存和重载：

* 继承 :py:class:`~evofabric.core.factory.BaseComponent` 获取工厂支持和 ``BaseModel`` 特性。

* 使用 ``pydantic`` 风格定义类的入参。

* 显示声明函数类型和其他不可序列化的入参的序列化、反序列化方法。(对函数的 ``pickle`` 序列化可以使用 :py:func:`~evofabric.core.factory.get_func_serializer` 获取函数序列化/反序列化器)

示例：

.. code-block:: python

    import json
    from typing import Annotated, Any, Callable
    from pydantic import Field, field_serializer, field_validator
    from evofabric.core.clients import ChatClientBase, OpenAIChatClient
    from evofabric.core.factory import BaseComponent, FactoryTypeAdapter, get_func_serializer


    class CustomNode(BaseComponent):

        client: Annotated[ChatClientBase, FactoryTypeAdapter, Field(description="a parameter need FactoryTypeAdapter")]

        custom_router: Callable = Field(description="a router that need specific (de)serialize method")

        @field_serializer("custom_router")
        def serialize_stream_parser(self, _value: Callable) -> str:
            return get_func_serializer().serialize(_value)

        @field_validator('custom_router', mode='before')
        @classmethod
        def deserialize_stream_parser(cls, v: Any) -> Callable:
            if callable(v):
                return v
            return get_func_serializer().deserialize(v)

        async def __call__(self, *args, **kwargs):
            ...


    def custom_router():
        # do something
        ...


    node = CustomNode(client=OpenAIChatClient(model="your-model-name"), custom_router=custom_router)

    # dump agent to json string
    node_config = node.model_dump_json()

    # reload node from json string
    reload_node = CustomNode.model_validate(json.loads(node_config))

