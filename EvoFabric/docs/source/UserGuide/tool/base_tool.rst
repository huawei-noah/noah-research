.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

BaseTool
=========================


概述
------------------

:py:class:`~evofabric.core.tool.BaseTool` 是单个工具基础类型，它负责记录具体的工具函数接口、解析函数入参、提供工具 ``schema`` 定义、维护工具内部状态等基础功能。它是 :py:class:`~evofabric.core.tool.ToolManager` 管理的基础工具单元。

特性
------------------

* **工具调用能力**: 通过 :py:class:`~evofabric.core.tool.BaseTool.__call__` 可完成对工具的调用。
* **工具状态管理**：在调用工具时，如果工具本身带有 ``inner_state`` 参数，会将 BaseTool 管理的状态填充进工具入参。工具状态还支持导出和重载。
* **流式消息管理**: 在调用工具时，如果工具本身带有 ``stream_writer`` 参数，会将 :py:class:`~evofabric.core.graph.StreamWriter` 填充进工具入参，以接收工具输出的流式消息。

.. Note::
    
    EvoFabric 框架中工具的内部状态本质上是指该工具需要的输入参数。该参数区别于普通参数的关键在于：该参数的值只需要初始化，之后工具调用时不再需要显式输入，而是依赖于该工具上次调用后该参数的变动结果。


工具调用 & 获取 ToolSchema
---------------------------------------

.. code-block:: python

    from evofabric.core.tool import BaseTool
    import asyncio

    async def main():
        # The core of the tool is a Python function
        def add(a: float, b: float):
            "add two numbers"
            return a + b

        # Initialize BaseTool
        new_tool = BaseTool(
            name='add',
            description='add two float numbers.',
            func=add
        )

        # Get tool schema
        tool_schema = new_tool.get_tool_schema()
        print(tool_schema)

        # Call BaseTool
        ans = await new_tool(a=1, b=2)
        print(ans)

    asyncio.run(main())


带状态的工具调用
---------------------------------

.. code-block:: python

    from evofabric.core.tool import BaseTool
    import asyncio
    import os
    from evofabric.core.typing import ToolInnerState


    async def main():

        # Define the tool
        async def mycd(path: str, inner_state: ToolInnerState):
            '''
            change directory.
            '''
            new_path = os.path.join(inner_state.state['current_dir'], path)
            inner_state.state['current_dir'] = new_path
            return new_path

        # Tool internal state
        inner_state = ToolInnerState(
            state={
                "current_dir": "/xxx/xxx"
            }
        )

        # Initialize BaseTool
        new_tool = BaseTool(
            name="mycd",  # tool name
            description="change directory.",  # tool description
            func=mycd,  # core of the tool
            inner_state=inner_state,  # tool internal state
            exclude_params=["inner_state"]  # exclude inner_state (tool internal state) from tool schema
        )

        # Get BaseTool tool schema
        tool_schema = new_tool.get_tool_schema()
        print(tool_schema)  # inner_state will not appear

        # Call BaseTool
        ans = await new_tool(path="sss/")
        print(ans)  # "/xxx/xxx/sss/"

        # BaseTool internal state changes
        print(new_tool.inner_state.state['current_dir'])  # "/xxx/xxx/sss/"

        # Get BaseTool internal state
        state1 = await new_tool.dump_state()

        # Overwrite BaseTool internal state
        await new_tool.load_state(input_state=state1)

    asyncio.run(main())





