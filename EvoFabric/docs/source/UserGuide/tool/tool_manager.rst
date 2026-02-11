.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

ToolManager
==============================


概述
------------------

:py:class:`~evofabric.core.tool.ToolManager` 继承自 :py:class:`~evofabric.core.tool.ToolManagerBase` ，主要用于管理使用Python定义的工具。类型包括：

- 工具基类 :py:class:`~evofabric.core.tool.BaseTool`
- Python函数
- Python类的成员函数
- python文件下的全部Python函数 (支持指定 ``pattern`` 纳入和排除)

在Agent场景，负责向 :py:class:`~evofabric.core.agent.AgentNode` 提供工具列表，根据大语言模型(LLM)输出的工具调用指令执行对应工具，并向LLM反馈工具执行结果。

同时，还支持批量导出和重载所有工具的内部状态。


工具的添加、删除、查找和更新
----------------------------

:py:class:`~evofabric.core.tool.ToolManager` 可通过不同的接口对内部工具进行增、删、改、查。


.. code-block:: python

    import asyncio
    import math

    from evofabric.core.tool import ToolManager, BaseTool


    async def MULTIPLY(a: int, b: int):
        """MULTIPLY two numbers."""
        return a * b


    def mydivide(a: float, b: float):
        """divide two numbers."""
        return a / b


    def noinputfun():
        """The function has no params"""

        return "The function has no params."


    class MyMath:
        def __init__(self) -> None:
            pass

        @staticmethod
        def cubic(a: float):
            """get a^3."""
            return a ** 3

        def mypow(self, a: int, power: int):
            """get any power of a."""
            return pow(a, power)

        @classmethod
        def logfun(cls, a: float):
            """my log function. a > 0."""
            return math.log(a)


    async def main():
        mymath = MyMath()

        # =======================================================

        # ToolManager initialize
        tool_manager = ToolManager(
            tools=[MULTIPLY, noinputfun],
        )

        # =======================================================

        # add tools from python function
        tool_manager.add_callable_tools(
            tools=[mydivide, MyMath.logfun, mymath.mypow])

        async def add(a: float, b: float):
            "add two numbers"
            return a + b

        new_tool = BaseTool(
            name='add',
            description='add two float numbers.',
            func=add
        )
        # add tools from BaseTool instance
        tool_manager.add_callable_tools(
            tools=[new_tool])

        # add tools from python scripts
        tool_manager.add_python_file_tools(
            file_paths=["py_file_with_tools.py"],
            exclude_pattern_list=[["mysq*"]])

        # =======================================================

        # list tools, get all tool schemas in the toolmanager
        res = await tool_manager.list_tools()
        print("tools after update tools: \n", res)

        # =======================================================

        # delete tools
        tool_manager.delete_tools(['mydivide'])

        # =======================================================

        # update tools
        new_tool_v2 = BaseTool(
            name='add',
            description='add two FLOAT NUMBERS.',
            func=add
        )
        tool_manager.update_tools([new_tool_v2])

        # =======================================================

        # find tools
        ans = tool_manager.find_tools(['add', 'MULTIPLY'])
        print("found tools: \n", ans)

        # =======================================================


    if __name__ == '__main__':
        asyncio.run(main())

工具调用
--------------------

调用 :py:meth:`~evofabric.core.tool.ToolManager.call_tools` 可以并行执行多个工具调用指令。

.. code-block:: python

    import json
    import math

    from evofabric.core.tool import ToolManager
    from evofabric.core.typing import Function, ToolCall


    async def MULTIPLY(a: int, b: int):
        """MULTIPLY two numbers."""
        return a * b


    def mydivide(a: float, b: float):
        """divide two numbers."""
        return a / b


    def noinputfun():
        """The function has no params."""

        return "The function has no params."


    class MyMath:
        def __init__(self) -> None:
            pass

        @staticmethod
        def cubic(a: float):
            """get a^3."""
            return a ** 3

        def mypow(self, a: int, power: int):
            """get any power of a."""
            return pow(a, power)

        @classmethod
        def logfun(cls, a: float):
            """my log function. a > 0."""
            return math.log(a)


    async def main():
        mymath = MyMath()

        toolcall_1 = ToolCall(
            id="1",
            function=Function(
                name="MULTIPLY",
                arguments=json.dumps(
                    {
                        "a": 3,
                        "b": 4
                    }
                )
            )

        )
        toolcall_2 = ToolCall(
            id="2",
            function=Function(
                name="mydivide",
                arguments=json.dumps(
                    {
                        "a": 4,
                        "b": 0
                    }
                )
            )

        )
        toolcall_3 = ToolCall(
            id="3",
            function=Function(
                name="noinputfun",
                arguments=json.dumps({}),
            )

        )

        tool_manager = ToolManager(
            tools=[MULTIPLY, mydivide, noinputfun],
        )

        # =======================================================

        # call tools
        res = await tool_manager.call_tools([toolcall_1, toolcall_2, toolcall_3])
        print("call tool result: \n", res)


工具状态的导出和重载
-------------------------------------------------

针对带有 ``inner_state`` 的工具，ToolManager可以导出、重载工具的内部状态。

.. code-block:: python

    import asyncio
    import json
    import os

    from evofabric.core.tool import ToolManager
    from evofabric.core.typing import Function, ToolCall, ToolInnerState


    async def mycd(path: str, inner_state: ToolInnerState):
        new_path = os.path.join(inner_state.state['current_dir'], path)
        inner_state.state['current_dir'] = new_path
        return new_path


    async def main():
        inner_state = ToolInnerState(
            state={
                "current_dir": "/xxx/xxx"
            }
        )

        tool_manager = ToolManager(
            tools=[(mycd, inner_state)],
        )

        # Save the state of all tools in tool_manager to "test_tool_manager_state_start.json"
        await tool_manager.save_state(save_path="test_tool_manager_state_start.json")

        # Tool call information, only explicitly pass parameters that are not internal state;
        # parameters that belong to internal state are automatically passed from the tool's internal attributes
        toolcall_5 = ToolCall(
            id="5",
            function=Function(
                name="mycd",
                arguments=json.dumps({
                    "path": "sss/",
                }),
            )

        )

        # When calling the tool, the tool's internal state is updated in real time according to its logic
        res1 = await tool_manager.call_tools([toolcall_5])

        # Reload the internal state of tool_manager to the state saved at the beginning
        await tool_manager.load_state(load_path="test_tool_manager_state_start.json")


    if __name__ == '__main__':
        asyncio.run(main())
