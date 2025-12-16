.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

Tool Controller
===============

概述
----------------
我们同样支持通过 :py:class:`~evofabric.core.tool.ToolController` 动态管理工具的激活状态，以便Agent框架可以在某些情况下以最小改动关闭或打开某些工具。

核心功能
--------
:py:class:`~evofabric.core.tool.ToolController` 提供了基于规则的工具状态管理系统，核心功能包括：

1. **动态控制工具激活状态** - 通过规则模式匹配工具名称并激活/禁用工具
2. **默认行为配置** - 设置未匹配规则时的默认工具行为
3. **规则优先级** - 支持按优先级顺序应用规则，最先匹配的规则生效
4. **模式匹配** - 支持通配符模式匹配（如 `math_*` 匹配所有math服务器工具）
5. **动态工具管理** - 提供API在运行时激活/禁用特定工具

创建工具控制器
---------------
以下是创建和配置基本工具控制器的示例：

.. code-block:: python

    from evofabric.core.tool import ToolController

    # Create tool controller
    controller = ToolController(
        default_mode="deactivate",  # default_mode: deactivate all tools by default
        rules=[
            {
                "mode": "activate",
                "pattern": "math_*"  # activate all tools from the math server
            },
            {
                "mode": "deactivate",
                "pattern": "calc_*"   # deactivate tools from the calc server
            }
        ]
    )

    # After validation, rules will be automatically converted to objects
    print(controller.rules)
    # [ToolControlPattern(mode='activate', pattern='math_*'),
    #  ToolControlPattern(mode='deactivate', pattern='calc_*')]

规则详解
--------
规则是ToolController的核心，每个规则由两部分组成：

.. code-block:: python

    Rule = {
        "mode": "activate" | "deactivate",
        "pattern": "glob_wildcard_string"
    }

模式匹配规则遵循Unix shell通配符规范：

.. code-block:: python

    # Example patterns
    "math_*"         # Matches all tools from the math server
    "text_*"         # Matches all tools from the text server
    "math_calculator" # Matches the tool with exact name
    "*_calculator"   # Matches any tool ending with _calculator
    "upload_*"       # Matches all tools starting with upload

规则优先级示例
--------------
规则按顺序匹配最先匹配的规则生效：

.. code-block:: python

    controller = ToolController(
        default_mode="deactivate",
        rules=[
            {"mode": "activate", "pattern": "math_*"},    # First priority
            {"mode": "deactivate", "pattern": "math_add"}, # Second priority
        ]
    )

    # For tool name "math_add":
    # 1. First matches rule #1 -> activate
    # 2. Although rule #2 also matches, it has lower priority and is ignored
    print(controller.check_tool_status("math_add"))  # True

动态工具管理
------------
您可以在运行时激活或禁用特定工具，这些调整会立即生效并具有最高优先级：

.. code-block:: python

    from evofabric.core.tool import ToolManager

    # Create a tool manager with controller
    controller = ToolController(default_mode="deactivate")
    tool_manager = ToolManager(tools=[], tool_controller=controller)

    # Create some tools
    # ... (tool definitions) ...

    # Dynamically activate specific tools
    controller.activate_tool("text_translator")

    # Dynamically deactivate specific tools
    controller.deactivate_tool("math_complex")

    # Check tool status
    print(controller.check_tool_status("text_translator"))  # True
    print(controller.check_tool_status("math_complex"))     # False

过滤工具列表
-----------
ToolController提供了工具列表过滤功能，确保只返回激活的工具：

.. code-block:: python

    # Original tool list
    all_tools = [
        {"name": "math_add"},
        {"name": "math_subtract"},
        {"name": "text_translate"}
    ]

    # Filter active tools based on rules
    active_tools = controller.filter_tool_list(all_tools)

    # Assume rules:
    # - default_mode="deactivate"
    # - rules=[{"mode": "activate", "pattern": "math_*"}]
    # Result: Only math tools are activated
    print([tool["name"] for tool in active_tools])
    # Output: ['math_add', 'math_subtract']

最佳实践建议
------------
1. **规则顺序优化**：将精确匹配规则放在通用规则之前
2. **默认模式选择**：

   - `activate`：适合白名单场景（仅管控禁用特定工具）
   - `deactivate`：适合黑名单场景（仅管控激活特定工具）

3. **模式命名**：

   - 使用服务器名前缀如 `math_*` 组织规则
   - 避免过于宽泛的通配符（如 `*_*` 会匹配所有工具）

4. **动态管理优先级**：通过 `activate_tool()` 和 `deactivate_tool()` 实现紧急开关