.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

示例代码
=================


顺序图的构建
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    from typing import Annotated

    from pydantic import BaseModel

    from evofabric.core.graph import GraphBuilder
    from evofabric.core.typing import State, StateDelta


    def node_a(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node a"}],
            "marker": "node_a"
        }


    def node_b(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node b"}],
            "marker": "node_b"
        }


    def node_c(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node c"}],
            "marker": "node_c"
        }


    class StateSchema(BaseModel):
        messages: Annotated[list, "append_messages"]
        marker: Annotated[str, "overwrite"]


    async def main():
        graph_builder = GraphBuilder(state_schema=StateSchema)
        graph_builder.add_node("a", node_a)
        graph_builder.add_node("b", node_b)
        graph_builder.add_node("c", node_c)

        graph_builder.add_edge("a", "b")
        graph_builder.add_edge("b", "c")
        graph_builder.set_entry_point("a")

        engine = graph_builder.build()
        engine.draw_graph()

        response = await engine.run({
            "messages": [{"role": "user", "content": "hello"}]
        })
        print(response.model_dump_json(indent=4))


    if __name__ == "__main__":
        asyncio.run(main())

    

可视化结果：

.. mermaid::

    flowchart LR
        a(a)
        b(b)
        c(c)
        __start__(__start__)
        __end__(__end__)
        a -->  b
        b -->  c
        __start__ -->  a
        c -->  __end__


输出结果：

.. code-block:: json

    {
        "messages": [
            {
                "content": "hello",
                "node_name": null,
                "msg_id": "1ae3c964-ec7f-4d22-83c7-4ec64d6e6042",
                "role": "user"
            },
            {
                "content": "Msg from node a",
                "node_name": "a",
                "msg_id": "c4c206fe-0b59-4632-9db4-e064e819dbae",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node b",
                "node_name": "b",
                "msg_id": "91cb5a6b-21d6-46f5-ba1a-dedc92bfda29",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node c",
                "node_name": "c",
                "msg_id": "572c3689-7235-4919-9f62-e4033602aaf7",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            }
        ],
        "marker": "node_c"
    }


分支图的构建
~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    from typing import Annotated

    from pydantic import BaseModel

    from evofabric.core.graph import GraphBuilder
    from evofabric.core.typing import State, StateDelta


    def node_a(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node a"}],
            "marker": "node_a"
        }


    def node_b(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node b"}],
            "marker": "node_b"
        }


    def node_c(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node c"}],
            "marker": "node_c"
        }


    class StateSchema(BaseModel):
        messages: Annotated[list, "append_messages"]
        marker: Annotated[str, "overwrite"]


    async def main():
        graph_builder = GraphBuilder(state_schema=StateSchema)
        graph_builder.add_node("a", node_a)
        graph_builder.add_node("b", node_b)
        graph_builder.add_node("c", node_c)

        graph_builder.add_edge("a", "b")
        graph_builder.add_edge("a", "c")
        graph_builder.set_entry_point("a")

        engine = graph_builder.build()
        engine.draw_graph()

        response = await engine.run({
            "messages": [{"role": "user", "content": "hello"}]
        })
        print(response.model_dump_json(indent=4))


    if __name__ == "__main__":
        asyncio.run(main())


可视化结果：

.. mermaid::

    flowchart LR
        a(a)
        b(b)
        c(c)
        __start__(__start__)
        __end__(__end__)
        a -->  b
        a -->  c
        __start__ -->  a
        b -->  __end__
        c -->  __end__


输出结果：

.. code-block:: json

    {
        "messages": [
            {
                "content": "hello",
                "node_name": null,
                "msg_id": "7f7bdab9-ab85-458c-aab6-35137d583f40",
                "role": "user"
            },
            {
                "content": "Msg from node a",
                "node_name": "a",
                "msg_id": "c2e70210-8b31-40a9-a5b9-ce2a239608eb",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node b",
                "node_name": "b",
                "msg_id": "e80ad96e-71c2-4e3b-8f4a-93f0da0fa816",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node c",
                "node_name": "c",
                "msg_id": "12bf38e0-2857-45eb-881a-2b32f97c90ed",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            }
        ],
        "marker": "node_c"
    }


带有多输入节点的图的构建
~~~~~~~~~~~~~~~~~~~~~~~~

ALL模式
--------------

节点在具有多个前驱输入时，可以通过设置 ``action_mode='all'`` 来等待所有前置节点运行完成。此时该节点的输入也是所有前置节点的输出按 ``state_schema`` 声明的更新策略合并后的结果。

.. code-block:: python

    import asyncio
    from typing import Annotated

    from pydantic import BaseModel

    from evofabric.core.factory import safe_get_attr
    from evofabric.core.graph import GraphBuilder
    from evofabric.core.typing import State, StateDelta


    def node_a(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node a"}],
            "marker": "node_a"
        }


    def node_b(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node b"}],
            "marker": "node_b"
        }


    def node_c(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node c"}],
            "marker": "node_c"
        }


    def node_d(state: State) -> StateDelta:
        messages = safe_get_attr(state, "messages")

        merged_msg = []
        for msg in messages:
            merged_msg.append(safe_get_attr(msg, "content"))

        return {
            "messages": [{"role": "assistant", "content": "I have received msgs: " + "\n".join(merged_msg)}],
            "marker": "node_d"
        }


    class StateSchema(BaseModel):
        messages: Annotated[list, "append_messages"]
        marker: Annotated[str, "overwrite"]


    async def main():
        graph_builder = GraphBuilder(state_schema=StateSchema)
        graph_builder.add_node("a", node_a)
        graph_builder.add_node("b", node_b)
        graph_builder.add_node("c", node_c)
        graph_builder.add_node("d", node_d, action_mode="all")

        graph_builder.add_edge("a", "b")
        graph_builder.add_edge("a", "c")
        graph_builder.add_edge("c", "d")
        graph_builder.add_edge("b", "d")
        graph_builder.set_entry_point("a")

        engine = graph_builder.build()
        engine.draw_graph()

        response = await engine.run({
            "messages": [{"role": "user", "content": "hello"}]
        })
        print(response.model_dump_json(indent=4))


    if __name__ == "__main__":
        asyncio.run(main())


可视化结果：

.. mermaid::

    flowchart LR
        a(a)
        b(b)
        c(c)
        d(d)
        __start__(__start__)
        __end__(__end__)
        a -->  b
        a -->  c
        c -->  d
        b -->  d
        __start__ -->  a
        d -->  __end__

示例输出：

.. code-block:: json

    {
        "messages": [
            {
                "content": "hello",
                "node_name": null,
                "msg_id": "e8e1ba56-aae9-49dc-86fa-a2b91de9d30e",
                "role": "user"
            },
            {
                "content": "Msg from node a",
                "node_name": "a",
                "msg_id": "09a19de1-04e2-40a9-9023-3a71d9b3d561",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node c",
                "node_name": "c",
                "msg_id": "fde95bab-15d1-47e0-95aa-f5e12866f8b9",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node b",
                "node_name": "b",
                "msg_id": "4321f2ea-811d-4103-aede-a2997782f5f7",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "I have received msgs: hello\nMsg from node a\nMsg from node c\nMsg from node b",
                "node_name": "d",
                "msg_id": "3991a4ec-3ba4-40bd-959d-0550fcf45370",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            }
        ],
        "marker": "node_d"
    }


ANY模式
--------------------

将上述代码 ``action_mode`` 改为 ``any`` 时，会发现 ``d`` 节点会被触发两次，这是因为 ``any`` 模式下任意一个前驱节点运行结束都会触发一次当前节点的运行。


示例输出

.. code-block:: json

    {
        "messages": [
            {
                "content": "hello",
                "node_name": null,
                "msg_id": "a2785b99-50d6-4b71-9829-414b21f6d957",
                "role": "user"
            },
            {
                "content": "Msg from node a",
                "node_name": "a",
                "msg_id": "bf6214ef-16e6-46e3-bb0c-4eb0ddbc9b35",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node b",
                "node_name": "b",
                "msg_id": "63976b07-51f3-41f2-b838-9ea550d89f52",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "I have received msgs: hello\nMsg from node a\nMsg from node b",
                "node_name": "d",
                "msg_id": "15356ef0-cd1f-4534-9a62-924e580df240",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node c",
                "node_name": "c",
                "msg_id": "29943db7-0733-4a35-a335-240dd89f13c2",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "I have received msgs: hello\nMsg from node a\nMsg from node c",
                "node_name": "d",
                "msg_id": "c7efdf3f-fb57-4c97-b5dd-c6bee414e901",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            }
        ],
        "marker": "node_d"
    }


带条件分支图的构建
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    import random
    from typing import Annotated

    from pydantic import BaseModel

    from evofabric.core.factory import safe_get_attr
    from evofabric.core.graph import GraphBuilder
    from evofabric.core.typing import State, StateDelta


    def node_a(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node a"}],
            "marker": "node_a"
        }


    def node_b(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node b"}],
            "marker": "node_b",
            "random_num": random.randint(1, 100)
        }


    def node_c(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node c"}],
            "marker": "node_c"
        }


    def node_d(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node d"}],
            "marker": "node_d"
        }


    class StateSchema(BaseModel):
        messages: Annotated[list, "append_messages"]
        marker: Annotated[str, "overwrite"]
        random_num: Annotated[int, "overwrite"]


    def condition_router(state):
        random_num = safe_get_attr(state, "random_num")

        if random_num > 50:
            return "c"
        return "d"


    async def main():
        graph_builder = GraphBuilder(state_schema=StateSchema)

        graph_builder.add_node("a", node_a)
        graph_builder.add_node("b", node_b)
        graph_builder.add_node("c", node_c)
        graph_builder.add_node("d", node_d)

        graph_builder.add_edge("a", "b")
        graph_builder.add_condition_edge("b", condition_router, possible_targets={"c", "d"})
        graph_builder.set_entry_point("a")
        engine = graph_builder.build()

        engine.draw_graph()

        response = await engine.run({
            "messages": [{"role": "user", "content": "hello"}]
        })
        print(response.model_dump_json(indent=4))

    if __name__ == "__main__":
        asyncio.run(main())



可视化结果：

.. mermaid::

    flowchart LR
        a(a)
        b(b)
        c(c)
        d(d)
        __start__(__start__)
        __end__(__end__)
        a -->  b
        b -.->  c
        b -.->  d
        __start__ -->  a
        c -->  __end__
        d -->  __end__


示例输出：

.. code-block:: json

    {
        "messages": [
            {
                "content": "hello",
                "node_name": null,
                "msg_id": "02c0d4b4-0eab-4e9d-b8d5-651eeb16060b",
                "role": "user"
            },
            {
                "content": "Msg from node a",
                "node_name": "a",
                "msg_id": "4a7eaa34-f048-497f-94c4-2859d0f9f95d",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node b",
                "node_name": "b",
                "msg_id": "784d4e44-202f-4254-bec9-563caf38ab62",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node d",
                "node_name": "d",
                "msg_id": "191c4b43-3488-435b-a489-9db0c2b37895",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            }
        ],
        "marker": "node_d",
        "random_num": 23
    }


用条件分支构造自循环
~~~~~~~~~~~~~~~~~~~~~~~~~

自循环可以通过条件边构造。需要注意，如果节点 ``action_mode='all'``，需要单独为自循环条件边指定一个非默认的组名。否则这个节点将永远无法触发。

或者可以简单地将改节点的 ``action_mode`` 设为 ``any`` 。


.. code-block:: python

    import asyncio
    import random
    from typing import Annotated

    from pydantic import BaseModel

    from evofabric.core.factory import safe_get_attr
    from evofabric.core.graph import GraphBuilder
    from evofabric.core.typing import State, StateDelta


    def node_a(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node a"}],
            "marker": "node_a"
        }


    def node_b(state: State) -> StateDelta:
        random_num = random.randint(1, 100)
        return {
            "messages": [{"role": "assistant", "content": f"Msg from node b, random num is {random_num}"}],
            "marker": "node_b",
            "random_num": random_num
        }


    def node_c(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node c"}],
            "marker": "node_c"
        }


    class StateSchema(BaseModel):
        messages: Annotated[list, "append_messages"]
        marker: Annotated[str, "overwrite"]
        random_num: Annotated[int, "overwrite"]


    def self_loop_condition_router(state):
        random_num = safe_get_attr(state, "random_num")

        if random_num > 50:
            return "b"
        return "c"


    async def main():
        graph_builder = GraphBuilder(state_schema=StateSchema)

        graph_builder.add_node("a", node_a)
        graph_builder.add_node("b", node_b, action_mode="all")
        graph_builder.add_node("c", node_c)

        graph_builder.add_edge("a", "b")
        graph_builder.add_condition_edge("b", self_loop_condition_router, possible_targets={"b", "c"}, group="b-self-loop")
        graph_builder.set_entry_point("a")
        engine = graph_builder.build()

        engine.draw_graph()

        response = await engine.run({
            "messages": [{"role": "user", "content": "hello"}]
        })
        print(response.model_dump_json(indent=4))

    if __name__ == "__main__":
        asyncio.run(main())



可视化结果：

.. mermaid::

    flowchart LR
        a(a)
        b(b)
        c(c)
        __start__(__start__)
        __end__(__end__)
        a -->  b
        b -.-> |"b-self-loop"| c
        b -.-> |"b-self-loop"| b
        __start__ -->  a
        c -->  __end__

示例输出：

.. code-block:: json

    {
        "messages": [
            {
                "content": "hello",
                "node_name": null,
                "msg_id": "5be38bca-5a75-4336-acd4-793de5c4ab8e",
                "role": "user"
            },
            {
                "content": "Msg from node a",
                "node_name": "a",
                "msg_id": "f8b0f010-8c35-4fd9-8e66-9fcdd3aae851",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node b, random num is 95",
                "node_name": "b",
                "msg_id": "80b0e523-9df6-4927-be16-c1ea6ecf2c0c",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node b, random num is 57",
                "node_name": "b",
                "msg_id": "c208f9a8-fdd6-4ecd-b70b-36706077749a",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node b, random num is 17",
                "node_name": "b",
                "msg_id": "beb25ed1-6c69-48c1-867a-8f2726c71643",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "Msg from node c",
                "node_name": "c",
                "msg_id": "976b6afe-de0b-4bfa-9ca2-2185f99a37aa",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            }
        ],
        "marker": "node_c",
        "random_num": 17
    }


带multi_input_merge_strategy的图
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当节点需要从一到多组路径下接收多个前驱节点的输入，并指定特殊的合并策略时，可以使用 ``multi_input_merge_strategy`` 按边的组别指定合并策略。

合并策略要求是一个函数，接收 ``list[State]`` 作为入参，并输出合并后的 ``State`` 状态。

.. note::

    ``multi_input_merge_strategy`` 会导致状态根节点的重置和不可恢复。

    :doc:`参考状态维护与重构机制 <state_while_running>`

.. code-block:: python

    import asyncio
    from typing import Annotated

    from pydantic import BaseModel

    from evofabric.core.factory import safe_get_attr
    from evofabric.core.graph import GraphBuilder
    from evofabric.core.typing import DEFAULT_EDGE_GROUP, State, StateDelta


    def node_a(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node a"}],
            "marker": "node_a"
        }


    def node_b(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node b"}],
            "marker": "node_b"
        }


    def node_c(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node c"}],
            "marker": "node_c"
        }


    def node_d(state: State) -> StateDelta:
        messages = safe_get_attr(state, "messages")

        merged_msg = []
        for msg in messages:
            merged_msg.append(safe_get_attr(msg, "content"))

        return {
            "messages": [{"role": "assistant", "content": "I have received msgs: " + "\n".join(merged_msg)}],
            "marker": "node_d"
        }


    def node_b2(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node b2"}],
            "marker": "node_b"
        }


    def node_c2(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node c2"}],
            "marker": "node_c"
        }


    def node_e(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node e"}],
            "marker": "node_e"
        }


    def node_f(state: State) -> StateDelta:
        return {
            "messages": [{"role": "assistant", "content": "Msg from node f"}],
            "marker": "node_f"
        }


    class StateSchema(BaseModel):
        messages: Annotated[list, "append_messages"]
        marker: Annotated[str, "overwrite"]


    def default_group_merge(state_list) -> State:
        merged_msg = {}
        for state in state_list:
            msgs = safe_get_attr(state, "messages")
            for msg in msgs:
                merged_msg[safe_get_attr(msg, "msg_id")] = safe_get_attr(msg, "content")
        return {"messages": [{"role": 'user', "content": "Default Input message: [" + "\n".join(list(merged_msg.values())) + "]"}]}


    def v2_group_merge(state_list) -> State:
        merged_msg = {}
        for state in state_list:
            msgs = safe_get_attr(state, "messages")
            for msg in msgs:
                merged_msg[safe_get_attr(msg, "msg_id")] = safe_get_attr(msg, "content")
        return {"messages": [{"role": 'user', "content": "V2 Input message: [" + "\n".join(list(merged_msg.values())) + "]"}]}


    async def main():
        graph_builder = GraphBuilder(state_schema=StateSchema)
        graph_builder.add_node("a", node_a)
        graph_builder.add_node("b", node_b)
        graph_builder.add_node("c", node_c)
        graph_builder.add_node("b2", node_b2)
        graph_builder.add_node("c2", node_c2)
        graph_builder.add_node("d", node_d, action_mode="all", multi_input_merge_strategy={
            DEFAULT_EDGE_GROUP: default_group_merge,
            "v2": v2_group_merge
        })
        graph_builder.add_node("e", node_e)
        graph_builder.add_node("f", node_f)

        graph_builder.add_edge("a", "b")
        graph_builder.add_edge("a", "c")
        graph_builder.add_edge("a", "e")
        graph_builder.add_edge("a", "f")

        graph_builder.add_edge("e", "b2")
        graph_builder.add_edge("f", "c2")

        graph_builder.add_edge("c", "d")
        graph_builder.add_edge("b", "d")

        graph_builder.add_edge("c2", "d", group="v2")
        graph_builder.add_edge("b2", "d", group="v2")

        graph_builder.set_entry_point("a")

        engine = graph_builder.build()
        engine.draw_graph()

        response = await engine.run({
            "messages": [{"role": "user", "content": "hello"}]
        })
        print(response.model_dump_json(indent=4))


    if __name__ == "__main__":
        asyncio.run(main())

可视化结果：

.. mermaid::

    flowchart TD
        a(a)
        b(b)
        c(c)
        b2(b2)
        c2(c2)
        d(d)
        e(e)
        f(f)
        __start__(__start__)
        __end__(__end__)
        a -->  b
        a -->  c
        a -->  e
        a -->  f
        e -->  b2
        f -->  c2
        c -->  d
        b -->  d
        c2 --> |"v2"| d
        b2 --> |"v2"| d
        __start__ -->  a
        d -->  __end__

示例输出：

.. code-block:: json

    {
        "messages": [
            {
                "content": "Default Input message: [hello\nMsg from node a\nMsg from node c\nMsg from node b]",
                "node_name": null,
                "msg_id": "f8ffaee9-f0da-4ba4-9ec2-44f67d96ad30",
                "role": "user"
            },
            {
                "content": "I have received msgs: Default Input message: [hello\nMsg from node a\nMsg from node c\nMsg from node b]",
                "node_name": "d",
                "msg_id": "f0fa7037-2c2e-4a2e-8af9-9f272cf1f33f",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            },
            {
                "content": "V2 Input message: [hello\nMsg from node a\nMsg from node f\nMsg from node c2\nMsg from node e\nMsg from node b2]",
                "node_name": null,
                "msg_id": "35d6f21a-1fdd-4425-94dc-a120103c4f14",
                "role": "user"
            },
            {
                "content": "I have received msgs: V2 Input message: [hello\nMsg from node a\nMsg from node f\nMsg from node c2\nMsg from node e\nMsg from node b2]",
                "node_name": "d",
                "msg_id": "15060c35-30a8-4c09-9645-32d7b888cfe7",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            }
        ],
        "marker": "node_d"
    }

带StateFilter的图
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

向某些节点传递消息时，有时并不希望子节点看到全量的上下文。

例如在中控 -> 执行器场景：

- 执行器往往只需要看到中控器下发的任务来避免潜在的干扰。

- 执行器也只需要返回执行结果，而不需要返回执行过程。

此时，可以使用带有 ``state_filter`` 参数的边对状态里的上下文 ``messages`` 做过滤。

.. note::

    由于 ``state_filter`` 会导致状态根节点的重置和不可恢复，如果仍然希望保留历史上下文信息，我们建议您将信息暂存在状态的其他字段中，并在下一次调用时取用。

    :doc:`参考状态维护与重构机制 <state_while_running>`

.. code-block:: python

    import asyncio
    from typing import Annotated

    from pydantic import BaseModel

    from evofabric.core.factory import safe_get_attr, safe_set_attr
    from evofabric.core.graph import GraphBuilder
    from evofabric.core.typing import State


    class StateSchema(BaseModel):
        messages: Annotated[list, "append_messages"]


    def central(state: State):
        return {
            "messages": [{"role": "assistant", "content": "task ... assigned to execution1"}]
        }


    def execution1(state: State):
        return {"messages": [{"role": "assistant", "content": "execution1 done"}]}


    def execution2(state: State):
        return {"messages": [{"role": "assistant", "content": "execution2 done"}]}


    def last_assistant_to_user(state: State) -> State:
        """
        keep last assistant msg as user query for execution node
        """
        messages = safe_get_attr(state, "messages")
        for msg in reversed(messages):
            if safe_get_attr(msg, "role") == "assistant":
                safe_set_attr(state, "messages", [{"role": "user", "content": safe_get_attr(msg, "content")}])
                return state
        return state


    def central_router(state: State):
        """A fixed router for the demo."""
        return "execution1", last_assistant_to_user


    async def main():
        builder = GraphBuilder(state_schema=StateSchema)

        builder.add_node("central", central)
        builder.add_node("execution1", execution1)
        builder.add_node("execution2", execution2)

        builder.add_condition_edge(
            "central",
            router=central_router,
            possible_targets={"execution1", "execution2"}
        )

        builder.add_edge("execution1", "end")

        builder.set_entry_point("central")
        engine = builder.build()
        engine.draw_graph()

        result = await engine.run({
            "messages": [{"role": "user", "content": "hello, do something"}]
        })
        print(result.model_dump_json(indent=4))


    if __name__ == "__main__":
        asyncio.run(main())



可视化结果：

.. mermaid::

    flowchart TD
        central(central)
        execution1(execution1)
        execution2(execution2)
        __start__(__start__)
        __end__(__end__)
        central -.->  execution2
        central -.->  execution1
        execution1 -->  __end__
        __start__ -->  central
        execution2 -->  __end__

示例输出：

.. code-block:: json

    {
        "messages": [
            {
                "content": "task ... assigned to execution1",
                "node_name": null,
                "msg_id": "65f09c4a-959b-4976-8354-69d6e1b65da5",
                "role": "user"
            },
            {
                "content": "execution1 done",
                "node_name": "execution1",
                "msg_id": "d1d869f3-107f-432f-8a6f-9fb6f663c245",
                "role": "assistant",
                "reasoning_content": null,
                "tool_calls": null,
                "usage": null
            }
        ]
    }