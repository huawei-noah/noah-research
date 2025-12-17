.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

===================
MemBase
===================

------------------
概述
------------------

MemBase 定义了Memory与智能体交互基础协议。在使用Evofabric框架时, 可以通过继承MemBase并实现其基础方法完成自定义记忆使用

------------------
特性
------------------

* **统一接口**: 所有记忆系统继承自 :py:class:`~evofabric.core.mem.MemBase`，提供一致的调用方式。
* **记忆拓展**: 用户可继承MemBase并定义自己的方法。
* **示例记忆**: EvoFabric 在MemBase的基础上实现了RetrievalMem，ChatMem和TaskMem三类常用记忆类型，供用户直接使用。

开源记忆适配
--------------------
实现其中retrieval_update, add_messages和clear方法（均为异步）

.. code-block:: python

    from typing import Dict, Optional, Union, List
    from pydantic import Field, PrivateAttr
    from mem0 import AsyncMemory
    from evofabric.core.mem import MemBase
    from evofabric.core.typing import StateMessage, UserMessage


    # Example implementation of Mem0
    class Mem0Module(MemBase):
            message_rounds: int = Field(default=20,
                                        description="Messages rounds considered in retrieval and save")
            mem_config: Dict[str, Union[str, bool, Dict]] = Field(description="Mem0 initialization config")
            user_id: str = Field(default="default", description="mem0 user_id")
            _mem0_client: Optional[AsyncMemory] = PrivateAttr(default=None)

            async def get_client(self) -> AsyncMemory:
                """Initialize AsyncMem when first call"""
                if self._mem0_client is None:
                    self._mem0_client = await AsyncMemory.from_config(self.mem_config)
                return self._mem0_client

            async def retrieval_update(self, messages: List[StateMessage], **kwargs) -> List[StateMessage]:
                """Implement the retrieval_update method"""
                context = "context: "
                for msg in messages[-self.message_rounds:]:
                    context += f"\n{msg.role}: {msg.content}"
                _mem0_client = await self.get_client()
                items = await _mem0_client.search(context, user_id=self.user_id)
                contents = [item['memory'] for item in items['results']]
                messages = [UserMessage(content="Historical memory: " + str(contents))] + messages
                return messages

            # Implement memory update functionality
            async def add_messages(self, messages: List[StateMessage], **kwargs) -> None:
                """Implement the add_messages method"""
                _mem0_client = await self.get_client()
                message_dict = [
                    {"role": msg.role, "content": msg.content} for msg in messages[-self.message_rounds:]
                ]
                await _mem0_client.add(message_dict, user_id=self.user_id)
                return

            async def clear(self):
                """Implement the clear method"""
                _mem0_client = await self.get_client()
                await _mem0_client.reset()


接入Agent
--------------------
当前Agent支持传入一个或多个Memory系统。
检索：在Agent推理推理前，会依据上下文信息逐个记忆系统检索，并在消息列表中拼接UserMessage信息。入口方法即retrieval_update。
更新：在智能体推理结束后，将上下文信息逐一更新记忆库。入口方法即 add_messages。
以上执行流程均在智能体内部实现，只需将记忆模块接入即可。

.. code-block:: python

    from evofabric.core.agent import AgentNode

    # Create an English chat mem
    chat_mem = ChatMem(
        zh_mode=True,
        vectorstore=vectorstore,
        chat_client=test_client_common,
        feat_define_prompt="Please extract the user's professional identity and personal habits."
    )
 
    # Create an agent node using memory
    agent = AgentNode(client=chat_client, memory=[retrieval_mem])