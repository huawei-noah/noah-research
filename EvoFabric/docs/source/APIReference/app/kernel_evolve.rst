.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

evofabric.app.kernel_evolve
==================================


.. py:module:: evofabric.app.kernel_evolve


.. py:class:: BaseEvaluator

    内核演化过程的抽象评估器

    .. py:method:: evaluate(initial_code: str, evolve_code: str) -> Metrics

        执行生成代码的评估

        :param initial_code: 内核的初始代码
        :type initial_code: str
        :param evolve_code: 内核的演化代码
        :type evolve_code: str
        :return: 生成内核代码的评估指标
        :rtype: Metrics
        :raises: NotImplementedError


.. py:class:: GPUEvaluator

    基于GPU设备实现的内核演化评估器

    .. py:method:: evaluate(initial_code: str, evolve_code: str) -> Metrics

        执行生成代码的评估

        :param initial_code: 内核的初始代码
        :type initial_code: str
        :param evolve_code: 内核的演化代码
        :type evolve_code: str
        :return: 生成内核代码的评估指标
        :rtype: Metrics


.. py:class:: Metrics(speedup: float, original_time: float, optimized_time: float, error: Optional[str], traceback: Optional[str])

    内核演化的指标类

    :param speedup: 内核重写后的加速比
    :type speedup: float
    :param original_time: 初始代码的执行时间
    :type original_time: float
    :param optimized_time: 优化后生成代码的执行时间
    :type optimized_time: float
    :param error: 评估过程中产生的错误
    :type error: str
    :param traceback: 评估过程的错误追踪信息
    :type traceback: str


.. py:class:: LLMConfig(model_class: str, model_name: str, api_key: str, base_url:str, **kwargs)

    内核演化的LLM配置类

    :param model_class: 模型类名
    :type model_class: str
    :param model_name: 模型名称
    :type model_name: str
    :param api_key: 模型的API密钥
    :type api_key: str
    :param base_url: 模型的基础URL
    :type base_url: str
    :param \**kwargs: 任意关键字参数

    **示例用法：**

    .. code-block:: python

        from evofabric.app.kernel_evolve import LLMConfig

        LLMConfig(
            model_class="OpenAIChatClient",
            model_name='your-model-name',
            api_key="xxxx",
            base_url="xxxx",
        )


.. py:class:: KernelEvolve(initial_code: str, llm_config: LLMConfig, evaluator: BaseEvaluator)

    内核演化控制器，用于启动内核演化

    :param initial_code: 待演化的操作代码
    :type initial_code: str
    :param llm_config: 内核演化的LLM配置类
    :type llm_config: LLMConfig
    :param evaluator: 用于生成内核代码的评估器实现
    :type evaluator: BaseEvaluator

    .. py:method:: evolve()

        执行自我演化并返回生成的内核代码

        :return:
            - flag (boolean): 执行成功的标志
            - result (str): 生成的内核代码或错误信息
        :rtype: tuple

    **示例用法：**

    .. code-block:: python

        from evofabric.app.kernel_evolve import BaseEvaluator, LLMConfig
        from evofabric.app.kernel_evolve.core.controller import KernelEvolve
        original_code = '''
                import torch
                import torch.nn as nn

                class Model(nn.Module):
                    """
                    calculate C = diag(A) * B + D
                    A: (N,)
                    B: (N, M)
                    D: (N, M)
                    C: (N, M)
                    """

                    def __init__(self, BLOCK_M=128):
                        super(Model, self).__init__()
                        self.BLOCK_M = BLOCK_M

                    def forward(self, A, B, D):
                        return torch.diag(A) @ B + D


                def get_inputs():
                    N, M = 4096, 4096
                    A = torch.randn(N, dtype=torch.float32)
                    B = torch.randn(N, M, dtype=torch.float32)
                    D = torch.randn(N, M, dtype=torch.float32)
                    return [A, B, D]


                def get_init_inputs():
                    return []
                '''
        config = LLMConfig(
            model_class="PanguClient",
            model_name='Pangu_38b',
            api_key="xxxx",
            base_url="xxxx",
            default_headers={"csb-token": "xxxx"}
        )
        evaluator = GPUKernelEvaluator()
        kernel_evolve = KernelEvolve(
            initial_code=original_code,
            llm_config=config,
            evaluator=evaluator)
        success_flag, result = kernel_evolve.evolve()
