.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

===================
KernelEvolve
===================

KernelEvolve是一个基于graph agent实现的自进化Kernel改写应用API工具，用于将Pytorch的kernel代码通过自进化模式改写为Triton实现，
模块已经集成了GPU的评估器。

1. 条件与限制
===================

- 设置的模型需要支持Function calling，否则无法使用此API
- 进化结果与模型能力相关，如果无法生成，可以多次尝试重新生成
- Kernel格式规范必须符合以下格式，必须要以Model为开头，kernel部分放到forward中实现，此外还必须包含get_inputs与get_init_inputs初始化与测试参数，否则可能无法正常评估生成的kernel：

.. code-block:: python

    import torch
    import torch.nn as nn

    class Model(nn.Module):
        """
        Compute C = diag(A) * B + D
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

2. 使用说明
===================

使用kernel evolve完成kernel进化重写只需要四步骤即可：
1. 请先准备要改写的kernel代码，例如：

.. code-block:: python

    torch.diag(A) @ B + D

因此，请改写成Model的格式，并且写好初始化和测试参数的获取函数：

.. code-block:: python

    import torch
    import torch.nn as nn

    class Model(nn.Module):
        """
        Compute C = diag(A) * B + D
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

2. 准备好代码后，请引入依赖 :py:class:`~evofabric.app.kernel_evolve.LLMConfig` ，配置好模型参数

.. code-block:: python

    LLMConfig(
        model_class="PanguClient",
        model_name='Pangu_38b_5.0.3.1',
        api_key="xxxx",
        base_url="xxxx",
        default_headers={"csb-token": "xxxx"}
    )

3. 你可以直接使用evofabric.app.kernel_evolve.GPUEvaluator作为评估器进行GPU算子的进化评估，此外，也可以根据需求继承 :py:class:`~evofabric.app.kernel_evolve.BaseEvaluator` 实现评估的方法作为进化的参考：

.. code-block:: python

    class GPUKernelEvaluator(BaseEvaluator):
        def evaluate(self, initial_code, evolve_code) -> Metrics:
            logger.info(f"Initial code: {initial_code}")
            logger.info(f"Evolve code: {evolve_code}")
            metrics = {
                "speedup": 1.5,
                "original_time": 380,
                "optimized_time": 190,
            }
            return Metrics(**metrics)


4. 将刚刚的配置作为进化器的构造函数，初始化 :py:class:`~evofabric.app.kernel_evolve.KernelEvolve` ，启动进化评估，获取重写结果：

.. code-block:: python

    kernel_evolve = KernelEvolve(
        initial_code=original_code,
        llm_config=config,
        evaluator=evaluator)
    success_flag, result = kernel_evolve.evolve()


- 如果成功改写，则返回success_flag为True；
- 如果失败改写，则会返回错误信息。