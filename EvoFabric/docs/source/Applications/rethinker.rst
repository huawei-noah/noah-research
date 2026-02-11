.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

ReThinker-Agent-Framework
========================================================================================

`论文：arXiv <https://arxiv.org/abs/2602.04496>`_


简介
--------

**ReThinker** 是一个推理框架，旨在通过 **引导式反思** 与 **置信度控制**，
使大型语言模型能够对其中间结论进行 *重新思考*，
从而显著提升其 **科学推理能力**。

不同于一次性完成的 chain-of-thought 推理方式，
ReThinker 引入了一种迭代式机制，用于：

    - 检测低置信度的推理步骤
    - 主动回溯并修正这些步骤
    - 生成更加稳健、可靠的最终答案

该方法在复杂科学问题以及多步推理任务中，
能够显著提升推理准确率与结果稳定性。

论文地址：
https://arxiv.org/abs/2602.04496


核心贡献
-----------------

**具备重思能力的智能体框架**

ReThinker 将“重新思考”机制整合进 **基于智能体的工作流** 中，
使系统能够进行迭代式决策与持续优化。
这种设计带来了 **稳定且一致的解质量提升**，
而非仅依赖随机反思所带来的偶发性收益。

**引导式反思机制**

与让 critic 从简化后的轨迹中自行推断问题不同，
ReThinker 在轨迹总结阶段 **显式提出具体的改进点**，
并将这些信息传递给受约束的 critic 角色进行反思与修正。
这种引导方式确保了重新思考过程的聚焦性、高效性与稳定性。

**置信度控制策略**

在候选选择阶段，ReThinker 采用迭代式选择策略，
每一轮不仅基于 **之前已选候选项**，
还显式结合这些候选项的 **置信度评分**。
这种置信度反馈机制引导模型在多轮重思过程中
做出更加可靠、稳定的选择决策。


实验结果
--------------------
.. list-table:: Benchmark results
   :header-rows: 1
   :widths: 22 58 8 8 10

   * - Category
     - Model / Framework
     - HLE
     - GAIA
     - XBench
   * - Foundation Model w. tools
     - Kimi K2 (Kimi et al., 2025)
     - 18.1
     - 57.7
     - 50.0
   * - Foundation Model w. tools
     - Claude-4.5-Sonnet (Anthropic, 2025)
     - 24.5
     - 71.2
     - 66.0
   * - Foundation Model w. tools
     - DeepSeek-V3.2 (Liu et al., 2025a)
     - 27.2
     - 63.5
     - 71.0
   * - Foundation Model w. tools
     - GLM-4.6 (Zhipu, 2025)
     - 30.4
     - 71.9
     - 70.0
   * - Foundation Model w. tools
     - GPT-5-high (OpenAI, 2025b)
     - 35.2
     - 76.4
     - 77.8
   * - Foundation Model w. tools
     - Gemini-3-Pro (Google, 2025)
     - 38.3
     - 79.0
     - 87.0
   * - Inference Framework
     - WebExplorer (Liu et al., 2025b)
     - 17.3
     - 50.0
     - 53.7
   * - Inference Framework
     - OpenAI DeepResearch (OpenAI, 2025a)
     - 26.6
     - 67.4
     - -
   * - Inference Framework
     - Kimi Researcher (Kimi, 2025)
     - 26.9
     - -
     - 69.0
   * - Inference Framework
     - Tongyi DeepResearch (30BA3B) (Tongyi et al., 2025)
     - 32.9
     - 70.9
     - 75.0
   * - Inference Framework
     - MiroThinker-v1.0 (30B) (MiroMind et al., 2025)
     - 33.4
     - 73.5
     - 70.6
   * - Inference Framework
     - ReThinker (OpenPangu-72B) (Ours)
     - 33.1
     - 72.8
     - 78.0
   * - Inference Framework
     - ReThinker (Gemini-3-pro) (Ours)
     - 52.2
     - 81.6
     - 90.0

使用方式
-----------

1. 依赖安装
~~~~~~~~~~~~~~~~~~~~~~~

首先安装所需依赖。除基础包外，还需要 *rethinker* 模块的额外依赖::

    pip install evofabric[rethinker]


2. 配置
~~~~~~~~~~~~~~~~~~~~~~~

配置项目文件::

    configs/config.yaml

请确保正确填写以下字段：


.. code-block:: yaml

    llm_resources:
      llm_config_name:
        model_name: your-model-name  # model name
        api_key: your-api-key        # openai api key
        base_url: your-api-base-url  # openai base url
        top_p: 1.0
        temperature: 1.0
        fast_think: false
        stop_condition: '<code[^>]*>((?:(?!<code).)*?)</code>'
        http_client_kwargs:
          verify: false


    web_parser:
      model: pangu_web_parser  # must exist a corresponding llm config in llm_resources
      use_jina: true
      jina_api_key: your-jina-api-key  # jina api key

    web_search:
      serper_api_key: your-serper-api-key  # serper api key

3. 运行Rethinker
~~~~~~~~~~~~~~~~

运行主程序::

    python run.py --config configs/config.yaml


4. 输出结构
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    如果你计划运行评测脚本，此步骤是必须的。

当 ``config.exp.output_root`` 被设置时，输出文件结构如下::

    output_root/
        qid00001/
            node1.json
            node2.json
            ...
            result.json
        qid00002/
        ...

每个 ``qidXXXXX`` 目录对应一次独立查询或实验，
其中包含中间节点结果以及最终汇总结果。


5. 测试
~~~~~~~~~~~~~~~~~

运行评测脚本::

    python evaluation.py \
        --api-key=your-api-key \
        --model-name=your-model-name \
        --base-url=your-base-url \
        --save-result=eval.json \
        --benchmark=hle

该命令将运行指定基准测试，
并将评测结果保存至 ``eval.json``。


致谢
---------------

本项目部分工作受益于
`Eigen-1 <https://github.com/tangxiangru/Eigen-1/tree/main>`_。


引用
--------

::

    @article{tang2026rethinker,
        author       = {Zhentao Tang and Yuqi Cui and Shixiong Kai and Wenqian Zhao
                        and Ke Ye and Xing Li and Anxin Tian and Zehua Pei
                        and Hui‐Ling Zhen and Shoubo Hu and Xiaoguang Li
                        and Yunhe Wang and Mingxuan Yuan},
        title        = {ReThinker: Scientific Reasoning by Rethinking with Guided
                        Reflection and Confidence Control},
        year         = {2026},
        url          = {https://arxiv.org/abs/2602.04496}
    }

