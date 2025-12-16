.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

图引擎
===================

本指南提供了 EvoFabric 中图引擎模块的全面文档。

一个完整的图引擎主要由以下三部分组成：

* 状态声明，声明图中传递的状态变量类型和更新机制。

* 节点，是图中处理任务的最小单元，节点的输入是状态，输出是状态的增量。

* 边，边帮助节点将状态传递给下一个节点，支持条件判断、状态过滤、分组等特性。

接下来，会分别介绍状态声明、添加节点、添加边、可视化等功能特性和用法。

.. toctree::
   :maxdepth: 1

   state_schema
   node
   edge
   build_graph
   state_while_running
   ctx_and_streaming
   examples