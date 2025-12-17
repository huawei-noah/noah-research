.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

Logger
================

.. py:module:: evofabric.logger

get_logger
~~~~~~~~~~~~~

.. py:function:: get_logger()

   获取全局日志记录器。如果尚未初始化，则使用 ``loguru.logger`` 。

   :returns: 全局日志记录器对象
   :rtype: logger


set_logger
~~~~~~~~~~~~~~~~~~

.. py:function:: set_logger(logger)

   设置全局日志记录器。

   :param logger: 要设置为全局使用的日志记录器
   :type logger: logger