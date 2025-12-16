.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

CodeSandbox
===================



概述
------------------

:py:class:`~evofabric.core.tool.CodeSandbox` 基于docker SDK构建，用于帮助用户在安全环境下执行代码片段。目前支持执行Python代码片段和 bash 命令。


先决条件
------------------

使用沙箱功能需要安装docker。推荐安装docker>=7.1.0。

.. code-block:: python

    pip install docker

配置沙箱环境
--------------------------

.. code-block:: python

    from evofabric.core.typing import CodeExecDockerConfig
    from evofabric.core.tool import CodeSandbox

    # Base configuration for the sandbox environment; unspecified attributes use default values
    config = CodeExecDockerConfig(
        name="cmd_sandbox"
    )

    # Initialize CodeSandbox
    sandbox = CodeSandbox(config=config)

    # Create and start the sandbox
    sandbox.start()

    # Command snippet to run inside the sandbox (you can install environment dependencies via pip; if there are network issues, you can use domestic mirrors)
    cmd = """
    pip3 install openpyxl
    """

    # Run the command snippet inside the sandbox
    result1 = sandbox.run_cmd(cmd=cmd)
    print("cmd result: ", result1)


在沙箱中执行Python代码
-----------------------------

.. code-block:: python

    from evofabric.core.typing import CodeExecDockerConfig
    from evofabric.core.tool import CodeSandbox

    # Base configuration for the sandbox environment
    config = CodeExecDockerConfig(
        image="python:3-slim",
        auto_remove=True,
        working_dir="/tmp",
        tty=True,
        detach=True,
        mem_limit="4096m",
        cpu_quota=50000,
        entrypoint="/bin/sh",
        name="python_sandbox",
        network="host",
        volumes={"/mnt/temp_test": {"bind": "/tmp", "mode": "rw"}} # mount /mnt/temp_test on the host to /tmp in the container, with mode set to 'rw' (read and write).
    )

    # Initialize CodeSandbox
    sandbox = CodeSandbox(config=config)

    # Create and start the sandbox
    sandbox.start()

    # Code snippet to run inside the sandbox
    code1="""
    def fib(n):
        if n <= 2:
            return 1
        else:
            return fib(n-1) + fib(n-2)

    res = fib(10)
    print("10th fib number: ", res)
    """
    # Run code inside the sandbox
    result1 = sandbox.run_python(code1)
    print("python result: ", result1)

    # Stop the sandbox. If auto_remove=True in config, the sandbox will be automatically removed after stopping.
    sandbox.stop()


