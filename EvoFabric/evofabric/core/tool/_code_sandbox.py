# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import uuid
from typing import Any

import docker
from pydantic import Field, PrivateAttr

from ..factory import BaseComponent
from ..typing import CodeExecDockerConfig


class CodeSandbox(BaseComponent):
    config: CodeExecDockerConfig = Field(
        description="Docker configs"
    )

    _container: Any = PrivateAttr(default=None)

    def start(self):
        client = docker.from_env()
        self._container = client.containers.create(
            **self.config.model_dump(),
        )
        self._container.start()

    def run_python(self, code: str):
        if not self._container:
            raise ValueError("code sandbox have not started yet!")

        try:
            if self._container.status != "created":
                raise ValueError("code sandbox have not created yet!")

            filename = f"script_{uuid.uuid4().hex}.py"
            exec_script = f"""
#!/usr/bin/env python3
{code}
"""
            result = self._container.exec_run(
                cmd=["/bin/sh", "-c", f"echo '{exec_script}' > /tmp/{filename} && python3 /tmp/{filename}"])
            return result
        except Exception as e:
            raise ValueError("code exec falied")

    def run_cmd(self, cmd: str = None):
        if not self._container:
            raise ValueError("code sandbox have not started yet!")

        try:
            if self._container.status != "created":
                raise ValueError("code sandbox have not created yet!")

            result = self._container.exec_run(cmd=["/bin/sh", "-c", cmd])

            return result
        except Exception as e:
            raise ValueError("code exec failed")

    def stop(self):
        self._container.stop()
