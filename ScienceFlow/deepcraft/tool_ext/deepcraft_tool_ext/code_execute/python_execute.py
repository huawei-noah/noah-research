# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

import multiprocessing
import sys
import contextlib
import io
import logging
from typing import Dict
import logging
import threading
from deepcraft_core.tool import BaseTool

import ast, re

PYTHON_EXECUTE_DESCRIPTION = """
Executes Python code string. Note: Only print outputs are visible, function return values are not captured. 
Use print statements to see results.
"""

logger = logging.getLogger(__name__)
from datetime import datetime

MAX_RETRIES = 2

class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions.
    Attributes:
        name (str): The name of the tool, set to "python_execute".
        description (str): A description of the tool, taken from PYTHON_EXECUTE_DESCRIPTION.
        parameters (dict): A dictionary defining the parameters required for code execution.
    """

    name: str = "python_execute"
    description: str = PYTHON_EXECUTE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }

    async def execute(
        self,
        code: str,
        exec_env,
        tools_context,
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.
            **context: Additional context variables to be passed to the code execution environment.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """
        logger.info(f"execute: \n{code}")
        result_dict = {"observation": "", "success": False, "code": code, "prev_keys": set(tools_context.keys())}

        with exec_env as f:
            obs, error = f(code)
        obs = re.sub(r"Out\[\d+\]:", "", obs)

        try:
            result = ast.literal_eval(obs.strip())
        except Exception as e:
            result = obs

        if not error.strip():
            result_dict["observation"] = result
            result_dict["success"] = True
            result_dict["new_vars"] = {
                    k: v for k, v in tools_context.items()
                    if k not in result_dict["prev_keys"] and not k.startswith("__")
                }
        else:
            result_dict["observation"] = str(error)
            result_dict["success"] = False
      

        return result_dict

if __name__ == "__main__":
    import asyncio
    Model = PythonExecute()
    code  = """
x = 10
y = 20
print(f'Sum: {x + y}')
    """

    rst   = asyncio.run(Model.execute(code, timeout=10))
    print(rst)