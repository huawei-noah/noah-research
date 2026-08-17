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

import logging
from typing import List, Optional, Callable
from pydantic import BaseModel, Field, model_validator
import sys, re
from deepcraft_core.message import ROLE_TYPE, Role
from deepcraft_core import Message
from deepcraft_tool_ext import PythonExecute

from ..base import BaseAgent
from .codeact_prompts import CODEACT_PROMPT, TOOL_FUNC_PROMPT
from .utils import extract_and_combine_codeblocks, response_to_function_calling_dict
from deepcraft_tool_ext.code_execute.code_exec_env import PythonREPL
import json

logger = logging.getLogger(__name__)

import re, ast

class CodeActAgent(BaseAgent):
    """A specialized agent for executing code actions.
      Attributes:
        tools (Optional[List[Callable]]): A list of callable tools available to the agent.
    """
    tools: Optional[List[Callable]] = Field([], description="Agent tools")
    extra_system_prompt: Optional[str] = Field("", description="Extra system prompt")
    codeact_prompt: Optional[str] = Field(CODEACT_PROMPT, description="Codeact system prompt")


    @model_validator(mode="after")
    def initializeCodeActAgent(self) -> "CodeActAgent":
        """Initialize the CodeActAgent instance.
        Returns:
            CodeActAgent: The initialized CodeActAgent instance.
        """
        self = super().initializeAgent()

        tool_func_prompts = ""
        self.tools_context = {}

        if self.exec_mode == "multi_turn":
            if self.availableTools is not None:
                for idx, tool in enumerate(self.availableTools.tools):
                    tool_func_prompts += TOOL_FUNC_PROMPT.format(
                        tool_name=tool.name,
                        tool_description=tool.description,
                        tool_func_signature=str(tool.func_signature),
                        )
                    
                    self.tools_context[tool.name] = self.tools[idx]

        if self.systemPrompt is None:
            self.systemPrompt = self.extra_system_prompt + self.codeact_prompt.format(Tool_Func_Prompts=tool_func_prompts)

        #if isinstance(__builtins__, dict):
        #    self.safe_globals = {"__builtins__": __builtins__, **self.tools_context}
        #else:
        #    self.safe_globals = {"__builtins__": __builtins__.__dict__.copy(), **self.tools_context}


        if self.exec_mode == "multi_turn":
            self.repl = PythonREPL(self.tools_context)
        else:
            self.repl = None

        return self
    

    async def step(self) -> str:
        """Execute a single step in the agent's workflow.
        Must be implemented by subclasses to define specific behavior.
        Returns:
            str: The result of the executed code or the direct response from the language model.
        """
        response = await self.llm.ask(
            messages=self.memory.messages,
            system_msgs=[Message.system_message(self.systemPrompt)]
        )

        output = extract_and_combine_codeblocks(response)      

        if self.exec_mode == "single_turn":
            return {"observation": output, "type": "natural_language"}, ""

        try:
            ast.parse(output)
            output_type = "python_code"
        except SyntaxError:
            output_type = "natural_language"

        self.updateMemory(Role.ASSISTANT, output)     
        
        if output_type == "natural_language":
            self.updateMemory(Role.USER, output)
            return {'observation':output, 'type':output_type}, ""
           
        logger.info(f"code: {output}")

        
        python_env = PythonExecute()
        #ret = await python_env.execute(code, self.safe_globals, self.memory.messages)
        ret = await python_env.execute(output, self.repl, self.tools_context)
            


        self.updateMemory(Role.USER, json.dumps(ret['observation']))
    
        return ret, ""
      

