# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import fnmatch
from typing import Any, Dict, List, Literal, Union

from pydantic import Field, TypeAdapter

from ..factory import BaseComponent
from ..typing import ActiveToolPattern, DeactivateToolPattern, ToolControlPattern


class ToolController(BaseComponent):
    default_mode: Literal['activate', 'deactivate'] = "activate"
    """ 
    Default tool behavior when no rules match.
    Can be either:
    - "activate": Automatically activate tools without matching rules (default)
    - "deactivate": Automatically deactivate tools without matching rules

    Example settings:
    default_mode = "activate"  # All tools active by default
    default_mode = "deactivate"  # All tools inactive by default
    """

    rules: list[Union[ToolControlPattern, dict]] = Field(default_factory=dict)
    """ 
    List of rules for controlling tool activation/deactivation. 
    Rules are applied in order, with the first matching rule determining the tool's status.

    Each rule can be either:
    1. A `ToolControlPattern` instance with attributes:
       - `mode`: 'activate' or 'deactivate'
       - `pattern`: glob wildcard string
    2. A dictionary with keys:
       - 'mode': 'activate' or 'deactivate'
       - 'pattern': glob wildcard string

    .. note::
        When using with McpToolManager:
        - Actual tool names follow format [server_name]_[tool_name]
        - Wildcard patterns must prefix with server_name
        - Example: To match all tools from server "math", use pattern "math_*"
                  (matches "math_calculator", "math_grapher", etc.)
    
        Pattern examples:
        math_*       → Matches all tools from server "math"  
                      (e.g., "math_calculator", "math_grapher")
        text_*       → Matches all tools from server "text"
        math_calculator → Matches only this specific tool
        *_calculator   → Matches any tool ending with "*_calculator" from any server
    """

    def model_post_init(self, context: Any, /):
        """Convert dictionary rules to Pattern objects after initialization"""
        converted_rules = []
        for rule in self.rules:
            if isinstance(rule, dict):
                # Convert dict to appropriate Pattern class based on mode
                rule = TypeAdapter(ToolControlPattern).validate_python(rule)

            converted_rules.append(rule)
        self.rules = converted_rules

    def check_tool_status(self, tool_name: str) -> bool:
        """Check if a tool is active based on applied rules

        Args:
            tool_name: Name of the tool to check

        Returns:
            bool: True if tool is active, False if deactivated
        """
        for rule in self.rules:
            if fnmatch.fnmatch(tool_name, rule.pattern):
                return rule.mode == 'activate'

        # Return default_mode if no rules matched
        return self.default_mode == 'activate'

    def filter_tool_list(self, tool_list: List[Dict]) -> List[Dict]:
        """Filter tool list to only include active tools

        Args:
            tool_list: List of tool dictionaries

        Returns:
            List[Dict]: List of active tools
        """
        return [
            tool for tool in tool_list
            if self.check_tool_status(tool['name'])
        ]

    def activate_tool(self, tool_name: str):
        """Activate a specific tool with highest priority
        Removes any existing rule for this exact tool name

        Args:
            tool_name: Exact name of the tool to activate
        """
        # Remove existing rule for this exact tool name
        self.rules = [r for r in self.rules if r.pattern != tool_name]
        # Add new activation rule at the beginning (highest priority)
        self.rules.insert(0, ActiveToolPattern(pattern=tool_name))

    def deactivate_tool(self, tool_name: str):
        """Deactivate a specific tool with highest priority
        Removes any existing rule for this exact tool name

        Args:
            tool_name: Exact name of the tool to deactivate
        """
        # Remove existing rule for this exact tool name
        self.rules = [r for r in self.rules if r.pattern != tool_name]
        # Add new deactivation rule at the beginning (highest priority)
        self.rules.insert(0, DeactivateToolPattern(pattern=tool_name))
