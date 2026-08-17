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

import re
from deepcraft_core import ToolCollection
import sys

BACKTICK_PATTERN = r"(?:^|\n)```(.*?)(?:```(?:\n|$))"


def extract_and_combine_codeblocks(text: str) -> str:
    """
    Extracts all codeblocks from a text string and combines them into a single code string.

    Args:
        text: A string containing zero or more codeblocks, where each codeblock is
            surrounded by triple backticks (```).

    Returns:
        str: A string containing the combined code from all codeblocks, with each codeblock
        separated by a newline.

    Example:
        text = '''Here's some code:

        ```python
        print('hello')
        ```
        And more:

        ```
        print('world')
        ```'''

        result = extract_and_combine_codeblocks(text)

        Result:

        print('hello')

        print('world')
    """
    # Find all code blocks in the text using regex
    # Pattern matches anything between triple backticks, with or without a language identifier
   
    code_blocks = re.findall(BACKTICK_PATTERN, text, re.DOTALL)
   
    if not code_blocks:
        return text

    # Process each codeblock
    processed_blocks = []
    for block in code_blocks:
        # Strip leading and trailing whitespace
        block = block.strip()

        # If the first line looks like a language identifier, remove it
        lines = block.split("\n")
        if lines and (not lines[0].strip() or " " not in lines[0].strip()):
            # First line is empty or likely a language identifier (no spaces)
            block = "\n".join(lines[1:])

        processed_blocks.append(block)

    # Combine all codeblocks with newlines between them
    combined_code = "\n\n".join(processed_blocks)
    return combined_code


def split_string(code_str: str):
    """
    Parsing function call strings to extract function names and parameter values

    Args:
        code_str: A string containing the function call.

    Returns:
        A dictionary containing the function name and a list of argument values.
    """
    # Extract function name
    func_name_match = re.search(r'(\w+)\s*\(', code_str)
    if not func_name_match:
        raise ValueError("No valid function call found")
    function_name = func_name_match.group(1)

    # Extract parameter string inside parentheses
    param_match = re.search(r'\((.*?)\)', code_str)
    param_str = param_match.group(1).strip() if param_match else ''

    # Parse parameters
    args = []
    if param_str:
        # Handle string parameters with quotes (allowing commas within strings)
        param_pattern = r'''
            (?:                   # Non-capturing group: match string or non-string parameters
              "((?:\\.|[^"\\])*)"  # Double-quoted string, allowing escape characters
              |                   # or
              '((?:\\.|[^'\\])*)'  # Single-quoted string, allowing escape characters
              |                   # or
              ([^,"'\s]+)         # Non-string parameters (numbers, identifiers, etc.)
            )
            \s*                   # Ignore whitespace between parameters
            (?:,|$)               # End with comma or end of line
        '''
        param_regex = re.compile(param_pattern, re.VERBOSE)

        for match in param_regex.finditer(param_str):
            # Try to match string parameters
            string_value = match.group(1) or match.group(2)
            if string_value is not None:
                # Handle escape characters
                string_value = string_value.replace(r'\"', '"').replace(r"\'", "'").replace(r'\\', '\\')
                args.append(string_value)
                continue

            # Try to match numeric parameters
            numeric_value = match.group(3)
            if numeric_value is not None:
                # Try to convert to integer or float
                try:
                    num = float(numeric_value)
                    args.append(int(num) if num.is_integer() else num)
                except ValueError:
                    # Not a number, skip parameter name, and keep only the value
                    value_part = numeric_value.split('=')[-1].strip()
                    try:
                        num = float(value_part)
                        args.append(int(num) if num.is_integer() else num)
                    except ValueError:
                        # Still can't convert to a number, treat as a regular string
                        args.append(value_part)

    return function_name, args


def response_to_function_calling_dict(
        codeblocks: str,
        availableTools: ToolCollection,
) -> dict:
    """
    Convert a string of code blocks into a dictionary mapping tool names to function calls.
    Args:
        codeblocks (str): A string containing code blocks with function calls.
        availableTools (ToolCollection): The available tools

    Returns:
        dict: A dictionary mapping tool names to function calls.


    """
    func_list = codeblocks.split("\n")
   
    toolCalls = []
    for func in func_list:
        print(func)
        try:
            func_name, params_list = split_string(func)

        except Exception as e:
            print(f'[WARNING] {repr(e)}')
            continue

        tool = availableTools.tool_map.get(func_name)

        if tool is None:
            continue

        param_dict = {}
        if 'properties' in  tool.parameters.keys():
            for idx, param_name in enumerate(tool.parameters['properties'].keys()):
                param_dict[param_name] = params_list[idx]

        toolCalls.append({
            'name':func_name,
            'arguments':param_dict,
        })

    return toolCalls