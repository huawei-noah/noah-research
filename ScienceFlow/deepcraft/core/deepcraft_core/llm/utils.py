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

import json
import re
import asyncio
from typing import Dict, Tuple, Optional, Union
from lazy_loader import load as lazy_load

ASK_TOOL_PROMPT = """
Available Tools:
{tool_descriptions}

Determine if a tool is needed to answer the user's question.
If yes, generate a tool call instruction in the following format:
<|FunctionCallBegin|>
{{
"name": "tool_name",
"arguments": {{
"argument1": "value1",
"argument2": "value2"
}}
}}
<|FunctionCallEnd|>
If no tool is needed, directly answer the user's question.
"""


class FunctionCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = str(arguments)  # arguments is string
        self.arguments = self.arguments.replace("\'", "\"")


class ToolCall:
    def __init__(self, id, type, function):
        self.id = id
        self.type = type
        self.function = FunctionCall(**function)  # function is dict


class ResponseCall:
    def __init__(self, role, content, tool_calls):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls  # assume function is a dict


def response2content_and_tool_calls(
        response: str,
        begin_marker: str = "<|FunctionCallBegin|>",
        end_marker: str = "<|FunctionCallEnd|>",
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Translate the response of the model to the content and tool calls format
    Args:
        response: A string containing the function call and content.
        begin_marker: The marker indicating the beginning of the function call.
        end_marker: The marker indicating the end of the function call.

    Returns:
        thought: A string containing the content.
        function_call_matches: A list of function calls.
    """
    pattern = re.escape(begin_marker) + r'\s*(.*?)\s*' + re.escape(end_marker)
    function_call_matches = list(re.finditer(pattern, response, re.DOTALL))

    if not function_call_matches:
        return response.strip(), None

    thought_parts = []
    last_end = 0

    for match in function_call_matches:
        start, end = match.span()
        thought_parts.append(response[last_end:start].strip())
        last_end = end

    thought_parts.append(response[last_end:].strip())
    thought = '\n'.join([part for part in thought_parts if part])

    return thought, function_call_matches


def response2tool_call(
        response,
        parallel_tool_calls: bool = False
) -> ResponseCall:
    """
    Translate the response of the model to the tool call format
    Args:
        response: A string containing the function call.
        parallel_tool_calls: Whether to allow multiple tool calls.

    Returns:
        ResponseCall: A class containing the function call.

    """
    content, function_call_matches = response2content_and_tool_calls(response)

    function_calls = []
    for match in function_call_matches:
        try:
            function_call_str = match.group(1).strip()
            function_call = json.loads(function_call_str)

            if isinstance(function_call, list):
                for call in function_call:
                    if "name" in call and "arguments" in call:
                        function_calls.append(call)
            elif isinstance(function_call, dict) and "name" in function_call:
                if "arguments" not in function_call:
                    if "parameters" in function_call:
                        function_call["arguments"] = function_call["parameters"]
                    else:
                        function_call["arguments"] = {}

        except json.JSONDecodeError:
            function_call = {
                "raw": match.group(1).strip(),
                "error": "JSON ERROR"
            }

        tmp_ToolCall = ToolCall(
            id=r"call_{idx}",
            type="function",
            function=function_call
        )
        function_calls.append(tmp_ToolCall)

    if not parallel_tool_calls and function_calls:
        function_calls = [function_calls[0]]

    res = ResponseCall(
        role="assistant",
        content=content,
        tool_calls=function_calls,
    )

    return res


def openapi2pangu(tools_list):
    """
    Translate the paradigm of OpenAI tool definition to the tool paradigm of pangu equipment
    Args:
        list: A list containing the tools' parameters.

    Returns:
        str: A string containing the tools' information.
    """
    ans = ""
    for tool_info in tools_list:
        tmp = tool_info['function']
        tmp["principle"] = tool_info['function']['description']

        ans += "\n" + json.dumps(tmp, ensure_ascii=False)

    return ans

# Note:
def lazy_import(module_path, attr_name=None):
    if attr_name is None:
        return lazy_load

    def wrapper(*args, **kwargs):
        module = lazy_load(module_path)
        attr = getattr(module, attr_name)
        return attr(*args, **kwargs)
    return wrapper


def count_chars(text: str) -> dict:
    """
    Count the number of Chinese and English characters in a string.

    Args:
        text (str): The input string.

    Returns:
        dict: A dictionary with counts for 'chinese', 'english' & 'symbol' characters.
    """
    chinese_count = 0
    symbol_count = 0
    english_count = 0

    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1
        elif ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
            english_count += 1
        else:
            symbol_count += 1
    return {'chinese': chinese_count, 'english': english_count, 'symbol': symbol_count}


def get_text_from_content(content: Union[str, list]) -> str:
    """
    Extract plain text from message content for token estimation.
    Handles OpenAI-style content: str or list of dicts (e.g. {"type": "text", "text": "..."}).

    Args:
        content: Message content (string or list of content parts).

    Returns:
        Concatenated text string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


def estimate_tokens(
    text: str,
    model: Optional[str] = None,
    encoding_name: Optional[str] = None,
    use_tiktoken: bool = True,
    fallback_chinese_ratio: float = 0.6,
    fallback_english_ratio: float = 0.25,
    fallback_symbol_ratio: float = 0.33,
) -> int:
    """
    Estimate token count for a string. Prefers tiktoken when available and model/encoding
    is set (accurate for OpenAI-compatible APIs); otherwise uses an improved heuristic.

    Why heuristic is inaccurate:
    - Different tokenizers (OpenAI/Claude/Gemini/DeepSeek) tokenize differently.
    - English is tokenized by subwords, not by character (e.g. "tokenization" can be 2–3 tokens).
    - Symbols/numbers often merge with adjacent tokens; 1 token per symbol overcounts.

    Args:
        text: Input string to count.
        model: Model name for tiktoken (e.g. "gpt-4", "gpt-3.5-turbo"). Used when encoding_name is None.
        encoding_name: tiktoken encoding name (e.g. "cl100k_base"). Overrides model when set.
        use_tiktoken: If True, use tiktoken when available; else always use heuristic.
        fallback_chinese_ratio: Tokens per Chinese character in fallback (default from DeepSeek doc).
        fallback_english_ratio: Tokens per English character in fallback (~4 chars/token).
        fallback_symbol_ratio: Tokens per symbol in fallback (symbols often merge).

    Returns:
        Estimated token count (int).
    """
    if use_tiktoken and (model or encoding_name):
        try:
            import tiktoken
            if encoding_name:
                enc = tiktoken.get_encoding(encoding_name)
            elif model:
                enc = tiktoken.encoding_for_model(model)
            else:
                enc = None
            if enc is not None:
                return len(enc.encode(text))
        except Exception:
            pass  # fall through to heuristic

    num = count_chars(text)
    return int(
        num["chinese"] * fallback_chinese_ratio
        + num["english"] * fallback_english_ratio
        + num["symbol"] * fallback_symbol_ratio
    )


class AsyncLineIterator:
    def __init__(self, response):
        """
        AsyncLineIterator is an asynchronous iterator that iterates over the lines of a response.
        Args:
            response: The response to iterate over.

        """
        self.lines = response.iter_lines()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            line = next(self.lines).decode("utf-8")
            await asyncio.sleep(0.001)  # Asynchronous operations
            return line
        except StopIteration:
            raise StopAsyncIteration
