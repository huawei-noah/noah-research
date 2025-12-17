# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import copy
import fnmatch
import functools
import importlib.util
import inspect
import os
import types
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Annotated, Any, AsyncGenerator, Callable, get_args, get_origin, List, Optional

import mcp
from docstring_parser import Docstring, parse
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from pydantic import ConfigDict, create_model, Field

from ..typing import McpServerLink, SseLink, StdioLink, StreamableHttpLink


def load_function(
        file_path: str,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None
):
    """
    Import all functions from a given Python file and return them as a dictionary.

    This function dynamically loads a Python file as a module, then extracts all module-level
    functions (excluding classes and methods) that pass the include/exclude pattern filters.

    Args:
        file_path (str):
            Absolute or relative path to the target Python file. Must be a valid .py file.
        include_patterns (List[str], optional):
            List of Unix shell-style wildcards (e.g., 'test_*') that function names must match ALL patterns.
            If None, no include filtering is applied. Defaults to None.
        exclude_patterns (List[str], optional):
            List of Unix shell-style wildcards (e.g., '_*') that function names must NOT match ANY pattern.
            If None, no exclude filtering is applied. Defaults to None.

    Returns:
        dict:
            Dictionary mapping function names (str) to their callable objects. Only includes:
            - Module-level functions (not class methods)
            - Functions passing the include_patterns (if specified)
            - Functions NOT excluded by exclude_patterns (if specified)

    Raises:
        ImportError:
            If the file cannot be loaded as a Python module (invalid syntax, missing path, etc.)
        FileNotFoundError:
            If file_path points to a non-existent file (raised by underlying filesystem calls)
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"{file_path}")

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    functions = {}
    for name, obj in module.__dict__.items():
        if (include_patterns is not None) and (not all(fnmatch.fnmatch(name, pattern) for pattern in include_patterns)):
            continue

        if (exclude_patterns is not None) and (any(fnmatch.fnmatch(name, pattern) for pattern in exclude_patterns)):
            continue

        if callable(obj) and isinstance(obj, types.FunctionType):
            functions[name] = obj

    return functions


def _safe_peel_partial(function):
    """Peel off partial wrapper"""
    while isinstance(function, functools.partial):
        function = function.func
    return function


def _parse_docstring(
        function: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        include_long_description: bool = True,
):
    """
    Parses the docstring of a function to extract structured documentation metadata.

    Args:
        function: The function whose docstring is to be parsed.
        name (Optional[str]): Custom name to use for the function. If None, uses `function.__name__`.
        description (Optional[str]): Custom description to override the docstring content.
                                    If None, builds description from docstring.
        include_long_description (bool): Whether to include the long description from the docstring.
                                         Defaults to True.

    Returns:
        tuple: (parsed_docstring, function_name, formatted_description)
            - parsed_docstring: Parsed docstring object from `docstring_parser.parse()`
            - function_name: Name of the function (either specified or inferred)
            - formatted_description: Formatted description string built from docstring or provided
    """
    function = _safe_peel_partial(function)
    docs = parse(function.__doc__)

    if name is None:
        name = function.__name__

    if description is None:
        description_parts = []
        if docs.short_description:
            description_parts.append(docs.short_description)
        if docs.long_description and include_long_description:
            description_parts.append(docs.long_description)
    else:
        description_parts = [description]
    description = "\n\n".join(description_parts)

    return docs, name, description


def _is_inspect_empty(sth):
    """Checks if an object matches `inspect.Parameter.empty` sentinel value."""
    return sth == inspect.Parameter.empty


def _parse_func_params(
        function: Callable,
        docstring: Docstring = None,
        include_var_positional: bool = True,
        include_var_keyword: bool = True,
        exclude_params: Optional[List[str]] = None,
):
    """
    Parses function parameters using signature inspection and docstring metadata.

    Args:
        function: Target function for parameter inspection.
        docstring (Docstring): Parsed docstring object (from `docstring_parser.parse()`).
                              If None, will parse the function's docstring.
        include_var_positional (bool): Whether to include `*args` style parameters. Defaults to True.
        include_var_keyword (bool): Whether to include `**kwargs` style parameters. Defaults to True.
        exclude_params (Optional[List[str]]): Parameter names to exclude from processing.

    Returns:
        tuple: (parameter_fields, excluded_parameters_list)
            - parameter_fields: Dict mapping parameter names to (type, Field) tuples
            - excluded_parameters_list: Names of parameters actually excluded
    """

    def get_annotation(param, default_type=Any):
        return default_type if _is_inspect_empty(param.annotation) else param.annotation

    def create_field(description, default_value):
        return Field(default=default_value, description=description)

    excluded_param_list = []
    exclude_params = copy.deepcopy(exclude_params) or []
    sig = inspect.signature(function)

    # for partial function, will exclude bound params
    while isinstance(function, functools.partial):
        arguments: inspect.BoundArguments = sig.bind_partial(*function.args, **function.keywords)
        exclude_params.extend(arguments.arguments.keys())
        function = function.func

    is_class_method = function.__qualname__ and "." in function.__qualname__
    docstring = docstring or parse(function.__doc__)
    param_desp_map: dict[str, str] = {x.arg_name: x.description for x in docstring.params}

    fields = {}
    # iteratively define param field
    for pidx, (name, param) in enumerate(sig.parameters.items()):
        description = param_desp_map.get(name, None)
        default = param.default if not _is_inspect_empty(param.default) else ...
        annotation = get_annotation(param)

        if description is None and get_origin(annotation) is Annotated:
            # update description if argument define `Annotated`
            args = get_args(annotation)
            for arg in args[1:]:
                if isinstance(arg, str):
                    description = arg
                    break

        if is_class_method and pidx == 0 and name in {"cls", "self"}:
            # handle self and cls
            continue

        if name in exclude_params:
            # handle exclude params
            excluded_param_list.append(name)
            continue

        if param.kind == param.VAR_POSITIONAL:
            if include_var_positional:
                # handle function(*args)
                fields[name] = (
                    list[annotation]
                    if not _is_inspect_empty(annotation)
                    else list,
                    create_field(description, None)
                )

        elif param.kind == param.VAR_KEYWORD:
            if include_var_keyword:
                # handle function(**kwargs)
                fields[name] = (
                    dict[str, annotation]
                    if not _is_inspect_empty(annotation)
                    else dict,
                    create_field(description, None)
                )

        else:
            # handle other params
            fields[name] = (
                annotation,
                create_field(description, default)
            )

    return fields, excluded_param_list


def parse_callable_schema(
        function: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        include_long_description: bool = True,
        include_var_positional: bool = True,
        include_var_keyword: bool = True,
        exclude_params: Optional[List[str]] = None,
):
    """
    Generates a JSON schema representation of a callable function for LLM tool usage.

    This function transforms a Python callable into a structured schema compatible with
    LLM tool/function calling formats. It parses the function's signature, docstring,
    and parameter defaults to create a complete description of the function's interface.

    Support function type:
    - python function
    - class method
    - @classmethod
    - @staticmethod
    - partial(function / class method)
    - lambda

    Args:
        function: Callable function to convert into a schema.
        name (Optional[str]): Custom function name. Defaults to function's __name__.
        description (Optional[str]): Custom description. Defaults to docstring summary.
        include_long_description (bool): Whether to include long description from docstring.
        include_var_positional (bool): Whether to include *args parameters.
        include_var_keyword (bool): Whether to include **kwargs parameters.
        exclude_params (Optional[List[str]]): Parameter names to exclude.

    Returns:
        tuple: (json_schema, excluded_parameters_list)
            - json_schema: Dictionary with OpenAI-compatible function schema
            - excluded_parameters_list: Names of parameters actually excluded
    """
    docstring, name, description = _parse_docstring(
        function,
        name,
        description,
        include_long_description
    )

    fields_info, excluded_params = _parse_func_params(
        function,
        docstring,
        include_var_positional,
        include_var_keyword,
        exclude_params,
    )

    model_name = f"pyfunc_{name}_{uuid.uuid4().hex[:4]}"
    pydantic_model = create_model(
        model_name,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **fields_info,
    )
    param_schema = pydantic_model.model_json_schema()
    for _, param in param_schema["properties"].items():
        param.pop("title", "")
    full_json_schema: dict = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": param_schema,
        },
    }
    full_json_schema["function"]["parameters"].pop("title", None)
    return full_json_schema, excluded_params


def parse_mcp_tool_function(
        mcp_tool: mcp.types.Tool,
        server_name: str = None,
        include_long_description: bool = True,
) -> dict:
    """Converts MCP tool schema to OpenAI-compatible function call format.

    This function:
    1. Parses the docstring from the tool's description to extract structured metadata
    2. Combines function description components into a coherent narrative
    3. Updates parameter schemas with human-readable descriptions from docstring
    4. Generates a function-call-ready JSON schema with proper naming conventions

    Args:
        mcp_tool:
            Input tool schema containing the core function definition
            and raw docstring (triple-quoted content).
        server_name:
            Optional server identifier to prefix function names
            (e.g., "myserver_calculate" when server_name="myserver").
        include_long_description:
            Controls whether detailed documentation sections should be
            included in the function's main description (default: True).

    Returns:
        A dictionary representing an OpenAI function call schema with:
        - Type specification: "function"
        - Function metadata including:
            * Name: Prefixed with server_name if specified
            * Parameters: JSON schema with docstring-resolved descriptions
            * Description: Consolidated short/long description

    Note:
        The function relies on the docstring containing:
          - Short description (one-line summary)
          - Long description (detailed documentation, optional)
          - Parameter documentation (arg_name:description pairs)
    """
    docstring = parse(mcp_tool.description)
    params_docstring = {
        param.arg_name: param.description for param in docstring.params
    }

    # Function description
    description_parts = []
    if docstring.short_description is not None:
        description_parts.append(docstring.short_description)

    if include_long_description and docstring.long_description is not None:
        description_parts.append(docstring.long_description)

    description = "\n\n".join(description_parts)

    params_json_schema = copy.deepcopy(mcp_tool.inputSchema['properties'])

    for name, info in params_json_schema.items():
        params_json_schema[name]['description'] = params_docstring.get(name, None)

    func_json_schema: dict = {
        "type": "function",
        "function": {
            "name": mcp_tool.name
            if server_name is None
            else server_name + "_" + mcp_tool.name,
            "parameters": params_json_schema,
            "description": description
        },
    }

    return func_json_schema


@asynccontextmanager
async def create_mcp_session(
        server_link: McpServerLink,
        sampling_callback=None,
) -> AsyncGenerator[ClientSession, None]:
    """
    Create an MCP client session for the given server parameters.

    Yields:
        An initialized ClientSession.
    """
    # Shared setup for all transports
    read_timeout = timedelta(seconds=_get_read_timeout_seconds(server_link))

    # Select and invoke the appropriate transport client
    if isinstance(server_link, StdioLink):
        client_ctx = stdio_client(server_link)
    elif isinstance(server_link, SseLink):
        client_ctx = sse_client(**server_link.model_dump(exclude={"type"}))
    elif isinstance(server_link, StreamableHttpLink):
        params = server_link.model_dump(exclude={"type"})
        params["timeout"] = timedelta(seconds=params["timeout"])
        params["sse_read_timeout"] = timedelta(
            seconds=params["sse_read_timeout"])
        client_ctx = streamablehttp_client(**params)
    else:
        raise NotImplementedError(
            f"Unsupported server params type: {type(server_link)}")

    # Enter transport context and extract streams
    async with client_ctx as streams:
        read, write = streams[0], streams[1]

        # Create and yield the ClientSession
        async with ClientSession(
                read_stream=read,
                write_stream=write,
                read_timeout_seconds=read_timeout,
                sampling_callback=sampling_callback,
        ) as session:
            yield session


def _get_read_timeout_seconds(server_link: McpServerLink, default=30.) -> float:
    """Extract the appropriate read timeout in seconds based on server params type."""
    if isinstance(server_link, StdioLink):
        return server_link.read_time_out
    elif isinstance(server_link, (SseLink, StreamableHttpLink)):
        return getattr(server_link, "sse_read_timeout", 300.0)
    else:
        return default
