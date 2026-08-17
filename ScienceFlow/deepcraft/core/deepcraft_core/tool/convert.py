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

from inspect import signature, getdoc
from typing import Callable, get_type_hints
import asyncio
from functools import wraps

from .base import BaseTool

def async_wrapper(func):
    @wraps(func)
    async def wrapped(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapped

def func_to_openai_format(func: Callable):
    """Converts a function into a format compatible with OpenAI function calling specifications.
        Args:
            func (Callable): The function to convert.

        Returns:
            Dict[str, Any]: A dictionary representing the function in OpenAI format.
    """
    sig = signature(func)
    docstring = getdoc(func)
    type_hints = get_type_hints(func)

    # Build the parameter description
    params = {}
    for param_name, param in sig.parameters.items():
        param_type = type_hints[param_name].__name__
        if param.default != param.empty:
            required = False
            default_value = param.default
        else:
            required = True
            default_value = None

        params[param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}",
            "required": required
        }
        if default_value is not None:
            params[param_name]["default"] = default_value

    # Build the function description
    func_description = {
        "name": func.__name__,
        "description": docstring,
        "parameters": {
            "type": "object",
            "properties": params,
            "required": [param_name for param_name, param in sig.parameters.items() if param.default == param.empty]
        }
    }

    return func_description

def func_to_tool_instance(func: Callable):
    """
    Dynamically creates a subclass of the given base class with the specified method implementation.

    :param func: The implementation of the method as a function.
    :return: An instance of the dynamically created subclass.
    """
    func_description = func_to_openai_format(func)

    # Define the new class using type
    class_attributes = {
        "execute": staticmethod(async_wrapper(func)),
        "codeact_func": staticmethod(func),
        "__init__": lambda self, *args, **kwargs: (
            super(type(self), self).__init__(
                name=func_description["name"],
                description=func_description["description"],
                parameters=func_description["parameters"],
                func_signature=str(signature(func)),
                *args,
                **kwargs)
        )
    }
    name = func_description["name"]
    new_class = type(f"{name}", (BaseTool,), class_attributes)
    instance = new_class()

    return instance

def dict_to_tool_instance(func_dict: dict):
    """
    Dynamically creates a subclass of the given base class with the specified method's dictionary.
    :param func_dict: The dictionary containing the method's information.
    e.g.,
    func_description = {
    'name': 'add',
    'description': 'Add two numbers together.',
    'parameters':
        {
            'type': 'object',
            'properties':
                {
                    'a': {'type': 'float', 'description': 'Parameter a', 'required': True},
                    'b': {'type': 'float', 'description': 'Parameter b', 'required': True}
                },
            'required': ['a', 'b']
        },
    'func': "return a + b",
}

    :return: An instance of the dynamically created subclass.
    """
    func = "return"
    if "func" in func_dict:
        func = func_dict["func"]

    signature_info = "("
    if "properties" in func_dict['parameters']:
        for key in func_dict['parameters']['properties'].keys():
            signature_info += key + ": " + func_dict['parameters']['properties'][key]['type']
            signature_info += ", "

    signature_info += "):"

    func_name = func_dict["name"]
    description = func_dict["description"]

    func_code = f'''
def {func_name}{signature_info}
    """{description}"""
    {func}
    '''

    local_scope = {}
    exec(func_code, {}, local_scope)
    func_method = local_scope[func_name]

    # Define the new class using type
    class_attributes = {
        "execute": staticmethod(async_wrapper(func_method)),
        "codeact_func": staticmethod(func_method),
        "__init__": lambda self, *args, **kwargs: (
            super(type(self), self).__init__(
                name=func_name,
                description=description,
                parameters=func_dict["parameters"],
                func_signature=signature_info[:-1], # delete ":"
                *args,
                **kwargs)
        )
    }

    new_class = type(f"{func_name}", (BaseTool,), class_attributes)
    instance = new_class()

    return instance

def dict_to_tool_fun(func_dict: dict):
    func = "return"
    if "func" in func_dict:
        func = func_dict["func"]

    signature_info = "("
    if "properties" in func_dict['parameters']:
        for key in func_dict['parameters']['properties'].keys():
            signature_info += key + ": " + func_dict['parameters']['properties'][key]['type']
            signature_info += ", "

    signature_info += "):"

    func_name = func_dict["name"]
    description = func_dict["description"]

    func_code = f'''
def {func_name}{signature_info}
    """{description}"""
    {func}
    '''

    exec(func_code, globals())

    return globals().get(func_name)