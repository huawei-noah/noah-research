# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import functools
import inspect
from ...logger import get_logger

logger = get_logger()


def trace_chat(function):
    if inspect.isasyncgenfunction(function):
        @functools.wraps(function)
        async def wrapper(*args, **kwargs):
            async for item in function(*args, **kwargs):
                yield item

        return wrapper
    elif asyncio.iscoroutinefunction(function):
        @functools.wraps(function)
        async def wrapper(*args, **kwargs):
            result = await function(*args, **kwargs)
            return result
        return wrapper
    else:
        logger.warning(f"Unrecognized function: {function}, "
                       f"trace_chat only support async function or async generator")

