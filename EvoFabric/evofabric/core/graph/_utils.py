# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import inspect
import uuid
from typing import Any, Callable


def _make_class_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _effective_func(callable_obj: Callable[..., Any]) -> Callable[..., Any]:
    """get callable object"""
    if not inspect.isfunction(callable_obj) and hasattr(callable_obj, "__call__"):
        return callable_obj.__call__
    return callable_obj


def _has_stream_writer(func: Callable[..., Any]) -> bool:
    try:
        return "stream_writer" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
