# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import contextvars
import threading
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Optional, Union

from pydantic import BaseModel

from ..factory import BaseComponent

from ...logger import get_logger


logger = get_logger()


def default_handler(x):
    """The default handler does nothing."""
    ...


_ON_MESSAGE: Callable[[dict], Union[None, Awaitable[None]]] = default_handler
_LOCK = threading.Lock()


def set_streaming_handler(
        callback: Callable[[dict], Union[None, Awaitable[None]]]
) -> None:
    """Register the callback of stream messages"""
    global _ON_MESSAGE
    with _LOCK:
        _ON_MESSAGE = callback


class StreamCtx(BaseModel):
    # node level context
    node_name: Optional[str] = None
    call_id: Optional[str] = None

    # tool level context
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def __bool__(self) -> bool:
        """If all attr is empty, return False, otherwise return true."""
        return any(
            [self.node_name, self.call_id, self.tool_name, self.tool_call_id]
        )

    def __repr__(self) -> str:
        """Return repr of StreamCtx"""
        parts = []
        if self.node_name or self.call_id:
            parts.append(
                f"Node(name={self.node_name or '?'}, id={self.call_id or '?'})"
            )
        if self.tool_name or self.tool_call_id:
            parts.append(
                f"Tool(name={self.tool_name or '?'}, id={self.tool_call_id or '?'})"
            )

        return " -> ".join(parts) if parts else "StreamCtx(empty)"


_STREAM_CTX: contextvars.ContextVar[StreamCtx] = contextvars.ContextVar("ctx", default=StreamCtx())


def current_ctx() -> StreamCtx:
    """Return current context"""
    return _STREAM_CTX.get()


@contextmanager
def stream_writer_env(ctx_updates: StreamCtx):
    """
    Generate stream context information for StreamWriter

    Example:
        with stream_writer_env(StreamCtx(node_name='NodeA')):
            # ...
    """
    parent_ctx = _STREAM_CTX.get()

    update_data = ctx_updates.model_dump(exclude_unset=True)
    new_ctx = parent_ctx.model_copy(update=update_data)

    token = _STREAM_CTX.set(new_ctx)

    try:
        yield
    finally:
        _STREAM_CTX.reset(token)


class StreamWriter(BaseComponent):
    @staticmethod
    def put(payload: Any) -> None:
        """Put streaming msg into stream writer and trigger the msg handler"""
        ctx = current_ctx()
        envelope = {
            **ctx.model_dump(exclude_unset=True),
            "payload": payload,
        }

        handler = _ON_MESSAGE
        if asyncio.iscoroutinefunction(handler):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            loop.create_task(handler(envelope))
        else:
            handler(envelope)


_G_STREAM_WRITER = StreamWriter()


def get_stream_writer():
    """Return global stream writer"""
    return _G_STREAM_WRITER
