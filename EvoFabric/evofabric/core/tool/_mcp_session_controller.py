# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import asyncio
import traceback
from contextlib import AsyncExitStack
from typing import Optional
from mcp import ClientSession
from pydantic import Field, PrivateAttr

from ._tool_utils import create_mcp_session
from ..factory import BaseComponent
from ..typing import McpServerLink
from ...logger import get_logger

logger = get_logger()


class McpSessionController(BaseComponent):
    """
    Manages MCP server connection lifecycle and session operations.

    Handles connection establishment, session management, and graceful
    disconnection from MCP servers using async contexts.
    """

    server_link: McpServerLink
    """Reference to MCP server communication link object"""

    server_name: str
    """Name identifier for the connected MCP server"""

    persistent_link: bool = Field(default=False)
    """If true, __aexit__ will not disconnect the Mcp server. Default is false."""

    _connected: bool = PrivateAttr(default=False)
    """Tracks whether an active server connection exists"""

    _session: Optional[ClientSession] = PrivateAttr(default=None)
    """Stores the active HTTP client session for MCP communication"""

    _session_task: Optional = PrivateAttr(default=None)
    """Holds the background session maintenance task"""

    @property
    def session(self):
        """Returns the current active client session (read-only)"""
        return self._session

    @property
    def is_connect(self):
        """Returns connection status (True/False) for monitoring purposes"""
        return self._connected

    async def connect(self):
        """
        Initiates and waits for MCP server connection.

        Creates a background task to maintain the session loop and blocks until
        connection is fully established. Safe to call multiple times.
        """
        if self._session_task:
            return

        ready_event = asyncio.Event()
        self._session_task = asyncio.create_task(self._run_session_loop(ready_event))
        await ready_event.wait()
        self._connected = True
        logger.debug(f"Mcp session {self.server_name} connected")

    async def disconnect(self):
        """
        Terminates the MCP connection.

        Cancels session task, waits for cleanup, and resets all connection state.
        """
        if self._session_task:
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError as e:
                ...
            finally:
                self._session_task = None
                self._session = None
                self._connected = False
                logger.debug(f"Mcp session {self.server_name} disconnected")
        else:
            logger.debug("Mcp session do not exist, no need to disconnect.")

    async def _run_session_loop(self, ready_event: asyncio.Event):
        try:
            async with create_mcp_session(self.server_link) as session:
                await session.initialize()
                self._session = session
                ready_event.set()

                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    raise
        except Exception as e:
            logger.info(f"MCP session encountered error. Disconnecting..."
                        f"Server name: {self.server_name}, "
                        f"error: {e}, "
                        f"traceback: {traceback.format_exc()}")
        finally:
            self._session = None
            self._session_task = None
            self._connected = False

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.persistent_link:
            return
        await self.disconnect()
