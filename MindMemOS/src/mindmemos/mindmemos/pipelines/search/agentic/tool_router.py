"""Agentic tool selection."""

from __future__ import annotations

from ....logging import get_logger
from .base import SearchTool

logger = get_logger(__name__)


class DefaultToolRouter:
    """Select a search tool from configured tool names."""

    def __init__(
        self,
        *,
        tools: dict[str, SearchTool],
        enabled_tools: list[str],
        default_tool: str,
    ) -> None:
        self._tools = {name: tool for name, tool in tools.items() if name in enabled_tools}
        self._default_tool = default_tool if default_tool in self._tools else next(iter(self._tools), "")
        if not self._tools:
            raise ValueError("Agentic search requires at least one configured tool")

    def select(self, requested_tool: str | None = None) -> SearchTool:
        """Return the requested tool or the configured default."""

        if requested_tool and requested_tool in self._tools:
            return self._tools[requested_tool]
        if requested_tool and requested_tool not in self._tools:
            logger.warning("agentic_tool_not_configured", requested_tool=requested_tool)
        return self._tools[self._default_tool]
