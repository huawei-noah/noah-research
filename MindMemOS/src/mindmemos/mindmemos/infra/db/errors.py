"""Database-layer exceptions."""

from __future__ import annotations

from ...errors import MindMemOSError


class MemoryDbError(MindMemOSError):
    """Base class for memory database errors."""


class MemoryDbConfigurationError(MemoryDbError):
    """Database adapter is missing required configuration."""


class MemoryDbValidationError(MemoryDbError):
    """Invalid DB input such as project or vector mismatch."""


class MemoryNotFoundError(MemoryDbError):
    """Requested memory does not exist or is outside the caller scope."""
