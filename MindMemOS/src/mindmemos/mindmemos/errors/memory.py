"""Memory operation errors."""

from .base import MindMemOSError


class MemoryUpdateError(MindMemOSError):
    """Raised when an in-place memory update cannot be completed safely."""
