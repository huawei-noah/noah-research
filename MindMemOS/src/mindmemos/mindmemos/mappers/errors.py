"""Mapper-specific errors."""

from __future__ import annotations

from ..errors import MindMemOSError


class MappingError(MindMemOSError, ValueError):
    """Base error raised when DTO and DB model conversion is invalid."""


class ProjectIsolationError(MappingError):
    """Raised when a DTO tries to cross the request project boundary."""


class MissingSourceIdError(MappingError):
    """Raised when a SourceRef must be persisted but has no source_id."""
