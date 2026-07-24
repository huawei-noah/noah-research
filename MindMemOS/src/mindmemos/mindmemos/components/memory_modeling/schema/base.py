"""Modeling component protocols."""

from __future__ import annotations

from typing import Any, Protocol


class EntitySchemaProvider(Protocol):
    """Protocol for components that provide entity modeling schemas."""

    def get_all_dicts(self) -> list[dict[str, Any]]: ...

    def list_types(self) -> list[str]: ...
