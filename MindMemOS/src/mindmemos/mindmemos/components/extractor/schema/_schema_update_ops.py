"""Schema-add update operation DTOs."""

from __future__ import annotations

from dataclasses import dataclass

from ....typing import MemoryWrite


@dataclass(slots=True)
class SchemaMemoryUpdate:
    """One schema-add update targeting an existing memory id.

    The writer owns new version id generation; this DTO only carries the target
    memory id plus the desired updated content/vector material.
    """

    target_memory_id: str
    memory: MemoryWrite
    reason: str
