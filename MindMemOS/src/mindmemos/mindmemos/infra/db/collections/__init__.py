"""Per-collection Qdrant repositories.

Each module binds exactly one logical Qdrant table and exposes a small typed
adapter; shared mechanics live in :mod:`.base` and the connection-owning
:class:`mindmemos.infra.db.engine.QdrantEngine`.
"""

from __future__ import annotations

from .add_record import AddRecordRepository
from .base import CollectionRepository
from .entity import EntityRepository
from .memory import MemoryRepository
from .schema_add_buffer import SchemaAddBufferRepository
from .search_record import SearchRecordRepository
from .skill import SkillVersionRepository
from .source import SourceRepository

__all__ = [
    "AddRecordRepository",
    "CollectionRepository",
    "EntityRepository",
    "MemoryRepository",
    "SchemaAddBufferRepository",
    "SearchRecordRepository",
    "SkillVersionRepository",
    "SourceRepository",
]
