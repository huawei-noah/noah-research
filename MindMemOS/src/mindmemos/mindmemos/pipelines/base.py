"""Shared pipeline dependencies and internal contracts."""

from __future__ import annotations

from typing import Protocol

from .memory_db import (
    MemoryDbReader,
    MemoryDbWriter,
    MemoryOperationRecorder,
)


class HasMemoryDbAccess(Protocol):
    """Internal contract for pipeline implementations that use memory DB access."""

    db_reader: MemoryDbReader
    db_writer: MemoryDbWriter


class MemoryDbPipelineMixin:
    """Provide shared memory DB reader/writer dependencies for concrete pipelines."""

    def __init__(
        self,
        *,
        db_reader: MemoryDbReader | None = None,
        db_writer: MemoryDbWriter | None = None,
        recorder: MemoryOperationRecorder | None = None,
    ) -> None:
        self.db_reader = db_reader or MemoryDbReader()
        self.db_writer = db_writer or MemoryDbWriter()
        self.recorder = recorder or MemoryOperationRecorder()
