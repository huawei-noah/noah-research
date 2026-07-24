"""Extractor component protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ...typing import ExtractionEnvelope, MemoryRequestContext, MemoryView, PreprocessedText, RelatedMemoryRecallResult

if TYPE_CHECKING:
    from .vanilla.memory import MemoryExtractionResult


class MemoryExtractor(Protocol):
    """Protocol for memory extraction components."""

    async def extract_from_envelope(
        self,
        envelope: ExtractionEnvelope,
        preprocessed_texts: list[PreprocessedText],
        context: MemoryRequestContext,
    ) -> "MemoryExtractionResult": ...


class AddRecallStrategy(Protocol):
    """Protocol for add-time related memory recall."""

    async def recall(
        self,
        ctx: MemoryRequestContext,
        preprocessed: PreprocessedText,
        *,
        active_memories: list[MemoryView] | None = None,
    ) -> RelatedMemoryRecallResult: ...
