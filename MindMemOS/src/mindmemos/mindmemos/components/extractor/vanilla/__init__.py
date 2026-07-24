"""Vanilla add extraction components."""

from ._dedup import CandidateDeduplicator
from ._safety_gate import AddSafetyGate, PlannedAddAction
from .add_builder import AddCoreBuilder
from .add_recall import RelatedMemoryRecall
from .memory import (
    ExtractedEntityCandidate,
    ExtractedMemoryCandidate,
    ExtractedSourceCandidate,
    MemoryExtractionResult,
    PropertyBinding,
    VanillaMemoryExtractor,
    parse_memory_extraction_json,
)

__all__ = [
    "AddSafetyGate",
    "AddCoreBuilder",
    "CandidateDeduplicator",
    "ExtractedEntityCandidate",
    "ExtractedMemoryCandidate",
    "ExtractedSourceCandidate",
    "MemoryExtractionResult",
    "PlannedAddAction",
    "PropertyBinding",
    "RelatedMemoryRecall",
    "VanillaMemoryExtractor",
    "parse_memory_extraction_json",
]
