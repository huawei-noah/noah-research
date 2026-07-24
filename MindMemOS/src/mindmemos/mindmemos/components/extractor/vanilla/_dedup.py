"""Intra-batch candidate deduplication for LLM-extracted memory candidates."""

from __future__ import annotations

from .memory import ExtractedMemoryCandidate


def _normalize_entity_name(name: str) -> str:
    """Normalize entity name for comparison: lowercase, strip whitespace."""
    return name.strip().lower()


class CandidateDeduplicator:
    """Deduplicate memory candidates within a single extraction batch.

    Stateless component: each call processes one batch independently. Uses
    content hash + memory type to identify duplicate candidates, then merges
    by keeping the highest-confidence content and union-merging entities,
    source_refs, and related memory ids.
    """

    def dedup(self, candidates: list[ExtractedMemoryCandidate]) -> list[ExtractedMemoryCandidate]:
        """Deduplicate a list of memory candidates from one extraction batch.

        Merge rules for candidates with the same (content_hash, mem_type):
        - content: keep highest-confidence candidate's content
        - confidence: keep max
        - entities: union-merge (normalized)
        - source_refs: union-merge
        - related_memory_ids: union-merge
        - metadata: shallow merge, winner's values take precedence
        """
        if len(candidates) <= 1:
            return list(candidates)

        groups: dict[tuple[str, str], list[ExtractedMemoryCandidate]] = {}
        order: list[tuple[str, str]] = []

        for candidate in candidates:
            content_hash = candidate.metadata.get("content_hash", "") or candidate.content
            key = (content_hash, candidate.mem_type)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(candidate)

        result: list[ExtractedMemoryCandidate] = []
        for key in order:
            group = groups[key]
            if len(group) == 1:
                result.append(group[0])
                continue
            result.append(self._merge_group(group))
        return result

    def _merge_group(self, group: list[ExtractedMemoryCandidate]) -> ExtractedMemoryCandidate:
        """Merge a group of duplicate candidates into one."""
        # Pick winner: highest confidence, first occurrence as tiebreaker
        winner = max(group, key=lambda c: c.confidence or 0.0)

        # Union-merge entities (normalized dedup)
        seen_entities: set[str] = set()
        merged_entities: list[str] = []
        for candidate in group:
            for entity_id in candidate.entities:
                normalized = _normalize_entity_name(entity_id)
                if normalized not in seen_entities:
                    seen_entities.add(normalized)
                    merged_entities.append(entity_id)

        # Union-merge source_refs
        merged_source_refs: list[str] = []
        seen_sources: set[str] = set()
        for candidate in group:
            for ref in candidate.source_refs:
                if ref not in seen_sources:
                    seen_sources.add(ref)
                    merged_source_refs.append(ref)

        # Union-merge related_memory_ids
        merged_related: list[str] = []
        seen_related: set[str] = set()
        for candidate in group:
            for mid in candidate.related_memory_ids:
                if mid not in seen_related:
                    seen_related.add(mid)
                    merged_related.append(mid)

        # Merge metadata (shallow, winner takes precedence)
        merged_metadata: dict[str, object] = {}
        for candidate in group:
            merged_metadata.update(dict(candidate.metadata))
        merged_metadata.update(dict(winner.metadata))

        return winner.model_copy(
            update={
                "entities": merged_entities,
                "source_refs": merged_source_refs,
                "related_memory_ids": merged_related,
                "metadata": merged_metadata,
            }
        )
