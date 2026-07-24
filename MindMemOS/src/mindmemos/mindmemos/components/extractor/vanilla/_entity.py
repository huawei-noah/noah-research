"""Entity resolution and deduplication utilities.

Resolves LLM-extracted entity candidates into business ``Entity`` DTOs,
and deduplicates by canonical name + entity type.
"""

from __future__ import annotations

from ....typing import Entity
from .memory import ExtractedEntityCandidate, ExtractedMemoryCandidate

_LOCAL_FALLBACK_ENTITY_STOPWORDS = {"user", "the_user", "用户", "assistant", "助手"}


def resolve_candidate_entities(
    candidate: ExtractedMemoryCandidate,
    extracted_entities: list[ExtractedEntityCandidate],
    fallback_entities: list[Entity],
) -> list[Entity]:
    """Resolve LLM entity candidate references into Entity DTOs.

    Looks up each entity ref_id from the candidate in the extraction
    result. Falls back to preprocessor-detected entities only when the
    extractor omitted the entities field entirely.
    """
    if "entities" not in candidate.model_fields_set:
        return _filter_local_fallback_entities(fallback_entities)

    by_ref_id = {entity.ref_id: entity for entity in extracted_entities}
    entities: list[Entity] = []
    for ref_id in candidate.entities:
        extracted = by_ref_id.get(ref_id)
        if extracted is None:
            continue
        entities.append(
            Entity(
                name=extracted.entity_name,
                canonical_name=extracted.entity_name,
                entity_type=extracted.entity_type,
                description=extracted.description,
                confidence=extracted.confidence,
                extractor="vanilla_llm",
                metadata=dict(extracted.metadata),
            )
        )
    return entities


def _filter_local_fallback_entities(entities: list[Entity]) -> list[Entity]:
    filtered: list[Entity] = []
    for entity in entities:
        name = (entity.canonical_name or entity.name or "").strip().lower()
        if name in _LOCAL_FALLBACK_ENTITY_STOPWORDS:
            continue
        filtered.append(entity)
    return filtered


def deduplicate_entities(entities: list[Entity]) -> list[Entity]:
    """Deduplicate entities by (canonical_name, entity_type), keeping first occurrence."""
    unique: dict[tuple[str, str | None], Entity] = {}
    for entity in entities:
        name = entity.canonical_name or entity.name
        if not name:
            continue
        unique.setdefault((name, entity.entity_type), entity)
    return list(unique.values())


def entity_names(entities: list[Entity]) -> list[str]:
    """Extract canonical or raw names from an entity list."""
    return [entity.canonical_name or entity.name for entity in entities if entity.canonical_name or entity.name]
