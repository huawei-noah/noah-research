"""Memory extraction components for add pipeline candidates."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from ....logging import get_logger
from ....typing import (
    ChunkBoundary,
    ExtractionEnvelope,
    MemoryRequestContext,
    MemoryType,
    PreprocessedText,
    Turn,
    TurnMessageRef,
)

logger = get_logger(__name__)


def _message_time_payload(timestamp_ms: int | None) -> dict[str, Any]:
    if timestamp_ms is None:
        return {}
    message_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    return {
        "timestamp_ms": timestamp_ms,
        "message_time": message_time.strftime("%Y-%m-%d %H:%M:%S"),
        "message_date": message_time.strftime("%Y-%m-%d"),
    }


class ExtractedMemoryCandidate(BaseModel):
    """A memory candidate produced by LLM extraction or fallback extraction."""

    ref_id: str
    content: str
    mem_type: MemoryType = "fact"
    confidence: float | None = None
    importance: float | None = None
    entities: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    related_memory_ids: list[str] = Field(default_factory=list)
    action_hint: Literal["add", "reinforce", "update", "merge", "skip"] = "add"
    target_memory_id: str | None = None
    reason: str | None = None
    segment_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedEntityCandidate(BaseModel):
    """An entity candidate referenced by extracted memory candidates."""

    ref_id: str
    entity_name: str
    entity_type: str | None = None
    description: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedSourceCandidate(BaseModel):
    """A source candidate referenced by extracted memory candidates."""

    ref_id: str
    source_type: str = "message"
    message_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PropertyBinding(BaseModel):
    """Schema-mode property binding candidate.

    Add online does not execute timeline updates from this object; later schema
    and consolidation phases can route it to the right operator.
    """

    entity_ref_id: str
    memory_ref_id: str
    property_name: str
    schema_version: str | None = None
    update_candidate_memory_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryExtractionResult(BaseModel):
    """Structured extractor output consumed by add planning."""

    memories: list[ExtractedMemoryCandidate] = Field(default_factory=list)
    entities: list[ExtractedEntityCandidate] = Field(default_factory=list)
    sources: list[ExtractedSourceCandidate] = Field(default_factory=list)
    property_bindings: list[PropertyBinding] = Field(default_factory=list)


class MemoryExtractor(Protocol):
    async def extract_from_envelope(
        self,
        envelope: ExtractionEnvelope,
        preprocessed_texts: list[PreprocessedText],
        context: MemoryRequestContext,
    ) -> MemoryExtractionResult: ...


class VanillaMemoryExtractor:
    """Extract vanilla-mode memories with optional LLM and deterministic fallback."""

    def __init__(self, *, llm_client=None, enable_entities: bool = False) -> None:
        self._llm_client = llm_client
        self._enable_entities = enable_entities

    async def extract_from_envelope(
        self,
        envelope: ExtractionEnvelope,
        preprocessed_texts: list[PreprocessedText],
        context: MemoryRequestContext,
    ) -> MemoryExtractionResult:
        """Extract memories from an ExtractionEnvelope (chunked path).

        The envelope separates extractable evidence from non-extractable
        context (history, prior events, recall). Boundary metadata
        influences extraction conservatism.

        Args:
            envelope: Structured extraction context with extractable/context split.
            preprocessed_texts: Preprocessed text for the extractable messages.
            context: Request context for logging.

        Returns:
            MemoryExtractionResult with candidates extracted from evidence only.
        """
        if self._llm_client is None:
            return self._envelope_fallback(envelope, preprocessed_texts)
        try:
            response = await self._llm_client.chat(
                task="memory.add.extract",
                messages=_envelope_prompt_messages(
                    envelope, preprocessed_texts, context, enable_entities=self._enable_entities
                ),
                format_parser=parse_memory_extraction_json,
            )
            result = MemoryExtractionResult.model_validate(_normalize_extraction_payload(response.parsed))
            return _mark_extractor(result, "vanilla_llm_chunked")
        except Exception:
            logger.warning(
                "llm_chunked_extraction_failed",
                request_id=context.request_id,
                chunk_index=envelope.chunk_index,
                boundary=envelope.boundary,
                exc_info=True,
            )
            result = self._envelope_fallback(envelope, preprocessed_texts)
            return _mark_extractor(result, "fallback_chunked")

    def _envelope_fallback(
        self,
        envelope: ExtractionEnvelope,
        preprocessed_texts: list[PreprocessedText],
    ) -> MemoryExtractionResult:
        """Deterministic fallback for chunked extraction."""
        memories: list[ExtractedMemoryCandidate] = []
        entities: list[ExtractedEntityCandidate] = []
        sources: list[ExtractedSourceCandidate] = []

        base_confidence = _boundary_confidence(envelope.boundary)

        for i, (msg_ref, preprocessed) in enumerate(
            zip(envelope.extractable_messages, preprocessed_texts, strict=False)
        ):
            if not msg_ref.is_extractable:
                continue
            if not preprocessed.normalized_text.strip():
                continue

            seg_tag = f"chunk{envelope.chunk_index}_msg{i}"
            memory_ref_id = f"m_{seg_tag}"
            source_ref_id = f"s_{seg_tag}"

            memories.append(
                ExtractedMemoryCandidate(
                    ref_id=memory_ref_id,
                    content=preprocessed.normalized_text,
                    mem_type="fact",
                    confidence=base_confidence,
                    source_refs=[source_ref_id],
                    reason="fallback_chunked_memory",
                    segment_id=seg_tag,
                    metadata={
                        "extractor": "fallback_chunked",
                        "chunk_index": envelope.chunk_index,
                        "boundary": envelope.boundary,
                    },
                )
            )
            sources.append(
                ExtractedSourceCandidate(
                    ref_id=source_ref_id,
                    source_type="message",
                    message_index=msg_ref.message_index,
                    metadata={"evidence_index": i},
                )
            )

        return MemoryExtractionResult(
            memories=memories,
            entities=entities,
            sources=sources,
        )


def parse_memory_extraction_json(content: str) -> dict[str, Any]:
    """Parse extractor JSON, tolerating simple markdown JSON fences."""

    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return json.loads(text)


def _normalize_extraction_payload(payload: Any) -> Any:
    """Normalize common LLM shape drift before Pydantic validation."""
    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)
    extracted_entities = normalized.get("entities")
    has_resolvable_entities = isinstance(extracted_entities, list) and bool(extracted_entities)
    sources = list(normalized.get("sources") or [])
    seen_source_refs = {source.get("ref_id") for source in sources if isinstance(source, dict)}
    memories: list[Any] = []
    for raw_memory in normalized.get("memories") or []:
        if not isinstance(raw_memory, dict):
            memories.append(raw_memory)
            continue

        memory = dict(raw_memory)
        source_refs: list[str] = []
        for source_ref in memory.get("source_refs") or []:
            if isinstance(source_ref, str):
                source_refs.append(source_ref)
                continue
            if not isinstance(source_ref, dict):
                continue
            ref_id = source_ref.get("ref_id")
            if not isinstance(ref_id, str) or not ref_id:
                continue
            source_refs.append(ref_id)
            if ref_id not in seen_source_refs:
                sources.append(source_ref)
                seen_source_refs.add(ref_id)
        memory["source_refs"] = source_refs
        if not has_resolvable_entities:
            memory.pop("entities", None)
        elif "entities" in memory and not memory["entities"]:
            memory.pop("entities")
        for empty_field in ("related_memory_ids", "target_memory_id"):
            if memory.get(empty_field) in (None, [], ""):
                memory.pop(empty_field, None)
        cleaned_metadata = _clean_memory_metadata(memory.get("metadata"))
        if cleaned_metadata:
            memory["metadata"] = cleaned_metadata
        else:
            memory.pop("metadata", None)
        memories.append(memory)

    normalized["memories"] = memories
    if not has_resolvable_entities:
        normalized.pop("entities", None)
    normalized["sources"] = sources
    return normalized


def _clean_memory_metadata(metadata: Any) -> dict[str, Any]:
    """Keep only compact, useful temporal metadata from extractor output."""

    if not isinstance(metadata, dict):
        return {}
    resolved_date = metadata.get("resolved_event_date")
    resolved_range = metadata.get("resolved_event_range")
    has_resolved_time = bool(resolved_date) or bool(resolved_range)
    cleaned: dict[str, Any] = {}
    if has_resolved_time and metadata.get("temporal_text"):
        cleaned["temporal_text"] = metadata["temporal_text"]
    if resolved_date:
        cleaned["resolved_event_date"] = resolved_date
    if resolved_range:
        cleaned["resolved_event_range"] = resolved_range
    return cleaned


def _dominant_lang(preprocessed_texts: list[PreprocessedText]) -> str:
    """Determine the dominant language across preprocessed texts.

    Returns 'zh' if any segment is primarily Chinese, else 'en'.
    """
    for pt in preprocessed_texts:
        if pt.lang == "zh":
            return "zh"
    return "en"


def _boundary_confidence(boundary: ChunkBoundary) -> float:
    """Map chunk boundary to a base confidence for fallback extraction.

    COMPLETE: full confidence. OPEN_HEAD / OPEN_TAIL: moderate.
    ORPHAN: conservative. COMPACTED: full (head+tail preserved).
    """
    mapping: dict[str, float] = {
        "complete": 1.0,
        "compacted": 0.9,
        "open_head": 0.7,
        "open_tail": 0.7,
        "orphan": 0.5,
    }
    return mapping.get(boundary, 0.7)


def _envelope_prompt_messages(
    envelope: ExtractionEnvelope,
    preprocessed_texts: list[PreprocessedText],
    context: MemoryRequestContext,
    *,
    enable_entities: bool = False,
) -> list[dict[str, Any]]:
    """Build LLM prompt messages from an ExtractionEnvelope.

    The prompt clearly separates:
    1. Extractable evidence (current chunk messages) — can produce candidates.
    2. Context section (history, prior events, recall) — NON-extractable.
    3. Boundary metadata — informs extraction conservatism.
    """
    extractable_entries: list[dict[str, Any]] = []
    for i, (msg_ref, preprocessed) in enumerate(zip(envelope.extractable_messages, preprocessed_texts, strict=False)):
        extractable_entries.append(
            {
                "index": msg_ref.message_index,
                "evidence_index": i,
                **_message_time_payload(msg_ref.timestamp),
                "role": msg_ref.role,
                "raw_role": msg_ref.raw_role,
                "speaker": msg_ref.speaker,
                "text": preprocessed.normalized_text if i < len(preprocessed_texts) else msg_ref.text,
                "is_extractable": msg_ref.is_extractable,
            }
        )

    # Build context section (non-extractable)
    context_section: dict[str, Any] = {}

    history_turns: list[dict[str, Any]] = []
    for turn in envelope.history.in_request_history:
        history_item = _context_turn_payload(turn)
        if history_item:
            history_turns.append(history_item)
    if history_turns:
        context_section["history"] = history_turns

    ext_turns: list[dict[str, Any]] = []
    for turn in envelope.history.external_history:
        history_item = _context_turn_payload(turn)
        if history_item:
            ext_turns.append(history_item)
    if ext_turns:
        context_section["external_history"] = ext_turns

    if envelope.recalled_memories:
        context_section["related_memories"] = envelope.recalled_memories

    # Current context messages (non-extractable chunk context, e.g. compaction summaries)
    if envelope.current_context_messages:
        context_section["current_context"] = [
            _context_message_payload(msg) for msg in envelope.current_context_messages
        ]

    payload: dict[str, Any] = {
        "request_id": context.request_id,
        "project_id": context.project_id,
        "chunk_index": envelope.chunk_index,
        "boundary": envelope.boundary,
        "instruction": (
            "EXTRACT memories ONLY from the 'extractable' section below. "
            "The 'context' section is provided for reference resolution and "
            "duplicate detection only — do NOT create new memories from context. "
            "For multi-speaker dialogue, first-person pronouns refer to the message speaker. "
            'Do not rewrite unknown speakers as "the user"; preserve the named speaker as the subject. '
            'Use source_refs in the form "s{evidence_index}" for provenance; '
            "the system binds message sources automatically. "
            + (
                "Do not output top-level sources or property_bindings."
                if enable_entities
                else "Do not output entities, top-level sources, or property_bindings."
            )
        ),
        "extractable": extractable_entries,
        "context": context_section,
    }

    # Add boundary-specific guidance
    boundary_guidance = _boundary_guidance(envelope.boundary)
    if boundary_guidance:
        payload["boundary_guidance"] = boundary_guidance

    from ....prompts import get_extraction_system_prompt

    lang = _dominant_lang(preprocessed_texts) if preprocessed_texts else "en"
    system_prompt = get_extraction_system_prompt(lang, enable_entities=enable_entities)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _context_turn_payload(turn: Turn) -> dict[str, Any] | None:
    messages = [_context_message_payload(message) for message in turn.messages if message.is_extractable]
    if not messages:
        return None
    text = "\n".join(_context_message_text(message) for message in turn.messages if message.is_extractable)
    return {"boundary": turn.boundary, "text": text, "messages": messages}


def _context_message_payload(message: TurnMessageRef) -> dict[str, Any]:
    return {
        "role": message.role,
        "raw_role": message.raw_role,
        "speaker": message.speaker,
        "text": message.text,
        "message_index": message.message_index,
    }


def _context_message_text(message: TurnMessageRef) -> str:
    label = _context_message_label(message)
    return f"{label}: {message.text}" if label else message.text


def _context_message_label(message: TurnMessageRef) -> str | None:
    if message.speaker:
        return message.speaker
    if message.raw_role:
        return message.raw_role
    return message.role or None


def _boundary_guidance(boundary: ChunkBoundary) -> str:
    """Return extraction guidance based on chunk boundary type."""
    guides: dict[str, str] = {
        "complete": "",
        "compacted": (
            "This chunk contains a compacted turn. Extract memories from the "
            "head and tail text only. The middle summary is context-only."
        ),
        "open_head": (
            "This chunk starts without the user context that likely triggered it. "
            "Mark resulting candidates as partial provenance. Be conservative."
        ),
        "open_tail": (
            "This chunk ends with an unfinished turn. Do not treat the last "
            "assistant message as a final conclusion. Extract only clear facts."
        ),
        "orphan": (
            "This chunk has no user context at all. Extract only explicit, "
            "stable facts. Mark all candidates with lower confidence."
        ),
    }
    return guides.get(boundary, "")


def _mark_extractor(result: MemoryExtractionResult, extractor: str) -> MemoryExtractionResult:
    return result.model_copy(
        update={
            "memories": [
                memory.model_copy(update={"metadata": {**dict(memory.metadata), "extractor": extractor}})
                for memory in result.memories
            ]
        }
    )
