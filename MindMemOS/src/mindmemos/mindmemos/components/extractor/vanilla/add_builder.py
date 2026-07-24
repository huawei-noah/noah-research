"""Pure orchestration of vanilla add extraction phases.

Composed by ``VanillaAddPipeline``. Each phase delegates to a component;
no operator implementations live here.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ....config import VanillaAddConfig
from ....logging import get_logger
from ....typing import (
    AddPipelineInput,
    EntityVectorWrite,
    EntityWrite,
    ExtractionEnvelope,
    MemoryAddEventItem,
    MemoryDbUpdateCommand,
    MemoryDbWritePlan,
    MemoryRequestContext,
    MemoryWrite,
    PreprocessedText,
    RelatedMemoryRecallResult,
    SourceRef,
    SourceWrite,
    TurnMessageRef,
    VectorWrite,
)
from ...chunker import SourceAwareSegment
from ...chunker.vanilla import (
    ChunkPlanner,
    HistoryPacker,
    LongTurnCompactor,
    LongTurnSummarizer,
    TurnGrouper,
)
from ...id import generate_entity_id, generate_memory_id, generate_source_id
from ...memory_modeling.vanilla import (
    build_extracted_from_edge,
    build_mentioned_in_source_edge,
    build_mentions_edge,
    build_relates_to_edge,
)
from ...text import TextPreprocessor
from ...text.vectorizer import MemoryVectorizer
from ._dedup import CandidateDeduplicator
from ._entity import (
    deduplicate_entities,
    entity_names,
    resolve_candidate_entities,
)
from ._safety_gate import AddSafetyGate, PlannedAddAction
from ._update_commands import (
    build_merge_archive_commands,
    build_reinforcement_command,
    build_update_command,
)
from .add_recall import RelatedMemoryRecall
from .memory import (
    ExtractedEntityCandidate,
    ExtractedMemoryCandidate,
    VanillaMemoryExtractor,
)

logger = get_logger(__name__)


def _source_time_metadata(segment: SourceAwareSegment) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if segment.timestamp is not None:
        metadata["source_timestamp_ms"] = segment.timestamp
    if segment.role is not None:
        metadata["source_role"] = segment.role
    raw_role = segment.metadata.get("raw_role")
    if isinstance(raw_role, str) and raw_role:
        metadata["source_raw_role"] = raw_role
    speaker = segment.metadata.get("speaker")
    if isinstance(speaker, str) and speaker:
        metadata["source_speaker"] = speaker
    return metadata


def _message_source_metadata(msg_ref: TurnMessageRef) -> dict[str, object]:
    metadata: dict[str, object] = {
        "message_index": msg_ref.message_index,
        "source_role": msg_ref.role,
    }
    if msg_ref.raw_role:
        metadata["source_raw_role"] = msg_ref.raw_role
    if msg_ref.speaker:
        metadata["source_speaker"] = msg_ref.speaker
    return metadata


def _datetime_from_millis(value: int | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def _segment_event_timestamp(segment: SourceAwareSegment, inp: AddPipelineInput) -> int:
    return segment.timestamp if segment.timestamp is not None else inp.event_timestamp


def _extract_file_url_source_refs(inp: AddPipelineInput) -> list[SourceRef]:
    """Extract SourceRef for FileMessage and UrlMessage from input.

    DialogueMessage and TextMessage are handled by the chunking pipeline.
    This only collects file/URL sources for graph edges.
    """
    from ....typing import FileMessage, UrlMessage

    source_refs: list[SourceRef] = []
    for index, message in enumerate(inp.messages):
        if isinstance(message, FileMessage):
            source_refs.append(
                SourceRef(
                    source_type="file",
                    file_path=message.file_path,
                    file_name=message.file_name,
                    mime_type=message.file_type or None,
                    is_parsed=False,
                    metadata={"message_index": index},
                )
            )
        elif isinstance(message, UrlMessage):
            source_refs.append(
                SourceRef(
                    source_type="url",
                    uri=message.url,
                    title=message.url,
                    is_parsed=False,
                    metadata={"message_index": index},
                )
            )
    return source_refs


def _recall_memory_ids(recall_result: RelatedMemoryRecallResult | None) -> set[str]:
    if recall_result is None:
        return set()
    memory_ids = {candidate.memory_id for candidate in recall_result.candidates}
    if recall_result.duplicate is not None:
        memory_ids.add(recall_result.duplicate.memory_id)
    return memory_ids


def _filter_known_memory_ids(memory_ids: list[str], known_memory_ids: set[str]) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for memory_id in memory_ids:
        if memory_id not in known_memory_ids or memory_id in seen:
            continue
        filtered.append(memory_id)
        seen.add(memory_id)
    return filtered


def _attach_search_fields(entity: EntityWrite, fields: list[str]) -> None:
    if not fields:
        return
    metadata = dict(entity.metadata or {})
    existing = [field for field in metadata.get("search_fields", []) if isinstance(field, str)]
    merged: list[str] = []
    for field in [*existing, *fields]:
        normalized = field.strip()
        if normalized and normalized not in merged:
            merged.append(normalized)
    metadata["search_fields"] = merged
    entity.metadata = metadata


def _parse_evidence_index(ref: object) -> int | None:
    """Parse a message ref ``s{evidence_index}`` into an int evidence index."""
    if not isinstance(ref, str) or not ref.startswith("s"):
        return None
    base = ref[1:].split("_", 1)[0]
    return int(base) if base.isdigit() else None


def _entity_evidence_source_refs(
    entity: Entity,
    source_by_evidence: dict[int, SourceRef],
    fallback: SourceRef,
) -> list[SourceRef]:
    """Resolve entity provenance refs to SourceRef objects for MENTIONED_IN_SOURCE edges.

    ``entity.metadata["source_refs"]`` carries message refs in the form
    ``s{evidence_index}`` produced by the entity-aware extraction prompt. Each
    is mapped through ``source_by_evidence`` (evidence_index -> SourceRef) and
    de-duplicated by source_id. When the entity has no resolvable source_refs
    (e.g. local NER fallback entities), falls back to the memory's primary
    source so prior single-edge behavior is preserved.
    """
    raw = entity.metadata.get("source_refs") if entity.metadata else None
    resolved: list[SourceRef] = []
    seen: set[str | None] = set()
    if isinstance(raw, list):
        for ref in raw:
            idx = _parse_evidence_index(ref)
            if idx is None:
                continue
            src = source_by_evidence.get(idx)
            if src is None or src.source_id in seen:
                continue
            seen.add(src.source_id)
            resolved.append(src)
    if not resolved and fallback.source_id not in seen:
        resolved.append(fallback)
    return resolved


@dataclass
class _PendingMemoryVector:
    memory: MemoryWrite
    preprocessed: PreprocessedText
    content: str


def _prefix_segment_ref_ids(
    candidates: list[ExtractedMemoryCandidate],
    entities: list[ExtractedEntityCandidate],
    segment_index: int,
) -> tuple[list[ExtractedMemoryCandidate], list[ExtractedEntityCandidate]]:
    """Prefix ref_ids with segment index to prevent cross-segment collisions.

    When extract() is called per-segment, the LLM may return the same
    ref_id (e.g. "m1") for every call.  Prefixing ensures global
    uniqueness so that the downstream candidate_by_ref_id dict never
    silently overwrites entries from earlier segments.
    """
    prefix = f"s{segment_index}_"

    remapped_candidates = []
    for candidate in candidates:
        updates = {
            "ref_id": prefix + candidate.ref_id,
            "source_refs": [prefix + source_ref for source_ref in candidate.source_refs],
        }
        if "entities" in candidate.model_fields_set:
            updates["entities"] = [prefix + entity_ref for entity_ref in candidate.entities]
        remapped_candidates.append(candidate.model_copy(update=updates))

    remapped_entities = [e.model_copy(update={"ref_id": prefix + e.ref_id}) for e in entities]

    return remapped_candidates, remapped_entities


@dataclass
class _MessageExtractionContext:
    """Per-message context needed by Phase 5-6 for source resolution."""

    msg_ref: TurnMessageRef
    source_ref: SourceRef
    segment: SourceAwareSegment
    preprocessed: PreprocessedText


def _resolve_candidate_source(
    candidate: ExtractedMemoryCandidate,
    source_context_by_ref_id: dict[str, _MessageExtractionContext],
    message_context_by_index: dict[int, _MessageExtractionContext],
) -> _MessageExtractionContext | None:
    """Resolve a candidate to its primary source message context.

    Resolution order:
    1. If candidate.source_refs is non-empty, look up the first match
       in source_context_by_ref_id.
    2. If no ref resolves and there is exactly one message context,
       auto-bind to it.
    3. Otherwise return None (ambiguous or unresolvable).
    """
    for ref_id in candidate.source_refs:
        ctx = _resolve_source_context(
            ref_id,
            source_context_by_ref_id,
            message_context_by_index,
            candidate_content=candidate.content,
        )
        if ctx is not None:
            return ctx

    if len(message_context_by_index) == 1:
        return next(iter(message_context_by_index.values()))

    return None


def _resolve_source_context(
    ref_id: str,
    source_context_by_ref_id: dict[str, _MessageExtractionContext],
    message_context_by_index: dict[int, _MessageExtractionContext],
    *,
    candidate_content: str | None = None,
) -> _MessageExtractionContext | None:
    ctx = source_context_by_ref_id.get(ref_id)
    if ctx is not None:
        return ctx

    candidates = [
        ctx
        for index in _source_ref_index_candidates(ref_id)
        if (ctx := message_context_by_index.get(index)) is not None
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return max(
            candidates,
            key=lambda candidate: _text_overlap_score(
                candidate_content or "",
                candidate.preprocessed.normalized_text,
            ),
        )
    return None


def _source_ref_index_candidates(ref_id: str) -> list[int]:
    """Infer evidence-index candidates from LLM source ref labels."""
    token = ref_id.rsplit("_", 1)[-1]
    if token.startswith("s"):
        token = token[1:]
    if not token.isdigit():
        return []

    index = int(token)
    candidates = [index - 1, index] if index > 0 else [index]
    return candidates


def _text_overlap_score(left: str, right: str) -> int:
    left_tokens = set(re.findall(r"[A-Za-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[A-Za-z0-9]+", right.lower()))
    return len(left_tokens & right_tokens)


class AddCoreBuilder:
    """Pure orchestration of the add pipeline phases.

    Receives all component dependencies via constructor injection.
    Contains no operator implementations — only phase ordering and
    data flow between components.
    """

    def __init__(
        self,
        *,
        text_preprocessor: TextPreprocessor,
        memory_extractor: VanillaMemoryExtractor,
        candidate_deduplicator: CandidateDeduplicator,
        related_memory_recall: RelatedMemoryRecall,
        safety_gate: AddSafetyGate,
        vectorizer: MemoryVectorizer,
        llm_client=None,
    ) -> None:
        self._text_preprocessor = text_preprocessor
        self._memory_extractor = memory_extractor
        self._candidate_deduplicator = candidate_deduplicator
        self._related_memory_recall = related_memory_recall
        self._safety_gate = safety_gate
        self._vectorizer = vectorizer
        self._llm_client = llm_client

    def dedup_candidates(
        self,
        candidates: list[ExtractedMemoryCandidate],
    ) -> list[ExtractedMemoryCandidate]:
        """Deduplicate candidates across all chunks (batch-wide)."""
        return self._candidate_deduplicator.dedup(candidates)

    def plan(
        self,
        preprocessed: PreprocessedText,
        mem_type: str | None,
        action_hint: str | None = None,
        confidence: float | None = None,
        target_memory_id: str | None = None,
        related_memory_ids: list[str] | None = None,
    ) -> PlannedAddAction:
        """Plan write action for one candidate."""
        return self._safety_gate.gate_segment(
            preprocessed,
            mem_type=mem_type,
            action_hint=action_hint,
            confidence=confidence,
            target_memory_id=target_memory_id,
            related_memory_ids=related_memory_ids,
        )

    async def vectorize(
        self,
        memory_id: str,
        preprocessed: PreprocessedText,
        content: str,
        consistency: str = "fast",
    ) -> tuple[VectorWrite, bool]:
        """Generate sparse + dense vectors for a memory."""
        return await self._vectorizer.vectorize(memory_id, preprocessed, content, consistency)

    async def vectorize_many(
        self,
        items: list[tuple[str, PreprocessedText, str]],
        consistency: str = "fast",
    ) -> tuple[list[VectorWrite], list[bool]]:
        """Generate sparse + dense vectors for multiple new memories."""
        vectorize_many = getattr(self._vectorizer, "vectorize_many", None)
        if vectorize_many is not None:
            return await vectorize_many(items, consistency)

        vectors: list[VectorWrite] = []
        pending: list[bool] = []
        for memory_id, preprocessed, content in items:
            vector, is_pending = await self._vectorizer.vectorize(memory_id, preprocessed, content, consistency)
            vectors.append(vector)
            pending.append(is_pending)
        return vectors, pending

    async def _preprocess_text(self, text: str, **kwargs: Any) -> PreprocessedText:
        return await asyncio.to_thread(self._text_preprocessor.preprocess_text, text, **kwargs)

    async def _preprocess_many(
        self,
        texts: list[str],
        *,
        source_refs: list[SourceRef | None],
        segment_ids: list[str | None],
    ) -> list[PreprocessedText]:
        return await asyncio.to_thread(
            self._text_preprocessor.preprocess_many,
            texts,
            source_refs=source_refs,
            segment_ids=segment_ids,
        )

    async def _vectorize_entities(
        self,
        entities: list[EntityWrite],
        memories: list[MemoryWrite],
        consistency: str,
    ) -> tuple[list[EntityVectorWrite], bool]:
        memories_by_entity: dict[str, list[MemoryWrite]] = {}
        for memory in memories:
            if memory.entity_id:
                memories_by_entity.setdefault(memory.entity_id, []).append(memory)
        return await self._vectorizer.vectorize_entities(
            entities,
            memories_by_entity=memories_by_entity,
            consistency=consistency,
        )

    async def build(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        consistency: str = "fast",
        config: VanillaAddConfig | None = None,
    ) -> tuple[MemoryDbWritePlan, list[MemoryAddEventItem], list[MemoryDbUpdateCommand]]:
        """Execute all phases with turn-aware chunking.

        Replaces per-message extraction with per-chunk extraction:
        1. Group messages into turns → plan turns into chunks
        2. For each chunk: preprocess → recall → build envelope → extract
        3. Batch-wide dedup across all chunks
        4. Phase 5-6: plan + vectorize per candidate (unchanged logic)
        """
        from ....pipelines.utils import build_entity_write, build_source_write

        cfg = config or VanillaAddConfig()
        enable_entities = bool(cfg.enable_entities)
        now = datetime.now(UTC)
        memories: list[MemoryWrite] = []
        entities_by_id: dict[str, EntityWrite] = {}
        sources_by_id: dict[str, SourceWrite] = {}
        vectors: list[VectorWrite] = []
        pending_memory_vectors: list[_PendingMemoryVector] = []
        relationships: list = []
        events: list[MemoryAddEventItem] = []
        update_commands: list[MemoryDbUpdateCommand] = []

        # Phase 1 chunked: Group → Plan
        from ....typing import DialogueMessage, TextMessage

        indexed_dialogue = [(i, m) for i, m in enumerate(inp.messages) if isinstance(m, (DialogueMessage, TextMessage))]
        grouper = TurnGrouper(cfg)
        turns = grouper.group(indexed_dialogue)
        planner = ChunkPlanner(cfg)
        chunks = planner.plan(turns)

        compactor = LongTurnCompactor(cfg)
        summarizer = LongTurnSummarizer(cfg, self._llm_client)
        for chunk in chunks:
            if not chunk.needs_compaction:
                continue
            for idx in chunk.compacted_turn_indices:
                original_turn = chunk.turns[idx]
                parts = compactor.split(original_turn)
                summary = await summarizer.summarize(parts.middle_text)
                compacted_turn, _result = compactor.compact(original_turn, summary=summary, parts=parts)
                chunk.turns[idx] = compacted_turn
            chunk.token_count = sum(turn.token_count for turn in chunk.turns)
            if chunk.boundary == "complete":
                chunk.boundary = "compacted"

        file_url_source_refs = _extract_file_url_source_refs(inp)
        for source_ref in file_url_source_refs:
            source_ref = generate_source_id(source_ref, context)
            sources_by_id.setdefault(source_ref.source_id or "", build_source_write(source_ref, context, now))

        active_memories = await self._related_memory_recall.list_active_memories(context)

        # Per-chunk extraction loop
        history_packer = HistoryPacker(cfg)
        prev_pack = None
        prev_chunk = None
        all_raw_candidates: list[ExtractedMemoryCandidate] = []

        # Track chunk-level data for Phase 5-6
        chunk_candidate_data: list[
            tuple[
                list[ExtractedMemoryCandidate],
                list[ExtractedEntityCandidate],
                RelatedMemoryRecallResult | None,
                dict[str, _MessageExtractionContext],
                dict[int, _MessageExtractionContext],
            ]
        ] = []
        source_contexts_by_ref_id: dict[str, _MessageExtractionContext] = {}

        for chunk in chunks:
            if chunk.chunk_index == 0:
                history_pack = history_packer.pack_for_first_chunk()
            else:
                history_pack = history_packer.pack_for_chunk(
                    chunk.chunk_index,
                    prev_pack,
                    prev_chunk,
                )

            # Collect extractable and non-extractable context message refs from chunk
            extractable_refs: list[TurnMessageRef] = []
            current_context_refs: list[TurnMessageRef] = []
            for turn in chunk.turns:
                for msg in turn.messages:
                    if msg.is_extractable:
                        extractable_refs.append(msg)
                    else:
                        current_context_refs.append(msg)

            if not extractable_refs:
                prev_pack = history_pack
                prev_chunk = chunk
                continue

            # Phase 2: Preprocess extractable messages and build per-message context
            preprocessed_texts: list[PreprocessedText] = []
            message_context_by_index: dict[int, _MessageExtractionContext] = {}
            original_message_contexts_by_index: dict[int, list[_MessageExtractionContext]] = {}
            pending_message_contexts: list[tuple[int, TurnMessageRef, SourceRef, SourceAwareSegment]] = []
            for evidence_index, msg_ref in enumerate(extractable_refs):
                source_ref = SourceRef(
                    source_type="message",
                    message_id=f"chunk{chunk.chunk_index}-evidence-{evidence_index}-message-{msg_ref.message_index}",
                    is_parsed=True,
                    metadata={
                        **_message_source_metadata(msg_ref),
                        "evidence_index": evidence_index,
                    },
                )
                source_ref = generate_source_id(source_ref, context)
                segment = _ref_to_segment(msg_ref, inp)
                segment = segment.model_copy(
                    update={
                        "segment_id": f"chunk{chunk.chunk_index}_evidence{evidence_index}_msg{msg_ref.message_index}",
                        "source_ref": source_ref,
                        "metadata": {
                            **dict(segment.metadata),
                            "evidence_index": evidence_index,
                            "chunk_index": chunk.chunk_index,
                        },
                    }
                )
                sources_by_id.setdefault(
                    source_ref.source_id or "",
                    build_source_write(source_ref, context, now),
                )
                pending_message_contexts.append((evidence_index, msg_ref, source_ref, segment))

            message_preprocessed = await self._preprocess_many(
                [msg_ref.text for _, msg_ref, _, _ in pending_message_contexts],
                source_refs=[source_ref for _, _, source_ref, _ in pending_message_contexts],
                segment_ids=[segment.segment_id for _, _, _, segment in pending_message_contexts],
            )
            for (evidence_index, msg_ref, source_ref, segment), pp in zip(
                pending_message_contexts,
                message_preprocessed,
                strict=True,
            ):
                preprocessed_texts.append(pp)
                context_entry = _MessageExtractionContext(
                    msg_ref=msg_ref,
                    source_ref=source_ref,
                    segment=segment,
                    preprocessed=pp,
                )
                message_context_by_index[evidence_index] = context_entry
                original_message_contexts_by_index.setdefault(msg_ref.message_index, []).append(context_entry)

            # Phase 3: Recall per chunk
            combined_text = " ".join(pp.normalized_text for pp in preprocessed_texts)
            recall_preprocessed = await self._preprocess_text(
                combined_text,
                source_ref=SourceRef(source_type="message", is_parsed=True),
                segment_id=f"chunk{chunk.chunk_index}_recall",
            )
            recall_result = await self._related_memory_recall.recall(
                context, recall_preprocessed, active_memories=active_memories
            )

            recalled_memories: list[dict[str, Any]] = []
            if recall_result is not None:
                if recall_result.duplicate is not None:
                    recalled_memories.append(
                        {
                            "memory_id": recall_result.duplicate.memory_id,
                            "content": recall_result.duplicate.memory.content if recall_result.duplicate.memory else "",
                            "source": recall_result.duplicate.source,
                            "is_exact_duplicate": True,
                        }
                    )
                for candidate in recall_result.candidates:
                    if candidate.source == "hash":
                        continue
                    recalled_memories.append(
                        {
                            "memory_id": candidate.memory_id,
                            "content": candidate.memory.content if candidate.memory else "",
                            "source": candidate.source,
                            "score": candidate.score,
                        }
                    )

            envelope = ExtractionEnvelope(
                extractable_messages=extractable_refs,
                current_context_messages=current_context_refs,
                history=history_pack,
                recalled_memories=recalled_memories,
                boundary=chunk.boundary,
                chunk_index=chunk.chunk_index,
            )

            # Phase 4: Extract from envelope
            extraction = await self._memory_extractor.extract_from_envelope(
                envelope,
                preprocessed_texts,
                context,
            )
            raw_candidates = extraction.memories
            extracted_entities = extraction.entities

            raw_candidates, extracted_entities = _prefix_segment_ref_ids(
                raw_candidates,
                extracted_entities,
                chunk.chunk_index,
            )

            # Build source-ref-id → message context mapping for this chunk
            source_context_by_ref_id: dict[str, _MessageExtractionContext] = {}
            source_offset_by_original_index: dict[int, int] = {}
            for evidence_idx, ctx_entry in message_context_by_index.items():
                prefixed_id = f"s{chunk.chunk_index}_s{evidence_idx}"
                source_context_by_ref_id[prefixed_id] = ctx_entry
                source_contexts_by_ref_id[prefixed_id] = ctx_entry
            for src_candidate in extraction.sources:
                prefixed_id = f"s{chunk.chunk_index}_{src_candidate.ref_id}"
                msg_idx = src_candidate.message_index
                ctx_entry = None
                evidence_idx = src_candidate.metadata.get("evidence_index")
                if isinstance(evidence_idx, int):
                    ctx_entry = message_context_by_index.get(evidence_idx)
                elif isinstance(evidence_idx, str) and evidence_idx.isdigit():
                    ctx_entry = message_context_by_index.get(int(evidence_idx))

                if ctx_entry is None and msg_idx is not None:
                    original_contexts = original_message_contexts_by_index.get(msg_idx, [])
                    if len(original_contexts) > 1:
                        offset = source_offset_by_original_index.get(msg_idx, 0)
                        ctx_entry = original_contexts[min(offset, len(original_contexts) - 1)]
                        source_offset_by_original_index[msg_idx] = offset + 1
                    else:
                        ctx_entry = message_context_by_index.get(msg_idx)
                        if ctx_entry is None and original_contexts:
                            ctx_entry = original_contexts[0]
                if ctx_entry is not None:
                    source_context_by_ref_id[prefixed_id] = ctx_entry
                    source_contexts_by_ref_id[prefixed_id] = ctx_entry

            chunk_candidate_data.append(
                (
                    raw_candidates,
                    extracted_entities,
                    recall_result,
                    source_context_by_ref_id,
                    message_context_by_index,
                )
            )
            all_raw_candidates.extend(raw_candidates)

            prev_pack = history_pack
            prev_chunk = chunk

        # Batch-wide dedup across all chunks
        deduped_candidates = self.dedup_candidates(all_raw_candidates)
        candidate_by_ref_id = {c.ref_id: c for c in deduped_candidates}

        # Phase 5-6: Plan + Vectorize per surviving candidate
        for (
            raw_candidates,
            extracted_entities,
            seg_recall_result,
            source_context_by_ref_id,
            message_context_by_index,
        ) in chunk_candidate_data:
            surviving = [c for c in raw_candidates if c.ref_id in candidate_by_ref_id]
            if not surviving:
                continue

            for candidate in surviving:
                candidate = candidate_by_ref_id.get(candidate.ref_id, candidate)

                # Resolve per-candidate source context from source_refs
                primary_context = _resolve_candidate_source(
                    candidate,
                    source_context_by_ref_id,
                    message_context_by_index,
                )
                if primary_context is None and candidate.source_refs:
                    primary_context = _resolve_candidate_source(
                        candidate,
                        source_contexts_by_ref_id,
                        {},
                    )
                if primary_context is None:
                    logger.warning(
                        "candidate_source_unresolvable",
                        ref_id=candidate.ref_id,
                        source_refs=candidate.source_refs,
                    )
                    continue

                source_ref = primary_context.source_ref
                segment = primary_context.segment
                preprocessed = primary_context.preprocessed

                # Collect all source contexts for multi-source edges
                all_source_contexts: list[_MessageExtractionContext] = []
                seen_source_ids: set[str] = set()
                for ref_id in candidate.source_refs:
                    ctx = source_contexts_by_ref_id.get(ref_id)
                    if ctx is not None and ctx.source_ref.source_id not in seen_source_ids:
                        all_source_contexts.append(ctx)
                        seen_source_ids.add(ctx.source_ref.source_id)
                if not all_source_contexts:
                    all_source_contexts = [primary_context]

                if candidate.content == preprocessed.normalized_text:
                    candidate_preprocessed = preprocessed
                else:
                    candidate_preprocessed = await self._preprocess_text(
                        candidate.content,
                        source_ref=source_ref,
                        segment_id=candidate.segment_id or preprocessed.segment_id,
                    )
                event_timestamp = _segment_event_timestamp(segment, inp)
                event_time = _datetime_from_millis(event_timestamp)
                known_memory_ids = _recall_memory_ids(seg_recall_result)
                target_memory_id = (
                    candidate.target_memory_id if candidate.target_memory_id in known_memory_ids else None
                )
                related_memory_ids = _filter_known_memory_ids(candidate.related_memory_ids, known_memory_ids)

                planned = self.plan(
                    candidate_preprocessed,
                    candidate.mem_type,
                    action_hint=candidate.action_hint,
                    confidence=candidate.confidence,
                    target_memory_id=target_memory_id,
                    related_memory_ids=related_memory_ids,
                )

                # Fallback path: when LLM was unavailable, the fallback extractor
                # outputs action_hint="add" with no target_memory_id. Use chunk
                # recall to detect hash-duplicates and upgrade to REINFORCE.
                if planned.action == "ADD" and candidate.metadata.get("extractor", "").startswith("fallback"):
                    if seg_recall_result is not None and seg_recall_result.duplicate is not None:
                        planned = PlannedAddAction(
                            action="REINFORCE",
                            content=planned.content,
                            mem_type=planned.mem_type,
                            reason="content_hash_duplicate_fallback",
                            target_memory_id=seg_recall_result.duplicate.memory_id,
                            related_memory_ids=[seg_recall_result.duplicate.memory_id],
                            metadata={"duplicate_source": seg_recall_result.duplicate.source},
                            extractor_action_hint="add",
                        )
                    elif seg_recall_result is not None and not planned.related_memory_ids:
                        related = [c.memory_id for c in seg_recall_result.candidates if c.source != "hash"]
                        if related:
                            planned = planned.model_copy(update={"related_memory_ids": related})

                if planned.action == "SKIP":
                    continue

                if planned.action == "ADD":
                    memory_id = generate_memory_id(
                        context.project_id,
                        context.request_id,
                        candidate_preprocessed.content_hash,
                    )
                    unique_entities = (
                        deduplicate_entities(
                            resolve_candidate_entities(candidate, extracted_entities, candidate_preprocessed.entities)
                        )
                        if enable_entities
                        else []
                    )
                    memory_metadata: dict = {
                        **dict(inp.metadata),
                        **dict(candidate.metadata),
                        **_source_time_metadata(segment),
                        "content_hash": candidate_preprocessed.content_hash,
                        "bm25_text": candidate_preprocessed.bm25_text,
                        "tokens": list(candidate_preprocessed.tokens),
                        "lang": candidate_preprocessed.lang,
                        "source_id": source_ref.source_id,
                        "source_type": source_ref.source_type,
                        "source_message_index": segment.message_index,
                        "source_timestamp_ms": event_timestamp,
                        "source_role": segment.role,
                        "chunk_index": segment.segment_id,
                        "entity_count": len(unique_entities),
                        "entities": entity_names(unique_entities),
                        "related_memory_ids": list(planned.related_memory_ids),
                        "extractor": candidate.metadata.get("extractor", "vanilla_llm_chunked"),
                        "extractor_confidence": candidate.confidence,
                        "extractor_importance": candidate.importance,
                        "extractor_reason": candidate.reason,
                        "planner_action": planned.action,
                        "planner_reason": planned.reason,
                    }

                    memory = MemoryWrite(
                        memory_id=memory_id,
                        account_id=context.account_id,
                        project_id=context.project_id,
                        api_key_uuid=context.api_key_uuid,
                        user_id=context.user_id,
                        app_id=context.app_id,
                        session_id=context.session_id,
                        agent_id=context.agent_id,
                        request_id=context.request_id,
                        content=planned.content,
                        mem_type=planned.mem_type or "fact",
                        mem_extract_type="vanilla",
                        mem_extract_version="default_add_v1_chunked",
                        metadata=memory_metadata,
                        validate_from=event_time,
                        created_at=now,
                        root_id=[memory_id],
                    )
                    memories.append(memory)

                    pending_memory_vectors.append(
                        _PendingMemoryVector(
                            memory=memory,
                            preprocessed=candidate_preprocessed,
                            content=planned.content,
                        )
                    )

                    memory_relationships = []
                    for ctx in all_source_contexts:
                        memory_relationships.append(
                            build_extracted_from_edge(memory_id, ctx.source_ref, context, ctx.segment)
                        )
                    if enable_entities:
                        source_by_evidence = {i: ctx.source_ref for i, ctx in message_context_by_index.items()}
                        for entity in unique_entities:
                            eid = generate_entity_id(context.project_id, entity)
                            entity_write = entities_by_id.setdefault(eid, build_entity_write(entity, eid, context, now))
                            _attach_search_fields(entity_write, [memory.content])
                            memory_relationships.append(build_mentions_edge(memory_id, eid, entity, context))
                            for entity_source in _entity_evidence_source_refs(entity, source_by_evidence, source_ref):
                                memory_relationships.append(
                                    build_mentioned_in_source_edge(eid, entity_source, entity, context)
                                )
                    for related_memory_id in planned.related_memory_ids:
                        memory_relationships.append(build_relates_to_edge(memory_id, related_memory_id, context))
                    relationships.extend(memory_relationships)

                    events.append(
                        MemoryAddEventItem(
                            operation="add",
                            content=memory.content,
                            memory_id=memory.memory_id,
                            mem_type=memory.mem_type,
                            confidence=candidate.confidence,
                            related_memory_ids=list(planned.related_memory_ids),
                            graph_edge_count=len(memory_relationships),
                        )
                    )

                elif planned.action == "UPDATE":
                    if planned.target_memory_id is not None:
                        updated_preprocessed = await self._preprocess_text(
                            planned.content,
                            source_ref=source_ref,
                            segment_id=segment.segment_id,
                        )
                        updated_vw, updated_vector_pending = await self.vectorize(
                            planned.target_memory_id,
                            updated_preprocessed,
                            planned.content,
                            consistency,
                        )
                        metadata_refresh = {
                            "content_hash": updated_preprocessed.content_hash,
                            "bm25_text": updated_preprocessed.bm25_text,
                            "tokens": list(updated_preprocessed.tokens),
                            "lang": updated_preprocessed.lang,
                            "entity_count": len(updated_preprocessed.entities),
                            "entities": [e.name for e in updated_preprocessed.entities],
                        }
                        if updated_vector_pending:
                            metadata_refresh["vector_pending"] = True
                        update_commands.append(
                            build_update_command(
                                planned.target_memory_id,
                                planned.content,
                                context,
                                now,
                                dense_vector=updated_vw.semantic_vector,
                                sparse_vectors={
                                    "bm25_indices": updated_vw.bm25_indices,
                                    "bm25_values": updated_vw.bm25_values,
                                }
                                if updated_vw.bm25_indices is not None
                                else None,
                                metadata_refresh=metadata_refresh,
                                consistency=consistency,
                            )
                        )
                    events.append(
                        MemoryAddEventItem(
                            operation="update",
                            content=planned.content,
                            memory_id=planned.target_memory_id,
                            mem_type=planned.mem_type,
                            related_memory_ids=list(planned.related_memory_ids),
                        )
                    )

                elif planned.action == "MERGE":
                    merge_memory_id = generate_memory_id(
                        context.project_id,
                        context.request_id,
                        candidate_preprocessed.content_hash,
                    )
                    merge_entities = (
                        deduplicate_entities(
                            resolve_candidate_entities(candidate, extracted_entities, candidate_preprocessed.entities)
                        )
                        if enable_entities
                        else []
                    )
                    merge_metadata: dict = {
                        **dict(inp.metadata),
                        **dict(candidate.metadata),
                        **_source_time_metadata(segment),
                        "content_hash": candidate_preprocessed.content_hash,
                        "bm25_text": candidate_preprocessed.bm25_text,
                        "tokens": list(candidate_preprocessed.tokens),
                        "lang": candidate_preprocessed.lang,
                        "source_id": source_ref.source_id,
                        "source_type": source_ref.source_type,
                        "source_message_index": segment.message_index,
                        "source_timestamp_ms": event_timestamp,
                        "source_role": segment.role,
                        "chunk_index": segment.segment_id,
                        "entity_count": len(merge_entities),
                        "entities": entity_names(merge_entities),
                        "related_memory_ids": list(planned.related_memory_ids),
                        "extractor": candidate.metadata.get("extractor", "vanilla_llm_chunked"),
                        "extractor_confidence": candidate.confidence,
                        "extractor_importance": candidate.importance,
                        "extractor_reason": candidate.reason,
                        "planner_action": planned.action,
                        "planner_reason": planned.reason,
                        "merged_from": list(planned.related_memory_ids),
                    }
                    merge_memory = MemoryWrite(
                        memory_id=merge_memory_id,
                        account_id=context.account_id,
                        project_id=context.project_id,
                        api_key_uuid=context.api_key_uuid,
                        user_id=context.user_id,
                        app_id=context.app_id,
                        session_id=context.session_id,
                        agent_id=context.agent_id,
                        request_id=context.request_id,
                        content=planned.content,
                        mem_type=planned.mem_type or "fact",
                        mem_extract_type="vanilla",
                        mem_extract_version="default_add_v1_chunked",
                        metadata=merge_metadata,
                        validate_from=event_time,
                        created_at=now,
                        root_id=list(planned.related_memory_ids),
                    )
                    memories.append(merge_memory)
                    pending_memory_vectors.append(
                        _PendingMemoryVector(
                            memory=merge_memory,
                            preprocessed=candidate_preprocessed,
                            content=planned.content,
                        )
                    )
                    merge_relationships = []
                    for ctx in all_source_contexts:
                        merge_relationships.append(
                            build_extracted_from_edge(merge_memory_id, ctx.source_ref, context, ctx.segment)
                        )
                    for related_memory_id in planned.related_memory_ids:
                        merge_relationships.append(build_relates_to_edge(merge_memory_id, related_memory_id, context))
                    if enable_entities:
                        source_by_evidence = {i: ctx.source_ref for i, ctx in message_context_by_index.items()}
                        for entity in merge_entities:
                            eid = generate_entity_id(context.project_id, entity)
                            entity_write = entities_by_id.setdefault(eid, build_entity_write(entity, eid, context, now))
                            _attach_search_fields(entity_write, [merge_memory.content])
                            merge_relationships.append(build_mentions_edge(merge_memory_id, eid, entity, context))
                            for entity_source in _entity_evidence_source_refs(entity, source_by_evidence, source_ref):
                                merge_relationships.append(
                                    build_mentioned_in_source_edge(eid, entity_source, entity, context)
                                )
                    relationships.extend(merge_relationships)
                    update_commands.extend(
                        build_merge_archive_commands(planned.related_memory_ids, context, now, consistency=consistency)
                    )
                    events.append(
                        MemoryAddEventItem(
                            operation="merge",
                            content=planned.content,
                            memory_id=merge_memory_id,
                            mem_type=planned.mem_type,
                            related_memory_ids=list(planned.related_memory_ids),
                            graph_edge_count=len(merge_relationships),
                        )
                    )

                elif planned.action == "REINFORCE" and planned.target_memory_id is not None:
                    update_commands.append(
                        build_reinforcement_command(planned.target_memory_id, context, now, consistency=consistency)
                    )
                    reinforce_relationships = []
                    for ctx in all_source_contexts:
                        reinforce_relationships.append(
                            build_extracted_from_edge(planned.target_memory_id, ctx.source_ref, context, ctx.segment)
                        )
                    relationships.extend(reinforce_relationships)
                    events.append(
                        MemoryAddEventItem(
                            operation="reinforcement",
                            content=planned.content,
                            memory_id=planned.target_memory_id,
                            mem_type=planned.mem_type,
                            related_memory_ids=list(planned.related_memory_ids),
                            graph_edge_count=len(reinforce_relationships),
                        )
                    )

        memory_vectors: list[VectorWrite] = []
        entity_vectors: list[EntityVectorWrite] = []
        memory_vector_items = [
            (pending.memory.memory_id, pending.preprocessed, pending.content) for pending in pending_memory_vectors
        ]
        entity_values = list(entities_by_id.values()) if enable_entities and entities_by_id else []

        memory_vector_pending: list[bool] = []
        entity_vector_pending = False
        if memory_vector_items and entity_values:
            memory_task = asyncio.create_task(self.vectorize_many(memory_vector_items, consistency))
            entity_task = asyncio.create_task(self._vectorize_entities(entity_values, memories, consistency))
            try:
                (memory_vectors, memory_vector_pending), (entity_vectors, entity_vector_pending) = await asyncio.gather(
                    memory_task,
                    entity_task,
                )
            except Exception:
                for task in (memory_task, entity_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(memory_task, entity_task, return_exceptions=True)
                raise
        elif memory_vector_items:
            memory_vectors, memory_vector_pending = await self.vectorize_many(memory_vector_items, consistency)
        elif entity_values:
            entity_vectors, entity_vector_pending = await self._vectorize_entities(
                entity_values,
                memories,
                consistency,
            )

        vectors.extend(memory_vectors)
        for pending, is_pending in zip(pending_memory_vectors, memory_vector_pending, strict=True):
            if is_pending:
                pending.memory.metadata["vector_pending"] = True

        if entity_values:
            if entity_vector_pending:
                for entity in entities_by_id.values():
                    entity.metadata = {**dict(entity.metadata), "vector_pending": True}

        return (
            MemoryDbWritePlan(
                memories=memories,
                entities=list(entities_by_id.values()),
                sources=list(sources_by_id.values()),
                vectors=vectors,
                entity_vectors=entity_vectors,
                relationships=relationships,
            ),
            events,
            update_commands,
        )


def _ref_to_segment(msg_ref: TurnMessageRef, inp: AddPipelineInput) -> SourceAwareSegment:
    """Build a synthetic SourceAwareSegment from a TurnMessageRef.

    This bridges chunk-level messages back to the segment-based Phase 5-6
    code so that metadata, graph edges, and vectorization work unchanged.
    """
    source_ref = SourceRef(
        source_type="message",
        message_id=f"message-{msg_ref.message_index}",
        is_parsed=True,
        metadata=_message_source_metadata(msg_ref),
    )
    return SourceAwareSegment(
        segment_id=f"chunk-msg-{msg_ref.message_index}",
        text=msg_ref.text,
        source_ref=source_ref,
        message_index=msg_ref.message_index,
        role=msg_ref.role,
        timestamp=msg_ref.timestamp,
        end_offset=len(msg_ref.text),
        metadata={
            "message_type": "chunked",
            "raw_role": msg_ref.raw_role,
            "speaker": msg_ref.speaker,
        },
    )
