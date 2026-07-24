"""Algorithm-level DTOs shared by memory components and pipelines.

This module hosts intermediate types produced and consumed *inside* the
algorithmic layer (text preprocessing, BM25, sparse encoding, entity
extraction, language detection). Cross-layer business DTOs live in
``typing.memory``; pipeline I/O contracts live in ``typing.service``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .service import MemorySearchItem

LanguageCode = Literal["zh", "en", "mixed", "unknown"]
SparseFallbackMode = Literal["tf", "log_tf"]


class LanguageResult(BaseModel):
    """Language detection result for a single text.

    Purpose: Carries the language chosen by the lightweight detector plus the
    ratios used to explain that decision.
    Used in: TextPreprocessor, Bm25TextAnalyzer routing, EntityExtractor
    routing, and debug traces.
    """

    lang: LanguageCode = Field(description="Detected language code.")
    confidence: float = Field(description="Detector confidence, usually derived from character ratios.")
    zh_ratio: float = Field(default=0.0, description="Ratio of CJK characters among counted characters.")
    latin_ratio: float = Field(default=0.0, description="Ratio of Latin letters among counted characters.")


class BM25TokenizationResult(BaseModel):
    """BM25-specific lexical analysis result.

    Purpose: Carries terms used for BM25 or keyword retrieval without
    implying LLM tokenization semantics.
    Used in: TextPreprocessor, CorpusStatsProvider, SparseVectorEncoder,
    add/update writes, and BM25 search.
    """

    lang: str = Field(description="Language strategy used by the analyzer.")
    terms: list[str] = Field(default_factory=list, description="BM25 terms. Repeated terms are preserved for TF.")
    bm25_text: str = Field(description="Space-joined BM25 terms for payload storage and debugging.")
    term_count: int = Field(description="Total number of BM25 terms, equal to len(terms).")


class SparseVector(BaseModel):
    """Storage-agnostic sparse vector.

    Purpose: Represents the sparse indices and weights produced by the BM25
    encoder before they are mapped to Qdrant models.
    Used in: SparseVectorEncoder output, DB mapper input, BM25 search, and
    reindex jobs.
    """

    indices: list[int] = Field(default_factory=list, description="Sorted sparse vector indices.")
    values: list[float] = Field(default_factory=list, description="Sparse vector values aligned with indices.")
    model: str = Field(description="Sparse encoder model identifier from config.")
    hash_dim: int = Field(description="Hash trick dimension used to create indices.")


class CorpusStats(BaseModel):
    """Corpus statistics required for BM25 weighting.

    Purpose: Provides project-scoped document count, average document length,
    and document frequency values for the query or document terms being
    encoded.
    Used in: SparseVectorEncoder.encode_document(), SparseVectorEncoder.
    encode_query(), PersistentCorpusStatsProvider, and reindex jobs.
    """

    project_id: str = Field(description="Project ID that owns this corpus statistics snapshot.")
    doc_count: int = Field(default=0, description="Number of documents observed for the project.")
    avg_doc_len: float = Field(default=0.0, description="Average BM25 term count per document.")
    document_frequency: dict[str, int] = Field(
        default_factory=dict,
        description="Document frequency for the requested terms.",
    )


class SupplementalSearchQuery(BaseModel):
    """Supplemental query rewritten from an operation record query.

    Purpose: Carries the query used to recall additional memories for implicit
    feedback analysis.
    Used in: Feedback query rewriting components and implicit feedback
    collection.
    """

    query: str = Field(description="Query rewritten for supplemental memory recall.")


ImplicitFeedbackCategory = Literal["task_temporary", "scenario_specific", "long_term"]


class ImplicitFeedbackSignal(BaseModel):
    """One implicit feedback signal detected in a compact conversation round.

    Purpose: Identifies one implicit feedback signal in a compact conversation
    round that may affect memory, with a category used by later action
    planning. Multiple independent signals may point to the same round. The
    full round context is passed separately to the action planner; this DTO
    should not carry rewritten conversation content.
    Used in: Implicit feedback signal detection and later feedback planning.
    """

    round_index: int = Field(description="Zero-based round index in the collected session material.")
    category: ImplicitFeedbackCategory = Field(
        description=(
            "Feedback category: task_temporary for current-task-only feedback, "
            "scenario_specific for feedback bound to the current task type or scene, "
            "or long_term for general durable memory."
        )
    )
    reason: str | None = Field(default=None, description="Short reason why this round is considered feedback.")


class ImplicitFeedbackSignalResult(BaseModel):
    """Detected implicit feedback signals for one session.

    Purpose: Groups negative feedback signals produced by the detector.
    Used in: Implicit feedback pipeline orchestration.
    """

    signals: list[ImplicitFeedbackSignal] = Field(
        default_factory=list,
        description="Rounds that contain negative feedback signals.",
    )


class ImplicitFeedbackRound(BaseModel):
    """Compact add-record conversation round.

    Purpose: Holds the initial user query and final assistant response for one
    add-record round, without tool-call traces.
    Used in: Implicit feedback material collection and signal detection.
    """

    messages: list[dict] = Field(default_factory=list, description="Compact round messages.")


class ImplicitFeedbackSessionMaterial(BaseModel):
    """Session-level material collected for implicit feedback.

    Purpose: Carries compact conversation rounds and a deduplicated candidate
    memory pool.
    Used in: Implicit feedback pipeline orchestration.
    """

    session_id: str = Field(description="Session ID represented by this material.")
    rounds: list[ImplicitFeedbackRound] = Field(default_factory=list, description="Compact conversation rounds.")
    messages: list[dict] = Field(default_factory=list, description="Flattened compact messages.")
    memories: list[MemorySearchItem] = Field(default_factory=list, description="Deduplicated memory candidates.")
    source_add_record_ids: list[str] = Field(
        default_factory=list,
        description="Add record IDs used only for marking implicit feedback processing progress.",
    )


# Trace classification & extraction background

TraceKind = Literal["dialogue", "agent_trace", "skill_trace", "file_or_url", "mixed"]
TraceCompleteness = Literal["complete", "partial", "unknown"]


class TraceClassification(BaseModel):
    """Purpose: Classify the current add request's input trace type.

    Used in: AddCoreBuilder background construction and extraction context
    routing. Carries the classification result with confidence and reason
    for downstream consumption.
    """

    kind: TraceKind = Field(default="dialogue", description="Detected trace kind.")
    completeness: TraceCompleteness = Field(default="partial", description="Trace completeness assessment.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Classification confidence [0, 1].")
    reason: str = Field(default="unknown", description="Why this classification was chosen.")


class ExtractionBackground(BaseModel):
    """Purpose: Carry history context and trace classification for memory extraction.

    Used in: AddCoreBuilder extraction phase. Passed to VanillaMemoryExtractor
    so the LLM prompt can include relevant history, user query, and trace
    context. Safe to construct with all-defaults; no field is required.
    """

    trace_kind: TraceKind = Field(default="dialogue")
    trace_completeness: TraceCompleteness = Field(default="partial")
    classification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    classification_reason: str = Field(default="unknown")
    recent_dialogue: list[dict[str, str]] = Field(
        default_factory=list,
        description="Up to 10 recent dialogue turns with role and content.",
    )
    full_agent_trace: list[dict[str, str]] | None = Field(
        default=None,
        description="Full agent trace when trace_kind is agent_trace.",
    )
    skill_trace: list[dict[str, str]] | None = Field(
        default=None,
        description="Full skill trace when trace_kind is skill_trace.",
    )
    related_memory_seed: list[str] = Field(
        default_factory=list,
        description="Memory IDs used to seed related-memory recall.",
    )
    user_intent_summary: str = Field(
        default="",
        description="Summary of user intent passed to LLM extraction prompt as context.",
    )
    user_query: str = Field(
        default="",
        description="Extracted root user query from the trace.",
    )
    session_summary: str = Field(
        default="",
        description="Optional session summary for extraction context.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings encountered during background construction.",
    )


class ConsolidationCreate(BaseModel):
    """A business-level memory creation proposed by dreaming consolidation.

    Purpose: Carries a create action emitted by the consolidation prompt
    without exposing storage primitives.
    Used in: Dreaming consolidation parser and DefaultDreamingPipeline action
    application.
    """

    content: str
    mem_type: str = "fact"
    entity_id: str | None = None
    entity_type: str | None = None
    property_name: str | None = None
    root_id: list[str] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)
    evidence_memory_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsolidationUpdate(BaseModel):
    """A memory content, quality signal, or metadata update from consolidation.

    Purpose: Carries a patch action emitted by the consolidation prompt.
    Used in: Dreaming consolidation parser and DefaultDreamingPipeline action
    application.
    """

    memory_id: str
    content: str | None = None
    quality_signal: Literal["reinforce", "low_value", "conflict", "stale", "canonical", "ambiguous"] | None = None
    reinforcement_count: int | None = None
    metadata_patch: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ConsolidationMerge(BaseModel):
    """A merge operation: archive source memories and create a consolidated target.

    Purpose: Carries a merge action emitted by the consolidation prompt.
    Used in: Dreaming consolidation parser and DefaultDreamingPipeline action
    application.
    """

    source_memory_ids: list[str] = Field(default_factory=list)
    target_content: str
    target_entity_id: str | None = None
    target_entity_type: str | None = None
    target_property_name: str | None = None
    target_root_id: list[str] = Field(default_factory=list)
    merge_reason: str = ""


class ConsolidationArchive(BaseModel):
    """A memory to archive with replacement lineage.

    Purpose: Carries an archive action emitted by the consolidation prompt.
    Used in: Dreaming consolidation parser and DefaultDreamingPipeline action
    application.
    """

    memory_id: str
    reason: str = ""
    replacement_memory_id: str | None = None


class ConsolidationLink(BaseModel):
    """A graph edge intent between existing memories/entities in the cluster.

    Purpose: Carries a graph link action emitted by the consolidation prompt.
    Used in: Dreaming consolidation parser and DefaultDreamingPipeline action
    application. Links must not reference memories created by the same action
    plan; the pipeline creates evidence/timeline edges for new memories.
    """

    source_kind: Literal["Memory", "Entity"]
    source_id: str
    target_kind: Literal["Memory", "Entity"]
    target_id: str
    relation_type: str = "related"
    property_name: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsolidationAction(BaseModel):
    """Structured business intent from LLM consolidation decisions.

    Purpose: Groups all consolidation mutations returned by the prompt.
    Used in: Dreaming consolidation parser and DefaultDreamingPipeline action
    application.
    """

    creates: list[ConsolidationCreate] = Field(default_factory=list)
    updates: list[ConsolidationUpdate] = Field(default_factory=list)
    merges: list[ConsolidationMerge] = Field(default_factory=list)
    archives: list[ConsolidationArchive] = Field(default_factory=list)
    links: list[ConsolidationLink] = Field(default_factory=list)


# Turn grouping, chunk planning, history packing, extraction envelope

TurnBoundary = Literal["complete", "open_head", "open_tail", "orphan"]
ChunkBoundary = Literal["complete", "open_head", "open_tail", "orphan", "compacted"]


class TurnMessageRef(BaseModel):
    """Reference to a single message within a turn.

    Purpose: Carry the text, role, timestamp, and extractability flag for one
    message inside a Turn. System messages are marked non-extractable.
    Used in: Turn construction by TurnGrouper and downstream chunk components.
    """

    text: str = Field(description="Message content.")
    role: str = Field(description="Normalized message role: user, assistant, system, tool, or speaker.")
    raw_role: str | None = Field(default=None, description="Original message role before normalization.")
    speaker: str | None = Field(default=None, description="Speaker identity for arbitrary named-speaker dialogue.")
    timestamp: int | None = Field(default=None, description="Millisecond timestamp if available.")
    message_index: int = Field(description="Original index in AddPipelineInput.messages.")
    is_extractable: bool = Field(
        default=True,
        description="Whether this message can produce memory candidates. System messages are False.",
    )


class Turn(BaseModel):
    """Purpose: A semantically grouped set of messages forming one conversational unit.

    Used in: TurnGrouper output, ChunkPlanner input, history packing, and
    extraction envelope construction. A turn represents one user intent and
    the assistant response(s) associated with it.
    """

    messages: list[TurnMessageRef] = Field(description="Ordered messages in this turn.")
    boundary: TurnBoundary = Field(description="Boundary type: complete, open_head, open_tail, or orphan.")
    token_count: int = Field(default=0, description="Total token count across all messages in this turn.")

    @property
    def is_compacted(self) -> bool:
        """Whether this turn has been compacted (head+summary+tail)."""
        return hasattr(self, "_compaction_result") and self._compaction_result is not None

    @property
    def extractable_messages(self) -> list[TurnMessageRef]:
        """Messages that are extractable evidence (excludes system messages)."""
        return [m for m in self.messages if m.is_extractable]

    @property
    def text(self) -> str:
        """Concatenated text of all extractable messages."""
        return "\n".join(m.text for m in self.extractable_messages)


class Chunk(BaseModel):
    """Purpose: A token-budgeted group of turns sent to LLM extraction as one unit.

    Used in: ChunkPlanner output, extraction loop, and history packing. One
    chunk produces one LLM extraction call.
    """

    turns: list[Turn] = Field(description="Ordered turns packed into this chunk.")
    boundary: ChunkBoundary = Field(
        description="Derived boundary: if any turn is open_head, chunk is open_head; "
        "if last turn is open_tail, chunk is open_tail; all complete = complete.",
    )
    token_count: int = Field(default=0, description="Total token count across all turns.")
    chunk_index: int = Field(default=0, description="Zero-based chunk index in the plan.")
    needs_compaction: bool = Field(
        default=False,
        description="True when a single turn exceeds hard turn budget and requires compaction.",
    )
    compacted_turn_indices: list[int] = Field(
        default_factory=list,
        description="Indices into self.turns for turns that have been compacted.",
    )


class HistoryPack(BaseModel):
    """Purpose: Sliding context window carried between chunks.

    Used in: History packer output and extraction envelope construction. For
    chunk 0, external history may be included. For later chunks, only sliding
    in-request history is available.
    """

    external_history: list[Turn] = Field(
        default_factory=list,
        description="External DB/add-record history. Only used for chunk 0.",
    )
    in_request_history: list[Turn] = Field(
        default_factory=list,
        description="Packed turns from prior chunks (complete-turn backward packing).",
    )
    token_usage: int = Field(default=0, description="Total tokens used by history.")


class ExtractionEnvelope(BaseModel):
    """Purpose: Structured context for one LLM extraction call.

    Used in: VanillaMemoryExtractor input. Separates extractable evidence from
    non-extractable context so the LLM knows the boundary.
    """

    extractable_messages: list[TurnMessageRef] = Field(
        description="Primary evidence: current chunk messages that can produce memory candidates.",
    )
    current_context_messages: list[TurnMessageRef] = Field(
        default_factory=list,
        description="Non-extractable context within the current chunk (e.g. compaction summaries). "
        "Visible to the LLM for reference but must not produce memory candidates.",
    )
    history: HistoryPack = Field(
        default_factory=HistoryPack,
        description="Non-extractable context: history + prior events.",
    )
    recalled_memories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Non-extractable: related memory recall results for dedup/action decisions.",
    )
    boundary: ChunkBoundary = Field(
        description="Chunk boundary metadata for extraction conservatism.",
    )
    chunk_index: int = Field(default=0, description="Zero-based chunk index for logging.")


class TurnCompactionSummary(BaseModel):
    """Purpose: Structured output from the long-turn compaction summary prompt.

    Used in: LongTurnCompactor output and TurnCompactionResult assembly. The
    summary preserves context needed for later extraction without creating
    memories itself.
    """

    general_summary: str = Field(default="", description="Overall summary of the middle section.")
    key_entities: list[str] = Field(default_factory=list, description="Named entities mentioned in the middle.")
    user_intent: str = Field(default="", description="User intent preserved from the middle section.")
    confirmed_facts: list[str] = Field(default_factory=list, description="Facts confirmed in the middle.")
    decisions: list[str] = Field(default_factory=list, description="Decisions made in the middle.")
    open_questions: list[str] = Field(default_factory=list, description="Unresolved questions from the middle.")
    warnings: list[str] = Field(default_factory=list, description="Potential issues or ambiguities.")


class TurnCompactionResult(BaseModel):
    """Purpose: Result of compacting an oversized turn.

    Used in: LongTurnCompactor output. Carries head (raw evidence), middle
    (structured summary), and tail (raw evidence). Head and tail remain
    extractable; middle is non-extractable context.
    """

    head_text: str = Field(description="Raw head section preserved as evidence.")
    head_tokens: int = Field(description="Token count of head section.")
    tail_text: str = Field(description="Raw tail section preserved as evidence.")
    tail_tokens: int = Field(description="Token count of tail section.")
    middle_summary: TurnCompactionSummary = Field(
        default_factory=TurnCompactionSummary,
        description="Structured summary of the compacted middle.",
    )
    original_token_count: int = Field(description="Token count of the original turn before compaction.")
    is_lossy: bool = Field(default=True, description="Whether compaction lost information (always true for v1).")
