"""Shared memory DTOs for business objects and storage-bound write primitives.

These models define business contracts only. They do not import database
clients, build Qdrant points, emit Cypher, or call LLM/embedding services.
Conversions between these DTOs and low-level database primitives live in
``mindmemos.mappers``.

Layered roles inside this module:

* Input message shapes consumed at the pipeline boundary
  (``FileMessage`` / ``TextMessage`` / ``UrlMessage`` / ``DialogueMessage``)
  and per-request context (``MemoryRequestContext``, ``SourceRef``).
* Extraction intermediates shared by ``components.text`` and add/search
  preprocessing (``Entity`` / ``PreprocessedText``).
* Internal DB filter trees (``SearchFilter`` / ``FieldCondition``).
  The public filter DSL is parsed into ``SearchFilter`` by
  ``mappers.api.parse_search_dsl`` against ``DSL_FILTERABLE_MEMORY_FIELDS``.
* Write DTOs consumed by ``typing.memory_db.MemoryDbWritePlan`` and the mapper layer
  (``MemoryWrite`` / ``EntityWrite`` / ``SourceWrite`` / ``VectorWrite`` /
  ``EntityVectorWrite`` / graph relationship).
* DB-layer operation DTOs prefixed with ``MemoryDb*`` live in
  ``mindmemos.typing.memory_db``. Keep them out of this module so the business
  memory contract does not absorb storage operation semantics.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AddMode = Literal["sync", "async"]
ConsistencyMode = Literal["fast", "strong"]
GraphNodeKind = Literal["Memory", "Entity", "Source"]
GraphDirection = Literal["out", "in", "both"]
GraphNeighborSource = Literal["shared_entity", "direct_memory_relation"]
MemoryStatus = Literal["active", "archived", "delete"]
MemoryRelationType = Literal["RELATES_TO", "RELATED_TO"]
MemoryType = Literal["profile", "fact", "experience", "episodic", "tool_trace", "skill_candidate", "file_knowledge"]
SearchMode = Literal["semantic", "bm25", "rrf", "graph", "hybrid"]
RankingMode = Literal["none", "score", "hybrid"]
MemoryOperation = Literal["add", "delete", "update", "reinforcement", "merge"]

# Graph relationship type names (Neo4j edge labels). Centralized here so that
# mappers, pipelines, and services share one source of truth instead of
# hardcoding string literals.
REL_HAS_PROPERTY_MEMORY = "HAS_PROPERTY_MEMORY"
REL_NEXT_IN_PROPERTY_TIMELINE = "NEXT_IN_PROPERTY_TIMELINE"
REL_RELATES_TO = "RELATES_TO"
REL_RELATED_TO = "RELATED_TO"
REL_MENTIONS = "MENTIONS"
REL_EXTRACTED_FROM = "EXTRACTED_FROM"
REL_MENTIONED_IN_SOURCE = "MENTIONED_IN_SOURCE"
REL_DERIVED_FROM = "DERIVED_FROM"


class DatabaseRequestBudget(BaseModel):
    """Purpose: Track remaining database read budget for one request.

    Used in: memory DB readers that can amplify one graph lookup into many
    Neo4j rows or Qdrant point reads.
    """

    qdrant_reads: int | None = Field(default=None, ge=0)
    neo4j_rows: int | None = Field(default=None, ge=0)


class MemoryRequestContext(BaseModel):
    """Purpose: Carry request identity and project isolation for one memory operation.

    Used in: API service, mapper boundary, and pipeline orchestration before
    database primitives are built. Assembled in the service layer by
    ``api.mappers.to_memory_request_context`` from the security-only
    ``api.schemas.AuthContext`` (resolved by ``api.deps``) plus request-body
    actor identity on add/search/feedback/dreaming. Actor fields are absent
    (``None``) on endpoints that do not accept them (get/delete/update).
    """

    request_id: str
    account_id: str
    project_id: str
    api_key_uuid: str
    memory_algorithm: str | None = None
    user_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    scopes: list[str] = Field(default_factory=list)
    database_budget: DatabaseRequestBudget | None = None


class MemoryEdgeFilter(BaseModel):
    """Purpose: Constrain Memory-to-Memory graph traversal sources.

    Used in: graph recall readers and dreaming pipeline scope selection when
    direct memory relation neighbors are composed with shared-entity scopes.
    """

    rel_types: tuple[MemoryRelationType, ...] = (REL_RELATES_TO,)
    direction: GraphDirection = "both"
    edge_types: tuple[str, ...] | None = None
    relation_types: tuple[str, ...] | None = None
    active_only: bool = True


class GraphNeighborScope(BaseModel):
    """Purpose: Carry one graph recall scope for a seed memory.

    Used in: memory DB reader and dreaming scope selection. ``source`` makes
    the traversal origin explicit, e.g. shared entity or direct memory relation.
    """

    seed_memory_id: str
    entity_id: str
    entity_name: str | None = None
    entity_type: str | None = None
    memory_ids: tuple[str, ...] = Field(default_factory=tuple)
    source: GraphNeighborSource = "shared_entity"


class DirectRelatedMemory(BaseModel):
    """Purpose: Carry one direct Memory-to-Memory graph neighbor.

    Used in: memory DB reader and dreaming graph recall composition before
    direct neighbors are attached to seed/entity scopes.
    """

    seed_memory_id: str
    memory_id: str
    rel_type: MemoryRelationType
    direction: GraphDirection = "both"
    edge_type: str | None = None
    relation_type: str | None = None


class FileMessage(BaseModel):
    """Purpose: Carry an uploaded or referenced file attachment for memory add.

    Used in: ``AddPipelineInput.messages`` and add-side preprocessing.
    """

    model_config = ConfigDict(extra="forbid")

    file_name: str
    """Attachment file name."""

    file_path: str
    """Attachment path, either object storage or local path."""

    file_type: str = Field(default="")
    """File type. Defaults to the lowercase file_path extension without a dot, such as "pdf"."""

    @model_validator(mode="after")
    def _fill_file_type(self) -> FileMessage:
        """Infer missing file_type from file_path extension."""
        if not self.file_type:
            path = self.file_path.split("?", 1)[0].split("#", 1)[0]
            suffix = PurePosixPath(path).suffix
            self.file_type = suffix[1:].lower() if suffix else ""
        return self


class TextMessage(BaseModel):
    """Purpose: Carry a free-form text snippet for memory add.

    Used in: ``AddPipelineInput.messages`` and add-side preprocessing.
    """

    model_config = ConfigDict(extra="forbid")

    text: str


class UrlMessage(BaseModel):
    """Purpose: Carry a URL reference for memory add.

    Used in: ``AddPipelineInput.messages`` and add-side preprocessing.
    """

    model_config = ConfigDict(extra="forbid")

    url: str


class DialogueMessage(BaseModel):
    """Purpose: Carry one conversational turn for memory add.

    Used in: ``AddPipelineInput.messages`` and add-side preprocessing.
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    """Dialogue role, including user, assistant, system, tool, or any speaker name."""

    content: str
    """Dialogue content."""

    timestamp: int | None = None
    """Dialogue event time as a 13-digit millisecond timestamp, or None when omitted."""


class SourceRef(BaseModel):
    """Purpose: Describe an external source referenced by memory extraction.

    Used in: API add requests, text preprocessing, source writes, and graph
    ``EXTRACTED_FROM`` / ``MENTIONED_IN_SOURCE`` relationships.
    """

    source_id: str | None = None
    source_type: str = "message"
    file_path: str | None = None
    file_name: str | None = None
    is_parsed: bool = False
    parsed_content_path: str | None = None
    parsed_at: datetime | None = None
    parsed_cost: float | None = None
    uri: str | None = None
    title: str | None = None
    mime_type: str | None = None
    content_hash: str | None = None
    message_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    line_range: tuple[int, int] | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """Purpose: Represent one extracted entity before it is mapped to storage.

    Used in: text preprocessing, entity extraction components, graph write
    planning, and search/debug payloads.
    """

    name: str
    canonical_name: str | None = None
    entity_type: str | None = None
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    confidence: float | None = None
    extractor: str | None = None
    offsets: list[tuple[int, int]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreprocessedText(BaseModel):
    """Purpose: Hold normalized text and lexical/entity features before memory extraction.

    Used in: components/text and future add/search preprocessing pipelines.
    """

    segment_id: str | None = None
    text: str
    normalized_text: str
    lang: str | None = None
    content_hash: str
    bm25_text: str
    tokens: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    source_ref: SourceRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


FilterOp = Literal[
    "match",  # exact keyword / int / bool   -> MatchValue
    "any",  # value in list                -> MatchAny
    "except",  # value not in list            -> MatchExcept
    "text",  # full-text substring          -> MatchText
    "range",  # numeric gt/gte/lt/lte         -> Range
    "datetime",  # datetime gt/gte/lt/lte        -> DatetimeRange
    "is_empty",  # field empty                  -> IsEmptyCondition
    "is_null",  # field null                   -> IsNullCondition
]


class FieldCondition(BaseModel):
    """Purpose: One backend-neutral payload condition on a single field.

    Used in: ``SearchFilter`` clauses; translated to a Qdrant ``FieldCondition``
    (or ``IsEmpty``/``IsNull`` condition) only inside ``mappers.db``. The DTO
    layer never imports ``qdrant_client``.

    Field usage by ``op``:
        match/text   -> ``value``
        any/except   -> ``values``
        range/datetime -> any of ``gt`` / ``gte`` / ``lt`` / ``lte``
        is_empty/is_null -> none (only ``field`` matters)
    """

    field: str
    op: FilterOp
    value: str | int | float | bool | None = None
    values: list[str | int | float] | None = None
    gt: float | datetime | None = None
    gte: float | datetime | None = None
    lt: float | datetime | None = None
    lte: float | datetime | None = None


class SearchFilter(BaseModel):
    """Purpose: Internal boolean filter tree before project isolation.

    Used in: mapper/database-facing code and recursive nesting. Mirrors
    Qdrant's ``must`` / ``should`` / ``must_not`` structure so retrieval code can
    express payload conditions while staying free of ``qdrant_client`` types.

    Do not expose this DTO directly to external callers. ``project_id`` is NEVER
    honored from here: ``mappers.db`` force-injects ``ctx.project_id`` into
    ``must`` and rejects conditions on disallowed fields.
    """

    must: list[FieldCondition | SearchFilter] = Field(default_factory=list)
    should: list[FieldCondition | SearchFilter] = Field(default_factory=list)
    must_not: list[FieldCondition | SearchFilter] = Field(default_factory=list)


SearchFilter.model_rebuild()


# Business-facing field allowlist for the user-supplied search DSL. This is a
# deliberately narrower surface than the indexed memory payload: project
# isolation and lifecycle fields such as ``status`` are excluded and instead
# force-injected by the pipeline or DB mapper.
DSL_FILTERABLE_MEMORY_FIELDS: frozenset[str] = frozenset(
    {
        "memory_id",
        "account_id",
        "user_id",
        "app_id",
        "session_id",
        "agent_id",
        "mem_type",
        "mem_extract_type",
        "validate_from",
        "validate_to",
        "created_at",
        "property_name",
        "entity_id",
        "entity_type",
        "content",
    }
)


def search_filter_is_empty(sf: SearchFilter | None) -> bool:
    """Return whether a search filter has no effective clauses."""

    return sf is None or (not sf.must and not sf.should and not sf.must_not)


def combine_search_filters(*filters: SearchFilter | None) -> SearchFilter | None:
    """Combine filters with AND semantics while preserving nested bool clauses."""

    non_empty = [item for item in filters if not search_filter_is_empty(item)]
    if not non_empty:
        return None
    if len(non_empty) == 1:
        return non_empty[0]
    return SearchFilter(must=non_empty)


# DSL fields whose range comparisons map to a datetime range.
DSL_DATETIME_FIELDS: frozenset[str] = frozenset({"validate_from", "validate_to", "created_at"})

# DSL fields that support full-text ``contains`` / ``icontains`` operators.
DSL_TEXT_FIELDS: frozenset[str] = frozenset({"content"})


def build_tenant_conditions(ctx: MemoryRequestContext) -> list[FieldCondition]:
    """Build context-scope conditions for internal memory scans.

    Used by internal recall paths such as ``RelatedMemoryRecall`` when scanning
    active memories around an add request. Public search pipelines should not
    call this helper: their filter tree must match the FastAPI ``filters`` DSL,
    with only ``project_id`` injected later by the DB mapper.

    Always includes ``project_id``. Adds ``user_id``, ``app_id``, and
    ``agent_id`` when present (non-None and non-empty).
    """
    conditions = [FieldCondition(field="project_id", op="match", value=ctx.project_id)]
    if ctx.user_id:
        conditions.append(FieldCondition(field="user_id", op="match", value=ctx.user_id))
    if ctx.app_id:
        conditions.append(FieldCondition(field="app_id", op="match", value=ctx.app_id))
    if ctx.agent_id:
        conditions.append(FieldCondition(field="agent_id", op="match", value=ctx.agent_id))
    return conditions


class RelatedMemoryCandidate(BaseModel):
    """Purpose: One related-memory candidate from a recall channel.

    Used in: add-side recall (components/searcher), extractor prompt construction,
    and safety gate planning.
    """

    memory_id: str
    score: float
    source: str
    rank: int | None = None
    memory: MemoryView | None = None
    debug: dict[str, object] = Field(default_factory=dict)


class RelatedMemoryRecallResult(BaseModel):
    """Purpose: Recall result separated into exact duplicate and fused context.

    Used in: add pipeline recall phase, extractor prompt, and fallback planning.
    """

    duplicate: RelatedMemoryCandidate | None = None
    candidates: list[RelatedMemoryCandidate] = Field(default_factory=list)


class MemoryWrite(BaseModel):
    """Purpose: Define the minimal Qdrant ``memory_item_v1`` payload before mapping.

    Used in: add/update/merge algorithms and ``pipelines.memory_db.MemoryDbWriter``.
    """

    memory_id: str
    account_id: str
    project_id: str
    api_key_uuid: str
    user_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    request_id: str | None = None
    content: str
    mem_type: MemoryType = "fact"
    mem_extract_type: str = "vanilla"
    mem_extract_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    validate_from: datetime | None = None
    validate_to: datetime | None = None
    status: MemoryStatus = "active"
    reinforcement_count: int = 0
    created_at: datetime
    update_at: datetime | None = None
    status_changed_at: datetime | None = None
    parent_ids: list[str] = Field(default_factory=list)
    root_id: list[str] = Field(default_factory=list)
    property_name: str | None = None
    entity_id: str | None = None
    entity_type: str | None = None


class EntityWrite(BaseModel):
    """Purpose: Define the minimal Qdrant ``entity_item_v1`` payload before mapping.

    Used in: entity extraction outputs and graph mirror write planning.
    """

    entity_id: str
    account_id: str
    project_id: str
    api_key_uuid: str
    user_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    request_id: str | None = None
    entity_name: str
    entity_type: str | None = None
    description: str | None = None
    status: MemoryStatus = "active"
    created_at: datetime
    update_at: datetime | None = None
    status_changed_at: datetime | None = None
    parent_ids: list[str] = Field(default_factory=list)
    root_id: list[str] = Field(default_factory=list)
    schema_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceWrite(BaseModel):
    """Purpose: Define a source write intent whose target store depends on ``persist_payload``.

    Used in: file/url/message source ingestion and Neo4j source mirror planning.
    ``persist_payload=True`` (default for ``file``/``url``) maps the source to the
    Qdrant ``source_item_v1`` collection; ``persist_payload=False`` (``message``)
    keeps the source only as a Neo4j ``Source`` graph node used as a provenance
    edge endpoint, carrying no vector or parsed content.
    """

    source_id: str
    account_id: str
    project_id: str
    api_key_uuid: str
    user_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    request_id: str | None = None
    source_type: str
    file_path: str
    file_name: str
    is_parsed: bool = False
    parsed_content_path: str | None = None
    status: MemoryStatus = "active"
    created_at: datetime
    update_at: datetime | None = None
    parsed_at: datetime | None = None
    parsed_cost: float | None = None
    status_changed_at: datetime | None = None
    parent_ids: list[str] = Field(default_factory=list)
    root_id: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    persist_payload: bool = Field(
        default=True,
        description=(
            "Whether to persist this source's payload into the ``source_item_v1`` "
            "Qdrant collection. External resources (``file``/``url``) keep ``True``; "
            "``message`` sources set ``False`` and only exist as Neo4j ``Source`` "
            "graph nodes (provenance edge endpoints) carrying no vector or parsed content."
        ),
    )


class VectorWrite(BaseModel):
    """Purpose: Attach dense and sparse vectors to a memory write.

    Used in: mapper conversion from algorithm output into Qdrant point vectors.
    """

    memory_id: str
    semantic_vector: list[float] | None = None
    bm25_indices: list[int] = Field(default_factory=list)
    bm25_values: list[float] = Field(default_factory=list)


class EntityVectorWrite(BaseModel):
    """Purpose: Attach dense and sparse vectors to an entity write.

    Used in: add/merge pipelines when writing ``entity_item_v1`` through
    ``MemoryDbWritePlan`` without bypassing the repository boundary.
    """

    entity_id: str
    semantic_vector: list[float] | None = None
    bm25_indices: list[int] | None = None
    bm25_values: list[float] | None = None


class GraphNodeRef(BaseModel):
    """Purpose: Address a graph node with the keys defined by the database design.

    Used in: graph relationship write planning and Neo4j mapper conversion.
    """

    kind: GraphNodeKind
    project_id: str
    node_id: str


class GraphRelationship(BaseModel):
    """Purpose: Define one graph mirror relationship before Neo4j primitive mapping.

    Used in: memory write plans for ``HAS_PROPERTY_MEMORY``, ``MENTIONS``,
    ``EXTRACTED_FROM``, and related graph edges.
    """

    source: GraphNodeRef
    target: GraphNodeRef
    rel_type: str
    project_id: str
    property_name: str | None = None
    edge_type: str | None = None
    relation_type: str | None = None
    entity_id: str | None = None
    extraction_position: dict[str, Any] | None = None
    mention_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryView(BaseModel):
    """Purpose: Represent a business-visible memory hydrated from storage.

    Used in: get/search/list responses and API service return payloads.
    """

    memory_id: str
    project_id: str
    content: str
    mem_type: MemoryType | str
    mem_extract_type: str | None = None
    mem_extract_version: str | None = None
    status: MemoryStatus | str
    metadata: dict[str, Any] = Field(default_factory=dict)
    account_id: str | None = None
    api_key_uuid: str | None = None
    user_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    request_id: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    root_id: list[str] = Field(default_factory=list)
    property_name: str | None = None
    entity_id: str | None = None
    entity_type: str | None = None
    validate_from: datetime | None = None
    validate_to: datetime | None = None
    created_at: datetime | None = None
    update_at: datetime | None = None


class EntityView(BaseModel):
    """Purpose: Represent a business-visible entity hydrated from storage.

    Used in: add/merge pipelines for entity recall and graph-aware write
    planning without exposing database primitives.
    """

    entity_id: str
    project_id: str
    entity_name: str
    entity_type: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    account_id: str | None = None
    api_key_uuid: str | None = None
    user_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    request_id: str | None = None
    created_at: datetime | None = None
    update_at: datetime | None = None


class EntitySearchHit(BaseModel):
    """Purpose: Carry one entity recall hit with score and payload.

    Used in: add pipelines before LLM entity merge decisions.
    """

    entity_id: str
    """Canonical business entity id, never the storage-only search-field point id."""
    score: float
    entity: EntityView | None = None
    source: str | None = None
    rank: int | None = None
    best_search_field: str = ""
    """Highest-scoring matched search_field text for this query, empty for core-point hits."""
    best_search_field_index: int | None = None
    """Original index of the matched search_field on the entity metadata."""
    best_search_field_score: float | None = None
    """Raw recall score for the matched search_field."""
    matched_point_role: str = "core"
    """Storage point role for the winning hit: ``core`` or ``search_field``."""


class EntitySearchResult(BaseModel):
    """Purpose: Return project-scoped entity recall results from the DB reader.

    Used in: add/merge pipelines that need entity candidate recall while
    keeping direct Qdrant access inside infrastructure code.
    """

    query: str
    hits: list[EntitySearchHit] = Field(default_factory=list)
    total: int = 0
    debug: dict[str, Any] = Field(default_factory=dict)
