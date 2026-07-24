"""Memory DB operation DTOs.

These models describe database-bound memory operations and execution results.
They are intentionally separate from the business memory DTOs in
``mindmemos.typing.memory`` so the DB operator surface can evolve without
mixing storage mutation semantics into the core memory contract.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .memory import (
    ConsistencyMode,
    EntityVectorWrite,
    EntityWrite,
    GraphRelationship,
    MemoryStatus,
    MemoryView,
    MemoryWrite,
    RankingMode,
    SearchFilter,
    SearchMode,
    SourceWrite,
    VectorWrite,
)


class MemoryDbWritePlan(BaseModel):
    """Purpose: Batch all database writes produced by a memory operation.

    Used in: ``pipelines.memory_db.MemoryDbWriter`` as the boundary between
    business algorithms and low-level database stores.
    """

    memories: list[MemoryWrite] = Field(default_factory=list)
    entities: list[EntityWrite] = Field(default_factory=list)
    sources: list[SourceWrite] = Field(default_factory=list)
    vectors: list[VectorWrite] = Field(default_factory=list)
    entity_vectors: list[EntityVectorWrite] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)


class MemoryDbMemoryWriteCommand(BaseModel):
    """Purpose: Upsert one memory and its optional vector through the memory DB boundary.

    Used in: add, feedback add, dreaming consolidation, and other pipelines
    that create or overwrite memories.
    """

    memory: MemoryWrite
    vector: VectorWrite | None = None
    operation: Literal["upsert"] = "upsert"


class MemoryDbEntityWriteCommand(BaseModel):
    """Purpose: Upsert one entity with its core and search-field vectors.

    Used in: add and schema pipelines before mapping to Qdrant entity points
    and Neo4j entity nodes.
    """

    entity: EntityWrite
    core_vector: EntityVectorWrite | None = None
    search_field_vectors: list[EntityVectorWrite] = Field(default_factory=list)
    operation: Literal["upsert"] = "upsert"
    replace_search_fields: bool = True


class MemoryDbSourceWriteCommand(BaseModel):
    """Purpose: Upsert one source through the memory DB boundary.

    Used in: add pipelines before mapping to Qdrant source points and Neo4j
    source nodes.
    """

    source: SourceWrite
    operation: Literal["upsert"] = "upsert"


class MemoryDbRelationshipWriteCommand(BaseModel):
    """Purpose: Upsert one graph relationship through the memory DB boundary.

    Used in: add, dreaming, and schema pipelines when graph edges must be
    mirrored into Neo4j.
    """

    relationship: GraphRelationship
    operation: Literal["upsert"] = "upsert"


class MemoryDbMemoryUpdateCommand(BaseModel):
    """Purpose: Patch one existing memory through the memory DB boundary.

    Used in: update, feedback, dreaming, schema merge, and vanilla add
    update/merge flows.
    """

    memory_id: str
    content: str | None = None
    reinforcement_count: int | None = None
    metadata_patch: dict[str, Any] = Field(default_factory=dict)
    payload_patch: dict[str, Any] = Field(default_factory=dict)
    """Storage payload fields to carry into this mutation.

    This is intentionally narrow and DB-facing. Schema pipelines use it when a
    versioned content update also needs to refresh business fields such as
    ``property_name`` or ``validate_from`` without bypassing ``update_memory``.
    """
    status: MemoryStatus | None = None
    reinforcement_count_delta: int = 0
    reason: str | None = None
    consistency: ConsistencyMode = "strong"
    dedup_metadata_key: str | None = None
    """If set, skip this update when the existing metadata already has the same
    value for this key as ``metadata_patch[dedup_metadata_key]``.  Used to make
    reinforcement commands idempotent across Kafka retries."""
    dense_vector: list[float] | None = None
    """Dense semantic vector to replace on Qdrant.  Set by UPDATE actions that
    recompute the embedding after content changes."""
    sparse_vectors: dict[str, Any] | None = None
    """Sparse BM25 vector payload to replace on Qdrant (bm25_indices, bm25_values)."""
    embedding: list[float] | None = None
    """Precomputed dense semantic vector accepted by the public update command path."""
    bm25_indices: list[int] | None = None
    """Precomputed BM25 sparse indices; values default to 1.0 for API-side commands."""
    graph_content_sync: bool = False
    """If True, also update the Neo4j memory node content property."""


class MemoryDbEntityUpdateCommand(BaseModel):
    """Purpose: Update one existing entity through the memory DB boundary.

    Used in: schema merge/update flows. The first implementation may treat an
    entity update as a full entity upsert while preserving an explicit command
    slot for future partial patch semantics.
    """

    entity_id: str
    entity: EntityWrite | None = None
    description: str | None = None
    metadata_patch: dict[str, Any] = Field(default_factory=dict)
    status: MemoryStatus | None = None
    core_vector: EntityVectorWrite | None = None
    search_field_vectors: list[EntityVectorWrite] = Field(default_factory=list)
    replace_search_fields: bool = False
    consistency: ConsistencyMode = "strong"


class MemoryDbSourceUpdateCommand(BaseModel):
    """Purpose: Patch one existing source through the memory DB boundary.

    Used in: future source lifecycle flows. This is a structural command slot;
    execution can be implemented when source patch infra is available.
    """

    source_id: str
    payload_patch: dict[str, Any] = Field(default_factory=dict)
    metadata_patch: dict[str, Any] = Field(default_factory=dict)
    status: MemoryStatus | None = None
    consistency: ConsistencyMode = "strong"


class MemoryDbMemoryDeleteCommand(BaseModel):
    """Purpose: Archive or hard-delete one existing memory.

    Used in: delete, feedback delete, dreaming archive, and schema merge
    archive flows.
    """

    memory_id: str
    hard: bool = False
    reason: str = "user_request"
    consistency: ConsistencyMode = "strong"


class MemoryDbEntityDeleteCommand(BaseModel):
    """Purpose: Archive or hard-delete one existing entity.

    Used in: future entity lifecycle flows. This is a structural command slot
    until entity delete execution is implemented.
    """

    entity_id: str
    hard: bool = False
    reason: str = "user_request"
    consistency: ConsistencyMode = "strong"


class MemoryDbSourceDeleteCommand(BaseModel):
    """Purpose: Archive or hard-delete one existing source.

    Used in: future source lifecycle flows. This is a structural command slot
    until source delete execution is implemented.
    """

    source_id: str
    hard: bool = False
    reason: str = "user_request"
    consistency: ConsistencyMode = "strong"


class MemoryDbRelationshipDeleteCommand(BaseModel):
    """Purpose: Delete one graph relationship through the memory DB boundary.

    Used in: future graph maintenance flows. Relationship key semantics should
    be finalized before enabling execution.
    """

    relationship: GraphRelationship
    reason: str = "user_request"
    consistency: ConsistencyMode = "strong"


class MemoryDbMutationPlan(BaseModel):
    """Purpose: Batch all database mutations produced by memory pipelines.

    Used in: ``pipelines.memory_db.MemoryDbWriter`` as the unified boundary
    for add, update, delete, feedback, dreaming, and schema merge flows.
    """

    memory_writes: list[MemoryDbMemoryWriteCommand] = Field(default_factory=list)
    memory_updates: list[MemoryDbMemoryUpdateCommand] = Field(default_factory=list)
    memory_deletes: list[MemoryDbMemoryDeleteCommand] = Field(default_factory=list)
    entity_writes: list[MemoryDbEntityWriteCommand] = Field(default_factory=list)
    entity_updates: list[MemoryDbEntityUpdateCommand] = Field(default_factory=list)
    entity_deletes: list[MemoryDbEntityDeleteCommand] = Field(default_factory=list)
    source_writes: list[MemoryDbSourceWriteCommand] = Field(default_factory=list)
    source_updates: list[MemoryDbSourceUpdateCommand] = Field(default_factory=list)
    source_deletes: list[MemoryDbSourceDeleteCommand] = Field(default_factory=list)
    relationship_writes: list[MemoryDbRelationshipWriteCommand] = Field(default_factory=list)
    relationship_deletes: list[MemoryDbRelationshipDeleteCommand] = Field(default_factory=list)

    @classmethod
    def from_write_plan(cls, plan: MemoryDbWritePlan) -> "MemoryDbMutationPlan":
        """Build a mutation plan from the legacy flat write plan."""

        vector_by_memory_id = {vector.memory_id: vector for vector in plan.vectors}
        entity_vectors_by_owner: dict[str, list[EntityVectorWrite]] = {}
        for vector in plan.entity_vectors:
            owner_id = vector.entity_id.split("#sf", 1)[0]
            entity_vectors_by_owner.setdefault(owner_id, []).append(vector)

        entity_writes: list[MemoryDbEntityWriteCommand] = []
        for entity in plan.entities:
            vectors = entity_vectors_by_owner.get(entity.entity_id, [])
            core_vector = next((vector for vector in vectors if vector.entity_id == entity.entity_id), None)
            search_field_vectors = [vector for vector in vectors if vector.entity_id != entity.entity_id]
            entity_writes.append(
                MemoryDbEntityWriteCommand(
                    entity=entity,
                    core_vector=core_vector,
                    search_field_vectors=search_field_vectors,
                )
            )

        return cls(
            memory_writes=[
                MemoryDbMemoryWriteCommand(memory=memory, vector=vector_by_memory_id.get(memory.memory_id))
                for memory in plan.memories
            ],
            entity_writes=entity_writes,
            source_writes=[MemoryDbSourceWriteCommand(source=source) for source in plan.sources],
            relationship_writes=[
                MemoryDbRelationshipWriteCommand(relationship=relationship) for relationship in plan.relationships
            ],
        )

    def to_write_plan(self) -> MemoryDbWritePlan:
        """Project write/upsert commands into the legacy flat write plan."""

        memories: list[MemoryWrite] = []
        vectors: list[VectorWrite] = []
        entities: list[EntityWrite] = []
        entity_vectors: list[EntityVectorWrite] = []
        sources: list[SourceWrite] = []
        relationships: list[GraphRelationship] = []

        for command in self.memory_writes:
            memories.append(command.memory)
            if command.vector is not None:
                vectors.append(command.vector)
        for command in [*self.entity_writes, *self.entity_updates_as_writes()]:
            entities.append(command.entity)
            if command.core_vector is not None:
                entity_vectors.append(command.core_vector)
            entity_vectors.extend(command.search_field_vectors)
        sources.extend(command.source for command in self.source_writes)
        relationships.extend(command.relationship for command in self.relationship_writes)

        return MemoryDbWritePlan(
            memories=memories,
            entities=entities,
            sources=sources,
            vectors=vectors,
            entity_vectors=entity_vectors,
            relationships=relationships,
        )

    def entity_updates_as_writes(self) -> list[MemoryDbEntityWriteCommand]:
        """Return entity updates that can currently be executed as full upserts."""

        return [
            MemoryDbEntityWriteCommand(
                entity=command.entity,
                core_vector=command.core_vector,
                search_field_vectors=command.search_field_vectors,
                replace_search_fields=command.replace_search_fields,
            )
            for command in self.entity_updates
            if command.entity is not None
        ]

    def has_writes(self) -> bool:
        """Return whether the plan contains write/upsert commands."""

        return bool(
            self.memory_writes
            or self.entity_writes
            or any(command.entity is not None for command in self.entity_updates)
            or self.source_writes
            or self.relationship_writes
        )

    def has_updates_or_deletes(self) -> bool:
        """Return whether the plan contains non-write mutation commands."""

        return bool(
            self.memory_updates
            or self.memory_deletes
            or self.entity_updates
            or self.entity_deletes
            or self.source_updates
            or self.source_deletes
            or self.relationship_deletes
        )


class MemoryDbSearchQuery(BaseModel):
    """Purpose: DB-layer search request for ``pipelines.memory_db.reader``.

    Used in: ``MemoryDbReader.search_dense / search_sparse / search_rrf /
    search_by_filter``. Sparse search receives dense business-level
    ``indices`` / ``values`` instead of infra DB vector models. Higher-level pipelines translate their public
    ``SearchPipelineInput`` into this DB-shaped query before execution.
    """

    query: str
    top_k: int = 10
    filters: SearchFilter | None = None
    mode: SearchMode = "hybrid"
    ranking: RankingMode = "hybrid"
    include_debug: bool = False
    include_graph: bool = False
    include_content: bool = False
    include_patches: bool = True
    """Deprecated compatibility field; search engines own patch enrichment."""


class MemoryDbSearchHit(BaseModel):
    """Purpose: One retrieval hit returned by the DB layer.

    Used in: ``MemoryDbSearchResult.hits`` and mapper output. The pipeline
    layer projects this into the public ``MemorySearchItem`` shape before
    returning to the service layer.
    """

    memory_id: str
    score: float
    memory: MemoryView | None = None
    source: str | None = None
    rank: int | None = None
    debug: dict[str, Any] = Field(default_factory=dict)


class MemoryDbSearchResult(BaseModel):
    """Purpose: DB-layer search response with raw hits and execution metadata.

    Used in: ``MemoryDbReader`` return value and mapper ``to_search_result``.
    """

    query: str
    hits: list[MemoryDbSearchHit] = Field(default_factory=list)
    total: int = 0
    debug: dict[str, Any] = Field(default_factory=dict)


MemoryDbUpdateCommand = MemoryDbMemoryUpdateCommand
MemoryDbDeleteCommand = MemoryDbMemoryDeleteCommand


class MemoryDbMutationResult(BaseModel):
    """Purpose: Minimal DB-layer acknowledgement for update/delete operations.

    Used in: ``MemoryDbWriter.update_memory / delete_memory`` return value
    and mapper ``to_mutation_result``.
    """

    status: str = "ok"
    memory_id: str
    changed: bool = True
    hard: bool = False


class MemoryDbWriteSummary(BaseModel):
    """Purpose: Summarize a completed write plan dispatch at the DB boundary.

    Used in: mapper ``to_add_result`` and higher pipelines that surface
    write outcomes to the service layer.
    """

    status: str = "ok"
    memory_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class MemoryDbWriteResult(BaseModel):
    """Purpose: Summarize the outcome of writing one plan into Qdrant and Neo4j.

    Used in: ``pipelines.memory_db.MemoryDbWriter`` return value. For unified
    mutation plans, ``mutations`` carries per-command update/delete results.
    """

    memory_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    mutations: list[MemoryDbMutationResult] = Field(default_factory=list)
    graph_pending: bool = False
    errors: list[str] = Field(default_factory=list)
