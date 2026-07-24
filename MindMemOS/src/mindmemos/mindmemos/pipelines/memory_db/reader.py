"""Database read orchestration for memory records."""

from __future__ import annotations

import asyncio
from typing import Any

from qdrant_client import models as qmodels

from ...components.memory_modeling.schema import TemporalEntity
from ...config import get_config
from ...errors import ConfigNotInitializedError
from ...infra.db import (
    DatabaseClients,
    QdrantRecord,
    QdrantSearchRecord,
    SparseVectorData,
    resolve_database_clients,
)
from ...logging import get_logger, traced
from ...mappers import (
    search_filter_to_qdrant,
    to_entity_search_result,
    to_entity_view_from_record,
    to_memory_view_from_record,
    to_search_result,
)
from ...typing import (
    REL_RELATED_TO,
    REL_RELATES_TO,
    DatabaseRequestBudget,
    DirectRelatedMemory,
    EntitySearchResult,
    EntityView,
    FieldCondition,
    GraphNeighborScope,
    GraphNeighborSource,
    MemoryDbSearchQuery,
    MemoryDbSearchResult,
    MemoryEdgeFilter,
    MemoryRequestContext,
    MemoryView,
    SearchFilter,
    SparseVector,
    combine_search_filters,
)
from .add_record_store import AddRecordStore

logger = get_logger(__name__)

DEFAULT_ENTITY_SEARCH_FIELD_OVERFETCH_FACTOR = 3
DEFAULT_REQUEST_QDRANT_READ_BUDGET = 100
DEFAULT_REQUEST_NEO4J_ROW_BUDGET = 100


ACTIVE_MEMORY_FILTER = SearchFilter(must=[FieldCondition(field="status", op="match", value="active")])
_ALLOWED_MEMORY_REL_TYPES = {REL_RELATES_TO, REL_RELATED_TO}


def _to_sparse_vector_data(vector: SparseVector) -> SparseVectorData:
    return SparseVectorData(indices=list(vector.indices), values=list(vector.values))


def _entity_search_field_overfetch_factor() -> int:
    try:
        value = get_config().algo_config.search.schema_search.entity.search_field_overfetch_factor
    except ConfigNotInitializedError:
        return DEFAULT_ENTITY_SEARCH_FIELD_OVERFETCH_FACTOR
    except Exception:
        logger.warning("entity_search_field_overfetch_config_failed", exc_info=True)
        return DEFAULT_ENTITY_SEARCH_FIELD_OVERFETCH_FACTOR
    return max(1, int(value or DEFAULT_ENTITY_SEARCH_FIELD_OVERFETCH_FACTOR))


def _configured_request_qdrant_read_budget() -> int:
    try:
        value = get_config().database.qdrant.request_read_budget
    except ConfigNotInitializedError:
        return DEFAULT_REQUEST_QDRANT_READ_BUDGET
    except Exception:
        logger.warning("qdrant_request_read_budget_config_failed", exc_info=True)
        return DEFAULT_REQUEST_QDRANT_READ_BUDGET
    return max(0, int(value or 0))


def _configured_request_neo4j_row_budget() -> int:
    try:
        value = get_config().database.neo4j.request_row_budget
    except ConfigNotInitializedError:
        return DEFAULT_REQUEST_NEO4J_ROW_BUDGET
    except Exception:
        logger.warning("neo4j_request_row_budget_config_failed", exc_info=True)
        return DEFAULT_REQUEST_NEO4J_ROW_BUDGET
    return max(0, int(value or 0))


def _ensure_request_database_budget(ctx: MemoryRequestContext) -> DatabaseRequestBudget:
    budget = ctx.database_budget
    if budget is None:
        budget = DatabaseRequestBudget(
            qdrant_reads=_configured_request_qdrant_read_budget(),
            neo4j_rows=_configured_request_neo4j_row_budget(),
        )
        ctx.database_budget = budget
        return budget
    if budget.qdrant_reads is None:
        budget.qdrant_reads = _configured_request_qdrant_read_budget()
    if budget.neo4j_rows is None:
        budget.neo4j_rows = _configured_request_neo4j_row_budget()
    return budget


def _budget_value(value: int | None) -> int:
    return max(0, int(value or 0))


def _reserve_database_budget(ctx: MemoryRequestContext, *, limit: int | None = None) -> int:
    budget = _ensure_request_database_budget(ctx)
    candidates = [_budget_value(budget.qdrant_reads), _budget_value(budget.neo4j_rows)]
    if limit is not None:
        candidates.append(max(0, limit))
    reservation = min(candidates)
    if reservation <= 0:
        return 0
    budget.qdrant_reads = _budget_value(budget.qdrant_reads) - reservation
    budget.neo4j_rows = _budget_value(budget.neo4j_rows) - reservation
    return reservation


def _refund_database_budget(ctx: MemoryRequestContext, *, qdrant_reads: int = 0, neo4j_rows: int = 0) -> None:
    budget = _ensure_request_database_budget(ctx)
    if qdrant_reads > 0:
        budget.qdrant_reads = _budget_value(budget.qdrant_reads) + qdrant_reads
    if neo4j_rows > 0:
        budget.neo4j_rows = _budget_value(budget.neo4j_rows) + neo4j_rows


def _dedupe_entity_search_records(hits: list[QdrantSearchRecord], *, limit: int) -> list[QdrantSearchRecord]:
    """Group core/search-field entity points by canonical payload entity_id."""

    best_by_entity: dict[str, QdrantSearchRecord] = {}
    for fallback_rank, hit in enumerate(hits, start=1):
        payload = dict(hit.payload or {})
        canonical_entity_id = str(payload.get("entity_id") or hit.point_id)
        metadata = dict(payload.get("metadata") or {})
        is_search_field = bool(metadata.get("is_search_field"))
        debug = dict(hit.debug)
        debug.setdefault("rank", fallback_rank)
        debug["matched_point_id"] = hit.point_id
        debug["matched_point_role"] = "search_field" if is_search_field else "core"
        if is_search_field:
            search_field = str(metadata.get("search_field_content") or "")
            debug["best_search_field"] = search_field
            debug["best_search_field_index"] = metadata.get("search_field_index")
            debug["best_search_field_score"] = hit.score
        payload["entity_id"] = canonical_entity_id
        normalized = QdrantSearchRecord(
            point_id=canonical_entity_id,
            score=hit.score,
            payload=payload,
            vectors=hit.vectors,
            source=hit.source,
            debug=debug,
        )

        current = best_by_entity.get(canonical_entity_id)
        if current is None or normalized.score > current.score:
            best_by_entity[canonical_entity_id] = normalized
            continue

        if is_search_field and not current.debug.get("best_search_field"):
            current.debug["best_search_field"] = debug.get("best_search_field", "")
            current.debug["best_search_field_index"] = debug.get("best_search_field_index")
            current.debug["best_search_field_score"] = debug.get("best_search_field_score")

    return sorted(best_by_entity.values(), key=lambda item: item.score, reverse=True)[: max(0, limit)]


def _active_memory_filter(filters: SearchFilter | None = None) -> SearchFilter:
    return combine_search_filters(ACTIVE_MEMORY_FILTER, filters) or ACTIVE_MEMORY_FILTER


def _memory_rel_type_fragment(edge_filter: MemoryEdgeFilter) -> str:
    rel_types = tuple(dict.fromkeys(edge_filter.rel_types or (REL_RELATES_TO,)))
    invalid = [rel_type for rel_type in rel_types if rel_type not in _ALLOWED_MEMORY_REL_TYPES]
    if invalid:
        raise ValueError(f"unsupported memory relation types: {invalid}")
    return ":" + "|".join(rel_types)


def _attach_direct_neighbors(
    scope: GraphNeighborScope, direct_memory_ids: list[str] | tuple[str, ...]
) -> GraphNeighborScope:
    if not direct_memory_ids:
        return scope
    return scope.model_copy(update={"memory_ids": tuple(dict.fromkeys([*scope.memory_ids, *direct_memory_ids]))})


class MemoryDbReader:
    """Read stored memories through low-level database clients."""

    def __init__(self, clients: DatabaseClients | None = None, add_record_store: AddRecordStore | None = None) -> None:
        self._clients = resolve_database_clients(clients)
        self._add_records = add_record_store or AddRecordStore(clients=self._clients)
        self._entity_search_field_overfetch_factor = _entity_search_field_overfetch_factor()

    @traced("memory_db.get_memory")
    async def get_memory(self, ctx: MemoryRequestContext, memory_id: str) -> MemoryView | None:
        """Read one memory in the request project."""

        record = await self._clients.qdrant.get_memory(ctx.project_id, memory_id)
        return to_memory_view_from_record(record) if record else None

    @traced("memory_db.get_memory_record")
    async def get_memory_record(self, ctx: MemoryRequestContext, memory_id: str) -> QdrantRecord | None:
        """Read one raw memory record in the request project."""

        return await self._clients.qdrant.get_memory(ctx.project_id, memory_id, with_vectors=False)

    @traced("memory_db.get_memories")
    async def get_memories(self, ctx: MemoryRequestContext, memory_ids: list[str]) -> list[MemoryView]:
        """Read many memories in the request project."""

        records = await self._clients.qdrant.get_memories(ctx.project_id, memory_ids)
        return [to_memory_view_from_record(record) for record in records]


    @traced("memory_db.list_memories_by_shared_entities")
    async def list_memories_by_shared_entities(
        self,
        ctx: MemoryRequestContext,
        memory_ids: list[str],
        *,
        include_seed: bool = True,
        active_only: bool = True,
        limit_per_entity: int = 50,
    ) -> list[GraphNeighborScope]:
        """Return ``Memory -> Entity <- Memory`` scopes only."""

        seed_ids = [mid for mid in dict.fromkeys(memory_ids) if mid]
        if not seed_ids:
            return []
        status_clause = "WHERE coalesce(mentioned.status, 'active') = 'active'" if active_only else ""
        seed_expr = "mentioned_ids + [seed_memory_id]" if include_seed else "mentioned_ids"
        query = """
        UNWIND $memory_ids AS seed_memory_id
        MATCH (seed:Memory {{project_id: $project_id, memory_id: seed_memory_id}})-[:MENTIONS]->(e:Entity {{project_id: $project_id}})
        OPTIONAL MATCH (e)<-[:MENTIONS]-(mentioned:Memory {{project_id: $project_id}})
        {status_clause}
        WITH seed_memory_id, e, collect(DISTINCT mentioned.memory_id) AS mentioned_ids
        WITH seed_memory_id, e, {seed_expr} AS raw_memory_ids
        UNWIND raw_memory_ids AS memory_id
        WITH seed_memory_id, e, memory_id
        WHERE memory_id IS NOT NULL
        MATCH (m:Memory {{project_id: $project_id, memory_id: memory_id}})
        {active_memory_clause}
        ORDER BY m.memory_id
        WITH seed_memory_id,
             e,
             collect(DISTINCT m.memory_id)[0..$limit_per_entity] AS memory_ids
        RETURN seed_memory_id AS seed_memory_id,
               e.entity_id AS entity_id,
               e.entity_name AS entity_name,
               e.entity_type AS entity_type,
               memory_ids AS memory_ids
        """.format(
            status_clause=status_clause,
            seed_expr=seed_expr,
            active_memory_clause="WHERE coalesce(m.status, 'active') = 'active'" if active_only else "",
        )
        rows = await self._clients.neo4j.run_read(
            query,
            project_id=ctx.project_id,
            memory_ids=seed_ids,
            limit_per_entity=limit_per_entity,
        )
        scopes: list[GraphNeighborScope] = []
        for row in rows:
            entity_id = row.get("entity_id")
            if not entity_id:
                continue
            scopes.append(
                GraphNeighborScope(
                    seed_memory_id=str(row.get("seed_memory_id") or ""),
                    entity_id=str(entity_id),
                    entity_name=row.get("entity_name"),
                    entity_type=row.get("entity_type"),
                    memory_ids=tuple(str(mid) for mid in row.get("memory_ids") or [] if mid),
                    source="shared_entity",
                )
            )
        return scopes

    @traced("memory_db.list_direct_related_memories")
    async def list_direct_related_memories(
        self,
        ctx: MemoryRequestContext,
        memory_ids: list[str],
        *,
        edge_filter: MemoryEdgeFilter,
        limit_per_memory: int = 20,
        max_candidates: int = 200,
    ) -> list[DirectRelatedMemory]:
        """Return direct Memory-to-Memory neighbors only."""

        seed_ids = [mid for mid in dict.fromkeys(memory_ids) if mid]
        if not seed_ids:
            return []
        rel_fragment = _memory_rel_type_fragment(edge_filter)
        if edge_filter.direction == "out":
            pattern = f"(seed:Memory {{project_id: $project_id, memory_id: seed_memory_id}})-[r{rel_fragment}]->(related:Memory {{project_id: $project_id}})"
        elif edge_filter.direction == "in":
            pattern = f"(seed:Memory {{project_id: $project_id, memory_id: seed_memory_id}})<-[r{rel_fragment}]-(related:Memory {{project_id: $project_id}})"
        else:
            pattern = f"(seed:Memory {{project_id: $project_id, memory_id: seed_memory_id}})-[r{rel_fragment}]-(related:Memory {{project_id: $project_id}})"
        filters = []
        if edge_filter.active_only:
            filters.append("coalesce(related.status, 'active') = 'active'")
        if edge_filter.edge_types:
            filters.append("r.edge_type IN $edge_types")
        if edge_filter.relation_types:
            filters.append("r.relation_type IN $relation_types")
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = f"""
        UNWIND $memory_ids AS seed_memory_id
        MATCH {pattern}
        {where_clause}
        WITH seed_memory_id,
             related,
             type(r) AS rel_type,
             r.edge_type AS edge_type,
             r.relation_type AS relation_type
        ORDER BY related.memory_id
        WITH seed_memory_id,
             collect({{
                 memory_id: related.memory_id,
                 rel_type: rel_type,
                 edge_type: edge_type,
                 relation_type: relation_type
             }})[0..$limit_per_memory] AS rows
        UNWIND rows AS row
        RETURN seed_memory_id AS seed_memory_id,
               row.memory_id AS memory_id,
               row.rel_type AS rel_type,
               row.edge_type AS edge_type,
               row.relation_type AS relation_type
        LIMIT $max_candidates
        """
        rows = await self._clients.neo4j.run_read(
            query,
            project_id=ctx.project_id,
            memory_ids=seed_ids,
            limit_per_memory=limit_per_memory,
            max_candidates=max_candidates,
            edge_types=list(edge_filter.edge_types or []),
            relation_types=list(edge_filter.relation_types or []),
        )
        related: list[DirectRelatedMemory] = []
        for row in rows:
            memory_id = row.get("memory_id")
            seed_memory_id = row.get("seed_memory_id")
            rel_type = row.get("rel_type")
            if not memory_id or not seed_memory_id or rel_type not in _ALLOWED_MEMORY_REL_TYPES:
                continue
            related.append(
                DirectRelatedMemory(
                    seed_memory_id=str(seed_memory_id),
                    memory_id=str(memory_id),
                    rel_type=rel_type,
                    direction=edge_filter.direction,
                    edge_type=row.get("edge_type"),
                    relation_type=row.get("relation_type"),
                )
            )
        return related

    @traced("memory_db.list_memory_neighbor_scopes")
    async def list_memory_neighbor_scopes(
        self,
        ctx: MemoryRequestContext,
        memory_ids: list[str],
        *,
        sources: tuple[GraphNeighborSource, ...] = ("shared_entity", "direct_memory_relation"),
        edge_filter: MemoryEdgeFilter = MemoryEdgeFilter(),
        include_seed: bool = True,
        active_only: bool = True,
        limit_per_entity: int = 50,
        limit_direct_per_memory: int = 20,
        attach_direct_neighbors_to_entity_scopes: bool = True,
    ) -> list[GraphNeighborScope]:
        """Compose explicit graph traversal sources for dreaming."""

        scopes: list[GraphNeighborScope] = []
        if "shared_entity" in sources:
            scopes.extend(
                await self.list_memories_by_shared_entities(
                    ctx,
                    memory_ids,
                    include_seed=include_seed,
                    active_only=active_only,
                    limit_per_entity=limit_per_entity,
                )
            )
        direct_by_seed: dict[str, list[str]] = {}
        if "direct_memory_relation" in sources:
            direct = await self.list_direct_related_memories(
                ctx,
                memory_ids,
                edge_filter=edge_filter,
                limit_per_memory=limit_direct_per_memory,
            )
            for item in direct:
                direct_by_seed.setdefault(item.seed_memory_id, []).append(item.memory_id)
        if direct_by_seed and attach_direct_neighbors_to_entity_scopes:
            return [_attach_direct_neighbors(scope, direct_by_seed.get(scope.seed_memory_id, ())) for scope in scopes]
        if direct_by_seed:
            for seed_memory_id, related_ids in direct_by_seed.items():
                raw_ids = [seed_memory_id, *related_ids] if include_seed else list(related_ids)
                ids = tuple(dict.fromkeys(raw_ids))
                scopes.append(
                    GraphNeighborScope(
                        seed_memory_id=seed_memory_id,
                        entity_id="direct_memory_relation",
                        entity_name=None,
                        entity_type=None,
                        memory_ids=ids,
                        source="direct_memory_relation",
                    )
                )
        return scopes

    @traced("memory_db.get_related_memory_ids")
    async def get_related_memory_ids(
        self,
        ctx: MemoryRequestContext,
        memory_ids: list[str],
        *,
        limit_per_memory: int = 3,
        max_candidates: int = 10,
    ) -> list[dict[str, str]]:
        """Read one-hop ``RELATES_TO`` memory neighbors through Neo4j."""

        if not memory_ids:
            return []
        rows = await self._clients.neo4j.get_related_memory_ids(
            ctx.project_id,
            memory_ids,
            limit_per_memory=limit_per_memory,
            max_candidates=max_candidates,
        )
        related: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            memory_id = str(row.get("memory_id") or "")
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            related.append(
                {
                    "memory_id": memory_id,
                    "seed_memory_id": str(row.get("seed_memory_id") or ""),
                }
            )
        return related

    @traced("memory_db.list_memories")
    async def list_memories(
        self,
        ctx: MemoryRequestContext,
        *,
        filters: SearchFilter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
    ) -> tuple[list[MemoryView], Any | None]:
        """List memories in the request project."""

        qfilter = search_filter_to_qdrant(ctx, filters)
        records, next_cursor = await self._clients.qdrant.scroll_memories(
            ctx.project_id,
            filter_=qfilter,
            limit=limit,
            cursor=cursor,
        )
        return [to_memory_view_from_record(record) for record in records], next_cursor

    @traced("memory_db.list_memory_records")
    async def list_memory_records(
        self,
        ctx: MemoryRequestContext,
        *,
        filters: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """List raw memory records in the request project."""

        return await self._clients.qdrant.scroll_memories(
            ctx.project_id,
            filter_=filters,
            limit=limit,
            cursor=cursor,
            with_vectors=False,
        )

    @traced("memory_db.list_add_records")
    async def list_add_records(
        self,
        ctx: MemoryRequestContext,
        *,
        filters: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """List add records in the request project."""

        return await self._add_records.list(
            ctx.project_id,
            filters=filters,
            limit=limit,
            cursor=cursor,
        )

    @traced("memory_db.get_add_records_by_ids")
    async def get_add_records_by_ids(self, ctx: MemoryRequestContext, add_record_ids: list[str]) -> list[QdrantRecord]:
        """Load add records by IDs in the request project."""

        return await self._add_records.get_by_ids(ctx.project_id, add_record_ids)

    @traced("memory_db.list_search_records")
    async def list_search_records(
        self,
        ctx: MemoryRequestContext,
        *,
        filters: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """List search records in the request project."""

        return await self._clients.qdrant.scroll_search_records(
            ctx.project_id,
            filter_=filters,
            limit=limit,
            cursor=cursor,
        )

    async def get_memory_lineage(self, ctx: MemoryRequestContext, memory_ids: list[str]) -> dict[str, list[str]]:
        """Load directed DERIVED_FROM ancestor IDs for returned search memories."""

        seed_ids = list(dict.fromkeys(memory_id for memory_id in memory_ids if memory_id))
        if not seed_ids:
            return {}

        rows = await self._clients.neo4j.get_memory_lineage(ctx.project_id, seed_ids)
        result: dict[str, list[str]] = {}
        for row in rows:
            memory_id = str(row.get("memory_id") or "")
            if not memory_id:
                continue
            result[memory_id] = _dedupe_ids(row.get("derived_from_memory_ids", []))
        return result

    @traced("memory_db.search_dense", record_args=False)
    async def search_dense(
        self,
        ctx: MemoryRequestContext,
        req: MemoryDbSearchQuery,
        *,
        query_vector: list[float],
    ) -> MemoryDbSearchResult:
        """Search memories by dense vector."""

        hits = await self._clients.qdrant.search_memory_dense(
            ctx.project_id,
            query_vector,
            filter_=search_filter_to_qdrant(ctx, _active_memory_filter(req.filters)),
            limit=req.top_k,
        )
        result = to_search_result(req.query, hits)
        return result

    @traced("memory_db.search_entities_dense", record_args=False)
    async def search_entities_dense(
        self,
        ctx: MemoryRequestContext,
        *,
        query: str,
        query_vector: list[float],
        filters: SearchFilter | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> EntitySearchResult:
        """Search entities by dense vector through the project-scoped DB reader."""

        hits = await self._clients.qdrant.search_entity_dense(
            ctx.project_id,
            query_vector,
            filter_=search_filter_to_qdrant(ctx, filters, target="entity"),
            limit=self._entity_raw_limit(limit),
            score_threshold=score_threshold,
        )
        return to_entity_search_result(query, _dedupe_entity_search_records(hits, limit=limit))

    @traced("memory_db.search_entity_property_memories", record_args=False)
    async def search_entity_property_memories(
        self,
        ctx: MemoryRequestContext,
        *,
        query_vector: list[float],
        entity_id: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> MemoryDbSearchResult:
        """Search property memories belonging to a specific entity by dense vector."""

        entity_filter = SearchFilter(
            must=[
                FieldCondition(field="entity_id", op="match", value=entity_id),
            ]
        )
        hits = await self._clients.qdrant.search_memory_dense(
            ctx.project_id,
            query_vector,
            filter_=search_filter_to_qdrant(ctx, _active_memory_filter(entity_filter)),
            limit=limit,
            score_threshold=score_threshold,
        )
        return to_search_result(entity_id, hits)

    @traced("memory_db.search_sparse", record_args=False)
    async def search_sparse(
        self,
        ctx: MemoryRequestContext,
        req: MemoryDbSearchQuery,
        *,
        indices: list[int],
        values: list[float],
    ) -> MemoryDbSearchResult:
        """Search memories by sparse vector."""

        hits = await self._clients.qdrant.search_memory_sparse(
            ctx.project_id,
            SparseVectorData(indices=indices, values=values),
            filter_=search_filter_to_qdrant(ctx, _active_memory_filter(req.filters)),
            limit=req.top_k,
        )
        result = to_search_result(req.query, hits)
        return result

    @traced("memory_db.search_hybrid", record_args=False)
    async def search_hybrid(
        self,
        ctx: MemoryRequestContext,
        req: MemoryDbSearchQuery,
        *,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        dense_limit: int | None = None,
        sparse_limit: int | None = None,
    ) -> MemoryDbSearchResult:
        """Search memories by Qdrant-side dense/sparse hybrid fusion."""

        hits = await self._clients.qdrant.search_memory_hybrid(
            ctx.project_id,
            dense_vector,
            _to_sparse_vector_data(sparse_vector),
            filter_=search_filter_to_qdrant(ctx, _active_memory_filter(req.filters)),
            limit=req.top_k,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
        )
        result = to_search_result(req.query, hits)
        return result

    @traced("memory_db.search_by_filter")
    async def search_by_filter(self, ctx: MemoryRequestContext, req: MemoryDbSearchQuery) -> MemoryDbSearchResult:
        """Fallback search that returns project-scoped records without vector scoring."""

        records, _ = await self._clients.qdrant.scroll_memories(
            ctx.project_id,
            filter_=search_filter_to_qdrant(ctx, _active_memory_filter(req.filters)),
            limit=req.top_k,
        )
        memories = [to_memory_view_from_record(record) for record in records]
        result = MemoryDbSearchResult(
            query=req.query,
            hits=[
                {
                    "memory_id": memory.memory_id,
                    "score": 0.0,
                    "memory": memory,
                    "source": "filter",
                    "rank": index,
                }
                for index, memory in enumerate(memories, start=1)
            ],
            total=len(memories),
            debug={"mode": "filter_fallback"},
        )
        return result

    @traced("memory_db.search_entities_sparse", record_args=False)
    async def search_entities_sparse(
        self,
        ctx: MemoryRequestContext,
        *,
        indices: list[int],
        values: list[float],
        filters: SearchFilter | None = None,
        limit: int = 10,
    ) -> EntitySearchResult:
        """Search entities by sparse BM25 vector."""

        hits = await self._clients.qdrant.search_entity_sparse(
            ctx.project_id,
            SparseVectorData(indices=indices, values=values),
            filter_=search_filter_to_qdrant(ctx, filters, target="entity"),
            limit=self._entity_raw_limit(limit),
        )
        return to_entity_search_result("", _dedupe_entity_search_records(hits, limit=limit))

    @traced("memory_db.search_entities_hybrid", record_args=False)
    async def search_entities_hybrid(
        self,
        ctx: MemoryRequestContext,
        *,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        filters: SearchFilter | None = None,
        limit: int = 10,
    ) -> EntitySearchResult:
        """Search entities by Qdrant-side dense/sparse hybrid fusion."""

        hits = await self._clients.qdrant.search_entity_hybrid(
            ctx.project_id,
            dense_vector,
            _to_sparse_vector_data(sparse_vector),
            filter_=search_filter_to_qdrant(ctx, filters, target="entity"),
            limit=self._entity_raw_limit(limit),
        )
        return to_entity_search_result("", _dedupe_entity_search_records(hits, limit=limit))

    def _entity_raw_limit(self, limit: int) -> int:
        requested = max(1, limit)
        factor = max(1, self._entity_search_field_overfetch_factor)
        return max(requested * factor, requested + 20)

    @traced("memory_db.search_entity_property_memories_sparse", record_args=False)
    async def search_entity_property_memories_sparse(
        self,
        ctx: MemoryRequestContext,
        *,
        indices: list[int],
        values: list[float],
        entity_id: str,
        limit: int = 5,
    ) -> MemoryDbSearchResult:
        """Search property memories of a specific entity by sparse BM25 vector."""

        entity_filter = SearchFilter(
            must=[
                FieldCondition(field="entity_id", op="match", value=entity_id),
            ]
        )
        hits = await self._clients.qdrant.search_memory_sparse(
            ctx.project_id,
            SparseVectorData(indices=indices, values=values),
            filter_=search_filter_to_qdrant(ctx, _active_memory_filter(entity_filter)),
            limit=limit,
        )
        return to_search_result(entity_id, hits)

    @traced("memory_db.get_entity_with_memories")
    async def get_entity_with_memories(
        self,
        ctx: MemoryRequestContext,
        entity_id: str,
        *,
        filters: SearchFilter | None = None,
    ) -> TemporalEntity | None:
        """Load one entity and hydrate its active memories.

        Args:
            ctx: Tenant and project context used for storage isolation.
            entity_id: Entity identifier to load from the vector store.
            filters: Optional additional memory filters.

        Returns:
            A hydrated temporal entity, or None when the entity does not exist.
        """

        record = await self._clients.qdrant.get_entity(ctx.project_id, entity_id)
        if record is None:
            return None
        entity_view = to_entity_view_from_record(record)
        memories, _ = await self.list_memories(
            ctx,
            filters=combine_search_filters(
                SearchFilter(
                    must=[
                        FieldCondition(field="entity_id", op="match", value=entity_id),
                        FieldCondition(field="status", op="match", value="active"),
                    ]
                ),
                filters,
            ),
            limit=500,
        )
        return TemporalEntity.from_views(entity_view, memories)

    @traced("memory_db.get_entity_neighbors")
    async def get_entity_neighbors(
        self,
        ctx: MemoryRequestContext,
        entity_id: str,
        *,
        direction: str = "both",
        rel_type: str | None = None,
        limit: int | None = None,
    ) -> list[EntityView]:
        """Return bounded one-hop entity graph neighbors hydrated from Qdrant."""

        reserved = _reserve_database_budget(ctx, limit=limit)
        if reserved <= 0:
            return []

        try:
            neighbor_rows = await self._clients.neo4j.get_entity_neighbors(
                ctx.project_id,
                entity_id,
                direction=direction,
                rel_type=rel_type,
                limit=reserved,
            )
        except Exception:
            _refund_database_budget(ctx, qdrant_reads=reserved, neo4j_rows=reserved)
            raise

        if len(neighbor_rows) < reserved:
            _refund_database_budget(
                ctx, qdrant_reads=reserved - len(neighbor_rows), neo4j_rows=reserved - len(neighbor_rows)
            )
        if not neighbor_rows:
            return []

        row_by_id: dict[str, dict[str, Any]] = {}
        for row in neighbor_rows:
            nid = row.get("entity_id")
            if nid:
                row_by_id[nid] = row
        neighbor_ids = list(row_by_id)[:reserved]
        qdrant_refund = min(len(neighbor_rows), reserved) - len(neighbor_ids)
        if qdrant_refund > 0:
            _refund_database_budget(ctx, qdrant_reads=qdrant_refund)
        if len(neighbor_ids) < len(row_by_id):
            logger.warning(
                "entity_neighbor_hydration_truncated",
                entity_id=entity_id,
                returned_rows=len(neighbor_rows),
                unique_neighbors=len(row_by_id),
                hydrated_neighbors=len(neighbor_ids),
            )

        async def _fetch(nid: str) -> EntityView | None:
            try:
                record = await self._clients.qdrant.get_entity(ctx.project_id, nid)
            except Exception:
                return None
            if record is None:
                return None
            if ctx.user_id and record.payload.get("user_id") != ctx.user_id:
                return None
            view = to_entity_view_from_record(record)
            row = row_by_id.get(nid, {})
            view.metadata["_relation"] = row.get("relation", "")
            view.metadata["_direction"] = row.get("direction", "")
            return view

        results = await asyncio.gather(*[_fetch(nid) for nid in neighbor_ids])
        return [v for v in results if v is not None]


def _dedupe_ids(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
