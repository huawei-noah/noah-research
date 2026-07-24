"""Neo4j primitive graph store."""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, RoutingControl
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from ...config import Neo4jConfig
from ...logging import get_logger
from ..retry import retry_delay
from .concurrency import AsyncClientConcurrencyLimiter, capped_db_client_concurrency
from .errors import MemoryDbValidationError
from .models import EntityNode, GraphRelationship, MemoryNode, NodeRef, SourceNode
from .schema import NEO4J_SCHEMA_STATEMENTS

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_SCHEMA_STATEMENTS = NEO4J_SCHEMA_STATEMENTS
logger = get_logger(__name__)


@dataclass(frozen=True)
class _RelationshipBatchKey:
    source_label: str
    target_label: str
    rel_type: str
    source_key_fields: tuple[str, ...]
    target_key_fields: tuple[str, ...]
    rel_key_fields: tuple[str, ...]

    @classmethod
    def from_relationship(cls, rel: GraphRelationship) -> "_RelationshipBatchKey":
        return cls(
            rel.source.label,
            rel.target.label,
            rel.rel_type,
            tuple(rel.source.key.keys()),
            tuple(rel.target.key.keys()),
            tuple(rel.key.keys()),
        )


class Neo4jStore:
    """Thin Neo4j adapter with no memory business logic."""

    def __init__(self, cfg: Neo4jConfig, *, driver: AsyncDriver | None = None) -> None:
        self._cfg = cfg
        max_client_concurrency = capped_db_client_concurrency(
            cfg.max_client_concurrency,
            cap=cfg.max_client_concurrency_cap,
        )
        raw_driver = driver or AsyncGraphDatabase.driver(
            cfg.uri,
            auth=(cfg.username, cfg.password),
            max_connection_lifetime=cfg.max_connection_lifetime,
            max_connection_pool_size=min(cfg.max_connection_pool_size, max_client_concurrency),
            connection_acquisition_timeout=cfg.connection_acquisition_timeout,
        )
        self._driver = AsyncClientConcurrencyLimiter(raw_driver, max_concurrency=max_client_concurrency)

    async def ensure_schema(self, statements: list[str] | None = None) -> None:
        """Create constraints and indexes."""

        if not self._cfg.auto_create_schema:
            return
        for statement in statements or list(DEFAULT_SCHEMA_STATEMENTS):
            await self._run_write(statement)

    async def upsert_memory_node(self, node: MemoryNode) -> None:
        """Upsert one ``Memory`` node using the documented node key."""

        await self.upsert_nodes(memories=[node])

    async def upsert_entity_node(self, node: EntityNode) -> None:
        """Upsert one ``Entity`` node using the documented node key."""

        await self.upsert_nodes(entities=[node])

    async def upsert_source_node(self, node: SourceNode) -> None:
        """Upsert one ``Source`` node using the documented node key."""

        await self.upsert_nodes(sources=[node])

    async def upsert_nodes(
        self,
        *,
        memories: list[MemoryNode] | None = None,
        entities: list[EntityNode] | None = None,
        sources: list[SourceNode] | None = None,
    ) -> None:
        """Upsert graph nodes in label-specific batches."""

        if memories:
            await self._upsert_nodes(
                label="Memory",
                key_fields=("project_id", "memory_id"),
                rows=[
                    {
                        "project_id": node.project_id,
                        "memory_id": node.memory_id,
                        "properties": _normalize_properties({"content": node.content, "status": "active"}),
                    }
                    for node in memories
                ],
            )
        if entities:
            await self._upsert_nodes(
                label="Entity",
                key_fields=("project_id", "entity_id"),
                rows=[
                    {
                        "project_id": node.project_id,
                        "entity_id": node.entity_id,
                        "properties": _normalize_properties(
                            {
                                "entity_name": node.entity_name,
                                "description": node.description,
                                "entity_type": node.entity_type,
                            }
                        ),
                    }
                    for node in entities
                ],
            )
        if sources:
            await self._upsert_nodes(
                label="Source",
                key_fields=("project_id", "source_id"),
                rows=[
                    {
                        "project_id": node.project_id,
                        "source_id": node.source_id,
                        "properties": _normalize_properties({"parsed_content_path": node.parsed_content_path}),
                    }
                    for node in sources
                ],
            )

    async def get_entity_neighbors(
        self,
        project_id: str,
        entity_id: str,
        *,
        direction: str = "both",
        rel_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query graph neighbors for an entity with a hard limit."""
        if limit is not None and limit <= 0:
            return []

        if direction == "out":
            pattern = "(e)-[r]->(n)"
        elif direction == "in":
            pattern = "(e)<-[r]-(n)"
        else:
            pattern = "(e)-[r]-(n)"

        rel_filter = f"AND type(r) = $rel_type" if rel_type else ""
        limit_clause = "LIMIT $limit" if limit is not None else ""
        query = f"""
        MATCH {pattern}
        WHERE e.`project_id` = $project_id
          AND e.`entity_id` = $entity_id
          AND n:`Entity`
          AND n.`project_id` = $project_id
          {rel_filter}
        RETURN DISTINCT n.`entity_id` AS entity_id,
               n.`entity_name` AS entity_name,
               n.`entity_type` AS entity_type,
               type(r) AS relation,
               CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END AS direction
        {limit_clause}
        """
        params: dict[str, Any] = {"project_id": project_id, "entity_id": entity_id}
        if rel_type:
            params["rel_type"] = rel_type
        if limit is not None:
            params["limit"] = limit
        return await self.run_read(query, **params)

    async def upsert_relationship(self, rel: GraphRelationship) -> None:
        """Upsert one relationship by source, target, type, and key properties."""

        await self.upsert_relationships([rel])

    async def upsert_relationships(self, relationships: list[GraphRelationship]) -> None:
        """Upsert relationships in shape-specific batches."""

        batches: dict[_RelationshipBatchKey, list[GraphRelationship]] = defaultdict(list)
        for rel in relationships:
            batch_key = _RelationshipBatchKey.from_relationship(rel)
            batches[batch_key].append(rel)

        for batch_key, rels in batches.items():
            await self._upsert_relationship_batch(batch_key, rels)

    async def delete_memory_node(self, project_id: str, memory_id: str, *, detach: bool = True) -> None:
        """Delete one ``Memory`` node."""

        await self.delete_node(
            NodeRef(label="Memory", key={"project_id": project_id, "memory_id": memory_id}),
            detach=detach,
        )

    async def archive_memory_node(self, project_id: str, memory_id: str, *, reason: str | None = None) -> None:
        """Mark one ``Memory`` node as archived without removing graph edges."""

        properties: dict[str, Any] = {
            "status": "archived",
            "status_changed_at": datetime.now(UTC),
        }
        if reason:
            properties["delete_reason"] = reason
        await self._upsert_node(
            label="Memory",
            key={"project_id": project_id, "memory_id": memory_id},
            properties=properties,
        )

    async def get_related_memory_ids(
        self,
        project_id: str,
        memory_ids: list[str],
        *,
        limit_per_memory: int = 3,
        max_candidates: int = 10,
    ) -> list[dict[str, Any]]:
        """Return one-hop ``RELATES_TO`` memory neighbors for seed memories."""

        if not memory_ids or limit_per_memory <= 0 or max_candidates <= 0:
            return []
        query = """
        MATCH (m:`Memory` {project_id: $project_id})-[r:`RELATES_TO`]-(n:`Memory` {project_id: $project_id})
        WHERE m.memory_id IN $memory_ids
          AND n.memory_id <> m.memory_id
        WITH m.memory_id AS seed_memory_id,
             n.memory_id AS memory_id,
             type(r) AS relation
        ORDER BY seed_memory_id, memory_id
        WITH seed_memory_id, collect({memory_id: memory_id, relation: relation})[..$limit_per_memory] AS neighbors
        UNWIND neighbors AS neighbor
        RETURN DISTINCT neighbor.memory_id AS memory_id,
               seed_memory_id,
               neighbor.relation AS relation
        LIMIT $max_candidates
        """
        return await self.run_read(
            query,
            project_id=project_id,
            memory_ids=memory_ids,
            limit_per_memory=limit_per_memory,
            max_candidates=max_candidates,
        )

    async def get_memory_lineage(self, project_id: str, memory_ids: list[str]) -> list[dict[str, Any]]:
        """Return directed DERIVED_FROM ancestors for each memory."""

        return await self.run_read(
            """
            MATCH (m:Memory {project_id: $project_id})
            WHERE m.memory_id IN $memory_ids
            OPTIONAL MATCH (m)-[:DERIVED_FROM*1..]->(ancestor:Memory {project_id: $project_id})
            WITH m.memory_id AS memory_id,
                 ancestor.memory_id AS ancestor_id,
                 ancestor.memory_id AS ancestor_sort_key
            WITH memory_id, ancestor_id, max(ancestor_sort_key) AS ancestor_sort_key
            ORDER BY ancestor_sort_key
            WITH memory_id, [row IN collect({id: ancestor_id}) WHERE row.id IS NOT NULL | row.id] AS derived_from_memory_ids
            RETURN memory_id, derived_from_memory_ids
            """,
            project_id=project_id,
            memory_ids=memory_ids,
        )

    async def update_memory_content(self, project_id: str, memory_id: str, content: str) -> None:
        """Sync the ``content`` property of one ``Memory`` node after a content edit.

        Mirrors the content stored in Qdrant; ``MENTIONS`` edges are left untouched
        (entity re-extraction is handled by a dedicated command, not here).
        """

        await self._upsert_node(
            label="Memory",
            key={"project_id": project_id, "memory_id": memory_id},
            properties={"content": content},
        )

    async def delete_node(self, ref: NodeRef, *, detach: bool = True) -> None:
        """Delete one node."""

        label = _safe_identifier(ref.label)
        key_expr, params = _match_properties("key", ref.key)
        delete_clause = "DETACH DELETE n" if detach else "DELETE n"
        query = f"""
        MATCH (n:{label} {{{key_expr}}})
        {delete_clause}
        """
        await self._run_write(query, **params)

    async def run_read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Run a read query and return record dictionaries."""

        max_retries = max(1, self._cfg.read_max_retries)
        base_delay = self._cfg.read_retry_base_delay
        for attempt in range(1, max_retries + 1):
            try:
                result = await self._driver.execute_query(
                    query,
                    params,
                    routing_=RoutingControl.READ,
                    database_=self._cfg.database,
                )
                return [record.data() for record in result.records]
            except Exception as exc:
                if not _is_retryable_driver_error(exc):
                    raise
                if attempt == max_retries:
                    logger.error("neo4j_read_failed", attempt=attempt, max_retries=max_retries)
                    raise
                delay = retry_delay(base_delay, attempt)
                logger.warning("neo4j_read_retryable_error", attempt=attempt, retry_after=delay, error=str(exc))
                await asyncio.sleep(delay)

        raise RuntimeError("unreachable neo4j read retry state")

    async def _run_write(self, query: str, **params: Any) -> None:
        """Run an internal write query with transient-error retry."""

        max_retries = max(1, self._cfg.write_max_retries)
        base_delay = self._cfg.write_retry_base_delay
        for attempt in range(1, max_retries + 1):
            try:
                await self._driver.execute_query(
                    query,
                    params,
                    routing_=RoutingControl.WRITE,
                    database_=self._cfg.database,
                )
                return
            except Exception as exc:
                if not _is_retryable_driver_error(exc):
                    raise
                if attempt == max_retries:
                    logger.error("neo4j_write_failed", attempt=attempt, max_retries=max_retries)
                    raise
                delay = retry_delay(base_delay, attempt)
                logger.warning("neo4j_write_retryable_error", attempt=attempt, retry_after=delay, error=str(exc))
                await asyncio.sleep(delay)

    async def close(self) -> None:
        """Close the underlying driver."""

        await self._driver.close()

    async def _upsert_node(self, *, label: str, key: dict[str, Any], properties: dict[str, Any]) -> None:
        if not key:
            raise MemoryDbValidationError("Neo4j node key cannot be empty")
        safe_label = _safe_identifier(label)
        key_expr, params = _match_properties("key", key)
        normalized = _normalize_properties({k: v for k, v in properties.items() if v is not None})
        query = f"""
        MERGE (n:{safe_label} {{{key_expr}}})
        SET n += $properties
        """
        await self._run_write(query, **params, properties=normalized)

    async def _upsert_nodes(self, *, label: str, key_fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
        if not key_fields:
            raise MemoryDbValidationError("Neo4j node key cannot be empty")
        if not rows:
            return

        safe_label = _safe_identifier(label)
        key_expr = ", ".join(f"{_safe_identifier(field)}: row.{field}" for field in key_fields)
        query = f"""
        UNWIND $rows AS row
        MERGE (n:{safe_label} {{{key_expr}}})
        SET n += row.properties
        """
        await self._run_write(query, rows=rows)

    async def _upsert_relationship_batch(
        self,
        batch_key: _RelationshipBatchKey,
        relationships: list[GraphRelationship],
    ) -> None:
        if not relationships:
            return

        source_label = _safe_identifier(batch_key.source_label)
        target_label = _safe_identifier(batch_key.target_label)
        rel_type = _safe_identifier(batch_key.rel_type)
        source_expr = _row_match_properties("source", batch_key.source_key_fields)
        target_expr = _row_match_properties("target", batch_key.target_key_fields)
        rel_expr = _row_match_properties("rel", batch_key.rel_key_fields)
        rel_selector = f"[r:{rel_type} {{{rel_expr}}}]" if rel_expr else f"[r:{rel_type}]"
        query = f"""
        UNWIND $rows AS row
        MATCH (source:{source_label} {{{source_expr}}})
        MATCH (target:{target_label} {{{target_expr}}})
        MERGE (source)-{rel_selector}->(target)
        SET r += row.properties
        """
        rows = [
            {
                "source": dict(rel.source.key),
                "target": dict(rel.target.key),
                "rel": dict(rel.key),
                "properties": _normalize_properties(rel.properties),
            }
            for rel in relationships
        ]
        await self._run_write(query, rows=rows)


def _safe_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise MemoryDbValidationError(f"invalid Neo4j identifier: {value!r}")
    return f"`{value}`"


def _match_properties(prefix: str, values: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not values:
        return "", {}
    parts: list[str] = []
    params: dict[str, Any] = {}
    for key, value in values.items():
        safe_key = _safe_identifier(key)
        param_name = f"{prefix}_{key}"
        parts.append(f"{safe_key}: ${param_name}")
        params[param_name] = value
    return ", ".join(parts), params


def _row_match_properties(prefix: str, fields: tuple[str, ...]) -> str:
    if not fields:
        return ""
    return ", ".join(f"{_safe_identifier(field)}: row.{prefix}.{field}" for field in fields)


def _normalize_properties(properties: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in properties.items():
        if value is None:
            continue
        if isinstance(value, dict):
            normalized[f"{key}_json"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif isinstance(value, list) and any(isinstance(item, dict) for item in value):
            normalized[f"{key}_json"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            normalized[key] = value
    return normalized


def _is_retryable_driver_error(exc: Exception) -> bool:
    return isinstance(exc, (TransientError, ServiceUnavailable, SessionExpired, TimeoutError))
