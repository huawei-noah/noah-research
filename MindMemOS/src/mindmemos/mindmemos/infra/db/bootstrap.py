"""Schema bootstrap helpers for low-level database clients."""

from __future__ import annotations

from .registry import DatabaseClients, get_database_clients


async def ensure_database_schema(clients: DatabaseClients | None = None) -> None:
    """Create Qdrant and Neo4j schemas for configured database clients."""

    db_clients = clients or get_database_clients()
    await db_clients.qdrant.ensure_schema()
    await db_clients.neo4j.ensure_schema()
