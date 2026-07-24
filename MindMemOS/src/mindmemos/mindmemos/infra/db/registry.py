"""Lazy low-level database client registry."""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass
from typing import Any

from ...config import get_config
from .collections import SkillVersionRepository
from .neo4j import Neo4jStore
from .qdrant import QdrantStore


@dataclass(slots=True)
class _LoopDatabaseClients:
    """Event-loop-scoped async database clients used by pipelines."""

    qdrant: QdrantStore
    neo4j: Neo4jStore
    skill: SkillVersionRepository

    async def close(self) -> None:
        """Close all underlying database clients.

        ``skill`` shares ``qdrant``'s engine and does not own a connection, so
        closing ``qdrant`` releases both.
        """

        await self.qdrant.close()
        await self.neo4j.close()


class DatabaseClients:
    """Synchronous provider for the current event loop's async database clients.

    The provider itself is safe to construct and pass around from synchronous
    code. Accessing ``qdrant``, ``neo4j`` or ``skill`` resolves the real clients
    for the currently running event loop.
    """

    @property
    def qdrant(self) -> QdrantStore:
        """Return Qdrant clients for the currently running event loop."""

        return _get_loop_database_clients().qdrant

    @property
    def neo4j(self) -> Neo4jStore:
        """Return Neo4j clients for the currently running event loop."""

        return _get_loop_database_clients().neo4j

    @property
    def skill(self) -> SkillVersionRepository:
        """Return skill repository clients for the currently running event loop."""

        return _get_loop_database_clients().skill

    async def close(self) -> None:
        """Close database clients for the currently running event loop."""

        await close_database_clients()


_database_clients = DatabaseClients()
_database_clients_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopDatabaseClients] = (
    weakref.WeakKeyDictionary()
)


def get_database_clients() -> DatabaseClients:
    """Get the process-wide database client provider.

    The returned object can be stored from synchronous code. Its attributes
    resolve to real async clients for the running event loop when used.
    """

    return _database_clients


def _get_loop_database_clients() -> _LoopDatabaseClients:
    """Get low-level database clients for the current event loop."""

    loop = asyncio.get_running_loop()
    clients = _database_clients_by_loop.get(loop)
    if clients is None:
        clients = _create_database_clients()
        _database_clients_by_loop[loop] = clients
    return clients


def _create_database_clients() -> _LoopDatabaseClients:
    cfg = get_config().database
    qdrant = QdrantStore(cfg.qdrant)
    return _LoopDatabaseClients(
        qdrant=qdrant,
        neo4j=Neo4jStore(cfg.neo4j),
        skill=SkillVersionRepository(cfg.qdrant, engine=qdrant.engine),
    )


def resolve_database_clients(clients: Any | None = None) -> Any:
    """Return provided database clients or the current event-loop clients."""

    return clients if clients is not None else get_database_clients()


async def close_database_clients() -> None:
    """Close and forget database clients for the current event loop."""

    loop = asyncio.get_running_loop()
    clients = _database_clients_by_loop.pop(loop, None)
    if clients is not None:
        await clients.close()


def reset_database_clients() -> None:
    """Forget database client registry entries, mainly for tests and config refreshes.

    Call ``close_database_clients`` first when live clients may hold network
    resources.
    """

    _database_clients_by_loop.clear()
