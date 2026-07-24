import asyncio
from types import SimpleNamespace

import pytest
from mindmemos.pipelines.memory_db.add_record_store import AddRecordStore
from mindmemos.pipelines.memory_db.reader import MemoryDbReader
from mindmemos.pipelines.memory_db.writer import MemoryDbWriter

from mindmemos.config import init_config, reset_config
from mindmemos.infra.db import close_database_clients, get_database_clients, reset_database_clients
from mindmemos.infra.db import registry as db_registry


@pytest.fixture(autouse=True)
def db_registry_config():
    init_config(config_path="config/mindmemos/dev.example.yaml")
    reset_database_clients()
    try:
        yield
    finally:
        reset_database_clients()
        reset_config()


def test_get_database_clients_returns_sync_provider_without_running_loop():
    first = get_database_clients()
    second = get_database_clients()

    assert first is second


def test_memory_db_boundaries_can_be_constructed_without_running_loop(monkeypatch):
    def fail_create_clients():
        raise AssertionError("client creation should be deferred until async DB access")

    monkeypatch.setattr(db_registry, "_create_database_clients", fail_create_clients)

    AddRecordStore()
    MemoryDbReader()
    MemoryDbWriter()


def test_database_clients_provider_resolves_clients_per_event_loop(monkeypatch):
    created: list[object] = []

    def fake_create_clients():
        clients = SimpleNamespace(qdrant=object(), neo4j=object(), skill=object())
        created.append(clients)
        return clients

    monkeypatch.setattr(db_registry, "_create_database_clients", fake_create_clients)

    async def get_once():
        provider = get_database_clients()
        assert provider is get_database_clients()
        first_qdrant = provider.qdrant
        assert provider.qdrant is first_qdrant
        return first_qdrant

    first_loop_qdrant = asyncio.run(get_once())
    second_loop_qdrant = asyncio.run(get_once())

    assert first_loop_qdrant is not second_loop_qdrant
    assert [clients.qdrant for clients in created] == [first_loop_qdrant, second_loop_qdrant]


@pytest.mark.asyncio
async def test_close_database_clients_closes_current_loop_clients(monkeypatch):
    closed: list[object] = []
    created: list[object] = []

    class FakeClients:
        def __init__(self) -> None:
            self.qdrant = object()
            self.neo4j = object()
            self.skill = object()
            created.append(self)

        async def close(self):
            closed.append(self)

    monkeypatch.setattr(db_registry, "_create_database_clients", FakeClients)

    provider = get_database_clients()
    qdrant = provider.qdrant
    await close_database_clients()

    assert closed == created[:1]
    assert provider.qdrant is not qdrant
