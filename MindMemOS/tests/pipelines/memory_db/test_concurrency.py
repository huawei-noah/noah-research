import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from mindmemos.config import init_config, reset_config
from mindmemos.infra.db import reset_database_clients
from mindmemos.pipelines.memory_db.reader import MemoryDbReader
from mindmemos.typing.memory import MemoryRequestContext


class TrackingQdrant:
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.active = 0
        self.max_active = 0

    async def get_memory(self, project_id: str, memory_id: str, *, with_vectors: bool = False):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self.release.wait()
            return None
        finally:
            self.active -= 1


class TrackingNeo4j:
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.active = 0
        self.max_active = 0

    async def get_entity_neighbors(
        self,
        project_id: str,
        entity_id: str,
        *,
        direction: str = "both",
        rel_type=None,
        limit=None,
    ):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self.release.wait()
            return []
        finally:
            self.active -= 1


@pytest.fixture(autouse=True)
def memory_db_concurrency_config(tmp_path):
    # Client concurrency is yaml-only config (no longer env-overridable), so the
    # low caps used to exercise the limiter are written into a temporary config file.
    base = yaml.safe_load(Path("config/mindmemos/dev.example.yaml").read_text(encoding="utf-8")) or {}
    database = base.setdefault("database", {})
    database.setdefault("qdrant", {})["max_client_concurrency"] = 3
    database.setdefault("neo4j", {})["max_client_concurrency"] = 3
    config_path = tmp_path / "concurrency.yaml"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")

    init_config(config_path=str(config_path))
    reset_database_clients()
    try:
        yield
    finally:
        reset_database_clients()
        reset_config()


@pytest.mark.asyncio
async def test_memory_db_reader_uses_injected_clients_without_extra_qdrant_limiter():
    release = asyncio.Event()
    qdrant = TrackingQdrant(release)
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    readers = [MemoryDbReader(clients=clients) for _ in range(4)]

    tasks = [asyncio.create_task(reader.get_memory(_ctx(), f"mem-{index}")) for index, reader in enumerate(readers * 3)]
    await _wait_for_active(qdrant, len(tasks))

    assert qdrant.max_active == len(tasks)

    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_memory_db_reader_does_not_limit_injected_qdrant_or_neo4j_clients():
    qdrant_release = asyncio.Event()
    neo4j_release = asyncio.Event()
    qdrant = TrackingQdrant(qdrant_release)
    neo4j = TrackingNeo4j(neo4j_release)
    clients = SimpleNamespace(qdrant=qdrant, neo4j=neo4j)
    reader = MemoryDbReader(clients=clients)

    qdrant_tasks = [asyncio.create_task(reader.get_memory(_ctx(), f"mem-{index}")) for index in range(10)]
    neo4j_tasks = [asyncio.create_task(reader.get_entity_neighbors(_ctx(), f"ent-{index}")) for index in range(10)]
    await _wait_for_active(qdrant, len(qdrant_tasks))
    await _wait_for_active(neo4j, len(neo4j_tasks))

    assert qdrant.max_active == len(qdrant_tasks)
    assert neo4j.max_active == len(neo4j_tasks)
    assert qdrant.active + neo4j.active == len(qdrant_tasks) + len(neo4j_tasks)

    qdrant_release.set()
    neo4j_release.set()
    await asyncio.gather(*qdrant_tasks, *neo4j_tasks)


def _ctx() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-concurrency",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
    )


async def _wait_for_active(client, expected: int) -> None:
    for _ in range(100):
        if client.active == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {expected} active calls, got {client.active}")
