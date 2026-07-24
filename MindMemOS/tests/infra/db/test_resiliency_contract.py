import asyncio

import pytest
from mindmemos.infra.db.neo4j import Neo4jStore
from mindmemos.infra.db.qdrant import QdrantStore
from mindmemos.infra.retry import retry_delay
from neo4j.exceptions import TransientError
from qdrant_client.http.exceptions import ResponseHandlingException

from mindmemos.config import Neo4jConfig, QdrantConfig


class _QdrantFailingRetrieveClient:
    def __init__(self, errors: list[Exception]) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def retrieve(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return []

    async def close(self):
        return None


class _QdrantRetryBlockingClient:
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.entered_ids: list[str] = []

    async def retrieve(self, **kwargs):
        self.calls += 1
        ids = kwargs.get("ids") or []
        if ids:
            self.entered_ids.append(ids[0])
        if self.calls == 1:
            raise ResponseHandlingException(RuntimeError("temporary transport failure"))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self.release.wait()
            return []
        finally:
            self.active -= 1


class _Neo4jFailingDriver:
    def __init__(self, errors: list[Exception]) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def execute_query(self, query, params=None, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return _Neo4jResult()

    async def close(self):
        return None


class _Neo4jResult:
    records = []


def test_retry_delay_is_exponential_and_capped() -> None:
    assert retry_delay(0.1, 1) == 0.1
    assert retry_delay(0.1, 3) == 0.4
    assert retry_delay(0.0, 3) == 0.0
    assert retry_delay(10.0, 3) == 10.0


@pytest.mark.asyncio
async def test_qdrant_retry_backoff_does_not_hold_client_concurrency_slot() -> None:
    release = asyncio.Event()
    client = _QdrantRetryBlockingClient(release)
    store = QdrantStore(_qdrant_config(max_retries=2, retry_base_delay=0.2, max_client_concurrency=1), client=client)

    first = asyncio.create_task(store.get_memories("proj", ["mem-1"]))
    await _wait_for_calls(client, 1)

    second = asyncio.create_task(store.get_memories("proj", ["mem-2"]))
    await _wait_for_entered_id(client, "mem-2")

    assert client.max_active == 1

    release.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_qdrant_retries_retryable_transport_errors_only() -> None:
    retryable = ResponseHandlingException(RuntimeError("temporary transport failure"))
    client = _QdrantFailingRetrieveClient([retryable, ValueError("caller bug")])
    store = QdrantStore(_qdrant_config(max_retries=3), client=client)

    with pytest.raises(ValueError, match="caller bug"):
        await store.get_memories("proj", ["mem-1"])

    assert client.calls == 2


@pytest.mark.asyncio
async def test_qdrant_does_not_retry_non_retryable_errors() -> None:
    client = _QdrantFailingRetrieveClient([ValueError("bad request shape")])
    store = QdrantStore(_qdrant_config(max_retries=3), client=client)

    with pytest.raises(ValueError, match="bad request shape"):
        await store.get_memories("proj", ["mem-1"])

    assert client.calls == 1


@pytest.mark.asyncio
async def test_neo4j_retries_transient_errors_only() -> None:
    driver = _Neo4jFailingDriver([TransientError("deadlock"), ValueError("invalid cypher")])
    store = Neo4jStore(_neo4j_config(read_max_retries=3), driver=driver)

    with pytest.raises(ValueError, match="invalid cypher"):
        await store.run_read("MATCH (n) RETURN n")

    assert driver.calls == 2


@pytest.mark.asyncio
async def test_neo4j_does_not_retry_non_retryable_errors() -> None:
    driver = _Neo4jFailingDriver([ValueError("invalid cypher")])
    store = Neo4jStore(_neo4j_config(read_max_retries=3), driver=driver)

    with pytest.raises(ValueError, match="invalid cypher"):
        await store.run_read("MATCH (n) RETURN n")

    assert driver.calls == 1


def _qdrant_config(
    *,
    max_retries: int,
    retry_base_delay: float = 0.0,
    max_client_concurrency: int = 100,
) -> QdrantConfig:
    return QdrantConfig(
        url="http://unused",
        memory_collection="test_memos",
        entity_collection="test_entities",
        source_collection="test_sources",
        add_record_collection="test_add_records",
        search_record_collection="test_search_records",
        vector_size=2,
        max_client_concurrency=max_client_concurrency,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )


def _neo4j_config(*, read_max_retries: int) -> Neo4jConfig:
    return Neo4jConfig(
        uri="bolt://unused",
        read_max_retries=read_max_retries,
        read_retry_base_delay=0.0,
    )


async def _wait_for_calls(client, expected: int) -> None:
    for _ in range(100):
        if client.calls >= expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected at least {expected} calls, got {client.calls}")


async def _wait_for_active(client, expected: int) -> None:
    for _ in range(100):
        if client.active == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {expected} active calls, got {client.active}")


async def _wait_for_entered_id(client, expected: str) -> None:
    for _ in range(100):
        if expected in client.entered_ids:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {expected} to enter client, got {client.entered_ids}")
