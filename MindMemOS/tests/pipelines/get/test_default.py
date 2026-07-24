from datetime import UTC, datetime
from typing import Any

import pytest
from mindmemos.typing.memory import MemoryRequestContext, MemoryView, SearchFilter
from mindmemos.typing.service import GetPipelineInput

from mindmemos.pipelines.get import DefaultGetPipeline


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeReader:
    def __init__(self, memories: list[MemoryView]) -> None:
        self.memories = memories
        self.calls: list[tuple[str, SearchFilter | None, int]] = []

    async def list_memories(
        self,
        ctx: MemoryRequestContext,
        *,
        filters: SearchFilter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
    ) -> tuple[list[MemoryView], Any | None]:
        self.calls.append((ctx.project_id, filters, limit))
        return self.memories, None


class FakeWriter:
    pass


@pytest.mark.asyncio
async def test_get_returns_hydrated_memory_items() -> None:
    memory = MemoryView(
        memory_id="mem-1",
        project_id="proj-1",
        content="Project uses Qdrant.",
        mem_type="fact",
        status="active",
        metadata={"source_timestamp_ms": 1700000000000},
        validate_from=datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, 8, 30, tzinfo=UTC),
        update_at=datetime(2026, 1, 2, 9, 45, tzinfo=UTC),
    )
    reader = FakeReader([memory])
    pipeline = DefaultGetPipeline(db_reader=reader, db_writer=FakeWriter())

    result = await pipeline.get(GetPipelineInput(top_k=5), make_context())

    assert len(reader.calls) == 1
    project_id, sent_filter, limit = reader.calls[0]
    assert project_id == "proj-1"
    assert limit == 5
    # An "active" status condition is always appended to the parsed filter.
    assert any(cond.field == "status" and cond.value == "active" for cond in sent_filter.must)
    assert result.status == "ok"
    assert result.message is None
    assert len(result.memories) == 1
    assert result.memories[0].id == "mem-1"
    assert result.memories[0].memory == "Project uses Qdrant."
    assert result.memories[0].memory_type == "fact"
    assert result.memories[0].last_update_at == "2026-01-02 09:45:00"
    assert result.memories[0].event_time == "2023-11-14 22:13:20"
    assert result.memories[0].source_timestamp == "2023-11-14 22:13:20"


@pytest.mark.asyncio
async def test_get_prefers_resolved_event_date_over_source_time() -> None:
    memory = MemoryView(
        memory_id="mem-1",
        project_id="proj-1",
        content="On 2023-01-19 (yesterday), Caroline visited an LGBTQ support group.",
        mem_type="episodic",
        status="active",
        metadata={
            "resolved_event_date": "2023-01-19",
            "event_time_text": "yesterday",
            "source_timestamp_ms": 1674172800000,
        },
        validate_from=datetime(2023, 1, 20, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, 8, 30, tzinfo=UTC),
    )
    pipeline = DefaultGetPipeline(db_reader=FakeReader([memory]), db_writer=FakeWriter())

    result = await pipeline.get(GetPipelineInput(filters={"memory_id": "mem-1"}), make_context())

    assert result.memories[0].event_time == "2023-01-19 00:00:00"
    assert result.memories[0].source_timestamp == "2023-01-20 00:00:00"


@pytest.mark.asyncio
async def test_get_returns_ok_with_empty_list_when_nothing_matches() -> None:
    reader = FakeReader([])
    pipeline = DefaultGetPipeline(db_reader=reader, db_writer=FakeWriter())

    result = await pipeline.get(GetPipelineInput(), make_context())

    # top_k defaults to None -> no explicit limit passed, reader uses its default (50).
    assert reader.calls[0][2] == 50
    assert result.status == "ok"
    assert result.memories == []
    assert result.message is None
