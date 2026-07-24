"""Tests for the RecentActivityCollector component."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from mindmemos.components.activity import RecentActivityCollector
from mindmemos.typing.activity import ActivityScope
from qdrant_client import models as qmodels

from mindmemos.errors import ActivityCollectionError
from mindmemos.infra.db import QdrantRecord

WINDOW_END = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)


def _iso(minutes_before: int) -> str:
    return (WINDOW_END - timedelta(minutes=minutes_before)).isoformat()


def _add_record(point_id: str, *, session_id: str | None, submitted: int, memories, messages) -> QdrantRecord:
    return QdrantRecord(
        point_id=point_id,
        payload={
            "project_id": "proj-1",
            "user_id": "user-1",
            "session_id": session_id,
            "request_id": f"req-{point_id}",
            "status": "ok",
            "request_submitted_at": _iso(submitted),
            "task_completed_at": _iso(submitted - 1),
            "messages": messages,
            "memories": memories,
        },
    )


def _search_record(point_id: str, *, session_id: str | None, submitted: int, query: str, memories) -> QdrantRecord:
    return QdrantRecord(
        point_id=point_id,
        payload={
            "project_id": "proj-1",
            "user_id": "user-1",
            "session_id": session_id,
            "request_id": f"req-{point_id}",
            "status": "ok",
            "request_submitted_at": _iso(submitted),
            "task_completed_at": _iso(submitted - 1),
            "query": query,
            "top_k": 5,
            "deep_search": False,
            "memories": memories,
        },
    )


class FakeQdrant:
    def __init__(self, add_records, search_records, *, fail: str | None = None):
        self._add = add_records
        self._search = search_records
        self._fail = fail
        self.add_calls: list[dict] = []
        self.search_calls: list[dict] = []

    async def scroll_add_records(self, project_id, *, filter_=None, limit=50, cursor=None, order_by=None):
        if self._fail == "add":
            raise RuntimeError("boom")
        self.add_calls.append({"project_id": project_id, "filter": filter_, "limit": limit, "order_by": order_by})
        return list(self._add), None

    async def scroll_search_records(self, project_id, *, filter_=None, limit=50, cursor=None, order_by=None):
        if self._fail == "search":
            raise RuntimeError("boom")
        self.search_calls.append({"project_id": project_id, "filter": filter_, "limit": limit, "order_by": order_by})
        return list(self._search), None


def make_scope() -> ActivityScope:
    return ActivityScope(project_id="proj-1", user_id="user-1")


async def _collect(qdrant, **kwargs):
    collector = RecentActivityCollector(qdrant)
    return await collector.collect(make_scope(), window_end=WINDOW_END, lookback=timedelta(hours=24), **kwargs)


@pytest.mark.asyncio
async def test_groups_by_session_and_aligns_search_query() -> None:
    add = _add_record(
        "add-1",
        session_id="sess-1",
        submitted=10,
        memories=[{"operation": "add", "content": "likes coffee", "memory_id": "m-1"}],
        messages=[{"role": "user", "content": "I like coffee", "timestamp": 1}],
    )
    search = _search_record(
        "search-1",
        session_id="sess-1",
        submitted=20,
        query="what does the user drink?",
        memories=[{"id": "m-9", "memory": "drinks tea"}],
    )
    bundle = await _collect(FakeQdrant([add], [search]))

    assert len(bundle.conversations) == 1
    conv = bundle.conversations[0]
    assert conv.session_id == "sess-1"
    assert conv.add_record_ids == ["add-1"]
    assert conv.search_record_ids == ["search-1"]
    # add messages reconstructed into DialogueMessage, search query not mixed in.
    assert [type(m).__name__ for m in conv.messages] == ["DialogueMessage"]
    assert conv.messages[0].content == "I like coffee"

    assert len(conv.search_events) == 1
    event = conv.search_events[0]
    assert event.query == "what does the user drink?"
    assert [r.memory_id for r in event.recalled_memories] == ["m-9"]
    assert event.recalled_memories[0].rank == 0
    assert event.recalled_memories[0].content == "drinks tea"

    assert [w.memory_id for w in conv.written_memories] == ["m-1"]
    assert conv.written_memory_ids == ["m-1"]
    assert conv.feedback_add_record_ids == ["add-1"]
    assert conv.dreaming_add_record_ids == ["add-1"]


@pytest.mark.asyncio
async def test_window_and_filter_injection() -> None:
    qdrant = FakeQdrant([], [])
    bundle = await _collect(qdrant)

    assert bundle.window_end == WINDOW_END
    assert bundle.window_start == WINDOW_END - timedelta(hours=24)
    # scope user_id + status==ok + datetime window pushed into the filter.
    call = qdrant.add_calls[0]
    assert call["limit"] == 2000
    assert isinstance(call["order_by"], qmodels.OrderBy)
    keys = [c.key for c in call["filter"].must if isinstance(c, qmodels.FieldCondition)]
    assert "request_submitted_at" in keys
    assert "user_id" in keys
    assert "status" in keys


@pytest.mark.asyncio
async def test_records_without_session_degrade_to_one_conversation_each() -> None:
    add_a = _add_record("add-a", session_id=None, submitted=5, memories=[], messages=[{"text": "note a"}])
    add_b = _add_record("add-b", session_id=None, submitted=6, memories=[], messages=[{"text": "note b"}])
    bundle = await _collect(FakeQdrant([add_a, add_b], []))

    assert len(bundle.conversations) == 2
    assert {c.add_record_ids[0] for c in bundle.conversations} == {"add-a", "add-b"}


@pytest.mark.asyncio
async def test_written_and_recalled_global_dedup() -> None:
    add_1 = _add_record(
        "add-1",
        session_id="sess-1",
        submitted=10,
        memories=[{"operation": "add", "content": "v1", "memory_id": "m-1"}],
        messages=[],
    )
    add_2 = _add_record(
        "add-2",
        session_id="sess-2",
        submitted=8,
        memories=[{"operation": "update", "content": "v2", "memory_id": "m-1"}],
        messages=[],
    )
    search_1 = _search_record(
        "s-1", session_id="sess-1", submitted=12, query="q1", memories=[{"id": "r-1", "memory": "a"}]
    )
    search_2 = _search_record(
        "s-2", session_id="sess-2", submitted=9, query="q2", memories=[{"id": "r-1", "memory": "a"}]
    )
    bundle = await _collect(FakeQdrant([add_1, add_2], [search_1, search_2]))

    # m-1 deduped globally across two sessions; add_record_ids merged.
    assert [w.memory_id for w in bundle.written_memories] == ["m-1"]
    written = bundle.written_memories[0]
    assert set(written.add_record_ids) == {"add-1", "add-2"}
    assert len(written.payloads) == 2
    # newest record (add-2 at -8min is more recent than add-1 at -10min) wins.
    assert written.operation == "update"
    assert written.content == "v2"

    # r-1 recalled in both search events, deduped globally.
    assert [r.memory_id for r in bundle.recalled_memories] == ["r-1"]
    assert len(bundle.search_events) == 2


@pytest.mark.asyncio
async def test_scroll_failure_raises_component_error() -> None:
    with pytest.raises(ActivityCollectionError):
        await _collect(FakeQdrant([], [], fail="add"))
