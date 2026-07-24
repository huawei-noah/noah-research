from __future__ import annotations

from dataclasses import dataclass

import pytest
from mindmemos.pipelines.search.agentic.base import SearchToolRequest, SearchToolResult
from mindmemos.pipelines.search.agentic.loop import AgenticLoop
from mindmemos.typing.memory import MemoryRequestContext

from mindmemos.components.memory_modeling.schema import TemporalEntity
from mindmemos.config.algo.search import AgenticConfig


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeTool:
    name = "schema_search"

    def __init__(self) -> None:
        self.requests: list[SearchToolRequest] = []

    async def search(self, request: SearchToolRequest) -> SearchToolResult:
        self.requests.append(request)
        entity = TemporalEntity(entity_id=f"ent-{len(self.requests)}", name="User", entity_type="person")
        entity.modify_property("preference", request.query, "2026-01-01 00:00:00")
        return SearchToolResult(entities=[entity])


@dataclass
class FakeRouter:
    tool: FakeTool

    def select(self, requested_tool: str | None = None) -> FakeTool:
        return self.tool


class FakePlanner:
    async def generate_next_queries(self, **kwargs):
        return [
            {
                "query": f"follow up {len(kwargs['query_history'])}",
                "time_range": ("2026-01-01 00:00:00", "2026-02-01 00:00:00"),
            }
        ]


class FakeSufficiency:
    def __init__(self) -> None:
        self.calls = 0

    async def evaluate_sufficiency(self, **kwargs):
        self.calls += 1
        return False, "missing", ["more"]

    async def filter_entities_by_relevance(self, entities, query):
        return entities, []


@pytest.mark.asyncio
async def test_agentic_loop_adds_time_relaxed_query_after_second_round() -> None:
    tool = FakeTool()
    loop = AgenticLoop(
        config=AgenticConfig(max_rounds=3, num_hops=2),
        tool_router=FakeRouter(tool),
        planner=FakePlanner(),
        sufficiency=FakeSufficiency(),
    )

    await loop.run(
        query="what did I discuss",
        context=make_context(),
        initial_time_window=("2026-01-01 00:00:00", "2026-02-01 00:00:00"),
    )

    assert [request.query for request in tool.requests] == [
        "what did I discuss",
        "follow up 1",
        "follow up 2",
        "what did I discuss",
    ]
    assert [request.num_hops for request in tool.requests] == [2, 1, 1, 1]
    relaxed_request = tool.requests[-1]
    assert relaxed_request.time_window is None
    assert relaxed_request.allow_time_extraction is False


@pytest.mark.asyncio
async def test_agentic_loop_uses_two_hops_only_for_first_round() -> None:
    tool = FakeTool()
    loop = AgenticLoop(
        config=AgenticConfig(max_rounds=2, num_hops=4),
        tool_router=FakeRouter(tool),
        planner=FakePlanner(),
        sufficiency=FakeSufficiency(),
    )

    await loop.run(query="what did I discuss", context=make_context())

    assert [request.num_hops for request in tool.requests] == [2, 1]
