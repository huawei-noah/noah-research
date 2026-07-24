from __future__ import annotations

import pytest
from mindmemos.pipelines.search.agentic.base import SearchToolRequest
from mindmemos.pipelines.search.agentic.wrapper import EngineSearchTool
from mindmemos.pipelines.search.base import SearchEngineOptions
from mindmemos.typing.memory import MemoryRequestContext
from mindmemos.typing.service import MemorySearchItem, SearchPipelineInput


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.inputs: list[SearchPipelineInput] = []
        self.options: list[SearchEngineOptions | None] = []

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        self.inputs.append(inp)
        self.options.append(options)
        return [MemorySearchItem(id="mem-1", memory="Kai uses Qdrant.", last_update_at="")]


@pytest.mark.asyncio
async def test_engine_search_tool_drops_schema_only_filters_for_default_engine() -> None:
    engine = FakeEngine("default")
    tool = EngineSearchTool(
        engine=engine,
        template=SearchPipelineInput(query="original"),
        recall_top_k=7,
        result_top_n=4,
        use_reranker=False,
    )

    result = await tool.search(
        SearchToolRequest(
            query="what does Kai use",
            original_query="what does Kai use",
            time_window=None,
            num_hops=2,
            context=make_context(),
            filters={"project_id": "proj-1", "user_id": "user-1"},
        )
    )

    assert result.entities[0].entity_id == "mem-1"
    assert engine.inputs[0].query == "what does Kai use"
    assert engine.inputs[0].top_k == 4
    assert engine.inputs[0].filters == {"user_id": "user-1"}
    assert engine.inputs[0].agentic is False
    assert engine.inputs[0].rerank is False
    assert engine.options[0] == SearchEngineOptions(
        num_hops=2,
        recall_top_k=7,
        result_top_n=4,
        use_reranker=False,
    )


@pytest.mark.asyncio
async def test_engine_search_tool_keeps_schema_filters_for_schema_engine() -> None:
    engine = FakeEngine("schema")
    tool = EngineSearchTool(
        engine=engine,
        template=SearchPipelineInput(query="original"),
        recall_top_k=6,
        result_top_n=3,
        use_reranker=True,
    )
    filters = {"project_id": "proj-1", "entity_type": "person"}

    await tool.search(
        SearchToolRequest(
            query="what does Kai use",
            original_query="what does Kai use",
            time_window=None,
            num_hops=1,
            context=make_context(),
            filters=filters,
        )
    )

    assert engine.inputs[0].filters == filters
