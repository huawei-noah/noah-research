from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.components.searcher.final_filter import SearchFinalFilter
from mindmemos.pipelines.search.base import SearchEngineOptions
from mindmemos.pipelines.search.pipeline import SearchPipelineImpl
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
    name = "default"

    def __init__(self) -> None:
        self.inputs: list[SearchPipelineInput] = []

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        self.inputs.append(inp)
        return [MemorySearchItem(id="mem-1", memory=f"{inp.search_pipeline}:{inp.query}", last_update_at="")]


class ExplodingAgenticWrapper:
    async def run(self, inp, context, engine):
        raise AssertionError("agentic wrapper should not run for non-agentic search")



@pytest.mark.asyncio
async def test_search_pipeline_uses_selected_engine_without_agentic_wrapper() -> None:
    engine = FakeEngine()
    pipeline = SearchPipelineImpl(
        engines={"default": engine},
        agentic_wrapper=ExplodingAgenticWrapper(),
        final_filter=SearchFinalFilter(),
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
    )

    result = await pipeline.search(SearchPipelineInput(query="Qdrant", search_pipeline="default"), make_context())

    assert result.memories[0].id == "mem-1"
    assert engine.inputs[0].agentic is False



@pytest.mark.asyncio
async def test_search_pipeline_rejects_unknown_strategy_with_available_names() -> None:
    pipeline = SearchPipelineImpl(
        engines={"default": FakeEngine()},
        final_filter=SearchFinalFilter(),
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="Available strategies: default"):
        await pipeline.search(SearchPipelineInput(query="Qdrant", search_pipeline="schema"), make_context())
