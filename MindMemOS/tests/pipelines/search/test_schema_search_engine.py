from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.typing.memory import MemoryRequestContext, MemoryView
from mindmemos.typing.memory_db import MemoryDbSearchHit, MemoryDbSearchResult
from mindmemos.typing.service import SearchPipelineInput

from mindmemos.components.memory_modeling.schema import TemporalEntity
from mindmemos.config import init_config, reset_config
from mindmemos.config.algo.search import SearchConfig
from mindmemos.pipelines.search.schema import SchemaSearchEngine


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeEntityManager:
    def get_all_dicts(self):
        return [{"entity_type": "person", "properties": ["preference"]}]


class FakeQueryBuilder:
    def __init__(self, *, current_time_mode: str, min_time_window_days: int | None) -> None:
        self.current_time_mode = current_time_mode
        self.min_time_window_days = min_time_window_days

    def all_property_filter(self) -> dict[str, list[str]]:
        return {"person": ["preference"]}

    async def extract_time_from_query(self, query: str, **kwargs):
        return ("2026-01-01", "2026-01-02")


class FakeExpander:
    def __init__(self) -> None:
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return [TemporalEntity(entity_id="ent-1", name="Kai", entity_type="person", description="Kai likes Qdrant.")]


class EmptyExpander:
    def __init__(self) -> None:
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return []


class FakeFallbackReader:
    def __init__(self) -> None:
        self.calls = []

    async def search_sparse(self, context, query, *, indices, values):
        self.calls.append(
            {
                "context": context,
                "query": query,
                "indices": indices,
                "values": values,
            }
        )
        memory = MemoryView(
            memory_id="mem-1",
            project_id=context.project_id,
            content="Kai likes Qdrant.",
            mem_type="fact",
            status="active",
            created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        )
        return MemoryDbSearchResult(
            query=query.query,
            hits=[MemoryDbSearchHit(memory_id="mem-1", score=0.91, memory=memory, source="bm25", rank=1)],
            total=1,
        )


@pytest.fixture
def config_scope():
    init_config(config_path="config/mindmemos/dev.example.yaml")
    try:
        yield
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_schema_search_engine_uses_schema_search_config_not_agentic_round_config(config_scope) -> None:
    search_config = SearchConfig()
    search_config.schema_search = replace(
        search_config.schema_search,
        multi_hop=4,
        current_time_mode="system",
        min_time_window_days=9,
        include_edges=True,
        output_max_edge_num=2,
    )
    search_config.agentic = replace(search_config.agentic, top_k_per_round=99, top_n_per_round=88, num_hops=7)
    query_builder = FakeQueryBuilder(
        current_time_mode=search_config.schema_search.current_time_mode,
        min_time_window_days=search_config.schema_search.min_time_window_days,
    )
    expander = FakeExpander()
    engine = SchemaSearchEngine(
        search_config=search_config,
        llm_client=SimpleNamespace(),
        embed_client=SimpleNamespace(),
        rerank_client=None,
        entity_manager=FakeEntityManager(),
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
    )
    engine._query_builder = query_builder
    engine._expander = expander

    result = await engine.search_candidates(
        SearchPipelineInput(query="Qdrant", search_pipeline="schema", top_k=5, rerank=True),
        make_context(),
    )

    assert result[0].id == "ent-1"
    assert query_builder.current_time_mode == "system"
    call = expander.calls[0]
    assert call["num_hops"] == 4
    assert call["top_k"] is None
    assert call["top_n"] == 5
    assert call["use_reranker"] is None


@pytest.mark.asyncio
async def test_schema_search_engine_falls_back_to_direct_memory_when_schema_results_are_empty(config_scope) -> None:
    search_config = SearchConfig()
    query_builder = FakeQueryBuilder(
        current_time_mode=search_config.schema_search.current_time_mode,
        min_time_window_days=search_config.schema_search.min_time_window_days,
    )
    expander = EmptyExpander()
    reader = FakeFallbackReader()
    engine = SchemaSearchEngine(
        search_config=search_config,
        llm_client=SimpleNamespace(),
        embed_client=SimpleNamespace(),
        rerank_client=None,
        entity_manager=FakeEntityManager(),
        db_reader=reader,
        db_writer=SimpleNamespace(),
    )
    engine._query_builder = query_builder
    engine._expander = expander

    result = await engine.search_candidates(
        SearchPipelineInput(query="Qdrant", search_pipeline="schema", top_k=3, rerank=True),
        make_context(),
    )

    assert len(result) == 1
    assert result[0].id == "mem-1"
    assert result[0].memory == "Kai likes Qdrant."
    assert result[0].memory_type == "fact"
    assert result[0].last_update_at == "2026-01-02 03:04:05"
    assert result[0].event_time is None
    assert len(expander.calls) == 1
    assert len(reader.calls) == 1
    fallback_call = reader.calls[0]
    assert fallback_call["context"].project_id == "proj-1"
    assert fallback_call["query"].top_k == 3
    assert fallback_call["query"].mode == "bm25"
    assert fallback_call["query"].ranking == "score"
    assert fallback_call["query"].filters is None
    assert fallback_call["indices"]
    assert fallback_call["values"]
