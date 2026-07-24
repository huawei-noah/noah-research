from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.pipelines.search.default import DefaultSearchEngine
from mindmemos.typing.memory import (
    FieldCondition,
    MemoryRequestContext,
    MemoryView,
)
from mindmemos.typing.memory_db import MemoryDbSearchHit, MemoryDbSearchResult
from mindmemos.typing.service import SearchPipelineInput

from mindmemos.config import TextProcessingConfig


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeReader:
    def __init__(self) -> None:
        self.calls = []

    async def search_sparse(self, context: MemoryRequestContext, req, *, indices, values):
        self.calls.append(SimpleNamespace(context=context, req=req, indices=indices, values=values))
        return MemoryDbSearchResult(
            query=req.query,
            hits=[
                MemoryDbSearchHit(
                    memory_id="mem-1",
                    score=0.8,
                    memory=MemoryView(
                        memory_id="mem-1",
                        project_id=context.project_id,
                        content="Kai uses Qdrant.",
                        mem_type="fact",
                        status="active",
                        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
                    ),
                    source="bm25",
                    rank=1,
                )
            ],
            total=1,
        )


def make_engine(reader: FakeReader) -> DefaultSearchEngine:
    return DefaultSearchEngine(
        db_reader=reader,
        db_writer=SimpleNamespace(),
        text_config=TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        ),
    )


@pytest.mark.asyncio
async def test_search_uses_bm25_sparse_vector_with_default_scope() -> None:
    reader = FakeReader()
    engine = make_engine(reader)

    result = await engine.search_candidates(SearchPipelineInput(query="Qdrant", top_k=3), make_context())

    assert result[0].id == "mem-1"
    assert result[0].memory == "Kai uses Qdrant."
    assert result[0].memory_type == "fact"
    assert result[0].last_update_at == "2026-01-02 03:04:05"

    assert len(reader.calls) == 1
    call = reader.calls[0]
    assert call.req.query == "Qdrant"
    assert call.req.top_k == 3
    assert call.req.mode == "bm25"
    assert call.indices

    # Pipeline injects only status=active; other scope fields come from the public filters DSL.
    must_fields = [c.field for c in call.req.filters.must if isinstance(c, FieldCondition)]
    assert must_fields == ["status"]
    assert call.req.filters.should == []
    assert call.req.filters.must_not == []


@pytest.mark.asyncio
async def test_search_empty_query_tokens_returns_empty_result_without_db_call() -> None:
    reader = FakeReader()
    engine = make_engine(reader)

    result = await engine.search_candidates(SearchPipelineInput(query="   ", top_k=3), make_context())

    assert result == []
    assert reader.calls == []
