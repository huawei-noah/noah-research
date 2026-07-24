from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.components.extractor.vanilla.add_recall import RelatedMemoryRecall
from mindmemos.components.text import SparseVectorEncoder, TextPreprocessor
from mindmemos.config import TextProcessingConfig
from mindmemos.typing import MemoryDbSearchHit, MemoryDbSearchResult
from mindmemos.typing.memory import Entity, MemoryRequestContext, MemoryView


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


def make_text_tools():
    cfg = TextProcessingConfig(
        bm25_use_spacy_lemma=False,
        spacy_en_model="missing_en_model",
        spacy_zh_model="missing_zh_model",
        sparse_hash_dim=128,
    )
    return TextPreprocessor(cfg), SparseVectorEncoder(cfg)


def memory(memory_id: str, content: str, *, metadata: dict | None = None) -> MemoryView:
    return MemoryView(
        memory_id=memory_id,
        project_id="proj-1",
        content=content,
        mem_type="fact",
        status="active",
        metadata=metadata or {},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class FakeReader:
    def __init__(self) -> None:
        self.list_calls = []
        self.sparse_calls = []
        self.listed_memories = []
        self.sparse_hits = []

    async def list_memories(self, context: MemoryRequestContext, *, filters=None, limit=50, cursor=None):
        self.list_calls.append(SimpleNamespace(context=context, filters=filters, limit=limit, cursor=cursor))
        return self.listed_memories, None

    async def search_sparse(self, context: MemoryRequestContext, req, *, indices, values):
        self.sparse_calls.append(SimpleNamespace(context=context, req=req, indices=indices, values=values))
        return MemoryDbSearchResult(query=req.query, hits=self.sparse_hits, total=len(self.sparse_hits))


@pytest.mark.asyncio
async def test_related_memory_recall_marks_exact_hash_duplicate_before_fusion() -> None:
    text_preprocessor, sparse_encoder = make_text_tools()
    preprocessed = text_preprocessor.preprocess_text("Kai uses Qdrant.", segment_id="segment-1")
    preprocessed = preprocessed.model_copy(update={"entities": [Entity(name="Kai")]})
    reader = FakeReader()
    reader.listed_memories = [
        memory("mem-1", "Kai uses Qdrant.", metadata={"content_hash": preprocessed.content_hash}),
    ]

    result = await RelatedMemoryRecall(
        db_reader=reader,
        sparse_encoder=sparse_encoder,
        top_k=5,
    ).recall(make_context(), preprocessed)

    assert result.duplicate is not None
    assert result.duplicate.memory_id == "mem-1"
    assert result.duplicate.source == "hash"
    assert result.candidates[0].memory_id == "mem-1"


@pytest.mark.asyncio
async def test_related_memory_recall_does_not_pass_project_id_as_caller_filter() -> None:
    text_preprocessor, sparse_encoder = make_text_tools()
    preprocessed = text_preprocessor.preprocess_text("Kai uses Qdrant.", segment_id="segment-1")
    reader = FakeReader()
    reader.listed_memories = [memory("mem-existing", "Kai uses something else.")]

    await RelatedMemoryRecall(
        db_reader=reader,
        sparse_encoder=sparse_encoder,
        top_k=5,
    ).recall(make_context(), preprocessed)

    list_filter = reader.list_calls[0].filters
    sparse_filter = reader.sparse_calls[0].req.filters

    assert [condition.field for condition in list_filter.must] == ["user_id", "status"]
    assert [condition.field for condition in sparse_filter.must] == ["user_id", "status"]


@pytest.mark.asyncio
async def test_related_memory_recall_skips_sparse_search_without_active_memories() -> None:
    text_preprocessor, sparse_encoder = make_text_tools()
    preprocessed = text_preprocessor.preprocess_text("Kai uses Qdrant.", segment_id="segment-1")
    reader = FakeReader()

    result = await RelatedMemoryRecall(
        db_reader=reader,
        sparse_encoder=sparse_encoder,
        top_k=5,
    ).recall(make_context(), preprocessed)

    assert result.candidates == []
    assert reader.sparse_calls == []


@pytest.mark.asyncio
async def test_related_memory_recall_combines_entity_and_bm25_candidates() -> None:
    text_preprocessor, sparse_encoder = make_text_tools()
    preprocessed = text_preprocessor.preprocess_text("Kai uses Qdrant.", segment_id="segment-1")
    preprocessed = preprocessed.model_copy(update={"entities": [Entity(name="Kai")]})
    reader = FakeReader()
    reader.listed_memories = [
        memory("mem-entity", "Kai likes vector databases.", metadata={"entities": ["Kai"]}),
    ]
    reader.sparse_hits = [
        MemoryDbSearchHit(
            memory_id="mem-bm25",
            score=0.7,
            memory=memory("mem-bm25", "Qdrant is used for memory search."),
            source="bm25",
            rank=1,
        )
    ]

    result = await RelatedMemoryRecall(
        db_reader=reader,
        sparse_encoder=sparse_encoder,
        top_k=5,
    ).recall(make_context(), preprocessed)

    assert {hit.memory_id for hit in result.candidates} == {"mem-entity", "mem-bm25"}
    assert reader.sparse_calls[0].req.mode == "bm25"
    assert reader.sparse_calls[0].indices
