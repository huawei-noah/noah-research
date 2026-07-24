import asyncio
from types import SimpleNamespace

import pytest
from mindmemos.llm.rerank import RerankClient, _extract_terms, _keyword_score


class FailingRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def arerank(self, **kwargs):
        self.calls += 1
        raise RuntimeError("reranker unavailable")


class HangingRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def arerank(self, **kwargs):
        import asyncio

        self.calls += 1
        await asyncio.sleep(1)
        raise RuntimeError("should be cancelled by rerank timeout")


class TrackingRouter:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def arerank(self, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return SimpleNamespace(
            results=[{"index": 0, "relevance_score": 1.0}],
            model="tracking-reranker",
        )


@pytest.mark.asyncio
async def test_rerank_router_failure_falls_back_to_keyword() -> None:
    router = FailingRouter()
    client = RerankClient(router)

    response = await client.rerank("hello world", ["hello there", "goodbye world", "hello world"], top_n=2)

    assert router.calls == 1
    assert response.model == "keyword_fallback"
    assert response.results[0].index == 2


@pytest.mark.asyncio
async def test_rerank_router_timeout_falls_back_to_keyword() -> None:
    router = HangingRouter()
    client = RerankClient(router, request_timeout=0.01)

    response = await client.rerank("hello world", ["goodbye", "hello world"], top_n=2)

    assert router.calls == 1
    assert response.model == "keyword_fallback"
    assert response.results[0].index == 1


@pytest.mark.asyncio
async def test_batched_rerank_router_failures_fall_back_to_keyword() -> None:
    router = FailingRouter()
    client = RerankClient(router, max_batch_size=1)

    response = await client.rerank("hello", ["hello there", "goodbye", "hello world"], top_n=2)

    assert router.calls == 3
    assert response.model == "keyword_fallback"
    for hit in response.results:
        assert hit.index in (0, 2)


@pytest.mark.asyncio
async def test_batched_rerank_defaults_to_one_concurrent_batch() -> None:
    router = TrackingRouter()
    client = RerankClient(router, max_batch_size=1)

    response = await client.rerank("hello", ["a", "b", "c", "d", "e"], top_n=3)

    assert router.max_active == 1
    assert response.model == "rerank_batched"


def test_rerank_with_router_reports_external_model() -> None:
    client = RerankClient(FailingRouter())

    assert client.available
    assert client.has_external_model


def test_rerank_without_router_has_no_external_model() -> None:
    client = RerankClient(None)

    assert client.available
    assert not client.has_external_model


@pytest.mark.asyncio
async def test_disabled_keyword_fallback_preserves_input_order() -> None:
    client = RerankClient(None, use_keyword_fallback=False)

    response = await client.rerank("query", ["first", "second"], top_n=2)

    assert not client.available
    assert not client.has_external_model
    assert response.model == "identity"
    assert [hit.index for hit in response.results] == [0, 1]


@pytest.mark.asyncio
async def test_keyword_fallback_without_router() -> None:
    client = RerankClient(None)

    response = await client.rerank("python programming", ["python is great", "java code", "programming in python"], top_n=2)

    assert response.model == "keyword_fallback"
    assert response.results[0].index == 2


def test_extract_terms_filters_stopwords() -> None:
    terms = _extract_terms("what is the best programming language for data science")
    assert "what" not in terms
    assert "is" not in terms
    assert "the" not in terms
    assert "for" not in terms
    assert "best" in terms
    assert "programming" in terms
    assert "data" in terms
    assert "science" in terms


def test_extract_terms_handles_chinese() -> None:
    terms = _extract_terms("hello 你好 world 世界")
    assert "hello" in terms
    assert "world" in terms
    assert "你好" in terms
    assert "世界" in terms


def test_keyword_score_full_coverage() -> None:
    query_terms = {"python", "programming"}
    score = _keyword_score(query_terms, "python programming is great")
    assert score == pytest.approx(1.0, abs=0.2)


def test_keyword_score_partial_coverage() -> None:
    query_terms = {"python", "programming"}
    score = _keyword_score(query_terms, "java programming is great")
    assert 0.4 < score < 0.7


def test_keyword_score_no_overlap() -> None:
    query_terms = {"python", "programming"}
    score = _keyword_score(query_terms, "java and rust are great")
    assert score == 0.0
