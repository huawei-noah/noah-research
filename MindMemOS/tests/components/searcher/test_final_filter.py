import pytest
from mindmemos.components.searcher.final_filter import SearchFinalFilter
from mindmemos.typing.llm import RerankHit, RerankResponse
from mindmemos.typing.service import MemorySearchItem


class FakeRerankClient:
    def __init__(
        self,
        indices: list[int] | None = None,
        *,
        fail: bool = False,
        has_external_model: bool = True,
        available: bool = True,
    ) -> None:
        self.available = available
        self.has_external_model = has_external_model
        self.indices = indices or [1, 0]
        self.fail = fail
        self.calls = 0

    async def rerank(self, query: str, documents: list[str], top_n: int):
        self.calls += 1
        if self.fail:
            raise RuntimeError("rerank unavailable")
        return RerankResponse(
            results=[
                RerankHit(index=index, relevance_score=1.0 - offset * 0.01)
                for offset, index in enumerate(self.indices[:top_n])
            ],
            model="fake",
        )


def item(memory_id: str, text: str) -> MemorySearchItem:
    return MemorySearchItem(id=memory_id, memory=text, last_update_at="")


@pytest.mark.asyncio
async def test_final_filter_truncates_without_rerank() -> None:
    final_filter = SearchFinalFilter()
    candidates = [item("a", "A"), item("b", "B"), item("c", "C")]

    result = await final_filter.apply(query="q", candidates=candidates, top_k=2, rerank=False)

    assert [entry.id for entry in result] == ["a", "b"]


@pytest.mark.asyncio
async def test_final_filter_returns_all_when_top_k_is_none() -> None:
    final_filter = SearchFinalFilter()
    candidates = [item("a", "A"), item("b", "B"), item("c", "C")]

    result = await final_filter.apply(query="q", candidates=candidates, top_k=None, rerank=False)

    assert [entry.id for entry in result] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_final_filter_reranks_before_truncation() -> None:
    reranker = FakeRerankClient(indices=[2, 0, 1])
    final_filter = SearchFinalFilter(rerank_client=reranker)
    candidates = [item("a", "A"), item("b", "B"), item("c", "C")]

    result = await final_filter.apply(query="q", candidates=candidates, top_k=2, rerank=True)

    assert [entry.id for entry in result] == ["c", "a"]
    assert reranker.calls == 1


@pytest.mark.asyncio
async def test_final_filter_ignores_negative_rerank_indices() -> None:
    async def fake_rerank(rerank_client, query: str, documents: list[str], top_n: int) -> list[int]:
        return [-1, 0]

    final_filter = SearchFinalFilter(rerank_client=FakeRerankClient(), rerank_fn=fake_rerank)
    candidates = [item("a", "A"), item("b", "B"), item("c", "C")]

    result = await final_filter.apply(query="q", candidates=candidates, top_k=2, rerank=True)

    assert [entry.id for entry in result] == ["a"]


@pytest.mark.asyncio
async def test_final_filter_falls_back_to_input_order_when_rerank_fails() -> None:
    reranker = FakeRerankClient(fail=True)
    final_filter = SearchFinalFilter(rerank_client=reranker)
    candidates = [item("a", "A"), item("b", "B"), item("c", "C")]

    result = await final_filter.apply(query="q", candidates=candidates, top_k=2, rerank=True)

    assert [entry.id for entry in result] == ["a", "b"]
    assert reranker.calls == 1


@pytest.mark.asyncio
async def test_final_filter_skips_rerank_without_external_model() -> None:
    reranker = FakeRerankClient(indices=[1, 0], has_external_model=False)
    final_filter = SearchFinalFilter(rerank_client=reranker)
    candidates = [item("a", "A"), item("b", "B")]

    result = await final_filter.apply(query="q", candidates=candidates, top_k=2, rerank=True)

    assert [entry.id for entry in result] == ["a", "b"]
    assert reranker.calls == 0


@pytest.mark.asyncio
async def test_final_filter_does_not_rerank_when_request_disables_it() -> None:
    reranker = FakeRerankClient(indices=[1, 0])
    final_filter = SearchFinalFilter(rerank_client=reranker)
    candidates = [item("a", "A"), item("b", "B")]

    result = await final_filter.apply(query="q", candidates=candidates, top_k=2, rerank=False)

    assert [entry.id for entry in result] == ["a", "b"]
    assert reranker.calls == 0


@pytest.mark.asyncio
async def test_score_threshold_filters_low_score_results() -> None:
    async def fake_rerank_with_scores(client, query, docs, top_n):
        return [(0, 0.95), (1, 0.5), (2, 0.3)]

    final_filter = SearchFinalFilter(
        rerank_client=FakeRerankClient(),
        rerank_with_scores_fn=fake_rerank_with_scores,
    )
    candidates = [item("a", "high"), item("b", "medium"), item("c", "low")]

    result = await final_filter.apply(
        query="q", candidates=candidates, top_k=10, rerank=True, score_threshold=0.6,
    )

    assert [entry.id for entry in result] == ["a"]


@pytest.mark.asyncio
async def test_score_threshold_ignored_when_rerank_false() -> None:
    final_filter = SearchFinalFilter()
    candidates = [item("a", "A"), item("b", "B"), item("c", "C")]

    result = await final_filter.apply(
        query="q", candidates=candidates, top_k=10, rerank=False, score_threshold=0.99,
    )

    assert [entry.id for entry in result] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_score_threshold_none_uses_indices_only_rerank() -> None:
    reranker = FakeRerankClient(indices=[2, 0, 1])
    final_filter = SearchFinalFilter(rerank_client=reranker)
    candidates = [item("a", "A"), item("b", "B"), item("c", "C")]

    result = await final_filter.apply(
        query="q", candidates=candidates, top_k=3, rerank=True, score_threshold=None,
    )

    assert [entry.id for entry in result] == ["c", "a", "b"]
    assert reranker.calls == 1
