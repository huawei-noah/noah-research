import pytest
from mindmemos.components.searcher.rerank import rerank, rerank_with_scores
from mindmemos.typing.llm import RerankHit, RerankResponse


class FakeRerankClient:
    async def rerank(self, query: str, documents: list[str], top_n: int) -> RerankResponse:
        return RerankResponse(
            results=[
                RerankHit(index=-1, relevance_score=0.9),
                RerankHit(index=2, relevance_score=0.8),
                RerankHit(index=3, relevance_score=0.7),
                RerankHit(index=0, relevance_score=0.6),
            ],
            model="fake",
        )


@pytest.mark.asyncio
async def test_rerank_filters_negative_and_out_of_range_indices() -> None:
    indices = await rerank(FakeRerankClient(), "query", ["a", "b", "c"], top_n=4)

    assert indices == [2, 0]


@pytest.mark.asyncio
async def test_rerank_with_scores_filters_negative_and_out_of_range_indices() -> None:
    scored = await rerank_with_scores(FakeRerankClient(), "query", ["a", "b", "c"], top_n=4)

    assert scored == [(2, 0.8), (0, 0.6)]
