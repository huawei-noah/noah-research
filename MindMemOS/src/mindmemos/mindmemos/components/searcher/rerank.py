"""Rerank component wrapping RerankClient for search pipeline use."""

from __future__ import annotations

from ...llm import RerankClient
from ...logging import get_logger

logger = get_logger(__name__)


async def rerank(
    rerank_client: RerankClient | None,
    query: str,
    documents: list[str],
    top_n: int,
) -> list[int]:
    """Rerank documents and return document indexes in relevance order."""
    if not rerank_client or not documents:
        return list(range(min(top_n, len(documents))))

    response = await rerank_client.rerank(query, documents, top_n)
    indices = [hit.index for hit in response.results if 0 <= hit.index < len(documents)]
    logger.debug("rerank_complete", input_docs=len(documents), output_count=len(indices))
    return indices


async def rerank_with_scores(
    rerank_client: RerankClient | None,
    query: str,
    documents: list[str],
    top_n: int,
) -> list[tuple[int, float]]:
    """Rerank documents and return indexes with relevance scores."""
    if not rerank_client or not documents:
        return [(i, 1.0) for i in range(min(top_n, len(documents)))]

    response = await rerank_client.rerank(query, documents, top_n)
    scored = [(hit.index, hit.relevance_score) for hit in response.results if 0 <= hit.index < len(documents)]
    logger.debug("rerank_with_scores_complete", input_docs=len(documents), output_count=len(scored))
    return scored
