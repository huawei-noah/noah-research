"""Rerank client with litellm Router backend and keyword-similarity fallback."""

from __future__ import annotations

import asyncio
import re
from time import perf_counter
from typing import TYPE_CHECKING, Any

from ..logging import get_logger, traced
from ..typing import RerankHit, RerankResponse
from .router import usage_tokens

if TYPE_CHECKING:
    from litellm import Router

logger = get_logger(__name__)

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "and", "or", "but", "nor", "not", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "into",
    "about", "between", "through", "during", "before", "after",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "when", "where", "how", "why",
})


class RerankClient:
    """Rerank client that routes through litellm.Router.

    Fallback chain:
    1. litellm Router rerank endpoint (if configured)
    2. Keyword overlap scoring (no external API needed)
    3. Identity ordering (preserve input order)
    """

    ALIAS = "rerank"

    def __init__(
        self,
        router: Router | None,
        *,
        max_query_length: int = 500,
        max_doc_length: int = 500,
        max_batch_size: int = 40,
        max_concurrent_batches: int = 1,
        request_timeout: float = 5.0,
        use_keyword_fallback: bool = True,
    ) -> None:
        self._use_keyword_fallback = use_keyword_fallback
        self._router = router
        self._has_model = router is not None
        self._default_model = self.ALIAS
        self._max_query_length = max_query_length
        self._max_doc_length = max_doc_length
        self._max_batch_size = max_batch_size
        self._max_concurrent_batches = max(1, max_concurrent_batches)
        self._request_timeout = request_timeout

    @property
    def available(self) -> bool:
        """Whether any reranking capability is available."""
        return self._has_model or self._use_keyword_fallback

    @property
    def has_external_model(self) -> bool:
        """Whether a configured external reranker endpoint is active."""
        return self._has_model

    @traced("llm.rerank")
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        **kwargs: Any,
    ) -> RerankResponse:
        """Rerank documents and return document indexes in relevance order."""
        if not documents:
            return RerankResponse(results=[], model="none")

        top_n = min(top_n, len(documents))

        if self._has_model and self._router is not None:
            try:
                return await self._rerank_via_router(query, documents, top_n, **kwargs)
            except Exception:
                return await self._fallback_rerank(query, documents, top_n)

        return await self._fallback_rerank(query, documents, top_n)

    async def _fallback_rerank(self, query: str, documents: list[str], top_n: int) -> RerankResponse:
        if self._use_keyword_fallback:
            return self._rerank_via_keyword(query, documents, top_n)

        return self._identity_rerank(documents, top_n)

    async def _rerank_via_router(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        **kwargs: Any,
    ) -> RerankResponse:
        truncated_query = query[: self._max_query_length]
        truncated_docs = [doc[: self._max_doc_length] for doc in documents]

        if len(truncated_docs) <= self._max_batch_size:
            return await self._rerank_single_batch(truncated_query, truncated_docs, top_n, **kwargs)

        return await self._rerank_batched(truncated_query, truncated_docs, top_n, **kwargs)

    async def _rerank_single_batch(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        **kwargs: Any,
    ) -> RerankResponse:
        start = perf_counter()
        try:
            resp = await asyncio.wait_for(
                self._router.arerank(
                    model=self._default_model,
                    query=query,
                    documents=documents,
                    top_n=top_n,
                    **kwargs,
                ),
                timeout=self._request_timeout,
            )
        except Exception as exc:
            logger.info(
                "litellm_call",
                kind="rerank",
                model=self._default_model,
                status="error",
                latency_ms=round((perf_counter() - start) * 1000, 2),
                error=str(exc),
            )
            raise

        hits = []
        for item in getattr(resp, "results", []) or []:
            if isinstance(item, dict):
                hits.append(RerankHit(index=item.get("index", 0), relevance_score=item.get("relevance_score", 0.0)))
            else:
                hits.append(
                    RerankHit(
                        index=getattr(item, "index", 0),
                        relevance_score=getattr(item, "relevance_score", 0.0),
                    )
                )

        hits = [hit for hit in hits if 0 <= hit.index < len(documents)]
        if top_n > 0 and not hits:
            raise RuntimeError("rerank returned no valid indices")

        usage = usage_tokens(getattr(resp, "usage", None))
        model = getattr(resp, "model", self._default_model) or self._default_model
        logger.info(
            "litellm_call",
            kind="rerank",
            model=self._default_model,
            status="ok",
            latency_ms=round((perf_counter() - start) * 1000, 2),
        )
        return RerankResponse(results=hits[:top_n], model=model, usage=usage)

    async def _rerank_batched(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        **kwargs: Any,
    ) -> RerankResponse:
        import asyncio

        batches: list[list[str]] = []
        offsets: list[int] = []
        for start in range(0, len(documents), self._max_batch_size):
            batch = documents[start : start + self._max_batch_size]
            batches.append(batch)
            offsets.append(start)

        semaphore = asyncio.Semaphore(self._max_concurrent_batches)

        async def run_batch(batch: list[str]) -> RerankResponse:
            async with semaphore:
                return await self._rerank_single_batch(query, batch, min(top_n, len(batch)), **kwargs)

        tasks = [run_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_hits: list[RerankHit] = []
        for batch_resp, offset in zip(batch_results, offsets, strict=False):
            if isinstance(batch_resp, Exception):
                continue
            for hit in batch_resp.results:
                all_hits.append(RerankHit(index=hit.index + offset, relevance_score=hit.relevance_score))

        all_hits.sort(key=lambda h: h.relevance_score, reverse=True)
        if not all_hits:
            raise RuntimeError("all rerank batches failed")
        return RerankResponse(results=all_hits[:top_n], model="rerank_batched")

    @staticmethod
    def _rerank_via_keyword(
        query: str,
        documents: list[str],
        top_n: int,
    ) -> RerankResponse:
        query_terms = _extract_terms(query)
        if not query_terms:
            return RerankClient._identity_rerank(documents, top_n)

        hits: list[RerankHit] = []
        for idx, doc in enumerate(documents):
            score = _keyword_score(query_terms, doc)
            hits.append(RerankHit(index=idx, relevance_score=score))

        hits.sort(key=lambda h: h.relevance_score, reverse=True)
        return RerankResponse(results=hits[:top_n], model="keyword_fallback")

    @staticmethod
    def _identity_rerank(documents: list[str], top_n: int) -> RerankResponse:
        return RerankResponse(
            results=[RerankHit(index=i, relevance_score=1.0 - i * 0.001) for i in range(min(top_n, len(documents)))],
            model="identity",
        )


def _extract_terms(text: str) -> set[str]:
    """Tokenize and filter stopwords."""
    return {t for t in re.findall(r"[a-z0-9一-鿿]+", text.lower()) if t not in _STOPWORDS}


def _keyword_score(query_terms: set[str], document: str) -> float:
    """Query-term coverage score with position and phrase bonuses.

    Returns a value in [0, 1+] where 1.0 means all query terms found.
    """
    doc_lower = document.lower()
    doc_terms = set(re.findall(r"[a-z0-9一-鿿]+", doc_lower))

    overlap = query_terms & doc_terms
    if not overlap:
        return 0.0

    coverage = len(overlap) / len(query_terms)

    phrase_bonus = 0.0
    query_sorted = sorted(overlap)
    for i in range(len(query_sorted) - 1):
        bigram = f"{query_sorted[i]} {query_sorted[i + 1]}"
        if bigram in doc_lower:
            phrase_bonus += 0.1

    return min(coverage + phrase_bonus, 2.0)
