"""LLM query rewriting for implicit feedback memory recall."""

from __future__ import annotations

import json

from ...llm import LLMClient, get_llm_client
from ...prompts.EN.feedback import QUERY_REWRITE_PROMPT
from ...typing import SupplementalSearchQuery


class ImplicitFeedbackQueryRewriter:
    """Rewrite user queries for supplemental implicit-feedback memory recall."""

    def __init__(self, *, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    async def rewrite(self, original_query: str) -> SupplementalSearchQuery:
        response = await self._client.chat(
            task="feedback.implicit.rewrite_search_query",
            messages=[
                {"role": "system", "content": QUERY_REWRITE_PROMPT},
                {"role": "user", "content": json.dumps({"query": original_query}, ensure_ascii=False)},
            ],
            format_parser=_parse_supplemental_search_query,
            temperature=0,
        )
        query = response.parsed
        if not isinstance(query, SupplementalSearchQuery):
            msg = "implicit feedback query rewriter expected parsed supplemental query"
            raise TypeError(msg)
        return query

    @property
    def _client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client


def _parse_supplemental_search_query(content: str) -> SupplementalSearchQuery:
    return SupplementalSearchQuery.model_validate_json(_json_object_text(content))


def _json_object_text(content: str) -> str:
    text = content.strip()
    try:
        json.loads(text)
        return text
    except ValueError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        return text[start : end + 1]
