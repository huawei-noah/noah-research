"""Asynchronous LLM summarization for compacted long-turn middles."""

from __future__ import annotations

import json
from typing import Any

from ....config import VanillaAddConfig
from ....logging import get_logger
from ....prompts.EN.add import LONG_TURN_SUMMARY_PROMPT
from ....typing import TurnCompactionSummary
from .compactor import LongTurnCompactor
from .turn_grouper import _estimate_tokens

logger = get_logger(__name__)


def parse_turn_compaction_summary(content: str) -> TurnCompactionSummary:
    """Parse a structured long-turn summary, tolerating markdown JSON fences."""
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return TurnCompactionSummary.model_validate(json.loads(text))


class LongTurnSummarizer:
    """Summarize long-turn middle text under stable context and output budgets."""

    def __init__(self, config: VanillaAddConfig, llm_client: Any = None) -> None:
        self._config = config
        self._llm_client = llm_client

    async def summarize(self, middle_text: str) -> TurnCompactionSummary:
        """Return one structured summary, recursively reducing oversized input."""
        if not middle_text:
            return TurnCompactionSummary()
        if self._llm_client is None:
            return self._fallback(middle_text)
        try:
            segments = self._split_for_context(middle_text)
            summaries = [await self._summarize_once(segment, mode="segment") for segment in segments]
            while len(summaries) > 1:
                summaries = await self._reduce(summaries)
            return summaries[0]
        except Exception:
            logger.warning("long_turn_summary_failed", exc_info=True)
            return self._fallback(middle_text)

    async def _reduce(self, summaries: list[TurnCompactionSummary]) -> list[TurnCompactionSummary]:
        text = "\n".join(summary.model_dump_json() for summary in summaries)
        segments = self._split_for_context(text)
        return [await self._summarize_once(segment, mode="reduce") for segment in segments]

    async def _summarize_once(self, middle_text: str, *, mode: str) -> TurnCompactionSummary:
        response = await self._llm_client.chat(
            task="memory.add.long_turn_summary",
            messages=[
                {"role": "system", "content": LONG_TURN_SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps({"mode": mode, "middle_text": middle_text}, ensure_ascii=False),
                },
            ],
            format_parser=parse_turn_compaction_summary,
            max_tokens=self._config.compaction_summary_output_token_budget,
        )
        parsed = response.parsed
        if isinstance(parsed, TurnCompactionSummary):
            return parsed
        return TurnCompactionSummary.model_validate(parsed)

    def _split_for_context(self, text: str) -> list[str]:
        budget = self._config.compaction_summary_context_token_budget
        if _estimate_tokens(text) <= budget:
            return [text]

        segments: list[str] = []
        remaining = text
        while remaining:
            end = LongTurnCompactor._prefix_end_for_budget(remaining, budget)
            if end <= 0:
                end = 1
            segments.append(remaining[:end])
            remaining = remaining[end:]
        return segments

    @staticmethod
    def _fallback(middle_text: str) -> TurnCompactionSummary:
        tokens = _estimate_tokens(middle_text)
        return TurnCompactionSummary(general_summary=f"[Compacted middle section omitted: {tokens} tokens]")
