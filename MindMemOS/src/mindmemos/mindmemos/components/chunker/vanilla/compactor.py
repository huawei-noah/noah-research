"""Long-turn compaction for vanilla add pipeline chunking.

When a single turn exceeds the hard turn token budget, the compactor
preserves the head and tail as raw evidence and summarizes the middle
via a dedicated LLM call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ....config import VanillaAddConfig
from ....typing import (
    Turn,
    TurnCompactionResult,
    TurnCompactionSummary,
    TurnMessageRef,
)
from .turn_grouper import _estimate_tokens


@dataclass(frozen=True, slots=True)
class TurnCompactionParts:
    """Deterministic source ranges selected for long-turn compaction."""

    head_text: str
    middle_text: str
    tail_text: str
    head_messages: tuple[TurnMessageRef, ...] = ()
    tail_messages: tuple[TurnMessageRef, ...] = ()


class LongTurnCompactor:
    """Compact oversized turns by preserving head/tail and summarizing middle.

    The compactor keeps the beginning (user intent/setup) and end (final
    answer/decision) as raw extractable evidence. The middle is summarized
    via a dedicated LLM call that preserves context without creating memories.
    """

    def __init__(self, config: VanillaAddConfig | None = None) -> None:
        self._config = config or VanillaAddConfig()

    def needs_compaction(self, turn: Turn) -> bool:
        """Check if a turn exceeds the hard turn token budget."""
        return turn.token_count > self._config.turn_hard_token_budget

    def compact(
        self,
        turn: Turn,
        summarize_fn: SummarizeFn | None = None,
        *,
        summary: TurnCompactionSummary | None = None,
        parts: TurnCompactionParts | None = None,
    ) -> tuple[Turn, TurnCompactionResult]:
        """Compact an oversized turn.

        Args:
            turn: The turn to compact.
            summarize_fn: Callable that takes the middle text and returns
                a TurnCompactionSummary. If None, a stub summary is used.

        Returns:
            Tuple of (compacted Turn, TurnCompactionResult with metadata).
        """
        full_text, _ = self._flatten_extractable_messages(turn)
        full_token_count = _estimate_tokens(full_text)
        selected = parts or self.split(turn)
        head_text = selected.head_text
        middle_text = selected.middle_text
        tail_text = selected.tail_text

        head_token_count = _estimate_tokens(head_text)
        tail_token_count = _estimate_tokens(tail_text)

        if summary is None and summarize_fn and middle_text:
            summary = summarize_fn(middle_text)
        if summary is None:
            summary = TurnCompactionSummary(
                general_summary=f"[Compacted middle section: {full_token_count - head_token_count - tail_token_count} tokens omitted]",
            )

        compaction_result = TurnCompactionResult(
            head_text=head_text,
            head_tokens=head_token_count,
            tail_text=tail_text,
            tail_tokens=tail_token_count,
            middle_summary=summary,
            original_token_count=full_token_count,
            is_lossy=True,
        )

        # Build compacted turn: replace messages with head + summary + tail
        compacted_messages = self._build_compacted_messages(turn, head_text, tail_text, summary, selected)
        summary_text = self._format_summary(summary)

        compacted_turn = Turn(
            messages=compacted_messages,
            boundary="complete",
            token_count=head_token_count + tail_token_count + _estimate_tokens(summary_text),
        )

        return compacted_turn, compaction_result

    def split(self, turn: Turn) -> TurnCompactionParts:
        """Select head, middle, and tail source ranges for an oversized turn."""
        text, message_ranges = self._flatten_extractable_messages(turn)
        if not text:
            return TurnCompactionParts(head_text="", middle_text="", tail_text="")

        head_end = self._prefix_end_for_budget(text, self._config.compaction_head_tokens)
        first_user_end = next((end for message, _start, end in message_ranges if message.role == "user"), None)
        if first_user_end is not None:
            head_end = max(head_end, first_user_end)

        tail_start = self._suffix_start_for_budget(text, self._config.compaction_tail_tokens)
        tail_start = max(head_end, tail_start)

        return TurnCompactionParts(
            head_text=text[:head_end],
            middle_text=text[head_end:tail_start],
            tail_text=text[tail_start:],
            head_messages=tuple(self._slice_message_refs(message_ranges, 0, head_end)),
            tail_messages=tuple(self._slice_message_refs(message_ranges, tail_start, len(text))),
        )

    @staticmethod
    def _flatten_extractable_messages(turn: Turn) -> tuple[str, list[tuple[TurnMessageRef, int, int]]]:
        """Flatten extractable messages while retaining exact source ranges."""
        parts: list[str] = []
        ranges: list[tuple[TurnMessageRef, int, int]] = []
        position = 0
        for message in turn.extractable_messages:
            if parts:
                parts.append("\n")
                position += 1
            start = position
            parts.append(message.text)
            position += len(message.text)
            ranges.append((message, start, position))
        return "".join(parts), ranges

    @staticmethod
    def _slice_message_refs(
        message_ranges: list[tuple[TurnMessageRef, int, int]],
        range_start: int,
        range_end: int,
    ) -> list[TurnMessageRef]:
        """Return source-preserving message refs intersecting a flattened range."""
        refs: list[TurnMessageRef] = []
        for message, message_start, message_end in message_ranges:
            start = max(range_start, message_start)
            end = min(range_end, message_end)
            if start >= end:
                continue
            text = message.text[start - message_start : end - message_start]
            if text:
                refs.append(message.model_copy(update={"text": text}))
        return refs

    @staticmethod
    def _prefix_end_for_budget(text: str, token_budget: int) -> int:
        """Return the largest prefix end whose estimated tokens fit the budget."""
        if token_budget <= 0:
            return 0
        low = 0
        high = len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if _estimate_tokens(text[:mid]) <= token_budget:
                low = mid
            else:
                high = mid - 1
        return low

    @staticmethod
    def _suffix_start_for_budget(text: str, token_budget: int) -> int:
        """Return the smallest suffix start whose estimated tokens fit the budget."""
        if token_budget <= 0:
            return len(text)
        low = 0
        high = len(text)
        while low < high:
            mid = (low + high) // 2
            if _estimate_tokens(text[mid:]) <= token_budget:
                high = mid
            else:
                low = mid + 1
        return low

    def _build_compacted_messages(
        self,
        original: Turn,
        head_text: str,
        tail_text: str,
        summary: TurnCompactionSummary,
        parts: TurnCompactionParts,
    ) -> list[TurnMessageRef]:
        """Build message refs for the compacted turn."""
        messages: list[TurnMessageRef] = []

        # Head: raw evidence
        if parts.head_messages:
            messages.extend(parts.head_messages)
        elif head_text:
            messages.append(
                TurnMessageRef(
                    text=head_text,
                    role=original.messages[0].role if original.messages else "user",
                    timestamp=original.messages[0].timestamp if original.messages else None,
                    message_index=original.messages[0].message_index if original.messages else 0,
                    is_extractable=True,
                )
            )

        # Middle summary: non-extractable context
        summary_text = self._format_summary(summary)
        if summary_text:
            messages.append(
                TurnMessageRef(
                    text=f"[Compacted context summary]\n{summary_text}",
                    role="system",
                    timestamp=None,
                    message_index=-1,
                    is_extractable=False,
                )
            )

        # Tail: raw evidence
        if parts.tail_messages:
            messages.extend(parts.tail_messages)
        elif tail_text:
            last_msg = original.messages[-1] if original.messages else None
            messages.append(
                TurnMessageRef(
                    text=tail_text,
                    role=last_msg.role if last_msg else "assistant",
                    timestamp=last_msg.timestamp if last_msg else None,
                    message_index=last_msg.message_index if last_msg else 0,
                    is_extractable=True,
                )
            )

        return messages

    @staticmethod
    def _format_summary(summary: TurnCompactionSummary) -> str:
        data = summary.model_dump(exclude_defaults=True)
        return json.dumps(data, ensure_ascii=False) if data else ""


SummarizeFn = callable  # type: ignore[type-arg]
"""Callable[[str], TurnCompactionSummary] — takes middle text, returns structured summary."""
