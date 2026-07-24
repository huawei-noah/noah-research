"""History packing for vanilla add pipeline chunking.

Builds a sliding context window across chunks. External DB history is only
used for chunk 0. Later chunks derive context only from the previous chunk's
packed history and raw messages.
"""

from __future__ import annotations

from ....config import VanillaAddConfig
from ....typing import (
    Chunk,
    HistoryPack,
    Turn,
)


class HistoryPacker:
    """Build history context for each chunk in a chunk plan.

    For chunk 0: external history (DB recall) may be injected.
    For chunk i > 0: sliding in-request history from chunk i-1 only.
    History is packed in complete turns, backward from most recent,
    within token budgets.
    """

    def __init__(self, config: VanillaAddConfig | None = None) -> None:
        self._config = config or VanillaAddConfig()

    def pack_for_first_chunk(
        self,
        external_history: list[Turn] | None = None,
    ) -> HistoryPack:
        """Build history pack for chunk 0 (may include external history).

        Args:
            external_history: Optional DB recall / prior conversation turns.

        Returns:
            HistoryPack with external history packed under budget.
        """
        ext = external_history or []
        packed_ext = self._pack_turns_backward(ext, self._config.history_soft_token_budget)
        token_usage = sum(t.token_count for t in packed_ext)
        return HistoryPack(
            external_history=packed_ext,
            in_request_history=[],
            token_usage=token_usage,
        )

    def pack_for_chunk(
        self,
        chunk_index: int,
        prev_pack: HistoryPack,
        prev_chunk: Chunk,
    ) -> HistoryPack:
        """Build history pack for chunk i > 0.

        Derives context from prev_pack + prev_chunk messages only.
        External history is not included.

        Args:
            chunk_index: Zero-based chunk index (> 0).
            prev_pack: HistoryPack from the previous chunk.
            prev_chunk: The previous Chunk (for raw messages).

        Returns:
            HistoryPack with sliding in-request history.
        """
        if chunk_index == 0:
            return self.pack_for_first_chunk()

        # Available sources: prev pack turns + prev chunk turns
        available_turns = list(prev_pack.in_request_history) + list(prev_chunk.turns)

        packed = self._pack_turns_backward(available_turns, self._config.history_soft_token_budget)

        token_usage = sum(t.token_count for t in packed)

        return HistoryPack(
            external_history=[],
            in_request_history=packed,
            token_usage=token_usage,
        )

    def _pack_turns_backward(
        self,
        turns: list[Turn],
        soft_budget: int,
    ) -> list[Turn]:
        """Pack complete turns backward (most recent first) within budget.

        Starts from the last turn and moves backward while staying within
        the soft budget. Guarantees at least `history_min_turn_count` turns
        if available, even if they exceed soft budget (but not hard budget).
        """
        if not turns:
            return []

        hard_budget = self._config.history_hard_token_budget
        min_turns = self._config.history_min_turn_count

        packed: list[Turn] = []
        total_tokens = 0

        for turn in reversed(turns):
            turn_tokens = turn.token_count

            # If we haven't met minimum turn count yet, include regardless of soft budget
            if len(packed) < min_turns:
                if total_tokens + turn_tokens > hard_budget:
                    # Even hard budget exceeded — stop
                    break
                packed.append(turn)
                total_tokens += turn_tokens
                continue

            # After minimum met, check soft budget
            if total_tokens + turn_tokens > soft_budget:
                break

            if total_tokens + turn_tokens > hard_budget:
                break

            packed.append(turn)
            total_tokens += turn_tokens

        packed.reverse()
        return packed
