"""Chunk planning for vanilla add pipeline chunking.

Packs complete turns into token-budgeted chunks. The planner does not split
turns — it prefers turn integrity over maximizing chunk fullness.
"""

from __future__ import annotations

from ....config import VanillaAddConfig
from ....typing import Chunk, ChunkBoundary, Turn


class ChunkPlanner:
    """Pack turns into token-budgeted chunks for LLM extraction.

    The planner allocates the hard budget across template, history, recall,
    output headroom, and extractable tokens. It greedily packs complete turns
    into chunks, closing a chunk when the next turn would exceed the soft
    extractable budget. Single turns exceeding the hard budget are flagged
    for compaction.
    """

    def __init__(self, config: VanillaAddConfig | None = None) -> None:
        self._config = config or VanillaAddConfig()

    @property
    def extractable_budget(self) -> int:
        """Token budget available for chunk messages (hard minus overhead)."""
        c = self._config
        return (
            c.chunk_hard_token_budget
            - c.template_tokens
            - c.history_hard_token_budget
            - c.recall_budget
            - c.output_headroom
        )

    @property
    def soft_extractable_budget(self) -> int:
        """Soft target for chunk messages (soft minus overhead)."""
        c = self._config
        return (
            c.chunk_soft_token_budget
            - c.template_tokens
            - c.history_soft_token_budget
            - c.recall_budget
            - c.output_headroom
        )

    def plan(self, turns: list[Turn]) -> list[Chunk]:
        """Pack turns into chunks according to token budgets.

        Args:
            turns: Ordered turns from TurnGrouper.

        Returns:
            Ordered list of Chunk objects. Each chunk carries its turns,
            boundary metadata, token count, and compaction flags.
        """
        if not turns:
            return []

        soft = self.soft_extractable_budget
        hard = self.extractable_budget
        turn_budget = self._config.turn_hard_token_budget

        chunks: list[Chunk] = []
        current_turns: list[Turn] = []
        current_tokens = 0
        chunk_index = 0

        for turn in turns:
            turn_tokens = turn.token_count

            # Check if this single turn exceeds hard turn budget → flag for compaction
            if turn_tokens > turn_budget or turn_tokens > hard:
                if current_turns:
                    chunks.append(self._build_chunk(current_turns, current_tokens, chunk_index))
                    chunk_index += 1
                    current_turns = []
                    current_tokens = 0

                # This turn gets its own chunk, flagged for compaction
                chunk = self._build_chunk([turn], turn_tokens, chunk_index)
                chunk.needs_compaction = True
                chunk.compacted_turn_indices = [0]
                chunks.append(chunk)
                chunk_index += 1
                continue

            # Default packing: check if adding this turn exceeds soft budget
            if current_turns and (current_tokens + turn_tokens > soft or current_tokens + turn_tokens > hard):
                # Close current chunk, start new one
                chunks.append(self._build_chunk(current_turns, current_tokens, chunk_index))
                chunk_index += 1
                current_turns = []
                current_tokens = 0

            current_turns.append(turn)
            current_tokens += turn_tokens

        if current_turns:
            chunks.append(self._build_chunk(current_turns, current_tokens, chunk_index))

        return chunks

    def _build_chunk(self, turns: list[Turn], token_count: int, chunk_index: int) -> Chunk:
        """Build a Chunk with derived boundary metadata."""
        boundary = self._derive_boundary(turns)
        return Chunk(
            turns=turns,
            boundary=boundary,
            token_count=token_count,
            chunk_index=chunk_index,
        )

    def _derive_boundary(self, turns: list[Turn]) -> ChunkBoundary:
        """Derive chunk boundary from constituent turn boundaries.

        Rules (order of precedence):
        1. If any turn has OPEN_HEAD → chunk is open_head
        2. If last turn has OPEN_TAIL → chunk is open_tail
        3. If any turn has ORPHAN → chunk inherits that boundary
        4. All turns complete → chunk is complete
        """
        if not turns:
            return "complete"

        boundaries = [t.boundary for t in turns]

        if "open_head" in boundaries:
            return "open_head"
        if boundaries[-1] == "open_tail":
            return "open_tail"
        if "orphan" in boundaries:
            # Multi-turn chunks with orphan context must be reopened from the head.
            return "open_head" if len(turns) > 1 else "orphan"
        return "complete"
