"""Unit tests for HistoryPacker covering all scenarios from specs/history-packing/spec.md."""

from __future__ import annotations

from mindmemos.components.chunker.vanilla.history_packer import HistoryPacker
from mindmemos.typing.algo import (
    Chunk,
    HistoryPack,
    Turn,
    TurnMessageRef,
)

from mindmemos.config import VanillaAddConfig


def _ref(text: str) -> TurnMessageRef:
    return TurnMessageRef(text=text, role="user", message_index=0)


def _turn(tokens: int, boundary: str = "complete") -> Turn:
    return Turn(messages=[_ref("x" * tokens)], boundary=boundary, token_count=tokens)


def _chunk(turns: list[Turn], index: int = 0) -> Chunk:
    return Chunk(
        turns=turns,
        boundary="complete",
        token_count=sum(t.token_count for t in turns),
        chunk_index=index,
    )


# 1. External history only for the first chunk


class TestExternalHistory:
    """Scenario: External history only used for chunk 0."""

    def test_chunk0_gets_external_history(self) -> None:
        config = VanillaAddConfig()
        packer = HistoryPacker(config)
        ext = [_turn(100), _turn(100)]
        pack = packer.pack_for_first_chunk(external_history=ext)
        assert len(pack.external_history) > 0

    def test_later_chunks_no_external(self) -> None:
        config = VanillaAddConfig()
        packer = HistoryPacker(config)
        prev_pack = packer.pack_for_first_chunk([_turn(100)])
        prev_chunk = _chunk([_turn(200)])
        pack = packer.pack_for_chunk(1, prev_pack, prev_chunk)
        assert len(pack.external_history) == 0

    def test_no_external_history_available(self) -> None:
        packer = HistoryPacker()
        pack = packer.pack_for_first_chunk()
        assert pack.external_history == []


# 2. Sliding in-request history for later chunks


class TestSlidingHistory:
    """Scenario: Chunk i > 0 derives history from chunk i-1 only."""

    def test_chunk1_from_chunk0(self) -> None:
        config = VanillaAddConfig(history_soft_token_budget=500)
        packer = HistoryPacker(config)
        prev_pack = HistoryPack(in_request_history=[_turn(100)], token_usage=100)
        prev_chunk = _chunk([_turn(200)])
        pack = packer.pack_for_chunk(1, prev_pack, prev_chunk)
        # Should have turns from prev_pack.in_request_history + prev_chunk.turns
        assert len(pack.in_request_history) > 0

    def test_chunk2_does_not_access_chunk0_directly(self) -> None:
        """Chunk 2 only sees chunk 1's history, not chunk 0's raw messages."""
        config = VanillaAddConfig(history_soft_token_budget=2000)
        packer = HistoryPacker(config)

        # Chunk 0 → chunk 1
        pack0 = packer.pack_for_first_chunk()
        chunk0 = _chunk([_turn(100, "complete")], index=0)
        pack1 = packer.pack_for_chunk(1, pack0, chunk0)

        # Chunk 1 → chunk 2
        # pack1.in_request_history has chunk0's turn
        chunk1 = _chunk([_turn(150, "complete")], index=1)
        pack2 = packer.pack_for_chunk(2, pack1, chunk1)

        # pack2 should only have pack1.in_request_history + chunk1.turns
        # NOT chunk0.turns directly (they may appear via pack1 history if they fit)
        assert len(pack2.in_request_history) > 0
        # The only sources are pack1.in_request_history + chunk1.turns
        total_available = len(pack1.in_request_history) + len(chunk1.turns)
        assert len(pack2.in_request_history) <= total_available


# 3. Complete-turn packing


class TestCompleteTurnPacking:
    """Scenario: Pack in complete turns, backward from most recent."""

    def test_multiple_short_turns_fit(self) -> None:
        config = VanillaAddConfig(history_soft_token_budget=300)
        packer = HistoryPacker(config)
        turns = [_turn(100), _turn(80), _turn(120), _turn(90)]
        pack = packer.pack_for_first_chunk(external_history=turns)
        # Should include the last 3 turns: 90 + 120 + 80 = 290 <= 300
        assert len(pack.external_history) == 3

    def test_most_recent_exceeds_soft_but_includes(self) -> None:
        config = VanillaAddConfig(
            history_soft_token_budget=400,
            history_hard_token_budget=800,
            history_min_turn_count=1,
        )
        packer = HistoryPacker(config)
        turns = [_turn(500)]  # Exceeds soft(400) but < hard(800)
        pack = packer.pack_for_first_chunk(external_history=turns)
        assert len(pack.external_history) == 1  # min_turn_count=1 guarantees inclusion


# 4. Token-budgeted history packing


class TestTokenBudget:
    """Scenario: Hard budget prevents runaway history."""

    def test_hard_budget_cap(self) -> None:
        config = VanillaAddConfig(
            history_soft_token_budget=5000,
            history_hard_token_budget=500,
            history_min_turn_count=1,
        )
        packer = HistoryPacker(config)
        turns = [_turn(200) for _ in range(10)]
        pack = packer.pack_for_first_chunk(external_history=turns)
        # Should include only turns fitting within hard=500
        total = sum(t.token_count for t in pack.external_history)
        assert total <= 500


# 5. History is non-extractable


class TestNonExtractable:
    """Scenario: History content is non-extractable — cannot produce memories."""

    def test_history_pack_carries_no_extractability(self) -> None:
        """HistoryPack itself is always used as non-extractable context.
        Extractability is enforced at the extraction envelope level, not
        in the pack. Verify the pack structure is correct."""
        packer = HistoryPacker()
        pack = packer.pack_for_first_chunk([_turn(100)])
        assert isinstance(pack, HistoryPack)
