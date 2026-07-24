"""Unit tests for ChunkPlanner covering all scenarios from specs/chunk-planning/spec.md."""

from __future__ import annotations

from mindmemos.components.chunker.vanilla.chunk_planner import ChunkPlanner
from mindmemos.typing.algo import Turn, TurnMessageRef

from mindmemos.config import VanillaAddConfig


def _ref(text: str, role: str = "user") -> TurnMessageRef:
    return TurnMessageRef(text=text, role=role, message_index=0)


def _turn(token_count: int, boundary: str = "complete") -> Turn:
    """Create a turn with an explicit token_count (text is a placeholder)."""
    return Turn(
        messages=[_ref("x" * token_count)],
        boundary=boundary,
        token_count=token_count,
    )


def _large_turn(token_count: int, boundary: str = "complete") -> Turn:
    """Create a turn with realistic text matching token_count."""
    # Use whitespace-separated words to approximate token count
    words = " ".join(["word"] * token_count)
    return Turn(
        messages=[_ref(words)],
        boundary=boundary,
        token_count=token_count,
    )


def _small_budget_config(**overrides: int) -> VanillaAddConfig:
    values = {
        "chunk_soft_token_budget": 3000,
        "chunk_hard_token_budget": 4000,
        "template_tokens": 500,
        "history_soft_token_budget": 600,
        "history_hard_token_budget": 800,
        "recall_budget": 300,
        "output_headroom": 200,
    }
    values.update(overrides)
    return VanillaAddConfig(**values)


# 1. Multiple short turns fit in one chunk


class TestMultipleShortTurns:
    """Scenario: Several short turns pack into one chunk under soft budget."""

    def test_three_short_turns_one_chunk(self) -> None:
        config = _small_budget_config()
        planner = ChunkPlanner(config)
        turns = [_turn(200), _turn(200), _turn(200)]
        chunks = planner.plan(turns)
        assert len(chunks) == 1
        assert len(chunks[0].turns) == 3
        assert chunks[0].token_count == 600

    def test_all_turns_fit(self) -> None:
        config = _small_budget_config(chunk_soft_token_budget=5000, chunk_hard_token_budget=6000)
        planner = ChunkPlanner(config)
        turns = [_turn(100) for _ in range(10)]
        chunks = planner.plan(turns)
        assert len(chunks) == 1


# 2. Next turn exceeds soft budget → new chunk


class TestSoftBudgetSplit:
    """Scenario: Close chunk when next turn would exceed soft budget."""

    def test_split_on_soft_exceed(self) -> None:
        # soft extractable = 3000 - 500 - 600 - 300 - 200 = 1400
        # 5 turns of 300 each = 1500 > 1400, so after 4 turns (1200), 5th triggers split
        config = _small_budget_config()
        planner = ChunkPlanner(config)
        turns = [_turn(300) for _ in range(5)]
        chunks = planner.plan(turns)
        # First 4 turns (1200 < 1400), 5th would make 1500 > 1400 → new chunk
        assert len(chunks) == 2
        assert len(chunks[0].turns) == 4
        assert len(chunks[1].turns) == 1

    def test_single_turn_under_soft_gets_own_chunk(self) -> None:
        config = _small_budget_config()
        planner = ChunkPlanner(config)
        turns = [_turn(500)]
        chunks = planner.plan(turns)
        assert len(chunks) == 1

    def test_hard_budget_splits_when_soft_budget_is_larger(self) -> None:
        config = _small_budget_config(
            chunk_soft_token_budget=3000,
            chunk_hard_token_budget=2000,
            template_tokens=0,
            history_soft_token_budget=0,
            history_hard_token_budget=0,
            recall_budget=0,
            output_headroom=0,
        )

        chunks = ChunkPlanner(config).plan([_turn(900), _turn(900), _turn(900)])

        assert [chunk.token_count for chunk in chunks] == [1800, 900]


# 3. Single turn exceeds hard turn budget → compaction flag


class TestCompactionFlag:
    """Scenario: Turn exceeding turn_hard_token_budget is flagged for compaction."""

    def test_oversized_turn_flagged(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=100)
        planner = ChunkPlanner(config)
        turns = [_turn(200)]
        chunks = planner.plan(turns)
        assert len(chunks) == 1
        assert chunks[0].needs_compaction is True
        assert chunks[0].compacted_turn_indices == [0]

    def test_normal_turn_not_flagged(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=500)
        planner = ChunkPlanner(config)
        turns = [_turn(200)]
        chunks = planner.plan(turns)
        assert chunks[0].needs_compaction is False

    def test_turn_above_hard_extractable_budget_is_flagged(self) -> None:
        config = VanillaAddConfig(
            chunk_hard_token_budget=1000,
            turn_hard_token_budget=5000,
            template_tokens=100,
            history_hard_token_budget=100,
            recall_budget=100,
            output_headroom=200,
        )
        planner = ChunkPlanner(config)
        chunks = planner.plan([_turn(600)])
        assert planner.extractable_budget == 500
        assert chunks[0].needs_compaction is True


# 4. Single turn fits within hard budget


class TestSingleTurnInBudget:
    """Scenario: Single turn exceeds soft but fits hard → own chunk."""

    def test_turn_over_soft_in_hard(self) -> None:
        # soft_extractable=1400, hard_extractable=2200
        config = _small_budget_config(
            chunk_soft_token_budget=3000,
            chunk_hard_token_budget=4000,
        )
        planner = ChunkPlanner(config)
        turns = [_turn(1800)]  # > soft(1400) but < hard(2200), and < turn_hard(3000)
        chunks = planner.plan(turns)
        assert len(chunks) == 1
        assert chunks[0].needs_compaction is False
        assert len(chunks[0].turns) == 1


# 5. Chunk boundary metadata


class TestChunkBoundary:
    """Scenario: Chunk boundary derived from turn boundaries."""

    def test_open_tail_from_last_turn(self) -> None:
        config = _small_budget_config(chunk_soft_token_budget=5000)
        planner = ChunkPlanner(config)
        turns = [_turn(100, "complete"), _turn(100, "complete"), _turn(100, "open_tail")]
        chunks = planner.plan(turns)
        assert chunks[0].boundary == "open_tail"

    def test_all_complete(self) -> None:
        config = _small_budget_config(chunk_soft_token_budget=5000)
        planner = ChunkPlanner(config)
        turns = [_turn(100, "complete"), _turn(100, "complete")]
        chunks = planner.plan(turns)
        assert chunks[0].boundary == "complete"

    def test_open_head_from_first_turn(self) -> None:
        config = _small_budget_config(chunk_soft_token_budget=5000)
        planner = ChunkPlanner(config)
        turns = [_turn(100, "open_head"), _turn(100, "complete")]
        chunks = planner.plan(turns)
        assert chunks[0].boundary == "open_head"

    def test_orphan_single_turn(self) -> None:
        turns = [_turn(100, "orphan")]
        chunks = ChunkPlanner().plan(turns)
        assert chunks[0].boundary == "orphan"


# 6. Budget allocation


class TestBudgetAllocation:
    """Scenario: Extractable budget includes all prompt context allocations."""

    def test_coherent_default_budgets(self) -> None:
        config = VanillaAddConfig()
        planner = ChunkPlanner(config)

        assert config.chunk_soft_token_budget == 26000
        assert config.chunk_hard_token_budget == 32000
        assert config.turn_hard_token_budget == 16000
        assert config.history_soft_token_budget == 2000
        assert config.history_hard_token_budget == 4000
        assert config.compaction_soft_token_budget == 16000
        assert config.compaction_head_tokens == 4000
        assert config.compaction_tail_tokens == 4000
        assert config.compaction_summary_context_token_budget == 200000
        assert config.compaction_summary_output_token_budget == 8000
        assert config.time_gap_threshold_seconds == 1800
        assert config.template_tokens == 1000
        assert config.recall_budget == 2000
        assert config.output_headroom == 4000
        assert planner.soft_extractable_budget == 17000
        assert planner.extractable_budget == 21000

    def test_extractable_budget_calculation(self) -> None:
        config = VanillaAddConfig(
            chunk_hard_token_budget=4000,
            template_tokens=500,
            history_hard_token_budget=800,
            recall_budget=300,
            output_headroom=200,
        )
        planner = ChunkPlanner(config)
        assert planner.extractable_budget == 4000 - 500 - 800 - 300 - 200

    def test_soft_extractable_budget(self) -> None:
        config = VanillaAddConfig(
            chunk_soft_token_budget=3000,
            template_tokens=500,
            history_soft_token_budget=600,
            recall_budget=300,
            output_headroom=200,
        )
        planner = ChunkPlanner(config)
        assert planner.soft_extractable_budget == 3000 - 500 - 600 - 300 - 200


# 7. Deterministic planning


class TestDeterministic:
    """Scenario: Same input + config → same output."""

    def test_reproducible(self) -> None:
        config = _small_budget_config()
        turns = [_turn(200 + i * 50) for i in range(8)]

        result1 = ChunkPlanner(config).plan(turns)
        result2 = ChunkPlanner(config).plan(turns)

        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2):
            assert c1.chunk_index == c2.chunk_index
            assert c1.token_count == c2.token_count
            assert c1.boundary == c2.boundary
            assert c1.needs_compaction == c2.needs_compaction
            assert len(c1.turns) == len(c2.turns)


# 8. Edge cases


class TestEdgeCases:
    """Edge cases for chunk planner."""

    def test_empty_turns(self) -> None:
        chunks = ChunkPlanner().plan([])
        assert chunks == []

    def test_mixed_compaction_and_normal(self) -> None:
        """Oversized turn between normal turns creates correct chunk boundaries."""
        config = _small_budget_config(
            chunk_soft_token_budget=5000,
            chunk_hard_token_budget=6000,
            turn_hard_token_budget=300,
        )
        planner = ChunkPlanner(config)
        turns = [
            _turn(100, "complete"),
            _turn(500, "complete"),  # Exceeds turn_hard_token_budget=300 → compaction
            _turn(100, "complete"),
        ]
        chunks = planner.plan(turns)
        # Chunk 0: turn 0 (100)
        # Chunk 1: turn 1 (500) — compacted
        # Chunk 2: turn 2 (100)
        assert len(chunks) == 3
        assert chunks[1].needs_compaction is True
        assert not chunks[0].needs_compaction
        assert not chunks[2].needs_compaction
