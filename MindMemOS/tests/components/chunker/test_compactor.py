"""Unit tests for LongTurnCompactor covering scenarios from specs/long-turn-compaction/spec.md."""

from __future__ import annotations

from mindmemos.components.chunker.vanilla.compactor import LongTurnCompactor
from mindmemos.components.chunker.vanilla.turn_grouper import _estimate_tokens
from mindmemos.typing.algo import Turn, TurnCompactionSummary, TurnMessageRef

from mindmemos.config import VanillaAddConfig


def _ref(text: str, role: str = "user") -> TurnMessageRef:
    return TurnMessageRef(text=text, role=role, message_index=0)


def _long_turn(token_count: int, role: str = "assistant") -> Turn:
    """Create a turn with word-count text approximating token_count."""
    words = " ".join([f"word{i}" for i in range(token_count)])
    return Turn(messages=[_ref(words, role)], boundary="complete", token_count=token_count)


def _mock_summarize(middle_text: str) -> TurnCompactionSummary:
    """Mock summary function for testing."""
    return TurnCompactionSummary(
        general_summary=f"Summary of {len(middle_text)} chars",
        key_entities=["entity1"],
        user_intent="test intent",
        confirmed_facts=["fact1"],
        decisions=["decision1"],
        open_questions=[],
        warnings=[],
    )


def _turn_from_messages(*messages: tuple[str, str]) -> Turn:
    refs = [TurnMessageRef(text=text, role=role, message_index=index) for index, (role, text) in enumerate(messages)]
    return Turn(
        messages=refs,
        boundary="complete",
        token_count=sum(_estimate_tokens(ref.text) for ref in refs),
    )


# 1. Compact turns exceeding hard turn budget


class TestCompactionTrigger:
    """Scenario: Turn exceeding turn_hard_token_budget is compacted."""

    def test_long_turn_compacted(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=50, compaction_head_tokens=10, compaction_tail_tokens=10)
        compactor = LongTurnCompactor(config)
        turn = _long_turn(100)
        assert compactor.needs_compaction(turn)
        compacted, result = compactor.compact(turn, summarize_fn=_mock_summarize)
        assert result.head_text != ""
        assert result.tail_text != ""

    def test_short_turn_not_compacted(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=500)
        compactor = LongTurnCompactor(config)
        turn = _long_turn(100)
        assert not compactor.needs_compaction(turn)


# 2. Compacted turn is structurally complete


class TestCompactedStructure:
    """Scenario: Head and tail are extractable, middle is non-extractable."""

    def test_head_tail_extractable(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=50, compaction_head_tokens=10, compaction_tail_tokens=10)
        compactor = LongTurnCompactor(config)
        turn = _long_turn(100)
        compacted, result = compactor.compact(turn, summarize_fn=_mock_summarize)

        head_msgs = [m for m in compacted.messages if m.is_extractable]
        assert len(head_msgs) == 2  # head + tail

    def test_middle_summary_non_extractable(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=50, compaction_head_tokens=10, compaction_tail_tokens=10)
        compactor = LongTurnCompactor(config)
        turn = _long_turn(100)
        compacted, _ = compactor.compact(turn, summarize_fn=_mock_summarize)

        summary_msgs = [m for m in compacted.messages if m.role == "system"]
        assert len(summary_msgs) == 1
        assert summary_msgs[0].is_extractable is False

    def test_middle_context_contains_all_structured_summary_fields(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=50, compaction_head_tokens=10, compaction_tail_tokens=10)
        compacted, _ = LongTurnCompactor(config).compact(_long_turn(100), summarize_fn=_mock_summarize)

        summary_text = next(message.text for message in compacted.messages if not message.is_extractable)
        assert "entity1" in summary_text
        assert "test intent" in summary_text
        assert "fact1" in summary_text
        assert "decision1" in summary_text


# 3. Dedicated summary prompt


class TestSummaryPrompt:
    """Scenario: Summary is produced by a dedicated call, not extraction prompt."""

    def test_summary_fn_called(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=50, compaction_head_tokens=10, compaction_tail_tokens=10)
        compactor = LongTurnCompactor(config)
        turn = _long_turn(100)

        called = {}

        def track_summary(text: str) -> TurnCompactionSummary:
            called["text"] = text
            return _mock_summarize(text)

        compactor.compact(turn, summarize_fn=track_summary)
        assert "text" in called
        assert len(called["text"]) > 0

    def test_no_summary_fn_uses_stub(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=50, compaction_head_tokens=10, compaction_tail_tokens=10)
        compactor = LongTurnCompactor(config)
        turn = _long_turn(100)
        _, result = compactor.compact(turn)
        assert "Compacted" in result.middle_summary.general_summary


# 4. Structured summary output


class TestStructuredSummary:
    """Scenario: Summary has structured sections."""

    def test_structured_fields(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=50, compaction_head_tokens=10, compaction_tail_tokens=10)
        compactor = LongTurnCompactor(config)
        turn = _long_turn(100)
        _, result = compactor.compact(turn, summarize_fn=_mock_summarize)
        s = result.middle_summary
        assert s.general_summary != ""
        assert s.key_entities == ["entity1"]
        assert s.user_intent == "test intent"
        assert s.confirmed_facts == ["fact1"]
        assert s.decisions == ["decision1"]


# 5. Compaction is lossy and marked


class TestLossyMetadata:
    """Scenario: Compaction result carries lossy metadata."""

    def test_lossy_flag(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=50, compaction_head_tokens=10, compaction_tail_tokens=10)
        compactor = LongTurnCompactor(config)
        turn = _long_turn(100)
        _, result = compactor.compact(turn, summarize_fn=_mock_summarize)
        assert result.is_lossy is True
        assert result.original_token_count == 100

    def test_compacted_turn_boundary_complete(self) -> None:
        config = VanillaAddConfig(turn_hard_token_budget=50, compaction_head_tokens=10, compaction_tail_tokens=10)
        compactor = LongTurnCompactor(config)
        turn = _long_turn(100)
        compacted, _ = compactor.compact(turn, summarize_fn=_mock_summarize)
        assert compacted.boundary == "complete"


# 6. Edge cases


class TestEdgeCases:
    """Edge cases for compactor."""

    def test_text_shorter_than_head_plus_tail(self) -> None:
        """Text too short to split — keep all as head."""
        config = VanillaAddConfig(
            turn_hard_token_budget=5,
            compaction_head_tokens=100,
            compaction_tail_tokens=100,
        )
        compactor = LongTurnCompactor(config)
        turn = Turn(
            messages=[_ref("short text")],
            boundary="complete",
            token_count=2,
        )
        compacted, result = compactor.compact(turn)
        # All text stays as head, no tail or middle
        assert result.head_text == "short text"
        assert result.tail_text == ""


class TestFirstUserPreservation:
    """The first user message controls the minimum preserved head range."""

    def test_first_user_inside_default_head_keeps_default_head(self) -> None:
        config = VanillaAddConfig(compaction_head_tokens=5, compaction_tail_tokens=2)
        turn = _turn_from_messages(
            ("user", "u0 u1"),
            ("assistant", "a0 a1 a2 a3 a4 a5 a6 a7"),
        )

        parts = LongTurnCompactor(config).split(turn)

        assert _estimate_tokens(parts.head_text) == 5
        assert parts.head_text.startswith("u0 u1")

    def test_first_user_crossing_default_head_extends_through_user_end(self) -> None:
        config = VanillaAddConfig(compaction_head_tokens=4, compaction_tail_tokens=2)
        user_text = "u0 u1 u2 u3 u4 u5"
        turn = _turn_from_messages(
            ("user", user_text),
            ("assistant", "a0 a1 a2 a3 a4 a5"),
        )

        parts = LongTurnCompactor(config).split(turn)

        assert parts.head_text == user_text

    def test_first_user_after_default_head_extends_through_user_end(self) -> None:
        config = VanillaAddConfig(compaction_head_tokens=2, compaction_tail_tokens=2)
        user_text = "u0 u1 u2"
        turn = _turn_from_messages(
            ("assistant", "a0 a1 a2"),
            ("user", user_text),
            ("assistant", "tail0 tail1 tail2 tail3"),
        )

        parts = LongTurnCompactor(config).split(turn)

        assert parts.head_text.endswith(user_text)
        assert parts.head_text.startswith("a0 a1 a2")

    def test_compacted_head_preserves_first_user_role_and_message_index(self) -> None:
        config = VanillaAddConfig(compaction_head_tokens=2, compaction_tail_tokens=2)
        turn = _turn_from_messages(
            ("assistant", "a0 a1 a2"),
            ("user", "u0 u1 u2"),
            ("assistant", "tail0 tail1 tail2 tail3"),
        )

        compacted, _ = LongTurnCompactor(config).compact(turn, summary=TurnCompactionSummary(general_summary="summary"))

        user_messages = [message for message in compacted.messages if message.role == "user"]
        assert [(message.text, message.message_index) for message in user_messages] == [("u0 u1 u2", 1)]

    def test_no_user_uses_default_head(self) -> None:
        config = VanillaAddConfig(compaction_head_tokens=3, compaction_tail_tokens=2)
        turn = _turn_from_messages(("assistant", "a0 a1 a2 a3 a4 a5 a6"))

        parts = LongTurnCompactor(config).split(turn)

        assert _estimate_tokens(parts.head_text) == 3


class TestPositionBasedSplitting:
    """Range selection works without word-value or whitespace assumptions."""

    def test_tail_overlap_is_not_duplicated(self) -> None:
        config = VanillaAddConfig(compaction_head_tokens=6, compaction_tail_tokens=6)
        turn = _turn_from_messages(("assistant", "a0 a1 a2 a3 a4 a5 a6 a7"))

        parts = LongTurnCompactor(config).split(turn)

        assert parts.middle_text == ""
        assert parts.tail_text == "a6 a7"
        assert parts.head_text == "a0 a1 a2 a3 a4 a5 "
        assert parts.head_text + parts.tail_text == "a0 a1 a2 a3 a4 a5 a6 a7"

    def test_no_whitespace_chinese_text_preserves_head_and_tail(self) -> None:
        config = VanillaAddConfig(compaction_head_tokens=10, compaction_tail_tokens=10)
        turn = _turn_from_messages(("assistant", "甲" * 90))

        parts = LongTurnCompactor(config).split(turn)

        assert parts.head_text
        assert parts.middle_text
        assert parts.tail_text
        assert parts.head_text + parts.middle_text + parts.tail_text == "甲" * 90
