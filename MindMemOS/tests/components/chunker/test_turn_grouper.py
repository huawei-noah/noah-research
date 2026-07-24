"""Unit tests for TurnGrouper covering all scenarios from specs/turn-grouping/spec.md."""

from __future__ import annotations

from mindmemos.components.chunker.vanilla.turn_grouper import TurnGrouper, _estimate_tokens
from mindmemos.typing.memory import DialogueMessage, TextMessage

from mindmemos.config import VanillaAddConfig


def _dm(role: str, content: str, timestamp: int | None = None) -> DialogueMessage:
    return DialogueMessage(role=role, content=content, timestamp=timestamp)


def _tm(text: str) -> TextMessage:
    return TextMessage(text=text)


def _roles(turns: list) -> list[list[str]]:
    return [[m.role for m in t.messages] for t in turns]


def _boundaries(turns: list) -> list[str]:
    return [t.boundary for t in turns]


LOCOMO_CAROLINE_MELANIE_D1_11_TO_D1_18 = [
    (
        "Caroline",
        "I'm keen on counseling or working in mental health - I'd love to support those with similar issues.",
    ),
    (
        "Melanie",
        "You'd be a great counselor! Your empathy and understanding will really help the people you work with. "
        "By the way, take a look at this.",
    ),
    ("Caroline", "Thanks, Melanie! That's really sweet. Is this your own painting?"),
    ("Melanie", "Yeah, I painted that lake sunrise last year! It's special to me."),
    (
        "Caroline",
        "Wow, Melanie! The colors really blend nicely. Painting looks like a great outlet for expressing yourself.",
    ),
    (
        "Melanie",
        "Thanks, Caroline! Painting's a fun way to express my feelings and get creative. "
        "It's a great way to relax after a long day.",
    ),
    ("Caroline", "Totally agree, Mel. Relaxing and expressing ourselves is key. Well, I'm off to go do some research."),
    (
        "Melanie",
        "Yep, Caroline. Taking care of ourselves is vital. I'm off to go swimming with the kids. Talk to you soon!",
    ),
]


# 1. Normal alternating dialogue


class TestNormalAlternating:
    """Scenario: Normal alternating user/assistant dialogue."""

    def test_two_complete_turns(self) -> None:
        msgs = [_dm("user", "u1"), _dm("assistant", "a1"), _dm("user", "u2"), _dm("assistant", "a2")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 2
        assert _boundaries(turns) == ["complete", "complete"]
        assert _roles(turns) == [["user", "assistant"], ["user", "assistant"]]

    def test_three_turns(self) -> None:
        msgs = [
            _dm("user", "u1"),
            _dm("assistant", "a1"),
            _dm("user", "u2"),
            _dm("assistant", "a2"),
            _dm("user", "u3"),
            _dm("assistant", "a3"),
        ]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 3
        assert all(t.boundary == "complete" for t in turns)


# 2. Consecutive user messages before assistant response


class TestConsecutiveUser:
    """Scenario: Consecutive user messages before assistant — grouped into one turn."""

    def test_two_user_then_assistant(self) -> None:
        msgs = [_dm("user", "u1"), _dm("user", "u2"), _dm("assistant", "a1")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 1
        assert turns[0].boundary == "complete"
        assert _roles(turns) == [["user", "user", "assistant"]]


# 3. Request starts with assistant message (open head)


class TestAssistantFirst:
    """Scenario: First message is assistant, followed by user messages."""

    def test_assistant_then_user(self) -> None:
        msgs = [_dm("assistant", "a1"), _dm("user", "u1")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 2
        assert turns[0].boundary == "open_head"
        assert turns[1].boundary == "open_tail"
        assert _roles(turns) == [["assistant"], ["user"]]

    def test_assistant_then_full_turn(self) -> None:
        msgs = [_dm("assistant", "a1"), _dm("user", "u1"), _dm("assistant", "a2")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert turns[0].boundary == "open_head"
        assert turns[1].boundary == "complete"


# 4. Request ends with user message (open tail)


class TestOpenTail:
    """Scenario: Last message is user without assistant response."""

    def test_ends_with_user(self) -> None:
        msgs = [_dm("user", "u1"), _dm("assistant", "a1"), _dm("user", "u2")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 2
        assert turns[0].boundary == "complete"
        assert turns[1].boundary == "open_tail"


# 5. Only user messages


class TestOnlyUser:
    """Scenario: All messages are user role — one open-tail turn."""

    def test_only_user(self) -> None:
        msgs = [_dm("user", "u1"), _dm("user", "u2")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 1
        assert turns[0].boundary == "open_tail"


# 6. Only assistant messages (orphan)


class TestOnlyAssistant:
    """Scenario: All messages are assistant role — orphan turn."""

    def test_only_assistant(self) -> None:
        msgs = [_dm("assistant", "a1"), _dm("assistant", "a2")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 1
        assert turns[0].boundary == "orphan"


# 7. Time-gap splitting


class TestTimeGap:
    """Scenario: Large timestamp gap splits consecutive same-role messages."""

    def test_default_gap_is_thirty_minutes(self) -> None:
        config = VanillaAddConfig()
        assert config.time_gap_threshold_seconds == 1800

    def test_large_gap_splits_user_messages(self) -> None:
        config = VanillaAddConfig(time_gap_threshold_seconds=600)
        msgs = [
            _dm("user", "u1", timestamp=0),
            _dm("user", "u2", timestamp=3600_000),
        ]
        turns = TurnGrouper(config).group(list(enumerate(msgs)))
        assert len(turns) == 2

    def test_small_gap_keeps_together(self) -> None:
        config = VanillaAddConfig(time_gap_threshold_seconds=600)
        msgs = [
            _dm("user", "u1", timestamp=0),
            _dm("user", "u2", timestamp=5000),
        ]
        turns = TurnGrouper(config).group(list(enumerate(msgs)))
        assert len(turns) == 1

    def test_no_timestamp_no_split(self) -> None:
        msgs = [_dm("user", "u1"), _dm("user", "u2")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 1


# 8. System messages


class TestSystemMessages:
    """Scenario: System messages are non-extractable, never start a turn."""

    def test_system_between_user_assistant(self) -> None:
        msgs = [
            _dm("user", "u1"),
            _dm("system", "sys1"),
            _dm("assistant", "a1"),
        ]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 1
        system_msgs = [m for m in turns[0].messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].is_extractable is False

    def test_extractable_flags(self) -> None:
        msgs = [_dm("user", "u1"), _dm("system", "sys1"), _dm("assistant", "a1")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        for m in turns[0].messages:
            if m.role == "system":
                assert m.is_extractable is False
            else:
                assert m.is_extractable is True


# 9. Boundary metadata


class TestBoundaryMetadata:
    """Scenario: Boundary assignment rules."""

    def test_complete_turn(self) -> None:
        turns = TurnGrouper().group(list(enumerate([_dm("user", "u1"), _dm("assistant", "a1")])))
        assert turns[0].boundary == "complete"

    def test_open_head_boundary(self) -> None:
        turns = TurnGrouper().group(list(enumerate([_dm("assistant", "a1"), _dm("user", "u1")])))
        assert turns[0].boundary == "open_head"

    def test_open_tail_boundary(self) -> None:
        turns = TurnGrouper().group(list(enumerate([_dm("user", "u1")])))
        assert turns[0].boundary == "open_tail"

    def test_orphan_boundary(self) -> None:
        turns = TurnGrouper().group(list(enumerate([_dm("assistant", "a1")])))
        assert turns[0].boundary == "orphan"


# 10. Token estimation


class TestTokenEstimation:
    """Basic sanity checks for _estimate_tokens."""

    def test_empty_string(self) -> None:
        assert _estimate_tokens("") == 0

    def test_latin_text(self) -> None:
        tokens = _estimate_tokens("hello world foo bar")
        assert tokens > 0

    def test_cjk_text(self) -> None:
        tokens = _estimate_tokens("你好世界测试")
        assert tokens > 0


# 11. TextMessage handling


class TestTextMessage:
    """TextMessage has no role — defaults to user."""

    def test_text_message_grouped_as_user(self) -> None:
        msgs = [_tm("some text"), _dm("assistant", "response")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 1
        assert turns[0].messages[0].role == "user"
        assert turns[0].messages[0].text == "some text"

    def test_empty_messages_skipped(self) -> None:
        msgs = [_dm("user", ""), _dm("assistant", "  ")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        assert len(turns) == 0


# 12. Arbitrary speaker role handling


class TestArbitrarySpeakerRoles:
    """Arbitrary role names are speaker identities, not user/assistant roles."""

    def test_two_named_speakers_form_exchange_turns(self) -> None:
        msgs = [
            _dm("Rose", "I moved to Boston."),
            _dm("Alice", "That is exciting."),
            _dm("Rose", "I like the parks."),
            _dm("Alice", "Great."),
        ]

        turns = TurnGrouper().group(list(enumerate(msgs)))

        assert len(turns) == 2
        assert _boundaries(turns) == ["complete", "complete"]
        assert [[m.speaker for m in t.messages] for t in turns] == [["Rose", "Alice"], ["Rose", "Alice"]]
        assert [[m.role for m in t.messages] for t in turns] == [["speaker", "speaker"], ["speaker", "speaker"]]
        assert [[m.raw_role for m in t.messages] for t in turns] == [["Rose", "Alice"], ["Rose", "Alice"]]
        assert all(m.is_extractable for turn in turns for m in turn.messages)

    def test_locomo_named_speakers_form_exchange_turns(self) -> None:
        """Regression sample from locomo10.json conversation 0, session_1 D1:11-D1:18."""
        msgs = [_dm(speaker, text) for speaker, text in LOCOMO_CAROLINE_MELANIE_D1_11_TO_D1_18]

        turns = TurnGrouper().group(list(enumerate(msgs)))

        assert len(turns) == 4
        assert _boundaries(turns) == ["complete", "complete", "complete", "complete"]
        assert [[m.speaker for m in turn.messages] for turn in turns] == [
            ["Caroline", "Melanie"],
            ["Caroline", "Melanie"],
            ["Caroline", "Melanie"],
            ["Caroline", "Melanie"],
        ]
        assert [[m.role for m in turn.messages] for turn in turns] == [["speaker", "speaker"]] * 4
        assert [message.message_index for turn in turns for message in turn.messages] == list(range(8))
        assert "I'm keen on counseling" in turns[0].messages[0].text
        assert "I'm off to go swimming" in turns[-1].messages[-1].text

    def test_consecutive_same_speaker_messages_stay_in_one_run(self) -> None:
        msgs = [
            _dm("Rose", "I moved to Boston."),
            _dm("Rose", "It was last month."),
            _dm("Alice", "That is exciting."),
        ]

        turns = TurnGrouper().group(list(enumerate(msgs)))

        assert len(turns) == 1
        assert [m.message_index for m in turns[0].messages] == [0, 1, 2]
        assert [m.speaker for m in turns[0].messages] == ["Rose", "Rose", "Alice"]

    def test_time_gap_splits_named_speaker_exchange(self) -> None:
        config = VanillaAddConfig(time_gap_threshold_seconds=600)
        msgs = [
            _dm("Rose", "I moved to Boston.", timestamp=0),
            _dm("Alice", "That is exciting.", timestamp=5000),
            _dm("Bob", "Which area?", timestamp=3600_000),
        ]

        turns = TurnGrouper(config).group(list(enumerate(msgs)))

        assert len(turns) == 2
        assert [[m.speaker for m in t.messages] for t in turns] == [["Rose", "Alice"], ["Bob"]]
        assert _boundaries(turns) == ["complete", "open_tail"]


# 13. Original index preservation (mixed-message regression)


class TestOriginalIndexPreservation:
    """message_index must reflect the position in AddPipelineInput.messages,
    not the position in a filtered dialogue-only sublist.

    This is the regression test for the mixed-message index misalignment bug:
    when non-dialogue messages (UrlMessage, FileMessage) precede dialogue
    messages, the filtered list gets re-enumerated starting from 0, producing
    wrong message_index values in TurnMessageRef.
    """

    def test_index_preserved_after_filtering(self) -> None:
        """Simulate filtering: UrlMessage at index 0 is skipped,
        DialogueMessage at original index 1 retains message_index=1.
        """
        # The original sequence starts with a URL message followed by a dialogue message.
        #   [UrlMessage(url="..."), DialogueMessage(role="user", content="hello")]
        # Filtering leaves only the dialogue message, but it keeps the original index.
        indexed = [(1, _dm("user", "hello"))]
        turns = TurnGrouper().group(indexed)
        assert len(turns) == 1
        assert turns[0].messages[0].message_index == 1

    def test_index_preserved_with_gap_in_middle(self) -> None:
        """Multiple non-dialogue messages create gaps in indices."""
        # Original: [UrlMsg(0), UrlMsg(1), DM(2, "q"), DM(3, "a")]
        indexed = [(2, _dm("user", "q")), (3, _dm("assistant", "a"))]
        turns = TurnGrouper().group(indexed)
        assert len(turns) == 1
        indices = [m.message_index for m in turns[0].messages]
        assert indices == [2, 3]

    def test_sequential_indices_when_no_filtering(self) -> None:
        """When no filtering is applied (enumerate), indices are 0-based sequential."""
        msgs = [_dm("user", "u1"), _dm("assistant", "a1")]
        turns = TurnGrouper().group(list(enumerate(msgs)))
        indices = [m.message_index for m in turns[0].messages]
        assert indices == [0, 1]

    def test_mixed_indices_across_multiple_turns(self) -> None:
        """Indices carry correctly across turn boundaries."""
        # Original: [FileMsg(0), DM(1, "q1"), DM(2, "a1"), DM(3, "q2"), DM(4, "a2")]
        indexed = [
            (1, _dm("user", "q1")),
            (2, _dm("assistant", "a1")),
            (3, _dm("user", "q2")),
            (4, _dm("assistant", "a2")),
        ]
        turns = TurnGrouper().group(indexed)
        assert len(turns) == 2
        assert [m.message_index for m in turns[0].messages] == [1, 2]
        assert [m.message_index for m in turns[1].messages] == [3, 4]
