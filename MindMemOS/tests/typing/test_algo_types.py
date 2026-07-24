"""Tests for TraceClassification and ExtractionBackground DTOs."""

import pytest
from mindmemos.typing.algo import ExtractionBackground, TraceClassification


class TestTraceClassification:
    def test_defaults(self):
        tc = TraceClassification()
        assert tc.kind == "dialogue"
        assert tc.completeness == "partial"
        assert tc.confidence == 0.0
        assert tc.reason == "unknown"

    def test_explicit_values(self):
        tc = TraceClassification(
            kind="agent_trace",
            completeness="complete",
            confidence=1.0,
            reason="explicit_metadata",
        )
        assert tc.kind == "agent_trace"
        assert tc.completeness == "complete"
        assert tc.confidence == 1.0

    def test_confidence_clamped_below_zero(self):
        with pytest.raises(Exception):
            TraceClassification(confidence=-0.1)

    def test_confidence_clamped_above_one(self):
        with pytest.raises(Exception):
            TraceClassification(confidence=1.1)

    def test_all_valid_kinds(self):
        for kind in ("dialogue", "agent_trace", "skill_trace", "file_or_url", "mixed"):
            tc = TraceClassification(kind=kind)
            assert tc.kind == kind

    def test_all_valid_completeness(self):
        for completeness in ("complete", "partial", "unknown"):
            tc = TraceClassification(completeness=completeness)
            assert tc.completeness == completeness


class TestExtractionBackground:
    def test_defaults(self):
        bg = ExtractionBackground()
        assert bg.trace_kind == "dialogue"
        assert bg.trace_completeness == "partial"
        assert bg.classification_confidence == 0.0
        assert bg.recent_dialogue == []
        assert bg.full_agent_trace is None
        assert bg.skill_trace is None
        assert bg.related_memory_seed == []
        assert bg.user_intent_summary == ""
        assert bg.user_query == ""
        assert bg.session_summary == ""
        assert bg.warnings == []

    def test_with_dialogue(self):
        bg = ExtractionBackground(
            trace_kind="dialogue",
            recent_dialogue=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            user_query="hello",
        )
        assert len(bg.recent_dialogue) == 2
        assert bg.user_query == "hello"

    def test_with_agent_trace(self):
        bg = ExtractionBackground(
            trace_kind="agent_trace",
            trace_completeness="complete",
            full_agent_trace=[
                {"role": "user", "content": "find files"},
                {"role": "assistant", "content": "using grep"},
                {"role": "tool", "content": "result.txt"},
            ],
            user_query="find files",
            user_intent_summary="find files + grep",
        )
        assert bg.full_agent_trace is not None
        assert len(bg.full_agent_trace) == 3
        assert bg.user_intent_summary == "find files + grep"

    def test_with_warnings(self):
        bg = ExtractionBackground(
            warnings=["history_unavailable", "ambiguous_trace"],
        )
        assert len(bg.warnings) == 2

    def test_confidence_clamped(self):
        with pytest.raises(Exception):
            ExtractionBackground(classification_confidence=2.0)
