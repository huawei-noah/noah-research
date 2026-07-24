"""Unit tests for ExtractionEnvelope integration covering scenarios from specs/extraction-envelope/spec.md."""

from __future__ import annotations

import json

import pytest
from mindmemos.components.extractor.vanilla.memory import (
    _boundary_confidence,
    _boundary_guidance,
    _envelope_prompt_messages,
)
from mindmemos.typing.algo import (
    ExtractionEnvelope,
    HistoryPack,
    TurnMessageRef,
)
from mindmemos.typing.memory import MemoryRequestContext, PreprocessedText

from mindmemos.components.extractor.vanilla import (
    VanillaMemoryExtractor,
)


def _ctx() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="test-req",
        account_id="acc1",
        project_id="proj1",
        api_key_uuid="key1",
        user_id="u1",
        session_id="s1",
    )


def _ref(text: str, role: str = "user", extractable: bool = True) -> TurnMessageRef:
    return TurnMessageRef(text=text, role=role, message_index=0, is_extractable=extractable)


def _preprocessed(text: str, lang: str = "en") -> PreprocessedText:
    return PreprocessedText(
        text=text,
        normalized_text=text,
        lang=lang,
        content_hash="hash_" + text[:8],
        bm25_text=text,
    )


def _envelope(
    messages: list[TurnMessageRef] | None = None,
    boundary: str = "complete",
    chunk_index: int = 0,
    history: HistoryPack | None = None,
    recalled: list[dict] | None = None,
) -> ExtractionEnvelope:
    return ExtractionEnvelope(
        extractable_messages=messages or [_ref("user says hello", "user")],
        history=history or HistoryPack(),
        recalled_memories=recalled or [],
        boundary=boundary,
        chunk_index=chunk_index,
    )


# 1. Structured extraction envelope


class TestEnvelopeStructure:
    """Scenario: Envelope has extractable, context, and boundary sections."""

    def test_complete_chunk_prompt(self) -> None:
        envelope = _envelope(
            messages=[
                _ref("Hello", "user"),
                _ref("Hi there", "assistant"),
            ],
            boundary="complete",
        )
        preprocessed = [_preprocessed("Hello"), _preprocessed("Hi there")]
        msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        assert len(msgs) == 2
        payload = json.loads(msgs[1]["content"])
        assert "extractable" in payload
        assert "context" in payload
        assert payload["boundary"] == "complete"
        assert len(payload["extractable"]) == 2

    def test_open_tail_chunk(self) -> None:
        envelope = _envelope(boundary="open_tail")
        preprocessed = [_preprocessed("test")]
        msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        payload = json.loads(msgs[1]["content"])
        assert payload["boundary"] == "open_tail"
        assert (
            "unfinished" in payload.get("boundary_guidance", "").lower()
            or "conservative" in payload.get("boundary_guidance", "").lower()
        )


# 2. Extractable section is primary evidence


class TestExtractableEvidence:
    """Scenario: Only extractable messages produce candidates."""

    def test_extractable_messages_in_prompt(self) -> None:
        msgs_refs = [
            _ref("user question", "user"),
            _ref("assistant answer", "assistant"),
        ]
        envelope = _envelope(messages=msgs_refs)
        preprocessed = [_preprocessed("user question"), _preprocessed("assistant answer")]
        prompt_msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        payload = json.loads(prompt_msgs[1]["content"])
        assert len(payload["extractable"]) == 2
        assert all(e["is_extractable"] for e in payload["extractable"])

    def test_compacted_slices_have_unique_evidence_index(self) -> None:
        msgs_refs = [
            TurnMessageRef(text="head text", role="assistant", message_index=0, is_extractable=True),
            TurnMessageRef(text="tail text", role="assistant", message_index=0, is_extractable=True),
        ]
        envelope = _envelope(messages=msgs_refs, boundary="compacted")
        preprocessed = [_preprocessed("head text"), _preprocessed("tail text")]

        prompt_msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        payload = json.loads(prompt_msgs[1]["content"])

        assert [entry["index"] for entry in payload["extractable"]] == [0, 0]
        assert all("original_message_index" not in entry for entry in payload["extractable"])
        assert [entry["evidence_index"] for entry in payload["extractable"]] == [0, 1]


# 3. Context section is non-extractable


class TestContextNonExtractable:
    """Scenario: Context section carries the non-extraction instruction."""

    def test_instruction_in_payload(self) -> None:
        envelope = _envelope()
        preprocessed = [_preprocessed("test")]
        msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        payload = json.loads(msgs[1]["content"])
        assert "instruction" in payload
        assert "ONLY" in payload["instruction"] or "only" in payload["instruction"].lower()
        assert "context" in payload["instruction"].lower()

    def test_history_in_context_section(self) -> None:
        from mindmemos.typing.algo import Turn

        history = HistoryPack(
            in_request_history=[
                Turn(
                    messages=[_ref("past user msg", "user")],
                    boundary="complete",
                    token_count=10,
                )
            ],
            token_usage=10,
        )
        envelope = _envelope(history=history)
        preprocessed = [_preprocessed("test")]
        msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        payload = json.loads(msgs[1]["content"])
        assert "history" in payload["context"]

    def test_history_context_preserves_named_speaker_messages(self) -> None:
        from mindmemos.typing.algo import Turn

        history = HistoryPack(
            in_request_history=[
                Turn(
                    messages=[
                        TurnMessageRef(
                            text="I moved to Boston.",
                            role="speaker",
                            raw_role="Rose",
                            speaker="Rose",
                            message_index=3,
                            is_extractable=True,
                        ),
                        TurnMessageRef(
                            text="That is exciting.",
                            role="speaker",
                            raw_role="Alice",
                            speaker="Alice",
                            message_index=4,
                            is_extractable=True,
                        ),
                    ],
                    boundary="complete",
                    token_count=6,
                )
            ],
            token_usage=6,
        )
        envelope = _envelope(history=history)
        preprocessed = [_preprocessed("current message")]

        msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        payload = json.loads(msgs[1]["content"])

        history_item = payload["context"]["history"][0]
        assert history_item["text"] == "Rose: I moved to Boston.\nAlice: That is exciting."
        assert history_item["messages"] == [
            {
                "role": "speaker",
                "raw_role": "Rose",
                "speaker": "Rose",
                "text": "I moved to Boston.",
                "message_index": 3,
            },
            {
                "role": "speaker",
                "raw_role": "Alice",
                "speaker": "Alice",
                "text": "That is exciting.",
                "message_index": 4,
            },
        ]


# 4. Boundary metadata informs extraction behavior


class TestBoundaryConservatism:
    """Scenario: Boundary type adjusts extraction behavior."""

    def test_open_head_guidance(self) -> None:
        guidance = _boundary_guidance("open_head")
        assert "partial" in guidance.lower() or "conservative" in guidance.lower()

    def test_open_tail_guidance(self) -> None:
        guidance = _boundary_guidance("open_tail")
        assert "unfinished" in guidance.lower() or "conclusion" in guidance.lower()

    def test_orphan_guidance(self) -> None:
        guidance = _boundary_guidance("orphan")
        assert "explicit" in guidance.lower() or "stable" in guidance.lower()

    def test_compacted_guidance(self) -> None:
        guidance = _boundary_guidance("compacted")
        assert "head" in guidance.lower() and "tail" in guidance.lower()

    def test_complete_no_guidance(self) -> None:
        guidance = _boundary_guidance("complete")
        assert guidance == ""


# 5. Per-chunk recall


class TestPerChunkRecall:
    """Scenario: Recall results passed via envelope for dedup decisions."""

    def test_recall_in_context(self) -> None:
        recalled = [{"memory_id": "mem1", "content": "user likes Python", "source": "bm25", "score": 0.9}]
        envelope = _envelope(recalled=recalled)
        preprocessed = [_preprocessed("test")]
        msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        payload = json.loads(msgs[1]["content"])
        assert "related_memories" in payload["context"]
        assert payload["context"]["related_memories"][0]["memory_id"] == "mem1"


# 7. Fallback extraction from envelope


class TestEnvelopeFallback:
    """Scenario: Fallback extraction without LLM."""

    @pytest.mark.asyncio
    async def test_fallback_uses_extractable_only(self) -> None:
        extractor = VanillaMemoryExtractor(llm_client=None)  # type: ignore[arg-type]
        envelope = _envelope(
            messages=[
                _ref("user fact", "user"),
                _ref("system note", "system", extractable=False),
            ],
            boundary="complete",
        )
        preprocessed = [_preprocessed("user fact"), _preprocessed("system note")]
        result = await extractor.extract_from_envelope(envelope, preprocessed, _ctx())
        assert len(result.memories) == 1
        assert result.memories[0].content == "user fact"

    @pytest.mark.asyncio
    async def test_fallback_boundary_confidence(self) -> None:
        extractor = VanillaMemoryExtractor(llm_client=None)  # type: ignore[arg-type]
        envelope = _envelope(boundary="orphan")
        preprocessed = [_preprocessed("test")]
        result = await extractor.extract_from_envelope(envelope, preprocessed, _ctx())

    @pytest.mark.asyncio
    async def test_fallback_sources_include_evidence_index_metadata(self) -> None:
        extractor = VanillaMemoryExtractor(llm_client=None)  # type: ignore[arg-type]
        envelope = _envelope(
            messages=[
                TurnMessageRef(text="head text", role="assistant", message_index=0, is_extractable=True),
                TurnMessageRef(text="tail text", role="assistant", message_index=0, is_extractable=True),
            ],
            boundary="compacted",
        )
        preprocessed = [_preprocessed("head text"), _preprocessed("tail text")]

        result = await extractor.extract_from_envelope(envelope, preprocessed, _ctx())

        assert [source.message_index for source in result.sources] == [0, 0]
        assert [source.metadata["evidence_index"] for source in result.sources] == [0, 1]


# 8. Boundary confidence mapping


class TestBoundaryConfidence:
    """Scenario: Boundary maps to extraction confidence."""

    def test_complete_confidence(self) -> None:
        assert _boundary_confidence("complete") == 1.0

    def test_compacted_confidence(self) -> None:
        assert _boundary_confidence("compacted") == 0.9

    def test_open_head_confidence(self) -> None:
        assert _boundary_confidence("open_head") == 0.7

    def test_open_tail_confidence(self) -> None:
        assert _boundary_confidence("open_tail") == 0.7

    def test_orphan_confidence(self) -> None:
        assert _boundary_confidence("orphan") == 0.5


# 9. Current context messages (non-extractable chunk context)


class TestCurrentContextMessages:
    """Scenario: Non-extractable messages (e.g. compaction summaries) are
    preserved in the envelope context section, not lost."""

    def test_current_context_in_prompt(self) -> None:
        """Middle summary appears in context.current_context, not in extractable."""
        head_ref = _ref("head text", "user", extractable=True)
        head_ref_with_index = head_ref.model_copy(update={"message_index": 0})
        summary_ref = TurnMessageRef(
            text="[Compacted context summary]\nSummary of middle",
            role="system",
            timestamp=None,
            message_index=-1,
            is_extractable=False,
        )
        tail_ref = _ref("tail text", "user", extractable=True)
        tail_ref_with_index = tail_ref.model_copy(update={"message_index": 1})

        envelope = ExtractionEnvelope(
            extractable_messages=[head_ref_with_index, tail_ref_with_index],
            current_context_messages=[summary_ref],
            history=HistoryPack(),
            recalled_memories=[],
            boundary="complete",
            chunk_index=0,
        )
        preprocessed = [_preprocessed("head text"), _preprocessed("tail text")]
        msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        payload = json.loads(msgs[1]["content"])

        assert len(payload["extractable"]) == 2
        assert all(e["is_extractable"] for e in payload["extractable"])

        # Summary is in context.current_context
        assert "current_context" in payload["context"]
        assert len(payload["context"]["current_context"]) == 1
        ctx_msg = payload["context"]["current_context"][0]
        assert ctx_msg["role"] == "system"
        assert "Compacted context summary" in ctx_msg["text"]

        extractable_texts = [e["text"] for e in payload["extractable"]]
        assert all("Compacted context summary" not in t for t in extractable_texts)

    def test_no_current_context_when_empty(self) -> None:
        """No current_context key when no non-extractable messages."""
        envelope = _envelope()
        preprocessed = [_preprocessed("test")]
        msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        payload = json.loads(msgs[1]["content"])
        assert "current_context" not in payload["context"]

    def test_instruction_still_forbids_extracting_from_context(self) -> None:
        """Instruction still says: only extract from extractable, not context."""
        envelope = _envelope()
        envelope.current_context_messages = [
            TurnMessageRef(text="summary", role="system", message_index=-1, is_extractable=False),
        ]
        preprocessed = [_preprocessed("test")]
        msgs = _envelope_prompt_messages(envelope, preprocessed, _ctx())
        payload = json.loads(msgs[1]["content"])
        instruction = payload["instruction"].lower()
        assert "extractable" in instruction
        assert "context" in instruction
