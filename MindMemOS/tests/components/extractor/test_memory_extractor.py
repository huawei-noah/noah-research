from types import SimpleNamespace

import pytest
from mindmemos.components.extractor.vanilla.memory import (
    _envelope_prompt_messages,
    _normalize_extraction_payload,
)
from mindmemos.typing.algo import ExtractionEnvelope, TurnMessageRef
from mindmemos.typing.llm import ChatResponse
from mindmemos.typing.memory import MemoryRequestContext
from mindmemos.typing.service import AddPipelineInput

from mindmemos.components.chunker import MessageSegmenter
from mindmemos.components.extractor.vanilla import (
    MemoryExtractionResult,
    VanillaMemoryExtractor,
)
from mindmemos.components.text import TextPreprocessor
from mindmemos.config import TextProcessingConfig


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


def make_text_preprocessor() -> TextPreprocessor:
    return TextPreprocessor(
        TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        )
    )


def make_segment_and_preprocessed(message: dict):
    inp = AddPipelineInput(messages=[message])
    segment = MessageSegmenter().segment(inp)[0][0]
    preprocessed = make_text_preprocessor().preprocess_text(
        segment.text,
        source_ref=segment.source_ref,
        segment_id=segment.segment_id,
    )
    return segment, preprocessed


def make_envelope(segments, *, recalled_memories=None) -> ExtractionEnvelope:
    return ExtractionEnvelope(
        extractable_messages=[
            TurnMessageRef(
                text=segment.text,
                role=segment.role or "user",
                timestamp=segment.timestamp,
                message_index=segment.message_index if segment.message_index is not None else index,
                is_extractable=True,
            )
            for index, segment in enumerate(segments)
        ],
        recalled_memories=recalled_memories or [],
        boundary="complete",
        chunk_index=0,
    )


def test_vanilla_memory_extractor_exposes_only_envelope_entrypoint() -> None:
    assert not hasattr(VanillaMemoryExtractor(llm_client=None), "extract")


class FakeLlmClient:
    def __init__(self, parsed=None, *, fail: bool = False) -> None:
        self.parsed = parsed
        self.fail = fail
        self.calls = []

    async def chat(self, task: str, messages: list[dict], format_parser=None, **kwargs):
        self.calls.append(SimpleNamespace(task=task, messages=messages, kwargs=kwargs))
        if self.fail:
            raise RuntimeError("llm failed")
        parsed = self.parsed
        if parsed is None and format_parser is not None:
            parsed = format_parser("{}")
        return ChatResponse(finish_reason="stop", content="{}", parsed=parsed)


@pytest.mark.asyncio
async def test_vanilla_memory_extractor_parses_llm_result() -> None:
    segment, preprocessed = make_segment_and_preprocessed({"text": "Kai prefers FastAPI."})
    fake_llm = FakeLlmClient(
        parsed={
            "memories": [
                {
                    "ref_id": "m1",
                    "content": "Kai prefers FastAPI.",
                    "mem_type": "profile",
                    "confidence": 0.86,
                    "importance": 0.7,
                    "entities": ["e1"],
                    "source_refs": ["s1"],
                    "related_memory_ids": ["mem-existing"],
                    "action_hint": "add",
                    "reason": "explicit preference",
                }
            ],
            "entities": [
                {
                    "ref_id": "e1",
                    "entity_name": "FastAPI",
                    "entity_type": "framework",
                    "confidence": 0.8,
                }
            ],
            "sources": [{"ref_id": "s1", "source_type": "message", "message_index": 0}],
            "property_bindings": [],
        }
    )

    result = await VanillaMemoryExtractor(llm_client=fake_llm).extract_from_envelope(
        make_envelope([segment]),
        [preprocessed],
        make_context(),
    )

    assert result.memories[0].content == "Kai prefers FastAPI."
    assert result.memories[0].mem_type == "profile"
    assert result.memories[0].related_memory_ids == ["mem-existing"]
    assert result.entities[0].entity_name == "FastAPI"
    assert result.memories[0].metadata["extractor"] == "vanilla_llm_chunked"
    assert fake_llm.calls[0].task == "memory.add.extract"


@pytest.mark.asyncio
async def test_vanilla_memory_extractor_falls_back_when_llm_fails() -> None:
    segment, preprocessed = make_segment_and_preprocessed(
        {"role": "user", "content": "Remember that Kai uses Qdrant.", "timestamp": 1770000000000}
    )

    result = await VanillaMemoryExtractor(llm_client=FakeLlmClient(fail=True)).extract_from_envelope(
        make_envelope([segment]),
        [preprocessed],
        make_context(),
    )

    assert result.memories[0].content == "Remember that Kai uses Qdrant."
    assert result.memories[0].mem_type == "fact"
    assert result.memories[0].metadata["extractor"] == "fallback_chunked"
    assert result.sources[0].message_index == 0


@pytest.mark.asyncio
async def test_vanilla_memory_extractor_normalizes_object_source_refs() -> None:
    segment, preprocessed = make_segment_and_preprocessed(
        {
            "role": "user",
            "content": "Besides classes, Jon offers one-on-one mentoring and training.",
            "timestamp": 1770000000000,
        }
    )
    fake_llm = FakeLlmClient(
        parsed={
            "memories": [
                {
                    "ref_id": "m1",
                    "content": "Jon offers one-on-one mentoring and training.",
                    "mem_type": "fact",
                    "confidence": 0.8,
                    "source_refs": [
                        {
                            "ref_id": "s1",
                            "source_type": "message",
                            "message_index": 0,
                            "metadata": {"evidence_index": 0},
                        }
                    ],
                    "action_hint": "add",
                }
            ],
            "entities": [],
            "sources": [],
        }
    )

    result = await VanillaMemoryExtractor(llm_client=fake_llm).extract_from_envelope(
        make_envelope([segment]),
        [preprocessed],
        make_context(),
    )

    assert result.memories[0].content == "Jon offers one-on-one mentoring and training."
    assert result.memories[0].source_refs == ["s1"]
    assert result.sources[0].ref_id == "s1"
    assert result.sources[0].metadata["evidence_index"] == 0
    assert result.memories[0].metadata["extractor"] == "vanilla_llm_chunked"


@pytest.mark.asyncio
async def test_vanilla_memory_extractor_ignores_top_level_empty_entities_for_local_fallback() -> None:
    segment, preprocessed = make_segment_and_preprocessed({"role": "user", "content": "我暑假去广州旅游。"})
    fake_llm = FakeLlmClient(
        parsed={
            "memories": [
                {
                    "ref_id": "m1",
                    "content": "用户计划暑假去广州旅游。",
                    "mem_type": "profile",
                    "confidence": 0.9,
                    "source_refs": ["s0"],
                    "action_hint": "add",
                }
            ],
            "entities": [],
        }
    )

    result = await VanillaMemoryExtractor(llm_client=fake_llm).extract_from_envelope(
        make_envelope([segment]),
        [preprocessed],
        make_context(),
    )

    assert result.memories[0].entities == []
    assert "entities" not in result.memories[0].model_fields_set


def test_normalize_extraction_payload_omits_empty_defaults_and_empty_metadata() -> None:
    normalized = _normalize_extraction_payload(
        {
            "memories": [
                {
                    "ref_id": "m1",
                    "content": "用户计划暑假去广州旅游。",
                    "mem_type": "fact",
                    "source_refs": ["s0"],
                    "related_memory_ids": [],
                    "target_memory_id": None,
                    "metadata": {
                        "temporal_text": "暑假",
                        "resolved_event_date": None,
                        "resolved_event_range": [],
                    },
                }
            ],
            "entities": [],
        }
    )

    memory = normalized["memories"][0]
    assert "entities" not in memory
    assert "related_memory_ids" not in memory
    assert "target_memory_id" not in memory
    assert "metadata" not in memory
    assert "entities" not in normalized


def test_normalize_extraction_payload_keeps_useful_metadata() -> None:
    normalized = _normalize_extraction_payload(
        {
            "memories": [
                {
                    "ref_id": "m1",
                    "content": "On 2026-06-17, the user booked a Guangzhou hotel.",
                    "mem_type": "fact",
                    "source_refs": ["s0"],
                    "metadata": {
                        "temporal_text": "yesterday",
                        "resolved_event_date": "2026-06-17",
                        "resolved_event_range": [],
                        "ignored": "drop me",
                    },
                }
            ],
        }
    )

    assert normalized["memories"][0]["metadata"] == {
        "temporal_text": "yesterday",
        "resolved_event_date": "2026-06-17",
    }


def test_memory_extraction_result_rejects_unknown_memory_type() -> None:
    with pytest.raises(ValueError):
        MemoryExtractionResult.model_validate(
            {
                "memories": [
                    {
                        "ref_id": "m1",
                        "content": "bad",
                        "mem_type": "unknown",
                        "confidence": 0.5,
                    }
                ]
            }
        )


def _make_preprocessed(lang: str = "en") -> object:
    """Build a minimal PreprocessedText-like object for prompt tests."""

    class _FakePreprocessed:
        def __init__(self) -> None:
            self.normalized_text = "User prefers FastAPI."
            self.lang = lang
            self.content_hash = "abc123"
            self.bm25_text = "user prefers fastapi"
            self.tokens = ["user", "prefers", "fastapi"]
            self.entities = []

    return _FakePreprocessed()


def _make_segment() -> object:
    """Build a minimal SourceAwareSegment-like object."""

    class _FakeSourceRef:
        source_type = "message"
        source_id = None

    class _FakeSegment:
        segment_id = "seg_0"
        text = "User prefers FastAPI."
        role = "user"
        message_index = 0
        timestamp = None
        source_ref = _FakeSourceRef()

    return _FakeSegment()


class TestPromptSelection:
    """Tests for language-aware system prompt selection."""

    def test_chinese_input_selects_chinese_prompt(self) -> None:
        ctx = make_context()
        preprocessed = _make_preprocessed(lang="zh")
        segment = _make_segment()
        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        assert messages[0]["role"] == "system"
        assert "你是 MindMemOS" in messages[0]["content"]

    def test_english_input_selects_english_prompt(self) -> None:
        ctx = make_context()
        preprocessed = _make_preprocessed(lang="en")
        segment = _make_segment()
        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        assert messages[0]["role"] == "system"
        assert "You are the memory extractor for MindMemOS" in messages[0]["content"]

    def test_english_prompt_preserves_core_contract(self) -> None:
        ctx = make_context()
        preprocessed = _make_preprocessed(lang="en")
        segment = _make_segment()
        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        prompt = messages[0]["content"]

        assert "Factual fidelity and subject attribution" in prompt
        assert "directly supported by source_refs" in prompt
        assert "Assistant suggestions, guesses, summaries" in prompt
        assert "retrieval anchors" in prompt
        assert "dates, times, places, person names, organization names" in prompt
        assert "meaning-changing qualifiers" in prompt
        assert "self-contained" in prompt
        assert "Do not output action_hint=skip" in prompt
        assert "mem_type must use only the values above" in prompt
        assert "may be reused in the future" in prompt
        assert "yesterday" in prompt
        assert "resolved_event_date" in prompt
        assert "message_time as the basis" in prompt
        assert "compacted context is only for resolution" in prompt
        assert "Output strict, one-line, minified JSON only" in prompt
        assert "Do not output entities" in prompt
        assert '"action_hint": "add | reinforce | update | merge"' in prompt

    def test_mixed_language_defaults_to_english_prompt(self) -> None:
        ctx = make_context()
        preprocessed = _make_preprocessed(lang="mixed")
        segment = _make_segment()
        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        assert messages[0]["role"] == "system"
        assert "You are the memory extractor for MindMemOS" in messages[0]["content"]

    def test_unknown_language_defaults_to_english_prompt(self) -> None:
        ctx = make_context()
        preprocessed = _make_preprocessed(lang="unknown")
        segment = _make_segment()
        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        assert "You are the memory extractor for MindMemOS" in messages[0]["content"]

    def test_chinese_prompt_preserves_core_contract(self) -> None:
        ctx = make_context()
        preprocessed = _make_preprocessed(lang="zh")
        segment = _make_segment()
        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        prompt = messages[0]["content"]

        assert "事实忠实与主体准确" in prompt
        assert "source_refs 直接支持" in prompt
        assert "assistant 的建议" in prompt
        assert "检索锚点" in prompt
        assert "日期、时间、地点、人名、组织名" in prompt
        assert "会改变含义的限定" in prompt
        assert "自包含" in prompt
        assert "mem_type 只能使用以上值" in prompt
        assert "未来可能复用" in prompt
        assert "不要输出 action_hint=skip" in prompt
        assert "today、yesterday、last Friday" in prompt
        assert "resolved_event_date" in prompt
        assert "以该消息的 message_time 为基准" in prompt
        assert "压缩上下文仅用于消解、去重和关联" in prompt
        assert "只输出严格、单行、minified JSON" in prompt
        assert "不输出 entities" in prompt
        assert '"action_hint": "add | reinforce | update | merge"' in prompt

    def test_user_payload_contains_extractable_messages(self) -> None:
        import json

        ctx = make_context()
        preprocessed = _make_preprocessed(lang="en")
        segment = _make_segment()
        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        user_payload = json.loads(messages[1]["content"])
        assert user_payload["request_id"] == "req-1"
        assert len(user_payload["extractable"]) == 1
        assert user_payload["extractable"][0]["index"] == 0
        assert user_payload["extractable"][0]["evidence_index"] == 0
        assert "original_message_index" not in user_payload["extractable"][0]

    def test_user_payload_instruction_uses_source_refs_not_top_level_sources(self) -> None:
        import json

        ctx = make_context()
        preprocessed = _make_preprocessed(lang="en")
        segment = _make_segment()

        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        user_payload = json.loads(messages[1]["content"])
        instruction = user_payload["instruction"]

        assert "source_refs" in instruction
        assert "s{evidence_index}" in instruction
        assert "Do not output entities, top-level sources, or property_bindings." in instruction
        assert "sources[]" not in instruction
        assert "original_message_index" not in instruction

    def test_user_payload_contains_message_time_for_relative_time_resolution(self) -> None:
        import json

        ctx = make_context()
        preprocessed = _make_preprocessed(lang="en")
        envelope = ExtractionEnvelope(
            extractable_messages=[
                TurnMessageRef(
                    text="I visited the support group yesterday.",
                    role="user",
                    timestamp=1674172800000,
                    message_index=0,
                    is_extractable=True,
                )
            ],
            boundary="complete",
            chunk_index=0,
        )

        messages = _envelope_prompt_messages(envelope, [preprocessed], ctx)
        user_payload = json.loads(messages[1]["content"])
        extractable = user_payload["extractable"][0]

        assert extractable["timestamp_ms"] == 1674172800000
        assert extractable["message_time"] == "2023-01-20 00:00:00"
        assert extractable["message_date"] == "2023-01-20"

    def test_user_payload_omits_preprocessed_entities_to_keep_prompt_small(self) -> None:
        import json

        class _Entity:
            canonical_name = "FastAPI"
            name = "fastapi"
            entity_type = "framework"
            confidence = 0.9

        ctx = make_context()
        preprocessed = _make_preprocessed(lang="en")
        preprocessed.entities = [_Entity()]
        segment = _make_segment()

        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        user_payload = json.loads(messages[1]["content"])

        assert "entities" not in user_payload["extractable"][0]

    def test_prompt_uses_source_refs_without_top_level_sources(self) -> None:
        ctx = make_context()
        preprocessed = _make_preprocessed(lang="zh")
        segment = _make_segment()

        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        prompt = messages[0]["content"]

        assert '"sources": [' not in prompt
        assert '"entities": [' not in prompt
        assert '"source_refs": ["s0"]' in prompt

    def test_named_speaker_payload_preserves_speaker_identity(self) -> None:
        import json

        ctx = make_context()
        preprocessed = _make_preprocessed(lang="en")
        envelope = ExtractionEnvelope(
            extractable_messages=[
                TurnMessageRef(
                    text="I moved to Boston.",
                    role="speaker",
                    raw_role="Rose",
                    speaker="Rose",
                    message_index=0,
                    is_extractable=True,
                )
            ],
            boundary="complete",
            chunk_index=0,
        )

        messages = _envelope_prompt_messages(envelope, [preprocessed], ctx)
        user_payload = json.loads(messages[1]["content"])

        assert user_payload["extractable"][0]["role"] == "speaker"
        assert user_payload["extractable"][0]["raw_role"] == "Rose"
        assert user_payload["extractable"][0]["speaker"] == "Rose"
        assert "first-person pronouns refer to the message speaker" in user_payload["instruction"]
        assert 'Do not rewrite unknown speakers as "the user"' in user_payload["instruction"]


class TestDominantLang:
    """Tests for _dominant_lang helper."""

    def test_all_zh_returns_zh(self) -> None:
        from mindmemos.components.extractor.vanilla.memory import _dominant_lang

        assert _dominant_lang([_make_preprocessed("zh"), _make_preprocessed("zh")]) == "zh"

    def test_single_zh_among_en_returns_zh(self) -> None:
        from mindmemos.components.extractor.vanilla.memory import _dominant_lang

        assert _dominant_lang([_make_preprocessed("en"), _make_preprocessed("zh")]) == "zh"

    def test_all_en_returns_en(self) -> None:
        from mindmemos.components.extractor.vanilla.memory import _dominant_lang

        assert _dominant_lang([_make_preprocessed("en")]) == "en"

    def test_mixed_without_zh_returns_en(self) -> None:
        from mindmemos.components.extractor.vanilla.memory import _dominant_lang

        assert _dominant_lang([_make_preprocessed("mixed")]) == "en"


class TestRealLlmChain:
    """Tests exercising the full LLM extraction chain with realistic output."""

    @pytest.mark.asyncio
    async def test_full_new_schema_extraction(self) -> None:
        """End-to-end: LLM returns output matching the compact prompt schema."""
        segment, preprocessed = make_segment_and_preprocessed({"text": "Kai prefers FastAPI for all backend APIs."})
        fake_llm = FakeLlmClient(
            parsed={
                "memories": [
                    {
                        "ref_id": "m1",
                        "content": "Kai prefers using FastAPI for building backend APIs.",
                        "mem_type": "profile",
                        "confidence": 0.92,
                        "importance": 0.8,
                        "entities": ["e1"],
                        "source_refs": ["s1"],
                        "related_memory_ids": [],
                        "action_hint": "add",
                        "reason": "Explicit long-term preference statement",
                        "segment_id": segment.segment_id,
                        "metadata": {
                            "evidence_summary": "User stated preference directly",
                            "rewrite_policy": "first_person_to_objective_user",
                        },
                    }
                ],
                "entities": [
                    {
                        "ref_id": "e1",
                        "entity_name": "FastAPI",
                        "entity_type": "framework",
                        "description": "Python web framework",
                        "confidence": 0.95,
                        "metadata": {},
                    }
                ],
                "sources": [{"ref_id": "s1", "source_type": "message", "message_index": 0, "metadata": {}}],
            }
        )

        result = await VanillaMemoryExtractor(llm_client=fake_llm).extract_from_envelope(
            make_envelope([segment]),
            [preprocessed],
            make_context(),
        )

        assert len(result.memories) == 1
        m = result.memories[0]
        assert m.ref_id == "m1"
        assert m.content == "Kai prefers using FastAPI for building backend APIs."
        assert m.mem_type == "profile"
        assert m.confidence == 0.92
        assert m.importance == 0.8
        assert m.action_hint == "add"
        assert m.entities == ["e1"]
        assert m.source_refs == ["s1"]
        assert m.reason == "Explicit long-term preference statement"
        assert m.metadata["extractor"] == "vanilla_llm_chunked"
        assert "evidence_summary" not in m.metadata
        assert "rewrite_policy" not in m.metadata

        assert len(result.entities) == 1
        assert result.entities[0].ref_id == "e1"
        assert result.entities[0].entity_name == "FastAPI"
        assert result.entities[0].entity_type == "framework"

    @pytest.mark.asyncio
    async def test_extraction_with_reinforce_action(self) -> None:
        """LLM returns reinforce action with target_memory_id."""
        segment, preprocessed = make_segment_and_preprocessed({"text": "Kai uses Qdrant."})
        fake_llm = FakeLlmClient(
            parsed={
                "memories": [
                    {
                        "ref_id": "m1",
                        "content": "Kai uses Qdrant.",
                        "mem_type": "fact",
                        "confidence": 0.85,
                        "importance": 0.5,
                        "entities": ["e1"],
                        "source_refs": ["s1"],
                        "related_memory_ids": ["mem_old_1"],
                        "action_hint": "reinforce",
                        "target_memory_id": "mem_old_1",
                        "reason": "Duplicates existing memory about Kai's tool preference",
                        "segment_id": segment.segment_id,
                        "metadata": {},
                    }
                ],
                "entities": [
                    {
                        "ref_id": "e1",
                        "entity_name": "Qdrant",
                        "entity_type": "tool",
                        "confidence": 0.9,
                        "metadata": {},
                    }
                ],
                "sources": [{"ref_id": "s1", "source_type": "message", "message_index": 0, "metadata": {}}],
            }
        )

        result = await VanillaMemoryExtractor(llm_client=fake_llm).extract_from_envelope(
            make_envelope([segment]),
            [preprocessed],
            make_context(),
        )

        assert result.memories[0].action_hint == "reinforce"
        assert result.memories[0].target_memory_id == "mem_old_1"
        assert result.memories[0].related_memory_ids == ["mem_old_1"]

    @pytest.mark.asyncio
    async def test_extraction_with_merge_action(self) -> None:
        """LLM returns merge action with multiple related_memory_ids."""
        segment, preprocessed = make_segment_and_preprocessed(
            {"text": "Kai uses FastAPI with Qdrant for memory storage."}
        )
        fake_llm = FakeLlmClient(
            parsed={
                "memories": [
                    {
                        "ref_id": "m1",
                        "content": "Kai uses FastAPI with Qdrant as the backend for memory storage.",
                        "mem_type": "fact",
                        "confidence": 0.88,
                        "importance": 0.7,
                        "entities": ["e1", "e2"],
                        "source_refs": ["s1"],
                        "related_memory_ids": ["mem_old_1", "mem_old_2"],
                        "action_hint": "merge",
                        "reason": "Combines separate facts about Kai's tech stack",
                        "segment_id": segment.segment_id,
                        "metadata": {},
                    }
                ],
                "entities": [
                    {
                        "ref_id": "e1",
                        "entity_name": "FastAPI",
                        "entity_type": "framework",
                        "confidence": 0.9,
                        "metadata": {},
                    },
                    {"ref_id": "e2", "entity_name": "Qdrant", "entity_type": "tool", "confidence": 0.9, "metadata": {}},
                ],
                "sources": [{"ref_id": "s1", "source_type": "message", "message_index": 0, "metadata": {}}],
            }
        )

        result = await VanillaMemoryExtractor(llm_client=fake_llm).extract_from_envelope(
            make_envelope([segment]),
            [preprocessed],
            make_context(),
        )

        assert result.memories[0].action_hint == "merge"
        assert result.memories[0].related_memory_ids == ["mem_old_1", "mem_old_2"]
        assert len(result.entities) == 2

    @pytest.mark.asyncio
    async def test_empty_candidates(self) -> None:
        """LLM returns empty arrays — no error, result is empty."""
        segment, preprocessed = make_segment_and_preprocessed({"text": "ok"})
        fake_llm = FakeLlmClient(parsed={"memories": [], "entities": [], "sources": []})

        result = await VanillaMemoryExtractor(llm_client=fake_llm).extract_from_envelope(
            make_envelope([segment]),
            [preprocessed],
            make_context(),
        )

        assert result.memories == []
        assert result.entities == []
        assert result.sources == []
        assert result.memories[0].metadata.get("extractor") != "fallback_chunked" if result.memories else True

    @pytest.mark.asyncio
    async def test_recall_context_injected_into_prompt(self) -> None:
        """Recall results appear as related_memories in the user payload."""
        import json

        from mindmemos.typing.memory import MemoryView, RelatedMemoryCandidate, RelatedMemoryRecallResult

        ctx = make_context()
        preprocessed = _make_preprocessed(lang="en")
        segment = _make_segment()

        old_memory = MemoryView(
            memory_id="mem_old_1",
            content="Kai uses PostgreSQL.",
            mem_type="fact",
            status="active",
            metadata={},
            project_id="proj-1",
        )
        recall_result = RelatedMemoryRecallResult(
            duplicate=None,
            candidates=[
                RelatedMemoryCandidate(
                    memory_id="mem_old_1",
                    score=0.85,
                    source="bm25",
                    rank=1,
                    memory=old_memory,
                    debug={},
                )
            ],
        )
        recalled_memories = [
            {
                "memory_id": candidate.memory_id,
                "content": candidate.memory.content if candidate.memory else "",
                "source": candidate.source,
                "score": candidate.score,
            }
            for candidate in recall_result.candidates
        ]

        messages = _envelope_prompt_messages(
            make_envelope([segment], recalled_memories=recalled_memories),
            [preprocessed],
            ctx,
        )
        user_payload = json.loads(messages[1]["content"])

        assert "related_memories" in user_payload["context"]
        assert len(user_payload["context"]["related_memories"]) == 1
        assert user_payload["context"]["related_memories"][0]["memory_id"] == "mem_old_1"
        assert user_payload["context"]["related_memories"][0]["content"] == "Kai uses PostgreSQL."

    @pytest.mark.asyncio
    async def test_multi_segment_extraction(self) -> None:
        """Multiple segments produce multiple candidates."""
        segment1, preprocessed1 = make_segment_and_preprocessed({"text": "Kai uses FastAPI."})
        segment2, preprocessed2 = make_segment_and_preprocessed({"text": "The project uses Qdrant."})
        fake_llm = FakeLlmClient(
            parsed={
                "memories": [
                    {
                        "ref_id": "m1",
                        "content": "Kai uses FastAPI.",
                        "mem_type": "profile",
                        "confidence": 0.8,
                        "entities": ["e1"],
                        "source_refs": ["s1"],
                        "action_hint": "add",
                        "reason": "Tool preference",
                        "segment_id": segment1.segment_id,
                        "metadata": {},
                    },
                    {
                        "ref_id": "m2",
                        "content": "The project uses Qdrant.",
                        "mem_type": "fact",
                        "confidence": 0.75,
                        "entities": ["e2"],
                        "source_refs": ["s2"],
                        "action_hint": "add",
                        "reason": "Infrastructure fact",
                        "segment_id": segment2.segment_id,
                        "metadata": {},
                    },
                ],
                "entities": [
                    {
                        "ref_id": "e1",
                        "entity_name": "FastAPI",
                        "entity_type": "framework",
                        "confidence": 0.9,
                        "metadata": {},
                    },
                    {"ref_id": "e2", "entity_name": "Qdrant", "entity_type": "tool", "confidence": 0.9, "metadata": {}},
                ],
                "sources": [
                    {"ref_id": "s1", "source_type": "message", "message_index": 0, "metadata": {}},
                    {"ref_id": "s2", "source_type": "message", "message_index": 1, "metadata": {}},
                ],
            }
        )

        result = await VanillaMemoryExtractor(llm_client=fake_llm).extract_from_envelope(
            make_envelope([segment1, segment2]),
            [preprocessed1, preprocessed2],
            make_context(),
        )

        assert len(result.memories) == 2
        assert result.memories[0].mem_type == "profile"
        assert result.memories[1].mem_type == "fact"
        assert len(result.entities) == 2


def test_envelope_prompt_uses_entity_prompt_when_enabled() -> None:
    import json

    ctx = make_context()
    segment, preprocessed = make_segment_and_preprocessed({"text": "Kai works at OpenAI."})
    messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx, enable_entities=True)
    system_prompt = messages[0]["content"]
    user_payload = json.loads(messages[1]["content"])

    # entity prompt is selected
    assert '"entities": ["e1"]' in system_prompt
    assert "entity_name" in system_prompt
    # user instruction must drop the entities ban to stay consistent with the entity prompt
    assert "Do not output entities" not in user_payload["instruction"]
    assert "top-level sources" in user_payload["instruction"]
