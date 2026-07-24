"""Integration tests for the chunked add pipeline (build).

Tests the full flow: turn grouping → chunk planning → history packing →
envelope extraction → dedup → plan → vectorize, using mock components.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mindmemos.components.extractor.vanilla import (
    CandidateDeduplicator,
    ExtractedEntityCandidate,
    ExtractedMemoryCandidate,
    ExtractedSourceCandidate,
    MemoryExtractionResult,
    VanillaMemoryExtractor,
)
from mindmemos.components.extractor.vanilla._safety_gate import AddSafetyGate
from mindmemos.components.extractor.vanilla.add_builder import AddCoreBuilder
from mindmemos.components.text import TextPreprocessor
from mindmemos.config import TextProcessingConfig, VanillaAddConfig
from mindmemos.typing.algo import TurnCompactionSummary
from mindmemos.typing.memory import (
    REL_EXTRACTED_FROM,
    REL_MENTIONED_IN_SOURCE,
    MemoryRequestContext,
    PreprocessedText,
    UrlMessage,
)
from mindmemos.typing.service import AddPipelineInput, DialogueMessage


def _ctx() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="test-req",
        account_id="acc1",
        project_id="proj1",
        api_key_uuid="key1",
        user_id="u1",
        session_id="s1",
    )


def _pp(text: str) -> PreprocessedText:
    return PreprocessedText(
        text=text,
        normalized_text=text,
        content_hash=f"hash_{text[:8]}",
        bm25_text=text,
    )


class MockPreprocessor:
    """Minimal mock that returns a PreprocessedText for any input."""

    def preprocess_text(self, text: str, **kwargs) -> PreprocessedText:
        return _pp(text)

    def preprocess_many(self, texts: list[str], **kwargs) -> list[PreprocessedText]:
        return [self.preprocess_text(text) for text in texts]


class MockExtractor(VanillaMemoryExtractor):
    """Extractor that returns one candidate per extractable message."""

    def __init__(self) -> None:
        super().__init__(llm_client=None)  # type: ignore[arg-type]

    async def extract_from_envelope(self, envelope, preprocessed_texts, context):
        from mindmemos.components.extractor.vanilla import ExtractedSourceCandidate

        memories = []
        sources = []
        for i, (msg_ref, pp) in enumerate(zip(envelope.extractable_messages, preprocessed_texts, strict=False)):
            if not msg_ref.is_extractable:
                continue
            source_ref_id = f"s{i}"
            memories.append(
                ExtractedMemoryCandidate(
                    ref_id=f"m{i}",
                    content=pp.normalized_text,
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=[source_ref_id],
                    action_hint="add",
                    reason="mock_chunked_extraction",
                    metadata={"extractor": "mock_chunked"},
                )
            )
            sources.append(
                ExtractedSourceCandidate(
                    ref_id=source_ref_id,
                    source_type="message",
                    message_index=msg_ref.message_index,
                )
            )
        return MemoryExtractionResult(memories=memories, sources=sources)


class MockRecall:
    async def list_active_memories(self, context):
        return []

    async def recall(self, context, preprocessed, **kwargs):
        return None


class MockVectorizer:
    async def vectorize(self, memory_id, preprocessed, content, consistency="fast"):
        from mindmemos.typing.memory import VectorWrite

        return VectorWrite(
            memory_id=memory_id,
            semantic_vector=[0.1] * 128,
            bm25_indices=[],
            bm25_values=[],
        ), False

    async def vectorize_entities(self, entities, *, memories_by_entity=None, consistency="fast"):
        return [], False


class RecordingVectorizer(MockVectorizer):
    def __init__(self) -> None:
        self.entity_batches = []

    async def vectorize_entities(self, entities, *, memories_by_entity=None, consistency="fast"):
        self.entity_batches.append(list(entities))
        return [], False


def _make_builder(*, llm_client=None, memory_extractor=None) -> AddCoreBuilder:
    return AddCoreBuilder(
        text_preprocessor=MockPreprocessor(),  # type: ignore[arg-type]
        memory_extractor=memory_extractor or MockExtractor(),  # type: ignore[arg-type]
        candidate_deduplicator=CandidateDeduplicator(),
        related_memory_recall=MockRecall(),  # type: ignore[arg-type]
        safety_gate=AddSafetyGate(),
        vectorizer=MockVectorizer(),  # type: ignore[arg-type]
        llm_client=llm_client,
    )


def _real_text_preprocessor() -> TextPreprocessor:
    return TextPreprocessor(
        TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        )
    )


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


class TestSingleMessageChunked:
    """Single message input should produce one chunk with one candidate."""

    @pytest.mark.asyncio
    async def test_single_user_message(self) -> None:
        builder = _make_builder()
        inp = AddPipelineInput(
            messages=[DialogueMessage(role="user", content="I live in Tokyo")],
        )
        plan, events, updates = await builder.build(inp, _ctx())
        assert len(events) >= 1
        assert events[0].operation == "add"
        assert "Tokyo" in events[0].content


class TestMultiTurnChunked:
    """Multi-turn dialogue should group turns into chunks."""

    @pytest.mark.asyncio
    async def test_two_turns_one_chunk(self) -> None:
        config = _small_budget_config(chunk_soft_token_budget=5000, chunk_hard_token_budget=8000)
        builder = _make_builder()
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="user", content="What is Python?"),
                DialogueMessage(role="assistant", content="Python is a programming language."),
                DialogueMessage(role="user", content="How do I install it?"),
                DialogueMessage(role="assistant", content="Use pip or conda."),
            ],
        )
        plan, events, updates = await builder.build(inp, _ctx(), config=config)
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_named_speaker_dialogue_is_extractable(self) -> None:
        config = VanillaAddConfig(chunk_soft_token_budget=5000, chunk_hard_token_budget=8000)
        builder = _make_builder()
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="Rose", content="I moved to Boston."),
                DialogueMessage(role="Alice", content="That is exciting."),
                DialogueMessage(role="Rose", content="I like the parks."),
            ],
        )

        plan, events, updates = await builder.build(inp, _ctx(), config=config)

        assert len(events) >= 1
        assert len(plan.memories) >= 1
        source_roles = {source.metadata.get("source_role") for source in plan.sources}
        assert source_roles == {"speaker"}
        source_speakers = {source.metadata.get("source_speaker") for source in plan.sources}
        assert source_speakers == {"Rose", "Alice"}

    @pytest.mark.asyncio
    async def test_locomo_named_speaker_dialogue_is_extractable(self) -> None:
        """Regression sample from locomo10.json conversation 0, session_1 D1:11-D1:18."""
        config = VanillaAddConfig(chunk_soft_token_budget=5000, chunk_hard_token_budget=8000)
        builder = _make_builder()
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role=speaker, content=text) for speaker, text in LOCOMO_CAROLINE_MELANIE_D1_11_TO_D1_18
            ],
        )

        plan, events, updates = await builder.build(inp, _ctx(), config=config)

        assert len(events) == len(LOCOMO_CAROLINE_MELANIE_D1_11_TO_D1_18)
        assert len(plan.memories) == len(LOCOMO_CAROLINE_MELANIE_D1_11_TO_D1_18)
        assert all(memory.content for memory in plan.memories)
        source_roles = {source.metadata.get("source_role") for source in plan.sources}
        assert source_roles == {"speaker"}
        source_speakers = {source.metadata.get("source_speaker") for source in plan.sources}
        assert source_speakers == {"Caroline", "Melanie"}
        assert {source.metadata.get("message_index") for source in plan.sources} == set(range(8))
        assert any("counseling" in event.content for event in events)
        assert any("swimming with the kids" in event.content for event in events)

    @pytest.mark.asyncio
    async def test_multiple_chunks(self) -> None:
        """Many turns with tight budgets should create multiple chunks."""
        config = _small_budget_config(
            chunk_soft_token_budget=1500,
            chunk_hard_token_budget=3000,
            turn_hard_token_budget=5000,
        )
        builder = _make_builder()
        msgs = []
        for i in range(10):
            msgs.append(DialogueMessage(role="user", content=f"User question {i}: " + "word " * 100))
            msgs.append(DialogueMessage(role="assistant", content=f"Assistant answer {i}: " + "response " * 100))

        inp = AddPipelineInput(messages=msgs)
        plan, events, updates = await builder.build(inp, _ctx(), config=config)
        assert len(events) >= 1


class _EnvelopeCaptureExtractor(VanillaMemoryExtractor):
    def __init__(self) -> None:
        super().__init__(llm_client=None)  # type: ignore[arg-type]
        self.envelopes = []

    async def extract_from_envelope(self, envelope, preprocessed_texts, context):
        self.envelopes.append(envelope)
        return MemoryExtractionResult()


class _SummaryLlmClient:
    def __init__(self) -> None:
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=TurnCompactionSummary(general_summary="middle summary"))


class TestLongTurnCompactionIntegration:
    @pytest.mark.asyncio
    async def test_builder_summarizes_long_turn_and_marks_compacted_envelope(self) -> None:
        summary_client = _SummaryLlmClient()
        extractor = _EnvelopeCaptureExtractor()
        builder = _make_builder(llm_client=summary_client, memory_extractor=extractor)
        config = VanillaAddConfig(
            chunk_soft_token_budget=30,
            chunk_hard_token_budget=40,
            turn_hard_token_budget=10,
            history_soft_token_budget=1,
            history_hard_token_budget=2,
            compaction_head_tokens=4,
            compaction_tail_tokens=4,
            compaction_summary_context_token_budget=20,
            compaction_summary_output_token_budget=8,
            template_tokens=1,
            recall_budget=1,
            output_headroom=1,
        )
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="user", content="keep this request"),
                DialogueMessage(role="assistant", content=" ".join(f"answer{i}" for i in range(40))),
            ],
        )

        await builder.build(inp, _ctx(), config=config)

        assert summary_client.calls
        assert summary_client.calls[0]["max_tokens"] == 8
        assert len(extractor.envelopes) == 1
        envelope = extractor.envelopes[0]
        assert envelope.boundary == "compacted"
        assert len(envelope.current_context_messages) == 1
        assert '"general_summary": "middle summary"' in envelope.current_context_messages[0].text
        assert envelope.extractable_messages[0].text.startswith("keep this request")
        assert envelope.extractable_messages[-1].text.endswith("answer39")

    @pytest.mark.asyncio
    async def test_compacted_orphan_turn_preserves_conservative_boundary(self) -> None:
        summary_client = _SummaryLlmClient()
        extractor = _EnvelopeCaptureExtractor()
        builder = _make_builder(llm_client=summary_client, memory_extractor=extractor)
        config = VanillaAddConfig(
            chunk_soft_token_budget=30,
            chunk_hard_token_budget=40,
            turn_hard_token_budget=10,
            history_soft_token_budget=1,
            history_hard_token_budget=2,
            compaction_head_tokens=4,
            compaction_tail_tokens=4,
            compaction_summary_context_token_budget=20,
            compaction_summary_output_token_budget=8,
            template_tokens=1,
            recall_budget=1,
            output_headroom=1,
        )
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="assistant", content=" ".join(f"answer{i}" for i in range(40))),
            ],
        )

        await builder.build(inp, _ctx(), config=config)

        assert len(extractor.envelopes) == 1
        assert extractor.envelopes[0].boundary == "orphan"


class TestEmptyInput:
    """Empty input should produce no candidates."""

    @pytest.mark.asyncio
    async def test_no_messages(self) -> None:
        builder = _make_builder()
        inp = AddPipelineInput(messages=[])
        plan, events, updates = await builder.build(inp, _ctx())
        assert len(events) == 0
        assert len(plan.memories) == 0


class TestHistoryPackingIntegration:
    """Verify history flows between chunks."""

    @pytest.mark.asyncio
    async def test_two_chunks_have_history_flow(self) -> None:
        """Two chunks: second should have history from first."""
        config = _small_budget_config(
            chunk_soft_token_budget=1500,
            chunk_hard_token_budget=3000,
            turn_hard_token_budget=5000,
        )
        builder = _make_builder()

        # Create enough messages to split into 2+ chunks
        msgs = []
        for i in range(6):
            msgs.append(DialogueMessage(role="user", content=" ".join([f"question{i}_word{j}" for j in range(80)])))
            msgs.append(DialogueMessage(role="assistant", content=" ".join([f"answer{i}_word{j}" for j in range(100)])))

        inp = AddPipelineInput(messages=msgs)
        plan, events, updates = await builder.build(inp, _ctx(), config=config)
        assert len(events) >= 2


class _ProvenanceExtractor(VanillaMemoryExtractor):
    """Extractor returning explicit per-message source provenance."""

    def __init__(self, candidates, sources, entities=None) -> None:
        super().__init__(llm_client=None)  # type: ignore[arg-type]
        self._candidates = candidates
        self._sources = sources
        self._entities = entities or []

    async def extract_from_envelope(self, envelope, preprocessed_texts, context):
        from mindmemos.components.extractor.vanilla import MemoryExtractionResult

        return MemoryExtractionResult(
            memories=self._candidates,
            entities=self._entities,
            sources=self._sources,
        )


class _CrossChunkDuplicateExtractor(VanillaMemoryExtractor):
    """Extractor returning the same candidate content from each chunk."""

    def __init__(self) -> None:
        super().__init__(llm_client=None)  # type: ignore[arg-type]

    async def extract_from_envelope(self, envelope, preprocessed_texts, context):
        msg_ref = envelope.extractable_messages[0]
        return MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m0",
                    content="User prefers Python.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["s0"],
                    action_hint="add",
                    metadata={"content_hash": "shared-python-preference"},
                )
            ],
            sources=[
                ExtractedSourceCandidate(
                    ref_id="s0",
                    source_type="message",
                    message_index=msg_ref.message_index,
                )
            ],
        )


class _HeadTailExtractor(VanillaMemoryExtractor):
    """Extractor returning separate candidates for compacted head and tail evidence."""

    def __init__(self) -> None:
        super().__init__(llm_client=None)  # type: ignore[arg-type]
        self.extractable_texts: list[str] = []

    async def extract_from_envelope(self, envelope, preprocessed_texts, context):
        self.extractable_texts = [msg_ref.text for msg_ref in envelope.extractable_messages]
        return MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m_head",
                    content="Head fact.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["s_head"],
                    action_hint="add",
                ),
                ExtractedMemoryCandidate(
                    ref_id="m_tail",
                    content="Tail fact.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["s_tail"],
                    action_hint="add",
                ),
            ],
            sources=[
                ExtractedSourceCandidate(
                    ref_id="s_head",
                    source_type="message",
                    message_index=envelope.extractable_messages[0].message_index,
                    metadata={"evidence_index": 0},
                ),
                ExtractedSourceCandidate(
                    ref_id="s_tail",
                    source_type="message",
                    message_index=envelope.extractable_messages[-1].message_index,
                    metadata={"evidence_index": len(envelope.extractable_messages) - 1},
                ),
            ],
        )


class TestMultiMessageSourceProvenance:
    """Verify candidates are attributed to correct source messages."""

    @pytest.mark.asyncio
    async def test_multi_message_chunk_assigns_correct_source_per_candidate(self):
        """Two candidates in one chunk, each from a different message."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m0",
                    content="User likes Python.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["s0"],
                    action_hint="add",
                ),
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User dislikes Java.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["s1"],
                    action_hint="add",
                ),
            ],
            sources=[
                ExtractedSourceCandidate(ref_id="s0", source_type="message", message_index=0),
                ExtractedSourceCandidate(ref_id="s1", source_type="message", message_index=1),
            ],
        )
        builder = _make_builder()
        builder._memory_extractor = extractor

        config = _small_budget_config(chunk_soft_token_budget=5000, chunk_hard_token_budget=8000)
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="user", content="I like Python."),
                DialogueMessage(role="assistant", content="I dislike Java."),
            ],
        )
        plan, events, _ = await builder.build(inp, _ctx(), config=config)

        assert len(plan.memories) == 2
        source_id_to_content = {m.metadata["source_id"]: m.content for m in plan.memories}
        assert len(source_id_to_content) == 2, "Each memory should have a distinct source_id"

        # Every EXTRACTED_FROM edge target must exist in plan.sources
        source_ids = {s.source_id for s in plan.sources}
        extracted_from = [r for r in plan.relationships if r.rel_type == "EXTRACTED_FROM"]
        assert len(extracted_from) == 2
        for rel in extracted_from:
            assert rel.target.node_id in source_ids

    @pytest.mark.asyncio
    async def test_candidate_source_refs_fall_back_to_evidence_index_when_sources_missing(self):
        """LLM source_refs like s1 should still bind when sources[] is missing."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="Jon offers one-on-one mentoring and training.",
                    mem_type="fact",
                    confidence=0.8,
                    source_refs=["s1"],
                    action_hint="add",
                ),
            ],
            sources=[],
        )
        builder = _make_builder(memory_extractor=extractor)

        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="Gina", content="What support are you offering dancers?"),
                DialogueMessage(
                    role="Jon",
                    content="Besides the dance classes and workshops, I'm offering one-on-one mentoring and training.",
                ),
            ],
        )

        plan, _, _ = await builder.build(inp, _ctx(), config=_small_budget_config())

        assert len(plan.memories) == 1
        assert plan.memories[0].content == "Jon offers one-on-one mentoring and training."
        assert plan.memories[0].metadata["source_message_index"] == 1

    @pytest.mark.asyncio
    async def test_explicit_empty_llm_entities_do_not_fallback_to_preprocessed_terms(self):
        """LLM-provided empty entities should not explode into rule-based term entities."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="用户计划暑假去广州旅游,选择七天酒店。",
                    mem_type="profile",
                    confidence=0.9,
                    entities=[],
                    source_refs=["s0"],
                    action_hint="add",
                ),
            ],
            sources=[],
        )
        vectorizer = RecordingVectorizer()
        builder = AddCoreBuilder(
            text_preprocessor=_real_text_preprocessor(),
            memory_extractor=extractor,  # type: ignore[arg-type]
            candidate_deduplicator=CandidateDeduplicator(),
            related_memory_recall=MockRecall(),  # type: ignore[arg-type]
            safety_gate=AddSafetyGate(),
            vectorizer=vectorizer,  # type: ignore[arg-type]
        )

        plan, _, _ = await builder.build(
            AddPipelineInput(messages=[DialogueMessage(role="user", content="我暑假去广州旅游,选择七天酒店。")]),
            _ctx(),
            config=_small_budget_config(),
        )

        assert len(plan.memories) == 1
        assert plan.entities == []
        assert vectorizer.entity_batches == []

    @pytest.mark.asyncio
    async def test_source_message_index_uses_original_input_index_after_filtering(self):
        """Extractor source indices remain original message indices when file/URL inputs are filtered."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m0",
                    content="Dialogue fact.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["s0"],
                    action_hint="add",
                ),
            ],
            sources=[
                ExtractedSourceCandidate(ref_id="s0", source_type="message", message_index=1),
            ],
        )
        builder = _make_builder(memory_extractor=extractor)

        inp = AddPipelineInput(
            messages=[
                UrlMessage(url="https://example.test/source"),
                DialogueMessage(role="user", content="Dialogue fact."),
            ],
        )

        plan, _, _ = await builder.build(inp, _ctx(), config=_small_budget_config())

        assert len(plan.memories) == 1
        assert plan.memories[0].metadata["source_message_index"] == 1

    @pytest.mark.asyncio
    async def test_multi_message_chunk_creates_source_writes_for_all_messages(self):
        """Both messages in a chunk should get SourceWrite entries."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m0",
                    content="First fact.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["s0"],
                    action_hint="add",
                ),
            ],
            sources=[
                ExtractedSourceCandidate(ref_id="s0", source_type="message", message_index=0),
            ],
        )
        builder = _make_builder()
        builder._memory_extractor = extractor

        config = _small_budget_config(chunk_soft_token_budget=5000, chunk_hard_token_budget=8000)
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="user", content="First message."),
                DialogueMessage(role="assistant", content="Second message."),
            ],
        )
        plan, _, _ = await builder.build(inp, _ctx(), config=config)

        # Both messages should have SourceWrites, even though only one candidate was extracted
        assert len(plan.sources) == 2, f"Expected 2 source writes, got {len(plan.sources)}"

    @pytest.mark.asyncio
    async def test_candidate_with_multiple_source_refs_gets_multiple_edges(self):
        """A candidate referencing two messages gets EXTRACTED_FROM edges to both."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m0",
                    content="Combined fact from both messages.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["s0", "s1"],
                    action_hint="add",
                ),
            ],
            sources=[
                ExtractedSourceCandidate(ref_id="s0", source_type="message", message_index=0),
                ExtractedSourceCandidate(ref_id="s1", source_type="message", message_index=1),
            ],
        )
        builder = _make_builder()
        builder._memory_extractor = extractor

        config = _small_budget_config(chunk_soft_token_budget=5000, chunk_hard_token_budget=8000)
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="user", content="User said something."),
                DialogueMessage(role="assistant", content="Assistant responded."),
            ],
        )
        plan, events, _ = await builder.build(inp, _ctx(), config=config)

        assert len(plan.memories) == 1
        extracted_from = [r for r in plan.relationships if r.rel_type == "EXTRACTED_FROM"]
        assert len(extracted_from) == 2, (
            f"Expected 2 EXTRACTED_FROM edges for multi-source candidate, got {len(extracted_from)}"
        )
        target_ids = {r.target.node_id for r in extracted_from}
        assert len(target_ids) == 2

    @pytest.mark.asyncio
    async def test_prompt_schema_source_refs_create_multiple_edges_without_top_level_sources(self):
        """Prompt schema source_refs should bind directly without sources[]."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m0",
                    content="Combined fact from both messages.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["s0", "s1"],
                    action_hint="add",
                ),
            ],
            sources=[],
        )
        builder = _make_builder(memory_extractor=extractor)

        config = _small_budget_config(chunk_soft_token_budget=5000, chunk_hard_token_budget=8000)
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="user", content="First evidence."),
                DialogueMessage(role="assistant", content="Second evidence."),
            ],
        )

        plan, _, _ = await builder.build(inp, _ctx(), config=config)

        source_by_id = {source.source_id: source for source in plan.sources}
        extracted_from = [rel for rel in plan.relationships if rel.rel_type == REL_EXTRACTED_FROM]
        assert len(extracted_from) == 2
        assert {source_by_id[rel.target.node_id].metadata["evidence_index"] for rel in extracted_from} == {0, 1}

    @pytest.mark.asyncio
    async def test_prompt_schema_entity_source_refs_create_mentioned_in_source_edges(self):
        """Entity metadata.source_refs should bind to evidence sources without sources[]."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m0",
                    content="OpenAI and Qdrant are both mentioned.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["s0", "s1"],
                    entities=["e1"],
                    action_hint="add",
                ),
            ],
            sources=[],
            entities=[
                ExtractedEntityCandidate(
                    ref_id="e1",
                    entity_name="Qdrant",
                    entity_type="tool",
                    confidence=0.95,
                    metadata={"source_refs": ["s1"]},
                )
            ],
        )
        builder = _make_builder(memory_extractor=extractor)

        config = _small_budget_config(chunk_soft_token_budget=5000, chunk_hard_token_budget=8000)
        config.enable_entities = True
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="user", content="OpenAI is mentioned here."),
                DialogueMessage(role="assistant", content="Qdrant is mentioned here."),
            ],
        )

        plan, _, _ = await builder.build(inp, _ctx(), config=config)

        source_by_id = {source.source_id: source for source in plan.sources}
        mentioned_in_source = [rel for rel in plan.relationships if rel.rel_type == REL_MENTIONED_IN_SOURCE]
        assert len(mentioned_in_source) == 1
        assert source_by_id[mentioned_in_source[0].target.node_id].metadata["evidence_index"] == 1

    @pytest.mark.asyncio
    async def test_deduped_candidate_across_chunks_keeps_all_source_edges(self):
        """Cross-chunk dedup preserves provenance from every contributing chunk."""
        builder = _make_builder()
        builder._memory_extractor = _CrossChunkDuplicateExtractor()

        config = _small_budget_config(
            chunk_soft_token_budget=300,
            chunk_hard_token_budget=600,
            turn_hard_token_budget=5000,
        )
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="user", content="User prefers Python. " + "alpha " * 200),
                DialogueMessage(role="assistant", content="Noted. " + "beta " * 200),
                DialogueMessage(role="user", content="User prefers Python. " + "gamma " * 200),
                DialogueMessage(role="assistant", content="Still true. " + "delta " * 200),
            ],
        )

        plan, events, _ = await builder.build(inp, _ctx(), config=config)

        assert len(plan.memories) == 1
        assert len(plan.sources) >= 2
        extracted_from = [rel for rel in plan.relationships if rel.rel_type == "EXTRACTED_FROM"]
        target_ids = {rel.target.node_id for rel in extracted_from}
        assert len(target_ids) >= 2

    @pytest.mark.asyncio
    async def test_empty_source_refs_single_message_autobinds(self):
        """Candidate with empty source_refs in a single-message chunk auto-binds."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m0",
                    content="Auto-bound fact.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=[],
                    action_hint="add",
                ),
            ],
            sources=[],
        )
        builder = _make_builder()
        builder._memory_extractor = extractor

        inp = AddPipelineInput(
            messages=[DialogueMessage(role="user", content="Single message.")],
        )
        plan, events, _ = await builder.build(inp, _ctx())

        assert len(plan.memories) == 1
        assert plan.memories[0].content == "Auto-bound fact."

    @pytest.mark.asyncio
    async def test_unresolvable_source_refs_single_message_autobinds(self):
        """Candidate with bad source_refs in a single-message chunk falls back to the only message."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m0",
                    content="Fallback-bound fact.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=["not_a_numeric_source_ref"],
                    action_hint="add",
                ),
            ],
            sources=[],
        )
        builder = _make_builder()
        builder._memory_extractor = extractor

        inp = AddPipelineInput(
            messages=[DialogueMessage(role="user", content="Single message.")],
        )
        plan, events, _ = await builder.build(inp, _ctx())

        assert len(plan.memories) == 1
        assert plan.memories[0].content == "Fallback-bound fact."
        assert plan.memories[0].metadata["source_message_index"] == 0

    @pytest.mark.asyncio
    async def test_empty_source_refs_multi_message_skipped(self):
        """Candidate with empty source_refs in a multi-message chunk is skipped."""
        extractor = _ProvenanceExtractor(
            candidates=[
                ExtractedMemoryCandidate(
                    ref_id="m0",
                    content="Ambiguous fact.",
                    mem_type="fact",
                    confidence=0.9,
                    source_refs=[],
                    action_hint="add",
                ),
            ],
            sources=[],
        )
        builder = _make_builder()
        builder._memory_extractor = extractor

        config = _small_budget_config(chunk_soft_token_budget=5000, chunk_hard_token_budget=8000)
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(role="user", content="First message."),
                DialogueMessage(role="assistant", content="Second message."),
            ],
        )
        plan, events, _ = await builder.build(inp, _ctx(), config=config)

        assert len(plan.memories) == 0, "Ambiguous candidate should be skipped"

    @pytest.mark.asyncio
    async def test_compacted_head_tail_from_same_message_keep_distinct_sources(self):
        """Head and tail slices from one original message should not overwrite provenance."""
        extractor = _HeadTailExtractor()
        builder = _make_builder(llm_client=_SummaryLlmClient(), memory_extractor=extractor)

        config = VanillaAddConfig(
            chunk_soft_token_budget=30,
            chunk_hard_token_budget=40,
            turn_hard_token_budget=10,
            history_soft_token_budget=1,
            history_hard_token_budget=2,
            compaction_head_tokens=4,
            compaction_tail_tokens=4,
            compaction_summary_context_token_budget=20,
            compaction_summary_output_token_budget=8,
            template_tokens=1,
            recall_budget=1,
            output_headroom=1,
        )
        inp = AddPipelineInput(
            messages=[
                DialogueMessage(
                    role="assistant",
                    content=" ".join(
                        [
                            *(f"head{i}" for i in range(6)),
                            *(f"mid{i}" for i in range(8)),
                            *(f"tail{i}" for i in range(6)),
                        ]
                    ),
                )
            ],
        )

        plan, _, _ = await builder.build(inp, _ctx(), config=config)

        assert len(plan.memories) == 2
        source_by_memory = {memory.content: memory.metadata["source_id"] for memory in plan.memories}
        assert source_by_memory["Head fact."] != source_by_memory["Tail fact."]
        source_text_by_id = {source.source_id: source.file_path for source in plan.sources}
        assert source_text_by_id[source_by_memory["Head fact."]] != source_text_by_id[source_by_memory["Tail fact."]]
