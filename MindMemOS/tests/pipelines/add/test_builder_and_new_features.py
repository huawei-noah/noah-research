"""Tests for AddCoreBuilder, LLM wiring, dense vectors, and integration scenarios."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.components.extractor import ExtractedEntityCandidate, ExtractedMemoryCandidate, MemoryExtractionResult
from mindmemos.components.extractor.vanilla import AddSafetyGate, CandidateDeduplicator
from mindmemos.components.extractor.vanilla.add_builder import AddCoreBuilder
from mindmemos.components.extractor.vanilla.add_recall import RelatedMemoryRecall
from mindmemos.components.extractor.vanilla.memory import VanillaMemoryExtractor
from mindmemos.components.text import SparseVectorEncoder, TextPreprocessor
from mindmemos.components.text.vectorizer import MemoryVectorizer
from mindmemos.config import TextProcessingConfig, VanillaAddConfig
from mindmemos.pipelines.add.vanilla import VanillaAddPipeline
from mindmemos.typing.llm import EmbeddingResponse
from mindmemos.typing.memory import EntityVectorWrite, MemoryRequestContext, MemoryView, VectorWrite
from mindmemos.typing.memory_db import MemoryDbMutationResult, MemoryDbWritePlan, MemoryDbWriteResult
from mindmemos.typing.service import AddPipelineInput

TEXT_CONFIG = TextProcessingConfig(
    bm25_use_spacy_lemma=False,
    spacy_en_model="missing_en_model",
    spacy_zh_model="missing_zh_model",
    sparse_hash_dim=128,
)


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeWriter:
    def __init__(self) -> None:
        self.calls: list = []
        self.update_calls: list = []
        self.mutation_plans: list = []

    async def apply_mutation_plan(self, context, plan, *, consistency="fast"):
        self.mutation_plans.append(SimpleNamespace(context=context, plan=plan, consistency=consistency))
        write_plan = plan.to_write_plan()
        if plan.has_writes():
            self.calls.append(SimpleNamespace(context=context, plan=write_plan, consistency=consistency))
        mutations = []
        for command in plan.memory_updates:
            self.update_calls.append(SimpleNamespace(context=context, command=command))
            mutations.append(MemoryDbMutationResult(memory_id=command.memory_id, changed=True))
        return MemoryDbWriteResult(
            memory_ids=[m.memory_id for m in write_plan.memories],
            entity_ids=[e.entity_id for e in write_plan.entities],
            mutations=mutations,
        )

    async def write(self, context, plan, *, consistency="fast"):
        self.calls.append(SimpleNamespace(context=context, plan=plan, consistency=consistency))
        return MemoryDbWriteResult(
            memory_ids=[m.memory_id for m in plan.memories],
            entity_ids=[e.entity_id for e in plan.entities],
        )

    async def update_memory(self, context, command):
        self.update_calls.append(SimpleNamespace(context=context, command=command))
        return SimpleNamespace(status="ok", memory_id=command.memory_id, changed=True)


class FakeReader:
    def __init__(self) -> None:
        self.listed_memories: list = []
        self.sparse_hits: list = []

    async def list_memories(self, context, *, filters=None, limit=50, cursor=None):
        return self.listed_memories, None

    async def search_sparse(self, context, req, *, indices, values):
        from mindmemos.typing.memory_db import MemoryDbSearchResult

        return MemoryDbSearchResult(query=req.query, hits=self.sparse_hits, total=len(self.sparse_hits))


class FakeEmbedClient:
    """Fake embed client that returns EmbeddingResponse like the real one."""

    def __init__(self, dim: int = 8, fail: bool = False) -> None:
        self._dim = dim
        self._fail = fail
        self.calls: list[str | list[str]] = []
        self.task_calls: list[tuple[str, str | list[str]]] = []

    async def embed(self, task: str, text: str | list[str], **kwargs) -> EmbeddingResponse:
        self.calls.append(text)
        self.task_calls.append((task, text))
        if self._fail:
            raise RuntimeError("embedding service unavailable")
        texts = text if isinstance(text, list) else [text]
        return EmbeddingResponse(embeddings=[[float(index + 1)] * self._dim for index, _ in enumerate(texts)])


class FakeLLMClient:
    """Fake LLM client that returns structured extraction JSON."""

    def __init__(self, result: MemoryExtractionResult | None = None, fail: bool = False) -> None:
        self._result = result
        self._fail = fail
        self.calls: list = []

    async def chat(self, *, task: str, messages, format_parser=None):
        self.calls.append(SimpleNamespace(task=task, messages=messages))
        if self._fail:
            raise RuntimeError("LLM unavailable")
        if self._result is None:
            parsed = {
                "memories": [{"ref_id": "m1", "content": "test", "mem_type": "fact"}],
                "entities": [],
                "sources": [],
                "property_bindings": [],
            }
        else:
            parsed = self._result.model_dump(exclude_unset=True)
        return SimpleNamespace(parsed=parsed)


def make_builder(
    *,
    embed_client=None,
    memory_extractor=None,
    reader=None,
) -> AddCoreBuilder:
    text_preprocessor = TextPreprocessor(TEXT_CONFIG)
    sparse_encoder = SparseVectorEncoder(TEXT_CONFIG)
    extractor = memory_extractor or VanillaMemoryExtractor()
    recall = RelatedMemoryRecall(
        db_reader=reader or FakeReader(),
        sparse_encoder=sparse_encoder,
    )
    vectorizer = MemoryVectorizer(
        sparse_encoder=sparse_encoder,
        embed_client=embed_client,
        text_preprocessor=text_preprocessor,
    )
    return AddCoreBuilder(
        text_preprocessor=text_preprocessor,
        memory_extractor=extractor,
        candidate_deduplicator=CandidateDeduplicator(),
        related_memory_recall=recall,
        safety_gate=AddSafetyGate(),
        vectorizer=vectorizer,
    )


class _ConfigCaptureBuilder:
    def __init__(self) -> None:
        self.config = None

    async def build(self, inp, context, *, consistency, config=None):
        self.config = config
        return MemoryDbWritePlan(), [], []


@pytest.mark.asyncio
async def test_vanilla_pipeline_passes_configured_chunk_budgets_to_builder(monkeypatch) -> None:
    configured = VanillaAddConfig(turn_hard_token_budget=12345)
    monkeypatch.setattr(
        "mindmemos.pipelines.add.vanilla.vanilla_add.get_config",
        lambda: SimpleNamespace(algo_config=SimpleNamespace(add=SimpleNamespace(vanilla=configured))),
    )
    pipeline = VanillaAddPipeline(
        db_reader=FakeReader(),
        db_writer=FakeWriter(),
        text_config=TEXT_CONFIG,
        consistency="fast",
        llm_client=None,
        embed_client=None,
    )
    capture = _ConfigCaptureBuilder()
    pipeline._builder = capture

    await pipeline.add_sync(AddPipelineInput(messages=[]), make_context())

    assert capture.config is configured


# Task 4.6: LLM extraction integration tests


class TestLLMExtractionWiring:
    @pytest.mark.asyncio
    async def test_llm_extraction_with_mock_client(self):
        """Pipeline uses LLM path when llm_client is provided."""
        extractor_result = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User prefers Python for backend.",
                    mem_type="profile",
                    confidence=0.9,
                    action_hint="add",
                    reason="explicit preference",
                )
            ]
        )
        llm = FakeLLMClient(result=extractor_result)
        extractor = VanillaMemoryExtractor(llm_client=llm)
        writer = FakeWriter()
        pipeline = VanillaAddPipeline(
            db_reader=FakeReader(),
            db_writer=writer,
            memory_extractor=extractor,
            text_config=TEXT_CONFIG,
            consistency="fast",
            llm_client=None,
            embed_client=None,
        )

        result = await pipeline.add_sync(
            AddPipelineInput(messages=[{"text": "I prefer Python."}]),
            make_context(),
        )

        assert result.status == "ok"
        assert llm.calls
        plan = writer.calls[0].plan
        assert plan.memories[0].content == "User prefers Python for backend."
        assert plan.memories[0].mem_type == "profile"

    @pytest.mark.asyncio
    async def test_fallback_on_llm_exception(self):
        """Pipeline falls back to deterministic extraction when LLM fails."""
        llm = FakeLLMClient(fail=True)
        extractor = VanillaMemoryExtractor(llm_client=llm)
        writer = FakeWriter()
        pipeline = VanillaAddPipeline(
            db_reader=FakeReader(),
            db_writer=writer,
            memory_extractor=extractor,
            text_config=TEXT_CONFIG,
            consistency="fast",
            llm_client=None,
            embed_client=None,
        )

        result = await pipeline.add_sync(
            AddPipelineInput(messages=[{"text": "Hello world."}]),
            make_context(),
        )

        assert result.status == "ok"
        assert llm.calls
        plan = writer.calls[0].plan
        assert plan.memories[0].content == "Hello world."


# Task 5.6: Dense vector tests


class TestDenseVectors:
    @pytest.mark.asyncio
    async def test_dense_vector_attached_to_write(self):
        """Dense vector is generated and attached when embed_client is provided."""
        embed = FakeEmbedClient(dim=8)
        builder = make_builder(embed_client=embed)
        ctx = make_context()
        inp = AddPipelineInput(messages=[{"text": "Hello dense world."}])

        plan, events, _ = await builder.build(inp, ctx, consistency="fast")

        assert len(plan.vectors) == 1
        assert plan.vectors[0].semantic_vector is not None
        assert len(plan.vectors[0].semantic_vector) == 8
        assert plan.vectors[0].bm25_indices
        assert embed.calls

    @pytest.mark.asyncio
    async def test_add_candidates_use_one_batch_embedding_call(self):
        """Multiple ADD candidates are embedded in one batch request."""
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User likes Python.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="add",
                ),
                ExtractedMemoryCandidate(
                    ref_id="m2",
                    content="User likes Rust.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="add",
                ),
                ExtractedMemoryCandidate(
                    ref_id="m3",
                    content="User likes Go.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="add",
                ),
            ]
        )
        embed = FakeEmbedClient(dim=4)
        builder = make_builder(
            embed_client=embed,
            memory_extractor=FakeExtractor(extraction),
        )

        plan, _, _ = await builder.build(
            AddPipelineInput(messages=[{"text": "I like Python, Rust, and Go."}]),
            make_context(),
            consistency="fast",
        )

        assert [text for task, text in embed.task_calls if task == "memory.add.embed"] == [
            ["User likes Python.", "User likes Rust.", "User likes Go."]
        ]
        assert len(plan.vectors) == 3
        assert [vector.semantic_vector for vector in plan.vectors] == [[1.0] * 4, [2.0] * 4, [3.0] * 4]

    @pytest.mark.asyncio
    async def test_no_embed_client_skips_dense_vector(self):
        """When embed_client is None, only sparse vector is generated."""
        builder = make_builder(embed_client=None)
        ctx = make_context()
        inp = AddPipelineInput(messages=[{"text": "Sparse only."}])

        plan, events, _ = await builder.build(inp, ctx, consistency="fast")

        assert len(plan.vectors) == 1
        assert plan.vectors[0].semantic_vector is None
        assert plan.vectors[0].bm25_indices

    @pytest.mark.asyncio
    async def test_embed_failure_fast_mode_marks_pending(self):
        """Embedding failure in fast mode sets vector_pending=True."""
        embed = FakeEmbedClient(fail=True)
        builder = make_builder(embed_client=embed)
        ctx = make_context()
        inp = AddPipelineInput(messages=[{"text": "Will fail embed."}])

        plan, events, _ = await builder.build(inp, ctx, consistency="fast")

        assert len(plan.vectors) == 1
        assert plan.vectors[0].semantic_vector is None
        assert plan.memories[0].metadata.get("vector_pending") is True

    @pytest.mark.asyncio
    async def test_embed_failure_strong_mode_raises(self):
        """Embedding failure in strong mode raises exception."""
        embed = FakeEmbedClient(fail=True)
        builder = make_builder(embed_client=embed)
        ctx = make_context()
        inp = AddPipelineInput(messages=[{"text": "Will fail embed."}])

        with pytest.raises(RuntimeError, match="embedding"):
            await builder.build(inp, ctx, consistency="strong")

    @pytest.mark.asyncio
    async def test_reinforce_skips_embedding(self):
        """REINFORCE actions don't generate embeddings."""
        embed = FakeEmbedClient(dim=8)
        reader = FakeReader()
        preprocessor = TextPreprocessor(TEXT_CONFIG)
        preprocessed = preprocessor.preprocess_text("Duplicate content.", segment_id="seg1")
        reader.listed_memories = [
            MemoryView(
                memory_id="mem-dup",
                project_id="proj-1",
                content="Duplicate content.",
                mem_type="fact",
                status="active",
                metadata={"content_hash": preprocessed.content_hash},
            )
        ]

        builder = make_builder(embed_client=embed, reader=reader)
        ctx = make_context()
        inp = AddPipelineInput(messages=[{"text": "Duplicate content."}])

        plan, events, reinforcement_commands = await builder.build(inp, ctx, consistency="fast")

        assert plan.memories == []
        assert plan.vectors == []
        assert len(reinforcement_commands) == 1
        assert embed.calls == []

    @pytest.mark.asyncio
    async def test_memory_and_entity_vectorization_start_concurrently(self):
        """Memory and entity embedding batches start without serially waiting for each other."""

        class BlockingVectorizer:
            def __init__(self) -> None:
                self.memory_started = asyncio.Event()
                self.entity_started = asyncio.Event()
                self.call_order: list[str] = []

            async def vectorize_many(self, items, consistency="fast"):
                self.call_order.append("memory_start")
                self.memory_started.set()
                await asyncio.wait_for(self.entity_started.wait(), timeout=0.2)
                return [
                    VectorWrite(memory_id=memory_id, semantic_vector=[1.0], bm25_indices=[1], bm25_values=[1.0])
                    for memory_id, _, _ in items
                ], [False] * len(items)

            async def vectorize_entities(self, entities, *, memories_by_entity=None, consistency="fast"):
                self.call_order.append("entity_start")
                self.entity_started.set()
                await asyncio.wait_for(self.memory_started.wait(), timeout=0.2)
                return [
                    EntityVectorWrite(
                        entity_id=entity.entity_id, semantic_vector=[2.0], bm25_indices=[2], bm25_values=[2.0]
                    )
                    for entity in entities
                ], False

        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="Kai uses Python.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="add",
                    entities=["e1"],
                )
            ],
            entities=[
                ExtractedEntityCandidate(
                    ref_id="e1",
                    entity_name="Python",
                    entity_type="technology",
                    confidence=0.95,
                )
            ],
        )
        vectorizer = BlockingVectorizer()
        text_preprocessor = TextPreprocessor(TEXT_CONFIG)
        builder = AddCoreBuilder(
            text_preprocessor=text_preprocessor,
            memory_extractor=FakeExtractor(extraction),
            candidate_deduplicator=CandidateDeduplicator(),
            related_memory_recall=RelatedMemoryRecall(
                db_reader=FakeReader(),
                sparse_encoder=SparseVectorEncoder(TEXT_CONFIG),
            ),
            safety_gate=AddSafetyGate(),
            vectorizer=vectorizer,  # type: ignore[arg-type]
        )

        plan, _, _ = await builder.build(
            AddPipelineInput(messages=[{"text": "Kai uses Python."}]),
            make_context(),
            config=VanillaAddConfig(enable_entities=True),
        )

        assert vectorizer.call_order == ["memory_start", "entity_start"]
        assert len(plan.vectors) == 1
        assert len(plan.entity_vectors) == 1


# Task 6.11: Phase method isolation tests


class TestSafetyGateActionHints:
    """Verify that the safety gate handles all action_hint values from the extractor."""

    @staticmethod
    def _preprocessed(text: str = "test content") -> "PreprocessedText":
        from mindmemos.typing.memory import PreprocessedText

        return PreprocessedText(
            text=text,
            normalized_text=text,
            content_hash="hash123",
            bm25_text=text,
        )

    def test_skip_hint(self):
        gate = AddSafetyGate()
        result = gate.gate_segment(self._preprocessed(), action_hint="skip")
        assert result.action == "SKIP"
        assert result.reason == "extractor_skip_hint"

    def test_update_hint_passes_through(self):
        """UPDATE: target_memory_id provided + high confidence → UPDATE."""
        gate = AddSafetyGate()
        result = gate.gate_segment(
            self._preprocessed(),
            action_hint="update",
            confidence=0.9,
            target_memory_id="mem-1",
        )
        assert result.action == "UPDATE"
        assert result.target_memory_id == "mem-1"

    def test_merge_hint_passes_through(self):
        """MERGE: >= 2 related_memory_ids + high confidence → MERGE."""
        gate = AddSafetyGate()
        result = gate.gate_segment(
            self._preprocessed(),
            action_hint="merge",
            confidence=0.9,
            related_memory_ids=["mem-1", "mem-2"],
        )
        assert result.action == "MERGE"
        assert len(result.related_memory_ids) == 2

    def test_reinforce_hint_adds_when_no_target(self):
        """reinforce hint with no target_memory_id → ADD (downgrade)."""
        gate = AddSafetyGate()
        result = gate.gate_segment(self._preprocessed(), action_hint="reinforce")
        assert result.action == "ADD"
        assert result.reason == "reinforce_no_target"

    def test_reinforce_hint_with_target(self):
        """reinforce hint + target_memory_id → REINFORCE."""
        gate = AddSafetyGate()
        result = gate.gate_segment(
            self._preprocessed(),
            action_hint="reinforce",
            target_memory_id="mem-dup",
        )
        assert result.action == "REINFORCE"
        assert result.target_memory_id == "mem-dup"

    def test_default_add_action(self):
        """No action_hint → default ADD."""
        gate = AddSafetyGate()
        result = gate.gate_segment(self._preprocessed())
        assert result.action == "ADD"
        assert result.reason == "extractor_add_hint"


# Task 6.12: Integration test — builder produces same output shape


class TestBuilderIntegration:
    @pytest.mark.asyncio
    async def test_build_produces_write_plan_with_all_fields(self):
        builder = make_builder()
        ctx = make_context()
        inp = AddPipelineInput(messages=[{"text": "Kai uses Qdrant for search."}], timestamp=1683504000000)

        plan, events, reinforcement_commands = await builder.build(inp, ctx, consistency="fast")

        assert plan.memories
        assert plan.vectors
        assert plan.relationships
        assert len(plan.memories) == len(plan.vectors)
        assert plan.memories[0].project_id == "proj-1"
        assert plan.memories[0].content == "Kai uses Qdrant for search."
        assert plan.memories[0].validate_from == datetime(2023, 5, 8, tzinfo=UTC)
        assert events[0].operation == "add"
        assert events[0].memory_id == plan.memories[0].memory_id
        assert reinforcement_commands == []

    @pytest.mark.asyncio
    async def test_chunked_build_produces_source_writes_for_messages(self):
        """Regression: message-type SourceWrite must appear in plan.sources.

        The chunked build path used to skip build_source_write() for
        message-type SourceRefs, leaving plan.sources empty while
        EXTRACTED_FROM edges still referenced those source IDs.
        Neo4j would silently skip the edge writes and Qdrant source
        records would be missing.
        """
        builder = make_builder()
        ctx = make_context()
        inp = AddPipelineInput(messages=[{"text": "Chunked source test."}])

        plan, events, _ = await builder.build(inp, ctx, consistency="fast")

        # plan.sources must not be empty for text input
        assert plan.sources, "plan.sources is empty — message SourceWrite missing"

        # Every EXTRACTED_FROM edge target must exist in plan.sources
        source_ids = {s.source_id for s in plan.sources}
        extracted_from_targets = [rel for rel in plan.relationships if rel.rel_type == "EXTRACTED_FROM"]
        assert extracted_from_targets, "No EXTRACTED_FROM edges found"
        for rel in extracted_from_targets:
            assert rel.target.node_id in source_ids, (
                f"EXTRACTED_FROM edge targets source_id={rel.target.node_id!r} not found in plan.sources ({source_ids})"
            )

        # message sources stay in plan.sources as Neo4j graph nodes but must be
        # flagged graph-only so they are not persisted to source_item_v1.
        assert all(s.persist_payload is False for s in plan.sources), "message sources must have persist_payload=False"

    @pytest.mark.asyncio
    async def test_multi_message_add_no_content_collision(self):
        """Two distinct messages must produce two distinct memories.

        Regression: fallback extraction used per-call counter ref_id="m1"
        for every segment, causing candidate_by_ref_id to overwrite the
        first candidate with the second.
        """
        builder = make_builder()
        ctx = make_context()
        inp = AddPipelineInput(
            messages=[
                {"text": "First memory."},
                {"text": "Second memory."},
            ]
        )

        plan, events, _ = await builder.build(inp, ctx, consistency="fast")

        contents = [m.content for m in plan.memories]
        assert len(contents) == 2, f"Expected 2 memories, got {len(contents)}: {contents}"
        assert "First memory." in contents
        assert "Second memory." in contents

    @pytest.mark.asyncio
    async def test_llm_multi_chunk_collision_no_content_crossover(self):
        """LLM returning same ref_id across chunks must not cause content crossover.

        Regression: when the LLM returns ref_id="m1" for every chunk,
        candidate_by_ref_id silently overwrote the first candidate with
        the second, producing two copies of the second chunk's content.
        The builder now prefixes ref_ids with chunk index.
        """

        class StatefulFakeLLM:
            """Returns different content on each call but always uses ref_id="m1"."""

            def __init__(self, contents: list[str], msg_indices: list[int]) -> None:
                self._contents = list(contents)
                self._msg_indices = list(msg_indices)
                self._call_idx = 0

            async def chat(self, *, task: str, messages, format_parser=None):
                content = self._contents[min(self._call_idx, len(self._contents) - 1)]
                msg_idx = self._msg_indices[min(self._call_idx, len(self._msg_indices) - 1)]
                self._call_idx += 1
                parsed = {
                    "memories": [{"ref_id": "m1", "content": content, "mem_type": "fact", "source_refs": ["s1"]}],
                    "entities": [],
                    "sources": [{"ref_id": "s1", "source_type": "message", "message_index": msg_idx}],
                    "property_bindings": [],
                }
                return SimpleNamespace(parsed=parsed)

        llm = StatefulFakeLLM(["First LLM memory.", "Second LLM memory."], msg_indices=[0, 2])
        extractor = VanillaMemoryExtractor(llm_client=llm)
        builder = make_builder(memory_extractor=extractor)
        ctx = make_context()
        # Use alternating roles + large messages to force 2 turns → 2 chunks
        from mindmemos.config import VanillaAddConfig

        config = VanillaAddConfig(
            chunk_soft_token_budget=300,
            chunk_hard_token_budget=600,
            turn_hard_token_budget=5000,
        )
        inp = AddPipelineInput(
            messages=[
                {"role": "user", "content": "First LLM memory. " + "word " * 200},
                {"role": "assistant", "content": "Response one. " + "word " * 200},
                {"role": "user", "content": "Second LLM memory. " + "word " * 200},
            ]
        )

        plan, events, _ = await builder.build(inp, ctx, consistency="fast", config=config)

        contents = [m.content for m in plan.memories]
        assert len(contents) == 2, f"Expected 2 memories, got {len(contents)}: {contents}"
        assert "First LLM memory." in contents
        assert "Second LLM memory." in contents

    @pytest.mark.asyncio
    async def test_builder_persists_dialogue_source_timestamp_on_memory(self):
        extractor_result = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User visited an LGBTQ support group yesterday.",
                    mem_type="episodic",
                    source_refs=["s1"],
                )
            ],
            sources=[
                {
                    "ref_id": "s1",
                    "source_type": "message",
                    "message_index": 0,
                }
            ],
        )
        builder = make_builder(memory_extractor=VanillaMemoryExtractor(llm_client=FakeLLMClient(extractor_result)))
        ctx = make_context()

        plan, _, _ = await builder.build(
            AddPipelineInput(
                messages=[
                    {
                        "role": "user",
                        "content": "I visited an LGBTQ support group yesterday.",
                        "timestamp": 1683504000000,
                    }
                ]
            ),
            ctx,
            consistency="fast",
        )

        assert len(plan.memories) == 1
        memory = plan.memories[0]
        assert memory.validate_from == datetime(2023, 5, 8, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_builder_persists_resolved_event_time_metadata(self):
        extractor_result = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="On 2023-05-07 (yesterday), the user visited an LGBTQ support group.",
                    mem_type="episodic",
                    source_refs=["s1"],
                    metadata={
                        "temporal_text": "yesterday",
                        "resolved_event_date": "2023-05-07",
                        "temporal_resolution_basis": "message_time: 2023-05-08 00:00:00",
                    },
                )
            ],
            sources=[
                {
                    "ref_id": "s1",
                    "source_type": "message",
                    "message_index": 0,
                }
            ],
        )
        builder = make_builder(memory_extractor=VanillaMemoryExtractor(llm_client=FakeLLMClient(extractor_result)))

        plan, _, _ = await builder.build(
            AddPipelineInput(
                messages=[
                    {
                        "role": "user",
                        "content": "I visited an LGBTQ support group yesterday.",
                        "timestamp": 1683504000000,
                    }
                ]
            ),
            make_context(),
            consistency="fast",
        )

        assert len(plan.memories) == 1
        memory = plan.memories[0]
        assert memory.metadata["temporal_text"] == "yesterday"
        assert memory.metadata["resolved_event_date"] == "2023-05-07"
        assert "temporal_resolution_basis" not in memory.metadata
        assert memory.metadata["source_timestamp_ms"] == 1683504000000


# Task 7.3: End-to-end with LLM + dedup + recall + dense + sparse


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_llm_and_dense_vectors(self):
        llm_result = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User prefers Python.",
                    mem_type="profile",
                    confidence=0.9,
                    entities=["e1"],
                    action_hint="add",
                    reason="explicit",
                ),
                ExtractedMemoryCandidate(
                    ref_id="m2",
                    content="User prefers Python.",
                    mem_type="profile",
                    confidence=0.7,
                    entities=["e1"],
                    action_hint="add",
                    reason="duplicate",
                ),
            ],
            entities=[
                ExtractedEntityCandidate(
                    ref_id="e1",
                    entity_name="Python",
                    entity_type="language",
                    confidence=0.95,
                )
            ],
        )
        llm = FakeLLMClient(result=llm_result)
        extractor = VanillaMemoryExtractor(llm_client=llm)
        embed = FakeEmbedClient(dim=4)
        writer = FakeWriter()

        pipeline = VanillaAddPipeline(
            db_reader=FakeReader(),
            db_writer=writer,
            memory_extractor=extractor,
            text_config=TEXT_CONFIG,
            consistency="fast",
            llm_client=None,
            embed_client=embed,
        )

        result = await pipeline.add_sync(
            AddPipelineInput(messages=[{"text": "I prefer Python."}]),
            make_context(),
        )

        assert result.status == "ok"
        plan = writer.calls[0].plan

        assert len(plan.memories) == 1
        assert plan.memories[0].content == "User prefers Python."
        assert plan.memories[0].mem_type == "profile"
        assert plan.memories[0].metadata["extractor_confidence"] == 0.9

        # Dense + sparse vectors
        assert len(plan.vectors) == 1
        assert plan.vectors[0].semantic_vector is not None
        assert plan.vectors[0].bm25_indices

        # Vanilla search does not consume entity writes, so the default add path mutes them.
        assert plan.entities == []
        assert plan.entity_vectors == []
        rel_types = {r.rel_type for r in plan.relationships}
        assert "MENTIONS" not in rel_types
        assert "EXTRACTED_FROM" in rel_types

    @pytest.mark.asyncio
    async def test_enable_entities_uses_local_entity_fallback(self):
        llm_result = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User prefers Python for backend services.",
                    mem_type="profile",
                    confidence=0.9,
                    action_hint="add",
                )
            ],
        )
        llm = FakeLLMClient(result=llm_result)
        extractor = VanillaMemoryExtractor(llm_client=llm)
        writer = FakeWriter()

        pipeline = VanillaAddPipeline(
            db_reader=FakeReader(),
            db_writer=writer,
            memory_extractor=extractor,
            text_config=TEXT_CONFIG,
            consistency="fast",
            llm_client=None,
            embed_client=FakeEmbedClient(dim=4),
        )

        result = await pipeline.add_sync(
            AddPipelineInput(messages=[{"text": "I prefer Python for backend services."}]),
            make_context(),
        )

        assert result.status == "ok"
        plan = writer.calls[0].plan
        assert plan.entities == []
        assert plan.memories[0].metadata["entities"] == []

    @pytest.mark.asyncio
    async def test_enable_entities_config_restores_entity_writes(self):
        llm_result = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User prefers Python for backend services.",
                    mem_type="profile",
                    confidence=0.9,
                    action_hint="add",
                )
            ],
        )
        llm = FakeLLMClient(result=llm_result)
        extractor = VanillaMemoryExtractor(llm_client=llm)
        writer = FakeWriter()

        pipeline = VanillaAddPipeline(
            db_reader=FakeReader(),
            db_writer=writer,
            memory_extractor=extractor,
            text_config=TEXT_CONFIG,
            consistency="fast",
            llm_client=None,
            embed_client=FakeEmbedClient(dim=4),
            vanilla_add_config=VanillaAddConfig(enable_entities=True),
        )

        result = await pipeline.add_sync(
            AddPipelineInput(messages=[{"text": "I prefer Python for backend services."}]),
            make_context(),
        )

        assert result.status == "ok"
        plan = writer.calls[0].plan
        assert [entity.entity_name for entity in plan.entities] == ["python"]
        assert plan.memories[0].metadata["entities"] == ["python"]

    @pytest.mark.asyncio
    async def test_project_isolation(self):
        """Task 7.5: All write DTOs carry project_id from context."""
        writer = FakeWriter()
        pipeline = VanillaAddPipeline(
            db_reader=FakeReader(),
            db_writer=writer,
            text_config=TEXT_CONFIG,
            consistency="fast",
            llm_client=None,
            embed_client=None,
        )

        ctx = MemoryRequestContext(
            request_id="req-iso",
            account_id="acc-iso",
            project_id="proj-isolated",
            api_key_uuid="key-iso",
            user_id="user-iso",
            session_id="sess-iso",
        )
        result = await pipeline.add_sync(
            AddPipelineInput(messages=[{"text": "Isolated memory."}]),
            ctx,
        )

        plan = writer.calls[0].plan
        assert plan.memories[0].project_id == "proj-isolated"
        assert plan.entities == []
        assert all(r.project_id == "proj-isolated" for r in plan.relationships)

    @pytest.mark.asyncio
    async def test_update_hint_returns_event_without_db_write(self):
        """UPDATE action produces an 'update' event; DB write deferred until Phase 4."""
        extractor_result = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User moved to Shanghai.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="update",
                    target_memory_id="mem-old",
                    reason="contradicts existing memory",
                )
            ]
        )
        llm = FakeLLMClient(result=extractor_result)
        extractor = VanillaMemoryExtractor(llm_client=llm)
        writer = FakeWriter()
        reader = FakeReader()
        from mindmemos.typing.memory_db import MemoryDbSearchHit

        existing = MemoryView(
            memory_id="mem-old",
            project_id="proj-1",
            content="User lives in Beijing.",
            mem_type="fact",
            status="active",
        )
        reader.listed_memories = [existing]
        reader.sparse_hits = [
            MemoryDbSearchHit(
                memory_id="mem-old",
                score=0.8,
                memory=existing,
                source="bm25",
                rank=1,
            )
        ]
        pipeline = VanillaAddPipeline(
            db_reader=reader,
            db_writer=writer,
            memory_extractor=extractor,
            text_config=TEXT_CONFIG,
            consistency="fast",
            llm_client=None,
            embed_client=None,
        )

        result = await pipeline.add_sync(
            AddPipelineInput(messages=[{"text": "I moved to Shanghai."}]),
            make_context(),
        )

        assert result.status == "ok"
        assert len(result.memories) == 1
        assert result.memories[0].operation == "update"
        assert result.memories[0].content == "User moved to Shanghai."

        # UPDATE produces no new memories/vectors, but the source record must
        # still be persisted so the EXTRACTED_FROM edge has a valid target.
        plan = writer.calls[0].plan
        assert plan.memories == []
        assert plan.vectors == []
        assert plan.sources, "SourceWrite must be persisted for UPDATE actions"


# MERGE branch, REINFORCE multi-source, unhandled action guard


class FakeExtractor:
    """Configurable extractor that returns a fixed result."""

    def __init__(self, result: MemoryExtractionResult) -> None:
        self._result = result

    async def extract_from_envelope(self, envelope, preprocessed_texts, context):
        return self._result


def _reader_with_active_memories(*memory_ids: str) -> FakeReader:
    from mindmemos.typing.memory_db import MemoryDbSearchHit

    reader = FakeReader()
    reader.listed_memories = [
        MemoryView(
            memory_id=memory_id,
            project_id="proj-1",
            content=f"Existing memory {memory_id}",
            mem_type="fact",
            status="active",
        )
        for memory_id in memory_ids
    ]
    reader.sparse_hits = [
        MemoryDbSearchHit(
            memory_id=memory.memory_id,
            score=0.8,
            memory=memory,
            source="bm25",
            rank=index,
        )
        for index, memory in enumerate(reader.listed_memories, start=1)
    ]
    return reader


class TestPlannerMemoryIdValidation:
    """Prevent extractor-local ref_ids from reaching DB update paths."""

    @pytest.mark.asyncio
    async def test_update_hint_with_local_ref_target_downgrades_to_add(self):
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User moved to Shanghai.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="update",
                    target_memory_id="m2",
                )
            ],
        )
        builder = make_builder(memory_extractor=FakeExtractor(extraction))

        plan, events, update_commands = await builder.build(
            AddPipelineInput(messages=[{"text": "I moved to Shanghai."}]),
            make_context(),
        )

        assert update_commands == []
        assert len(plan.memories) == 1
        assert events[0].operation == "add"
        assert plan.memories[0].memory_id != "m2"
        from uuid import UUID

        UUID(plan.memories[0].memory_id)

    @pytest.mark.asyncio
    async def test_merge_hint_with_local_ref_targets_downgrades_to_add(self):
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m3",
                    content="User likes action and sci-fi movies.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="merge",
                    related_memory_ids=["m1", "m2"],
                )
            ],
        )
        builder = make_builder(memory_extractor=FakeExtractor(extraction))

        plan, events, update_commands = await builder.build(
            AddPipelineInput(messages=[{"text": "I like action and sci-fi movies."}]),
            make_context(),
        )

        assert update_commands == []
        assert len(plan.memories) == 1
        assert events[0].operation == "add"
        assert plan.memories[0].metadata["related_memory_ids"] == []
        assert all(rel.rel_type != "RELATES_TO" or rel.target.node_id not in {"m1", "m2"} for rel in plan.relationships)

    @pytest.mark.asyncio
    async def test_reinforce_hint_with_local_ref_target_downgrades_to_add(self):
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User likes Counter-Strike.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="reinforce",
                    target_memory_id="m2",
                )
            ],
        )
        builder = make_builder(memory_extractor=FakeExtractor(extraction))

        plan, events, update_commands = await builder.build(
            AddPipelineInput(messages=[{"text": "I like Counter-Strike."}]),
            make_context(),
        )

        assert update_commands == []
        assert len(plan.memories) == 1
        assert events[0].operation == "add"
        assert plan.memories[0].memory_id != "m2"


class TestMergeBranch:
    """Cover builder.py lines 373-460: MERGE action path."""

    @pytest.mark.asyncio
    async def test_merge_creates_new_memory_archives_old(self):
        """MERGE: creates new memory, archives related memories, generates RELATES_TO edges."""
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="Merged: user likes Python and Rust.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="merge",
                    related_memory_ids=["mem-old-1", "mem-old-2"],
                    reason="similar_memories_merged",
                )
            ],
        )
        extractor = FakeExtractor(extraction)
        writer = FakeWriter()
        reader = _reader_with_active_memories("mem-old-1", "mem-old-2")
        pipeline = VanillaAddPipeline(
            db_reader=reader,
            db_writer=writer,
            memory_extractor=extractor,
            text_config=TEXT_CONFIG,
            consistency="fast",
            llm_client=None,
            embed_client=None,
        )

        request_time = 1683504000000
        before = datetime.now(UTC)
        result = await pipeline.add_sync(
            AddPipelineInput(messages=[{"text": "I like Python and Rust."}], timestamp=request_time),
            make_context(),
        )
        after = datetime.now(UTC)

        assert result.status == "ok"
        assert result.memories[0].operation == "merge"

        plan = writer.calls[0].plan
        assert len(plan.memories) == 1
        merged = plan.memories[0]
        assert "Python" in merged.content or "Rust" in merged.content
        assert merged.validate_from == datetime(2023, 5, 8, tzinfo=UTC)
        assert before <= merged.created_at <= after
        assert merged.root_id == ["mem-old-1", "mem-old-2"]
        assert merged.metadata["planner_action"] == "MERGE"
        assert merged.metadata["merged_from"] == ["mem-old-1", "mem-old-2"]

        relates_to = [rel for rel in plan.relationships if rel.rel_type == "RELATES_TO"]
        assert len(relates_to) == 2
        assert {rel.target.node_id for rel in relates_to} == {"mem-old-1", "mem-old-2"}

        archive_cmds = writer.update_calls
        assert len(archive_cmds) == 2
        assert {cmd.command.memory_id for cmd in archive_cmds} == {"mem-old-1", "mem-old-2"}
        assert all(cmd.command.status == "archived" for cmd in archive_cmds)

    @pytest.mark.asyncio
    async def test_merge_includes_vectors(self):
        """MERGE: new memory gets both sparse and dense vectors."""
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="Merged fact.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="merge",
                    related_memory_ids=["mem-a", "mem-b"],
                )
            ],
        )
        embed = FakeEmbedClient(dim=4)
        builder = make_builder(
            memory_extractor=FakeExtractor(extraction),
            embed_client=embed,
            reader=_reader_with_active_memories("mem-a", "mem-b"),
        )

        plan, events, _ = await builder.build(
            AddPipelineInput(messages=[{"text": "Hello merge."}]),
            make_context(),
        )

        assert events[0].operation == "merge"
        assert len(plan.vectors) == 1
        assert plan.vectors[0].semantic_vector is not None
        assert len(plan.vectors[0].semantic_vector) == 4

    @pytest.mark.asyncio
    async def test_merge_with_entities(self):
        """MERGE: entities from merged memory get MENTIONS edges."""
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="Kai uses Python.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="merge",
                    related_memory_ids=["mem-a", "mem-b"],
                    entities=["e1"],
                )
            ],
            entities=[
                ExtractedEntityCandidate(
                    ref_id="e1",
                    entity_name="Python",
                    entity_type="technology",
                    confidence=0.95,
                )
            ],
        )
        builder = make_builder(
            memory_extractor=FakeExtractor(extraction),
            reader=_reader_with_active_memories("mem-a", "mem-b"),
        )

        plan, events, _ = await builder.build(
            AddPipelineInput(messages=[{"text": "Kai uses Python."}]),
            make_context(),
            config=VanillaAddConfig(enable_entities=True),
        )

        assert events[0].operation == "merge"
        assert len(plan.entities) == 1
        assert plan.entities[0].entity_name == "Python"
        mentions = [
            rel
            for rel in plan.relationships
            if rel.rel_type == "MENTIONS" and rel.source.node_id == plan.memories[0].memory_id
        ]
        assert len(mentions) >= 1


class TestUpdateConsistency:
    """Verify UPDATE action recomputes vectors, refreshes metadata, and syncs Neo4j."""

    @pytest.mark.asyncio
    async def test_update_recomputes_dense_vector(self):
        """UPDATE: the update command carries a new dense vector for the updated content."""
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User moved to Shanghai.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="update",
                    target_memory_id="mem-old",
                    reason="contradicts existing memory",
                )
            ],
        )
        embed = FakeEmbedClient(dim=4)
        builder = make_builder(
            memory_extractor=FakeExtractor(extraction),
            embed_client=embed,
            reader=_reader_with_active_memories("mem-old"),
        )

        _, _, update_commands = await builder.build(
            AddPipelineInput(messages=[{"text": "I moved to Shanghai."}]),
            make_context(),
        )

        assert len(update_commands) == 1
        cmd = update_commands[0]
        assert cmd.memory_id == "mem-old"
        assert cmd.content == "User moved to Shanghai."
        assert cmd.dense_vector is not None
        assert len(cmd.dense_vector) == 4
        assert len(embed.calls) == 1

    @pytest.mark.asyncio
    async def test_update_refreshes_derived_metadata(self):
        """UPDATE: metadata_patch includes refreshed content_hash, bm25_text, tokens, lang."""
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User moved to Shanghai.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="update",
                    target_memory_id="mem-old",
                )
            ],
        )
        builder = make_builder(
            memory_extractor=FakeExtractor(extraction),
            reader=_reader_with_active_memories("mem-old"),
        )

        _, _, update_commands = await builder.build(
            AddPipelineInput(messages=[{"text": "I moved to Shanghai."}]),
            make_context(),
        )

        assert len(update_commands) == 1
        cmd = update_commands[0]
        patch = cmd.metadata_patch
        assert "content_hash" in patch
        assert "bm25_text" in patch
        assert "tokens" in patch
        assert "lang" in patch
        assert "last_updated_request_id" in patch
        assert "last_updated_at" in patch

    @pytest.mark.asyncio
    async def test_update_carries_sparse_vector(self):
        """UPDATE: the update command carries sparse BM25 vector data."""
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User moved to Shanghai.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="update",
                    target_memory_id="mem-old",
                )
            ],
        )
        builder = make_builder(
            memory_extractor=FakeExtractor(extraction),
            reader=_reader_with_active_memories("mem-old"),
        )

        _, _, update_commands = await builder.build(
            AddPipelineInput(messages=[{"text": "I moved to Shanghai."}]),
            make_context(),
        )

        cmd = update_commands[0]
        assert cmd.sparse_vectors is not None
        assert "bm25_indices" in cmd.sparse_vectors
        assert "bm25_values" in cmd.sparse_vectors

    @pytest.mark.asyncio
    async def test_update_signals_graph_content_sync(self):
        """UPDATE: graph_content_sync is True so Neo4j node content will be updated."""
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="User moved to Shanghai.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="update",
                    target_memory_id="mem-old",
                )
            ],
        )
        builder = make_builder(
            memory_extractor=FakeExtractor(extraction),
            reader=_reader_with_active_memories("mem-old"),
        )

        _, _, update_commands = await builder.build(
            AddPipelineInput(messages=[{"text": "I moved to Shanghai."}]),
            make_context(),
        )

        cmd = update_commands[0]
        assert cmd.graph_content_sync is True


class TestMergeArchiveConsistency:
    """Verify MERGE action archives old memories with proper DB sync signals."""

    @pytest.mark.asyncio
    async def test_merge_archive_commands_signal_archived_status(self):
        """MERGE: archive commands carry status='archived' which triggers Neo4j sync."""
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="Merged fact.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="merge",
                    related_memory_ids=["mem-a", "mem-b"],
                )
            ],
        )
        builder = make_builder(
            memory_extractor=FakeExtractor(extraction),
            reader=_reader_with_active_memories("mem-a", "mem-b"),
        )

        _, _, update_commands = await builder.build(
            AddPipelineInput(messages=[{"text": "Hello merge."}]),
            make_context(),
        )

        # Two archive commands, one per old memory
        assert len(update_commands) == 2
        for cmd in update_commands:
            assert cmd.status == "archived"
            assert cmd.memory_id in ("mem-a", "mem-b")
            assert cmd.reason == "add_merge_archive"

    @pytest.mark.asyncio
    async def test_merge_new_memory_has_complete_metadata(self):
        """MERGE: the new merged memory has content_hash, tokens, lang, entities metadata."""
        extraction = MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="Kai uses Python.",
                    mem_type="fact",
                    confidence=0.9,
                    action_hint="merge",
                    related_memory_ids=["mem-a", "mem-b"],
                    entities=["e1"],
                )
            ],
            entities=[
                ExtractedEntityCandidate(
                    ref_id="e1",
                    entity_name="Python",
                    entity_type="technology",
                    confidence=0.95,
                )
            ],
        )
        builder = make_builder(
            memory_extractor=FakeExtractor(extraction),
            reader=_reader_with_active_memories("mem-a", "mem-b"),
        )

        plan, events, _ = await builder.build(
            AddPipelineInput(messages=[{"text": "Kai uses Python."}]),
            make_context(),
        )

        assert len(plan.memories) == 1
        meta = plan.memories[0].metadata
        assert "content_hash" in meta
        assert "bm25_text" in meta
        assert "tokens" in meta
        assert "lang" in meta
        assert meta["planner_action"] == "MERGE"
        assert meta["merged_from"] == ["mem-a", "mem-b"]
        assert "entity_count" in meta
