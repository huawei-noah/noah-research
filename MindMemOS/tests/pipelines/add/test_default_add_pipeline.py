from datetime import UTC, datetime
from types import SimpleNamespace
from typing import get_type_hints

import pytest
from mindmemos.config import TextProcessingConfig, VanillaAddConfig
from mindmemos.config.algo.search import VanillaSearchConfig
from mindmemos.pipelines.add.base import AddPipeline
from mindmemos.pipelines.add.vanilla import VanillaAddPipeline
from mindmemos.pipelines.search.pipeline import SearchPipelineImpl
from mindmemos.typing.memory import (
    REL_EXTRACTED_FROM,
    REL_MENTIONED_IN_SOURCE,
    REL_MENTIONS,
    REL_RELATES_TO,
    MemoryRequestContext,
    MemoryView,
)
from mindmemos.typing.memory_db import (
    MemoryDbMutationResult,
    MemoryDbSearchHit,
    MemoryDbSearchResult,
    MemoryDbWriteResult,
)
from mindmemos.typing.service import AddPipelineAsyncResult, AddPipelineInput, SearchPipelineInput


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
        self.calls = []
        self.update_calls = []
        self.mutation_plans = []

    async def apply_mutation_plan(self, context: MemoryRequestContext, plan, *, consistency: str = "fast"):
        self.mutation_plans.append(SimpleNamespace(context=context, plan=plan, consistency=consistency))
        write_plan = plan.to_write_plan()
        if plan.has_writes():
            self.calls.append(SimpleNamespace(context=context, plan=write_plan, consistency=consistency))
        mutations = []
        for command in plan.memory_updates:
            self.update_calls.append(SimpleNamespace(context=context, command=command))
            mutations.append(MemoryDbMutationResult(memory_id=command.memory_id, changed=True))
        return MemoryDbWriteResult(
            memory_ids=[memory.memory_id for memory in write_plan.memories],
            entity_ids=[entity.entity_id for entity in write_plan.entities],
            mutations=mutations,
        )

    async def write(self, context: MemoryRequestContext, plan, *, consistency: str = "fast"):
        self.calls.append(SimpleNamespace(context=context, plan=plan, consistency=consistency))
        return MemoryDbWriteResult(
            memory_ids=[memory.memory_id for memory in plan.memories],
            entity_ids=[entity.entity_id for entity in plan.entities],
        )

    async def update_memory(self, context: MemoryRequestContext, command):
        self.update_calls.append(SimpleNamespace(context=context, command=command))
        return SimpleNamespace(status="ok", memory_id=command.memory_id, changed=True)


class FakeReader:
    def __init__(self) -> None:
        self.listed_memories = []
        self.sparse_hits = []

    async def list_memories(self, context: MemoryRequestContext, *, filters=None, limit=50, cursor=None):
        return self.listed_memories, None

    async def search_sparse(self, context: MemoryRequestContext, req, *, indices, values):
        from mindmemos.typing.memory_db import MemoryDbSearchResult

        return MemoryDbSearchResult(query=req.query, hits=self.sparse_hits, total=len(self.sparse_hits))


class FakeRecorder:
    def __init__(self) -> None:
        self.completed = []

    async def mark_add_completed(self, ctx, add_record_id, result) -> None:
        self.completed.append((add_record_id, result))


class FakeExtractor:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    async def extract_from_envelope(self, envelope, preprocessed_texts, context):
        self.calls.append(SimpleNamespace(envelope=envelope, preprocessed_texts=preprocessed_texts, context=context))
        return self.result


class InMemoryVanillaStore:
    def __init__(self) -> None:
        self.memories: dict[str, MemoryView] = {}
        self.entities = {}
        self.vectors = {}
        self.entity_vectors = {}
        self.sources = {}
        self.relationships = []
        self.mutation_plans = []

    async def apply_mutation_plan(self, context: MemoryRequestContext, plan, *, consistency: str = "fast"):
        self.mutation_plans.append(SimpleNamespace(context=context, plan=plan, consistency=consistency))
        write_plan = plan.to_write_plan()
        for memory in write_plan.memories:
            self.memories[memory.memory_id] = MemoryView(
                memory_id=memory.memory_id,
                project_id=memory.project_id,
                content=memory.content,
                mem_type=memory.mem_type,
                mem_extract_type=memory.mem_extract_type,
                mem_extract_version=memory.mem_extract_version,
                status=memory.status,
                metadata=dict(memory.metadata),
                account_id=memory.account_id,
                api_key_uuid=memory.api_key_uuid,
                user_id=memory.user_id,
                app_id=memory.app_id,
                session_id=memory.session_id,
                agent_id=memory.agent_id,
                request_id=memory.request_id,
                parent_ids=list(memory.parent_ids),
                root_id=list(memory.root_id),
                property_name=memory.property_name,
                entity_id=memory.entity_id,
                entity_type=memory.entity_type,
                validate_from=memory.validate_from,
                validate_to=memory.validate_to,
                created_at=memory.created_at,
                update_at=memory.update_at,
            )
        self.entities.update({entity.entity_id: entity for entity in write_plan.entities})
        self.vectors.update({vector.memory_id: vector for vector in write_plan.vectors})
        self.entity_vectors.update({vector.entity_id: vector for vector in write_plan.entity_vectors})
        self.sources.update({source.source_id: source for source in write_plan.sources})
        self.relationships.extend(write_plan.relationships)
        mutations = []
        for command in plan.memory_updates:
            memory = self.memories.get(command.memory_id)
            if memory is not None:
                metadata = {**dict(memory.metadata), **dict(command.metadata_patch)}
                content = command.content if command.content is not None else memory.content
                status = command.status or memory.status
                self.memories[command.memory_id] = memory.model_copy(
                    update={
                        "content": content,
                        "status": status,
                        "metadata": metadata,
                        "update_at": datetime.now(UTC),
                    }
                )
                mutations.append(MemoryDbMutationResult(memory_id=command.memory_id, changed=True))
            else:
                mutations.append(MemoryDbMutationResult(memory_id=command.memory_id, changed=False))
        return MemoryDbWriteResult(
            memory_ids=[memory.memory_id for memory in write_plan.memories],
            entity_ids=[entity.entity_id for entity in write_plan.entities],
            mutations=mutations,
        )

    async def list_memories(self, context: MemoryRequestContext, *, filters=None, limit=50, cursor=None):
        memories = [
            memory
            for memory in self.memories.values()
            if memory.project_id == context.project_id and memory.status == "active"
        ]
        return memories[:limit], None

    async def search_hybrid(self, context: MemoryRequestContext, req, *, dense_vector, sparse_vector):
        return MemoryDbSearchResult(query=req.query, hits=self._matching_hits(context, req.query, req.top_k), total=1)

    async def search_sparse(self, context: MemoryRequestContext, req, *, indices, values):
        return MemoryDbSearchResult(query=req.query, hits=self._matching_hits(context, req.query, req.top_k), total=1)

    async def get_related_memory_ids(
        self,
        context: MemoryRequestContext,
        memory_ids: list[str],
        *,
        limit_per_memory: int,
        max_candidates: int,
    ):
        return []

    async def get_memories(self, context: MemoryRequestContext, memory_ids: list[str]):
        return [self.memories[memory_id] for memory_id in memory_ids if memory_id in self.memories]

    def _matching_hits(self, context: MemoryRequestContext, query: str, limit: int):
        terms = [term.casefold() for term in query.split() if term.strip()]
        hits = []
        for memory in self.memories.values():
            if memory.project_id != context.project_id or memory.status != "active":
                continue
            content = memory.content.casefold()
            overlap = sum(1 for term in terms if term in content)
            if overlap <= 0:
                continue
            hits.append(
                MemoryDbSearchHit(
                    memory_id=memory.memory_id,
                    score=float(overlap),
                    memory=memory,
                    source="in_memory",
                    rank=len(hits) + 1,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.rank or 0))
        return hits[:limit]


def make_pipeline(writer: FakeWriter, *, recorder=None) -> VanillaAddPipeline:
    return make_pipeline_with_reader(FakeReader(), writer, recorder=recorder)


def make_pipeline_with_reader(
    reader: FakeReader,
    writer: FakeWriter,
    *,
    vanilla_add_config: VanillaAddConfig | None = None,
    recorder=None,
) -> VanillaAddPipeline:
    return VanillaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        text_config=TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        ),
        consistency="fast",
        llm_client=None,
        embed_client=None,
        vanilla_add_config=vanilla_add_config,
        recorder=recorder,
    )


def test_add_pipeline_protocol_async_returns_queued_result() -> None:
    return_type = get_type_hints(AddPipeline.add_async)["return"]

    assert return_type is AddPipelineAsyncResult


def _fake_search_config():
    return SimpleNamespace(
        algo_config=SimpleNamespace(
            text_processing=TextProcessingConfig(
                bm25_use_spacy_lemma=False,
                spacy_en_model="missing_en_model",
                spacy_zh_model="missing_zh_model",
                sparse_hash_dim=128,
            ),
            search=SimpleNamespace(vanilla=VanillaSearchConfig(recall_size=4, use_reranker=False)),
        ),
    )


@pytest.mark.asyncio
async def test_add_sync_mutes_entity_writes_by_default() -> None:
    writer = FakeWriter()
    pipeline = make_pipeline(writer)

    result = await pipeline.add_sync(
        AddPipelineInput(messages=[{"text": 'Kai uses QDRANT in "Memory Service".'}]),
        make_context(),
    )

    assert result.status == "ok"
    assert len(writer.calls) == 1

    plan = writer.calls[0].plan
    assert len(plan.memories) == 1
    assert plan.memories[0].project_id == "proj-1"
    assert plan.memories[0].content == 'Kai uses QDRANT in "Memory Service".'
    assert plan.memories[0].metadata["content_hash"]
    assert plan.memories[0].metadata["tokens"]

    assert len(plan.sources) == 1
    assert plan.sources[0].project_id == "proj-1"
    assert plan.sources[0].source_type == "message"
    assert plan.memories[0].metadata["source_id"] == plan.sources[0].source_id
    assert plan.memories[0].metadata["planner_action"] == "ADD"

    assert len(plan.vectors) == 1
    assert plan.vectors[0].memory_id == plan.memories[0].memory_id
    assert plan.vectors[0].bm25_indices
    assert len(plan.vectors[0].bm25_indices) == len(plan.vectors[0].bm25_values)

    assert plan.entities == []
    assert plan.entity_vectors == []

    assert plan.relationships
    rel_types = {rel.rel_type for rel in plan.relationships}
    assert REL_MENTIONS not in rel_types
    assert REL_EXTRACTED_FROM in rel_types
    assert REL_MENTIONED_IN_SOURCE not in rel_types
    assert {source.project_id for source in plan.sources} == {"proj-1"}
    assert {rel.project_id for rel in plan.relationships} == {"proj-1"}
    assert any(
        rel.source.kind == "Memory"
        and rel.target.kind == "Source"
        and rel.source.node_id == plan.memories[0].memory_id
        and rel.target.node_id == plan.sources[0].source_id
        for rel in plan.relationships
        if rel.rel_type == REL_EXTRACTED_FROM
    )
    assert result.memories[0].memory_id == plan.memories[0].memory_id
    assert result.memories[0].mem_type == "fact"
    assert result.memories[0].graph_edge_count >= 1


@pytest.mark.asyncio
async def test_add_sync_can_enable_entity_writes_and_mentions_relationships() -> None:
    writer = FakeWriter()
    pipeline = make_pipeline_with_reader(
        FakeReader(),
        writer,
        vanilla_add_config=VanillaAddConfig(enable_entities=True),
    )

    await pipeline.add_sync(
        AddPipelineInput(messages=[{"text": 'Kai uses QDRANT in "Memory Service".'}]),
        make_context(),
    )

    plan = writer.calls[0].plan
    assert plan.entities
    assert {entity.project_id for entity in plan.entities} == {"proj-1"}
    assert all(entity.metadata.get("search_fields") for entity in plan.entities)
    entity_vector_ids = {vector.entity_id for vector in plan.entity_vectors}
    assert {entity.entity_id for entity in plan.entities}.issubset(entity_vector_ids)
    assert any("#sf" in vector_id for vector_id in entity_vector_ids)
    assert {entity.entity_id for entity in plan.entities} == {
        rel.target.node_id for rel in plan.relationships if rel.rel_type == REL_MENTIONS
    }
    rel_types = {rel.rel_type for rel in plan.relationships}
    assert REL_MENTIONS in rel_types
    assert REL_MENTIONED_IN_SOURCE in rel_types
    assert all(
        rel.source.kind == "Entity" and rel.target.kind == "Source"
        for rel in plan.relationships
        if rel.rel_type == REL_MENTIONED_IN_SOURCE
    )
    assert all(rel.metadata["extractor"] for rel in plan.relationships if rel.rel_type == REL_MENTIONS)


@pytest.mark.asyncio
async def test_add_sync_without_add_record_id_does_not_write_back() -> None:
    writer = FakeWriter()
    recorder = FakeRecorder()
    pipeline = make_pipeline(writer, recorder=recorder)

    result = await pipeline.add_sync(
        AddPipelineInput(
            messages=[
                {"text": "Kai uses Qdrant."},
                {"text": "Kai also uses FastAPI."},
            ],
            timestamp=1770000000000,
            force_generation=True,
        ),
        make_context(),
    )

    assert result.status == "ok"
    assert len(result.memories) == 2
    assert recorder.completed == []


@pytest.mark.asyncio
async def test_add_sync_writes_output_back_onto_add_record_id() -> None:
    writer = FakeWriter()
    recorder = FakeRecorder()
    pipeline = make_pipeline(writer, recorder=recorder)

    result = await pipeline.add_sync(
        AddPipelineInput(
            messages=[
                {"text": "Kai uses Qdrant."},
                {"text": "Kai also uses FastAPI."},
            ],
            timestamp=1770000000000,
            force_generation=True,
        ),
        make_context(),
        add_record_id="rec-1",
    )

    assert result.status == "ok"
    assert [add_record_id for add_record_id, _ in recorder.completed] == ["rec-1"]
    assert recorder.completed[0][1] is result


@pytest.mark.asyncio
async def test_add_sync_writes_file_and_url_sources_without_text_memories() -> None:
    writer = FakeWriter()
    pipeline = make_pipeline(writer)

    result = await pipeline.add_sync(
        AddPipelineInput(
            messages=[
                {"file_name": "notes.pdf", "file_path": "oss://bucket/notes.pdf"},
                {"url": "https://example.com/design"},
            ]
        ),
        make_context(),
    )

    assert result.status == "ok"
    assert result.memories == []
    assert len(writer.calls) == 1

    plan = writer.calls[0].plan
    assert plan.memories == []
    assert {source.source_type for source in plan.sources} == {"file", "url"}
    assert {source.is_parsed for source in plan.sources} == {False}


@pytest.mark.asyncio
async def test_add_sync_planner_skip_does_not_write_empty_segments() -> None:
    writer = FakeWriter()
    pipeline = make_pipeline(writer)

    result = await pipeline.add_sync(
        AddPipelineInput(messages=[{"text": "   "}]),
        make_context(),
    )

    assert result.status == "ok"
    assert result.memories == []
    assert writer.calls == []


@pytest.mark.asyncio
async def test_add_sync_builds_message_source_from_messages() -> None:
    writer = FakeWriter()
    pipeline = make_pipeline(writer)

    await pipeline.add_sync(
        AddPipelineInput(messages=[{"text": "Kai uses FastAPI."}]),
        make_context(),
    )

    plan = writer.calls[0].plan
    assert {source.source_type for source in plan.sources} == {"message"}
    assert plan.sources[0].metadata["message_index"] == 0


@pytest.mark.asyncio
async def test_add_sync_exact_duplicate_reinforces_existing_memory_and_links_source() -> None:
    writer = FakeWriter()
    reader = FakeReader()
    pipeline = make_pipeline_with_reader(reader, writer)
    preprocessed = pipeline._text_preprocessor.preprocess_text("Kai uses Qdrant.", segment_id="existing")
    reader.listed_memories = [
        MemoryView(
            memory_id="mem-existing",
            project_id="proj-1",
            content="Kai uses Qdrant.",
            mem_type="fact",
            status="active",
            metadata={"content_hash": preprocessed.content_hash, "reinforcement_count": 2},
        )
    ]

    result = await pipeline.add_sync(AddPipelineInput(messages=[{"text": "Kai uses Qdrant."}]), make_context())

    assert result.memories[0].operation == "reinforcement"
    assert result.memories[0].memory_id == "mem-existing"
    assert result.memories[0].related_memory_ids == ["mem-existing"]

    assert len(writer.update_calls) == 1
    command = writer.update_calls[0].command
    assert command.memory_id == "mem-existing"
    assert command.content is None
    assert command.reinforcement_count_delta == 1
    assert command.metadata_patch["last_reinforced_request_id"] == "req-1"

    assert len(writer.calls) == 1
    plan = writer.calls[0].plan
    assert plan.memories == []
    assert plan.vectors == []
    assert {rel.rel_type for rel in plan.relationships} == {REL_EXTRACTED_FROM}
    assert plan.relationships[0].source.node_id == "mem-existing"


@pytest.mark.asyncio
async def test_add_sync_related_memory_adds_relates_to_edge_for_new_memory() -> None:
    from mindmemos.typing.memory_db import MemoryDbSearchHit

    writer = FakeWriter()
    reader = FakeReader()
    existing = MemoryView(
        memory_id="mem-related",
        project_id="proj-1",
        content="Kai uses vector search.",
        mem_type="fact",
        status="active",
    )
    reader.listed_memories = [existing]
    reader.sparse_hits = [
        MemoryDbSearchHit(
            memory_id="mem-related",
            score=0.7,
            memory=existing,
            source="bm25",
            rank=1,
        )
    ]
    pipeline = make_pipeline_with_reader(reader, writer)

    await pipeline.add_sync(AddPipelineInput(messages=[{"text": "Kai uses Qdrant for memory search."}]), make_context())

    plan = writer.calls[0].plan
    memory_id = plan.memories[0].memory_id
    relates = [rel for rel in plan.relationships if rel.rel_type == REL_RELATES_TO]
    assert len(relates) == 1
    assert relates[0].source.node_id == memory_id
    assert relates[0].target.node_id == "mem-related"
    assert relates[0].edge_type == "related_to"


@pytest.mark.asyncio
async def test_add_sync_uses_extractor_candidate_content_and_memory_type() -> None:
    from mindmemos.components.extractor.vanilla import (
        ExtractedMemoryCandidate,
        ExtractedSourceCandidate,
        MemoryExtractionResult,
    )

    writer = FakeWriter()
    extractor = FakeExtractor(
        MemoryExtractionResult(
            memories=[
                ExtractedMemoryCandidate(
                    ref_id="m1",
                    content="Kai prefers FastAPI for backend APIs.",
                    mem_type="profile",
                    confidence=0.86,
                    importance=0.7,
                    source_refs=["s1"],
                    action_hint="add",
                    reason="explicit preference",
                )
            ],
            sources=[
                ExtractedSourceCandidate(ref_id="s1", message_index=0),
            ],
        )
    )
    pipeline = VanillaAddPipeline(
        db_reader=FakeReader(),
        db_writer=writer,
        memory_extractor=extractor,
        text_config=TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        ),
        consistency="fast",
        llm_client=None,
        embed_client=None,
    )

    result = await pipeline.add_sync(
        AddPipelineInput(messages=[{"text": "I prefer FastAPI."}]),
        make_context(),
    )

    plan = writer.calls[0].plan
    assert extractor.calls
    assert plan.memories[0].content == "Kai prefers FastAPI for backend APIs."
    assert plan.memories[0].mem_type == "profile"
    assert plan.memories[0].metadata["extractor"] == "vanilla_llm_chunked"
    assert plan.memories[0].metadata["extractor_confidence"] == 0.86
    assert result.memories[0].mem_type == "profile"


@pytest.mark.asyncio
async def test_add_sync_deterministic_memory_id_on_same_request() -> None:
    """Two add_sync calls with the same request context produce the same memory_id."""
    writer1 = FakeWriter()
    pipeline1 = make_pipeline(writer1)
    ctx = make_context()

    result1 = await pipeline1.add_sync(
        AddPipelineInput(messages=[{"text": "Kai uses Qdrant."}]),
        ctx,
    )

    writer2 = FakeWriter()
    pipeline2 = make_pipeline(writer2)

    result2 = await pipeline2.add_sync(
        AddPipelineInput(messages=[{"text": "Kai uses Qdrant."}]),
        ctx,
    )

    id1 = writer1.calls[0].plan.memories[0].memory_id
    id2 = writer2.calls[0].plan.memories[0].memory_id
    assert id1 == id2, f"Same request should produce same memory_id: {id1} != {id2}"


@pytest.mark.asyncio
async def test_add_sync_different_request_id_produces_different_memory_id() -> None:
    """Different request_id produces different memory_id for the same content."""
    writer1 = FakeWriter()
    pipeline1 = make_pipeline(writer1)

    ctx_a = MemoryRequestContext(
        request_id="req-A",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )
    result1 = await pipeline1.add_sync(
        AddPipelineInput(messages=[{"text": "Kai uses Qdrant."}]),
        ctx_a,
    )

    writer2 = FakeWriter()
    pipeline2 = make_pipeline(writer2)

    ctx_b = MemoryRequestContext(
        request_id="req-B",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )
    result2 = await pipeline2.add_sync(
        AddPipelineInput(messages=[{"text": "Kai uses Qdrant."}]),
        ctx_b,
    )

    id1 = writer1.calls[0].plan.memories[0].memory_id
    id2 = writer2.calls[0].plan.memories[0].memory_id
    assert id1 != id2, "Different request_id should produce different memory_id"


@pytest.mark.asyncio
async def test_vanilla_add_then_vanilla_search_round_trips_through_in_memory_store(monkeypatch) -> None:
    monkeypatch.setattr(
        "mindmemos.pipelines.search.pipeline.get_config",
        _fake_search_config,
    )
    monkeypatch.setattr(
        "mindmemos.pipelines.search.vanilla.engine.get_config",
        _fake_search_config,
    )
    # VanillaSearchEngine lazy-load path falls back to get_text_preprocessor() when
    # text_config is not supplied, which reads its own module-level get_config.
    monkeypatch.setattr(
        "mindmemos.components.text.preprocessor.get_config",
        _fake_search_config,
    )
    store = InMemoryVanillaStore()
    add_pipeline = VanillaAddPipeline(
        db_reader=store,
        db_writer=store,
        text_config=TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        ),
        consistency="fast",
        llm_client=None,
        embed_client=None,
    )
    search_pipeline = SearchPipelineImpl(
        db_reader=store,
        db_writer=store,
        final_filter=None,
        rerank_client=None,
    )
    ctx = make_context()

    add_result = await add_pipeline.add_sync(
        AddPipelineInput(messages=[{"text": "Kai uses Qdrant for memory search."}]),
        ctx,
    )
    search_result = await search_pipeline.search(
        SearchPipelineInput(query="Qdrant memory", search_pipeline="vanilla", top_k=3),
        ctx,
    )

    assert add_result.memories[0].operation == "add"
    assert search_result.status == "ok"
    assert [memory.memory for memory in search_result.memories] == ["Kai uses Qdrant for memory search."]
    assert search_result.memories[0].id == add_result.memories[0].memory_id
    assert store.memories[add_result.memories[0].memory_id].metadata["source_id"] in store.sources
    assert store.vectors[add_result.memories[0].memory_id].bm25_indices
    assert store.entities == {}
    assert store.entity_vectors == {}
