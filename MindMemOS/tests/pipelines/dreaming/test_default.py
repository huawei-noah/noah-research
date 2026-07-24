from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from mindmemos.pipelines.dreaming.default import DefaultDreamingPipeline
from mindmemos.typing.activity import ActivityScope, RecentActivityBundle, WrittenMemoryRef
from mindmemos.typing.algo import ConsolidationAction, ConsolidationCreate, ConsolidationLink, ConsolidationMerge
from mindmemos.typing.memory import GraphNeighborScope, MemoryRequestContext, MemoryView
from mindmemos.typing.service import DreamingPipelineInput

from mindmemos.config import DreamingConfig, TextProcessingConfig
from mindmemos.infra.db import QdrantRecord


class FakeReader:
    def __init__(self, memories: list[MemoryView]) -> None:
        self.memories = memories
        self.add_record_payloads: dict[str, dict] = {}

    async def get_memories(self, _ctx, memory_ids: list[str]) -> list[MemoryView]:
        requested = set(memory_ids)
        return [memory for memory in self.memories if memory.memory_id in requested]

    async def list_memories(self, _ctx, *, filters=None, limit=50, cursor=None):
        return self.memories[:limit], None

    async def list_add_records(self, _ctx, *, filters=None, limit=50, cursor=None):
        return [], None

    async def get_add_records_by_ids(self, _ctx, add_record_ids: list[str]):
        return [
            QdrantRecord(point_id=add_record_id, payload=self.add_record_payloads.get(add_record_id, {}))
            for add_record_id in add_record_ids
        ]

    async def list_memory_neighbor_scopes(
        self,
        _ctx,
        memory_ids: list[str],
        *,
        edge_filter=None,
        limit_per_entity: int = 50,
        limit_direct_per_memory: int = 20,
        attach_direct_neighbors_to_entity_scopes: bool = True,
    ) -> list[GraphNeighborScope]:
        all_ids = tuple(memory.memory_id for memory in self.memories[:limit_per_entity])
        return [
            GraphNeighborScope(
                seed_memory_id=memory_id,
                entity_id="entity-1",
                entity_name="Alice",
                entity_type="person",
                memory_ids=all_ids,
                source="shared_entity",
            )
            for memory_id in memory_ids
        ]

    async def search_sparse(self, _ctx, _req, *, indices, values):
        return SimpleNamespace(hits=[])


class FakeWriter:
    def __init__(self) -> None:
        self.plans: list[object] = []
        self.deleted: list[tuple[str, str]] = []
        self.updated: list[object] = []
        self.add_record_patches: list[tuple[str, dict]] = []

    async def apply_mutation_plan(self, _ctx, plan, *, consistency="fast"):
        write_plan = plan.to_write_plan()
        if write_plan.memories or write_plan.relationships:
            self.plans.append(write_plan)
        for command in plan.memory_updates:
            self.updated.append(command)
        for command in plan.memory_deletes:
            self.deleted.append((command.memory_id, command.reason))
        return SimpleNamespace(
            memory_ids=[m.memory_id for m in write_plan.memories],
            mutations=[SimpleNamespace(memory_id=command.memory_id, changed=True) for command in plan.memory_updates]
            + [SimpleNamespace(memory_id=command.memory_id, changed=True) for command in plan.memory_deletes],
            errors=[],
            graph_pending=False,
        )

    async def write(self, _ctx, plan, *, consistency="fast"):
        self.plans.append(plan)
        return SimpleNamespace(memory_ids=[m.memory_id for m in plan.memories])

    async def patch_add_record(self, _ctx, add_record_id, payload):
        self.add_record_patches.append((add_record_id, payload))
        return SimpleNamespace(changed=True)

    async def delete_memory(self, _ctx, req):
        self.deleted.append((req.memory_id, req.reason))
        return SimpleNamespace(changed=True)

    async def update_memory(self, _ctx, req):
        self.updated.append(req)
        return SimpleNamespace(changed=True)


class FakeEmbed:
    async def embed(self, *, task: str, text):
        texts = text if isinstance(text, list) else [text]
        return SimpleNamespace(embeddings=[[0.1, 0.2, 0.3] for _ in texts])


class FakeLLM:
    def __init__(self, action: ConsolidationAction) -> None:
        self.action = action
        self.calls = 0

    async def chat(self, **_kwargs):
        self.calls += 1
        if self.calls % 2 == 1:
            return SimpleNamespace(
                parsed=SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            candidate_type="needs_consolidation",
                            primary_memory_id="m1",
                            neighbor_memory_id="m2",
                            primary_value_hint="",
                            neighbor_value_hint="",
                            confidence="high",
                            reason="test relation",
                        )
                    ]
                )
            )
        return SimpleNamespace(parsed=self.action)


class FakeActivityCollector:
    def __init__(self, memories: list[MemoryView]) -> None:
        self.memories = memories

    async def collect(self, scope: ActivityScope, **_kwargs) -> RecentActivityBundle:
        now = datetime.now(UTC)
        return RecentActivityBundle(
            window_start=now - timedelta(days=1),
            window_end=now,
            scope=scope,
            written_memories=[
                WrittenMemoryRef(
                    memory_id=memory.memory_id,
                    content=memory.content,
                    add_record_ids=[f"add-{memory.memory_id}"],
                    session_id="sess",
                    user_id="user",
                )
                for memory in self.memories
            ],
        )


def ctx() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acct",
        project_id="proj",
        api_key_uuid="key",
        user_id="user",
        session_id="sess",
        scopes=["memory:write"],
    )


def memory(
    memory_id: str,
    *,
    content: str,
    entity_id: str = "entity-1",
    property_name: str = "preference",
    content_hash: str | None = None,
    created_offset: int = 0,
) -> MemoryView:
    now = datetime.now(UTC)
    metadata = {}
    if content_hash:
        metadata["content_hash"] = content_hash
    return MemoryView(
        memory_id=memory_id,
        project_id="proj",
        content=content,
        mem_type="fact",
        status="active",
        metadata=metadata,
        root_id=["root-1"],
        entity_id=entity_id,
        entity_type="person",
        property_name=property_name,
        created_at=now - timedelta(minutes=created_offset),
        update_at=now - timedelta(minutes=created_offset),
    )


def pipeline(
    *, memories: list[MemoryView], action: ConsolidationAction
) -> tuple[DefaultDreamingPipeline, FakeReader, FakeWriter]:
    reader = FakeReader(memories)
    writer = FakeWriter()
    pipe = DefaultDreamingPipeline(
        dreaming_config=DreamingConfig(
            lookback_days=7,
            max_scopes_per_run=5,
            max_seed_memories=20,
            max_memories_per_scope=20,
            min_scope_updates=1,
            min_cluster_size=2,
        ),
        text_config=TextProcessingConfig(),
        llm_client=FakeLLM(action),
        embed_client=FakeEmbed(),
        activity_collector=FakeActivityCollector(memories),
        db_reader=reader,
        db_writer=writer,
        consistency="fast",
    )
    return pipe, reader, writer


@pytest.mark.asyncio
async def test_dreaming_archives_exact_duplicates_before_llm_actions():
    memories = [
        memory("m1", content="Alice likes tea", content_hash="same", created_offset=1),
        memory("m2", content="Alice likes tea", content_hash="same", created_offset=2),
    ]
    pipe, reader, writer = pipeline(memories=memories, action=ConsolidationAction())

    result = await pipe.dream_sync(DreamingPipelineInput(), ctx())

    assert result.status == "ok"
    assert writer.deleted == [("m2", "duplicate_of:m1")]
    assert writer.plans == []


@pytest.mark.asyncio
async def test_dreaming_skips_done_add_records_when_selecting_hot_scopes():
    memories = [
        memory("m1", content="Alice likes tea", created_offset=1),
        memory("m2", content="Alice likes coffee", created_offset=2),
    ]
    pipe, reader, _writer = pipeline(memories=memories, action=ConsolidationAction())
    reader.add_record_payloads = {
        "add-m1": {"consolidation_status": "done"},
        "add-m2": {"consolidation_status": "pending"},
    }

    scopes = await pipe._select_hot_scopes(ctx())

    assert len(scopes) == 1
    assert scopes[0].primary_memory_id == "m2"
    assert scopes[0].add_record_ids == ("add-m2",)


@pytest.mark.asyncio
async def test_dreaming_dedupes_duplicate_clusters_before_llm_calls():
    memories = [
        memory("m1", content="Alice likes green tea", created_offset=2),
        memory("m2", content="Alice prefers jasmine tea", created_offset=1),
    ]
    action = ConsolidationAction()
    pipe, _reader, writer = pipeline(memories=memories, action=action)

    result = await pipe.dream_sync(DreamingPipelineInput(), ctx())

    assert result.status == "ok"
    assert pipe._llm_client.calls == 2
    patched_ids = {add_record_id for add_record_id, _payload in writer.add_record_patches}
    assert patched_ids == {"add-m1", "add-m2"}


@pytest.mark.asyncio
async def test_dreaming_merge_creates_vectorized_memory_and_archives_sources():
    memories = [
        memory("m1", content="Alice likes green tea", created_offset=2),
        memory("m2", content="Alice prefers jasmine tea", created_offset=1),
    ]
    action = ConsolidationAction(
        merges=[
            ConsolidationMerge(
                source_memory_ids=["m1", "m2"],
                target_content="Alice prefers green or jasmine tea.",
                target_entity_id="entity-1",
                target_property_name="preference",
                merge_reason="fragments describe the same preference",
            )
        ]
    )
    pipe, reader, writer = pipeline(memories=memories, action=action)

    await pipe.dream_sync(DreamingPipelineInput(), ctx())

    assert [item[0] for item in writer.deleted] == ["m1", "m2"]
    assert len(writer.plans) == 1
    plan = writer.plans[0]
    assert len(plan.memories) == 1
    assert plan.memories[0].content == "Alice prefers green or jasmine tea."
    assert plan.memories[0].parent_ids == ["m1", "m2"]
    assert len(plan.vectors) == 1
    assert plan.vectors[0].semantic_vector == [0.1, 0.2, 0.3]
    assert plan.relationships


@pytest.mark.asyncio
async def test_dreaming_link_actions_cannot_reference_new_memories():
    memories = [
        memory("m1", content="Alice likes green tea", created_offset=2),
        memory("m2", content="Alice prefers jasmine tea", created_offset=1),
    ]
    action = ConsolidationAction(
        creates=[
            ConsolidationCreate(
                content="Alice has tea preferences.",
                evidence_memory_ids=["m1", "m2"],
                reason="generalized preference",
            )
        ],
        links=[
            ConsolidationLink(
                source_kind="Memory",
                source_id="new-memory",
                target_kind="Memory",
                target_id="m1",
                relation_type="generalizes",
            )
        ],
    )
    pipe, _reader, writer = pipeline(memories=memories, action=action)

    await pipe.dream_sync(DreamingPipelineInput(), ctx())

    plan = writer.plans[0]
    assert len(plan.memories) == 1
    assert all(rel.relation_type != "generalizes" for rel in plan.relationships)
    assert {rel.relation_type for rel in plan.relationships if rel.target.node_id in {"m1", "m2"}} == {
        "dreaming_evidence"
    }
