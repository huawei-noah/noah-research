"""End-to-end tests for the skill self-evolution pipeline against in-memory Qdrant.

A fake LLM client makes the summarize / propose / apply stages deterministic, so
the tests pin the trace-selection, threshold gating, add-order batching and
summary-consumption behavior rather than model output.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from mindmemos.components.skill import deserialize_bundle, serialize_bundle
from mindmemos.infra.db.models import AddRecordPoint
from mindmemos.infra.db.qdrant import QdrantStore
from mindmemos.pipelines.skill import SkillVersionStore
from mindmemos.pipelines.skill.evolution import SkillEvolver
from mindmemos.prompts.EN.skills import REWRITE_SKILL_SYSTEM, rewrite_skill_user
from mindmemos.typing.skill import SkillVersionStatus
from qdrant_client import AsyncQdrantClient

from mindmemos.config import QdrantConfig, SkillEvolutionConfig
from mindmemos.infra.db import SkillVersionRepository

PROJECT = "proj"
BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)
_ID_NS = uuid.UUID("11111111-2222-3333-4444-555555555555")


def _trace_id(label: str) -> str:
    """Deterministic UUID point id (local Qdrant requires UUID ids)."""

    return str(uuid.uuid5(_ID_NS, label))


class FakeChatResponse:
    def __init__(self, content: str, parsed=None) -> None:
        self.content = content
        self.finish_reason = "stop"
        self.parsed = parsed


def _gutter_line_count(messages) -> int:
    """Count the lines in the numbered gutter the apply prompt embedded.

    The apply prompt renders the current SKILL.md with an ``N| `` line-number
    gutter (see ``apply_patch_user`` / ``format_numbered``); the fake applier uses
    the count to emit a whole-document replace by line range.
    """

    user = messages[-1]["content"]
    block = user.split("part of the file)\n", 1)[1].split("\n\n# Change plan", 1)[0]
    return len(block.splitlines())


class FakeLLM:
    """Deterministic chat client keyed by the pipeline's task tags."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.apply_counter = 0

    async def chat(self, *, task: str, messages, format_parser=None, **kwargs):
        self.calls.append(task)
        if task == "skill_trajectory_summary":
            return FakeChatResponse("analytical summary of one session")
        if task == "skill_patch_propose":
            return FakeChatResponse("change plan: add a warning")
        if task == "skill_patch_apply":
            self.apply_counter += 1
            # Emit a line-addressed whole-document replace op and let the real
            # format_parser (components.skill.edit) apply it, exercising the new
            # edit-op path end to end.
            n = _gutter_line_count(messages)
            content = json.dumps(
                {"edits": [{"op": "replace", "start": 1, "end": n, "new": f"evolved skill body v{self.apply_counter}"}]}
            )
            parsed = format_parser(content) if format_parser is not None else None
            return FakeChatResponse(content, parsed=parsed)
        if task == "skill_rewrite":
            return FakeChatResponse("reformatted skill body")
        raise AssertionError(f"unexpected task {task}")


@pytest_asyncio.fixture
async def harness():
    client = AsyncQdrantClient(":memory:")
    cfg = QdrantConfig(
        url="http://unused",
        add_record_collection="test_add_record",
        skill_version_collection="test_skill_version",
        skill_blob_collection="test_skill_blob",
        skill_trace_pending_collection="test_skill_trace_pending",
        skill_trace_summary_collection="test_skill_trace_summary",
        vector_size=2,
    )
    qdrant = QdrantStore(cfg, client=client)
    await qdrant.ensure_schema()
    skill_repo = SkillVersionRepository(cfg, engine=qdrant.engine)
    await skill_repo.ensure_schema()
    store = SkillVersionStore(skill_repo=skill_repo, add_record_repo=qdrant.add_record)
    llm = FakeLLM()
    evolver = SkillEvolver(store=store, skill_repo=skill_repo, add_record_repo=qdrant.add_record, llm_client=llm)
    try:
        yield store, skill_repo, qdrant, evolver, llm
    finally:
        await client.close()


def _evolution_cfg(**overrides) -> SkillEvolutionConfig:
    base = {"min_aggregate": 4, "max_aggregate": 8, "summary_concurrency": 4, "rewrite_skill": False}
    base.update(overrides)
    return SkillEvolutionConfig(**base)


@pytest.fixture
def patched_cfg(monkeypatch):
    """Install a SkillEvolutionConfig into the global config for the pipeline."""

    def apply(cfg: SkillEvolutionConfig) -> None:
        from mindmemos.pipelines.skill import evolution as evolution_module

        class _Algo:
            skill_evolution = cfg

        class _Cfg:
            algo_config = _Algo()

        monkeypatch.setattr(evolution_module, "get_config", lambda: _Cfg())

    return apply


async def _seed_injected_traces(qdrant: QdrantStore, version_id: str, count: int, *, name: str = "prd-writer") -> None:
    for i in range(count):
        ts = (BASE_TIME + timedelta(minutes=i)).isoformat()
        await qdrant.upsert_add_record(
            AddRecordPoint(
                add_record_id=_trace_id(f"trace-{i}"),
                payload={
                    "project_id": PROJECT,
                    "task_completed_at": ts,
                    "messages": [{"role": "user", "content": f"do task {i}"}],
                    "skill_bindings": [
                        {"name": name, "content_hash": "h", "version_id": version_id, "usage": "injected"}
                    ],
                },
            )
        )


@pytest.mark.asyncio
async def test_below_threshold_does_not_summarize(harness, patched_cfg):
    store, skill_repo, qdrant, evolver, llm = harness
    patched_cfg(_evolution_cfg())
    root = await store.register(project_id=PROJECT, name="prd-writer", content=serialize_bundle({"SKILL.md": "v1"}))
    await _seed_injected_traces(qdrant, root.version_id, 3)

    result = await evolver.evolve(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)

    assert result.evolved is False
    assert result.pending_count == 3
    assert result.threshold == 4
    assert llm.calls == []  # statistics first, summarize only when enough


@pytest.mark.asyncio
async def test_evolves_single_version_when_threshold_met(harness, patched_cfg):
    store, skill_repo, qdrant, evolver, llm = harness
    patched_cfg(_evolution_cfg())
    root = await store.register(project_id=PROJECT, name="prd-writer", content=serialize_bundle({"SKILL.md": "v1"}))
    await _seed_injected_traces(qdrant, root.version_id, 5)

    result = await evolver.evolve(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)

    assert result.evolved is True
    assert result.consumed_count == 5
    assert result.summarized_count == 5
    assert result.new_version_id is not None
    assert len(result.new_version_ids) == 1

    # New version is a cloud/draft child of root with the applied body.
    new_version = await store.get_content(
        project_id=PROJECT, cloud_skill_id=root.cloud_skill_id, version_id=result.new_version_id
    )
    assert new_version.version.parent_version_id == root.version_id
    assert new_version.version.status is SkillVersionStatus.DRAFT
    assert new_version.version.origin.value == "cloud"
    assert deserialize_bundle(new_version.content)["SKILL.md"] == "evolved skill body v1"


@pytest.mark.asyncio
async def test_consumed_summaries_not_reused(harness, patched_cfg):
    store, skill_repo, qdrant, evolver, llm = harness
    patched_cfg(_evolution_cfg())
    root = await store.register(project_id=PROJECT, name="prd-writer", content=serialize_bundle({"SKILL.md": "v1"}))
    await _seed_injected_traces(qdrant, root.version_id, 5)

    first = await evolver.evolve(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)
    assert first.evolved is True

    # Re-running with no new traces: all 5 summaries are consumed -> below threshold.
    second = await evolver.evolve(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)
    assert second.evolved is False
    assert second.pending_count == 0


@pytest.mark.asyncio
async def test_serial_batches_in_add_order(harness, patched_cfg):
    store, skill_repo, qdrant, evolver, llm = harness
    patched_cfg(_evolution_cfg())
    root = await store.register(project_id=PROJECT, name="prd-writer", content=serialize_bundle({"SKILL.md": "v1"}))
    await _seed_injected_traces(qdrant, root.version_id, 12)

    result = await evolver.evolve(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)

    # 12 summaries -> batches of 8 then 4 -> two chained versions.
    assert result.evolved is True
    assert result.consumed_count == 12
    assert len(result.new_version_ids) == 2

    v1 = await store.get_content(
        project_id=PROJECT, cloud_skill_id=root.cloud_skill_id, version_id=result.new_version_ids[0]
    )
    v2 = await store.get_content(
        project_id=PROJECT, cloud_skill_id=root.cloud_skill_id, version_id=result.new_version_ids[1]
    )
    assert v1.version.parent_version_id == root.version_id
    assert v2.version.parent_version_id == result.new_version_ids[0]


@pytest.mark.asyncio
async def test_scans_beyond_single_scroll_page(harness, patched_cfg):
    """Regression: the candidate scan must paginate past one 200-record page.

    Qdrant disables cursor pagination when ``order_by`` is set, which previously
    capped the scan at the oldest ~200 traces and silently stranded the rest. With
    256 injected traces (an exact multiple of ``max_aggregate``), every one must be
    summarized and consumed rather than capped at the first 200-record page.
    """

    store, skill_repo, qdrant, evolver, llm = harness
    patched_cfg(_evolution_cfg(min_aggregate=4, max_aggregate=8))
    root = await store.register(project_id=PROJECT, name="prd-writer", content=serialize_bundle({"SKILL.md": "v1"}))
    await _seed_injected_traces(qdrant, root.version_id, 256)

    total_consumed = 0
    # Drain across repeated passes the way the batched runner would.
    for _ in range(100):
        result = await evolver.evolve(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)
        total_consumed += result.consumed_count
        if not result.evolved and result.pending_count == 0:
            break

    assert total_consumed == 256


@pytest.mark.asyncio
async def test_leftover_below_threshold_deferred(harness, patched_cfg):
    store, skill_repo, qdrant, evolver, llm = harness
    patched_cfg(_evolution_cfg())
    root = await store.register(project_id=PROJECT, name="prd-writer", content=serialize_bundle({"SKILL.md": "v1"}))
    await _seed_injected_traces(qdrant, root.version_id, 10)

    result = await evolver.evolve(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)

    # 10 -> one batch of 8, 2 deferred (still summarized + unconsumed).
    assert result.evolved is True
    assert result.consumed_count == 8
    assert len(result.new_version_ids) == 1

    # A follow-up call sees only the 2 leftover unconsumed summaries.
    again = await evolver.evolve(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)
    assert again.evolved is False
    assert again.pending_count == 2


@pytest.mark.asyncio
async def test_ignores_non_injected_and_other_skill_bindings(harness, patched_cfg):
    store, skill_repo, qdrant, evolver, llm = harness
    patched_cfg(_evolution_cfg())
    root = await store.register(project_id=PROJECT, name="prd-writer", content=serialize_bundle({"SKILL.md": "v1"}))

    # 4 injected for this skill version.
    await _seed_injected_traces(qdrant, root.version_id, 4)
    # Noise: a modified binding and an injected binding for a foreign version.
    await qdrant.upsert_add_record(
        AddRecordPoint(
            add_record_id=_trace_id("noise-modified"),
            payload={
                "project_id": PROJECT,
                "task_completed_at": BASE_TIME.isoformat(),
                "messages": [{"role": "user", "content": "x"}],
                "skill_bindings": [
                    {"name": "prd-writer", "content_hash": "h", "version_id": root.version_id, "usage": "modified"}
                ],
            },
        )
    )
    await qdrant.upsert_add_record(
        AddRecordPoint(
            add_record_id=_trace_id("noise-foreign"),
            payload={
                "project_id": PROJECT,
                "task_completed_at": BASE_TIME.isoformat(),
                "messages": [{"role": "user", "content": "y"}],
                "skill_bindings": [
                    {"name": "other", "content_hash": "h", "version_id": "foreign-version", "usage": "injected"}
                ],
            },
        )
    )

    result = await evolver.evolve(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)
    assert result.evolved is True
    assert result.consumed_count == 4


@pytest.mark.asyncio
async def test_rewrite_pass_runs_when_enabled(harness, patched_cfg):
    store, skill_repo, qdrant, evolver, llm = harness
    patched_cfg(_evolution_cfg(rewrite_skill=True))
    root = await store.register(project_id=PROJECT, name="prd-writer", content=serialize_bundle({"SKILL.md": "v1"}))
    await _seed_injected_traces(qdrant, root.version_id, 4)

    result = await evolver.evolve(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)

    assert result.evolved is True
    assert "skill_rewrite" in llm.calls
    new_version = await store.get_content(
        project_id=PROJECT, cloud_skill_id=root.cloud_skill_id, version_id=result.new_version_id
    )
    assert deserialize_bundle(new_version.content)["SKILL.md"] == "reformatted skill body"


def test_skill_evolver_is_registered_and_selectable():
    """The evolve algorithm version is pluggable via the pipeline registry."""

    from mindmemos.pipelines.registry import create_pipeline, load_builtin_pipelines

    load_builtin_pipelines()
    evolver = create_pipeline(type="skill_evolve", name="trace_v2_summary")
    assert isinstance(evolver, SkillEvolver)


def test_unknown_evolve_version_raises():
    from mindmemos.pipelines.registry import create_pipeline

    with pytest.raises(ValueError, match="Unknown skill_evolve pipeline"):
        create_pipeline(type="skill_evolve", name="does_not_exist")


def test_rewrite_prompt_is_format_only():
    assert "format-repair" in REWRITE_SKILL_SYSTEM
    assert "Do not add, delete, merge, deduplicate, reorder, summarize, or rewrite" in rewrite_skill_user("x")
    assert "DEDUPLICATE AGGRESSIVELY" not in REWRITE_SKILL_SYSTEM
    assert "Consolidate duplicated guidance" not in rewrite_skill_user("x")
