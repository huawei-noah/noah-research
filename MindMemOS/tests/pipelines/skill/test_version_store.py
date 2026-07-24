"""End-to-end tests for SkillVersionStore against an in-memory Qdrant.

Covers register idempotency / lineage and the §2.1 trace-binding rules including
pending-trace rebind onto the originating add record.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from mindmemos.components.skill import compute_content_hash, serialize_bundle
from mindmemos.infra.db.models import AddRecordPoint
from mindmemos.infra.db.qdrant import QdrantStore
from mindmemos.pipelines.skill import SkillVersionStore
from mindmemos.typing.skill import SkillContext, SkillOrigin, SkillSyncRequestItem, SkillVersionStatus
from qdrant_client import AsyncQdrantClient

from mindmemos.config import QdrantConfig
from mindmemos.errors import SkillBundleError, SkillNotFoundError, SkillVersionNotFoundError
from mindmemos.infra.db import SkillVersionRepository
from mindmemos.mappers import to_skill_version_point

PROJECT = "proj"


def bundle(text: str) -> str:
    """Canonical bundle text for a single-file SKILL.md with ``text`` body."""

    return serialize_bundle({"SKILL.md": text})


def content_hash(text: str) -> str:
    return compute_content_hash({"SKILL.md": text})


@pytest_asyncio.fixture
async def store():
    client = AsyncQdrantClient(":memory:")
    cfg = QdrantConfig(
        url="http://unused",
        add_record_collection="test_add_record",
        skill_version_collection="test_skill_version",
        skill_blob_collection="test_skill_blob",
        skill_trace_pending_collection="test_skill_trace_pending",
        vector_size=2,
    )
    qdrant = QdrantStore(cfg, client=client)
    await qdrant.ensure_schema()
    skill_repo = SkillVersionRepository(cfg, engine=qdrant.engine)
    version_store = SkillVersionStore(skill_repo=skill_repo, add_record_repo=qdrant.add_record)
    try:
        yield version_store, skill_repo, qdrant
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_register_root_then_idempotent(store):
    version_store, _, _ = store
    v1 = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("hello"))
    assert v1.content_hash == content_hash("hello")
    assert v1.parent_version_id is None
    assert v1.status.value == "observed"
    assert v1.origin.value == "edge"

    # Same content + same key -> same version, same cloud_skill_id (no fork).
    again = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("hello"))
    assert again.version_id == v1.version_id
    assert again.cloud_skill_id == v1.cloud_skill_id


@pytest.mark.asyncio
async def test_register_branch_inherits_cloud_skill_id(store):
    version_store, _, _ = store
    root = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("v1"))
    child = await version_store.register(
        project_id=PROJECT,
        name="prd-writer",
        content=bundle("v2"),
        parent_version_id=root.version_id,
    )
    assert child.parent_version_id == root.version_id
    assert child.cloud_skill_id == root.cloud_skill_id
    assert child.version_id != root.version_id


@pytest.mark.asyncio
async def test_register_unknown_parent_raises(store):
    version_store, _, _ = store
    with pytest.raises(SkillVersionNotFoundError):
        await version_store.register(
            project_id=PROJECT,
            name="prd-writer",
            content=bundle("x"),
            parent_version_id="does-not-exist",
        )


@pytest.mark.asyncio
async def test_register_empty_bundle_raises(store):
    version_store, _, _ = store
    # A canonical bundle with no whitelisted file.
    with pytest.raises(SkillBundleError):
        await version_store.register(project_id=PROJECT, name="x", content="[]")


@pytest.mark.asyncio
async def test_register_accepts_raw_skill_md_body(store):
    """A bare SKILL.md body hashes identically to its canonical bundle form."""

    version_store, _, _ = store
    v = await version_store.register(project_id=PROJECT, name="prd-writer", content="raw body")
    assert v.content_hash == content_hash("raw body")


@pytest.mark.asyncio
async def test_list_get_versions_and_content(store):
    version_store, _, _ = store
    root = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("v1"))
    child = await version_store.register(
        project_id=PROJECT,
        name="prd-writer",
        content=bundle("v2"),
        parent_version_id=root.version_id,
    )

    skills = await version_store.list_skills(project_id=PROJECT)
    assert [skill.cloud_skill_id for skill in skills] == [root.cloud_skill_id]
    assert skills[0].latest_version.version_id == child.version_id
    assert skills[0].published_head is None

    detail = await version_store.get_skill(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)
    assert detail.latest_version.version_id == child.version_id

    versions = await version_store.versions_since(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)
    assert [version.version_id for version in versions] == [root.version_id, child.version_id]

    content = await version_store.get_content(
        project_id=PROJECT,
        cloud_skill_id=root.cloud_skill_id,
        version_id=child.version_id,
    )
    assert content.version.version_id == child.version_id
    assert content.content == bundle("v2")


@pytest.mark.asyncio
async def test_get_content_rejects_version_from_another_cloud_skill(store):
    version_store, _, _ = store
    first = await version_store.register(project_id=PROJECT, name="first", content=bundle("first"))
    second = await version_store.register(project_id=PROJECT, name="second", content=bundle("second"))

    with pytest.raises(SkillVersionNotFoundError):
        await version_store.get_content(
            project_id=PROJECT,
            cloud_skill_id=first.cloud_skill_id,
            version_id=second.version_id,
        )


@pytest.mark.asyncio
async def test_delete_skill_unmanages_versions(store):
    version_store, _, _ = store
    root = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("v1"))

    await version_store.delete_skill(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)

    with pytest.raises(SkillNotFoundError):
        await version_store.get_skill(project_id=PROJECT, cloud_skill_id=root.cloud_skill_id)


@pytest.mark.asyncio
async def test_sync_reports_published_head_diff(store):
    version_store, skill_repo, _ = store
    root = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("v1"))
    child = await version_store.register(
        project_id=PROJECT,
        name="prd-writer",
        content=bundle("v2"),
        parent_version_id=root.version_id,
    )
    published = child.model_copy(update={"status": SkillVersionStatus.PUBLISHED, "origin": SkillOrigin.CLOUD})
    await skill_repo.upsert_version(to_skill_version_point(published))

    results = await version_store.sync(
        project_id=PROJECT,
        items=[SkillSyncRequestItem(cloud_skill_id=root.cloud_skill_id, local_version_id=root.version_id)],
    )

    assert len(results) == 1
    assert results[0].has_update is True
    assert results[0].published_head is not None
    assert results[0].published_head.version_id == child.version_id
    assert results[0].gating_status == "published"


@pytest.mark.asyncio
async def test_sync_without_published_head_is_not_an_update(store):
    version_store, _, _ = store
    root = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("v1"))

    results = await version_store.sync(
        project_id=PROJECT,
        items=[SkillSyncRequestItem(cloud_skill_id=root.cloud_skill_id, local_version_id=root.version_id)],
    )

    assert results[0].has_update is False
    assert results[0].published_head is None
    assert results[0].gating_status == "no_published_head"


@pytest.mark.asyncio
async def test_bind_unchanged_binds_base(store):
    version_store, _, _ = store
    root = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("hello"))
    bindings = await version_store.bind_skill_context(
        project_id=PROJECT,
        add_record_id="rec-1",
        skill_context=[
            SkillContext(name="prd-writer", content_hash=content_hash("hello"), base_version_id=root.version_id)
        ],
    )
    assert len(bindings) == 1
    assert bindings[0].version_id == root.version_id


@pytest.mark.asyncio
async def test_bind_existing_derived_version(store):
    version_store, _, _ = store
    root = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("v1"))
    child = await version_store.register(
        project_id=PROJECT, name="prd-writer", content=bundle("v2"), parent_version_id=root.version_id
    )
    # Local content == child's content, base is still root -> binds the derived child.
    bindings = await version_store.bind_skill_context(
        project_id=PROJECT,
        add_record_id="rec-1",
        skill_context=[
            SkillContext(name="prd-writer", content_hash=content_hash("v2"), base_version_id=root.version_id)
        ],
    )
    assert bindings[0].version_id == child.version_id


@pytest.mark.asyncio
async def test_bind_miss_parks_pending(store):
    version_store, skill_repo, _ = store
    bindings = await version_store.bind_skill_context(
        project_id=PROJECT,
        add_record_id="rec-1",
        skill_context=[SkillContext(name="prd-writer", content_hash=content_hash("brand-new"), base_version_id="")],
    )
    assert bindings[0].version_id is None
    pending = await skill_repo.scroll_pending_traces(PROJECT, content_hash("brand-new"))
    assert len(pending) == 1
    assert pending[0].payload["add_record_id"] == "rec-1"


@pytest.mark.asyncio
async def test_register_rebinds_pending_trace_onto_add_record(store):
    version_store, skill_repo, qdrant = store
    new_hash = content_hash("brand-new")
    rec_id = str(uuid.uuid4())

    # 1) An add records a miss: binding has version_id=None and a pending trace exists.
    bindings = await version_store.bind_skill_context(
        project_id=PROJECT,
        add_record_id=rec_id,
        skill_context=[SkillContext(name="prd-writer", content_hash=new_hash, base_version_id="")],
    )
    await qdrant.upsert_add_record(
        AddRecordPoint(
            add_record_id=rec_id,
            payload={"project_id": PROJECT, "skill_bindings": [b.model_dump(mode="json") for b in bindings]},
        )
    )

    # 2) ensure/register uploads the content -> rebind fills the add record binding.
    version = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("brand-new"))

    record = await qdrant.add_record.get(PROJECT, rec_id)
    assert record is not None
    stored = record.payload["skill_bindings"]
    assert stored[0]["version_id"] == version.version_id
    # Pending trace cleared after rebind.
    assert await skill_repo.scroll_pending_traces(PROJECT, new_hash) == []


@pytest.mark.asyncio
async def test_register_rebinds_pending_traces_across_pages(store):
    version_store, skill_repo, qdrant = store
    new_hash = content_hash("many-pending")
    rec_ids = [str(uuid.uuid4()) for _ in range(205)]

    for rec_id in rec_ids:
        bindings = await version_store.bind_skill_context(
            project_id=PROJECT,
            add_record_id=rec_id,
            skill_context=[SkillContext(name="prd-writer", content_hash=new_hash, base_version_id="")],
        )
        await qdrant.upsert_add_record(
            AddRecordPoint(
                add_record_id=rec_id,
                payload={"project_id": PROJECT, "skill_bindings": [b.model_dump(mode="json") for b in bindings]},
            )
        )

    version = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("many-pending"))
    records = await qdrant.add_record.retrieve(PROJECT, rec_ids)

    assert len(records) == len(rec_ids)
    assert {record.payload["skill_bindings"][0]["version_id"] for record in records} == {version.version_id}
    assert await skill_repo.scroll_pending_traces(PROJECT, new_hash) == []


@pytest.mark.asyncio
async def test_register_keeps_pending_when_add_record_is_not_written_yet(store):
    version_store, skill_repo, _ = store
    new_hash = content_hash("delayed-add-record")
    rec_id = str(uuid.uuid4())

    await version_store.bind_skill_context(
        project_id=PROJECT,
        add_record_id=rec_id,
        skill_context=[SkillContext(name="prd-writer", content_hash=new_hash, base_version_id="")],
    )

    await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("delayed-add-record"))

    pending = await skill_repo.scroll_pending_traces(PROJECT, new_hash)
    assert len(pending) == 1
    assert pending[0].payload["add_record_id"] == rec_id


@pytest.mark.asyncio
async def test_existing_register_rebinds_leftover_pending_trace(store):
    version_store, skill_repo, qdrant = store
    new_hash = content_hash("late-retry")
    rec_id = str(uuid.uuid4())

    bindings = await version_store.bind_skill_context(
        project_id=PROJECT,
        add_record_id=rec_id,
        skill_context=[SkillContext(name="prd-writer", content_hash=new_hash, base_version_id="")],
    )
    version = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("late-retry"))
    assert len(await skill_repo.scroll_pending_traces(PROJECT, new_hash)) == 1

    await qdrant.upsert_add_record(
        AddRecordPoint(
            add_record_id=rec_id,
            payload={"project_id": PROJECT, "skill_bindings": [b.model_dump(mode="json") for b in bindings]},
        )
    )

    again = await version_store.register(project_id=PROJECT, name="prd-writer", content=bundle("late-retry"))

    record = await qdrant.add_record.get(PROJECT, rec_id)
    assert again.version_id == version.version_id
    assert record is not None
    assert record.payload["skill_bindings"][0]["version_id"] == version.version_id
    assert await skill_repo.scroll_pending_traces(PROJECT, new_hash) == []
