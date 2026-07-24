from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from mindmemos.config import QdrantConfig
from mindmemos.infra.db import SkillVersionRepository
from mindmemos.mappers import (
    skill_blob_from_record,
    skill_version_from_record,
    skill_version_id,
    to_skill_blob_point,
    to_skill_trace_pending_point,
    to_skill_version_point,
)
from mindmemos.typing import (
    SkillBlob,
    SkillOrigin,
    SkillTracePending,
    SkillVersion,
    SkillVersionStatus,
)

PROJECT = "proj"


@pytest_asyncio.fixture
async def repo():
    client = AsyncQdrantClient(":memory:")
    cfg = QdrantConfig(
        url="http://unused",
        skill_version_collection="test_skill_version",
        skill_blob_collection="test_skill_blob",
        skill_trace_pending_collection="test_skill_trace_pending",
        vector_size=2,
    )
    repository = SkillVersionRepository(cfg, client=client)
    await repository.ensure_schema()
    try:
        yield repository
    finally:
        await client.close()


def _version(
    *,
    content_hash: str,
    parent: str | None,
    cloud_skill_id: str,
    status: SkillVersionStatus = SkillVersionStatus.OBSERVED,
    created_at: datetime | None = None,
) -> SkillVersion:
    version_id = skill_version_id(PROJECT, content_hash, parent)
    return SkillVersion(
        version_id=version_id,
        project_id=PROJECT,
        cloud_skill_id=cloud_skill_id,
        skill_name="prd-writer",
        content_hash=content_hash,
        parent_version_id=parent,
        version_label="1.0.0",
        status=status,
        origin=SkillOrigin.EDGE,
        created_at=created_at or datetime.now(UTC),
    )


def test_version_id_is_deterministic_and_key_sensitive():
    a = skill_version_id(PROJECT, "h1", None)
    assert a == skill_version_id(PROJECT, "h1", None)
    # Same content under a different parent is a different identity.
    assert skill_version_id(PROJECT, "h1", "p1") != a
    # Project isolation is baked into the id.
    assert skill_version_id("other", "h1", None) != a


@pytest.mark.asyncio
async def test_upsert_and_get_version_roundtrip(repo):
    version = _version(content_hash="h1", parent=None, cloud_skill_id="skill-1")
    await repo.upsert_version(to_skill_version_point(version))

    record = await repo.get_version(PROJECT, version.version_id)
    assert record is not None
    loaded = skill_version_from_record(record)
    assert loaded == version
    # Project isolation: another project cannot read it.
    assert await repo.get_version("other", version.version_id) is None


@pytest.mark.asyncio
async def test_upsert_version_is_idempotent(repo):
    version = _version(content_hash="h1", parent=None, cloud_skill_id="skill-1")
    await repo.upsert_version(to_skill_version_point(version))
    await repo.upsert_version(to_skill_version_point(version))

    records = await repo.versions_since(PROJECT, "skill-1")
    assert len(records) == 1


@pytest.mark.asyncio
async def test_blob_roundtrip_and_dedup(repo):
    blob = SkillBlob(project_id=PROJECT, content_hash="h1", content="canonical-text", created_at=datetime.now(UTC))
    await repo.upsert_blob(to_skill_blob_point(blob))
    await repo.upsert_blob(to_skill_blob_point(blob))

    record = await repo.get_blob(PROJECT, "h1")
    assert record is not None
    assert skill_blob_from_record(record).content == "canonical-text"
    assert await repo.get_blob("other", "h1") is None


@pytest.mark.asyncio
async def test_published_head_picks_newest_published(repo):
    base = datetime.now(UTC)
    await repo.upsert_version(
        to_skill_version_point(_version(content_hash="h1", parent=None, cloud_skill_id="skill-1", created_at=base))
    )
    await repo.upsert_version(
        to_skill_version_point(
            _version(
                content_hash="h2",
                parent=skill_version_id(PROJECT, "h1", None),
                cloud_skill_id="skill-1",
                status=SkillVersionStatus.PUBLISHED,
                created_at=base + timedelta(seconds=10),
            )
        )
    )
    newest = _version(
        content_hash="h3",
        parent=skill_version_id(PROJECT, "h2", skill_version_id(PROJECT, "h1", None)),
        cloud_skill_id="skill-1",
        status=SkillVersionStatus.PUBLISHED,
        created_at=base + timedelta(seconds=20),
    )
    await repo.upsert_version(to_skill_version_point(newest))

    head = await repo.published_head(PROJECT, "skill-1")
    assert head is not None
    assert head.payload["content_hash"] == "h3"
    # No published version for an unknown skill.
    assert await repo.published_head(PROJECT, "skill-x") is None


@pytest.mark.asyncio
async def test_versions_since_returns_incremental_ascending(repo):
    base = datetime.now(UTC)
    v1 = _version(content_hash="h1", parent=None, cloud_skill_id="skill-1", created_at=base)
    v2 = _version(
        content_hash="h2",
        parent=v1.version_id,
        cloud_skill_id="skill-1",
        created_at=base + timedelta(seconds=10),
    )
    await repo.upsert_version(to_skill_version_point(v1))
    await repo.upsert_version(to_skill_version_point(v2))

    since = await repo.versions_since(PROJECT, "skill-1", since=base + timedelta(seconds=5))
    assert [r.payload["content_hash"] for r in since] == ["h2"]
    all_versions = await repo.versions_since(PROJECT, "skill-1")
    assert [r.payload["content_hash"] for r in all_versions] == ["h1", "h2"]


@pytest.mark.asyncio
async def test_list_versions_and_latest_version_are_project_scoped(repo):
    base = datetime.now(UTC)
    v1 = _version(content_hash="h1", parent=None, cloud_skill_id="skill-1", created_at=base)
    v2 = _version(
        content_hash="h2", parent=v1.version_id, cloud_skill_id="skill-1", created_at=base + timedelta(seconds=5)
    )
    other_project = SkillVersion(
        version_id=skill_version_id("other", "h3", None),
        project_id="other",
        cloud_skill_id="skill-2",
        skill_name="prd-writer",
        content_hash="h3",
        parent_version_id=None,
        version_label="1.0.0",
        status=SkillVersionStatus.OBSERVED,
        origin=SkillOrigin.EDGE,
        created_at=base + timedelta(seconds=10),
    )
    for version in (v1, v2, other_project):
        await repo.upsert_version(to_skill_version_point(version))

    records, cursor = await repo.list_versions(PROJECT)
    assert cursor is None
    assert [record.payload["content_hash"] for record in records] == ["h1", "h2"]

    latest = await repo.latest_version(PROJECT, "skill-1")
    assert latest is not None
    assert latest.payload["content_hash"] == "h2"
    assert await repo.latest_version(PROJECT, "skill-2") is None


@pytest.mark.asyncio
async def test_delete_versions_removes_metadata_only(repo):
    version = _version(content_hash="h1", parent=None, cloud_skill_id="skill-1")
    blob = SkillBlob(project_id=PROJECT, content_hash="h1", content="canonical-text", created_at=datetime.now(UTC))
    await repo.upsert_version(to_skill_version_point(version))
    await repo.upsert_blob(to_skill_blob_point(blob))

    await repo.delete_versions([version.version_id])

    assert await repo.get_version(PROJECT, version.version_id) is None
    assert await repo.get_blob(PROJECT, "h1") is not None


@pytest.mark.asyncio
async def test_iter_lineage_walks_parent_chain(repo):
    base = datetime.now(UTC)
    v1 = _version(content_hash="h1", parent=None, cloud_skill_id="skill-1", created_at=base)
    v2 = _version(content_hash="h2", parent=v1.version_id, cloud_skill_id="skill-1", created_at=base)
    v3 = _version(content_hash="h3", parent=v2.version_id, cloud_skill_id="skill-1", created_at=base)
    for v in (v1, v2, v3):
        await repo.upsert_version(to_skill_version_point(v))

    lineage = await repo.iter_lineage(PROJECT, v3.version_id)
    assert [r.payload["content_hash"] for r in lineage] == ["h3", "h2", "h1"]


@pytest.mark.asyncio
async def test_pending_trace_scroll_and_delete(repo):
    trace = SkillTracePending(
        trace_id="00000000-0000-0000-0000-0000000000aa",
        project_id=PROJECT,
        content_hash="h1",
        base_version_id="base-1",
        created_at=datetime.now(UTC),
    )
    other = SkillTracePending(
        trace_id="00000000-0000-0000-0000-0000000000bb",
        project_id=PROJECT,
        content_hash="h2",
        base_version_id="base-1",
        created_at=datetime.now(UTC),
    )
    await repo.add_pending_trace(to_skill_trace_pending_point(trace))
    await repo.add_pending_trace(to_skill_trace_pending_point(other))

    hits = await repo.scroll_pending_traces(PROJECT, "h1")
    assert [r.payload["trace_id"] for r in hits] == [trace.trace_id]

    await repo.delete_pending_traces([trace.trace_id])
    assert await repo.scroll_pending_traces(PROJECT, "h1") == []
