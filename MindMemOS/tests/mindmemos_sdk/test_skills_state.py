"""Tests for SDK skill state, bundle hashing, registry, and history storage."""

from __future__ import annotations

import json

import pytest
from mindmemos.components.skill import bundle as server_bundle
from mindmemos_sdk.errors import SkillBundleError, SkillRegistryError
from mindmemos_sdk.skills import bundle as sdk_bundle
from mindmemos_sdk.skills.history import SkillHistoryStore
from mindmemos_sdk.skills.models import (
    HashState,
    LocalSkillVersion,
    SkillOrigin,
    SkillRecord,
    SkillVersionStatus,
)
from mindmemos_sdk.skills.registry import SkillRegistry

from mindmemos_sdk.config import ConfigManager


def test_bundle_hash_matches_server_for_canonical_cases():
    cases = [
        {"SKILL.md": "name: demo\n---\nUse me.\n"},
        {"nested/SKILL.md": "line1\r\nline2\r", "ignored.txt": "nope"},
        {"C:\\skills\\demo\\SKILL.md": "中文内容\n"},
    ]

    for files in cases:
        assert sdk_bundle.normalize_bundle(files) == server_bundle.normalize_bundle(files)
        assert sdk_bundle.serialize_bundle(files) == server_bundle.serialize_bundle(files)
        assert sdk_bundle.compute_content_hash(files) == server_bundle.compute_content_hash(files)


def test_bundle_serialize_deserialize_and_bare_content():
    serialized = sdk_bundle.serialize_bundle({"skill/SKILL.md": "hello\r\n", "README.md": "ignored"})

    assert serialized == '[{"content":"hello\\n","path":"SKILL.md"}]'
    assert sdk_bundle.deserialize_bundle(serialized) == {"SKILL.md": "hello\n"}
    assert sdk_bundle.bundle_files_from_content("plain skill body") == {"SKILL.md": "plain skill body"}


def test_bundle_requires_skill_md(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "README.md").write_text("ignored", encoding="utf-8")

    with pytest.raises(SkillBundleError):
        sdk_bundle.read_local_bundle(skill_dir)


def test_read_local_bundle_reads_only_skill_md(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("line1\r\nline2\r", encoding="utf-8")
    (skill_dir / "README.md").write_text("ignored", encoding="utf-8")

    assert sdk_bundle.read_local_bundle(skill_dir) == {"SKILL.md": "line1\nline2\n"}
    assert sdk_bundle.read_local_bundle(skill_dir / "SKILL.md") == {"SKILL.md": "line1\nline2\n"}
    assert sdk_bundle.resolve_skill_dir(skill_dir / "SKILL.md") == skill_dir


def test_read_local_bundle_rejects_non_skill_file(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("ignored", encoding="utf-8")

    with pytest.raises(SkillBundleError, match="not whitelisted"):
        sdk_bundle.read_local_bundle(path)


def test_registry_upsert_list_lookup_and_remove(tmp_path):
    manager = ConfigManager(config_dir=tmp_path / "config")
    registry = SkillRegistry(manager)
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()

    record = registry.upsert(
        SkillRecord(
            path=str(skill_dir),
            alias="demo-local",
            skill_name="demo",
            cloud_skill_id="cloud-1",
            base_version_id="v1",
            content_hash="hash-1",
            hash_state=HashState.CONFIRMED,
            version_label="1.0.0",
        )
    )

    assert record.skill_id.startswith("sk_")
    assert record.path == str(skill_dir.resolve())
    assert record.registered_at is not None
    assert record.updated_at is not None
    assert registry.get_by_path(str(skill_dir)) == record
    assert registry.get_by_cloud_id("cloud-1") == record
    assert registry.get_by_skill_id(record.skill_id) == record
    assert registry.get_by_alias("demo-local") == record
    assert registry.get_by_ref("demo-local") == record
    assert registry.list() == [record]

    updated = registry.upsert(record.model_copy(update={"skill_name": "renamed", "content_hash": "hash-2"}))
    assert updated.skill_id == record.skill_id
    assert updated.registered_at == record.registered_at
    assert updated.skill_name == "renamed"
    assert updated.content_hash == "hash-2"

    raw = json.loads(manager.config_path.read_text(encoding="utf-8"))
    assert raw["skills"][0]["hash_state"] == "confirmed"
    assert raw["skills"][0]["alias"] == "demo-local"
    assert registry.remove(skill_id=record.skill_id).skill_name == "renamed"
    assert registry.list() == []


def test_registry_rejects_duplicate_alias(tmp_path):
    manager = ConfigManager(config_dir=tmp_path / "config")
    registry = SkillRegistry(manager)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    registry.upsert(SkillRecord(path=str(first), alias="demo", skill_name="first"))

    with pytest.raises(SkillRegistryError, match="alias already exists"):
        registry.upsert(SkillRecord(path=str(second), alias="demo", skill_name="second"))


def test_registry_rejects_invalid_records(tmp_path):
    manager = ConfigManager(config_dir=tmp_path / "config")
    manager.save(manager.load_or_default().model_copy(update={"skills": [{"path": "/tmp/missing-name"}]}))

    with pytest.raises(SkillRegistryError):
        SkillRegistry(manager).list()


def test_history_merges_versions_and_caches_content(tmp_path):
    manager = ConfigManager(config_dir=tmp_path / "config")
    config = manager.load_or_default()
    config.storage.skill_cache_dir = str(tmp_path / "cache")
    manager.save(config)
    store = SkillHistoryStore(manager)

    v1 = LocalSkillVersion(
        version_id="v1",
        status=SkillVersionStatus.PUBLISHED,
        origin=SkillOrigin.EDGE,
        content_hash="hash-1",
        created_at="2026-06-16T00:00:00Z",
    )
    v2 = LocalSkillVersion(
        version_id="v2",
        parent_version_id="v1",
        status=SkillVersionStatus.PUBLISHED,
        origin=SkillOrigin.CLOUD,
        content_hash="hash-2",
        created_at="2026-06-16T00:01:00Z",
    )

    entry = store.upsert_versions("cloud-1", skill_name="demo", versions=[v1], last_pulled_at="t1")
    assert entry.last_pulled_at == "t1"
    entry = store.upsert_versions("cloud-1", skill_name="demo", versions=[v2, v1], last_pulled_at="t2")

    assert [version.version_id for version in entry.versions] == ["v1", "v2"]
    assert entry.last_pulled_at == "t2"
    assert store.get("cloud-1").versions[1].parent_version_id == "v1"

    path = store.write_cached_content("hash-2", "bundle content")
    assert path == tmp_path / "cache" / "hash-2"
    assert store.read_cached_content("hash-2") == "bundle content"
    assert store.read_cached_content("missing") is None
    assert store.remove("cloud-1") is not None
    assert store.get("cloud-1") is None
