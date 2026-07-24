"""Tests for the high-level SDK SkillManager."""

from __future__ import annotations

import pytest
from mindmemos_sdk.skills import SkillCloudClient, SkillContentData, SkillManager, SkillRecord, SkillRegisterData
from mindmemos_sdk.skills.bundle import compute_content_hash, deserialize_bundle
from mindmemos_sdk.skills.detector import detect_skill_context
from mindmemos_sdk.skills.history import SkillHistoryStore
from mindmemos_sdk.skills.installer import SkillInstaller
from mindmemos_sdk.skills.models import (
    HashState,
    SkillOrigin,
    SkillVersion,
    SkillVersionStatus,
)
from mindmemos_sdk.skills.pending import SkillPendingUploadStore
from mindmemos_sdk.skills.registry import SkillRegistry

from mindmemos_sdk.config import ConfigManager


class _FakeCloud(SkillCloudClient):
    def __init__(self) -> None:
        self.register_calls = []
        self.versions_since_calls = []
        self.get_content_calls = []
        self.get_skill_calls = []
        self.sync_calls = []
        self.delete_calls = []
        self.fail_next_register = False

    def register(self, **kwargs):
        self.register_calls.append(kwargs)
        if self.fail_next_register:
            self.fail_next_register = False
            raise RuntimeError("temporary outage")
        content_hash = compute_content_hash(deserialize_bundle(kwargs["content"]))
        return SkillRegisterData(
            cloud_skill_id="cloud-1",
            version_id=f"v{len(self.register_calls)}",
            version_label=kwargs.get("version_label"),
            content_hash=content_hash,
            status=SkillVersionStatus.OBSERVED,
        )

    def versions_since(self, cloud_skill_id: str, *, since: str | None = None, request_id: str | None = None):
        self.versions_since_calls.append((cloud_skill_id, since, request_id))
        content = '[{"content":"name: demo\\nversion: \\"1.1.0\\"\\n\\nCloud body\\n","path":"SKILL.md"}]'
        content_hash = compute_content_hash(deserialize_bundle(content))
        return [
            SkillVersion(
                version_id="v2",
                project_id="project-1",
                cloud_skill_id=cloud_skill_id,
                skill_name="demo",
                content_hash=content_hash,
                parent_version_id="v1",
                version_label="1.1.0",
                status=SkillVersionStatus.PUBLISHED,
                origin=SkillOrigin.CLOUD,
                created_at="2026-06-16T00:01:00Z",
            )
        ]

    def get_content(self, cloud_skill_id: str, version_id: str, *, request_id: str | None = None):
        self.get_content_calls.append((cloud_skill_id, version_id, request_id))
        content = '[{"content":"name: demo\\nversion: \\"1.1.0\\"\\n\\nCloud body\\n","path":"SKILL.md"}]'
        return SkillContentData(
            version=SkillVersion(
                version_id=version_id,
                project_id="project-1",
                cloud_skill_id=cloud_skill_id,
                skill_name="demo",
                content_hash=compute_content_hash(deserialize_bundle(content)),
                parent_version_id="v1",
                version_label="1.1.0",
                status=SkillVersionStatus.PUBLISHED,
                origin=SkillOrigin.CLOUD,
                created_at="2026-06-16T00:01:00Z",
            ),
            content=content,
        )

    def get_skill(self, cloud_skill_id: str, *, request_id: str | None = None):
        self.get_skill_calls.append((cloud_skill_id, request_id))
        version = self.get_content(cloud_skill_id, "v2").version
        from mindmemos_sdk.skills import SkillSummary

        return SkillSummary(
            cloud_skill_id=cloud_skill_id, skill_name="demo", latest_version=version, published_head=version
        )

    def sync(self, items, *, request_id: str | None = None):
        self.sync_calls.append((items, request_id))
        from mindmemos_sdk.skills import SkillSyncData, SkillSyncResult

        return SkillSyncData(
            results=[
                SkillSyncResult(
                    cloud_skill_id="cloud-1",
                    local_version_id="v1",
                    has_update=True,
                    published_head=self.get_content("cloud-1", "v2").version,
                    gating_status="published",
                )
            ]
        )

    def delete_skill(self, cloud_skill_id: str, *, request_id: str | None = None) -> None:
        self.delete_calls.append((cloud_skill_id, request_id))


def _manager(tmp_path):
    config_manager = ConfigManager(config_dir=tmp_path / "config")
    config = config_manager.load_or_default()
    config.storage.skill_cache_dir = str(tmp_path / "cache")
    config_manager.save(config)
    cloud = _FakeCloud()
    manager = SkillManager(
        registry=SkillRegistry(config_manager),
        history=SkillHistoryStore(config_manager),
        pending=SkillPendingUploadStore(config_manager),
        cloud=cloud,
        installer=SkillInstaller(config.storage.skill_backup_dir),
    )
    return manager, cloud, config_manager


def _skill_dir(tmp_path):
    path = tmp_path / "demo"
    path.mkdir()
    (path / "SKILL.md").write_text('name: demo\nversion: "1.0.0"\n\nBody\n', encoding="utf-8")
    return path


def test_register_uploads_and_persists_registry_history_and_cache(tmp_path):
    manager, cloud, config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)

    record = manager.register(str(path), name="override", version_label="2.0.0", alias="demo-main")

    assert isinstance(record, SkillRecord)
    assert record.alias == "demo-main"
    assert record.skill_name == "override"
    assert record.cloud_skill_id == "cloud-1"
    assert record.base_version_id == "v1"
    assert record.hash_state == HashState.CONFIRMED
    assert cloud.register_calls[0]["name"] == "override"
    assert cloud.register_calls[0]["version_label"] == "2.0.0"
    assert cloud.register_calls[0]["parent_version_id"] is None

    history = SkillHistoryStore(config_manager).get("cloud-1")
    assert history.versions[0].version_id == "v1"
    assert SkillHistoryStore(config_manager).read_cached_content(record.content_hash).startswith('[{"content"')


def test_register_accepts_skill_md_file_path_and_stores_directory(tmp_path):
    manager, _cloud, _config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)

    record = manager.register(str(path / "SKILL.md"), alias="demo-file")

    assert record.path == str(path.resolve())
    assert manager.show("demo-file").path == str(path.resolve())


def test_alias_can_reference_skill_commands(tmp_path):
    manager, _cloud, _config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path), alias="demo-main")

    assert manager.show("demo-main").skill_id == record.skill_id
    assert manager.history("demo-main")[0].version_id == "v1"
    assert manager.unregister("demo-main").skill_id == record.skill_id


def test_pull_and_unregister(tmp_path):
    manager, cloud, _config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path))

    pulled = manager.pull(record.skill_id)
    removed = manager.unregister(record.skill_id)

    assert pulled[0].version_id == "v2"
    assert cloud.versions_since_calls[0][0] == "cloud-1"
    assert removed.skill_id == record.skill_id
    assert cloud.delete_calls == [("cloud-1", None)]


def test_ensure_skill_context_enqueues_pending_snapshot(tmp_path):
    manager, cloud, config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path))
    (path / "SKILL.md").write_text('name: demo\nversion: "1.1.0"\n\nChanged\n', encoding="utf-8")

    context = manager.ensure_skill_context(record.skill_id, usage="modified")

    assert context.name == "demo"
    assert context.base_version_id == "v1"
    assert context.content_hash != record.content_hash
    assert context.usage == "modified"
    pending = SkillPendingUploadStore(config_manager).list()
    assert len(pending) == 1
    assert pending[0].parent_version_id == "v1"
    assert SkillHistoryStore(config_manager).read_cached_content(context.content_hash).startswith('[{"content"')
    updated = manager.show(record.skill_id)
    assert updated.hash_state == HashState.PENDING_UPLOAD
    assert cloud.register_calls == [cloud.register_calls[0]]


def test_flush_pending_upload_advances_registry_when_disk_matches(tmp_path):
    manager, cloud, config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path))
    (path / "SKILL.md").write_text('name: demo\nversion: "1.1.0"\n\nChanged\n', encoding="utf-8")
    context = manager.ensure_skill_context(record.skill_id)

    results = manager.flush_pending_uploads()

    assert results[0].uploaded is True
    assert results[0].version_id == "v2"
    assert results[0].registry_advanced is True
    assert SkillPendingUploadStore(config_manager).list() == []
    updated = manager.show(record.skill_id)
    assert updated.hash_state == HashState.CONFIRMED
    assert updated.content_hash == context.content_hash
    assert updated.base_version_id == "v2"
    assert cloud.register_calls[-1]["parent_version_id"] == context.base_version_id


def test_flush_pending_upload_does_not_advance_registry_after_newer_disk_change(tmp_path):
    manager, cloud, config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path))
    (path / "SKILL.md").write_text('name: demo\nversion: "1.1.0"\n\nChanged once\n', encoding="utf-8")
    context = manager.ensure_skill_context(record.skill_id)
    (path / "SKILL.md").write_text('name: demo\nversion: "1.2.0"\n\nChanged twice\n', encoding="utf-8")

    results = manager.flush_pending_uploads()

    assert results[0].uploaded is True
    assert results[0].registry_advanced is False
    assert SkillPendingUploadStore(config_manager).list() == []
    updated = manager.show(record.skill_id)
    assert updated.hash_state == HashState.PENDING_UPLOAD
    assert updated.content_hash == context.content_hash
    assert updated.base_version_id == "v1"
    history = SkillHistoryStore(config_manager).get("cloud-1")
    assert [version.version_id for version in history.versions] == ["v1", "v2"]
    assert history.versions[1].content_hash == context.content_hash


def test_flush_pending_upload_keeps_job_on_failure(tmp_path):
    manager, cloud, config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path))
    (path / "SKILL.md").write_text('name: demo\nversion: "1.1.0"\n\nChanged\n', encoding="utf-8")
    manager.ensure_skill_context(record.skill_id)
    cloud.fail_next_register = True

    results = manager.flush_pending_uploads()

    assert results[0].uploaded is False
    assert "temporary outage" in results[0].error
    pending = SkillPendingUploadStore(config_manager).list()
    assert len(pending) == 1
    assert pending[0].attempts == 1
    assert "temporary outage" in pending[0].last_error


def test_rollback_downloads_content_replaces_files_and_advances_registry(tmp_path):
    manager, cloud, _config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path))
    manager.pull(record.skill_id)

    plan, updated = manager.rollback(record.skill_id, version_id="v2")

    assert plan.from_version_id == "v1"
    assert plan.to_version_id == "v2"
    assert plan.backup_path is not None
    assert (path / "SKILL.md").read_text(encoding="utf-8").endswith("Cloud body\n")
    assert (path / "README.md").exists() is False
    assert updated.base_version_id == "v2"
    assert updated.content_hash == plan.to_content_hash
    assert updated.hash_state == HashState.CONFIRMED
    assert cloud.get_content_calls == [("cloud-1", "v2", None)]
    assert (path / "SKILL.md").read_text(encoding="utf-8") != "Body\n"


def test_diff_returns_unified_text_without_changing_files(tmp_path):
    manager, cloud, _config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path))
    manager.pull(record.skill_id)
    before = (path / "SKILL.md").read_text(encoding="utf-8")

    result = manager.diff(record.skill_id, from_version_id="v1", to_version_id="v2")

    assert result.from_version_id == "v1"
    assert result.to_version_id == "v2"
    assert "--- v1/SKILL.md" in result.diff
    assert "+++ v2/SKILL.md" in result.diff
    assert "+Cloud body" in result.diff
    assert (path / "SKILL.md").read_text(encoding="utf-8") == before
    assert cloud.get_content_calls == [("cloud-1", "v2", None)]


def test_update_checks_published_head_and_applies_checkout(tmp_path):
    manager, cloud, _config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path))

    result = manager.update(record.skill_id)

    assert result.had_update is True
    assert result.plan.to_version_id == "v2"
    assert result.record.base_version_id == "v2"
    assert (path / "SKILL.md").read_text(encoding="utf-8").endswith("Cloud body\n")
    assert cloud.get_skill_calls == [("cloud-1", None)]


def test_update_rejects_unuploaded_local_changes(tmp_path):
    manager, _cloud, _config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path))
    (path / "SKILL.md").write_text('name: demo\nversion: "1.1.0"\n\nLocal edit\n', encoding="utf-8")

    with pytest.raises(Exception) as exc:
        manager.plan_update(record.skill_id)

    assert "skill push" in str(exc.value)


def test_push_uploads_local_changes_as_new_version(tmp_path):
    manager, cloud, _config_manager = _manager(tmp_path)
    path = _skill_dir(tmp_path)
    record = manager.register(str(path))
    (path / "SKILL.md").write_text('name: demo\nversion: "1.1.0"\n\nLocal edit\n', encoding="utf-8")

    pushed = manager.push(record.skill_id)

    assert pushed.base_version_id == "v2"
    assert pushed.content_hash != record.content_hash
    assert pushed.hash_state == HashState.CONFIRMED
    assert cloud.register_calls[-1]["parent_version_id"] == "v1"


def test_detect_skill_context_from_openclaw_tool_messages(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    registry = SkillRegistry(ConfigManager(config_dir=tmp_path / "config"))
    record = registry.upsert(
        SkillRecord(
            path=str(skill_dir),
            skill_name="demo",
            cloud_skill_id="cloud-1",
            base_version_id="v1",
            content_hash="hash-1",
            hash_state=HashState.CONFIRMED,
        )
    )
    content = 'name: demo\nversion: "1.2.0"\n\nLoaded\n'
    messages = [
        {"role": "assistant", "content": f'[tool_call] read({{"path": "{skill_dir / "SKILL.md"}"}})', "timestamp": 1},
        {"role": "tool", "content": content, "timestamp": 2},
    ]

    contexts = detect_skill_context(messages, registry=registry)

    assert len(contexts) == 1
    assert contexts[0].name == "demo"
    assert contexts[0].base_version_id == record.base_version_id
    assert contexts[0].content_hash == compute_content_hash({"SKILL.md": content})
    assert contexts[0].usage == "injected"
