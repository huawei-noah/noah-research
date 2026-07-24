"""Tests for the SDK ConfigManager and auth persistence."""

from __future__ import annotations

import json

import pytest
from mindmemos_sdk.errors import ConfigNotFoundError, ConfigValidationError

from mindmemos_sdk.config import ConfigManager, mask_secret


@pytest.fixture
def manager(tmp_path):
    return ConfigManager(config_dir=tmp_path)


def test_load_missing_raises(manager):
    assert manager.exists() is False
    with pytest.raises(ConfigNotFoundError):
        manager.load()


def test_update_auth_persists_and_roundtrips(manager):
    manager.update_auth(
        base_url="https://my.example.com",
        api_key="mk_secret",
        user_id="u_1",
    )

    assert manager.exists() is True
    loaded = manager.load()
    assert loaded.base_url == "https://my.example.com"
    assert loaded.auth.api_key == "mk_secret"
    assert loaded.defaults.user_id == "u_1"
    assert loaded.version == 1
    assert loaded.metadata.created_at is not None
    assert loaded.metadata.updated_at is not None


def test_update_auth_drops_legacy_actor_defaults(manager):
    manager.config_dir.mkdir(parents=True, exist_ok=True)
    manager.config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "base_url": "https://old.example.com",
                "auth": {"api_key": "old"},
                "defaults": {
                    "user_id": "old-user",
                    "app_id": "legacy-app",
                    "agent_id": "legacy-agent",
                    "session_id": "legacy-session",
                },
                "skills": [],
            }
        )
    )

    updated = manager.update_auth(base_url="https://new.example.com", api_key="new", user_id="new-user")
    raw = json.loads(manager.config_path.read_text())

    assert updated.defaults.user_id == "new-user"
    assert raw["defaults"] == {"user_id": "new-user"}


def test_update_auth_preserves_skills_and_created_at(manager):
    first = manager.update_auth(base_url="https://a", api_key="k1", user_id="u_1")
    created_at = first.metadata.created_at

    # Inject a skill record out-of-band to prove update_auth does not clobber it.
    raw = json.loads(manager.config_path.read_text())
    raw["skills"] = [{"id": "skill_1", "name": "demo"}]
    manager.config_path.write_text(json.dumps(raw))

    second = manager.update_auth(base_url="https://b", api_key="k2", user_id="u_2")
    assert second.base_url == "https://b"
    assert second.auth.api_key == "k2"
    assert second.skills == [{"id": "skill_1", "name": "demo"}]
    assert second.metadata.created_at == created_at


def test_reset_removes_file(manager):
    manager.update_auth(base_url="https://a", api_key="k", user_id="u")
    assert manager.reset() is True
    assert manager.exists() is False
    assert manager.reset() is False


def test_load_invalid_json_raises(manager):
    manager.config_dir.mkdir(parents=True, exist_ok=True)
    manager.config_path.write_text("{not json")
    with pytest.raises(ConfigValidationError):
        manager.load()


def test_save_is_atomic_no_leftover_temp(manager):
    manager.update_auth(base_url="https://a", api_key="k", user_id="u")
    leftovers = list(manager.config_dir.glob(".settings-*.tmp"))
    assert leftovers == []


@pytest.mark.parametrize(
    ("secret", "expected"),
    [
        (None, "(not set)"),
        ("", "(not set)"),
        ("abc", "***"),
        ("mk_abcdef", "*****cdef"),
    ],
)
def test_mask_secret(secret, expected):
    assert mask_secret(secret) == expected
