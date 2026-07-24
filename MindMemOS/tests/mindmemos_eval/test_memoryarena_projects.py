"""Tests for the per-episode MemoryArena project/key pool helpers."""

from __future__ import annotations

import yaml

from mindmemos_eval import (
    generate_project_keys,
    merge_keys_into_file,
    remove_keys_from_file,
)


def _load(path) -> list[dict]:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("api_keys") or []


def test_generate_project_keys_is_deterministic_and_distinct():
    keys = generate_project_keys("run1", "math", 3)
    assert len(keys) == 3
    assert [k.project_id for k in keys] == [
        "proj-memoryarena-math-run1-0",
        "proj-memoryarena-math-run1-1",
        "proj-memoryarena-math-run1-2",
    ]
    # Each episode maps to a distinct api key -> distinct project (no pollution).
    assert len({k.api_key for k in keys}) == 3


def test_merge_is_idempotent_and_preserves_existing(tmp_path):
    path = tmp_path / "api_keys.memoryarena.yaml"
    path.write_text(
        yaml.safe_dump(
            {"api_keys": [{"key_id": "key_dev", "api_key": "sk-dev", "project_id": "proj_dev", "enabled": True}]}
        ),
        encoding="utf-8",
    )

    keys = generate_project_keys("run1", "math", 2)
    merge_keys_into_file(path, keys)

    entries = _load(path)
    key_ids = [e["key_id"] for e in entries]
    assert key_ids == ["key_dev", "memoryarena-math-run1-0", "memoryarena-math-run1-1"]
    generated = next(e for e in entries if e["key_id"] == "memoryarena-math-run1-0")
    assert generated["scopes"] == ["memory:read", "memory:write"]


def test_remove_keys_only_drops_matching_run(tmp_path):
    path = tmp_path / "api_keys.memoryarena.yaml"
    merge_keys_into_file(path, generate_project_keys("runA", "math", 2))
    merge_keys_into_file(path, generate_project_keys("runB", "phys", 1))

    removed = remove_keys_from_file(path, "runA")
    assert removed == 2
    remaining = [e["key_id"] for e in _load(path)]
    assert remaining == ["memoryarena-phys-runB-0"]


def test_remove_keys_missing_file_is_noop(tmp_path):
    assert remove_keys_from_file(tmp_path / "does_not_exist.yaml", "run1") == 0
