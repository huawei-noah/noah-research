"""Per-episode API key / project pool for isolated MemoryArena evaluation.

MindMemOS scopes every memory by ``project_id``, and ``project_id`` is bound to
the API key (resolved server-side from the api-key YAML table). MemoryArena
episodes must not pollute each other's retrieval, so each episode is given its
own throwaway ``project_id`` by minting a dedicated API key.

These helpers generate a run-scoped key pool and merge/remove it from the
api-key file the running server reads (``ApiKeyAuthProvider`` re-reads the file
on every request, so appended keys are hot-loaded without a restart).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_SCOPES = ("memory:read", "memory:write")


@dataclass(frozen=True)
class ProjectKey:
    """One throwaway api-key row mapping to an isolated project.

    Mirrors the ``config/mindmemos/api_keys.yaml`` schema so it can be serialized straight
    into the file the server's ``ApiKeyAuthProvider`` loads.
    """

    key_id: str
    api_key: str
    project_id: str
    scopes: list[str] = field(default_factory=lambda: list(DEFAULT_SCOPES))
    enabled: bool = True

    def to_entry(self) -> dict[str, object]:
        """Render this key as one ``api_keys`` YAML entry."""
        return {
            "key_id": self.key_id,
            "api_key": self.api_key,
            "project_id": self.project_id,
            "enabled": self.enabled,
            "scopes": list(self.scopes),
        }


def generate_project_keys(
    run_id: str,
    env_name: str,
    count: int,
    *,
    prefix: str = "memoryarena",
) -> list[ProjectKey]:
    """Deterministically mint ``count`` run-scoped (api_key, project_id) pairs.

    Args:
        run_id: Stable per-run suffix isolating this run's keys from others.
        env_name: MemoryArena env name (``math`` / ``phys``), embedded in the
            project id for readability.
        count: Number of episodes; one key/project is minted per episode.
        prefix: Namespace prefix for key_id / api_key / project_id.

    Returns:
        A list of ``count`` :class:`ProjectKey`, indexable by episode index.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    keys: list[ProjectKey] = []
    for idx in range(count):
        keys.append(
            ProjectKey(
                key_id=f"{prefix}-{env_name}-{run_id}-{idx}",
                api_key=f"sk-{prefix}-{run_id}-{idx}",
                project_id=f"proj-{prefix}-{env_name}-{run_id}-{idx}",
            )
        )
    return keys


def _read_entries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("api_keys") or []
    return list(entries)


def _atomic_write_entries(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump({"api_keys": entries}, allow_unicode=True, sort_keys=False)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def merge_keys_into_file(path: str | Path, keys: list[ProjectKey]) -> None:
    """Idempotently upsert ``keys`` into the api-key YAML file (atomic write).

    Existing entries are keyed by ``key_id``; matching keys are replaced so the
    operation is safe to retry. The file is rewritten atomically via a temp file
    + ``os.replace`` to avoid the server reading a half-written table.
    """
    target = Path(path)
    existing = _read_entries(target)
    by_id: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for entry in existing:
        key_id = str(entry.get("key_id") or "")
        if key_id and key_id not in by_id:
            order.append(key_id)
        by_id[key_id] = entry
    for key in keys:
        if key.key_id not in by_id:
            order.append(key.key_id)
        by_id[key.key_id] = key.to_entry()
    _atomic_write_entries(target, [by_id[key_id] for key_id in order])


def remove_keys_from_file(path: str | Path, run_id: str) -> int:
    """Remove run-scoped temp keys (key_id containing ``-{run_id}-``).

    Returns:
        The number of removed entries. Missing file is treated as zero.
    """
    target = Path(path)
    existing = _read_entries(target)
    if not existing:
        return 0
    marker = f"-{run_id}-"
    kept = [entry for entry in existing if marker not in str(entry.get("key_id") or "")]
    removed = len(existing) - len(kept)
    if removed:
        _atomic_write_entries(target, kept)
    return removed
