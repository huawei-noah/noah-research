"""Config-file backed API key authentication provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import stat_result
from pathlib import Path
from typing import Any

import yaml

from ...errors import AuthenticationError
from ...logging import get_logger, traced
from ..algorithm import resolve_memory_algorithm
from .base import ResolvedIdentity

logger = get_logger(__name__)


@dataclass(frozen=True)
class ApiKeyEntry:
    """One API key row loaded from ``config/mindmemos/api_keys.yaml``."""

    key_id: str
    api_key: str
    project_id: str
    memory_algorithm: str
    enabled: bool = True
    scopes: list[str] = field(default_factory=list)
    user_override_config: dict[str, Any] | None = None
    project_override_config: dict[str, Any] | None = None


class ApiKeyAuthProvider:
    """Resolve bearer API keys from a local YAML key table."""

    def __init__(self, *, api_key_file: str | Path):
        self._api_key_file = _resolve_path(api_key_file)
        self._entries: dict[str, ApiKeyEntry] = {}
        self._signature: tuple[int, int] | None = None
        self._reload(allow_missing=True)

    @traced("auth.api_key.resolve", record_args=False, record_result=False)
    def resolve_api_key(self, api_key: str) -> ResolvedIdentity:
        self._reload_if_changed()
        entry = self._entries.get(api_key)
        if entry is None:
            raise AuthenticationError("invalid api key", code="auth.invalid_api_key")
        if not entry.enabled:
            raise AuthenticationError("api key is disabled", code="auth.api_key_disabled")
        return ResolvedIdentity(
            key_id=entry.key_id,
            project_id=entry.project_id,
            memory_algorithm=entry.memory_algorithm,
            scopes=list(entry.scopes),
            user_override_config=entry.user_override_config,
            project_override_config=entry.project_override_config,
        )

    def _reload_if_changed(self) -> None:
        signature = _file_signature(self._api_key_file)
        if signature == self._signature:
            return
        self._reload(allow_missing=True)

    def _reload(self, *, allow_missing: bool) -> None:
        if not self._api_key_file.exists():
            if not allow_missing:
                raise AuthenticationError("api key config file not found", code="auth.api_key_file_not_found")
            self._entries = {}
            self._signature = None
            return
        entries = _load_entries(self._api_key_file)
        self._entries = entries
        self._signature = _file_signature(self._api_key_file)


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat: stat_result = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


@traced("auth.api_key.load_entries", record_args=False, record_result=False)
def _load_entries(path: Path) -> dict[str, ApiKeyEntry]:
    if not path.exists():
        raise AuthenticationError("api key config file not found", code="auth.api_key_file_not_found")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    entries: dict[str, ApiKeyEntry] = {}
    for raw in data.get("api_keys") or []:
        entry = ApiKeyEntry(
            key_id=str(raw.get("key_id") or ""),
            api_key=str(raw.get("api_key") or ""),
            project_id=str(raw.get("project_id") or ""),
            memory_algorithm=str(raw.get("memory_algorithm") or ""),
            enabled=bool(raw.get("enabled", True)),
            scopes=list(raw.get("scopes") or []),
            user_override_config=_optional_mapping(raw.get("user_override_config"), field_name="user_override_config"),
            project_override_config=_optional_mapping(
                raw.get("project_override_config"),
                field_name="project_override_config",
            ),
        )
        if not entry.key_id or not entry.api_key or not entry.project_id:
            logger.warning("invalid api key entry ignored", api_key_file=str(path), key_id=entry.key_id)
            continue
        entry = ApiKeyEntry(
            key_id=entry.key_id,
            api_key=entry.api_key,
            project_id=entry.project_id,
            memory_algorithm=resolve_memory_algorithm(entry.memory_algorithm),
            enabled=entry.enabled,
            scopes=entry.scopes,
            user_override_config=entry.user_override_config,
            project_override_config=entry.project_override_config,
        )
        entries[entry.api_key] = entry
    return entries


def _optional_mapping(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AuthenticationError(f"{field_name} must be a mapping", code="auth.invalid_override_config")
    return dict(value)
