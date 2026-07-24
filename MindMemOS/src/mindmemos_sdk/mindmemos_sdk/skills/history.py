"""Local skill history metadata and content-cache storage."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from ..config import ConfigManager
from ..errors import SkillHistoryError
from .models import LocalSkillVersion, SkillHistoryEntry, SkillHistoryFile

HISTORY_FILE_NAME = "skill_history.json"


class SkillHistoryStore:
    """Manage ``skill_history.json`` and cached bundle content snapshots."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self._config_manager = config_manager
        self.history_path = config_manager.config_dir / HISTORY_FILE_NAME

    def load(self) -> SkillHistoryFile:
        """Load history from disk or return an empty history."""

        if not self.history_path.is_file():
            return SkillHistoryFile()
        try:
            return SkillHistoryFile.model_validate_json(self.history_path.read_text(encoding="utf-8"))
        except (ValueError, ValidationError) as exc:
            raise SkillHistoryError(f"invalid skill history at {self.history_path}: {exc}") from exc

    def save(self, history: SkillHistoryFile) -> None:
        """Atomically persist history metadata."""

        self._atomic_write_text(self.history_path, history.model_dump_json(indent=2) + "\n")

    def get(self, cloud_skill_id: str) -> SkillHistoryEntry | None:
        """Return one cloud skill history bucket."""

        return self.load().skills.get(cloud_skill_id)

    def upsert_versions(
        self,
        cloud_skill_id: str,
        *,
        skill_name: str,
        versions: list[LocalSkillVersion],
        last_pulled_at: str | None = None,
    ) -> SkillHistoryEntry:
        """Merge version metadata into a skill history bucket."""

        history = self.load()
        entry = history.skills.get(cloud_skill_id) or SkillHistoryEntry(skill_name=skill_name)
        entry.skill_name = skill_name
        if last_pulled_at is not None:
            entry.last_pulled_at = last_pulled_at

        by_id = {version.version_id: version for version in entry.versions}
        for version in versions:
            by_id[version.version_id] = version
        entry.versions = sorted(by_id.values(), key=lambda item: item.created_at)
        history.skills[cloud_skill_id] = entry
        self.save(history)
        return entry

    def remove(self, cloud_skill_id: str) -> SkillHistoryEntry | None:
        """Remove a skill history bucket."""

        history = self.load()
        removed = history.skills.pop(cloud_skill_id, None)
        if removed is not None:
            self.save(history)
        return removed

    def cache_path(self, content_hash: str) -> Path:
        """Return the local cache path for ``content_hash``."""

        config = self._config_manager.load_or_default()
        return Path(config.storage.skill_cache_dir).expanduser() / content_hash

    def write_cached_content(self, content_hash: str, content: str) -> Path:
        """Atomically write cached bundle content."""

        path = self.cache_path(content_hash)
        self._atomic_write_text(path, content)
        return path

    def read_cached_content(self, content_hash: str) -> str | None:
        """Read cached bundle content if present."""

        path = self.cache_path(content_hash)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
