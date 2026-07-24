"""Typed access to the SDK skill registry stored in ``settings.json``."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from ..config import ConfigManager
from ..errors import SkillRegistryError
from .models import SkillRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _skill_id_for_path(path: str) -> str:
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
    return f"sk_{digest}"


_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _normalize_alias(alias: str | None) -> str | None:
    if alias is None:
        return None
    normalized = alias.strip()
    if not normalized:
        return None
    if not _ALIAS_PATTERN.fullmatch(normalized):
        raise SkillRegistryError(
            "skill alias must be 1-64 characters and contain only letters, numbers, '.', '_', or '-'"
        )
    return normalized


class SkillRegistry:
    """Manage typed ``SkillRecord`` entries in ``SDKConfig.skills``."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self._config_manager = config_manager

    def list(self) -> list[SkillRecord]:
        """Return all registered skills sorted by name and path."""

        return sorted(self._load_records(), key=lambda item: (item.skill_name.lower(), item.path))

    def get_by_path(self, path: str) -> SkillRecord | None:
        """Return the registered skill at ``path`` if present."""

        target = _canonical_path(path)
        return next((record for record in self._load_records() if record.path == target), None)

    def get_by_cloud_id(self, cloud_skill_id: str) -> SkillRecord | None:
        """Return the registered skill with ``cloud_skill_id`` if present."""

        return next((record for record in self._load_records() if record.cloud_skill_id == cloud_skill_id), None)

    def get_by_skill_id(self, skill_id: str) -> SkillRecord | None:
        """Return the registered skill with local ``skill_id`` if present."""

        return next((record for record in self._load_records() if record.skill_id == skill_id), None)

    def get_by_alias(self, alias: str) -> SkillRecord | None:
        """Return the registered skill with local ``alias`` if present."""

        normalized = _normalize_alias(alias)
        if normalized is None:
            return None
        return next((record for record in self._load_records() if record.alias == normalized), None)

    def get_by_ref(self, skill_ref: str) -> SkillRecord | None:
        """Return a registered skill by local id or alias."""

        return self.get_by_skill_id(skill_ref) or self.get_by_alias(skill_ref)

    def ensure_alias_available(
        self, alias: str | None, *, skill_id: str | None = None, path: str | None = None
    ) -> str | None:
        """Validate ``alias`` and ensure no other registered skill owns it."""

        normalized = _normalize_alias(alias)
        if normalized is None:
            return None
        target_path = _canonical_path(path) if path else None
        for existing in self._load_records():
            same_record = (skill_id is not None and existing.skill_id == skill_id) or (
                target_path is not None and existing.path == target_path
            )
            if existing.alias == normalized and not same_record:
                raise SkillRegistryError(f"skill alias already exists: {normalized}")
            if existing.skill_id == normalized and not same_record:
                raise SkillRegistryError(f"skill alias conflicts with an existing skill id: {normalized}")
        return normalized

    def upsert(self, record: SkillRecord) -> SkillRecord:
        """Insert or replace one registry record and persist ``settings.json``."""

        config = self._config_manager.load_or_default()
        records = self._parse_records(config.skills)
        now = _utc_now_iso()
        canonical = _canonical_path(record.path)
        alias = _normalize_alias(record.alias)
        normalized = record.model_copy(
            update={
                "skill_id": record.skill_id or _skill_id_for_path(canonical),
                "alias": alias,
                "path": canonical,
                "registered_at": record.registered_at or now,
                "updated_at": now,
            }
        )

        self.ensure_alias_available(alias, skill_id=normalized.skill_id, path=canonical)
        for existing in records:
            if alias and normalized.skill_id == existing.alias and existing.skill_id != normalized.skill_id:
                raise SkillRegistryError(f"skill id conflicts with an existing skill alias: {normalized.skill_id}")

        replaced = False
        next_records: list[SkillRecord] = []
        for existing in records:
            same_path = existing.path == normalized.path
            same_cloud = normalized.cloud_skill_id and existing.cloud_skill_id == normalized.cloud_skill_id
            same_local = existing.skill_id == normalized.skill_id
            if same_path or same_cloud or same_local:
                if existing.registered_at and record.registered_at is None:
                    normalized = normalized.model_copy(update={"registered_at": existing.registered_at})
                if existing.skill_id and not record.skill_id:
                    normalized = normalized.model_copy(update={"skill_id": existing.skill_id})
                if existing.alias and record.alias is None:
                    normalized = normalized.model_copy(update={"alias": existing.alias})
                if existing.cloud_skill_id and normalized.cloud_skill_id is None:
                    normalized = normalized.model_copy(update={"cloud_skill_id": existing.cloud_skill_id})
                if not replaced:
                    next_records.append(normalized)
                    replaced = True
                continue
            next_records.append(existing)

        if not replaced:
            next_records.append(normalized)

        config.skills = [item.model_dump(mode="json") for item in next_records]
        self._config_manager.save(config)
        return normalized

    def remove(
        self,
        *,
        skill_id: str | None = None,
        path: str | None = None,
        cloud_skill_id: str | None = None,
    ) -> SkillRecord | None:
        """Remove one registry record by local id, path, or cloud id."""

        if skill_id is None and path is None and cloud_skill_id is None:
            raise SkillRegistryError("provide skill_id, path, or cloud_skill_id to remove a skill")

        target_path = _canonical_path(path) if path else None
        config = self._config_manager.load_or_default()
        records = self._parse_records(config.skills)
        removed: SkillRecord | None = None
        kept: list[SkillRecord] = []

        for record in records:
            matches = (
                (skill_id is not None and record.skill_id == skill_id)
                or (target_path is not None and record.path == target_path)
                or (cloud_skill_id is not None and record.cloud_skill_id == cloud_skill_id)
            )
            if matches and removed is None:
                removed = record
                continue
            kept.append(record)

        if removed is not None:
            config.skills = [item.model_dump(mode="json") for item in kept]
            self._config_manager.save(config)
        return removed

    def _load_records(self) -> list[SkillRecord]:
        config = self._config_manager.load_or_default()
        return self._parse_records(config.skills)

    @staticmethod
    def _parse_records(raw: list[dict]) -> list[SkillRecord]:
        records: list[SkillRecord] = []
        for item in raw:
            try:
                record = SkillRecord.model_validate(item)
            except ValidationError as exc:
                raise SkillRegistryError(f"invalid skill registry record: {exc}") from exc
            record = record.model_copy(
                update={"path": _canonical_path(record.path), "alias": _normalize_alias(record.alias)}
            )
            records.append(record)
        return records
