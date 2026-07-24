"""Local persistent outbox for skill uploads.

The queue stores metadata only. Bundle snapshots live in the content-addressed
skill cache, keyed by ``content_hash``, so retries upload the same bytes that
were referenced by ``/v1/memory/add`` even if the local files change later.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from ..config import ConfigManager
from ..errors import SkillPendingUploadError
from .models import SkillPendingUpload, SkillPendingUploadsFile

PENDING_UPLOADS_FILE_NAME = "skill_pending_uploads.json"


class SkillPendingUploadStore:
    """Manage ``skill_pending_uploads.json`` beside SDK settings/history."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self.pending_path = config_manager.config_dir / PENDING_UPLOADS_FILE_NAME

    def load(self) -> SkillPendingUploadsFile:
        """Load pending uploads from disk or return an empty queue."""

        if not self.pending_path.is_file():
            return SkillPendingUploadsFile()
        try:
            return SkillPendingUploadsFile.model_validate_json(self.pending_path.read_text(encoding="utf-8"))
        except (ValueError, ValidationError) as exc:
            raise SkillPendingUploadError(f"invalid skill pending uploads at {self.pending_path}: {exc}") from exc

    def save(self, pending: SkillPendingUploadsFile) -> None:
        """Atomically persist pending upload metadata."""

        self._atomic_write_text(self.pending_path, pending.model_dump_json(indent=2) + "\n")

    def list(self) -> list[SkillPendingUpload]:
        """Return pending uploads sorted by creation time."""

        return sorted(self.load().uploads.values(), key=lambda item: item.created_at)

    def upsert(self, upload: SkillPendingUpload) -> SkillPendingUpload:
        """Insert or replace one pending upload by deterministic job id."""

        pending = self.load()
        pending.uploads[upload.job_id] = upload
        self.save(pending)
        return upload

    def remove(self, job_id: str) -> SkillPendingUpload | None:
        """Remove one pending upload."""

        pending = self.load()
        removed = pending.uploads.pop(job_id, None)
        if removed is not None:
            self.save(pending)
        return removed

    def mark_failed(self, job_id: str, *, error: str, updated_at: str) -> SkillPendingUpload | None:
        """Record a failed retry without removing the upload."""

        pending = self.load()
        upload = pending.uploads.get(job_id)
        if upload is None:
            return None
        updated = upload.model_copy(
            update={
                "attempts": upload.attempts + 1,
                "last_error": error,
                "updated_at": updated_at,
                "next_retry_at": updated_at,
            }
        )
        pending.uploads[job_id] = updated
        self.save(pending)
        return updated

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
