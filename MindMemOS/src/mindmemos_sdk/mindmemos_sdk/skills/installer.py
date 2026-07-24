"""Local skill file replacement with backups and rollback on failure."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..errors import SkillInstallerError
from .bundle import bundle_files_from_content, compute_content_hash
from .models import SkillCheckoutPlan, SkillRecord


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class SkillInstaller:
    """Apply managed skill bundle snapshots to local skill directories."""

    def __init__(self, backup_root: str | os.PathLike[str]) -> None:
        self._backup_root = Path(backup_root).expanduser()

    def plan_checkout(
        self,
        record: SkillRecord,
        *,
        target_version_id: str,
        content: str,
    ) -> SkillCheckoutPlan:
        """Build a checkout plan without changing local files."""

        files = bundle_files_from_content(content)
        to_hash = compute_content_hash(files)
        return SkillCheckoutPlan(
            skill_id=record.skill_id,
            path=record.path,
            from_version_id=record.base_version_id,
            to_version_id=target_version_id,
            from_content_hash=record.content_hash,
            to_content_hash=to_hash,
            files=sorted(files),
            backup_path=str(self._backup_path(record)),
        )

    def apply_checkout(self, plan: SkillCheckoutPlan, *, content: str) -> SkillCheckoutPlan:
        """Back up current managed files and atomically replace them with ``content``."""

        files = bundle_files_from_content(content)
        to_hash = compute_content_hash(files)
        if to_hash != plan.to_content_hash:
            raise SkillInstallerError(f"checkout content hash mismatch: expected {plan.to_content_hash}, got {to_hash}")

        root = Path(plan.path).expanduser()
        if not root.is_dir():
            raise SkillInstallerError(f"skill directory does not exist: {root}")

        backup_path = Path(plan.backup_path) if plan.backup_path else self._fallback_backup_path(plan)
        backup_path.mkdir(parents=True, exist_ok=False)
        temp_dir = Path(tempfile.mkdtemp(dir=root.parent, prefix=f".{root.name}-checkout-"))

        try:
            self._backup_current_files(root, backup_path, files)
            for relative_path, text in files.items():
                target = root / relative_path
                temp_target = temp_dir / relative_path
                temp_target.parent.mkdir(parents=True, exist_ok=True)
                temp_target.write_text(text, encoding="utf-8")
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp_target, target)
        except BaseException as exc:
            self._restore_backup(root, backup_path, files)
            raise SkillInstallerError(f"failed to checkout skill version {plan.to_version_id}: {exc}") from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return plan.model_copy(update={"backup_path": str(backup_path)})

    def _backup_path(self, record: SkillRecord) -> Path:
        safe_name = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in record.skill_name)
        return self._backup_root / safe_name / _utc_timestamp()

    def _fallback_backup_path(self, plan: SkillCheckoutPlan) -> Path:
        return self._backup_root / plan.skill_id / _utc_timestamp()

    @staticmethod
    def _backup_current_files(root: Path, backup_path: Path, files: dict[str, str]) -> None:
        for relative_path in files:
            source = root / relative_path
            if source.is_file():
                target = backup_path / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    @staticmethod
    def _restore_backup(root: Path, backup_path: Path, files: dict[str, str]) -> None:
        for relative_path in files:
            backup = backup_path / relative_path
            target = root / relative_path
            if backup.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, target)
            else:
                with contextlib.suppress(FileNotFoundError):
                    target.unlink()
