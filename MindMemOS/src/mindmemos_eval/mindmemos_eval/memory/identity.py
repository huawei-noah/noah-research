"""Run identity and API-key file helpers."""

from __future__ import annotations

import os
import secrets
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCOPES = ("memory:read", "memory:write")


@dataclass(frozen=True)
class RunIdentity:
    """One benchmark run identity bound to a fixed memory algorithm."""

    benchmark: str
    run_id: str
    key_id: str
    api_key: str
    project_id: str
    memory_algorithm: str
    profile: str | None = None
    project_override_config: dict[str, Any] | None = None

    def to_api_key_entry(self) -> dict[str, Any]:
        """Render this identity as one ``api_keys`` YAML entry."""
        entry = {
            "key_id": self.key_id,
            "api_key": self.api_key,
            "project_id": self.project_id,
            "memory_algorithm": self.memory_algorithm,
            "enabled": True,
            "scopes": list(DEFAULT_SCOPES),
        }
        if self.project_override_config:
            entry["project_override_config"] = self.project_override_config
        return entry




def new_identity(
    benchmark: str,
    memory_algorithm: str,
    *,
    now: datetime | None = None,
    profile: str | None = None,
    project_override_config: dict[str, Any] | None = None,
) -> RunIdentity:
    """Create a unique run identity."""
    timestamp = now or datetime.now(UTC)
    suffix = f"{timestamp:%Y%m%d_%H%M%S}_{secrets.token_hex(4)}"
    return RunIdentity(
        benchmark=benchmark,
        run_id=suffix,
        key_id=f"key_{benchmark}_{memory_algorithm}_{suffix}",
        api_key=f"dev-api-key-{benchmark}-{memory_algorithm}-{suffix}".replace("_", "-"),
        project_id=f"proj_{benchmark}_{memory_algorithm}_{suffix}",
        memory_algorithm=memory_algorithm,
        profile=profile,
        project_override_config=project_override_config,
    )


def write_api_keys(path: str | Path, identities: list[RunIdentity]) -> None:
    """Write generated identities to an ``api_keys`` YAML file atomically."""
    target = Path(path)
    entries = [identity.to_api_key_entry() for identity in identities]
    _atomic_write_yaml(target, {"api_keys": entries})


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
