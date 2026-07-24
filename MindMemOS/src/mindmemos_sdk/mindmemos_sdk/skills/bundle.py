"""Local skill bundle whitelist, normalization, serialization and hashing.

This module intentionally mirrors the server-side skill bundle algorithm without
importing ``mindmemos``. The SDK is distributed independently, so tests pin the
byte-for-byte behavior against the server implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..errors import SkillBundleError

SKILL_WHITELIST: frozenset[str] = frozenset({"SKILL.md"})
CONTENT_HASH_ALGORITHM = "sha256"


def is_whitelisted(path: str) -> bool:
    """Return whether ``path`` points at a whitelisted skill bundle file."""

    return os.path.basename(path.replace("\\", "/")) in SKILL_WHITELIST


def resolve_skill_dir(skill_path: str | os.PathLike[str]) -> Path:
    """Resolve a skill directory from a directory path or a whitelisted skill file."""

    path = Path(skill_path).expanduser()
    if path.is_file():
        if not is_whitelisted(str(path)):
            raise SkillBundleError(f"skill file is not whitelisted: {path}")
        return path.parent
    return path


def read_local_bundle(skill_path: str | os.PathLike[str]) -> dict[str, str]:
    """Read whitelisted files from a local skill directory or skill file.

    Args:
        skill_path: Directory containing a skill, or a whitelisted skill file.

    Returns:
        Mapping of relative whitelisted path to text content.

    Raises:
        SkillBundleError: If the path is not a directory/SKILL.md file or has no whitelisted file.
    """

    path = Path(skill_path).expanduser()
    if path.is_file():
        if not is_whitelisted(str(path)):
            raise SkillBundleError(f"skill file is not whitelisted: {path}")
        return normalize_bundle({path.name: path.read_text(encoding="utf-8")})

    root = path
    if not root.is_dir():
        raise SkillBundleError(f"skill path does not exist or is not a directory/SKILL.md file: {root}")

    files: dict[str, str] = {}
    for name in sorted(SKILL_WHITELIST):
        path = root / name
        if path.is_file():
            files[name] = path.read_text(encoding="utf-8")
    return normalize_bundle(files)


def normalize_bundle(files: dict[str, str]) -> dict[str, str]:
    """Reduce raw bundle files to canonical whitelisted form."""

    normalized: dict[str, str] = {}
    for path, content in files.items():
        if not is_whitelisted(path):
            continue
        key = os.path.basename(path.replace("\\", "/"))
        normalized[key] = _normalize_newlines(content)
    if not normalized:
        raise SkillBundleError("skill bundle contains no whitelisted file (expected SKILL.md)")
    return normalized


def serialize_bundle(files: dict[str, str]) -> str:
    """Serialize a bundle to its canonical text representation."""

    normalized = normalize_bundle(files)
    records = [{"path": path, "content": normalized[path]} for path in sorted(normalized)]
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def deserialize_bundle(text: str) -> dict[str, str]:
    """Reconstruct the canonical bundle mapping from ``serialize_bundle`` output."""

    try:
        records = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SkillBundleError("skill bundle content is not valid canonical JSON") from exc
    if not isinstance(records, list):
        raise SkillBundleError("skill bundle content must be a list of {path, content} records")

    files: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or "path" not in record or "content" not in record:
            raise SkillBundleError("skill bundle record must be an object with 'path' and 'content'")
        path, content = record["path"], record["content"]
        if not isinstance(path, str) or not isinstance(content, str):
            raise SkillBundleError("skill bundle record 'path' and 'content' must be strings")
        files[path] = content
    return files


def bundle_files_from_content(content: str) -> dict[str, str]:
    """Parse upload content into a canonical bundle file mapping."""

    try:
        files = deserialize_bundle(content)
    except SkillBundleError:
        files = {"SKILL.md": content}
    return normalize_bundle(files)


def compute_content_hash(files: dict[str, str]) -> str:
    """Compute the SHA-256 content hash of canonical bundle text."""

    canonical = serialize_bundle(files)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_newlines(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")
