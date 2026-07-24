"""Skill bundle whitelist, normalization and content hashing (design §1 / §4.2).

A skill is a bundle of text files. Only *whitelisted* files participate in the
``content_hash``, in the registered bundle and in local change detection; every
non-whitelisted file is ignored entirely (it does not enter the bundle, does not
affect the hash and never triggers a new version). The whitelist is hard-coded
here and is NOT user-configurable; its scope is decided by what the algorithm
can currently process. Today it admits only ``SKILL.md``.

The hashing here is the single shared implementation used by both the edge SDK
(recognition / register) and the server, so a skill recognized on the edge and
the same skill registered on the server yield an identical ``content_hash``
(design §4.3.1 step 6).
"""

from __future__ import annotations

import json
import os

from ...errors import SkillBundleError
from ..text import digest_text

# Whitelisted bundle file names (matched by basename). Hard-coded, not
# user-configurable (design §4.2). Currently only ``SKILL.md``.
SKILL_WHITELIST: frozenset[str] = frozenset({"SKILL.md"})

# Hash algorithm shared by edge and server. SHA-256 per design §1.
CONTENT_HASH_ALGORITHM = "sha256"


def is_whitelisted(path: str) -> bool:
    """Return whether ``path`` points at a whitelisted bundle file.

    Matching is by basename so a full path (``skill-name/SKILL.md``) and a bare
    name (``SKILL.md``) both resolve to the same canonical bundle entry.
    """

    return os.path.basename(path.replace("\\", "/")) in SKILL_WHITELIST


def normalize_bundle(files: dict[str, str]) -> dict[str, str]:
    """Reduce raw bundle files to the canonical whitelisted form.

    Keys are canonicalized to their basename (the whitelist matches by name and
    every whitelisted name is unique) and content newlines are normalized to
    ``\\n`` so the hash is stable across platforms. Non-whitelisted files are
    dropped. Time/permission metadata is never considered.

    Args:
        files: Mapping of file path -> raw file content.

    Returns:
        Mapping of canonical (basename) path -> newline-normalized content,
        containing only whitelisted files.

    Raises:
        SkillBundleError: If no whitelisted file is present.
    """

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
    """Serialize a bundle to its canonical text representation.

    The result is deterministic (files sorted by canonical path) and is what gets
    stored in ``skill_blob.content``. ``deserialize_bundle`` is its inverse.

    Raises:
        SkillBundleError: If no whitelisted file is present.
    """

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
        # Each element must be a {path, content} object with string values; any
        # other shape (missing key, wrong type) is a malformed bundle, not a 500.
        if not isinstance(record, dict) or "path" not in record or "content" not in record:
            raise SkillBundleError("skill bundle record must be an object with 'path' and 'content'")
        path, content = record["path"], record["content"]
        if not isinstance(path, str) or not isinstance(content, str):
            raise SkillBundleError("skill bundle record 'path' and 'content' must be strings")
        files[path] = content
    return files


def bundle_files_from_content(content: str) -> dict[str, str]:
    """Parse register/upload ``content`` text into a bundle files mapping.

    The wire contract is the canonical bundle text produced by
    :func:`serialize_bundle` (design §3 / §5.2). For robustness a non-canonical
    payload (e.g. a bare ``SKILL.md`` body that is not the canonical JSON) is
    treated as the single whitelisted ``SKILL.md`` file, so the edge can upload
    either form and still get a stable, identical ``content_hash``.

    Raises:
        SkillBundleError: If the resulting bundle has no whitelisted file.
    """

    try:
        files = deserialize_bundle(content)
    except SkillBundleError:
        files = {"SKILL.md": content}
    # Validate the bundle yields a whitelisted file; raises SkillBundleError otherwise.
    normalize_bundle(files)
    return files


def compute_content_hash(files: dict[str, str]) -> str:
    """Compute the bundle ``content_hash`` (SHA-256 of the canonical text).

    This is the single source of truth shared by edge and server.

    Raises:
        SkillBundleError: If no whitelisted file is present.
    """

    return digest_text(serialize_bundle(files), algorithm=CONTENT_HASH_ALGORITHM)


def _normalize_newlines(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")
