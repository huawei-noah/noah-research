"""Content hashing helpers for memory preprocessing."""

from __future__ import annotations

import hashlib

from ...config import TextProcessingConfig


class ContentHasher:
    """Create stable hashes for normalized text."""

    def __init__(self, config: TextProcessingConfig):
        self.config = config

    def hash_text(self, normalized_text: str) -> str:
        return digest_text(normalized_text, algorithm=self.config.content_hash_algorithm)


def digest_text(text: str, *, algorithm: str) -> str:
    """Hash text with a hashlib algorithm name."""

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()
