"""Text normalization primitives."""

from __future__ import annotations

import re
import unicodedata

from ...config import TextProcessingConfig

ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\ufeff"


class TextNormalizer:
    """Normalize raw text before hashing, BM25 analysis, and entity extraction."""

    def __init__(self, config: TextProcessingConfig):
        self.config = config

    def normalize(self, text: str) -> str:
        value = text or ""
        if self.config.unicode_normal_form:
            value = unicodedata.normalize(self.config.unicode_normal_form, value)
        if self.config.strip_zero_width_chars:
            value = remove_zero_width_chars(value)
        if self.config.normalize_whitespace:
            value = re.sub(self.config.whitespace_regex, " ", value)
        if self.config.normalize_lowercase:
            value = value.lower()
        if self.config.strip_text:
            value = value.strip()
        return value


def remove_zero_width_chars(text: str) -> str:
    """Remove zero-width characters that commonly appear in copied text."""

    return text.translate({ord(ch): None for ch in ZERO_WIDTH_CHARS})
