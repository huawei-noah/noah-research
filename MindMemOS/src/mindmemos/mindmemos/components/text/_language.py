"""Lightweight language detection for memory preprocessing."""

from __future__ import annotations

from ...config import TextProcessingConfig
from ...typing import LanguageResult


class LanguageDetector:
    """Detect whether a text is Chinese, English, mixed, or unknown."""

    def __init__(self, config: TextProcessingConfig):
        self.config = config

    def detect(self, text: str) -> LanguageResult:
        chars = [ch for ch in text if not ch.isspace()]
        if not chars:
            return LanguageResult(lang="unknown", confidence=0.0)

        zh_count = sum(1 for ch in chars if is_cjk(ch))
        latin_count = sum(1 for ch in chars if is_latin(ch))
        zh_ratio = zh_count / len(chars)
        latin_ratio = latin_count / len(chars)

        if zh_ratio >= self.config.lang_mixed_zh_ratio and latin_ratio >= self.config.lang_mixed_latin_ratio:
            return LanguageResult(
                lang="mixed",
                confidence=max(zh_ratio, latin_ratio),
                zh_ratio=zh_ratio,
                latin_ratio=latin_ratio,
            )
        if zh_ratio >= self.config.lang_zh_ratio:
            return LanguageResult(lang="zh", confidence=zh_ratio, zh_ratio=zh_ratio, latin_ratio=latin_ratio)
        if latin_ratio >= self.config.lang_en_latin_ratio:
            return LanguageResult(lang="en", confidence=latin_ratio, zh_ratio=zh_ratio, latin_ratio=latin_ratio)
        return LanguageResult(
            lang="unknown",
            confidence=max(zh_ratio, latin_ratio),
            zh_ratio=zh_ratio,
            latin_ratio=latin_ratio,
        )


def is_cjk(ch: str) -> bool:
    """Return whether a character belongs to common CJK ideograph ranges."""

    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def is_latin(ch: str) -> bool:
    """Return whether a character is a Latin letter."""

    return ("A" <= ch <= "Z") or ("a" <= ch <= "z")


def detect_prompt_language(text: str, *, fallback: str = "EN") -> str:
    """检测文本语言并返回 prompt 语言码（"ZH" 或 "EN"）。

    Args:
        text: 待检测文本。
        fallback: 当检测结果为 mixed/unknown 时的回退值。

    Returns:
        "ZH" 或 "EN"。
    """
    from ...config import get_config

    detector = LanguageDetector(get_config().algo_config.text_processing)
    result = detector.detect(text)
    if result.lang == "zh":
        return "ZH"
    if result.lang == "en":
        return "EN"
    return fallback
