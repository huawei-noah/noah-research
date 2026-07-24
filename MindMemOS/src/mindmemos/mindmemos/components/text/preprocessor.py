"""High-level text preprocessing orchestration."""

from __future__ import annotations

from threading import Lock

from ...config import TextProcessingConfig, get_config
from ...typing import LanguageResult, PreprocessedText, SourceRef
from ._entity import EntityExtractor, LanguageAwareEntityExtractor
from ._hashing import ContentHasher
from ._language import LanguageDetector
from ._lexical import Bm25TextAnalyzer
from ._normalize import TextNormalizer


class TextPreprocessor:
    """Run normalization, language detection, hashing, BM25 analysis, and entities."""

    def __init__(
        self,
        config: TextProcessingConfig,
        *,
        normalizer: TextNormalizer | None = None,
        language_detector: LanguageDetector | None = None,
        content_hasher: ContentHasher | None = None,
        bm25_analyzer: Bm25TextAnalyzer | None = None,
        entity_extractor: EntityExtractor | None = None,
    ):
        self.config = config
        self.normalizer = normalizer or TextNormalizer(config)
        self.language_detector = language_detector or LanguageDetector(config)
        self.content_hasher = content_hasher or ContentHasher(config)
        self.bm25_analyzer = bm25_analyzer or Bm25TextAnalyzer(config)
        self.entity_extractor = entity_extractor or LanguageAwareEntityExtractor(config)

    def preprocess_text(
        self,
        text: str,
        *,
        lang: str | None = None,
        source_ref: SourceRef | None = None,
        segment_id: str | None = None,
        include_entities: bool = True,
    ) -> PreprocessedText:
        normalized = self.normalizer.normalize(text)
        detected = self._detect_language(normalized, lang)
        lexical = self.bm25_analyzer.analyze(normalized, detected.lang)
        entities = self.entity_extractor.extract(normalized, detected.lang) if include_entities else []
        content_hash = self.content_hasher.hash_text(normalized)

        return PreprocessedText(
            segment_id=segment_id,
            text=text,
            normalized_text=normalized,
            lang=detected.lang,
            content_hash=content_hash,
            bm25_text=lexical.bm25_text,
            tokens=lexical.terms,
            entities=entities,
            source_ref=source_ref,
            metadata={
                "language_confidence": detected.confidence,
                "zh_ratio": detected.zh_ratio,
                "latin_ratio": detected.latin_ratio,
                "term_count": lexical.term_count,
                "entity_count": len(entities),
            },
        )

    def preprocess_many(
        self,
        texts: list[str],
        *,
        lang: str | None = None,
        source_refs: list[SourceRef | None] | None = None,
        segment_ids: list[str | None] | None = None,
        include_entities: bool = True,
    ) -> list[PreprocessedText]:
        if source_refs is not None and len(source_refs) != len(texts):
            raise ValueError("source_refs and texts must have the same length")
        if segment_ids is not None and len(segment_ids) != len(texts):
            raise ValueError("segment_ids and texts must have the same length")

        normalized_texts = [self.normalizer.normalize(text) for text in texts]
        detected = [self._detect_language(normalized, lang) for normalized in normalized_texts]
        langs = [item.lang for item in detected]
        lexical = self.bm25_analyzer.analyze_many(normalized_texts, langs)
        if include_entities:
            entities = self.entity_extractor.extract_many(normalized_texts, langs)
        else:
            entities = [[] for _ in texts]

        refs = source_refs or [None] * len(texts)
        ids = segment_ids or [None] * len(texts)
        return [
            PreprocessedText(
                segment_id=segment_id,
                text=text,
                normalized_text=normalized,
                lang=detected_item.lang,
                content_hash=self.content_hasher.hash_text(normalized),
                bm25_text=lexical_item.bm25_text,
                tokens=lexical_item.terms,
                entities=entity_items,
                source_ref=source_ref,
                metadata={
                    "language_confidence": detected_item.confidence,
                    "zh_ratio": detected_item.zh_ratio,
                    "latin_ratio": detected_item.latin_ratio,
                    "term_count": lexical_item.term_count,
                    "entity_count": len(entity_items),
                },
            )
            for text, normalized, detected_item, lexical_item, entity_items, source_ref, segment_id in zip(
                texts,
                normalized_texts,
                detected,
                lexical,
                entities,
                refs,
                ids,
                strict=True,
            )
        ]

    def preprocess_query(
        self,
        query: str,
        *,
        lang: str | None = None,
        include_entities: bool = True,
    ) -> PreprocessedText:
        return self.preprocess_text(query, lang=lang, include_entities=include_entities)

    def _detect_language(self, normalized_text: str, lang: str | None) -> LanguageResult:
        if lang is None:
            return self.language_detector.detect(normalized_text)
        return LanguageResult(lang=lang, confidence=self.config.explicit_language_confidence)


_GLOBAL_TEXT_PREPROCESSOR: TextPreprocessor | None = None
_GLOBAL_TEXT_PREPROCESSOR_CONFIG: TextProcessingConfig | None = None
_GLOBAL_TEXT_PREPROCESSOR_LOCK = Lock()


def get_text_preprocessor(config: TextProcessingConfig | None = None) -> TextPreprocessor:
    global _GLOBAL_TEXT_PREPROCESSOR, _GLOBAL_TEXT_PREPROCESSOR_CONFIG
    cfg = config or get_config().algo_config.text_processing
    # TextProcessingConfig is frozen, so the heavy lazy NLP state can be shared process-wide.
    if _GLOBAL_TEXT_PREPROCESSOR is None or cfg != _GLOBAL_TEXT_PREPROCESSOR_CONFIG:
        with _GLOBAL_TEXT_PREPROCESSOR_LOCK:
            if _GLOBAL_TEXT_PREPROCESSOR is None or cfg != _GLOBAL_TEXT_PREPROCESSOR_CONFIG:
                _GLOBAL_TEXT_PREPROCESSOR = TextPreprocessor(cfg)
                _GLOBAL_TEXT_PREPROCESSOR_CONFIG = cfg
    return _GLOBAL_TEXT_PREPROCESSOR
