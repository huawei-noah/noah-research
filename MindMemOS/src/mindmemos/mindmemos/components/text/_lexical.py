"""BM25-oriented lexical analysis."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ...config import TextProcessingConfig
from ...logging import get_logger
from ...typing import BM25TokenizationResult
from ._nlp_retry import run_nlp_with_retry

logger = get_logger(__name__)

DEFAULT_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
DEFAULT_ZH_STOPWORDS = {
    "的",
    "了",
    "是",
    "和",
    "与",
    "或",
    "在",
    "就",
    "都",
    "而",
    "及",
    "一个",
    "我们",
    "你们",
    "他们",
}
PUNCTUATION = set(string.punctuation) | set("，。！？；：、“”‘’（）【】《》…—·")


class Bm25TextAnalyzer:
    """Convert normalized text into BM25 terms and BM25 payload text."""

    def __init__(self, config: TextProcessingConfig):
        self.config = config
        self.zh_stopwords = load_stopwords(config.stopwords_zh_path, DEFAULT_ZH_STOPWORDS)
        self.en_stopwords = load_stopwords(config.stopwords_en_path, DEFAULT_EN_STOPWORDS)
        self._spacy_en: Any | None = None
        self._spacy_en_unavailable = False

    def analyze(self, text: str, lang: str) -> BM25TokenizationResult:
        if lang == "zh":
            terms = self.analyze_zh(text)
        elif lang == "en":
            terms = self.analyze_en(text)
        else:
            terms = self.analyze_mixed(text)
        return BM25TokenizationResult(lang=lang, terms=terms, bm25_text=" ".join(terms), term_count=len(terms))

    def analyze_many(self, texts: list[str], langs: list[str]) -> list[BM25TokenizationResult]:
        if len(texts) != len(langs):
            raise ValueError("texts and langs must have the same length")

        results: list[BM25TokenizationResult | None] = [None] * len(texts)
        en_items = [(index, text) for index, (text, lang) in enumerate(zip(texts, langs, strict=True)) if lang == "en"]
        if en_items:
            en_terms = self._analyze_en_many([text for _, text in en_items])
            for (index, _), terms in zip(en_items, en_terms, strict=True):
                results[index] = BM25TokenizationResult(
                    lang="en",
                    terms=terms,
                    bm25_text=" ".join(terms),
                    term_count=len(terms),
                )

        for index, (text, lang) in enumerate(zip(texts, langs, strict=True)):
            if results[index] is None:
                results[index] = self.analyze(text, lang)

        return [result for result in results if result is not None]

    def analyze_zh(self, text: str) -> list[str]:
        terms: list[str] = []
        for raw_term in self._jieba_cut(text):
            term = normalize_term(raw_term)
            if not self._keep_term(term, self.zh_stopwords):
                continue
            terms.append(term)
        return terms

    def analyze_en(self, text: str) -> list[str]:
        if self.config.bm25_use_spacy_lemma:
            terms = self._analyze_en_spacy(text)
            if terms:
                return terms

        return self._analyze_en_fallback(text)

    def _analyze_en_many(self, texts: list[str]) -> list[list[str]]:
        if not self.config.bm25_use_spacy_lemma:
            return [self._analyze_en_fallback(text) for text in texts]

        spacy_terms = self._analyze_en_spacy_many(texts)
        return [terms if terms else self._analyze_en_fallback(text) for text, terms in zip(texts, spacy_terms, strict=True)]

    def _analyze_en_fallback(self, text: str) -> list[str]:
        source_text = text.lower() if self.config.bm25_lowercase_en else text
        raw_terms = re.findall(self.config.bm25_en_regex_pattern, source_text)
        terms = [normalize_term(term) for term in raw_terms]
        if self.config.bm25_use_stem_fallback:
            terms = [stem_term(term, algorithm=self.config.bm25_stemmer_name) for term in terms]
        return [term for term in terms if self._keep_term(term, self.en_stopwords)]

    def analyze_mixed(self, text: str) -> list[str]:
        return merge_preserving_order(self.analyze_zh(text), self.analyze_en(text))

    def _analyze_en_spacy(self, text: str) -> list[str]:
        if self._spacy_en_unavailable:
            return []
        try:
            nlp = self._load_spacy_en()
            doc = nlp(text)
            terms = [
                normalize_term(token.lemma_.lower())
                for token in doc
                if token.is_alpha and not token.is_stop and token.lemma_
            ]
            return [term for term in terms if self._keep_term(term, self.en_stopwords)]
        except Exception as exc:
            self._spacy_en_unavailable = True
            logger.warning("spacy_english_lexical_analysis_failed", error=str(exc))
            return []

    def _analyze_en_spacy_many(self, texts: list[str]) -> list[list[str]]:
        if self._spacy_en_unavailable:
            return [[] for _ in texts]
        try:
            nlp = self._load_spacy_en()
            docs = run_nlp_with_retry(
                lambda: list(nlp.pipe(texts)),
                config=self.config,
                operation_name="spacy_english_lexical_analysis",
            )
            return [
                [
                    term
                    for term in (normalize_term(token.lemma_.lower()) for token in doc if token.is_alpha and not token.is_stop and token.lemma_)
                    if self._keep_term(term, self.en_stopwords)
                ]
                for doc in docs
            ]
        except Exception as exc:
            self._spacy_en_unavailable = True
            logger.warning("spacy_english_lexical_analysis_failed", error=str(exc))
            return [[] for _ in texts]

    def _load_spacy_en(self) -> Any:
        if self._spacy_en is None:
            import spacy

            self._spacy_en = run_nlp_with_retry(
                lambda: spacy.load(self.config.spacy_en_model),
                config=self.config,
                operation_name="spacy_load",
            )
        return self._spacy_en

    def _jieba_cut(self, text: str) -> Iterable[str]:
        try:
            import jieba

            return run_nlp_with_retry(
                lambda: list(jieba.cut(text, cut_all=self.config.jieba_cut_all)),
                config=self.config,
                operation_name="jieba_cut",
            )
        except Exception as exc:
            logger.warning("jieba_cut_failed", error=str(exc))
            return re.findall(r"[\u4e00-\u9fff]+|[A-Za-z][A-Za-z0-9_+-]*", text)

    def _keep_term(self, term: str, stopwords: set[str]) -> bool:
        if not term:
            return False
        if len(term) < self.config.bm25_min_term_len:
            return False
        if self.config.bm25_drop_punctuation and is_punctuation(term):
            return False
        return term not in stopwords


def load_stopwords(path: str | None, defaults: set[str]) -> set[str]:
    """Load stopwords from a newline-separated file, falling back to defaults."""

    words = set(defaults)
    if not path:
        return words
    stopwords_path = Path(path)
    if not stopwords_path.exists():
        logger.warning("stopwords_file_missing", path=str(stopwords_path))
        return words
    for line in stopwords_path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word and not word.startswith("#"):
            words.add(word)
    return words


def normalize_term(term: str) -> str:
    """Normalize a single BM25 term without changing semantic content."""

    return term.strip().strip("".join(PUNCTUATION)).lower()


def is_punctuation(term: str) -> bool:
    """Return whether a term is made only of punctuation characters."""

    return all(ch in PUNCTUATION for ch in term)


def merge_preserving_order(*term_lists: list[str]) -> list[str]:
    """Merge multiple term lists while keeping the first occurrence order."""

    merged: list[str] = []
    seen: set[str] = set()
    for terms in term_lists:
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            merged.append(term)
    return merged


def stem_term(term: str, *, algorithm: str) -> str:
    """Stem an English fallback term with a small built-in suffix stemmer."""

    if algorithm != "porter":
        return term
    return simple_porter_like_stem(term)


def simple_porter_like_stem(term: str) -> str:
    """Apply a conservative Porter-like suffix normalization.

    This is intentionally small and deterministic. It keeps the component free
    of additional corpus downloads while still making fallback terms less
    brittle when spaCy is unavailable.
    """

    if len(term) <= 3:
        return term
    suffix_rules = (
        ("ization", "ize"),
        ("ational", "ate"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("tional", "tion"),
        ("ingly", ""),
        ("edly", ""),
        ("ing", ""),
        ("ed", ""),
        ("ies", "y"),
        ("s", ""),
    )
    for suffix, replacement in suffix_rules:
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            return term[: -len(suffix)] + replacement
    return term
