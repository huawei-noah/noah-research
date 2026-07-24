"""Text preprocessing and sparse vector configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextProcessingConfig:
    """Configuration for normalization, lexical analysis, sparse vectors, and fallback NER."""

    unicode_normal_form: str | None = field(default="NFKC")
    """Unicode normalization form. Set to None to disable."""

    strip_zero_width_chars: bool = field(default=True)
    """Whether zero-width characters are removed during normalization."""

    normalize_whitespace: bool = field(default=True)
    """Whether repeated whitespace is collapsed."""

    normalize_lowercase: bool = field(default=False)
    """Whether normalized text is lowercased globally."""

    whitespace_regex: str = field(default=r"\s+")
    """Regular expression used to collapse whitespace."""

    strip_text: bool = field(default=True)
    """Whether leading and trailing whitespace is stripped."""

    spacy_en_model: str = field(default="en_core_web_sm")
    """spaCy English model name."""

    spacy_zh_model: str = field(default="zh_core_web_sm")
    """spaCy Chinese model name."""

    content_hash_algorithm: str = field(default="md5")
    """Hash algorithm for normalized text content hashes."""

    explicit_language_confidence: float = field(default=1.0)
    """Confidence assigned when the caller passes an explicit language."""

    lang_zh_ratio: float = field(default=0.35)
    """Minimum CJK ratio for zh language detection."""

    lang_en_latin_ratio: float = field(default=0.5)
    """Minimum Latin ratio for en language detection."""

    lang_mixed_zh_ratio: float = field(default=0.35)
    """Minimum CJK ratio for mixed language detection."""

    lang_mixed_latin_ratio: float = field(default=0.15)
    """Minimum Latin ratio for mixed language detection."""

    jieba_cut_all: bool = field(default=False)
    """Whether jieba uses full mode for Chinese BM25 tokenization."""

    bm25_min_term_len: int = field(default=1)
    """Minimum term length kept by the BM25 analyzer."""

    bm25_drop_punctuation: bool = field(default=True)
    """Whether punctuation-only BM25 terms are dropped."""

    bm25_lowercase_en: bool = field(default=True)
    """Whether English fallback tokenization lowercases input text."""

    bm25_use_spacy_lemma: bool = field(default=True)
    """Whether English BM25 analysis tries spaCy lemma first."""

    bm25_en_regex_pattern: str = field(default=r"[A-Za-z][A-Za-z0-9_+-]*")
    """Regex pattern for English fallback lexical analysis."""

    bm25_use_stem_fallback: bool = field(default=True)
    """Whether English fallback terms are stemmed."""

    bm25_stemmer_name: str = field(default="porter")
    """Stemmer identifier for English fallback terms."""

    sparse_hash_dim: int = field(default=2_000_000)
    """Hash trick dimension for sparse BM25 vectors."""

    sparse_hash_algorithm: str = field(default="sha1")
    """Hash algorithm used to map BM25 terms to sparse vector indices."""

    sparse_k1: float = field(default=1.5)
    """BM25 k1 parameter."""

    sparse_b: float = field(default=0.75)
    """BM25 b parameter."""

    sparse_fallback_mode: str = field(default="log_tf")
    """Sparse fallback weighting mode when corpus stats are unavailable."""

    sparse_bm25_model_name: str = field(default="hash_bm25_v1")
    """Sparse vector model name when BM25 corpus stats are used."""

    sparse_fallback_model_name: str = field(default="hash_sparse_tf_v1")
    """Sparse vector model name when fallback TF weighting is used."""

    bm25_idf_smoothing: float = field(default=0.5)
    """BM25 IDF smoothing constant."""

    bm25_min_idf_denominator: float = field(default=1e-9)
    """Lower bound for the BM25 IDF denominator."""

    bm25_min_avg_doc_len: float = field(default=1.0)
    """Lower bound for average document length during BM25 normalization."""

    entity_fallback_on_empty: bool = field(default=True)
    """Whether rule-based entity extraction runs when NER returns no entities."""

    spacy_entity_default_confidence: float = field(default=1.0)
    """Confidence used for spaCy entities because spaCy does not expose per-entity scores."""

    rule_entity_default_confidence: float = field(default=0.6)
    """Confidence used for rule-based entity candidates."""

    max_entity_count: int = field(default=64)
    """Maximum number of entities returned for one text."""

    rule_zh_min_term_len: int = field(default=2)
    """Minimum jieba term length for Chinese rule-based entity candidates."""

    entity_rule_find_quoted_text: bool = field(default=True)
    """Whether rule-based extraction finds quoted text."""

    entity_rule_find_title_case: bool = field(default=True)
    """Whether English rule-based extraction finds title-case spans."""

    entity_rule_find_acronyms: bool = field(default=True)
    """Whether English rule-based extraction finds acronyms."""

    entity_rule_find_file_paths: bool = field(default=True)
    """Whether rule-based extraction finds file paths."""

    entity_rule_find_code_identifiers: bool = field(default=True)
    """Whether English rule-based extraction finds code identifiers."""

    entity_rule_find_book_titles: bool = field(default=True)
    """Whether Chinese rule-based extraction finds book-title or quoted spans."""

    entity_rule_find_english_terms: bool = field(default=True)
    """Whether Chinese rule-based extraction finds embedded English technical terms."""

    entity_rule_find_long_jieba_terms: bool = field(default=True)
    """Whether Chinese rule-based extraction uses long jieba terms as entity candidates."""

    stopwords_zh_path: str | None = field(default=None)
    """Optional Chinese stopwords file path."""

    stopwords_en_path: str | None = field(default=None)
    """Optional English stopwords file path."""

    nlp_max_retries: int = field(default=50)
    """Maximum attempts for loading NLP assets and running third-party tokenizers."""

    nlp_retry_base_delay: float = field(default=0.1)
    """Base delay in seconds for NLP retry backoff."""
