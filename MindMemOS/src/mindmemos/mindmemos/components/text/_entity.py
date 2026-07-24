"""Entity extraction components for preprocessed memory text."""

from __future__ import annotations

import re
from typing import Any, Protocol

from ...config import TextProcessingConfig
from ...logging import get_logger
from ...typing import Entity
from ._lexical import normalize_term
from ._nlp_retry import run_nlp_with_retry

logger = get_logger(__name__)

FILE_PATH_PATTERN = re.compile(r"(?:[\w.-]+/)+[\w.-]+|[\w.-]+\.(?:py|md|txt|json|yaml|yml|toml|js|ts|tsx|jsx)")
ENGLISH_TERM_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_+-]{1,}\b")
TITLE_CASE_PATTERN = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,4})\b")
ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")
CODE_IDENTIFIER_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_.]*\b")
QUOTED_PATTERN = re.compile(r'"([^"\n]{2,80})"|\'([^\'\n]{2,80})\'')
ZH_BOOK_TITLE_PATTERN = re.compile(r"《([^》]{2,80})》|“([^”]{2,80})”|「([^」]{2,80})」")


class EntityExtractor(Protocol):
    """Storage-agnostic entity extraction interface."""

    def extract(self, text: str, lang: str) -> list[Entity]:
        """Extract entities from text."""

    def extract_many(self, texts: list[str], langs: list[str]) -> list[list[Entity]]:
        """Extract entities from multiple texts."""


class LanguageAwareEntityExtractor:
    """Route entity extraction by detected language."""

    def __init__(self, config: TextProcessingConfig):
        self.config = config
        self.zh_extractor = FallbackEntityExtractor(
            primary=SpacyEntityExtractor(config, config.spacy_zh_model, "spacy_zh"),
            fallback=RuleBasedChineseEntityExtractor(config),
            config=config,
        )
        self.en_extractor = FallbackEntityExtractor(
            primary=SpacyEntityExtractor(config, config.spacy_en_model, "spacy_en"),
            fallback=RuleBasedEnglishEntityExtractor(config),
            config=config,
        )

    def extract(self, text: str, lang: str) -> list[Entity]:
        if lang == "zh":
            return self.zh_extractor.extract(text, lang)[: self.config.max_entity_count]
        if lang == "en":
            return self.en_extractor.extract(text, lang)[: self.config.max_entity_count]
        zh_entities = self.zh_extractor.extract(text, "zh")
        en_entities = self.en_extractor.extract(text, "en")
        return dedup_entities(zh_entities + en_entities)[: self.config.max_entity_count]

    def extract_many(self, texts: list[str], langs: list[str]) -> list[list[Entity]]:
        if len(texts) != len(langs):
            raise ValueError("texts and langs must have the same length")

        results: list[list[Entity] | None] = [None] * len(texts)
        for target_lang, extractor in (("zh", self.zh_extractor), ("en", self.en_extractor)):
            items = [
                (index, text)
                for index, (text, lang) in enumerate(zip(texts, langs, strict=True))
                if lang == target_lang
            ]
            if not items:
                continue
            extracted = extractor.extract_many([text for _, text in items], [target_lang] * len(items))
            for (index, _), entities in zip(items, extracted, strict=True):
                results[index] = entities[: self.config.max_entity_count]

        for index, (text, lang) in enumerate(zip(texts, langs, strict=True)):
            if results[index] is None:
                results[index] = self.extract(text, lang)

        return [entities for entities in results if entities is not None]


class FallbackEntityExtractor:
    """Try a primary extractor and fall back to rule-based extraction."""

    def __init__(self, primary: EntityExtractor, fallback: EntityExtractor, config: TextProcessingConfig):
        self.primary = primary
        self.fallback = fallback
        self.config = config

    def extract(self, text: str, lang: str) -> list[Entity]:
        try:
            entities = self.primary.extract(text, lang)
            if entities or not self.config.entity_fallback_on_empty:
                return entities[: self.config.max_entity_count]
        except Exception as exc:
            logger.warning("ner_entity_extract_failed", lang=lang, error=str(exc))
        return self.fallback.extract(text, lang)[: self.config.max_entity_count]

    def extract_many(self, texts: list[str], langs: list[str]) -> list[list[Entity]]:
        try:
            primary_results = self.primary.extract_many(texts, langs)
        except Exception as exc:
            lang = langs[0] if langs else "unknown"
            logger.warning("ner_entity_extract_failed", lang=lang, error=str(exc))
            return [self.fallback.extract(text, lang)[: self.config.max_entity_count] for text, lang in zip(texts, langs, strict=True)]

        results: list[list[Entity]] = []
        for text, lang, entities in zip(texts, langs, primary_results, strict=True):
            if entities or not self.config.entity_fallback_on_empty:
                results.append(entities[: self.config.max_entity_count])
            else:
                results.append(self.fallback.extract(text, lang)[: self.config.max_entity_count])
        return results


class SpacyEntityExtractor:
    """spaCy NER extractor with lazy model loading."""

    def __init__(self, config: TextProcessingConfig, model_name: str, extractor_name: str):
        self.config = config
        self.model_name = model_name
        self.extractor_name = extractor_name
        self._nlp: Any | None = None

    def extract(self, text: str, lang: str) -> list[Entity]:
        doc = self._load()(text)
        return self._entities_from_doc(doc)

    def extract_many(self, texts: list[str], langs: list[str]) -> list[list[Entity]]:
        del langs
        docs = run_nlp_with_retry(
            lambda: list(self._load().pipe(texts)),
            config=self.config,
            operation_name="spacy_entity_extract",
        )
        return [self._entities_from_doc(doc) for doc in docs]

    def _entities_from_doc(self, doc: Any) -> list[Entity]:
        entities = [
            make_entity(
                name=ent.text,
                entity_type=map_spacy_label(ent.label_),
                confidence=self.config.spacy_entity_default_confidence,
                extractor=self.extractor_name,
                offsets=[(ent.start_char, ent.end_char)],
            )
            for ent in doc.ents
        ]
        return dedup_entities(entities)[: self.config.max_entity_count]

    def _load(self) -> Any:
        if self._nlp is None:
            import spacy

            self._nlp = run_nlp_with_retry(
                lambda: spacy.load(self.model_name),
                config=self.config,
                operation_name="spacy_load",
            )
        return self._nlp


class RuleBasedEnglishEntityExtractor:
    """Rule-based English entity fallback."""

    def __init__(self, config: TextProcessingConfig):
        self.config = config

    def extract(self, text: str, lang: str) -> list[Entity]:
        entities: list[Entity] = []
        if self.config.entity_rule_find_quoted_text:
            entities.extend(find_quoted_text(text, self.config))
        if self.config.entity_rule_find_title_case:
            entities.extend(find_title_case_spans(text, self.config))
        if self.config.entity_rule_find_acronyms:
            entities.extend(find_acronyms(text, self.config))
        if self.config.entity_rule_find_file_paths:
            entities.extend(find_file_paths(text, self.config))
        if self.config.entity_rule_find_code_identifiers:
            entities.extend(find_code_identifiers(text, self.config))
        return dedup_entities(entities)[: self.config.max_entity_count]


class RuleBasedChineseEntityExtractor:
    """Rule-based Chinese entity fallback."""

    def __init__(self, config: TextProcessingConfig):
        self.config = config

    def extract(self, text: str, lang: str) -> list[Entity]:
        entities: list[Entity] = []
        if self.config.entity_rule_find_book_titles:
            entities.extend(find_book_title_or_quotes(text, self.config))
        if self.config.entity_rule_find_file_paths:
            entities.extend(find_file_paths(text, self.config))
        if self.config.entity_rule_find_english_terms:
            entities.extend(find_english_technical_terms(text, self.config))
        if self.config.entity_rule_find_long_jieba_terms:
            entities.extend(find_long_jieba_terms(text, self.config))
        return dedup_entities(entities)[: self.config.max_entity_count]


def make_entity(
    *,
    name: str,
    entity_type: str,
    confidence: float,
    extractor: str,
    offsets: list[tuple[int, int]] | None = None,
) -> Entity:
    """Create a normalized Entity DTO."""

    clean_name = name.strip()
    return Entity(
        name=clean_name,
        canonical_name=canonicalize_entity(clean_name),
        entity_type=entity_type,
        aliases=[clean_name],
        confidence=confidence,
        extractor=extractor,
        offsets=offsets,
    )


def canonicalize_entity(name: str) -> str:
    """Canonicalize entity names for project-local merging."""

    return normalize_term(name).replace(" ", "_")


def map_spacy_label(label: str) -> str:
    """Map spaCy labels to coarse MindMemOS entity types."""

    mapping = {
        "PERSON": "person",
        "ORG": "organization",
        "GPE": "location",
        "LOC": "location",
        "PRODUCT": "product",
        "EVENT": "event",
        "WORK_OF_ART": "work",
        "LAW": "law",
        "LANGUAGE": "language",
        "DATE": "date",
        "TIME": "time",
        "MONEY": "money",
    }
    return mapping.get(label, label.lower() if label else "entity")


def find_quoted_text(text: str, config: TextProcessingConfig) -> list[Entity]:
    """Find quoted spans in English text."""

    entities: list[Entity] = []
    for match in QUOTED_PATTERN.finditer(text):
        name = next(group for group in match.groups() if group)
        entities.append(
            make_entity(
                name=name,
                entity_type="quoted_text",
                confidence=config.rule_entity_default_confidence,
                extractor="rule_en",
                offsets=[match.span()],
            )
        )
    return entities


def find_title_case_spans(text: str, config: TextProcessingConfig) -> list[Entity]:
    """Find title-case spans that often represent names, projects, or products."""

    return [
        make_entity(
            name=match.group(0),
            entity_type="proper_noun",
            confidence=config.rule_entity_default_confidence,
            extractor="rule_en",
            offsets=[match.span()],
        )
        for match in TITLE_CASE_PATTERN.finditer(text)
    ]


def find_acronyms(text: str, config: TextProcessingConfig) -> list[Entity]:
    """Find uppercase acronyms."""

    return [
        make_entity(
            name=match.group(0),
            entity_type="acronym",
            confidence=config.rule_entity_default_confidence,
            extractor="rule_en",
            offsets=[match.span()],
        )
        for match in ACRONYM_PATTERN.finditer(text)
    ]


def find_file_paths(text: str, config: TextProcessingConfig) -> list[Entity]:
    """Find file path-like spans."""

    return [
        make_entity(
            name=match.group(0),
            entity_type="file",
            confidence=config.rule_entity_default_confidence,
            extractor="rule_file",
            offsets=[match.span()],
        )
        for match in FILE_PATH_PATTERN.finditer(text)
    ]


def find_code_identifiers(text: str, config: TextProcessingConfig) -> list[Entity]:
    """Find dotted code identifiers."""

    return [
        make_entity(
            name=match.group(0),
            entity_type="code",
            confidence=config.rule_entity_default_confidence,
            extractor="rule_en",
            offsets=[match.span()],
        )
        for match in CODE_IDENTIFIER_PATTERN.finditer(text)
    ]


def find_book_title_or_quotes(text: str, config: TextProcessingConfig) -> list[Entity]:
    """Find Chinese book-title or quoted spans."""

    entities: list[Entity] = []
    for match in ZH_BOOK_TITLE_PATTERN.finditer(text):
        name = next(group for group in match.groups() if group)
        entities.append(
            make_entity(
                name=name,
                entity_type="title",
                confidence=config.rule_entity_default_confidence,
                extractor="rule_zh",
                offsets=[match.span()],
            )
        )
    return entities


def find_english_technical_terms(text: str, config: TextProcessingConfig) -> list[Entity]:
    """Find English technical terms embedded in Chinese text."""

    return [
        make_entity(
            name=match.group(0),
            entity_type="technical_term",
            confidence=config.rule_entity_default_confidence,
            extractor="rule_zh",
            offsets=[match.span()],
        )
        for match in ENGLISH_TERM_PATTERN.finditer(text)
    ]


def find_long_jieba_terms(text: str, config: TextProcessingConfig) -> list[Entity]:
    """Find long jieba terms as coarse Chinese entity candidates."""

    try:
        import jieba

        terms = run_nlp_with_retry(
            lambda: list(jieba.cut(text, cut_all=config.jieba_cut_all)),
            config=config,
            operation_name="jieba_entity_cut",
        )
    except Exception as exc:
        logger.warning("jieba_entity_cut_failed", error=str(exc))
        terms = re.findall(r"[\u4e00-\u9fff]+", text)

    entities: list[Entity] = []
    cursor = 0
    for term in terms:
        clean = term.strip()
        if len(clean) < config.rule_zh_min_term_len:
            continue
        start = text.find(clean, cursor)
        end = start + len(clean) if start >= 0 else start
        if start >= 0:
            cursor = end
        entities.append(
            make_entity(
                name=clean,
                entity_type="term",
                confidence=config.rule_entity_default_confidence,
                extractor="rule_zh",
                offsets=[(start, end)] if start >= 0 else None,
            )
        )
    return entities


def dedup_entities(entities: list[Entity]) -> list[Entity]:
    """Deduplicate entities by canonical name and type, keeping first mention."""

    deduped: list[Entity] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        key = (entity.canonical_name, entity.entity_type)
        if key in seen or not entity.canonical_name:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped
