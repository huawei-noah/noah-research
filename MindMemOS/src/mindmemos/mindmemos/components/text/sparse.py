"""BM25-like sparse vector encoding."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Protocol

from ...config import TextProcessingConfig
from ...typing import CorpusStats, SparseVector
from ._hashing import digest_text


class CorpusStatsProvider(Protocol):
    """Interface for project-scoped BM25 corpus statistics."""

    def get_stats(self, project_id: str, terms: list[str]) -> CorpusStats:
        """Return corpus stats for the requested terms."""

    def observe_document(self, project_id: str, memory_id: str, terms: list[str]) -> None:
        """Add or replace a document in the corpus statistics."""

    def remove_document(self, project_id: str, memory_id: str) -> None:
        """Remove a document from the corpus statistics."""

    def replace_document(self, project_id: str, memory_id: str, old_terms: list[str], new_terms: list[str]) -> None:
        """Replace a document's terms in the corpus statistics."""


class InMemoryCorpusStatsProvider:
    """In-memory stats provider for tests and local single-process experiments."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, list[str]]] = defaultdict(dict)

    def get_stats(self, project_id: str, terms: list[str]) -> CorpusStats:
        documents = self._documents.get(project_id, {})
        doc_count = len(documents)
        avg_doc_len = sum(len(doc_terms) for doc_terms in documents.values()) / doc_count if doc_count else 0.0
        requested_terms = set(terms)
        document_frequency = {term: 0 for term in requested_terms}

        for doc_terms in documents.values():
            unique_terms = set(doc_terms)
            for term in requested_terms & unique_terms:
                document_frequency[term] += 1

        return CorpusStats(
            project_id=project_id,
            doc_count=doc_count,
            avg_doc_len=avg_doc_len,
            document_frequency=document_frequency,
        )

    def observe_document(self, project_id: str, memory_id: str, terms: list[str]) -> None:
        self._documents[project_id][memory_id] = list(terms)

    def remove_document(self, project_id: str, memory_id: str) -> None:
        self._documents.get(project_id, {}).pop(memory_id, None)

    def replace_document(self, project_id: str, memory_id: str, old_terms: list[str], new_terms: list[str]) -> None:
        self.observe_document(project_id, memory_id, new_terms)


class SparseVectorEncoder:
    """Encode BM25 terms into Qdrant-compatible sparse vector data."""

    def __init__(self, config: TextProcessingConfig):
        self.config = config

    def encode_document(self, terms: list[str], stats: CorpusStats | None = None) -> SparseVector:
        values_by_index: dict[int, float] = defaultdict(float)
        counts = Counter(terms)
        doc_len = len(terms)
        for term, tf in counts.items():
            index = term_to_index(term, self.config.sparse_hash_dim, self.config.sparse_hash_algorithm)
            if stats and stats.doc_count > 0:
                value = bm25_doc_weight(
                    tf=tf,
                    doc_len=doc_len,
                    avg_doc_len=stats.avg_doc_len,
                    k1=self.config.sparse_k1,
                    b=self.config.sparse_b,
                    min_avg_doc_len=self.config.bm25_min_avg_doc_len,
                )
            else:
                value = fallback_weight(tf, mode=self.config.sparse_fallback_mode)
            values_by_index[index] += value
        return to_sorted_sparse_vector(
            values_by_index,
            model=self.config.sparse_bm25_model_name
            if stats and stats.doc_count > 0
            else self.config.sparse_fallback_model_name,
            hash_dim=self.config.sparse_hash_dim,
        )

    def encode_query(self, terms: list[str], stats: CorpusStats | None = None) -> SparseVector:
        values_by_index: dict[int, float] = defaultdict(float)
        counts = Counter(terms)
        for term, tf in counts.items():
            index = term_to_index(term, self.config.sparse_hash_dim, self.config.sparse_hash_algorithm)
            if stats and stats.doc_count > 0:
                value = bm25_query_weight(
                    term=term,
                    stats=stats,
                    idf_smoothing=self.config.bm25_idf_smoothing,
                    min_idf_denominator=self.config.bm25_min_idf_denominator,
                )
            else:
                value = fallback_weight(tf, mode=self.config.sparse_fallback_mode)
            values_by_index[index] += value
        return to_sorted_sparse_vector(
            values_by_index,
            model=self.config.sparse_bm25_model_name
            if stats and stats.doc_count > 0
            else self.config.sparse_fallback_model_name,
            hash_dim=self.config.sparse_hash_dim,
        )


def term_to_index(term: str, hash_dim: int, hash_algorithm: str) -> int:
    """Map a term to a stable sparse vector index."""

    digest = digest_text(term, algorithm=hash_algorithm)
    return int(digest[:8], 16) % hash_dim


def bm25_doc_weight(
    *,
    tf: int,
    doc_len: int,
    avg_doc_len: float,
    k1: float,
    b: float,
    min_avg_doc_len: float,
) -> float:
    """Return BM25 document-side term-frequency normalization."""

    avg_len = max(avg_doc_len, min_avg_doc_len)
    norm = k1 * (1 - b + b * doc_len / avg_len)
    return (tf * (k1 + 1)) / (tf + norm)


def bm25_query_weight(
    *,
    term: str,
    stats: CorpusStats,
    idf_smoothing: float,
    min_idf_denominator: float,
) -> float:
    """Return BM25 query-side IDF weight."""

    df = stats.document_frequency.get(term, 0)
    numerator = stats.doc_count - df + idf_smoothing
    denominator = max(df + idf_smoothing, min_idf_denominator)
    return math.log1p(numerator / denominator)


def fallback_weight(tf: int, *, mode: str) -> float:
    """Return fallback sparse weight when corpus stats are unavailable."""

    if mode == "tf":
        return float(tf)
    if mode == "log_tf":
        return 1.0 + math.log(tf)
    raise ValueError(f"Unsupported sparse fallback mode: {mode}")


def to_sorted_sparse_vector(values_by_index: dict[int, float], *, model: str, hash_dim: int) -> SparseVector:
    """Convert an index-value mapping to a deterministic SparseVector."""

    indices = sorted(index for index, value in values_by_index.items() if value != 0.0)
    return SparseVector(
        indices=indices,
        values=[values_by_index[index] for index in indices],
        model=model,
        hash_dim=hash_dim,
    )
