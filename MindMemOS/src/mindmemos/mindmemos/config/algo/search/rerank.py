"""Reranker configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RerankConfig:
    """Reranker truncation and batching parameters."""

    enabled: bool = field(default=True)
    """Whether external reranker endpoints are enabled."""

    max_query_length: int = field(default=100)
    """Maximum query length before truncation."""

    max_doc_length: int = field(default=5000)
    """Maximum document length before truncation."""

    max_batch_size: int = field(default=20)
    """Auto-split batch size for reranking."""

    max_concurrent_batches: int = field(default=1)
    """Maximum concurrent external rerank batch requests per rerank call."""

    request_timeout: float = field(default=5.0)
    """Timeout in seconds for the external reranker request."""
