"""Dreaming and consolidation algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DreamingConfig:
    """Configuration for offline memory consolidation."""

    lookback_days: int = field(default=7)
    """Recent-day window used to select hot entity scopes."""

    max_scopes_per_run: int | None = field(default=None)
    """Maximum number of hot scopes processed in one run; None means unlimited."""

    max_seed_memories: int | None = field(default=None)
    """Maximum number of recent memories scanned when selecting hot scopes; None means unlimited."""

    max_memories_per_scope: int = field(default=40)
    """Maximum number of related memories inserted into one scope prompt."""

    min_scope_updates: int = field(default=1)
    """Minimum recent updates required for a scope to be processed."""

    min_cluster_size: int = field(default=2)
    """Clusters smaller than this value skip consolidation LLM calls."""

    concurrency: int = field(default=8)
    """Maximum number of unique dreaming clusters processed concurrently."""

    consolidation_model: str | None = field(default=None)
    """Optional model name dedicated to consolidation tasks."""

    scope_batch_size: int = field(default=2000)
    """Number of seed memories processed per neo4j query when selecting graph
    scopes.  Larger batches reduce round trips but consume more transaction
    memory.  Reduce for large graphs (e.g. 262k contexts)."""


__all__ = ["DreamingConfig"]
