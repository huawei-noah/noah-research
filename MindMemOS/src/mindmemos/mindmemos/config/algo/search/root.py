"""Root search algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .agentic import AgenticConfig
from .default_search import DefaultSearchConfig
from .rerank import RerankConfig
from .schema import SchemaSearchConfig
from .vanilla import VanillaSearchConfig


@dataclass
class SearchConfig:
    """Top-level search algorithm configuration."""

    request_top_k_max: int = field(default=100)
    """Maximum ``top_k`` accepted from public search requests."""

    include_patches: bool = field(default=True)
    """Deprecated compatibility field; public search no longer returns archived patch versions."""

    default: DefaultSearchConfig = field(default_factory=DefaultSearchConfig)
    """BM25 default search pipeline configuration."""

    vanilla: VanillaSearchConfig = field(default_factory=VanillaSearchConfig)
    """Vanilla hybrid search pipeline configuration."""

    schema_search: SchemaSearchConfig = field(default_factory=SchemaSearchConfig)
    """Schema-aware search pipeline configuration."""

    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    """Agentic search pipeline configuration."""

    rerank: RerankConfig = field(default_factory=RerankConfig)
    """Shared reranking configuration."""
