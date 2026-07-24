# src/consolidation/__init__.py
"""
Embedding Co-consolidation Module

Offline optimization for embedding storage layout.
"""

from .types import (
    TagCluster,
    ConsolidationPlan,
    ConsolidationMetrics,
    BlockLayoutInfo
)
from .clustering_strategy import TagClusteringStrategy
from .embedding_consolidator import EmbeddingConsolidator


__all__ = [
    'TagCluster',
    'ConsolidationPlan',
    'ConsolidationMetrics',
    'BlockLayoutInfo',
    'TagClusteringStrategy',
    'EmbeddingConsolidator',
]