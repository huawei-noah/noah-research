# storage/__init__.py
"""
Storage module

Provides three-layer indexing architecture:
- L1: Temporal Index (time-based filtering)
- L2: Tag DAG Index (semantic tag filtering)
- L3: Vector Index (locality-aware embedding search)
"""

from .unified_index import UnifiedIndex
from .indexing.temporal_index import TemporalIndex, TimeInterval
from .indexing.tag_dag_index import TagDAGIndex
from .indexing.vector_index import LocalityAwareVectorIndex

__all__ = [
    'UnifiedIndex',
    'TemporalIndex',
    'TimeInterval',
    'TagDAGIndex',
    'LocalityAwareVectorIndex',
]

