
# storage/indexing/__init__.py
"""
Three-layer indexing system

L1: Temporal Index - Fast time range queries
L2: Tag DAG Index - Semantic tag hierarchy
L3: Vector Index - Locality-aware embedding storage
"""

from .temporal_index import TemporalIndex, TimeInterval, TemporalNode
from .tag_dag_index import TagDAGIndex, TagNode
from .vector_index import LocalityAwareVectorIndex, VectorBlock

__all__ = [
    # L1: Temporal
    'TemporalIndex',
    'TimeInterval',
    'TemporalNode',
    
    # L2: Tag DAG
    'TagDAGIndex',
    'TagNode',
    
    # L3: Vector
    'LocalityAwareVectorIndex',
    'VectorBlock',
]