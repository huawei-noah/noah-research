# src/storage/backends/__init__.py
"""
存储后端模块
"""

from .episode_storage import EpisodeStorage
from .semantic_storage import SemanticStorage

__all__ = ['EpisodeStorage', 'SemanticStorage']