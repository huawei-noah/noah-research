# src/core/base.py
"""
抽象基类定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from .types import (
    MemoryQuery,
    MemorySearchResult,
    TimeRange,
    MultiTimeRange,
    IndexStats,
    MemoryType
)


class BaseIndex(ABC):
    """索引基类"""
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        self.user_id = user_id
        self.config = config
    
    @abstractmethod
    def add_memory(self, memory_id: str, timestamp: datetime, 
                   content: str, metadata: Dict[str, Any]) -> None:
        """添加memory到索引"""
        pass
    
    @abstractmethod
    def search(self, query: MemoryQuery) -> List[str]:
        """搜索并返回memory_id列表"""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """从索引中删除memory"""
        pass
    
    @abstractmethod
    def get_stats(self) -> IndexStats:
        """获取索引统计信息"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空索引"""
        pass


class TemporalIndexBase(BaseIndex):
    """时间索引基类"""
    
    @abstractmethod
    def query_time_range(self, time_range: TimeRange) -> List[str]:
        """查询单个时间范围"""
        pass
    
    @abstractmethod
    def query_multi_time_ranges(self, multi_range: MultiTimeRange) -> List[str]:
        """查询多个时间范围"""
        pass


class SemanticIndexBase(BaseIndex):
    """语义索引基类"""
    
    @abstractmethod
    def add_tag(self, tag: str, parent_tags: Optional[List[str]] = None) -> None:
        """添加标签到DAG"""
        pass
    
    @abstractmethod
    def get_related_tags(self, tag: str, max_depth: int = 3) -> List[str]:
        """获取相关标签"""
        pass
    
    @abstractmethod
    def query_by_tags(self, tags: List[str]) -> List[str]:
        """通过标签查询memory"""
        pass


class EmbeddingIndexBase(BaseIndex):
    """Embedding索引基类"""
    
    @abstractmethod
    def add_embedding(self, memory_id: str, embedding: List[float], 
                     tags: Optional[List[str]] = None) -> None:
        """添加embedding"""
        pass
    
    @abstractmethod
    def search_similar(self, query_embedding: List[float], 
                      top_k: int = 10,
                      tag_filter: Optional[List[str]] = None) -> List[tuple[str, float]]:
        """相似度搜索"""
        pass
    
    @abstractmethod
    def consolidate(self, tag_groups: Dict[str, List[str]]) -> None:
        """执行consolidation优化"""
        pass


__all__ = [
    'BaseIndex',
    'TemporalIndexBase',
    'SemanticIndexBase',
    'EmbeddingIndexBase'
]