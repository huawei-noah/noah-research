# src/core/types.py
"""
核心数据类型定义
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import uuid

class MemoryType(Enum):
    """Memory类型枚举"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class IndexType(Enum):
    """索引类型枚举"""
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    EMBEDDING = "embedding"


@dataclass
class TimeRange:
    """时间范围"""
    start: datetime
    end: datetime
    
    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("start time must be before end time")
    
    def contains(self, timestamp: datetime) -> bool:
        """检查时间戳是否在范围内"""
        return self.start <= timestamp <= self.end
    
    def overlaps(self, other: 'TimeRange') -> bool:
        """检查是否与另一个时间范围重叠"""
        return not (self.end < other.start or self.start > other.end)
    
    def duration_seconds(self) -> float:
        """获取时间范围的持续时间（秒）"""
        return (self.end - self.start).total_seconds()
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'TimeRange':
        """从字典创建"""
        return cls(
            start=datetime.fromisoformat(data["start"]),
            end=datetime.fromisoformat(data["end"])
        )


@dataclass
class MultiTimeRange:
    """多时间范围查询"""
    ranges: List[TimeRange] = field(default_factory=list)
    
    def add_range(self, start: datetime, end: datetime) -> None:
        """添加时间范围"""
        self.ranges.append(TimeRange(start, end))
    
    def contains(self, timestamp: datetime) -> bool:
        """检查时间戳是否在任意范围内"""
        return any(r.contains(timestamp) for r in self.ranges)
    
    def merge_overlapping(self) -> 'MultiTimeRange':
        """合并重叠的时间范围"""
        if not self.ranges:
            return MultiTimeRange()
        
        # 按开始时间排序
        sorted_ranges = sorted(self.ranges, key=lambda r: r.start)
        merged = [sorted_ranges[0]]
        
        for current in sorted_ranges[1:]:
            last = merged[-1]
            if current.start <= last.end:
                # 合并重叠范围
                merged[-1] = TimeRange(
                    start=last.start,
                    end=max(last.end, current.end)
                )
            else:
                merged.append(current)
        
        return MultiTimeRange(ranges=merged)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "ranges": [r.to_dict() for r in self.ranges]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiTimeRange':
        """从字典创建"""
        return cls(
            ranges=[TimeRange.from_dict(r) for r in data["ranges"]]
        )


@dataclass
class MemoryQuery:
    """统一的Memory查询接口"""
    user_id: str
    query_text: Optional[str] = None
    time_ranges: Optional[MultiTimeRange] = None
    tags: Optional[List[str]] = None
    memory_type: Optional[MemoryType] = None
    limit: int = 10
    similarity_threshold: float = 0.7
    
    # 高级查询选项
    use_temporal_index: bool = True
    use_semantic_index: bool = True
    use_embedding_search: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "query_text": self.query_text,
            "time_ranges": self.time_ranges.to_dict() if self.time_ranges else None,
            "tags": self.tags,
            "memory_type": self.memory_type.value if self.memory_type else None,
            "limit": self.limit,
            "similarity_threshold": self.similarity_threshold,
            "use_temporal_index": self.use_temporal_index,
            "use_semantic_index": self.use_semantic_index,
            "use_embedding_search": self.use_embedding_search
        }


@dataclass
class MemorySearchResult:
    """Memory搜索结果"""
    memory_id: str
    memory_type: MemoryType
    content: str
    timestamp: datetime
    score: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 调试信息
    matched_by: List[str] = field(default_factory=list)  # 匹配来源: temporal, semantic, embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "tags": self.tags,
            "metadata": self.metadata,
            "matched_by": self.matched_by
        }


@dataclass
class IndexStats:
    """索引统计信息"""
    index_type: IndexType
    total_entries: int
    memory_usage_bytes: int
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "index_type": self.index_type.value,
            "total_entries": self.total_entries,
            "memory_usage_bytes": self.memory_usage_bytes,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }


__all__ = [
    'MemoryType',
    'IndexType',
    'TimeRange',
    'MultiTimeRange',
    'MemoryQuery',
    'MemorySearchResult',
    'IndexStats'
]




@dataclass
class RetrievalResult:
    """检索结果"""
    memory_id: str
    content: str
    memory_type: str  # 'episode' 或 'semantic'
    score: float  # 相关性得分
    importance: float
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'memory_id': self.memory_id,
            'content': self.content,
            'memory_type': self.memory_type,
            'score': self.score,
            'importance': self.importance,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }