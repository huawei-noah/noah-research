# src/models/semantic.py
"""
语义记忆模型
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime
import uuid


@dataclass
class SemanticMemory:
    """语义记忆"""
    
    memory_id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    user_id: str = ""
    content: str = ""
    memory_type: str = "fact"  
    importance: float = 0.5    
    timestamp: datetime = field(default_factory=datetime.now)  
    source_episode_ids: List[str] = field(default_factory=list)  
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.user_id:
            raise ValueError("user_id 不能为空")
        if not self.content:
            raise ValueError("content 不能为空")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance 必须在 [0.0, 1.0] 范围内")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'memory_id': self.memory_id,
            'user_id': self.user_id,
            'content': self.content,
            'memory_type': self.memory_type,
            'importance': self.importance,
            'timestamp': self.timestamp.isoformat(),
            'source_episode_ids': self.source_episode_ids,
            'tags': self.tags,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticMemory':
        """从字典创建"""
        data = data.copy()
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def update_importance(self, new_importance: float) -> None:
        """更新重要性"""
        if not 0.0 <= new_importance <= 1.0:
            raise ValueError("importance 必须在 [0.0, 1.0] 范围内")
        self.importance = new_importance
    
    def add_source_episode(self, episode_id: str) -> None:
        """添加源 Episode"""
        if episode_id not in self.source_episode_ids:
            self.source_episode_ids.append(episode_id)
    
    def add_tag(self, tag: str) -> None:
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)


__all__ = ['SemanticMemory']