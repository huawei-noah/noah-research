# src/models/episode.py
"""
Episode模型 - 一段连续的对话片段
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import uuid

from .message import Message


@dataclass
class Episode:
    """对话片段（Episode）"""
    
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    messages: List[Message] = field(default_factory=list)
    title: str = ""
    content: str = ""  # Episode的摘要/内容
    boundary_reason: str = ""  # 为什么在此处分段
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    tag_relations: List[Tuple[str, str]] = field(default_factory=list)  # 标签层级关系
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def start_time(self) -> Optional[datetime]:
        """Episode开始时间"""
        if self.messages:
            return self.messages[0].timestamp
        return self.timestamp
    
    @property
    def end_time(self) -> Optional[datetime]:
        """Episode结束时间"""
        if self.messages:
            return self.messages[-1].timestamp
        return self.timestamp
    
    @property
    def duration_seconds(self) -> float:
        """Episode持续时间（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @classmethod
    def from_messages(
        cls,
        messages: List[Message],
        user_id: str,
        title: str = "",
        content: str = "",
        boundary_reason: str = "",
        timestamp: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        tag_relations: Optional[List[Tuple[str, str]]] = None  # 参数
    ) -> 'Episode':
        """从消息列表创建Episode"""
        if not messages:
            raise ValueError("消息列表不能为空")
        
        return cls(
            user_id=user_id,
            messages=messages,
            title=title,
            content=content,
            boundary_reason=boundary_reason,
            timestamp=timestamp or messages[0].timestamp,
            tags=tags or [],
            tag_relations=tag_relations or []  # 
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "episode_id": self.episode_id,
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "title": self.title,
            "content": self.content,
            "boundary_reason": self.boundary_reason,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "tag_relations": [list(r) for r in self.tag_relations],  # 转换为列表
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """从字典创建"""
        # 处理 tag_relations
        tag_relations_data = data.get("tag_relations", [])
        tag_relations = [tuple(r) for r in tag_relations_data] if tag_relations_data else []
        
        return cls(
            episode_id=data["episode_id"],
            user_id=data["user_id"],
            messages=[Message.from_dict(m) for m in data["messages"]],
            title=data["title"],
            content=data["content"],
            boundary_reason=data["boundary_reason"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tags=data.get("tags", []),
            tag_relations=tag_relations,  # 
            metadata=data.get("metadata", {})
        )