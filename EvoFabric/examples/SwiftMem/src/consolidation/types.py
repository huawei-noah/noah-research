# src/consolidation/types.py
"""
Co-consolidation 类型定义
"""

from dataclasses import dataclass, field
from typing import Set, List, Dict, Optional
from datetime import datetime
import numpy as np


@dataclass
class TagCluster:
    """
    标签簇 - 语义相关的标签集合
    
    Examples:
        - Programming cluster: {"python", "java", "programming", "coding"}
        - Cooking cluster: {"cooking", "recipe", "food"}
    """
    cluster_id: str
    tags: Set[str]
    centroid_tag: Optional[str] = None  # 簇的中心标签
    cohesion_score: float = 0.0  # 簇内聚度 [0, 1]
    
    def __post_init__(self):
        if not self.centroid_tag and self.tags:
            # 默认选第一个 tag 作为中心
            self.centroid_tag = next(iter(self.tags))
    
    def add_tag(self, tag: str):
        """添加标签到簇"""
        self.tags.add(tag)
    
    def remove_tag(self, tag: str):
        """从簇中移除标签"""
        self.tags.discard(tag)
    
    def size(self) -> int:
        """簇大小"""
        return len(self.tags)


@dataclass
class ConsolidationPlan:
    """
    Consolidation 执行计划
    
    定义如何重排 tag blocks 以优化局部性
    """
    clusters: List[TagCluster]
    cluster_order: List[str]  # cluster_id 的排列顺序
    estimated_improvement: float = 0.0  # 预期的性能提升比例
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_tag_order(self) -> List[str]:
        """
        获取全局 tag 排列顺序
        
        Returns:
            按簇分组的 tag 列表
        """
        tag_order = []
        cluster_map = {c.cluster_id: c for c in self.clusters}
        
        for cluster_id in self.cluster_order:
            if cluster_id in cluster_map:
                # 簇内按字母序排列（保证确定性）
                cluster_tags = sorted(cluster_map[cluster_id].tags)
                tag_order.extend(cluster_tags)
        
        return tag_order


@dataclass
class ConsolidationMetrics:
    """
    Consolidation 指标
    
    用于评估 consolidation 的效果
    """
    # 执行前
    before_num_blocks: int
    before_avg_block_size: float
    before_memory_fragmentation: float  # 内存碎片化程度 [0, 1]
    
    # 执行后
    after_num_blocks: int
    after_avg_block_size: float
    after_memory_fragmentation: float
    
    # 性能提升
    consolidation_time_ms: float
    estimated_search_speedup: float  # 预期搜索加速比
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_improvement(self) -> Dict[str, float]:
        """计算改进指标"""
        return {
            "fragmentation_reduction": self.before_memory_fragmentation - self.after_memory_fragmentation,
            "avg_block_size_increase": self.after_avg_block_size - self.before_avg_block_size,
            "estimated_speedup": self.estimated_search_speedup
        }


@dataclass
class BlockLayoutInfo:
    """
    Block 布局信息
    
    记录每个 tag block 在 consolidated memory 中的位置
    """
    tag: str
    start_offset: int  # 在 consolidated array 中的起始位置
    end_offset: int    # 结束位置
    cluster_id: str    # 所属簇
    num_embeddings: int
    
    def size(self) -> int:
        """Block 大小（embedding 数量）"""
        return self.end_offset - self.start_offset


__all__ = [
    'TagCluster',
    'ConsolidationPlan',
    'ConsolidationMetrics',
    'BlockLayoutInfo'
]