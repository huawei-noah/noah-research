"""
Retrieval-related type definitions
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class RetrievalStage(Enum):
    """检索阶段"""
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    VECTOR = "vector"
    RERANK = "rerank"


@dataclass
class TimeRange:
    """时间范围"""
    start: datetime
    end: datetime
    
    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("start must be before end")
    
    def contains(self, timestamp: datetime) -> bool:
        """检查时间戳是否在范围内"""
        return self.start <= timestamp <= self.end
    
    def overlaps(self, other: 'TimeRange') -> bool:
        """检查两个时间范围是否重叠"""
        return self.start <= other.end and other.start <= self.end


@dataclass
class SearchResult:
    """检索结果"""
    memory_id: str
    content: str
    timestamp: datetime
    tags: List[str]
    score: float
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 检索过程信息
    stage_scores: Dict[RetrievalStage, float] = field(default_factory=dict)
    matched_time_ranges: List[TimeRange] = field(default_factory=list)
    matched_tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'memory_id': self.memory_id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'score': self.score,
            'metadata': self.metadata,
            'stage_scores': {
                stage.value: score 
                for stage, score in self.stage_scores.items()
            }
        }


@dataclass
class RetrievalConfig:
    """检索配置"""
    # Temporal filtering
    use_temporal_filter: bool = True
    max_temporal_candidates: int = 1000
    
    # Semantic routing
    use_semantic_routing: bool = True
    max_tags_per_query: int = 10
    tag_expansion_depth: int = 1  # TAG-DAG 中向下扩展的深度0
    # Stage 2 候选数上限
    stage2_max_candidates: int = 100
        
    # 扩展后 tag 数量上限
    max_expanded_tags: int = 10
    
    # Vector search
    vector_top_k: int = 50
    vector_similarity_threshold: float = 0.1

    vector_top_k_multiplier: float = 1.5  # Stage 2 候选少时的倍数
    vector_dynamic_topk: bool = True
    
    # Reranking
    use_reranking: bool = True
    rerank_top_k: int = 20
    
    # 混合权重
    bm25_weight: float = 0.35
    vector_weight: float = 0.50
    tag_weight: float = 0.15
    
    # 时间衰减
    time_decay_factor: float = 0.95  # 每天的衰减系数

    stage3_min_candidates: int = 20  # Stage 2 结果少于此值时必用 Stage 3
    stage3_tag_coverage_threshold: float = 0.3  # 标签覆盖度低于此值时用 Stage 3

    vector_supplement_candidates: bool = True
    vector_min_candidates: int = 30  # 最少保留多少候选





@dataclass
class RetrievalMetrics:
    """检索性能指标"""
    total_memories: int
    temporal_filtered: int
    semantic_filtered: int
    vector_candidates: int
    final_results: int
    
    temporal_time_ms: float = 0.0
    semantic_time_ms: float = 0.0
    vector_time_ms: float = 0.0
    get_mem_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    def efficiency_ratio(self) -> float:
        """计算效率比（最终结果数 / 总记忆数）"""
        return self.final_results / self.total_memories if self.total_memories > 0 else 0.0
    
    def filter_reduction_ratio(self) -> float:
        """计算过滤减少比例"""
        if self.total_memories == 0:
            return 0.0
        return (self.total_memories - self.vector_candidates) / self.total_memories
    
    def __str__(self) -> str:
        return (
            f"RetrievalMetrics(\n"
            f"  Total Memories: {self.total_memories}\n"
            f"  After Temporal Filter: {self.temporal_filtered} "
            f"({self.temporal_filtered/self.total_memories*100:.1f}%)\n"
            f"  After Semantic Filter: {self.semantic_filtered} "
            f"({self.semantic_filtered/self.total_memories*100:.1f}%)\n"
            f"  Vector Candidates: {self.vector_candidates}\n"
            f"  Final Results: {self.final_results}\n"
            f"  ---\n"
            f"  Filter Reduction: {self.filter_reduction_ratio()*100:.1f}%\n"
            f"  ---\n"
            f"  Temporal Time: {self.temporal_time_ms:.2f}ms\n"
            f"  Semantic Time: {self.semantic_time_ms:.2f}ms\n"
            f"  Vector Time: {self.vector_time_ms:.2f}ms\n"
            f"  Rerank Time: {self.rerank_time_ms:.2f}ms\n"
            f"  Total Time: {self.total_time_ms:.2f}ms\n"
            f")"
        )