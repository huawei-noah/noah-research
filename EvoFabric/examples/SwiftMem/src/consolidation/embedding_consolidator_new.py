# src/consolidation/embedding_consolidator.py
"""
Embedding Co-consolidation - 核心实现

通过重排 tag blocks 的物理布局来优化检索性能
"""

import time
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from .types import (
    TagCluster,
    ConsolidationPlan,
    ConsolidationMetrics,
    BlockLayoutInfo
)
from .clustering_strategy import TagClusteringStrategy
from ..storage.indexing.vector_index import LocalityAwareVectorIndex, VectorBlock
from ..storage.indexing.tag_dag_index import TagDAGIndex


class EmbeddingConsolidator:
    """
    Embedding Co-consolidation 引擎
    
    核心功能：
    1. 基于 TAG-DAG 对标签进行语义聚类
    2. 重排 tag blocks 使相关标签的 embeddings 物理相邻
    3. 优化内存布局提高缓存局部性
    
    性能目标：
    - Consolidation 时间: O(N) where N = total embeddings
    - 搜索加速: 2-5x (通过减少内存跳跃)
    - 内存开销: 临时 2x（consolidation 过程中）
    """
    
    def __init__(
        self,
        vector_index: LocalityAwareVectorIndex,
        tag_dag_index: TagDAGIndex,
        clustering_strategy: Optional[TagClusteringStrategy] = None
    ):
        """
        初始化 Consolidator
        
        Args:
            vector_index: 向量索引
            tag_dag_index: TAG-DAG 索引
            clustering_strategy: 聚类策略（可选）
        """
        self.vector_index = vector_index
        self.tag_dag_index = tag_dag_index
        
        self.clustering_strategy = clustering_strategy or TagClusteringStrategy(
            min_cluster_size=0,
            max_cluster_size=20,
            min_cohesion=0.0
        )
        
        # 记录历史 consolidation
        self.consolidation_history: List[ConsolidationMetrics] = []
        
        # 当前的 consolidation plan
        self.current_plan: Optional[ConsolidationPlan] = None
        
        # Block 布局信息
        self.block_layout: Dict[str, BlockLayoutInfo] = {}
    
    def analyze(self) -> ConsolidationPlan:
        """
        分析当前布局，生成 consolidation plan
        
        Returns:
            Consolidation 执行计划
        """
        # 1. 获取活跃的 tags（有 embeddings 的）
        tag_blocks = self.vector_index.export_tag_blocks()
        active_tags = set(tag_blocks.keys())
        
        if not active_tags:
            # 没有数据，返回空计划
            return ConsolidationPlan(
                clusters=[],
                cluster_order=[],
                estimated_improvement=0.0
            )
        
        # 2. 对标签进行聚类
        clusters = self.clustering_strategy.cluster_tags(
            tag_dag=self.tag_dag_index,
            active_tags=active_tags
        )
        
        # 3. 确定簇的排列顺序（按内聚度降序）
        sorted_clusters = sorted(
            clusters,
            key=lambda c: c.cohesion_score,
            reverse=True
        )
        cluster_order = [c.cluster_id for c in sorted_clusters]
        
        # 4. 估算性能提升
        estimated_improvement = self._estimate_improvement(
            clusters=sorted_clusters,
            current_layout=tag_blocks
        )
        
        # 5. 创建计划
        plan = ConsolidationPlan(
            clusters=sorted_clusters,
            cluster_order=cluster_order,
            estimated_improvement=estimated_improvement
        )
        
        self.current_plan = plan
        return plan
    
    def consolidate(
        self,
        plan: Optional[ConsolidationPlan] = None,
        force: bool = False
    ) -> ConsolidationMetrics:
        """
        执行 consolidation
        
        Args:
            plan: 执行计划（如果为 None，则先调用 analyze()）
            force: 是否强制执行（即使预期提升不大）
            
        Returns:
            Consolidation 指标
        """
        start_time = time.time()
        
        # 1. 获取执行计划
        if plan is None:
            plan = self.analyze()
        
        # 2. 检查是否值得执行
        if not force and plan.estimated_improvement < 0.1:
            # 预期提升小于 10%，跳过
            raise ValueError(
                f"Estimated improvement ({plan.estimated_improvement:.1%}) too small. "
                f"Use force=True to override."
            )
        
        # 3. 收集执行前的指标
        before_metrics = self._collect_metrics()
        
        # 4. 执行 consolidation
        new_tag_blocks = self._reorganize_blocks(plan)
        
        # 5. 写回 vector index
        self.vector_index.import_tag_blocks(new_tag_blocks, clear_temp=True)
        
        # 6. 更新布局信息
        self._update_block_layout(plan, new_tag_blocks)
        
        # 7. 收集执行后的指标
        after_metrics = self._collect_metrics()
        
        # 8. 计算总指标
        consolidation_time_ms = (time.time() - start_time) * 1000
        
        metrics = ConsolidationMetrics(
            before_num_blocks=before_metrics['num_blocks'],
            before_avg_block_size=before_metrics['avg_block_size'],
            before_memory_fragmentation=before_metrics['fragmentation'],
            after_num_blocks=after_metrics['num_blocks'],
            after_avg_block_size=after_metrics['avg_block_size'],
            after_memory_fragmentation=after_metrics['fragmentation'],
            consolidation_time_ms=consolidation_time_ms,
            estimated_search_speedup=plan.estimated_improvement + 1.0
        )
        
        # 9. 记录历史
        self.consolidation_history.append(metrics)
        
        return metrics
    
    def _reorganize_blocks(
        self,
        plan: ConsolidationPlan
    ) -> Dict[str, VectorBlock]:
        """
        重组 tag blocks
        
        核心逻辑：按簇分组，簇内按字母序，确保物理相邻
        
        Args:
            plan: Consolidation 计划
            
        Returns:
            重组后的 tag blocks
        """
        # 1. 获取当前 blocks
        current_blocks = self.vector_index.export_tag_blocks()
        
        # 2. 按计划重组
        new_blocks = {}
        
        # 获取全局 tag 顺序
        tag_order = plan.get_tag_order()
        
        for tag in tag_order:
            if tag in current_blocks:
                old_block = current_blocks[tag]
                
                # 创建新 block（确保内存连续）
                new_embeddings = np.ascontiguousarray(old_block.embeddings)
                
                new_block = VectorBlock(
                    embeddings=new_embeddings,
                    episode_ids=old_block.episode_ids.copy()
                )
                
                new_blocks[tag] = new_block
        
        # 3. 处理不在任何簇中的孤立 tags
        orphan_tags = set(current_blocks.keys()) - set(tag_order)
        for tag in sorted(orphan_tags):  # 字母序
            old_block = current_blocks[tag]
            new_embeddings = np.ascontiguousarray(old_block.embeddings)
            
            new_block = VectorBlock(
                embeddings=new_embeddings,
                episode_ids=old_block.episode_ids.copy()
            )
            
            new_blocks[tag] = new_block
        
        return new_blocks
    
    def _update_block_layout(
        self,
        plan: ConsolidationPlan,
        new_blocks: Dict[str, VectorBlock]
    ) -> None:
        """
        更新 block 布局信息
        
        记录每个 tag block 在全局数组中的位置（逻辑上）
        """
        self.block_layout = {}
        
        cluster_map = {c.cluster_id: c for c in plan.clusters}
        offset = 0
        
        # 按 tag 顺序记录
        for tag in plan.get_tag_order():
            if tag not in new_blocks:
                continue
            
            block = new_blocks[tag]
            num_embeddings = len(block.episode_ids)
            
            # 找到所属簇
            cluster_id = "orphan"
            for c in plan.clusters:
                if tag in c.tags:
                    cluster_id = c.cluster_id
                    break
            
            layout_info = BlockLayoutInfo(
                tag=tag,
                start_offset=offset,
                end_offset=offset + num_embeddings,
                cluster_id=cluster_id,
                num_embeddings=num_embeddings
            )
            
            self.block_layout[tag] = layout_info
            offset += num_embeddings
    
    def _collect_metrics(self) -> Dict[str, float]:
        """
        收集当前的性能指标
        
        Returns:
            包含各种指标的字典
        """
        stats = self.vector_index.get_stats()
        layout = self.vector_index.get_block_memory_layout()
        
        # 计算内存碎片化程度
        fragmentation = self._compute_fragmentation(layout)
        
        return {
            'num_blocks': stats['total_blocks'],
            'avg_block_size': stats['avg_block_size'],
            'fragmentation': fragmentation
        }
    
    def _compute_fragmentation(
        self,
        layout: Dict[str, Dict[str, any]]
    ) -> float:
        """
        计算内存碎片化程度
        
        策略：检查有多少 blocks 的内存不是连续的
        
        Returns:
            碎片化程度 [0, 1]，越低越好
        """
        if not layout:
            return 0.0
        
        non_contiguous_count = sum(
            1 for info in layout.values()
            if not info['is_contiguous']
        )
        
        fragmentation = non_contiguous_count / len(layout)
        return fragmentation
    
    def _estimate_improvement(
        self,
        clusters: List[TagCluster],
        current_layout: Dict[str, VectorBlock]
    ) -> float:
        """
        估算 consolidation 的性能提升
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not clusters:
            return 0.0
        
        # 1. 平均簇内聚度
        avg_cohesion = sum(c.cohesion_score for c in clusters) / len(clusters)
        
        # 2. 簇的大小分布
        cluster_sizes = [c.size() for c in clusters]
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        
        # 3. 当前碎片化
        current_layout_info = self.vector_index.get_block_memory_layout()
        current_fragmentation = self._compute_fragmentation(current_layout_info)
        
        # ✅ 添加详细日志
        logger.info(f"📊 Consolidation Estimation:")
        logger.info(f"  - Clusters: {len(clusters)}")
        logger.info(f"  - Avg cohesion: {avg_cohesion:.4f}")
        logger.info(f"  - Avg cluster size: {avg_cluster_size:.2f}")
        logger.info(f"  - Max cluster size: {max_cluster_size}")
        logger.info(f"  - Current fragmentation: {current_fragmentation:.4f}")
        
        # 打印前 5 个簇的详细信息
        for i, cluster in enumerate(clusters[:5]):
            logger.info(f"  - Cluster {i}: size={cluster.size()}, cohesion={cluster.cohesion_score:.4f}, tags={list(cluster.tags)[:5]}")
        
        # 综合评分
        improvement = (
            0.3 * avg_cohesion +
            0.2 * min(avg_cluster_size / 10, 1.0) +
            0.5 * current_fragmentation
        )
        
        logger.info(f"  - Estimated improvement: {improvement:.4f} ({improvement:.2%})")
        
        return improvement
    
    def get_consolidation_stats(self) -> Dict[str, any]:
        """
        获取 consolidation 统计信息
        
        Returns:
            统计信息字典
        """
        if not self.consolidation_history:
            return {
                'total_consolidations': 0,
                'avg_consolidation_time_ms': 0.0,
                'avg_search_speedup': 1.0,
                'current_plan': None
            }
        
        total = len(self.consolidation_history)
        avg_time = sum(m.consolidation_time_ms for m in self.consolidation_history) / total
        avg_speedup = sum(m.estimated_search_speedup for m in self.consolidation_history) / total
        
        return {
            'total_consolidations': total,
            'avg_consolidation_time_ms': avg_time,
            'avg_search_speedup': avg_speedup,
            'current_plan': self.current_plan,
            'latest_metrics': self.consolidation_history[-1] if self.consolidation_history else None
        }
    
    def should_consolidate(
        self,
        threshold_improvement: float = 0.15,
        min_blocks: int = 10
    ) -> bool:
        """
        判断是否应该执行 consolidation
        
        Args:
            threshold_improvement: 最小提升阈值
            min_blocks: 最小 block 数量
            
        Returns:
            是否应该 consolidate
        """
        stats = self.vector_index.get_stats()
        
        # 1. Block 数量太少，不需要
        if stats['total_blocks'] < min_blocks:
            return False
        
        # 2. 分析潜在提升
        plan = self.analyze()
        
        # 3. 判断是否值得
        return plan.estimated_improvement >= threshold_improvement


__all__ = ['EmbeddingConsolidator']