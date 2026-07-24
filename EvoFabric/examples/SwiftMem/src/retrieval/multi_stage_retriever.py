"""
Multi-Stage Retriever

Three-stage retrieval pipeline:
1. Temporal filtering using multi-time-range index
2. Semantic routing via TAG-DAG
3. Vector search + reranking
"""

import time
from datetime import datetime
from typing import List, Optional, Set, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

from ..storage.unified_index import UnifiedIndex
from ..storage.indexing.temporal_index import TemporalIndex, TimeInterval
from ..storage.indexing.tag_dag_index import TagDAGIndex
from ..storage.indexing.vector_index import LocalityAwareVectorIndex
from ..storage.backends.episode_storage import EpisodeStorage
from ..storage.backends.semantic_storage import SemanticStorage
from ..embeddings.embedding_manager import EmbeddingManager
from .types import (
    TimeRange, SearchResult, RetrievalConfig, 
    RetrievalMetrics, RetrievalStage
)
from .reranker import Reranker


@dataclass
class StageResults:
    """每阶段的详细结果"""
    memory_ids: List[str]
    scores: Dict[str, float]  # memory_id -> score
    metadata: Dict[str, Any]  # 额外信息


@dataclass
class DetailedRetrievalResult:
    """包含每阶段详细结果"""
    final_results: List[SearchResult]
    metrics: RetrievalMetrics
    stage1_results: StageResults
    stage2_results: StageResults
    stage3_results: StageResults


class MultiStageRetriever:
    """
    多阶段检索器
    
    核心创新：
    1. Multi-time-range temporal filtering
    2. TAG-DAG semantic routing (vs Neeko's full scan)
    3. Embedding co-consolidation aware (optimized layout)
    """
    
    def __init__(
        self,
        temporal_index: TemporalIndex,
        tag_dag_index: TagDAGIndex,
        vector_index: LocalityAwareVectorIndex,
        episode_storage: EpisodeStorage,
        semantic_storage: SemanticStorage,
        embedding_manager: EmbeddingManager,
        unified_index: Optional['UnifiedIndex'] = None, 
        config: Optional[RetrievalConfig] = None
    ):
        self.temporal_index = temporal_index
        self.tag_dag_index = tag_dag_index
        self.vector_index = vector_index
        self.episode_storage = episode_storage
        self.semantic_storage = semantic_storage
        self.embedding_manager = embedding_manager
        self.unified_index = unified_index  
        
        self.config = config or RetrievalConfig()
        self.reranker = Reranker(self.config)

    def _stage2_direct_load(
        self,
        candidate_ids: Optional[Set[str]],
        metrics: RetrievalMetrics
    ) -> List[SearchResult]:
        """
        Stage2 足够时，直接加载候选记忆（不计算向量相似度）
        
        交给 Stage 4 的 BM25 rerank 来排序
        
        性能目标: < 50ms
        
        Args:
            candidate_ids: Stage 2 候选集
            metrics: 性能指标
            
        Returns:
            未排序的 SearchResult 列表（score 设为 1.0）
        """
        start_time = time.time()
        
        if not candidate_ids:
            metrics.vector_candidates = 0
            metrics.vector_time_ms = 0
            return []
        
        # 批量加载候选的 memory_data
        results = []
        for memory_id in candidate_ids:
            memory_data = self._get_memory_data(memory_id)
            if not memory_data:
                continue
            
            # 创建 SearchResult（score 暂时设为 1.0）
            result = SearchResult(
                memory_id=memory_id,
                content=memory_data['content'],
                timestamp=memory_data['timestamp'],
                tags=memory_data.get('tags', []),
                score=1.0,  # 占位分数，由 rerank 重新计算
                embedding=None,
                metadata=memory_data.get('metadata', {})
            )
            # 不设置 stage_scores，让 rerank 来决定
            results.append(result)
        
        metrics.vector_candidates = len(results)
        metrics.vector_time_ms = (time.time() - start_time) * 1000
        
        return results

    def _should_use_stage3(
        self,
        stage2_candidate_count: int,
        query_tags: List[str],
        expanded_tags: Set[str]
    ) -> Tuple[bool, str]:
        """
        快速评估是否需要 Stage 3
        
        基于方案 B：标签覆盖度判断
        
        Args:
            stage2_candidate_count: Stage 2 的候选数量
            query_tags: 原始查询标签
            expanded_tags: 扩展后的标签
            
        Returns:
            (是否使用 Stage 3, 原因)
            
        性能目标: < 1ms
        """
        # 规则 1: 候选太少，必须用 Stage 3 补充
        if stage2_candidate_count < self.config.stage3_min_candidates:
            return True, f"stage2_too_few({stage2_candidate_count}<{self.config.stage3_min_candidates})"
        
        # 规则 2: 没有查询标签，无法评估覆盖度，保守起见用 Stage 3
        if not query_tags:
            return True, "no_query_tags"
        
        # 规则 3: 标签覆盖度判断
        query_tag_set = set(query_tags)
        matched_tags = query_tag_set & expanded_tags
        
        coverage = len(matched_tags) / len(query_tag_set) if query_tag_set else 0.0
        
        if coverage < self.config.stage3_tag_coverage_threshold:
            return True, f"low_coverage({coverage:.2f}<{self.config.stage3_tag_coverage_threshold})"
        
        # 规则 4: 候选数量适中（20-50），总是用 Stage 3 确保质量
        if stage2_candidate_count < 50:
            return True, f"moderate_candidates({stage2_candidate_count})"
        
        # 否则跳过 Stage 3（Stage 2 已足够好）
        return False, f"stage2_sufficient({stage2_candidate_count},coverage={coverage:.2f})"
    
    def _get_memory_data(self, memory_id: str) -> Optional[dict]:
        """
        从 storage 获取记忆数据
        
        尝试从 episode storage 和 semantic storage 中查找
        """
        # 尝试从 episode storage
        episode = self.episode_storage.load_episode(memory_id)
        if episode:
            return {
                'content': episode.content or self._format_episode_content(episode),
                'timestamp': episode.timestamp,
                'tags': episode.tags,
                'metadata': episode.metadata,
                'type': 'episode'
            }
        
        # 尝试从 semantic storage
        semantic = self.semantic_storage.load_memory(memory_id)
        if semantic:
            return {
                'content': semantic.content,
                'timestamp': semantic.timestamp,
                'tags': semantic.tags,
                'metadata': semantic.metadata,
                'type': 'semantic'
            }
        
        return None

    def _get_memory_data_lightweight(self, memory_id: str) -> Optional[dict]:
        """
        轻量级加载记忆数据（只加载必要字段）
        """
        # 尝试从 episode storage
        episode = self.episode_storage.load_episode(memory_id)
        if episode:
            return {
                'content': episode.content or self._format_episode_content(episode),
                'tags': episode.tags,
                'timestamp': episode.timestamp
            }
        
        # 尝试从 semantic storage
        semantic = self.semantic_storage.load_memory(memory_id)
        if semantic:
            return {
                'content': semantic.content,
                'tags': semantic.tags,
                'timestamp': semantic.timestamp
            }
        
        return None

    def _batch_get_memory_data_fast(
        self,
        memory_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        从内存索引批量加载（O(n) 但常数极小）
        
        优先从 UnifiedIndex._episode_metadata 获取
        如果缺失，降级到 SQLite 查询
        
        性能目标: < 1ms for 100 items
        
        Args:
            memory_ids: Memory IDs
            
        Returns:
            [{memory_id, content, tags}, ...]
        """
        # 优先从内存索引获取
        if hasattr(self, 'unified_index') and self.unified_index:
            candidates_data = self.unified_index.batch_get_memory_data(memory_ids)
            
            # 检查是否有缺失
            loaded_ids = {item['memory_id'] for item in candidates_data}
            missing_ids = [mid for mid in memory_ids if mid not in loaded_ids]
            
            # 如果有缺失，降级到 SQLite
            if missing_ids:
                for memory_id in missing_ids:
                    data = self._get_memory_data_lightweight(memory_id)
                    if data:
                        candidates_data.append({
                            'memory_id': memory_id,
                            'content': data['content'],
                            'tags': data['tags']
                        })
            
            return candidates_data
        else:
            # 没有 unified_index，降级到逐个查询
            candidates_data = []
            for memory_id in memory_ids:
                data = self._get_memory_data_lightweight(memory_id)
                if data:
                    candidates_data.append({
                        'memory_id': memory_id,
                        'content': data['content'],
                        'tags': data['tags']
                    })
            return candidates_data

    def _format_episode_content(self, episode) -> str:
        """
        格式化 Episode 内容
        
        如果 Episode.content 为空，从 messages 生成摘要
        """
        if episode.content:
            return episode.content
        
        # 从 messages 生成简单摘要
        if not episode.messages:
            return episode.title or ""
        
        # 拼接前几条消息
        content_parts = []
        for msg in episode.messages[:3]:  # 只取前3条
            content_parts.append(f"{msg.role}: {msg.content[:100]}")
        
        return "\n".join(content_parts) + ("..." if len(episode.messages) > 3 else "")

    def _count_total_memories(self, user_id: str) -> int:
        """
        统计总记忆数
        
        包含 episodes 和 semantic memories
        """
        episode_count = self.episode_storage.count_episodes(user_id)
        semantic_count = self.semantic_storage.count_memories(user_id)
        return episode_count + semantic_count

    def _sort_by_tag_coverage(
        self,
        memory_ids: Set[str],
        query_tags: List[str],
        expanded_tags: Set[str],
        limit: int
    ) -> List[str]:
        """
        按 tag 覆盖度排序候选记忆
        
        覆盖度计算：
        - 原始 query tags 的匹配权重 = 1.0
        - 扩展 tags 的匹配权重 = 0.5
        - score = (原始匹配数 * 1.0 + 扩展匹配数 * 0.5) / len(query_tags)
        
        性能目标: < 5ms for 300 candidates
        
        Args:
            memory_ids: 候选 memory IDs
            query_tags: 原始查询标签
            expanded_tags: 扩展后的标签（包含原始标签）
            limit: 保留的最大数量
            
        Returns:
            排序后的 memory_ids 列表（最多 limit 个）
        """
        if len(memory_ids) <= limit:
            return list(memory_ids)
        
        # 快速计算每个 memory 的匹配分数
        scores = []
        query_tag_set = set(query_tags)
        expanded_only = expanded_tags - query_tag_set  # 只包含扩展的 tags
        
        for memory_id in memory_ids:
            # 从 tag_dag_index 的 episode_to_tags 获取（O(1)）
            memory_tags = self.tag_dag_index.episode_to_tags.get(memory_id, set())
            
            # 计算匹配分数
            original_match = len(query_tag_set & memory_tags)
            expanded_match = len(expanded_only & memory_tags)
            
            # 归一化分数
            if query_tags:
                score = (original_match * 1.0 + expanded_match * 0.5) / len(query_tags)
            else:
                score = 0.0
            
            scores.append((memory_id, score))
        
        # 排序并截断（按分数降序）
        scores.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in scores[:limit]]


    def retrieve(
        self,
        query: str,
        user_id: str,
        time_ranges: Optional[List[TimeRange]] = None,
        query_tags: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        return_metrics: bool = False,
        return_detailed: bool = False
    ):
        start_time = time.time()
        
        if top_k is None:
            top_k = self.config.rerank_top_k
        
        # 初始化指标
        metrics = RetrievalMetrics(
            total_memories=self._count_total_memories(user_id),
            temporal_filtered=0,
            semantic_filtered=0,
            vector_candidates=0,
            final_results=0
        )
        
        # Stage 1: Temporal Filtering
        candidate_ids = self._temporal_filter(user_id, time_ranges, metrics)
        
        stage1_results = StageResults(
            memory_ids=list(candidate_ids) if candidate_ids else [],
            scores={},
            metadata={"time_ranges": len(time_ranges) if time_ranges else 0}
        )
        
        # Stage 2: Semantic Routing via TAG-DAG
        candidate_ids, candidate_tags = self._semantic_filter(
            candidate_ids, query_tags, metrics
        )
        
        stage2_results = StageResults(
            memory_ids=list(candidate_ids) if candidate_ids else [],
            scores={},
            metadata={
                "query_tags": query_tags or [], 
                "expanded_tags": list(candidate_tags)
            }
        )
        
        # Stage 3 判断
        # stage2_count = len(candidate_ids) if candidate_ids else 0
        # use_stage3, reason = self._should_use_stage3(
        #     stage2_candidate_count=stage2_count,
        #     query_tags=query_tags or [],
        #     expanded_tags=candidate_tags
        # )
        
        # # 记录判断原因到 metadata
        # stage2_results.metadata["stage3_decision"] = {
        #     "use_stage3": use_stage3,
        #     "reason": reason
        # }

        # # Stage 3: Vector Search（条件执行）
        # if use_stage3:
        #     results = self._vector_search(
        #         query, candidate_ids, candidate_tags, metrics, stage2_count
        #     )
        # else:
        #     # Stage2 足够，直接加载候选，交给 Stage 4 rerank
        #     results = self._stage2_direct_load(
        #         candidate_ids, metrics
        #     )
        
        # stage3_results = StageResults(
        #     memory_ids=[r.memory_id for r in results],
        #     scores={r.memory_id: r.score for r in results},
        #     metadata={
        #         "used_stage3": use_stage3,
        #         "method": "vector_search" if use_stage3 else "stage2_direct_load"
        #     }
        # )
    
        stage3_results = stage2_results


        # Stage 4: Reranking
        rerank_start = time.time()

        if self.config.use_reranking and stage2_results.memory_ids:
            # 使用快速批量加载
            candidates_data = self._batch_get_memory_data_fast(
                stage2_results.memory_ids
            )
            
            get_mem_end = time.time()
            metrics.get_mem_time_ms = (get_mem_end - rerank_start) * 1000  
            
            # 调用 reranker（无需传递 load_fn）
            top_memory_ids = self.reranker.rerank_from_stage_results(
                query=query,
                stage2_results=stage2_results,
                candidates_data=candidates_data,
                top_k=top_k
            )
            
            metrics.rerank_time_ms = (time.time() - get_mem_end) * 1000  
        else:
            # 不使用 reranking，直接取前 top_k
            top_memory_ids = stage2_results.memory_ids[:top_k]
            metrics.get_mem_time_ms = 0.0  #  没有加载，设为 0
            metrics.rerank_time_ms = 0.0   # 没有 rerank，设为 0

        # 在统计 metrics 之前结束计时
        metrics.total_time_ms = (time.time() - start_time) * 1000

        metrics.final_results = len(top_memory_ids)

        results = []
        for memory_id in top_memory_ids:
            memory_data = self._get_memory_data(memory_id)
            if memory_data:
                result = SearchResult(
                    memory_id=memory_id,
                    content=memory_data['content'],
                    timestamp=memory_data['timestamp'],
                    tags=memory_data.get('tags', []),
                    score=1.0,
                    embedding=None,
                    metadata=memory_data.get('metadata', {})
                )
                result.stage_scores[RetrievalStage.RERANK] = 1.0
                results.append(result)

        # 返回结果
        if return_detailed:
            detailed = DetailedRetrievalResult(
                final_results=results,
                metrics=metrics,
                stage1_results=stage1_results,
                stage2_results=stage2_results,
                stage3_results=stage3_results
            )
            return results, metrics, detailed
        elif return_metrics:
            return results, metrics
        else:
            return results, None
            

    
    def _temporal_filter(
        self,
        user_id: str,
        time_ranges: Optional[List[TimeRange]],
        metrics: RetrievalMetrics
    ) -> Set[str]:
        """
        阶段1: 时间过滤
        
        使用 multi-time-range 查询快速过滤候选
        """
        if not self.config.use_temporal_filter or not time_ranges:

            metrics.temporal_filtered = metrics.total_memories
            return None  
        
        start_time = time.time()
        
        # 转换为 TemporalIndex 需要的格式
        time_intervals = [
            TimeInterval(start=tr.start, end=tr.end) 
            for tr in time_ranges
        ]
        
        # 使用 multi-range 查询
        candidate_ids = self.temporal_index.query_multi_range(
            user_id=user_id,
            time_ranges=time_intervals,
            limit=self.config.max_temporal_candidates
        )
        
        metrics.temporal_filtered = len(candidate_ids)
        metrics.temporal_time_ms = (time.time() - start_time) * 1000
        
        return candidate_ids



    def _semantic_filter(
        self,
        candidate_ids: Optional[Set[str]],
        query_tags: Optional[List[str]],
        metrics: RetrievalMetrics
    ) -> Tuple[Optional[Set[str]], Set[str]]:
        """
        阶段2: 语义路由（TAG-DAG）
        
        优化：当候选过多时，按 tag 覆盖度排序后截断
        """
        if not self.config.use_semantic_routing or not query_tags:
            if candidate_ids is not None:
                metrics.semantic_filtered = len(candidate_ids)
            else:
                metrics.semantic_filtered = metrics.temporal_filtered
            return candidate_ids, set()
        
        start_time = time.time()
        
        # 扩展 tags（通过 DAG 关系）
        expanded_tags = self._expand_tags(query_tags)
        
        # 获取这些 tags 关联的 memory IDs
        tag_related_ids = self.tag_dag_index.query_by_tags(
            query_tags=list(expanded_tags),
            expand_depth=0
        )
        
        # 如果有时间过滤的候选，取交集
        if candidate_ids is not None:
            filtered_ids = candidate_ids & tag_related_ids
        else:
            filtered_ids = tag_related_ids
        
        # 新增：智能排序 + 截断
        if filtered_ids and len(filtered_ids) > self.config.stage2_max_candidates:
            # 按 tag 覆盖度排序
            sorted_ids = self._sort_by_tag_coverage(
                memory_ids=filtered_ids,
                query_tags=query_tags,
                expanded_tags=expanded_tags,
                limit=self.config.stage2_max_candidates
            )
            filtered_ids = set(sorted_ids)  # 转回 set
        
        metrics.semantic_filtered = len(filtered_ids)
        metrics.semantic_time_ms = (time.time() - start_time) * 1000
        
        return filtered_ids, expanded_tags
    



    def _expand_tags(self, query_tags: List[str]) -> Set[str]:
        """
        通过 TAG-DAG 扩展 tags
        
        向下扩展 N 层子节点
        """
        expanded = set(query_tags)
        
        for tag in query_tags:
            # 获取相关标签（包括子标签）
            related = self.tag_dag_index.get_related_tags(
                tag=tag,
                relation_type='child',  # 只获取更具体的子标签
                max_distance=self.config.tag_expansion_depth
            )
            expanded.update(related)
        
        # 更激进的数量限制
        if len(expanded) > self.config.max_expanded_tags:
            priority_tags = set(query_tags)
            remaining = list(expanded - priority_tags)
            
            max_additional = self.config.max_expanded_tags - len(priority_tags)
            expanded = priority_tags | set(remaining[:max_additional])
        
        return expanded


    def _vector_search(
        self,
        query: str,
        candidate_ids: Optional[Set[str]],
        candidate_tags: Set[str],
        metrics: RetrievalMetrics,
        stage2_count: int = 0  # Stage 2 候选数（用于动态 top_k）
    ) -> List[SearchResult]:
        """
        阶段3: 向量检索
        
        在过滤后的候选集中进行向量检索
        """
        start_time = time.time()
        
        # 获取 query embedding
        query_embedding = self.embedding_manager.get_embedding(query)
        
        # 动态调整 top_k
        if self.config.vector_dynamic_topk:
            if stage2_count < self.config.stage3_min_candidates:
                # 候选太少，扩大搜索
                vector_top_k = int(self.config.vector_top_k * self.config.vector_top_k_multiplier * 2)
            elif stage2_count < 50:
                vector_top_k = int(self.config.vector_top_k * self.config.vector_top_k_multiplier)
            else:
                vector_top_k = self.config.vector_top_k
        else:
            vector_top_k = self.config.vector_top_k
        
        # 向量检索（使用 tag-aware search）
        if candidate_tags:
            search_results = self.vector_index.search(
                query_embedding=query_embedding,
                candidate_tags=candidate_tags,
                top_k=vector_top_k  # 使用动态 top_k
            )
        else:
            all_tags = set(self.tag_dag_index.tag_nodes.keys())
            search_results = self.vector_index.search(
                query_embedding=query_embedding,
                candidate_tags=all_tags if all_tags else {'default'},
                top_k=vector_top_k  # 使用动态 top_k
            )
        
        # 分层过滤 + 补充候选（逻辑不变，只是用新的 min_candidates）
        if candidate_ids is not None:
            in_candidates = [
                (memory_id, score) 
                for memory_id, score in search_results
                if memory_id in candidate_ids
            ]
            
            if self.config.vector_supplement_candidates and \
            len(in_candidates) < self.config.vector_min_candidates:
                
                out_candidates = [
                    (memory_id, score) 
                    for memory_id, score in search_results
                    if memory_id not in candidate_ids
                ]
                
                max_supplement = self.config.vector_min_candidates - len(in_candidates)
                search_results = in_candidates + out_candidates[:max_supplement]
            else:
                search_results = in_candidates
        
        metrics.vector_candidates = len(search_results)
        metrics.vector_time_ms = (time.time() - start_time) * 1000
        
        # 转换为 SearchResult 格式
        results = []
        for memory_id, score in search_results:
            memory_data = self._get_memory_data(memory_id)
            
            if memory_data:
                result = SearchResult(
                    memory_id=memory_id,
                    content=memory_data['content'],
                    timestamp=memory_data['timestamp'],
                    tags=memory_data.get('tags', []),
                    score=score,
                    embedding=None,
                    metadata=memory_data.get('metadata', {})
                )
                result.stage_scores[RetrievalStage.VECTOR] = score
                results.append(result)
        
        # 过滤低相似度结果
        results = [
            r for r in results 
            if r.score >= self.config.vector_similarity_threshold
        ]
        
        return results

 


    
    
    


    