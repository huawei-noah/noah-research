"""
Reranker for search results
Combines BM25, vector similarity, and temporal decay
"""

import math
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import Counter
import re
from dataclasses import dataclass

from .types import SearchResult, RetrievalConfig, RetrievalStage

@dataclass
class StageResults:
    """每阶段的详细结果"""
    memory_ids: List[str]
    scores: Dict[str, float]  # memory_id -> score
    metadata: Dict[str, Any]  

class BM25Scorer:
    """BM25 算法实现"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.corpus_size: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词"""
        # 转小写，分词，移除标点
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def fit(self, documents: List[str]):
        """在文档集上训练 BM25"""
        self.corpus_size = len(documents)
        self.doc_lens = []
        word_doc_counts: Dict[str, int] = {}
        
        # 统计每个词在多少文档中出现
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                word_doc_counts[token] = word_doc_counts.get(token, 0) + 1
        
        # 计算平均文档长度
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        
        # 计算 IDF
        for word, doc_count in word_doc_counts.items():
            self.idf[word] = math.log((self.corpus_size - doc_count + 0.5) / (doc_count + 0.5) + 1.0)
    
    def score(self, query: str, document: str, doc_index: int) -> float:
        """计算 BM25 分数"""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)
        doc_len = len(doc_tokens)
        
        # 文档中词频
        doc_term_freqs = Counter(doc_tokens)
        
        score = 0.0
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            # 词频
            tf = doc_term_freqs.get(token, 0)
            
            # BM25 公式
            idf = self.idf[token]
            norm = 1 - self.b + self.b * (doc_len / self.avg_doc_len) if self.avg_doc_len > 0 else 1
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        
        return score


class Reranker:
    """
    重排序器
    
    综合考虑：
    1. BM25 文本相似度
    2. Vector 语义相似度
    3. 时间衰减因子
    4. Tag 匹配度
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.bm25_scorer: Optional[BM25Scorer] = None
    
    def _init_bm25(self, documents: List[str]):
        """初始化 BM25 scorer"""
        self.bm25_scorer = BM25Scorer()
        self.bm25_scorer.fit(documents)
    
    def _compute_time_decay(self, timestamp: datetime, reference_time: Optional[datetime] = None) -> float:
        """
        计算时间衰减因子
        
        score = decay_factor ^ days_ago
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        days_diff = (reference_time - timestamp).days
        if days_diff < 0:
            days_diff = 0
        
        decay_score = self.config.time_decay_factor ** days_diff
        return decay_score
    
    def _compute_tag_score(self, result: SearchResult, query_tags: List[str]) -> float:
        """
        计算 tag 匹配分数
        
        Jaccard 相似度
        """
        if not query_tags or not result.tags:
            return 0.0
        
        result_tags_set = set(result.tags)
        query_tags_set = set(query_tags)
        
        intersection = len(result_tags_set & query_tags_set)
        union = len(result_tags_set | query_tags_set)
        
        return intersection / union if union > 0 else 0.0
    
    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        query_tags: Optional[List[str]] = None,
        reference_time: Optional[datetime] = None,
        has_stage2_filtering: bool = True  # 是否经过 Stage 2 过滤
    ) -> List[SearchResult]:
        """
        重排序候选结果
        
        Args:
            query: 查询文本
            candidates: 候选结果
            query_tags: 查询相关的 tags
            reference_time: 参考时间（用于计算时间衰减）
            has_stage2_filtering: 是否经过 Stage 2 标签过滤
        
        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return []
        
        if query_tags is None:
            query_tags = []
        
        # 初始化 BM25
        documents = [result.content for result in candidates]
        self._init_bm25(documents)
        
        # 计算各分数的归一化范围（用于更好的混合）
        bm25_scores = []
        vector_scores = []
        tag_scores = []
        
        for idx, result in enumerate(candidates):
            # BM25 分数
            bm25_raw = self.bm25_scorer.score(query, result.content, idx)
            bm25_scores.append(bm25_raw)
            
            # Vector 分数
            vector_scores.append(result.score)
            
            # Tag 匹配分数
            tag_scores.append(self._compute_tag_score(result, query_tags))
        
        # 归一化到 [0, 1]（使用 min-max normalization）
        def normalize(scores: List[float]) -> List[float]:
            if not scores or max(scores) == min(scores):
                return [0.5] * len(scores)
            min_s, max_s = min(scores), max(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        
        bm25_normalized = normalize(bm25_scores)
        vector_normalized = normalize(vector_scores)
        tag_normalized = tag_scores  # tag_score 已经是 Jaccard，范围 [0, 1]
        
        # 动态权重调整（如果没有 Stage 2 过滤，降低 tag 权重）
        if has_stage2_filtering:
            bm25_w = self.config.bm25_weight
            vector_w = self.config.vector_weight
            tag_w = self.config.tag_weight
        else:
            # 没有 Stage 2 过滤时，tag 不可信，降低权重
            bm25_w = 0.40
            vector_w = 0.55
            tag_w = 0.05
        
        # 计算最终分数
        for idx, result in enumerate(candidates):
            final_score = (
                bm25_w * bm25_normalized[idx] +
                vector_w * vector_normalized[idx] +
                tag_w * tag_normalized[idx]
            )
            
            # 更新分数
            result.score = final_score
            result.stage_scores[RetrievalStage.RERANK] = final_score
        
        # 排序
        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        # 返回 top-k
        return reranked[:self.config.rerank_top_k]
    

    def rerank_from_stage_results(
        self,
        query: str,
        stage2_results: StageResults,
        candidates_data: List[dict],  
        top_k: int = 10
    ) -> List[str]:
        """
        直接基于 Stage 2 结果重排序
        
        Args:
            query: 查询文本
            stage2_results: Stage 2 的结果对象
            candidates_data: 候选数据列表，每个元素包含:
                {
                    'memory_id': str,
                    'content': str,
                    'tags': List[str]
                }
            top_k: 返回数量
            
        Returns:
            排序后的 memory_ids（top-k）
        """
        if not candidates_data:
            return []
        
        # 从 metadata 中提取 query_tags
        query_tags = stage2_results.metadata.get('query_tags', [])
        
        # 初始化 BM25
        documents = [d['content'] for d in candidates_data]
        self._init_bm25(documents)
        
        # 计算分数
        bm25_scores = []
        tag_scores = []
        
        for idx, data in enumerate(candidates_data):
            # BM25
            bm25_raw = self.bm25_scorer.score(query, data['content'], idx)
            bm25_scores.append(bm25_raw)
            
            # Tag 匹配（使用 query_tags）
            tag_score = self._compute_tag_score_from_data(data, query_tags)
            tag_scores.append(tag_score)
        
        # 归一化
        def normalize(scores):
            if not scores or max(scores) == min(scores):
                return [0.5] * len(scores)
            min_s, max_s = min(scores), max(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        
        bm25_normalized = normalize(bm25_scores)
        tag_normalized = tag_scores
        
        # 动态权重
        has_stage2_filtering = bool(query_tags)
        if has_stage2_filtering:
            bm25_w = self.config.bm25_weight
            tag_w = self.config.tag_weight
        else:
            bm25_w = 0.60
            tag_w = 0.40
        
        # 计算最终分数
        final_scores = [
            bm25_w * bm25_normalized[i] + tag_w * tag_normalized[i]
            for i in range(len(candidates_data))
        ]
        
        # 排序并返回 top-k IDs
        memory_ids = [d['memory_id'] for d in candidates_data]
        scored_ids = list(zip(memory_ids, final_scores))
        scored_ids.sort(key=lambda x: x[1], reverse=True)
        
        return [memory_id for memory_id, _ in scored_ids[:top_k]]


    def _compute_tag_score_from_data(self, data: dict, query_tags: List[str]) -> float:
        """从 data dict 计算 tag 分数"""
        if not query_tags or not data.get('tags'):
            return 0.0
        
        result_tags_set = set(data['tags'])
        query_tags_set = set(query_tags)
        
        intersection = len(result_tags_set & query_tags_set)
        union = len(result_tags_set | query_tags_set)
        
        return intersection / union if union > 0 else 0.0