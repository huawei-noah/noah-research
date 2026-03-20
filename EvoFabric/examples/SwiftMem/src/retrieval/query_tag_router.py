"""
Query Tag Router - 将 query 映射到相关 tags
"""

from typing import List, Optional, Dict
import numpy as np
import logging

from ..storage.indexing.tag_dag_index import TagDAGIndex
from ..embeddings.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class QueryTagRouter:
    """
    Query Tag Router
    
    负责将用户查询映射到相关的 tags，支持三种方法：
    1. Embedding 相似度（默认）
    2. LLM 生成（TODO）
    3. 混合方法（TODO）
    """
    
    def __init__(
        self,
        tag_dag_index: TagDAGIndex,
        embedding_manager: EmbeddingManager
    ):
        """
        初始化 Router
        
        Args:
            tag_dag_index: Tag DAG 索引
            embedding_manager: Embedding 管理器
        """
        self.tag_dag = tag_dag_index
        self.embedding_manager = embedding_manager
        
        # 初始化 cache 为 None（确保属性存在）
        self._tag_embeddings_cache: Optional[Dict[str, any]] = None
    
    def infer_tags(
        self,
        query: str,
        method: str = "embedding",
        top_k: int = 5,
        query_embedding: Optional[np.ndarray] = None 
    ) -> List[str]:
        """
        统一入口：推断 query 相关的 tags
        
        Args:
            query: 查询文本
            method: 方法选择 ("embedding", "llm", "hybrid")
            top_k: 返回 tag 数量
            query_embedding: 可选的预计算 query embedding
            
        Returns:
            相关 tag 列表
        """
        if method == "embedding":
            return self.infer_tags_by_embedding(query, top_k, query_embedding)
        elif method == "llm":
            return self.infer_tags_by_llm(query, top_k)
        elif method == "hybrid":
            return self.infer_tags_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
    

    def infer_tags_by_embedding(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        方法1: 用 embedding 相似度找最相关 tags
        
        优化版本：使用预计算的归一化 tag embeddings cache
        """
        import time
        
        total_start = time.time()
        
        # 检查 cache
        if self._tag_embeddings_cache is None:
            print("❌ [ERROR] Tag embeddings cache is NONE! Rebuilding...")
            rebuild_start = time.time()
            self._rebuild_tag_embeddings_cache()
            rebuild_time = (time.time() - rebuild_start) * 1000
            print(f"⚠️  Emergency rebuild took {rebuild_time:.2f}ms")
        
        # 使用缓存的归一化矩阵
        tags_normalized = self._tag_embeddings_cache['matrix_normalized']
        valid_tags = self._tag_embeddings_cache['tags']
        
        if len(valid_tags) == 0:
            return []
        

        if query_embedding is None:
            query_embedding = self.embedding_manager.get_embedding(query)

        

        query_normalized = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)


        similarities = np.dot(tags_normalized, query_normalized)


        top_indices = np.argsort(similarities)[::-1][:top_k]
        result_tags = [valid_tags[idx] for idx in top_indices]


        
        logger.debug(
            f"Query tag inference (embedding): {query[:50]}... -> {result_tags}"
        )
        
        return result_tags


    def _rebuild_tag_embeddings_cache(self) -> None:
        """
        重建 tag embeddings 缓存（一次性构建 numpy matrix）
        
        这个方法只在第一次调用或 tags 变化时执行
        """
        all_tags = list(self.tag_dag.tag_nodes.keys())
        
        if not all_tags:
            logger.warning("No tags found in tag_dag, cache will be empty")
            self._tag_embeddings_cache = {
                'matrix': np.array([], dtype=np.float32).reshape(0, self.embedding_manager.dimension),
                'matrix_normalized': np.array([], dtype=np.float32).reshape(0, self.embedding_manager.dimension),
                'tags': []
            }
            return
        
        # 收集所有 tag embeddings（向量化）
        embeddings_list = []
        valid_tags = []
        
        for tag in all_tags:
            tag_node = self.tag_dag.tag_nodes[tag]
            
            if tag_node.embedding is not None:
                embeddings_list.append(tag_node.embedding)
                valid_tags.append(tag)
            else:
                # 如果预生成失败，立即报错（不要静默降级）
                raise RuntimeError(
                    f"Tag '{tag}' has no embedding! This should have been "
                    f"pre-generated in IndexPool._pregenerate_tag_embeddings()"
                )
        
        # 一次性转换为 numpy matrix
        if embeddings_list:
            tag_embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        else:
            tag_embeddings_matrix = np.array([], dtype=np.float32).reshape(0, self.embedding_manager.dimension)
        
        # 预计算归一化矩阵
        if tag_embeddings_matrix.shape[0] > 0:
            norms = np.linalg.norm(tag_embeddings_matrix, axis=1, keepdims=True) + 1e-8
            tag_embeddings_normalized = tag_embeddings_matrix / norms
        else:
            tag_embeddings_normalized = tag_embeddings_matrix
        
        # 缓存（包含原始和归一化两个版本）
        self._tag_embeddings_cache = {
            'matrix': tag_embeddings_matrix,
            'matrix_normalized': tag_embeddings_normalized,  # ← 新增
            'tags': valid_tags
        }
        
        logger.info(
            f"✅ Built tag embeddings cache: {len(valid_tags)} tags, "
            f"matrix shape {tag_embeddings_matrix.shape}"
        )

    def infer_tags_by_llm(
        self,
        query: str,
        llm_client = None,
        top_k: int = 5
    ) -> List[str]:
        """
        方法2: 用 LLM 直接生成 tags（预留接口）
        
        Args:
            query: 查询文本
            llm_client: LLM 客户端
            top_k: 返回数量
            
        Returns:
            相关 tag 列表
        """
        # TODO: 实现 LLM 方法
        logger.warning("LLM method not implemented yet, falling back to embedding")
        return self.infer_tags_by_embedding(query, top_k)
    
    def infer_tags_hybrid(
        self,
        query: str,
        top_k: int = 5
    ) -> List[str]:
        """
        方法3: 混合方法（预留接口）
        
        策略：
        1. 先用规则过滤（如关键词匹配）
        2. 再用 embedding 排序
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            相关 tag 列表
        """
        # TODO: 实现混合方法
        logger.warning("Hybrid method not implemented yet, falling back to embedding")
        return self.infer_tags_by_embedding(query, top_k)