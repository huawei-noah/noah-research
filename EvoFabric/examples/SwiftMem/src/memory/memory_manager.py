# src/memory/memory_manager.py
"""
Memory Manager - 统一的记忆管理接口
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np

from ..models.episode import Episode
from ..models.message import Message
from ..models.semantic import SemanticMemory
from ..storage.unified_index import UnifiedIndex
from ..storage.backends.episode_storage import EpisodeStorage
from ..storage.backends.semantic_storage import SemanticStorage
from ..embeddings.embedding_manager import EmbeddingManager


class MemoryManager:
    """统一的记忆管理器"""
    
    def __init__(
        self,
        # Storage paths
        index_dir: str = "./data/index",
        episode_db_path: str = "./data/episodes.db",
        semantic_db_path: str = "./data/semantic.db",
        
        # Embedding configuration
        openai_api_key: str = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_cache_path: Optional[str] = "./data/embedding_cache.json",
        embedding_dim: int = 1536,  # text-embedding-3-small 的维度
        
        # Optional: custom base URL for OpenAI API
        openai_base_url: Optional[str] = None
    ):
        """
        初始化 Memory Manager
        
        Args:
            index_dir: 索引目录（暂时未使用，为未来持久化预留）
            episode_db_path: Episode 数据库路径
            semantic_db_path: SemanticMemory 数据库路径
            openai_api_key: OpenAI API Key
            embedding_model: Embedding 模型名称
            embedding_cache_path: Embedding 缓存路径
            embedding_dim: 嵌入向量维度
            openai_base_url: 自定义 OpenAI API base URL
        """
        # 初始化 Embedding Manager
        if openai_api_key:
            cache_path = Path(embedding_cache_path) if embedding_cache_path else None
            self.embedding_manager = EmbeddingManager(
                api_key=openai_api_key,
                model=embedding_model,
                base_url=openai_base_url,
                cache_enabled=False,
                cache_path=cache_path
            )
            # 使用实际的 embedding 维度
            embedding_dim = self.embedding_manager.dimension
        else:
            self.embedding_manager = None
        
        # 初始化索引
        self.index = UnifiedIndex(embedding_dim=embedding_dim)
        
        # 初始化存储后端
        self.episode_storage = EpisodeStorage(episode_db_path)
        self.semantic_storage = SemanticStorage(semantic_db_path)
    
    # ========== Episode 管理 ==========
    
    def add_episode(self, episode: Episode) -> None:
        """
        添加 Episode（同时更新索引和存储）
        
        Args:
            episode: Episode 对象
        """
        # 1. 保存到存储
        self.episode_storage.save_episode(episode)
        
        # 2. 生成 embedding 并添加到索引
        if self.embedding_manager:
            # 构建用于 embedding 的文本
            text = self._episode_to_text(episode)
            
            # 生成 embedding
            embedding = self.embedding_manager.get_embedding(text)
            
            # 添加到索引（传递 content）
            self.index.add_episode(
                episode_id=episode.episode_id,
                user_id=episode.user_id,
                timestamp=episode.timestamp,
                tags=episode.tags or [],
                embedding=embedding,
                content=episode.content or text, 
                tag_relations=episode.tag_relations or []
            )
    
    def _episode_to_text(self, episode: Episode) -> str:
        """
        将 Episode 转换为用于 embedding 的文本
        
        Args:
            episode: Episode 对象
            
        Returns:
            文本表示
        """
        # 组合所有消息
        messages_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in episode.messages
        ])
        
        # 可选：添加 tags
        tags_text = f"Tags: {', '.join(episode.tags)}" if episode.tags else ""
        
        # 组合
        return f"{messages_text}\n{tags_text}".strip()
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """获取单个 Episode"""
        return self.episode_storage.load_episode(episode_id)
    
    def delete_episode(self, user_id: str, episode_id: str) -> bool:
        """
        删除 Episode（同时删除索引和存储）
        
        Args:
            user_id: 用户ID
            episode_id: Episode ID
        
        Returns:
            是否删除成功
        """
        # 1. 从索引删除
        self.index.remove_episode(episode_id)
        
        # 2. 从存储删除
        return self.episode_storage.delete_episode(episode_id)
    
    def list_episodes(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Episode]:
        """列出用户的 Episodes"""
        episode_ids = self.episode_storage.list_episodes(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        return self.episode_storage.batch_load_episodes(episode_ids)
    
    def count_episodes(self, user_id: str) -> int:
        """统计用户的 Episode 数量"""
        return self.episode_storage.count_episodes(user_id)
    
    # ========== SemanticMemory 管理 ==========
    
    def add_semantic_memory(self, memory: SemanticMemory) -> None:
        """
        添加 SemanticMemory（同时更新索引和存储）
        
        Args:
            memory: SemanticMemory 对象
        """
        # 1. 保存到存储
        self.semantic_storage.save_memory(memory)
        
        # 2. 生成 embedding 并添加到索引
        if self.embedding_manager:
            # 使用 content 生成 embedding
            embedding = self.embedding_manager.get_embedding(memory.content)
            
            # 添加到索引（传递 content）
            self.index.add_episode(
                episode_id=memory.memory_id,
                user_id=memory.user_id,
                timestamp=memory.timestamp,
                tags=memory.tags or [],
                embedding=embedding,
                content=memory.content,  
                tag_relations=None
            )
    
    def get_semantic_memory(self, memory_id: str) -> Optional[SemanticMemory]:
        """获取单个 SemanticMemory"""
        return self.semantic_storage.load_memory(memory_id)
    
    def delete_semantic_memory(self, user_id: str, memory_id: str) -> bool:
        """
        删除 SemanticMemory（同时删除索引和存储）
        
        Args:
            user_id: 用户ID
            memory_id: Memory ID
        
        Returns:
            是否删除成功
        """
        # 1. 从索引删除
        self.index.remove_episode(memory_id)
        
        # 2. 从存储删除
        return self.semantic_storage.delete_memory(memory_id)
    
    def list_semantic_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SemanticMemory]:
        """列出用户的 SemanticMemories"""
        memory_ids = self.semantic_storage.list_memories(
            user_id=user_id,
            memory_type=memory_type,
            limit=limit,
            offset=offset
        )
        return self.semantic_storage.batch_load_memories(memory_ids)
    
    def count_semantic_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None
    ) -> int:
        """统计用户的 SemanticMemory 数量"""
        return self.semantic_storage.count_memories(user_id, memory_type)
    
    def update_memory_importance(self, memory_id: str, new_importance: float) -> bool:
        """更新 Memory 的重要性"""
        return self.semantic_storage.update_importance(memory_id, new_importance)
    
    # ========== 混合检索 ==========
    
    def search_episodes(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Episode, float]]:
        """
        检索相关的 Episodes（语义搜索 + 过滤）
        
        Args:
            user_id: 用户ID
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件（时间范围、标签等）
            
        Returns:
            (Episode, 相似度分数) 列表
        """
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager not initialized. Please provide openai_api_key.")
        
        # 1. 生成查询 embedding
        query_embedding = self.embedding_manager.get_embedding(query)
        
        # 2. 使用索引检索
        query_tags = filters.get('tags') if filters else None
        time_ranges = filters.get('time_ranges') if filters else None
        
        results = self.index.search(
            user_id=user_id,
            query_embedding=query_embedding,
            query_tags=query_tags,
            time_ranges=time_ranges,
            top_k=top_k
        )
        
        # 3. 批量加载完整的 Episode 对象
        episode_ids = [r[0] for r in results]
        episodes = self.episode_storage.batch_load_episodes(episode_ids)
        
        # 4. 构建结果（保持顺序和分数）
        episode_dict = {ep.episode_id: ep for ep in episodes}
        return [
            (episode_dict[eid], score)
            for eid, score in results
            if eid in episode_dict
        ]
    
    def search_semantic_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[SemanticMemory, float]]:
        """
        检索相关的 SemanticMemories
        
        Args:
            user_id: 用户ID
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件
            
        Returns:
            (SemanticMemory, 相似度分数) 列表
        """
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager not initialized. Please provide openai_api_key.")
        
        # 1. 生成查询 embedding
        query_embedding = self.embedding_manager.get_embedding(query)
        
        # 2. 使用索引检索
        query_tags = filters.get('tags') if filters else None
        time_ranges = filters.get('time_ranges') if filters else None
        
        results = self.index.search(
            user_id=user_id,
            query_embedding=query_embedding,
            query_tags=query_tags,
            time_ranges=time_ranges,
            top_k=top_k
        )
        
        # 3. 批量加载完整对象
        memory_ids = [r[0] for r in results]
        memories = self.semantic_storage.batch_load_memories(memory_ids)
        
        # 4. 构建结果
        memory_dict = {mem.memory_id: mem for mem in memories}
        return [
            (memory_dict[mid], score)
            for mid, score in results
            if mid in memory_dict
        ]
    
    def hybrid_search(
        self,
        user_id: str,
        query: str,
        top_k_episodes: int = 5,
        top_k_memories: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """混合检索（同时搜索 Episodes 和 SemanticMemories）"""
        episodes = self.search_episodes(
            user_id=user_id,
            query=query,
            top_k=top_k_episodes,
            filters=filters
        )
        
        memories = self.search_semantic_memories(
            user_id=user_id,
            query=query,
            top_k=top_k_memories,
            filters=filters
        )
        
        return {
            'episodes': episodes,
            'memories': memories
        }
    
    # ========== 时间范围查询 ==========
    
    def get_episodes_in_time_range(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[Episode]:
        """获取指定时间范围内的 Episodes"""
        episode_ids = self.episode_storage.get_episodes_in_time_range(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        return self.episode_storage.batch_load_episodes(episode_ids)
    
    def get_memories_in_time_range(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[SemanticMemory]:
        """获取指定时间范围内的 Memories"""
        memory_ids = self.semantic_storage.get_memories_in_time_range(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        return self.semantic_storage.batch_load_memories(memory_ids)
    
    # ========== 标签查询 ==========
    
    def search_episodes_by_tags(
        self,
        user_id: str,
        tags: List[str],
        limit: int = 100
    ) -> List[Episode]:
        """根据标签搜索 Episodes"""
        episode_ids = self.episode_storage.search_by_tags(
            user_id=user_id,
            tags=tags,
            limit=limit
        )
        return self.episode_storage.batch_load_episodes(episode_ids)
    
    def search_memories_by_tags(
        self,
        user_id: str,
        tags: List[str],
        limit: int = 100
    ) -> List[SemanticMemory]:
        """根据标签搜索 Memories"""
        memory_ids = self.semantic_storage.search_by_tags(
            user_id=user_id,
            tags=tags,
            limit=limit
        )
        return self.semantic_storage.batch_load_memories(memory_ids)
    
    # ========== 统计信息 ==========
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户的记忆统计信息
        
        Args:
            user_id: 用户ID
        
        Returns:
            统计信息字典
        """
        # Episode 统计
        episode_count = self.episode_storage.count_episodes(user_id)
        
        # Memory 统计
        memory_stats = self.semantic_storage.get_memory_statistics(user_id)
        
        # 索引统计
        index_stats = self.index.get_statistics(user_id=user_id)
        
        # Embedding 统计
        embedding_stats = self.embedding_manager.get_stats() if self.embedding_manager else {}
        
        return {
            "episode_count": episode_count,
            "memory_stats": memory_stats,
            "index_stats": index_stats,
            "embedding_stats": embedding_stats,
            "user_id": user_id
        }
    
    # ========== 数据清理 ==========
    
    def clear_user_data(self, user_id: str) -> Dict[str, int]:
        """
        清空用户的所有数据
        
        Args:
            user_id: 用户ID
        
        Returns:
            删除统计字典
        """
        # 清空索引
        index_deleted = self.index.clear_user_index(user_id)
        
        # 清空 Episodes
        episode_ids = self.episode_storage.list_episodes(user_id)
        episode_count = len(episode_ids)
        for episode_id in episode_ids:
            self.episode_storage.delete_episode(episode_id)
        
        # 清空 Memories
        memory_ids = self.semantic_storage.list_memories(user_id)
        memory_count = len(memory_ids)
        for memory_id in memory_ids:
            self.semantic_storage.delete_memory(memory_id)
        
        return {
            "episodes_deleted": episode_count,
            "memories_deleted": memory_count,
            "index_entries_deleted": index_deleted
        }
    
    # ========== 批量操作 ==========
    
    def batch_add_episodes(self, episodes: List[Episode]) -> None:
        """批量添加 Episodes"""
        for episode in episodes:
            self.add_episode(episode)
    
    def batch_add_semantic_memories(self, memories: List[SemanticMemory]) -> None:
        """批量添加 SemanticMemories"""
        for memory in memories:
            self.add_semantic_memory(memory)
    
    # ========== 关联查询 ==========
    
    def get_memories_by_episode(
        self,
        user_id: str,
        episode_id: str,
        limit: int = 100
    ) -> List[SemanticMemory]:
        """查找关联到特定 Episode 的 Memories"""
        memory_ids = self.semantic_storage.get_memories_by_source_episode(
            user_id=user_id,
            episode_id=episode_id,
            limit=limit
        )
        return self.semantic_storage.batch_load_memories(memory_ids)
    
    # ========== 资源管理 ==========
    
    def close(self) -> None:
        """关闭所有连接"""
        self.episode_storage.close()
        self.semantic_storage.close()
        
        # 保存 embedding cache
        if self.embedding_manager:
            self.embedding_manager._save_cache()
    
    def __enter__(self):
        """上下文管理器支持"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()

    # ========== 索引持久化 ==========
    
    def save_index(self, index_dir: str) -> None:
        """
        保存索引到目录
        
        Args:
            index_dir: 索引保存目录
        """
        self.index.save(index_dir)
        print(f"✅ Index saved to {index_dir}")
    
    def load_index(self, index_dir: str) -> None:
        """
        从目录加载索引
        
        Args:
            index_dir: 索引目录
        """
        self.index.load(index_dir)
        print(f"✅ Index loaded from {index_dir}")
    

    # ========== Consolidation ==========

    def trigger_consolidation(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        触发离线 embedding consolidation
        
        Args:
            force: 是否强制执行（忽略收益评估）
        
        Returns:
            consolidation metrics，如果未执行则返回 None
        """
        import logging
        logger = logging.getLogger(__name__)
        
        from ..consolidation.embedding_consolidator import EmbeddingConsolidator
        
        consolidator = EmbeddingConsolidator(
            vector_index=self.index.vector_index,
            tag_dag_index=self.index.tag_dag_index
        )
        
        # 分析是否需要 consolidation
        plan = consolidator.analyze()
        
        # 判断条件：预计收益 > 10% 或 强制执行
        if plan.estimated_improvement > 0.1 or force:
            logger.info(
                f"Starting consolidation - estimated improvement: "
                f"{plan.estimated_improvement:.2%}"
            )
            try:
                metrics = consolidator.consolidate(plan)
                logger.info(f"Consolidation completed - {metrics}")
                
                # 返回简化的 metrics dict
                return {
                    'clusters_created': len(plan.clusters),
                    'estimated_improvement': plan.estimated_improvement,
                    'consolidation_time_ms': metrics.consolidation_time_ms,
                    'estimated_search_speedup': metrics.estimated_search_speedup,
                    'before_num_blocks': metrics.before_num_blocks,
                    'after_num_blocks': metrics.after_num_blocks,
                }
            except Exception as e:
                logger.error(f"Consolidation failed: {e}")
                return None
        else:
            logger.info(
                f"Skipping consolidation - improvement too small: "
                f"{plan.estimated_improvement:.2%}"
            )
            return None

__all__ = ['MemoryManager']