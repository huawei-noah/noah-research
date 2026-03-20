# storage/unified_index.py
"""
统一索引层 - 集成三层索引
"""

from typing import List, Optional, Set, Dict, Any
from datetime import datetime
import numpy as np

from .indexing.temporal_index import TemporalIndex, TimeInterval
from .indexing.tag_dag_index import TagDAGIndex
from .indexing.vector_index import LocalityAwareVectorIndex


class UnifiedIndex:
    """
    统一索引 - 三层过滤架构
    
    查询流程:
    Query -> L1 (Temporal) -> L2 (Tag DAG) -> L3 (Vector) -> Results
    
    性能保证:
    - L1 + L2: < 1ms (纯内存)
    - L3: < 10ms (只搜索候选集)
    - 总延迟: < 15ms (vs 纯向量搜索的 50-200ms)
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.temporal_index = TemporalIndex()
        self.tag_dag_index = TagDAGIndex()
        self.vector_index = LocalityAwareVectorIndex(embedding_dim)
        
        # 维护 episode_id -> (user_id, tags, timestamp, content) 映射
        self._episode_metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_episode(
        self,
        episode_id: str,
        user_id: str,
        timestamp: datetime,
        tags: List[str],
        embedding: np.ndarray,
        content: str,  
        tag_relations: Optional[List[tuple[str, str]]] = None
    ):
        """添加 episode 到所有索引"""
        # L1: Temporal
        self.temporal_index.add_episode(episode_id, user_id, timestamp)
        
        # L2: Tag DAG
        self.tag_dag_index.add_episode_tags(
            episode_id, 
            tags,
            tag_relations=tag_relations
        )
        
        # L3: Vector
        self.vector_index.add_embedding(episode_id, embedding, tags)
        
        # 保存元数据（包括 content）
        self._episode_metadata[episode_id] = {
            'user_id': user_id,
            'timestamp': timestamp,
            'tags': tags,
            'content': content 
        }
    
    def remove_episode(self, episode_id: str) -> bool:
        """
        从所有索引中删除 episode
        
        Args:
            episode_id: Episode ID
            
        Returns:
            是否删除成功
        """
        if episode_id not in self._episode_metadata:
            return False
        
        metadata = self._episode_metadata[episode_id]
        user_id = metadata['user_id']
        
        # L1: Temporal
        self.temporal_index.remove_episode(episode_id, user_id)
        
        # L2: Tag DAG
        self.tag_dag_index.remove_episode(episode_id)
        
        # L3: Vector
        self.vector_index.remove_embedding(episode_id)
        
        # 删除元数据
        del self._episode_metadata[episode_id]
        
        return True
    
    def search(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        query_tags: Optional[List[str]] = None,
        time_ranges: Optional[List[TimeInterval]] = None,
        top_k: int = 10,
        tag_expand_depth: int = 1
    ) -> List[tuple[str, float]]:
        """
        统一搜索接口
        
        Args:
            user_id: 用户ID
            query_embedding: 查询向量
            query_tags: 可选的查询标签
            time_ranges: 可选的时间范围
            top_k: 返回结果数量
            tag_expand_depth: 标签扩展深度
            
        Returns:
            [(episode_id, score), ...]
        """
        # L1: Temporal 过滤
        if time_ranges:
            temporal_candidates = set(
                self.temporal_index.query_multi_range(
                    user_id=user_id,
                    time_ranges=time_ranges
                )
            )
        else:
            # 如果没有时间限制，获取最近的一定数量（避免全量搜索）
            temporal_candidates = set(
                self.temporal_index.query_recent(
                    user_id=user_id,
                    limit=1000  # 限制最多搜索最近1000条
                )
            )
        
        # 如果 L1 没有候选，直接返回
        if not temporal_candidates:
            return []
        
        # L2: Tag DAG 过滤
        if query_tags:
            tag_candidates = self.tag_dag_index.query_by_tags(
                query_tags=query_tags,
                expand_depth=tag_expand_depth
            )
            
            # 交集：同时满足时间和标签条件
            candidates = temporal_candidates & tag_candidates
            
            # 如果交集为空，直接返回
            if not candidates:
                return []
            
            # 获取候选集的所有 tags（用于 L3）
            candidate_tags = set(query_tags)
            for tag in query_tags:
                candidate_tags.update(
                    self.tag_dag_index.get_related_tags(tag, max_distance=tag_expand_depth)
                )
        else:
            candidates = temporal_candidates
            candidate_tags = set()  # 空集表示搜索所有块
        
        # L3: Vector 搜索
        if candidate_tags:
            # 有标签：只搜索相关的 tag blocks
            results = self.vector_index.search(
                query_embedding=query_embedding,
                candidate_tags=candidate_tags,
                top_k=top_k * 3  # 多取一些，后面再过滤
            )
        else:
            # 无标签：需要在候选集中搜索
            results = self._search_in_candidates(
                query_embedding=query_embedding,
                candidates=candidates,
                top_k=top_k * 3
            )
        
        # 过滤：只返回在候选集中的结果
        filtered_results = [
            (eid, score) for eid, score in results
            if eid in candidates
        ]
        
        return filtered_results[:top_k]
    
    def _search_in_candidates(
        self,
        query_embedding: np.ndarray,
        candidates: Set[str],
        top_k: int
    ) -> List[tuple[str, float]]:
        """
        在候选集内进行向量搜索
        
        策略：
        1. 从候选集中提取所有相关的 tags
        2. 在这些 tags 对应的 blocks 中搜索
        3. 过滤出候选集中的结果
        
        Args:
            query_embedding: 查询向量
            candidates: 候选 episode IDs
            top_k: 返回数量
            
        Returns:
            [(episode_id, score), ...]
        """
        # 1. 收集候选集中所有的 tags
        candidate_tags = set()
        for episode_id in candidates:
            if episode_id in self._episode_metadata:
                candidate_tags.update(self._episode_metadata[episode_id]['tags'])
        
        # 2. 如果有 tags，搜索对应的 blocks
        if candidate_tags:
            results = self.vector_index.search(
                query_embedding=query_embedding,
                candidate_tags=candidate_tags,
                top_k=top_k * 2  # 多取一些
            )
        else:
            # 3. 没有 tags（不太可能），返回空
            return []
        
        # 4. 过滤并返回
        filtered = [
            (eid, score) for eid, score in results
            if eid in candidates
        ]
        return filtered[:top_k]
    
    def clear_user_index(self, user_id: str) -> int:
        """
        清空用户的所有索引数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            删除的 episode 数量
        """
        # 找到该用户的所有 episodes
        user_episodes = [
            eid for eid, meta in self._episode_metadata.items()
            if meta['user_id'] == user_id
        ]
        
        # 批量删除
        for episode_id in user_episodes:
            self.remove_episode(episode_id)
        
        return len(user_episodes)
    
    
    def get_memory_content(self, memory_id: str) -> Optional[str]:
        """
        O(1) 获取 memory content
        
        Args:
            memory_id: Memory ID
            
        Returns:
            content 或 None
        """
        data = self._episode_metadata.get(memory_id)
        return data['content'] if data else None
    
    def batch_get_memory_data(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
        """
        批量获取 memory 数据（O(n) 但常数极小）
        
        性能目标: < 1ms for 100 items
        
        Args:
            memory_ids: Memory IDs
            
        Returns:
            [{memory_id, content, tags}, ...]
        """
        result = []
        for memory_id in memory_ids:
            data = self._episode_metadata.get(memory_id)
            if data:
                result.append({
                    'memory_id': memory_id,
                    'content': data['content'],
                    'tags': data['tags']
                })
        return result
    
    # ========== 统计信息 ==========
    
    def get_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Args:
            user_id: 可选的用户ID，指定则返回该用户的统计
            
        Returns:
            统计信息字典
        """
        if user_id:
            # 用户级统计
            user_episodes = [
                eid for eid, meta in self._episode_metadata.items()
                if meta['user_id'] == user_id
            ]
            
            return {
                "user_id": user_id,
                "total_episodes": len(user_episodes),
                "L1_temporal": self.temporal_index.get_stats(),
                "L2_tag_dag": self.tag_dag_index.get_stats(),
                "L3_vector": self.vector_index.get_stats()
            }
        else:
            # 全局统计
            return {
                "total_episodes": len(self._episode_metadata),
                "L1_temporal": self.temporal_index.get_stats(),
                "L2_tag_dag": self.tag_dag_index.get_stats(),
                "L3_vector": self.vector_index.get_stats()
            }
    
    def consolidate_offline(self):
        """
        离线 consolidation
        在系统空闲时执行，优化物理布局
        """
        self.vector_index.consolidate()
    
    def get_stats(self) -> dict:
        """向后兼容的统计方法"""
        return self.get_statistics()
    
    # ========== 序列化支持 ==========
    
    def save(self, index_dir: str) -> None:
        """
        保存索引到目录
        
        文件结构:
        index_dir/
          ├── temporal_index.json
          ├── tag_dag_index.json
          ├── vector_index_metadata.json
          ├── vector_index_vectors.npz
          └── episode_metadata.json
        
        Args:
            index_dir: 索引目录路径
        """
        from pathlib import Path
        import json
        
        dir_path = Path(index_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存各个子索引
        self.temporal_index.save(str(dir_path / "temporal_index.json"))
        self.tag_dag_index.save(str(dir_path / "tag_dag_index.json"))
        self.vector_index.save(str(dir_path / "vector_index"))
        
        # 2. 保存 episode metadata（包括 content）
        metadata_path = dir_path / "episode_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            # 将 datetime 转换为 isoformat
            serializable_metadata = {}
            for ep_id, meta in self._episode_metadata.items():
                serializable_metadata[ep_id] = {
                    'user_id': meta['user_id'],
                    'timestamp': meta['timestamp'].isoformat(),
                    'tags': meta['tags'],
                    'content': meta['content'] 
                }
            
            json.dump(serializable_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Index saved to {index_dir}")
    
    def load(self, index_dir: str) -> None:
        """
        从目录加载索引
        
        Args:
            index_dir: 索引目录路径
        """
        from pathlib import Path
        import json
        
        dir_path = Path(index_dir)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")
        
        # 1. 加载各个子索引
        self.temporal_index.load(str(dir_path / "temporal_index.json"))
        self.tag_dag_index.load(str(dir_path / "tag_dag_index.json"))
        self.vector_index.load(str(dir_path / "vector_index"))
        
        # 2. 加载 episode metadata（包括 content）
        metadata_path = dir_path / "episode_metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            serialized_metadata = json.load(f)
            
            # 将 isoformat 转换回 datetime
            self._episode_metadata = {}
            for ep_id, meta in serialized_metadata.items():
                self._episode_metadata[ep_id] = {
                    'user_id': meta['user_id'],
                    'timestamp': datetime.fromisoformat(meta['timestamp']),
                    'tags': meta['tags'],
                    'content': meta.get('content', '') 
                }
        
        print(f"✅ Index loaded from {index_dir}")