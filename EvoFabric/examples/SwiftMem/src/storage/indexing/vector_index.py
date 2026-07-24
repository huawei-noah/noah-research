# storage/indexing/vector_index.py

import numpy as np
from typing import List, Set, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class VectorBlock:
    """
    Block of vectors with locality
    
    Stores vectors for a specific tag/context together
    to improve cache efficiency.
    """
    embeddings: np.ndarray  # Shape: (N, dim)
    episode_ids: List[str]
    
    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Search within this block
        
        Args:
            query: Query embedding (normalized)
            top_k: Number of results to return
        
        Returns:
            List of (episode_id, similarity_score) tuples
        """
        if len(self.episode_ids) == 0:
            return []
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query)
        
        # FIX: Handle case where top_k > number of vectors
        actual_k = min(top_k, len(similarities))
        
        # Get top-k using partial sort
        if actual_k == len(similarities):
            # If we want all results, just sort
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Use argpartition for efficiency
            top_indices = np.argpartition(similarities, -actual_k)[-actual_k:]
            # Sort the top-k
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        return [
            (self.episode_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]


class LocalityAwareVectorIndex:
    """
    L3: Locality-Aware Vector Index
    
    Stores embeddings grouped by tags for better cache locality.
    Uses incremental consolidation to balance write and search performance.
    """
    
    def __init__(self, embedding_dim: int, consolidation_threshold: int = 100):
        self.embedding_dim = embedding_dim
        self.consolidation_threshold = consolidation_threshold
        
        # Tag-specific blocks: tag -> VectorBlock
        self.tag_blocks: Dict[str, VectorBlock] = {}
        
        # Temporary block for recent additions (not yet consolidated)
        self.temp_embeddings: List[np.ndarray] = []
        self.temp_episode_ids: List[str] = []
        self.temp_tags: List[Set[str]] = []
        
        # Episode location tracking: episode_id -> (tags, is_in_temp)
        self.episode_location: Dict[str, Tuple[Set[str], bool]] = {}
    
    def add_embedding(self, episode_id: str, embedding: np.ndarray, tags: List[str]):
        """
        Add embedding to index
        
        Initially goes to temp block. Consolidates when threshold reached.
        """
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Add to temp block
        self.temp_embeddings.append(embedding)
        self.temp_episode_ids.append(episode_id)
        self.temp_tags.append(set(tags))
        
        # Track location
        self.episode_location[episode_id] = (set(tags), True)
        
        # Trigger consolidation if threshold reached
        if len(self.temp_embeddings) >= self.consolidation_threshold:
            self.consolidate()
    
    def consolidate(self):
        """
        Consolidate temporary embeddings into tag blocks
        
        Groups by tags and merges with existing blocks.
        """
        if not self.temp_embeddings:
            return
        
        # Group temp embeddings by tags
        tag_groups: Dict[str, List[Tuple[np.ndarray, str]]] = {}
        
        for emb, ep_id, tags in zip(
            self.temp_embeddings,
            self.temp_episode_ids,
            self.temp_tags
        ):
            for tag in tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append((emb, ep_id))
        
        # Merge into existing blocks or create new ones
        for tag, items in tag_groups.items():
            embeddings = np.array([item[0] for item in items])
            episode_ids = [item[1] for item in items]
            
            if tag in self.tag_blocks:
                # Merge with existing block
                old_block = self.tag_blocks[tag]
                new_embeddings = np.vstack([old_block.embeddings, embeddings])
                new_episode_ids = old_block.episode_ids + episode_ids
                
                self.tag_blocks[tag] = VectorBlock(
                    embeddings=new_embeddings,
                    episode_ids=new_episode_ids
                )
            else:
                # Create new block
                self.tag_blocks[tag] = VectorBlock(
                    embeddings=embeddings,
                    episode_ids=episode_ids
                )
        
        # Update episode locations
        for ep_id, tags in zip(self.temp_episode_ids, self.temp_tags):
            self.episode_location[ep_id] = (tags, False)
        
        # Clear temp block
        self.temp_embeddings = []
        self.temp_episode_ids = []
        self.temp_tags = []
    
    def search(
        self,
        query_embedding: np.ndarray,
        candidate_tags: Set[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query vector (will be normalized)
            candidate_tags: Tags to search within (from L2)
            top_k: Number of results to return
        
        Returns:
            List of (episode_id, similarity_score) sorted by score
        """
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        all_results = []
        
        # Search in relevant tag blocks
        for tag in candidate_tags:
            if tag in self.tag_blocks:
                block_results = self.tag_blocks[tag].search(
                    query_embedding,
                    top_k=top_k
                )
                all_results.extend(block_results)
        
        # Search in temp block
        if self.temp_embeddings:
            # Filter temp block by tags
            temp_indices = [
                i for i, tags in enumerate(self.temp_tags)
                if tags & candidate_tags  # intersection
            ]
            
            if temp_indices:
                temp_emb = np.array([self.temp_embeddings[i] for i in temp_indices])
                temp_ids = [self.temp_episode_ids[i] for i in temp_indices]
                
                similarities = np.dot(temp_emb, query_embedding)
                
                # FIX: Handle case where top_k > number of vectors
                actual_k = min(top_k, len(similarities))
                
                if actual_k == len(similarities):
                    top_temp_indices = np.argsort(similarities)[::-1]
                else:
                    top_temp_indices = np.argpartition(similarities, -actual_k)[-actual_k:]
                    top_temp_indices = top_temp_indices[np.argsort(similarities[top_temp_indices])[::-1]]
                
                for idx in top_temp_indices:
                    all_results.append((temp_ids[idx], float(similarities[idx])))
        
        # Merge and sort all results
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k unique results
        seen = set()
        unique_results = []
        for ep_id, score in all_results:
            if ep_id not in seen:
                seen.add(ep_id)
                unique_results.append((ep_id, score))
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
    
    def remove_embedding(self, episode_id: str):
        """Remove embedding from index"""
        if episode_id not in self.episode_location:
            return
        
        tags, is_in_temp = self.episode_location[episode_id]
        
        if is_in_temp:
            # Remove from temp block
            idx = self.temp_episode_ids.index(episode_id)
            self.temp_embeddings.pop(idx)
            self.temp_episode_ids.pop(idx)
            self.temp_tags.pop(idx)
        else:
            # Remove from tag blocks
            for tag in tags:
                if tag in self.tag_blocks:
                    block = self.tag_blocks[tag]
                    if episode_id in block.episode_ids:
                        idx = block.episode_ids.index(episode_id)
                        
                        # Remove from block
                        new_embeddings = np.delete(block.embeddings, idx, axis=0)
                        new_episode_ids = block.episode_ids[:idx] + block.episode_ids[idx+1:]
                        
                        if len(new_episode_ids) > 0:
                            self.tag_blocks[tag] = VectorBlock(
                                embeddings=new_embeddings,
                                episode_ids=new_episode_ids
                            )
                        else:
                            # Remove empty block
                            del self.tag_blocks[tag]
        
        # Remove from location tracking
        del self.episode_location[episode_id]
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        total_embeddings = sum(
            len(block.episode_ids) for block in self.tag_blocks.values()
        ) + len(self.temp_episode_ids)
        
        avg_block_size = 0
        if self.tag_blocks:
            avg_block_size = sum(
                len(block.episode_ids) for block in self.tag_blocks.values()
            ) / len(self.tag_blocks)
        
        return {
            'total_embeddings': total_embeddings,
            'total_blocks': len(self.tag_blocks),
            'temp_block_size': len(self.temp_episode_ids),
            'avg_block_size': avg_block_size,
        }
    
    def export_tag_blocks(self) -> Dict[str, 'VectorBlock']:
        """
        导出所有 tag blocks（用于 consolidation）
        
        Returns:
            tag -> VectorBlock 的映射
        """
        # return self.tag_blocks.copy()
        if self.temp_embeddings:
            self.consolidate()
    
        return self.tag_blocks.copy()
    
    def import_tag_blocks(
        self,
        new_tag_blocks: Dict[str, 'VectorBlock'],
        clear_temp: bool = True
    ) -> None:
        """
        导入新的 tag blocks（consolidation 后）
        
        Args:
            new_tag_blocks: 新的 tag blocks
            clear_temp: 是否清空临时块
        """
        # 更新 tag blocks
        self.tag_blocks = new_tag_blocks
        
        # 更新 episode_location（标记为非临时）
        for tag, block in new_tag_blocks.items():
            for episode_id in block.episode_ids:
                if episode_id in self.episode_location:
                    tags, _ = self.episode_location[episode_id]
                    self.episode_location[episode_id] = (tags, False)
        
        # 可选：清空临时块
        if clear_temp:
            self.temp_embeddings = []
            self.temp_episode_ids = []
            self.temp_tags = []
    
    def get_block_memory_layout(self) -> Dict[str, Dict[str, any]]:
        """
        获取 block 的内存布局信息
        
        Returns:
            每个 tag block 的内存信息
        """
        layout = {}
        
        for tag, block in self.tag_blocks.items():
            layout[tag] = {
                'num_embeddings': len(block.episode_ids),
                'memory_size_bytes': block.embeddings.nbytes,
                'is_contiguous': block.embeddings.flags['C_CONTIGUOUS'],
                'shape': block.embeddings.shape,
                'dtype': str(block.embeddings.dtype)
            }
        
        return layout
    

    # ========== 序列化支持 ==========
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化元数据为字典（不包含向量数据）"""
        return {
            "embedding_dim": self.embedding_dim,
            "consolidation_threshold": self.consolidation_threshold,
            "tag_blocks_metadata": {
                tag: {
                    "episode_ids": block.episode_ids,
                    "shape": list(block.embeddings.shape)
                }
                for tag, block in self.tag_blocks.items()
            },
            "temp_episode_ids": self.temp_episode_ids,
            "temp_tags": [list(tags) for tags in self.temp_tags],
            "episode_location": {
                ep_id: (list(tags), is_temp)
                for ep_id, (tags, is_temp) in self.episode_location.items()
            }
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典反序列化元数据"""
        self.embedding_dim = data["embedding_dim"]
        self.consolidation_threshold = data["consolidation_threshold"]
        
        # 恢复 temp 数据
        self.temp_episode_ids = data["temp_episode_ids"]
        self.temp_tags = [set(tags) for tags in data["temp_tags"]]
        
        # 恢复 episode_location
        self.episode_location = {
            ep_id: (set(tags), is_temp)
            for ep_id, (tags, is_temp) in data["episode_location"].items()
        }
        
        # tag_blocks 的 embeddings 需要从 npz 文件加载，这里只恢复 episode_ids
        # （实际的 embeddings 在 load() 方法中加载）
    
    def save(self, path: str) -> None:
        """
        保存索引到文件
        
        保存两个文件:
        - {path}_metadata.json: 元数据
        - {path}_vectors.npz: 所有向量数据
        """
        import json
        from pathlib import Path
        
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存元数据
        metadata_path = base_path.with_name(base_path.stem + "_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 2. 保存所有向量数据
        vectors_path = base_path.with_name(base_path.stem + "_vectors.npz")
        
        # 准备保存的数组字典
        arrays_to_save = {}
        
        # 保存 tag_blocks 的 embeddings
        for tag, block in self.tag_blocks.items():
            # 使用 tag 作为 key（替换特殊字符）
            safe_tag = tag.replace('/', '_').replace(' ', '_')
            arrays_to_save[f"tag_block_{safe_tag}"] = block.embeddings
        
        # 保存 temp_embeddings
        if self.temp_embeddings:
            arrays_to_save["temp_embeddings"] = np.array(self.temp_embeddings)
        
        # 保存
        if arrays_to_save:
            np.savez_compressed(vectors_path, **arrays_to_save)
    
    def load(self, path: str) -> None:
        """
        从文件加载索引
        
        加载两个文件:
        - {path}_metadata.json: 元数据
        - {path}_vectors.npz: 所有向量数据
        """
        import json
        from pathlib import Path
        
        base_path = Path(path)
        
        # 1. 加载元数据
        metadata_path = base_path.with_name(base_path.stem + "_metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.from_dict(data)
        
        # 2. 加载向量数据
        vectors_path = base_path.with_name(base_path.stem + "_vectors.npz")
        
        if vectors_path.exists():
            npz_data = np.load(vectors_path)
            
            # 恢复 tag_blocks
            self.tag_blocks = {}
            for tag, metadata in data["tag_blocks_metadata"].items():
                safe_tag = tag.replace('/', '_').replace(' ', '_')
                key = f"tag_block_{safe_tag}"
                
                if key in npz_data:
                    embeddings = npz_data[key]
                    episode_ids = metadata["episode_ids"]
                    
                    self.tag_blocks[tag] = VectorBlock(
                        embeddings=embeddings,
                        episode_ids=episode_ids
                    )
            
            # 恢复 temp_embeddings
            if "temp_embeddings" in npz_data:
                temp_array = npz_data["temp_embeddings"]
                self.temp_embeddings = [temp_array[i] for i in range(len(temp_array))]
            else:
                self.temp_embeddings = []
        else:
            self.tag_blocks = {}
            self.temp_embeddings = []