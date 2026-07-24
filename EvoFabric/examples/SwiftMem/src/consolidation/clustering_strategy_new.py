# src/consolidation/clustering_strategy.py

"""
标签聚类策略
"""

from typing import Set, List, Dict, Optional
from collections import defaultdict, deque
import numpy as np

from .types import TagCluster
from ..storage.indexing.tag_dag_index import TagDAGIndex


class TagClusteringStrategy:
    """
    基于 TAG-DAG 的标签聚类策略
    
    通过分析标签之间的层次关系和共现模式，将标签聚类成相关的组。
    """
    
    def __init__(
        self,
        min_cluster_size: int = 1,
        max_cluster_size: int = 50,
        min_cohesion: float = 0.0
    ):
        """
        初始化聚类策略
        
        Args:
            min_cluster_size: 最小簇大小（包含的标签数）
            max_cluster_size: 最大簇大小
            min_cohesion: 最小内聚度阈值
        """
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.min_cohesion = min_cohesion
    
    def cluster_tags(
        self,
        tag_dag: TagDAGIndex,
        active_tags: Set[str]
    ) -> List[TagCluster]:
        """
        对活跃标签进行聚类
        
        Args:
            tag_dag: TAG-DAG 索引
            active_tags: 活跃的标签集合
            
        Returns:
            标签簇列表，按内聚度降序排列
        """
        if not active_tags:
            return []
        
        # 1. 构建标签关系图
        tag_graph = self._build_tag_graph(tag_dag, active_tags)
        
        # 2. 使用连通分量算法进行初始聚类
        connected_components = self._find_connected_components(tag_graph, active_tags)
        
        # 3. 将大簇分解，小簇合并
        clusters = self._refine_clusters(connected_components, tag_graph)
        
        # 4. 计算每个簇的内聚度和质心
        result_clusters = []
        for i, cluster_tags in enumerate(clusters):
            if len(cluster_tags) < 1:  # 至少要有 1 个标签
                continue
            
            # 计算内聚度
            cohesion = self._calculate_cohesion(cluster_tags, tag_graph)
            
            # 选择质心（度数最高的标签）
            centroid = self._select_centroid(cluster_tags, tag_graph)
            
            # 创建簇对象
            cluster = TagCluster(
                cluster_id=f"cluster_{i}",  
                tags=cluster_tags,
                centroid_tag=centroid,
                cohesion_score=cohesion
            )
            
            result_clusters.append(cluster)
        
        # 5. 按内聚度降序排列
        result_clusters.sort(key=lambda c: c.cohesion_score, reverse=True)
        
        return result_clusters
    
    def _build_tag_graph(
        self,
        tag_dag: TagDAGIndex,
        active_tags: Set[str]
    ) -> Dict[str, Set[str]]:
        """
        构建标签关系图（无向图）
        
        Returns:
            邻接表表示的图 {tag: {connected_tags}}
        """
        import logging
        logger = logging.getLogger(__name__)
        
        graph = defaultdict(set)
        
        # 初始化所有活跃标签
        for tag in active_tags:
            graph[tag] = set()
        
        logger.info(f"🔍 Building tag graph for {len(active_tags)} active tags")
        
        edges_added = 0
        tags_with_relations = 0
        
       
        for tag in active_tags:
            try:
                # 获取相关标签（父标签 + 子标签，最大距离 2 层）
                related_tags = tag_dag.get_related_tags(
                    tag=tag,
                    relation_type='both',  # 'parent', 'child', 'both'
                    max_distance=2         # DAG 遍历深度
                )
                
                if related_tags:
                    tags_with_relations += 1
                    
                    # 只保留也在 active_tags 中的相关标签
                    for related_tag in related_tags:
                        if related_tag in active_tags and related_tag != tag:
                            graph[tag].add(related_tag)
                            graph[related_tag].add(tag)
                            edges_added += 1
                            
            except Exception as e:
                logger.debug(f"⚠️  Could not get relations for '{tag}': {e}")
                continue
        
        # 去重计数（每条边被计算了两次）
        edges_added = edges_added // 2
        
        logger.info(f"📊 Graph built: {edges_added} edges, {tags_with_relations}/{len(active_tags)} tags with relations")
        
        # Fallback: 如果没有任何关系，使用文本相似度
        if edges_added == 0:
            logger.warning("⚠️  No tag relations found in TAG-DAG, using text similarity fallback")
            graph = self._build_graph_by_text_similarity(active_tags)
        
        return dict(graph)

    def _build_graph_by_text_similarity(
        self,
        active_tags: Set[str]
    ) -> Dict[str, Set[str]]:
        """
        使用文本相似度构建图（fallback）
        
        策略：如果两个 tag 有公共词，认为它们相关
        
        Args:
            active_tags: 活跃标签集合
            
        Returns:
            邻接表表示的图
        """
        import logging
        logger = logging.getLogger(__name__)
        
        graph = defaultdict(set)
        
        # 将 tags 分词
        tag_words = {}
        for tag in active_tags:
            # 分词：下划线分隔 + 小写
            words = set(tag.lower().replace('_', ' ').split())
            tag_words[tag] = words
        
        # 计算 Jaccard 相似度
        edges_added = 0
        for tag1 in active_tags:
            for tag2 in active_tags:
                if tag1 >= tag2:  # 避免重复
                    continue
                
                words1 = tag_words[tag1]
                words2 = tag_words[tag2]
                
                # Jaccard 相似度
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                
                if union > 0:
                    similarity = intersection / union
                    
                    # 如果相似度 > 0.3，添加边
                    if similarity > 0.3:
                        graph[tag1].add(tag2)
                        graph[tag2].add(tag1)
                        edges_added += 1
        
        logger.info(f"📊 Text similarity graph: {edges_added} edges")
        
        return dict(graph)
    
    def _find_connected_components(
        self,
        graph: Dict[str, Set[str]],
        all_tags: Set[str]
    ) -> List[Set[str]]:
        """
        使用 BFS 找出所有连通分量
        
        Returns:
            连通分量列表，每个分量是一组标签
        """
        visited = set()
        components = []
        
        for tag in all_tags:
            if tag in visited:
                continue
            
            # BFS 遍历连通分量
            component = set()
            queue = deque([tag])
            visited.add(tag)
            
            while queue:
                current = queue.popleft()
                component.add(current)
                
                # 访问邻居
                for neighbor in graph.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
        
        return components
    
    def _refine_clusters(
        self,
        components: List[Set[str]],
        graph: Dict[str, Set[str]]
    ) -> List[Set[str]]:
        """
        优化簇：分解大簇，合并小簇
        
        Returns:
            优化后的簇列表
        """
        refined = []
        small_clusters = []
        
        for component in components:
            # 即使是单个标签也保留（如果 min_cluster_size <= 1）
            if len(component) <= self.max_cluster_size:
                if len(component) >= max(1, self.min_cluster_size):
                    refined.append(component)
                elif len(component) > 0:
                    small_clusters.append(component)
            else:
                # 分解大簇
                subclusters = self._split_large_cluster(component, graph)
                refined.extend(subclusters)
        
        # 尝试合并小簇
        if small_clusters:
            merged = self._merge_small_clusters(small_clusters, graph)
            refined.extend(merged)
        
        return refined
    
    def _split_large_cluster(
        self,
        cluster: Set[str],
        graph: Dict[str, Set[str]]
    ) -> List[Set[str]]:
        """分解过大的簇"""
        degrees = {
            tag: len(graph.get(tag, set()) & cluster)
            for tag in cluster
        }
        
        num_centers = max(2, len(cluster) // self.max_cluster_size)
        centers = sorted(degrees.keys(), key=lambda t: degrees[t], reverse=True)[:num_centers]
        
        subclusters = {center: {center} for center in centers}
        
        for tag in cluster:
            if tag in centers:
                continue
            
            connected_centers = [c for c in centers if c in graph.get(tag, set())]
            
            if connected_centers:
                best_center = max(connected_centers, key=lambda c: degrees[c])
                subclusters[best_center].add(tag)
            else:
                subclusters[centers[0]].add(tag)
        
        return [s for s in subclusters.values() if len(s) >= 1]
    
    def _merge_small_clusters(
        self,
        small_clusters: List[Set[str]],
        graph: Dict[str, Set[str]]
    ) -> List[Set[str]]:
        """合并小簇"""
        merged = []
        used = set()
        
        for i, cluster1 in enumerate(small_clusters):
            if i in used:
                continue
            
            current_cluster = cluster1.copy()
            used.add(i)
            
            for j, cluster2 in enumerate(small_clusters):
                if j in used or j <= i:
                    continue
                
                has_connection = any(
                    tag2 in graph.get(tag1, set())
                    for tag1 in current_cluster
                    for tag2 in cluster2
                )
                
                if has_connection and len(current_cluster) + len(cluster2) <= self.max_cluster_size:
                    current_cluster.update(cluster2)
                    used.add(j)
            
            merged.append(current_cluster)
        
        return merged
    
    def _calculate_cohesion(
        self,
        cluster_tags: Set[str],
        graph: Dict[str, Set[str]]
    ) -> float:
        """计算簇的内聚度"""
        if len(cluster_tags) <= 1:
            return 1.0
        
        edges_in_cluster = sum(
            len(graph.get(tag, set()) & cluster_tags)
            for tag in cluster_tags
        ) / 2
        
        max_edges = len(cluster_tags) * (len(cluster_tags) - 1) / 2
        
        if max_edges == 0:
            return 0.0
        
        cohesion = edges_in_cluster / max_edges
        return min(1.0, cohesion)
    
    def _select_centroid(
        self,
        cluster_tags: Set[str],
        graph: Dict[str, Set[str]]
    ) -> str:
        """选择簇的质心"""
        if len(cluster_tags) == 1:
            return list(cluster_tags)[0]
        
        degrees = {
            tag: len(graph.get(tag, set()) & cluster_tags)
            for tag in cluster_tags
        }
        
        return max(degrees.keys(), key=lambda t: degrees[t])

__all__ = ['TagClusteringStrategy']