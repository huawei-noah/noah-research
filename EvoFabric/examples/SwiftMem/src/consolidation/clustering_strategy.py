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

    """
    
    def __init__(
        self,
        min_cluster_size: int = 2,
        max_cluster_size: int = 50,
        min_cohesion: float = 0.3
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
        graph = defaultdict(set)
        
        # 初始化所有活跃标签
        for tag in active_tags:
            graph[tag] = set()
        
        # 尝试通过 get_related_tags 构建边
        for tag in active_tags:
            try:
                related_tags = tag_dag.get_related_tags(tag, min_cooccurrence=1)
                
                if related_tags:
                    for related_tag, score in related_tags:
                        if related_tag in active_tags and related_tag != tag:
                            graph[tag].add(related_tag)
                            graph[related_tag].add(tag)
            except Exception:
                # 方法不存在或出错，跳过
                pass
        
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
                if len(component) >= self.min_cluster_size or len(component) == 1:
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