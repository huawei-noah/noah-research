
# storage/indexing/tag_dag_index.py
"""
L2: Tag DAG Index
支持基于 DAG 的标签检索，比全量搜索更快
"""

from typing import List, Set, Dict, Optional, Any
from collections import defaultdict, deque
import networkx as nx


class TagNode:
    """标签节点"""
    
    __slots__ = ['tag', 'episode_ids', 'parent_tags', 'child_tags', 'embedding']
    
    def __init__(self, tag: str):
        self.tag = tag
        self.episode_ids: Set[str] = set()
        self.parent_tags: Set[str] = set()  # 更抽象的标签
        self.child_tags: Set[str] = set()   # 更具体的标签
        self.embedding: Optional[List[float]] = None  # tag的embedding


class TagDAGIndex:
    """
    标签 DAG 索引
    
    核心思想：
    1. Tags 形成 DAG 结构（如：工作 -> 编程 -> Python）
    2. 查询时沿着 DAG 定位相关标签
    3. 只在相关标签的 episodes 中进行语义搜索
    
    性能目标：
    - 标签定位: O(d)，d是DAG深度（通常 < 5）
    - 检索episodes: O(1) 哈希查找
    - 比全量搜索快 10-100x
    """
    
    def __init__(self):
        # 核心数据结构
        self.tag_nodes: Dict[str, TagNode] = {}
        self.dag = nx.DiGraph()  # 使用 NetworkX 管理 DAG
        
        # 辅助索引
        self.episode_to_tags: Dict[str, Set[str]] = defaultdict(set)
        
        # Query-Tag 对齐模型（预留RL接口）
        self.query_tag_aligner = None  # 后续可以接入 RL 模型
    
    def add_episode_tags(
        self,
        episode_id: str,
        tags: List[str],
        tag_relations: Optional[List[tuple[str, str]]] = None
    ):
        """
        添加 episode 的标签
        
        Args:
            episode_id: Episode ID
            tags: 标签列表
            tag_relations: 可选的标签关系 [(parent, child), ...]
        """
        for tag in tags:
            # 创建标签节点
            if tag not in self.tag_nodes:
                self.tag_nodes[tag] = TagNode(tag)
                self.dag.add_node(tag)
            
            # 关联 episode
            self.tag_nodes[tag].episode_ids.add(episode_id)
            self.episode_to_tags[episode_id].add(tag)
        
        # 建立标签关系
        if tag_relations:
            for parent, child in tag_relations:
                self._add_tag_relation(parent, child)
    
    def query_by_tags(
        self,
        query_tags: List[str],
        expand_depth: int = 1,
        use_alignment: bool = False
    ) -> Set[str]:
        """
        基于标签查询
        
        Args:
            query_tags: 查询标签
            expand_depth: DAG 扩展深度（0=精确匹配，1=包含子标签，2=包含孙标签...）
            use_alignment: 是否使用 Query-Tag 对齐模型
            
        Returns:
            符合条件的 episode_ids
            
        性能: O(t * d)，t是query_tags数量，d是expand_depth
        """
        # 1. Query-Tag 对齐（可选）
        if use_alignment and self.query_tag_aligner:
            # 使用 RL 模型找到最相关的标签
            aligned_tags = self.query_tag_aligner.align(query_tags)
        else:
            aligned_tags = query_tags
        
        # 2. DAG 扩展
        expanded_tags = self._expand_tags(aligned_tags, expand_depth)
        
        # 3. 收集所有相关的 episode_ids
        result_ids: Set[str] = set()
        
        for tag in expanded_tags:
            if tag in self.tag_nodes:
                result_ids.update(self.tag_nodes[tag].episode_ids)
        
        return result_ids
    
    def get_related_tags(
        self,
        tag: str,
        relation_type: str = 'both',  # 'parent', 'child', 'both'
        max_distance: int = 2
    ) -> Set[str]:
        """
        获取相关标签（用于 tag 扩展）
        """
        if tag not in self.tag_nodes:
            return set()
        
        related = {tag}
        
        # BFS 遍历 DAG
        if relation_type in ['parent', 'both']:
            related.update(
                self._bfs_traverse(tag, direction='predecessors', max_depth=max_distance)
            )
        
        if relation_type in ['child', 'both']:
            related.update(
                self._bfs_traverse(tag, direction='successors', max_depth=max_distance)
            )
        
        return related
    
    def _expand_tags(
        self,
        tags: List[str],
        depth: int
    ) -> Set[str]:
        """扩展标签（包含子标签）"""
        expanded = set(tags)
        
        for tag in tags:
            if depth > 0:
                # 包含所有后代标签
                descendants = self._bfs_traverse(
                    tag,
                    direction='successors',
                    max_depth=depth
                )
                expanded.update(descendants)
        
        return expanded
    
    def _bfs_traverse(
        self,
        start_tag: str,
        direction: str,  # 'predecessors' or 'successors'
        max_depth: int
    ) -> Set[str]:
        """BFS 遍历 DAG"""
        if start_tag not in self.dag:
            return set()
        
        visited = set()
        queue = deque([(start_tag, 0)])
        
        while queue:
            current_tag, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            visited.add(current_tag)
            
            # 获取邻居
            if direction == 'predecessors':
                neighbors = self.dag.predecessors(current_tag)
            else:
                neighbors = self.dag.successors(current_tag)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return visited
    
    def _add_tag_relation(self, parent: str, child: str):
        """添加标签关系"""
        # 确保节点存在
        for tag in [parent, child]:
            if tag not in self.tag_nodes:
                self.tag_nodes[tag] = TagNode(tag)
                self.dag.add_node(tag)
        
        # 添加边（检查是否会形成环）
        if not nx.has_path(self.dag, child, parent):
            self.dag.add_edge(parent, child)
            self.tag_nodes[parent].child_tags.add(child)
            self.tag_nodes[child].parent_tags.add(parent)
    
    def update_tag_embeddings(self, tag_embeddings: Dict[str, List[float]]):
        """更新标签的 embeddings（用于后续的对齐模型）"""
        for tag, embedding in tag_embeddings.items():
            if tag in self.tag_nodes:
                self.tag_nodes[tag].embedding = embedding
    
    def remove_episode(self, episode_id: str):
        """删除 episode"""
        if episode_id not in self.episode_to_tags:
            return
        
        for tag in self.episode_to_tags[episode_id]:
            if tag in self.tag_nodes:
                self.tag_nodes[tag].episode_ids.discard(episode_id)
        
        del self.episode_to_tags[episode_id]
    
    def get_stats(self) -> dict:
        return {
            "total_tags": len(self.tag_nodes),
            "total_relations": self.dag.number_of_edges(),
            "avg_episodes_per_tag": (
                sum(len(node.episode_ids) for node in self.tag_nodes.values())
                / len(self.tag_nodes) if self.tag_nodes else 0
            ),
            "dag_depth": (
                nx.dag_longest_path_length(self.dag)
                if self.dag.number_of_nodes() > 0 else 0
            )
        }
    
    # ========== 序列化支持 ==========
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "tag_nodes": {
                tag: {
                    "episode_ids": list(node.episode_ids),
                    "parent_tags": list(node.parent_tags),
                    "child_tags": list(node.child_tags),
                    "embedding": node.embedding
                }
                for tag, node in self.tag_nodes.items()
            },
            "dag_edges": list(self.dag.edges()),
            "episode_to_tags": {
                ep_id: list(tags)
                for ep_id, tags in self.episode_to_tags.items()
            }
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典反序列化"""
        # 恢复 tag_nodes
        self.tag_nodes = {}
        for tag, node_data in data["tag_nodes"].items():
            node = TagNode(tag)
            node.episode_ids = set(node_data["episode_ids"])
            node.parent_tags = set(node_data["parent_tags"])
            node.child_tags = set(node_data["child_tags"])
            node.embedding = node_data["embedding"]
            self.tag_nodes[tag] = node
        
        # 恢复 DAG
        self.dag = nx.DiGraph()
        self.dag.add_edges_from(data["dag_edges"])
        
        # 恢复 episode_to_tags
        self.episode_to_tags = defaultdict(set)
        for ep_id, tags in data["episode_to_tags"].items():
            self.episode_to_tags[ep_id] = set(tags)
    
    def save(self, path: str) -> None:
        """保存到文件"""
        import json
        from pathlib import Path
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """从文件加载"""
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.from_dict(data)
