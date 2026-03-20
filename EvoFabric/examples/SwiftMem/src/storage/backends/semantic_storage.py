# src/storage/backends/semantic_storage.py
"""
SemanticMemory 持久化存储（基于 SQLite）
"""

import sqlite3
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import threading

from ...models.semantic import SemanticMemory


class SemanticStorage:
    """SemanticMemory 持久化存储"""
    
    def __init__(self, db_path: str = "./data/semantic.db"):
        """
        初始化 SemanticMemory 存储
        
        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = db_path
        self._local = threading.local()  # 线程本地存储
        self._ensure_db_exists()
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取线程安全的数据库连接"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _ensure_db_exists(self) -> None:
        """确保数据库目录存在"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_schema(self) -> None:
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # SemanticMemories 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL,
                timestamp TEXT NOT NULL,
                source_episode_ids TEXT,
                tags TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_user_id 
            ON semantic_memories(user_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_type 
            ON semantic_memories(memory_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_importance 
            ON semantic_memories(importance DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_timestamp 
            ON semantic_memories(timestamp)
        """)
        
        conn.commit()
    
    def save_memory(self, memory: SemanticMemory) -> None:
        """
        保存 SemanticMemory
        
        Args:
            memory: SemanticMemory 对象
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO semantic_memories (
                    memory_id, user_id, content, memory_type, importance,
                    timestamp, source_episode_ids, tags, metadata,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.memory_id,
                memory.user_id,
                memory.content,
                memory.memory_type,
                memory.importance,
                memory.timestamp.isoformat(),
                json.dumps(memory.source_episode_ids, ensure_ascii=False),
                json.dumps(memory.tags, ensure_ascii=False),
                json.dumps(memory.metadata, ensure_ascii=False),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"保存 SemanticMemory 失败: {str(e)}")
    
    def load_memory(self, memory_id: str) -> Optional[SemanticMemory]:
        """
        加载单个 SemanticMemory
        
        Args:
            memory_id: Memory ID
            
        Returns:
            SemanticMemory 对象，如果不存在返回 None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM semantic_memories WHERE memory_id = ?
        """, (memory_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return self._row_to_memory(row)
    
    def batch_load_memories(self, memory_ids: List[str]) -> List[SemanticMemory]:
        """批量加载 Memories，保持输入顺序"""
        if not memory_ids:
            return []
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        placeholders = ','.join(['?'] * len(memory_ids))
        cursor.execute(f"""
            SELECT * FROM semantic_memories
            WHERE memory_id IN ({placeholders})
        """, memory_ids)
        
        rows = cursor.fetchall()
        
        # 创建字典以保持顺序
        memory_dict = {}
        for row in rows:
            memory_dict[row['memory_id']] = self._row_to_memory(row)
        
        # 按照输入的 memory_ids 顺序返回
        return [memory_dict[mid] for mid in memory_ids if mid in memory_dict]
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        删除 SemanticMemory
        
        Args:
            memory_id: Memory ID
            
        Returns:
            是否删除成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM semantic_memories WHERE memory_id = ?
            """, (memory_id,))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"删除 SemanticMemory 失败: {str(e)}")
    
    def list_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[str]:
        """列出用户的 Memory IDs"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if memory_type:
            cursor.execute("""
                SELECT memory_id FROM semantic_memories
                WHERE user_id = ? AND memory_type = ?
                ORDER BY importance DESC, memory_id ASC
                LIMIT ? OFFSET ?
            """, (user_id, memory_type, limit, offset))
        else:
            cursor.execute("""
                SELECT memory_id FROM semantic_memories
                WHERE user_id = ?
                ORDER BY importance DESC, memory_id ASC
                LIMIT ? OFFSET ?
            """, (user_id, limit, offset))
        
        return [row['memory_id'] for row in cursor.fetchall()]
    
    def count_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None
    ) -> int:
        """
        统计用户的 Memory 数量
        
        Args:
            user_id: 用户ID
            memory_type: 记忆类型过滤（可选）
            
        Returns:
            Memory 数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if memory_type:
            cursor.execute("""
                SELECT COUNT(*) as count FROM semantic_memories
                WHERE user_id = ? AND memory_type = ?
            """, (user_id, memory_type))
        else:
            cursor.execute("""
                SELECT COUNT(*) as count FROM semantic_memories
                WHERE user_id = ?
            """, (user_id,))
        
        return cursor.fetchone()['count']
    
    def get_memories_by_importance(
        self,
        user_id: str,
        min_importance: float = 0.0,
        max_importance: float = 1.0,
        limit: int = 100
    ) -> List[str]:
        """
        按重要性范围查询 Memories
        
        Args:
            user_id: 用户ID
            min_importance: 最小重要性
            max_importance: 最大重要性
            limit: 返回数量限制
            
        Returns:
            Memory ID 列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT memory_id FROM semantic_memories
            WHERE user_id = ? 
            AND importance >= ? 
            AND importance <= ?
            ORDER BY importance DESC
            LIMIT ?
        """, (user_id, min_importance, max_importance, limit))
        
        return [row['memory_id'] for row in cursor.fetchall()]
    
    def get_memories_in_time_range(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[str]:
        """
        获取指定时间范围内的 Memory IDs
        
        Args:
            user_id: 用户ID
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制
            
        Returns:
            Memory ID 列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT memory_id FROM semantic_memories
            WHERE user_id = ? 
            AND timestamp >= ? 
            AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (
            user_id,
            start_time.isoformat(),
            end_time.isoformat(),
            limit
        ))
        
        return [row['memory_id'] for row in cursor.fetchall()]
    
    def search_by_tags(
        self,
        user_id: str,
        tags: List[str],
        limit: int = 100
    ) -> List[str]:
        """
        根据标签搜索 Memories
        
        Args:
            user_id: 用户ID
            tags: 标签列表
            limit: 返回数量限制
            
        Returns:
            Memory ID 列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 构造查询条件（任意标签匹配）
        conditions = []
        params = [user_id]
        
        for tag in tags:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        
        query = f"""
            SELECT memory_id FROM semantic_memories
            WHERE user_id = ? AND ({' OR '.join(conditions)})
            ORDER BY importance DESC
            LIMIT ?
        """
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [row['memory_id'] for row in cursor.fetchall()]
    
    def get_memories_by_source_episode(
        self,
        user_id: str,
        episode_id: str,
        limit: int = 100
    ) -> List[str]:
        """
        查找来源于特定 Episode 的 Memories
        
        Args:
            user_id: 用户ID
            episode_id: Episode ID
            limit: 返回数量限制
            
        Returns:
            Memory ID 列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT memory_id FROM semantic_memories
            WHERE user_id = ? 
            AND source_episode_ids LIKE ?
            ORDER BY importance DESC
            LIMIT ?
        """, (user_id, f'%"{episode_id}"%', limit))
        
        return [row['memory_id'] for row in cursor.fetchall()]
    
    def update_importance(self, memory_id: str, new_importance: float) -> bool:
        """
        更新 Memory 的重要性
        
        Args:
            memory_id: Memory ID
            new_importance: 新的重要性分数
            
        Returns:
            是否更新成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE semantic_memories 
                SET importance = ?, updated_at = ?
                WHERE memory_id = ?
            """, (new_importance, datetime.now().isoformat(), memory_id))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"更新 importance 失败: {str(e)}")
    
    def clear_user_memories(self, user_id: str) -> int:
        """
        清空用户的所有 Memories
        
        Args:
            user_id: 用户ID
            
        Returns:
            删除的 Memory 数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM semantic_memories WHERE user_id = ?
            """, (user_id,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"清空 Memories 失败: {str(e)}")
    
    def get_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户的 Memory 统计信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            统计信息字典
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 总数和按类型统计
        cursor.execute("""
            SELECT 
                COUNT(*) as total_count,
                AVG(importance) as avg_importance,
                MAX(importance) as max_importance,
                MIN(importance) as min_importance
            FROM semantic_memories
            WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # 按类型统计
        cursor.execute("""
            SELECT memory_type, COUNT(*) as count
            FROM semantic_memories
            WHERE user_id = ?
            GROUP BY memory_type
        """, (user_id,))
        
        type_counts = {row['memory_type']: row['count'] for row in cursor.fetchall()}
        
        return {
            'total_count': row['total_count'],
            'avg_importance': row['avg_importance'] or 0.0,
            'max_importance': row['max_importance'] or 0.0,
            'min_importance': row['min_importance'] or 0.0,
            'type_counts': type_counts
        }
    
    def _row_to_memory(self, row: sqlite3.Row) -> SemanticMemory:
        """将数据库行转换为 SemanticMemory 对象"""
        return SemanticMemory(
            memory_id=row['memory_id'],
            user_id=row['user_id'],
            content=row['content'],
            memory_type=row['memory_type'],
            importance=row['importance'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            source_episode_ids=json.loads(row['source_episode_ids']) if row['source_episode_ids'] else [],
            tags=json.loads(row['tags']) if row['tags'] else [],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    def close(self) -> None:
        """关闭数据库连接"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')


__all__ = ['SemanticStorage']