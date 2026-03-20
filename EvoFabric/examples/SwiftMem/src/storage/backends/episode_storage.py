# src/storage/backends/episode_storage.py
"""
Episode 持久化存储（基于 SQLite）
"""

import sqlite3
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import threading

from ...models.episode import Episode
from ...models.message import Message


class EpisodeStorage:
    """Episode 持久化存储"""
    
    def __init__(self, db_path: str = "./data/episodes.db"):
        """
        初始化 Episode 存储
        
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
            
            # 启用外键约束
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            
        return self._local.connection
    
    def _ensure_db_exists(self) -> None:
        """确保数据库目录存在"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_schema(self) -> None:
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Episodes 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                content TEXT,
                boundary_reason TEXT,
                timestamp TEXT NOT NULL,
                tags TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Messages 表（与 Episode 关联）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                message_order INTEGER NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
            )
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_user_id 
            ON episodes(user_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_timestamp 
            ON episodes(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_episode_id 
            ON messages(episode_id)
        """)
        
        conn.commit()
    
    def save_episode(self, episode: Episode) -> None:
        """
        保存 Episode
        
        Args:
            episode: Episode 对象
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 保存 Episode 元数据
            cursor.execute("""
                INSERT OR REPLACE INTO episodes (
                    episode_id, user_id, title, content, boundary_reason,
                    timestamp, tags, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.episode_id,
                episode.user_id,
                episode.title,
                episode.content,
                episode.boundary_reason,
                episode.timestamp.isoformat(),
                json.dumps(episode.tags, ensure_ascii=False),
                json.dumps(episode.metadata, ensure_ascii=False),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            # 删除旧的 messages（如果存在）
            cursor.execute("""
                DELETE FROM messages WHERE episode_id = ?
            """, (episode.episode_id,))
            
            # 保存 Messages
            for order, message in enumerate(episode.messages):
                cursor.execute("""
                    INSERT INTO messages (
                        message_id, episode_id, role, content, timestamp,
                        metadata, message_order
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.message_id,
                    episode.episode_id,
                    message.role,
                    message.content,
                    message.timestamp.isoformat(),
                    json.dumps(message.metadata, ensure_ascii=False),
                    order
                ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"保存 Episode 失败: {str(e)}")
    
    def load_episode(self, episode_id: str) -> Optional[Episode]:
        """
        加载单个 Episode
        
        Args:
            episode_id: Episode ID
            
        Returns:
            Episode 对象，如果不存在返回 None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 加载 Episode 元数据
        cursor.execute("""
            SELECT * FROM episodes WHERE episode_id = ?
        """, (episode_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # 加载 Messages
        cursor.execute("""
            SELECT * FROM messages 
            WHERE episode_id = ?
            ORDER BY message_order ASC
        """, (episode_id,))
        
        message_rows = cursor.fetchall()
        messages = [
            Message(
                message_id=msg['message_id'],
                role=msg['role'],
                content=msg['content'],
                timestamp=datetime.fromisoformat(msg['timestamp']),
                metadata=json.loads(msg['metadata']) if msg['metadata'] else {}
            )
            for msg in message_rows
        ]
        
        # 构造 Episode 对象
        episode = Episode(
            episode_id=row['episode_id'],
            user_id=row['user_id'],
            messages=messages,
            title=row['title'] or "",
            content=row['content'] or "",
            boundary_reason=row['boundary_reason'] or "",
            timestamp=datetime.fromisoformat(row['timestamp']),
            tags=json.loads(row['tags']) if row['tags'] else [],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
        
        return episode
    
    def batch_load_episodes(self, episode_ids: List[str]) -> List[Episode]:
        """
        批量加载 Episodes（性能优化）
        
        Args:
            episode_ids: Episode ID 列表
            
        Returns:
            Episode 对象列表
        """
        if not episode_ids:
            return []
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 构造 SQL 查询
        placeholders = ','.join('?' * len(episode_ids))
        
        # 批量加载 Episodes
        cursor.execute(f"""
            SELECT * FROM episodes 
            WHERE episode_id IN ({placeholders})
        """, episode_ids)
        
        episode_rows = {row['episode_id']: row for row in cursor.fetchall()}
        
        # 批量加载 Messages
        cursor.execute(f"""
            SELECT * FROM messages 
            WHERE episode_id IN ({placeholders})
            ORDER BY episode_id, message_order ASC
        """, episode_ids)
        
        # 组织 Messages
        messages_by_episode: Dict[str, List[Message]] = {}
        for msg_row in cursor.fetchall():
            episode_id = msg_row['episode_id']
            if episode_id not in messages_by_episode:
                messages_by_episode[episode_id] = []
            
            messages_by_episode[episode_id].append(
                Message(
                    message_id=msg_row['message_id'],
                    role=msg_row['role'],
                    content=msg_row['content'],
                    timestamp=datetime.fromisoformat(msg_row['timestamp']),
                    metadata=json.loads(msg_row['metadata']) if msg_row['metadata'] else {}
                )
            )
        
        # 构造 Episode 对象
        episodes = []
        for episode_id in episode_ids:
            if episode_id not in episode_rows:
                continue
            
            row = episode_rows[episode_id]
            messages = messages_by_episode.get(episode_id, [])
            
            episode = Episode(
                episode_id=row['episode_id'],
                user_id=row['user_id'],
                messages=messages,
                title=row['title'] or "",
                content=row['content'] or "",
                boundary_reason=row['boundary_reason'] or "",
                timestamp=datetime.fromisoformat(row['timestamp']),
                tags=json.loads(row['tags']) if row['tags'] else [],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            episodes.append(episode)
        
        return episodes
    
    def delete_episode(self, episode_id: str) -> bool:
        """
        删除 Episode
        
        Args:
            episode_id: Episode ID
            
        Returns:
            是否删除成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 由于设置了 ON DELETE CASCADE，删除 Episode 会自动删除关联的 Messages
            cursor.execute("""
                DELETE FROM episodes WHERE episode_id = ?
            """, (episode_id,))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"删除 Episode 失败: {str(e)}")
    
    def list_episodes(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[str]:
        """
        列出用户的 Episode IDs
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Episode ID 列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT episode_id FROM episodes
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset))
        
        return [row['episode_id'] for row in cursor.fetchall()]
    
    def count_episodes(self, user_id: str) -> int:
        """
        统计用户的 Episode 数量
        
        Args:
            user_id: 用户ID
            
        Returns:
            Episode 数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) as count FROM episodes
            WHERE user_id = ?
        """, (user_id,))
        
        return cursor.fetchone()['count']
    
    def get_episodes_in_time_range(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[str]:
        """
        获取指定时间范围内的 Episode IDs
        
        Args:
            user_id: 用户ID
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制
            
        Returns:
            Episode ID 列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT episode_id FROM episodes
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
        
        return [row['episode_id'] for row in cursor.fetchall()]
    
    def search_by_tags(
        self,
        user_id: str,
        tags: List[str],
        limit: int = 100
    ) -> List[str]:
        """
        根据标签搜索 Episodes
        
        Args:
            user_id: 用户ID
            tags: 标签列表
            limit: 返回数量限制
            
        Returns:
            Episode ID 列表
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
            SELECT episode_id FROM episodes
            WHERE user_id = ? AND ({' OR '.join(conditions)})
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [row['episode_id'] for row in cursor.fetchall()]
    
    def clear_user_episodes(self, user_id: str) -> int:
        """
        清空用户的所有 Episodes
        
        Args:
            user_id: 用户ID
            
        Returns:
            删除的 Episode 数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM episodes WHERE user_id = ?
            """, (user_id,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"清空 Episodes 失败: {str(e)}")
    
    def close(self) -> None:
        """关闭数据库连接"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')


__all__ = ['EpisodeStorage']