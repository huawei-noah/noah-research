# storage/indexing/temporal_index.py

import bisect
from datetime import datetime
from typing import List, Set, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class TimeInterval:
    """Represents a time range"""
    start: datetime
    end: datetime
    
    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within this interval"""
        return self.start <= timestamp <= self.end
    
    def overlaps(self, other: 'TimeInterval') -> bool:
        """Check if this interval overlaps with another"""
        return not (self.end < other.start or other.end < self.start)


@dataclass
class TemporalNode:
    """Node in temporal index tree"""
    timestamp: datetime
    episode_id: str


class TemporalIndex:
    """
    L1: Temporal Index
    
    Fast time-based filtering using sorted timeline per user.
    Supports:
    - Single/multiple time range queries
    - Recent episode queries
    - Temporal joins
    """
    
    def __init__(self):
        # Per-user sorted timeline: user_id -> List[TemporalNode]
        self.user_timelines: Dict[str, List[TemporalNode]] = {}
        
        # Quick lookup: episode_id -> (user_id, timestamp)
        self.episode_to_time: Dict[str, Tuple[str, datetime]] = {}
        
        self.total_episodes = 0
    
    def add_episode(self, episode_id: str, user_id: str, timestamp: datetime):
        """Add episode to temporal index"""
        # Initialize user timeline if needed
        if user_id not in self.user_timelines:
            self.user_timelines[user_id] = []
        
        # Create node
        node = TemporalNode(timestamp=timestamp, episode_id=episode_id)
        
        # Insert in sorted order (binary search for insertion point)
        timeline = self.user_timelines[user_id]
        insert_pos = bisect.bisect_left([n.timestamp for n in timeline], timestamp)
        timeline.insert(insert_pos, node)
        
        # Update lookup
        self.episode_to_time[episode_id] = (user_id, timestamp)
        self.total_episodes += 1
    
    def query_multi_range(
        self,
        user_id: str,
        time_ranges: List[TimeInterval],
        limit: Optional[int] = None
    ) -> Set[str]:
        """
        Query episodes within multiple time ranges
        
        Args:
            user_id: User ID
            time_ranges: List of time intervals
            limit: Maximum number of results (most recent first)
        
        Returns:
            Set of episode IDs
        """
        if user_id not in self.user_timelines:
            return set()
        
        timeline = self.user_timelines[user_id]
        if not timeline:
            return set()
        
        # Merge overlapping intervals for efficiency
        merged_ranges = self._merge_intervals(time_ranges)
        
        results = []
        
        # For each interval, find matching episodes
        for interval in merged_ranges:
            # Binary search for start position
            start_idx = bisect.bisect_left(
                [n.timestamp for n in timeline],
                interval.start
            )
            
            # Binary search for end position
            end_idx = bisect.bisect_right(
                [n.timestamp for n in timeline],
                interval.end
            )
            
            # Add all episodes in this range
            for node in timeline[start_idx:end_idx]:
                results.append((node.timestamp, node.episode_id))
        
        # Sort by timestamp (most recent first) and apply limit
        results.sort(reverse=True)
        
        if limit:
            results = results[:limit]
        
        return {episode_id for _, episode_id in results}
    
    def query_recent(
        self,
        user_id: str,
        limit: int = 10,
        before: Optional[datetime] = None
    ) -> Set[str]:
        """
        Query most recent episodes
        
        Args:
            user_id: User ID
            limit: Number of episodes to return
            before: Only return episodes before this time
        
        Returns:
            Set of episode IDs
        """
        if user_id not in self.user_timelines:
            return set()
        
        timeline = self.user_timelines[user_id]
        if not timeline:
            return set()
        
        # Find the cutoff index
        if before:
            # Binary search for the position just before the 'before' timestamp
            cutoff_idx = bisect.bisect_left(
                [n.timestamp for n in timeline],
                before
            )
            # Get episodes before this index
            candidates = timeline[:cutoff_idx]
        else:
            candidates = timeline
        
        # Get the last 'limit' episodes
        recent_nodes = candidates[-limit:] if len(candidates) > limit else candidates
        
        return {node.episode_id for node in recent_nodes}
    
    def remove_episode(self, episode_id: str, user_id: str):
        """Remove episode from temporal index"""
        if episode_id not in self.episode_to_time:
            return
        
        # Remove from timeline
        timeline = self.user_timelines[user_id]
        self.user_timelines[user_id] = [
            node for node in timeline
            if node.episode_id != episode_id
        ]
        
        # Remove from lookup
        del self.episode_to_time[episode_id]
        self.total_episodes -= 1
    
    def _merge_intervals(self, intervals: List[TimeInterval]) -> List[TimeInterval]:
        """Merge overlapping time intervals"""
        if not intervals:
            return []
        
        # Sort by start time
        sorted_intervals = sorted(intervals, key=lambda x: x.start)
        
        merged = [sorted_intervals[0]]
        
        for current in sorted_intervals[1:]:
            last = merged[-1]
            
            if current.start <= last.end:
                # Overlapping - merge
                merged[-1] = TimeInterval(
                    start=last.start,
                    end=max(last.end, current.end)
                )
            else:
                # No overlap - add as new interval
                merged.append(current)
        
        return merged
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        total_users = len(self.user_timelines)
        avg_episodes = self.total_episodes / total_users if total_users > 0 else 0
        
        return {
            'total_episodes': self.total_episodes,
            'total_users': total_users,
            'avg_episodes_per_user': avg_episodes,
        }
    
    # ========== 序列化支持 ==========
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "user_timelines": {
                user_id: [
                    {
                        "timestamp": node.timestamp.isoformat(),
                        "episode_id": node.episode_id
                    }
                    for node in timeline
                ]
                for user_id, timeline in self.user_timelines.items()
            },
            "episode_to_time": {
                ep_id: (user_id, ts.isoformat())
                for ep_id, (user_id, ts) in self.episode_to_time.items()
            },
            "total_episodes": self.total_episodes
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典反序列化"""
        # 恢复 user_timelines
        self.user_timelines = {}
        for user_id, timeline_data in data["user_timelines"].items():
            self.user_timelines[user_id] = [
                TemporalNode(
                    timestamp=datetime.fromisoformat(node["timestamp"]),
                    episode_id=node["episode_id"]
                )
                for node in timeline_data
            ]
        
        # 恢复 episode_to_time
        self.episode_to_time = {
            ep_id: (user_id, datetime.fromisoformat(ts))
            for ep_id, (user_id, ts) in data["episode_to_time"].items()
        }
        
        self.total_episodes = data["total_episodes"]
    
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