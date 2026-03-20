"""
Episode Segmentor - 智能对话分段和Episode生成
将粗粒度的Session拆分为细粒度的Episodes
"""

from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging

from ..models.message import Message
from ..models.episode import Episode
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


# ========== Prompt Templates ==========

class SegmentationPrompts:
    """分段相关的Prompt模板（内部类）"""
    
    TOPIC_SPLIT_PROMPT = """You are analyzing a conversation to identify natural topic boundaries.

## Input Messages
You will receive {count} messages numbered from 1 to {count}:

{messages}

## Your Task
Identify topic boundaries and group messages into coherent episodes. Create NEW episodes when:

1. **Topic Shift** - New subject introduced
2. **Intent Change** - Purpose of conversation changes  
3. **Time Gap** - More than 30 minutes between messages
4. **Explicit Transitions** - Phrases like "by the way", "changing topics", etc.

## Output Format
Return JSON with episode groups:
{{
    "episodes": [
        {{
            "indices": [1, 2, 3, 4],
            "topic": "Brief description of this episode's topic"
        }},
        {{
            "indices": [5, 6, 7],
            "topic": "Brief description of this episode's topic"
        }}
    ]
}}

Focus on topic coherence. Return ONLY the JSON object."""

    EPISODE_SUMMARY_PROMPT = """You are generating a structured summary for an episode.

## Messages in this Episode:
{messages}

## Task
Generate a structured summary with:
1. **title**: Concise, descriptive title (10-20 words)
2. **content**: Third-person narrative including who, when, what, decisions, emotions

## Output Format
Return ONLY a JSON object:
{{
    "title": "Descriptive title here",
    "content": "Detailed narrative here...",
}}

Return ONLY the JSON object, no additional text."""


# ========== Main Segmentor Class ==========

class EpisodeSegmentor:
    """
    Episode分段器 - 智能拆分Session为多个Episodes
    """
    
    def __init__(
        self, 
        llm_client: LLMClient,
        split_threshold: int = 15,
        temperature: float = 0.2
    ):
        """
        初始化分段器
        
        Args:
            llm_client: LLM客户端
            split_threshold: 触发拆分的消息数阈值
            temperature: LLM生成的temperature参数
        """
        self.llm_client = llm_client
        self.split_threshold = split_threshold
        self.temperature = temperature
        self.prompts = SegmentationPrompts()
        
        logger.info(
            f"EpisodeSegmentor initialized: "
            f"threshold={split_threshold}, temperature={temperature}"
        )
    
    # ========== Public Methods ==========
    
    def should_split_session(self, message_count: int) -> Tuple[bool, str]:
        """
        判断Session是否需要拆分
        
        Args:
            message_count: 消息数量
            
        Returns:
            (是否需要拆分, 原因说明)
        """
        if message_count >= self.split_threshold:
            return True, f"Message count ({message_count}) exceeds threshold ({self.split_threshold})"
        return False, "Message count below threshold, no split needed"
    
    def split_session_to_episodes(
        self,
        user_id: str,
        messages: List[Message],
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Episode]:
        """
        将一个Session的消息拆分为多个Episodes
        
        Args:
            user_id: 用户ID
            messages: 消息列表
            session_metadata: Session元数据（可选）
            
        Returns:
            Episode对象列表
        """
        if not messages:
            logger.warning("Empty message list, returning empty episodes")
            return []
        
        # Step 1: 检测话题边界
        message_groups = self._detect_topic_boundaries(messages)
        
        logger.info(
            f"Split {len(messages)} messages into {len(message_groups)} episodes"
        )
        
        # Step 2: 为每个消息组生成Episode
        episodes = []
        for idx, group_indices in enumerate(message_groups):
            try:
                episode = self._build_episode_from_group(
                    user_id=user_id,
                    messages=messages,
                    group_indices=group_indices,
                    group_idx=idx,
                    total_groups=len(message_groups),
                    session_metadata=session_metadata
                )
                episodes.append(episode)
                
            except Exception as e:
                logger.error(f"Failed to build episode for group {idx}: {e}")
                # 创建fallback episode
                fallback = self._create_fallback_episode(
                    user_id, messages, group_indices, idx, session_metadata
                )
                episodes.append(fallback)
        
        return episodes
    
    # ========== Private Methods ==========
    
    def _detect_topic_boundaries(self, messages: List[Message]) -> List[List[int]]:
        """
        使用LLM检测话题边界
        
        Args:
            messages: 消息列表
            
        Returns:
            消息分组，例如 [[1,2,3], [4,5,6]]（1-based索引）
        """
        # 格式化消息
        formatted_messages = self._format_messages_for_prompt(messages)
        
        # 构建prompt
        prompt = self.prompts.TOPIC_SPLIT_PROMPT.format(
            count=len(messages),
            messages=formatted_messages
        )
        
        try:
            # 调用LLM
            response = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=4096,
                default_response={"episodes": []}
            )
            
            # 提取分组
            episodes_data = response.get("episodes", [])
            
            if not episodes_data:
                logger.warning("LLM returned no episodes, using fallback")
                return [[i + 1 for i in range(len(messages))]]
            
            groups = [ep["indices"] for ep in episodes_data if ep.get("indices")]
            
            # 验证分组
            if not self._validate_groups(groups, len(messages)):
                logger.warning("Invalid groups from LLM, using fallback")
                return [[i + 1 for i in range(len(messages))]]
            
            return groups
            
        except Exception as e:
            logger.error(f"Topic boundary detection failed: {e}")
            # Fallback: 所有消息作为一个episode
            return [[i + 1 for i in range(len(messages))]]
    
    def _build_episode_from_group(
        self,
        user_id: str,
        messages: List[Message],
        group_indices: List[int],
        group_idx: int,
        total_groups: int,
        session_metadata: Optional[Dict[str, Any]]
    ) -> Episode:
        """
        从消息组构建Episode
        
        Args:
            user_id: 用户ID
            messages: 完整消息列表
            group_indices: 当前组的消息索引（1-based）
            group_idx: 组索引
            total_groups: 总组数
            session_metadata: Session元数据
        """
        # 提取该组的消息（转换为0-based索引）
        group_messages = [messages[i - 1] for i in group_indices]
        
        # 使用LLM生成title和content
        episode_data = self._generate_episode_summary(group_messages)
        
        # 确定时间戳
        timestamp = self._extract_timestamp(episode_data, group_messages)
        
        # 构建metadata
        metadata = {
            "split_group_idx": group_idx,
            "split_total_groups": total_groups,
            "message_indices": group_indices,
            "original_message_count": len(group_messages)
        }
        
        if session_metadata:
            metadata.update({
                f"session_{k}": v for k, v in session_metadata.items()
            })
        
        # 创建Episode
        episode = Episode(
            user_id=user_id,
            messages=group_messages,
            title=episode_data.get("title", f"Episode {group_idx + 1}"),
            content=episode_data.get("content", "No content generated"),
            boundary_reason=f"Topic split: segment {group_idx + 1}/{total_groups}",
            timestamp=timestamp,
            metadata=metadata
        )
        
        return episode
    
    def _generate_episode_summary(self, messages: List[Message]) -> Dict[str, Any]:
        """
        使用LLM生成Episode摘要
        
        Args:
            messages: 消息列表
            
        Returns:
            包含title, content, timestamp的字典
        """
        # 格式化消息
        formatted_messages = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in messages
        ])
        
        # 构建prompt
        prompt = self.prompts.EPISODE_SUMMARY_PROMPT.format(
            messages=formatted_messages
        )
        
        try:
            response = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2048,
                default_response={
                    "title": "Episode Summary",
                    "content": "Summary generation failed"
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Episode summary generation failed: {e}")
            return {
                "title": "Episode Summary (Fallback)",
                "content": " ".join([msg.content[:100] for msg in messages[:3]])
            }
    
    def _format_messages_for_prompt(self, messages: List[Message]) -> str:
        """格式化消息用于prompt"""
        lines = []
        for i, msg in enumerate(messages, 1):
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            content = msg.content[:200]  # 限制长度
            lines.append(f"{i}. [{timestamp}] {msg.role}: {content}")
        
        return "\n".join(lines)
    
    def _validate_groups(self, groups: List[List[int]], total_messages: int) -> bool:
        """验证分组是否有效"""
        if not groups:
            return False
        
        all_indices = set()
        for group in groups:
            if not group:
                return False
            all_indices.update(group)
        
        # 检查是否覆盖所有消息
        expected = set(range(1, total_messages + 1))
        return all_indices == expected
    
    def _extract_timestamp(
        self, 
        episode_data: Dict[str, Any], 
        messages: List[Message]
    ) -> datetime:
        """提取Episode时间戳（直接使用消息的session时间戳）"""
        # 所有消息共享同一个 session 时间戳，直接取第一条即可
        if messages:
            return messages[0].timestamp 
        
        # Fallback: 如果消息列表为空（不应该发生）
        logger.warning("No messages provided for timestamp extraction")
        raise ValueError("Cannot extract timestamp from empty message list")

    def _create_fallback_episode(
        self,
        user_id: str,
        messages: List[Message],
        group_indices: List[int],
        group_idx: int,
        session_metadata: Optional[Dict[str, Any]]
    ) -> Episode:
        """创建fallback Episode（当LLM生成失败时）"""
        group_messages = [messages[i - 1] for i in group_indices]
        
        return Episode(
            user_id=user_id,
            messages=group_messages,
            title=f"Episode {group_idx + 1} (Fallback)",
            content=" ".join([msg.content[:50] for msg in group_messages[:3]]),
            boundary_reason=f"Fallback episode creation",
            timestamp=group_messages[0].timestamp,
            metadata={
                "fallback": True,
                "split_group_idx": group_idx,
                "message_indices": group_indices,
                **(session_metadata or {})
            }
        )