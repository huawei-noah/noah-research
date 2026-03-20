# src/memory/tag_generator.py
"""
Tag Generator - 使用LLM为对话生成语义标签
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import re

from ..llm.client import LLMClient
from .tag_hierarchy import infer_tag_relations

logger = logging.getLogger(__name__)


class TagGenerator:
    """
    对话标签生成器
    
    使用LLM分析对话内容，提取关键主题作为标签，并识别标签之间的层级关系
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化标签生成器
        
        Args:
            llm_client: LLM客户端（如果为None，会自动创建）
        """
        self.llm_client = llm_client or LLMClient()
        
        # 系统提示词
        self.system_prompt = """You are a semantic tag extraction assistant.
Your task is to analyze conversations and extract 3-8 meaningful tags that capture the main topics, themes, and contexts.

Guidelines:
1. Tags should be lowercase, single words or short phrases (max 3 words)
2. Focus on: topics, activities, locations, entities, emotions, intents
3. Prioritize specific over generic (e.g., "python_programming" over "technology")
4. Use underscores for multi-word tags (e.g., "machine_learning")
5. Avoid overly broad tags like "conversation" or "chat"
6. Return ONLY a JSON object with a "tags" array

Example output:
{
  "tags": ["travel", "paris", "vacation_planning", "hotel_booking", "food"]
}"""
        
        # 新系统提示词（用于生成 tags + relations）
        self.system_prompt_with_relations = """You are a semantic tag extraction assistant.
Your task is to:
1. Extract 3-8 meaningful tags that capture the main topics, themes, and contexts
2. Identify hierarchical relationships between these tags (parent-child)

Guidelines for tags:
- Tags should be lowercase, single words or short phrases (max 3 words)
- Focus on: topics, activities, locations, entities, emotions, intents
- Prioritize specific over generic (e.g., "python_programming" over "technology")
- Use underscores for multi-word tags (e.g., "machine_learning")
- Avoid overly broad tags like "conversation" or "chat"

Guidelines for relations:
- parent tag = broader/more abstract concept
- child tag = more specific concept
- Only include relations that are clear from the conversation
- Examples:
  * parent: "work", child: "programming"
  * parent: "lgbtq", child: "transgender_story"
  * parent: "food", child: "italian_cuisine"
  * parent: "identity", child: "self_acceptance"

Return ONLY a JSON object:
{
  "tags": ["tag1", "tag2", "tag3", ...],
  "relations": [
    {"parent": "broader_tag", "child": "specific_tag"},
    ...
  ]
}

If no clear hierarchical relations exist, return an empty "relations" array."""
    
    def generate_tags(
        self,
        conversation: Dict[str, Any],
        max_tags: int = 8,
        min_tags: int = 3
    ) -> List[str]:
        """
        为对话生成标签（向后兼容的方法）
        
        Args:
            conversation: 对话数据（LoCoMo格式）
            max_tags: 最大标签数量
            min_tags: 最小标签数量
            
        Returns:
            标签列表
        """
        # 调用新方法，只返回 tags
        tags, _ = self.generate_tags_with_relations(conversation, max_tags, min_tags)
        return tags
    
    def generate_tags_with_relations(
        self,
        conversation: Dict[str, Any],
        max_tags: int = 8,
        min_tags: int = 3
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        为对话生成标签和标签关系
        
        Args:
            conversation: 对话数据（LoCoMo格式）
            max_tags: 最大标签数量
            min_tags: 最小标签数量
            
        Returns:
            (tags, relations) 元组
            - tags: 标签列表
            - relations: [(parent, child), ...] 关系列表
        """
        try:
            # 1. 构建对话摘要
            conversation_summary = self._build_conversation_summary(conversation)
            
            # 2. 构建提示词
            user_prompt = f"""Analyze this conversation and extract {min_tags}-{max_tags} semantic tags with their hierarchical relations:

Conversation:
{conversation_summary}

Return tags and relations as a JSON object."""
            
            # 3. 调用LLM
            response = self.llm_client.generate_json_response(
                prompt=user_prompt,
                system_prompt=self.system_prompt_with_relations,
                temperature=0.3,
                max_tokens=300,  # 增加 token 限制以容纳 relations
                default_response={"tags": [], "relations": []},
                max_retries=3
            )
            
            # 4. 提取标签和关系
            tags = response.get("tags", [])
            relations_data = response.get("relations", [])
            
            # 5. 清理标签
            cleaned_tags = self._clean_tags(tags, max_tags)
            
            # 6. 清理关系
            cleaned_relations = self._clean_relations(relations_data, cleaned_tags)
            
            # 7. 如果没有 tags，回退到关键词提取
            if not cleaned_tags:
                logger.warning("LLM tag generation failed, falling back to keyword extraction")
                cleaned_tags = self._extract_keywords_fallback(conversation)
                # 使用预定义层级推断关系
                cleaned_relations = infer_tag_relations(cleaned_tags)
            
            # 8. 如果 LLM 没有返回 relations，使用预定义层级推断
            elif not cleaned_relations:
                logger.info("LLM did not return relations, inferring from hierarchy")
                cleaned_relations = infer_tag_relations(cleaned_tags)
            
            logger.info(f"Generated {len(cleaned_tags)} tags: {cleaned_tags}")
            logger.info(f"Generated {len(cleaned_relations)} relations: {cleaned_relations}")
            
            return cleaned_tags, cleaned_relations
            
        except Exception as e:
            logger.error(f"Error generating tags with relations: {e}")
            # 完全失败时返回基本标签 + 推断的关系
            fallback_tags = self._extract_keywords_fallback(conversation)
            fallback_relations = infer_tag_relations(fallback_tags)
            return fallback_tags, fallback_relations
    
    def _clean_relations(
        self,
        relations_data: List[Dict[str, str]],
        valid_tags: List[str]
    ) -> List[Tuple[str, str]]:
        """
        清理和验证标签关系
        
        Args:
            relations_data: LLM 返回的关系数据 [{"parent": "x", "child": "y"}, ...]
            valid_tags: 有效的标签列表
            
        Returns:
            清理后的关系列表 [(parent, child), ...]
        """
        cleaned = []
        valid_tag_set = set(valid_tags)
        seen = set()  # 去重
        
        for relation in relations_data:
            if not isinstance(relation, dict):
                continue
            
            parent = relation.get("parent", "").lower().strip().replace(" ", "_")
            child = relation.get("child", "").lower().strip().replace(" ", "_")
            
            # 验证：parent 和 child 都必须在 valid_tags 中
            if not (parent and child):
                continue
            
            if parent not in valid_tag_set or child not in valid_tag_set:
                continue
            
            # 验证：parent 不能等于 child
            if parent == child:
                continue
            
            # 去重
            relation_tuple = (parent, child)
            if relation_tuple in seen:
                continue
            
            seen.add(relation_tuple)
            cleaned.append(relation_tuple)
        
        return cleaned
    
    def _build_conversation_summary(
        self,
        conversation: Dict[str, Any],
        max_messages: int = 20,
        max_chars_per_msg: int = 150
    ) -> str:
        """
        构建对话摘要（避免token超限）
        
        Args:
            conversation: 对话数据
            max_messages: 最大消息数量
            max_chars_per_msg: 每条消息最大字符数
            
        Returns:
            对话摘要文本
        """
        parts = []
        
        # 添加speaker信息
        speaker_a = conversation.get("speaker_a", "User")
        speaker_b = conversation.get("speaker_b", "Assistant")
        parts.append(f"Speakers: {speaker_a} and {speaker_b}\n")
        
        # 收集所有消息
        all_messages = []
        for key, chats in conversation.items():
            if key in {"speaker_a", "speaker_b"} or key.endswith("_date_time"):
                continue
            
            for chat in chats or []:
                speaker = chat.get("speaker", speaker_a)
                text = chat.get("text", "")
                
                # 添加图片/搜索上下文
                if chat.get("blip_caption"):
                    text += f" [Image: {chat['blip_caption']}]"
                if chat.get("query"):
                    text += f" [Search: {chat['query']}]"
                
                all_messages.append(f"{speaker}: {text[:max_chars_per_msg]}")
        
        # 限制消息数量
        if len(all_messages) > max_messages:
            # 取前一半和后一半
            half = max_messages // 2
            selected = all_messages[:half] + ["..."] + all_messages[-half:]
        else:
            selected = all_messages
        
        parts.append("\n".join(selected))
        
        return "\n".join(parts)
    
    def _clean_tags(self, tags: List[str], max_tags: int) -> List[str]:
        """
        清理和规范化标签
        
        Args:
            tags: 原始标签列表
            max_tags: 最大标签数量
            
        Returns:
            清理后的标签列表
        """
        cleaned = []
        
        for tag in tags:
            if not isinstance(tag, str):
                continue
            
            # 转小写，去空格
            tag = tag.lower().strip()
            
            # 替换空格为下划线
            tag = tag.replace(" ", "_")
            
            # 移除特殊字符（保留字母、数字、下划线）
            tag = "".join(c for c in tag if c.isalnum() or c == "_")
            
            # 跳过过短或过长的标签
            if len(tag) < 2 or len(tag) > 30:
                continue
            
            # 跳过通用词
            generic_words = {
                "conversation", "chat", "talk", "discussion", "dialogue",
                "message", "text", "reply", "response"
            }
            if tag in generic_words:
                continue
            
            # 去重
            if tag not in cleaned:
                cleaned.append(tag)
        
        # 限制数量
        return cleaned[:max_tags]
    
    def _extract_keywords_fallback(
        self,
        conversation: Dict[str, Any]
    ) -> List[str]:
        """
        关键词提取回退方案（当LLM失败时）
        
        使用简单的关键词匹配
        """
        tags = set()
        
        # Speaker tags
        speaker_a = conversation.get("speaker_a", "")
        speaker_b = conversation.get("speaker_b", "")
        
        if speaker_a:
            tags.add(speaker_a.lower().replace(" ", "_"))
        if speaker_b:
            tags.add(speaker_b.lower().replace(" ", "_"))
        
        # 扩充关键词库（基于 LoCoMo 数据集高频词）
        keywords = {
            # 原有通用词
            "travel", "vacation", "trip", "holiday", "flight", "hotel",
            "food", "cooking", "recipe", "dinner", "lunch", "breakfast",
            "work", "job", "project", "meeting", "colleague",
            "family", "friend", "parent", "child", "brother", "sister",
            "movie", "music", "book", "game", "sport",
            "health", "exercise", "fitness", "doctor",
            "shopping", "buy", "purchase", "store",
            "restaurant", "coffee", "bar",
            "birthday", "party", "celebration",
            "weather", "rain", "snow", "sunny",
            "car", "drive", "traffic",
            "school", "study", "learn", "course",
            
            # LGBTQ+
            "lgbtq", "transgender", "pride", "support_group", "coming_out",
            "identity", "self_acceptance", "transition", "queer",
            "advocacy", "activism", "community", "ally",
            
            # 艺术/创作类
            "painting", "art", "pottery", "creative", "craft", "draw",
            "music", "instrument", "violin", "piano", "clarinet",
            
            # 教育/职业类
            "counseling", "therapy", "mental_health", "career",
            "education", "university", "certification",
            
            # 家庭/宠物类
            "adoption", "parenting", "children", "kids",
            "pet", "dog", "cat", "animal",
            
            # 物品类
            "figurine", "collectible", "souvenir", "gift", "treasure",
            "book", "reading", "library", "literature",
            
            # 活动类
            "camping", "hiking", "running", "swimming", "outdoor",
            "concert", "festival", "event", "workshop", "conference",
            "museum", "exhibition", "gallery",
            
            # 地点类
            "beach", "mountain", "park", "city", "neighborhood",
            "church", "cafe", "center"
        }
        
        # 遍历对话内容
        for key, chats in conversation.items():
            if key in {"speaker_a", "speaker_b"} or key.endswith("_date_time"):
                continue
            
            for chat in chats or []:
                text = chat.get("text", "").lower()
                
                # 使用词边界匹配（避免误匹配）
                for keyword in keywords:
                    # 使用正则确保完整单词匹配
                    import re
                    if re.search(rf'\b{keyword}\b', text):
                        tags.add(keyword)
        
        # 限制数量
        return sorted(list(tags))[:8] 


    def batch_generate_tags(
        self,
        conversations: List[Dict[str, Any]],
        max_tags: int = 8,
        min_tags: int = 3
    ) -> List[List[str]]:
        """
        批量生成标签（向后兼容）
        
        Args:
            conversations: 对话列表
            max_tags: 最大标签数量
            min_tags: 最小标签数量
            
        Returns:
            标签列表的列表
        """
        all_tags = []
        
        for i, conversation in enumerate(conversations):
            logger.info(f"Generating tags for conversation {i+1}/{len(conversations)}")
            tags = self.generate_tags(conversation, max_tags, min_tags)
            all_tags.append(tags)
        
        return all_tags
    
    def batch_generate_tags_with_relations(
        self,
        conversations: List[Dict[str, Any]],
        max_tags: int = 8,
        min_tags: int = 3
    ) -> List[Tuple[List[str], List[Tuple[str, str]]]]:
        """
        批量生成标签和关系
        
        Args:
            conversations: 对话列表
            max_tags: 最大标签数量
            min_tags: 最小标签数量
            
        Returns:
            [(tags, relations), ...] 列表
        """
        all_results = []
        
        for i, conversation in enumerate(conversations):
            logger.info(f"Generating tags and relations for conversation {i+1}/{len(conversations)}")
            tags, relations = self.generate_tags_with_relations(conversation, max_tags, min_tags)
            all_results.append((tags, relations))
        
        return all_results


__all__ = ['TagGenerator']