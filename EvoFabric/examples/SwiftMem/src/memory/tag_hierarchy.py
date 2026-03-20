# src/memory/tag_hierarchy.py
"""
Tag Hierarchy - 预定义的标签层级关系
用于在 LLM 失败时推断标签之间的层级关系
"""

from typing import List, Dict, Set, Tuple

# 预定义的标签层级 (parent -> children)
TAG_HIERARCHY: Dict[str, List[str]] = {
    # ========== 工作相关 ==========
    "work": ["programming", "meeting", "project", "deadline", "career", "job", "business"],
    "programming": ["python", "javascript", "debugging", "coding", "development"],
    "career": ["career_planning", "job_search", "promotion", "interview"],
    
    # ========== 食物相关 ==========
    "food": ["cooking", "recipe", "restaurant", "italian_cuisine", "chinese_cuisine", "dining"],
    "cooking": ["baking", "grilling", "meal_prep", "kitchen"],
    "restaurant": ["cafe", "bar", "fine_dining"],
    
    # ========== 旅行相关 ==========
    "travel": ["vacation", "trip", "holiday", "beach_vacation", "city_tour", "flight"],
    "vacation": ["beach_vacation", "ski_trip", "road_trip", "resort"],
    "trip": ["business_trip", "weekend_getaway"],
    
    # ========== 社交相关 ==========
    "social": ["family", "friend", "party", "gathering", "relationship"],
    "family": ["parent", "child", "sibling", "relative"],
    "friend": ["friendship", "hangout", "reunion"],
    "relationship": ["dating", "marriage", "romance"],
    
    # ========== 娱乐相关 ==========
    "entertainment": ["movie", "music", "book", "game", "tv_show"],
    "movie": ["comedy", "drama", "action", "horror", "documentary"],
    "music": ["concert", "album", "song", "band"],
    "game": ["video_game", "board_game", "sports"],
    
    # ========== 健康相关 ==========
    "health": ["exercise", "fitness", "diet", "mental_health", "wellness"],
    "exercise": ["running", "gym", "yoga", "swimming", "cycling"],
    "mental_health": ["therapy", "meditation", "stress", "anxiety"],
    "diet": ["nutrition", "weight_loss", "healthy_eating"],
    
    # ========== LGBTQ+ 相关 ==========
    "lgbtq": ["transgender_story", "lgbtq_support_group", "coming_out", "pride", "queer"],
    "transgender_story": ["transition", "gender_identity", "trans_rights"],
    "lgbtq_support_group": ["community", "activism", "advocacy"],
    
    # ========== 身份认同相关 ==========
    "identity": ["self_acceptance", "authentic_living", "self_discovery", "personal_growth"],
    "self_acceptance": ["self_love", "confidence", "body_positivity"],
    "authentic_living": ["being_yourself", "honesty", "integrity"],
    
    # ========== 教育相关 ==========
    "education": ["school", "study", "learn", "course", "university"],
    "school": ["homework", "exam", "class", "teacher"],
    "study": ["research", "reading", "learning"],
    
    # ========== 购物相关 ==========
    "shopping": ["buy", "purchase", "store", "online_shopping", "retail"],
    "store": ["mall", "boutique", "supermarket"],
    
    # ========== 情感相关 ==========
    "emotion": ["happiness", "sadness", "anger", "fear", "love"],
    "emotional_support": ["comfort", "encouragement", "empathy", "listening"],
    
    # ========== 兴趣爱好 ==========
    "hobby": ["photography", "painting", "writing", "crafts", "gardening"],
    "photography": ["camera", "photo_editing", "portrait"],
    "writing": ["blog", "journal", "creative_writing"],
    
    # ========== 财务相关 ==========
    "finance": ["money", "budget", "investment", "savings", "debt"],
    "investment": ["stocks", "real_estate", "retirement"],
    
    # ========== 住房相关 ==========
    "home": ["apartment", "house", "renovation", "decoration", "moving"],
    "decoration": ["furniture", "interior_design", "diy"],
    
    # ========== 交通相关 ==========
    "transportation": ["car", "bus", "train", "bike", "commute"],
    "car": ["driving", "parking", "traffic", "road_trip"],
    
    # ========== 天气相关 ==========
    "weather": ["rain", "snow", "sunny", "storm", "temperature"],
    
    # ========== 节日相关 ==========
    "celebration": ["birthday", "party", "anniversary", "wedding", "holiday"],
    "holiday": ["christmas", "thanksgiving", "new_year"],
    
    # ========== 宠物相关 ==========
    "pet": ["dog", "cat", "pet_care", "veterinarian"],
    
    # ========== 技术相关 ==========
    "technology": ["computer", "phone", "internet", "software", "hardware"],
    "software": ["app", "program", "website", "ai"],
}


def infer_tag_relations(tags: List[str]) -> List[Tuple[str, str]]:
    """
    根据预定义层级推断标签之间的关系
    
    Args:
        tags: 标签列表
        
    Returns:
        [(parent, child), ...] 关系列表
        
    Example:
        >>> infer_tag_relations(["work", "programming", "python"])
        [("work", "programming"), ("programming", "python")]
    """
    relations = []
    tag_set = set(tags)
    
    # 遍历所有预定义的层级关系
    for parent, children in TAG_HIERARCHY.items():
        if parent in tag_set:
            for child in children:
                if child in tag_set:
                    relations.append((parent, child))
    
    return relations


def get_tag_ancestors(tag: str, max_depth: int = 3) -> Set[str]:
    """
    获取标签的所有祖先（更抽象的标签）
    
    Args:
        tag: 标签
        max_depth: 最大深度
        
    Returns:
        祖先标签集合
        
    Example:
        >>> get_tag_ancestors("python")
        {"programming", "work"}
    """
    ancestors = set()
    
    # 构建反向映射 (child -> parents)
    child_to_parents: Dict[str, Set[str]] = {}
    for parent, children in TAG_HIERARCHY.items():
        for child in children:
            if child not in child_to_parents:
                child_to_parents[child] = set()
            child_to_parents[child].add(parent)
    
    # BFS 查找祖先
    current_level = {tag}
    for _ in range(max_depth):
        next_level = set()
        for current_tag in current_level:
            if current_tag in child_to_parents:
                parents = child_to_parents[current_tag]
                ancestors.update(parents)
                next_level.update(parents)
        
        if not next_level:
            break
        current_level = next_level
    
    return ancestors


def get_tag_descendants(tag: str, max_depth: int = 3) -> Set[str]:
    """
    获取标签的所有后代（更具体的标签）
    
    Args:
        tag: 标签
        max_depth: 最大深度
        
    Returns:
        后代标签集合
        
    Example:
        >>> get_tag_descendants("work")
        {"programming", "python", "meeting", ...}
    """
    descendants = set()
    
    if tag not in TAG_HIERARCHY:
        return descendants
    
    # BFS 查找后代
    current_level = set(TAG_HIERARCHY[tag])
    descendants.update(current_level)
    
    for _ in range(max_depth - 1):
        next_level = set()
        for current_tag in current_level:
            if current_tag in TAG_HIERARCHY:
                children = set(TAG_HIERARCHY[current_tag])
                descendants.update(children)
                next_level.update(children)
        
        if not next_level:
            break
        current_level = next_level
    
    return descendants


def expand_tags_with_hierarchy(
    tags: List[str],
    include_ancestors: bool = True,
    include_descendants: bool = False,
    max_depth: int = 2
) -> Set[str]:
    """
    基于层级扩展标签集合
    
    Args:
        tags: 原始标签列表
        include_ancestors: 是否包含祖先标签
        include_descendants: 是否包含后代标签
        max_depth: 最大深度
        
    Returns:
        扩展后的标签集合
        
    Example:
        >>> expand_tags_with_hierarchy(["python"], include_ancestors=True)
        {"python", "programming", "work"}
    """
    expanded = set(tags)
    
    for tag in tags:
        if include_ancestors:
            expanded.update(get_tag_ancestors(tag, max_depth))
        
        if include_descendants:
            expanded.update(get_tag_descendants(tag, max_depth))
    
    return expanded


__all__ = [
    'TAG_HIERARCHY',
    'infer_tag_relations',
    'get_tag_ancestors',
    'get_tag_descendants',
    'expand_tags_with_hierarchy'
]