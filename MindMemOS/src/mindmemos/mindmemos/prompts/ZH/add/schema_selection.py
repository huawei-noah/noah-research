SCHEMA_SELECTION_FOR_GENERATION_PROMPT = """你是一个记忆提取schema专家。根据给定的对话内容，选择与提取结构化记忆相关的实体类型及其动态属性。

对话内容：
{dialogue_text}

说话人说明：对话行可能使用 `speaker=Name: ...` 表示具名说话人。请将 `Name` 视为该行真实说话人；该行中的第一人称陈述属于 `Name`，不要自动归因到用户。

可用的实体类型和属性：
{entity_schema}

选择与对话信息相关的实体类型和属性。

输出格式（JSON）：
{{
    "selected_entities": [
        {{
            "entity_type": "person",
            "relevant_properties": ["position_event", "hobby_activity", "plan_event"]
        }},
        {{
            "entity_type": "animal",
            "relevant_properties": ["all"]
        }}
    ],
    "reasoning": "简要说明选择这些类型和属性的原因"
}}

规则：
1. 始终包含 "episodes" 实体类型（系统会自动添加，无需列出）
2. "default_property" 始终为每个选中的实体类型保留（无需列出）
3. 不确定某个属性是否相关时，请包含它——漏选比多选更糟糕
4. 当某个实体类型的大部分属性都可能相关时，使用 ["all"] 保留所有属性
5. 仅排除与对话内容明显无关的属性
6. 如果对话中提到或暗示了任何人物，始终包含 "person" 实体类型
7. 关注对话实际包含的信息，而非理论上可能关联的内容
"""
