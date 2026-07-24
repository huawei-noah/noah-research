SCHEMA_SELECTION_FOR_GENERATION_PROMPT = """You are a memory extraction schema expert. Given a dialogue, select which entity types and their dynamic properties are relevant for extracting structured memories.

Dialogue:
{dialogue_text}

Speaker note: Lines may be formatted as `speaker=Name: ...` for named-speaker dialogue. Treat `Name` as the real speaker of that line; first-person statements in that line belong to `Name`, not automatically to the user.

Available Entity Types and Properties:
{entity_schema}

Select the entity types and properties that are relevant to the information in this dialogue.

Output Format (JSON):
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
    "reasoning": "Brief explanation of why these types and properties were selected"
}}

Rules:
1. ALWAYS include "episodes" entity type (it will be added automatically, no need to list it)
2. "default_property" is ALWAYS included for every selected entity type (no need to list it)
3. When unsure whether a property is relevant, INCLUDE it — false negatives are worse than false positives
4. Use ["all"] to keep all properties of an entity type when most properties could be relevant
5. Only EXCLUDE properties that are clearly irrelevant to the dialogue content
6. ALWAYS include "person" entity type if any person is mentioned or implied in the dialogue
7. Focus on what information the dialogue CONTAINS, not what it might theoretically relate to
"""
