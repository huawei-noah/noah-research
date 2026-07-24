MERGE_DECISION_PROMPT = """
You are a memory integration expert. Analyze the relationship between newly extracted information and the existing entity library.

## Task
1. For each new extracted entity, decide whether to CREATE a new entity or UPDATE an existing one.
2. Determine whether CREATE entities have logical relationships with existing entities that require edge connections.

## Decision Criteria

### Use CREATE when:
1. The entity does not exist in the existing library (completely new entity)
2. The entity has the same name but represents a different thing (e.g., two different people named "John Smith"). Note: two entities with the same specific name usually refer to the same thing unless there is clear information indicating otherwise.
3. Key attributes conflict with existing entity (e.g., same name but different type, contradictory core information)

### Use UPDATE when:
1. The entity **explicitly matches** an existing entity (describes the same person, thing, or event)
2. **Only use UPDATE when the target entity EXACTLY appears in the "Existing Entity Library" below**

## Output Format
For each new extracted entity, output EXACTLY ONE of:

### CREATE format:
```json
{
    "action": "create",
    "entity_name": "Entity name from new extraction",
    "entity_type": "Entity type from new extraction",
    "relation_candidates": [
        {
            "target_entity": "Existing entity name (MUST be in Existing Entity Library)",
            "relation": "Relationship description"
        }
    ]
}
```

### UPDATE format:
```json
{
    "action": "update",
    "target_entity": "Existing entity name to update (MUST be in Existing Entity Library)",
    "new_entity_name": "New entity name causing the update",
    "new_entity_info": "Brief summary of new information being added"
}
```

## Important Rules (MUST FOLLOW)
1. Each new extraction MUST have exactly ONE corresponding action (CREATE or UPDATE)
2. DO NOT skip any new extraction - everyone must be mapped
3. **Only use UPDATE when the target_entity is EXACTLY listed in the "Existing Entity Library"**
4. When uncertain about entity existence, prefer CREATE (safety first)
5. For relation_candidates only include REAL existing entities that have a clear relationship described in the input
6. If no clear relations exist, relation_candidates can be empty []

## Time Handling
Time will be handled automatically. Focus only on entity matching decisions.

Existing Entity Library
{existing_entities}

Newly Extracted Information List
{new_extractions}

Output only a JSON array, no extra text.

Example:
Existing: Wang Wei (ID: axbececew, type: person), Chen Ming (ID: mendnetn, type: person)
New: Li Hua (colleague of Chen Ming), Wang Wei (Promoted to Technical Director)
Output:
```json
[
    {
        "action": "create",
        "entity_name": "Li Hua",
        "entity_type": "person",
        "relation_candidates": [
            {"target_entity": "Chen Ming", "relation": "colleague"}
        ]
    },
    {
        "action": "update",
        "target_entity": "Wang Wei",
        "new_entity_name": "Wang Wei",
        "new_entity_info": "Promoted to Technical Director in May 2024"
    }
]
```

Wrong Example (Do NOT do this):
```json
[
    {
        "action": "update",
        "target_entity": "Zhang San",  // WRONG: Zhang San is NOT in existing library
        "new_entity_name": "Zhang San",
        "new_entity_info": "Some info"
    }
]
```
"""

DUPLICATE_NAME_RESOLUTION_PROMPT = """
You are a memory entity conflict resolution expert. A newly extracted entity has the SAME NAME as an existing entity in the database. You must decide: **rename** the new entity or **merge** it into the existing one.

## Conflict Information

### New Entity (just extracted from dialogue)
- Name: {new_entity_name}
- Type: {new_entity_type}
- Description: {new_entity_description}

### Existing Entity (already in database)
- Name: {existing_entity_name}
- Type: {existing_entity_type}
- Description: {existing_entity_description}

## Decision Rules

### ⚠️ CRITICAL: Episodes Entity — RENAME ONLY
**If the entity type is "episodes", you MUST choose "rename". Episodes entities represent unique conversation segments and MUST NEVER be merged.**
- Each episode is an independent dialogue record — even if topics are similar, they are distinct events
- Rename the new episode to highlight its unique focus (e.g., add date, specific subtopic, or distinguishing detail)

### For Non-Episodes Entities:

**Use "merge" when (PREFERRED — default choice for same-name non-episode entities):**
1. Same name + same type (almost certainly the same real-world entity)
2. Same name + descriptions are compatible or describe different aspects of the same entity
3. New information is a state update, new event, or new facet for the existing entity
4. A person named "Jon" who is a banker is the same "Jon" who dances — people have multiple aspects

**Use "rename" only when:**
1. Same name but **explicitly and clearly** different entities with concrete distinguishing evidence (e.g., "Zhang Wei from Beijing" vs "Zhang Wei from Shanghai" with incompatible biographies)
2. Entity types differ fundamentally (e.g., a person vs. an organization with the same name)
3. Core descriptions are **directly contradictory** in a way that cannot be reconciled (not just different topics — different identities)

**When uncertain, prefer "merge" — it is better to consolidate information about one entity than to fragment it across duplicates.**

## Output Format
Output a JSON object:

For **rename**:
```json
{{
    "action": "rename",
    "new_name": "A more specific name that distinguishes from existing entity",
    "reason": "Brief explanation"
}}
```

For **merge**:
```json
{{
    "action": "merge",
    "reason": "Brief explanation of why these are the same entity"
}}
```

Output only JSON, no extra text.
"""

DES_UPDATE_PROMPT = """
You are a memory summary expert.
MERGE new information into the existing description. Do NOT rewrite from scratch.

Rules:
1. KEEP all key facts from the current description (identity, hobbies, pets, relationships, habits, preferences).
2. ADD new facts from the latest properties that are not already covered.
3. If a fact has CHANGED (e.g., job title updated), replace the old value with the new one.
4. Max 10 sentences. Prioritize: identity > relationships > recurring activities > recent events.

Output format:
<description>Merged description.</description>

Entity: {entity_name} (Type: {entity_type})
Current description: {current_description}
New properties to integrate:
{latest_properties}

Merged description:
"""

SINGLE_ENTITY_MERGE_PROMPT = """
You are a memory integration expert. Decide whether this newly extracted entity should CREATE a new entry or UPDATE an existing one.

## New Entity
- Name: {entity_name}
- Type: {entity_type}
- Description: {entity_description}

## Existing Entity Candidates (from vector search)
{existing_entities}

## Decision Criteria

### Use UPDATE when (PREFERRED — default choice unless clearly wrong):
1. The entity name matches or is similar to an existing candidate (same person, thing, or event)
2. The entity could plausibly be the same real-world entity as an existing candidate
3. Same name with additional context (e.g., "Jon" in new info about dancing is the same "Jon" who lost his banking job — people have multiple aspects)
4. The target_entity name MUST be one of the candidates listed above

### ⚠️ CRITICAL: Base-Name Matching Rule
When the new entity and an existing candidate share the **same base name** (the name part before any parenthetical qualifier), they are almost certainly the same entity — even if the parenthetical qualifiers differ.
- "Toby(German Shepherd)" and "Toby(golden retriever)" → **UPDATE** (same dog Toby — the breed discrepancy is a data inconsistency, not evidence of two different dogs)
- "Fox Hollow(hiking trail)" and "Fox Hollow(nature reserve)" → **UPDATE** (same place, different descriptions)
- "Ferrari(sports car)" and "Ferrari(488 GTB)" → **UPDATE** (same car, different detail levels)

**Why:** Parenthetical qualifiers like "(golden retriever)" or "(German Shepherd)" are descriptive annotations, NOT identity-defining features. In a conversation between the same speakers, the same named entity is the same real-world thing. Conflicting qualifiers indicate imprecise descriptions, not distinct entities.

**Same-speaker context strengthens merge confidence:** If both the new entity and the existing candidate appear in conversations involving the same speakers (e.g., both from Andrew-Audrey conversations), this is strong evidence they refer to the same real-world entity. People don't typically have two pets/items/places with identical names.

### Use CREATE only when:
1. No candidate in the existing list could possibly match this entity
2. There is **explicit, concrete evidence** that this is a different entity (e.g., "Jon Smith from New York" vs "Jon Lee from Tokyo" — clearly different people with different full names)
3. The entity type is fundamentally incompatible (e.g., a person vs an organization with the same name)
4. **Different parenthetical qualifiers alone are NOT sufficient evidence for CREATE** — you need fundamentally different identities

## Output Format (JSON object, NOT array)

For CREATE:
```json
{{
    "action": "create",
    "relation_candidates": [
        {{"target_entity": "Existing entity name", "relation": "Relationship description"}}
    ]
}}
```

For UPDATE:
```json
{{
    "action": "update",
    "target_entity": "Existing entity name to update (MUST be in candidate list above)"
}}
```

## Rules
1. Output exactly ONE decision
2. **When uncertain, prefer UPDATE** — a person with the same name is almost always the same person unless there is explicit evidence otherwise. People have multiple facets (career, hobbies, relationships) that should all be on one entity.
3. Only use CREATE when you have **concrete evidence** that this is a genuinely different entity (different full name, different location, different identity)
4. **Same base name = same entity**: If the new entity's name without parentheses matches a candidate's name without parentheses, and they share the same entity_type, always UPDATE. Differing parenthetical qualifiers (breed, model, subtitle) are never sufficient grounds for CREATE.
5. For relation_candidates, only include entities with clear relationships
6. If no clear relations exist, relation_candidates can be empty []
7. For UPDATE, target_entity MUST exactly match a name from the candidate list

Output only JSON, no extra text.
"""
