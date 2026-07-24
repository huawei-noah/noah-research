SEARCH_FIELD_GENERATION_PROMPT = """
You are a search optimization expert. Generate diverse search retrieval fields for a memory entity so it can be found by various natural language queries.

## Entity Information
- Name: {entity_name}
- Type: {entity_type}
- Description: {entity_description}
- Properties:
{entity_properties}

## Task
Generate up to {max_fields} short search phrases (each 5-20 words) that capture different aspects of this entity. These will be independently embedded for vector similarity search.

## Guidelines
1. **Diversity**: Each field should capture a DIFFERENT aspect or angle. Avoid redundancy.
2. **Query-oriented**: Write phrases as a user would naturally search for this information. Think about what questions or keywords someone might use.
3. **Comprehensive coverage**: Cover identity, activities, relationships, preferences, events, and temporal aspects mentioned in the properties.
4. **Self-contained**: Each phrase should be independently meaningful without needing the others.
5. **Include entity name**: Each field should mention the entity name at least once for disambiguation.
6. **Group and summarize when content is large**: When the entity has too many properties to cover individually within {max_fields} fields, group related information into higher-level summary phrases to maximize coverage. When full coverage is impossible, prioritize fields with broader scope and higher importance (e.g., core identity, key relationships, major events) over minor details.

## Examples
For entity "Caroline (person)" with properties about attending support groups, painting, and having pets:
```json
["Caroline attends LGBTQ support group meetings regularly",
 "Caroline's hobby is painting including a painting of Aragorn",
 "Caroline owns a dog named Toby",
 "Caroline felt inspired by transgender stories at support group",
 "Caroline's emotional well-being and social activities"]
```

## Output Format
Return a JSON array of strings. Each string is one search field.
```json
["field 1", "field 2", ...]
```

Output only the JSON array, no extra text.
"""

SEARCH_FIELD_UPDATE_PROMPT = """
You are a search optimization expert. Update the existing search retrieval fields for a memory entity based on newly added information.

## Entity: {entity_name} ({entity_type})

## Current Search Fields
{current_fields}

## New Information Added
{new_information}

## Task
Return an updated list of up to {max_fields} search phrases. You may:
- **Keep** existing fields that are still relevant
- **Update** existing fields to incorporate new information
- **Replace** less important fields with new ones covering the new information
- **Add** new fields if there is room (< {max_fields} total)

## Rules
1. Total fields must not exceed {max_fields}. If already at {max_fields}, you must replace or merge to add new coverage.
2. Each field should be 5-20 words, mention the entity name, and be independently meaningful.
3. Prioritize fields with broader scope and higher importance (core identity, key relationships, major events).
4. Do not create redundant fields — each should cover a different aspect.
5. Preserve existing fields that still accurately represent the entity. Only change what the new information requires.

## Output Format
Return a JSON array of strings (the complete updated field list).
```json
["field 1", "field 2", ...]
```

Output only the JSON array, no extra text.
"""

EPISODE_SEARCH_FIELD_AUGMENT_PROMPT = """
You are a search optimization expert. An episode (conversation segment) already has search fields derived from structured entity properties. Your task is to discover facts present in the original conversation that the existing property-based fields have NOT covered, and generate additional search phrases for them.

## Original Conversation
{episode_text}

## Existing Search Fields (already covered)
{existing_fields}

## Task
Generate up to {augment_count} short search phrases (each 5-20 words). You may generate fewer if the existing fields already cover the conversation well.

Requirements:
1. Each phrase must capture a fact from the conversation that is NOT already covered by the existing fields
2. Focus on: implicit facts, causal relationships, specific details (names, places, items, dates), emotional context, plans, suggestions, or any retrievable information missed by the property extraction
3. Write as declarative statements, NOT as questions
4. Each phrase must be independently meaningful and self-contained
5. Do NOT repeat or rephrase information already in the existing fields

## Output Format
Return a JSON array of 1 to {augment_count} strings.
```json
["phrase 1", "phrase 2", ...]
```

Output only the JSON array, no extra text.
"""
