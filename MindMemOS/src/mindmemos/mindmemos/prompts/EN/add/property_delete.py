PROPERTY_DELETE_DECISION_PROMPT = """You are a memory fusion expert. For entity "{entity_name}", decide which old property values should be deleted.

## Decision Rules (Conservative - Keep More, Delete Less)
- **KEEP all historical records by default** - memory is valuable and should not be deleted simply because it's old
- Only delete when ONE of these conditions is TRUE:
  1. The previous memory was wrong (e.g., the user says "oh no, it should be XXX" - an explicit correction or rectification of information)
  2. The information is **highly unlikely to ever be asked** again, typically intermediate reasoning steps or content with no informational value
- **DO NOT delete** just because:
  - The value is old or from past events
  - The new value is "newer"
  - There's a simple update (e.g., position changed - keep both old and new)
- When in doubt, **KEEP** the historical record

## Context
{context_text}

## Output Format
Output only a JSON array, no extra text:
```json
[
  {"property_name": "property name", "timestamp": "time stamp"}
]
```

If no properties need to be deleted, output empty array [].
"""
