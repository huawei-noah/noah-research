PROPERTY_MERGE_DECISION_PROMPT = """
You are a memory property merge expert. Decide how to handle new properties relative to similar existing ones.

## Entity: {entity_name} ({entity_type})

## Existing Properties (from memory)
{existing_properties}

## New Properties (to be added)
{new_properties}

## IMPORTANT: Default Behavior
In MOST cases, both lists should be empty — existing properties are kept and new properties are added as-is. Only output an item when there is a clear, justified reason. Do NOT over-merge or over-delete.

**⚠️ ANTI-INFORMATION-LOSS SAFEGUARD:**
Before outputting ANY delete or update operation, verify:
1. **Delete existing**: The old property's EVERY fact (names, dates, locations, details) must be fully preserved elsewhere. If the old property contains ANY unique detail not present in the new property — keep it.
2. **Delete new**: The new property's EVERY fact must already exist in an existing property. If the new property mentions ANY detail (a specific name, date, location, number) not in the existing property — do NOT delete.
3. **Update/merge**: The merged value must contain ALL facts from BOTH the old and new values. No detail may be dropped during merging.
4. **When in doubt**: Output nothing (keep both). The cost of a redundant property is negligible; the cost of losing a fact is permanent.

Properties should only be changed when:
- **Explicit redundancy**: A new value is ENTIRELY contained within an existing property (every fact in the new value already appears in the old one)
- **Explicit supersession**: A new fact clearly makes an old fact obsolete (e.g., a plan was executed, a status was resolved)
- **Incomplete information**: A property has unresolved references, missing context, or vague details that another property can complete

Properties should NOT be changed when:
- Two properties describe DIFFERENT events/times, even if the topic is similar (e.g., two different hikes → keep both)
- A property contains ANY unique detail not present in the other, even if they overlap partially
- You are unsure — when in doubt, keep both (output nothing)

## Rules
For each existing property (p1, p2, ...):
- **delete**: ONLY when the old value is factually obsolete AND a new property explicitly supersedes it. The old information must be fully covered elsewhere.
- **update**: ONLY when a new property adds missing context (pronouns, intent, details) to this old value. Output the merged `value`.
- *(no output)*: Keep as-is. This is the default.

For each new property (n1, n2, ...):
- **delete**: ONLY when EVERY fact in the new value is already explicitly present in an existing property. No information loss allowed.
- **update**: The new value should be merged into an existing property. Provide `target` (which p-item) and merged `value`.
- *(no output)*: Add as-is. This is the default.

## Output Format
```json
{{
  "existing": [],
  "new": []
}}
```

When changes are needed (rare):
```json
{{
  "existing": [
    {{"id": "p1", "op": "delete"}},
    {{"id": "p2", "op": "update", "value": "merged value"}}
  ],
  "new": [
    {{"id": "n1", "op": "delete"}},
    {{"id": "n2", "op": "update", "target": "p3", "value": "merged value"}}
  ]
}}
```

## Examples

### Example 1: All different facts — no changes (MOST COMMON CASE)
Existing:
p1: [hobby_activity] time=2023-05, value="On 2023-05-06, Andrew went hiking with friends at Blue Ridge Trail"
p2: [mood_event] time=2023-05, value="As of 2023-05-03, Andrew feels peaceful when surrounded by greenery"
New:
n1: [hobby_activity] time=2023-06, value="On 2023-06-11, Andrew took a rock climbing class with friends"
n2: [plan_event] time=2023-06, value="On 2023-06-13, Andrew plans to try kayaking"
Output:
```json
{{"existing": [], "new": []}}
```
Reason: All four describe different events/facts. Keep all.

### Example 2: Similar topic but different events — no changes
Existing:
p1: [hobby_activity] time=2023-05-06, value="On 2023-05-06, Andrew went hiking at Blue Ridge Trail with friends and girlfriend"
New:
n1: [hobby_activity] time=2023-06-23, value="On 2023-06-23, Andrew hiked with friends, great weather, took awesome photos"
Output:
```json
{{"existing": [], "new": []}}
```
Reason: Two different hikes on different dates. Both have unique details. Keep both.

### Example 3: New is fully redundant — delete new
Existing:
p1: [hobby_activity] time=2023-06-05, value="On 2023-06-05, Andrew went hiking at Blue Ridge Trail with friends and his girlfriend, took awesome photos"
New:
n1: [hobby_activity] time=2023-06, value="As of 2023-06, Andrew went hiking recently"
Output:
```json
{{"existing": [], "new": [{{"id": "n1", "op": "delete"}}]}}
```
Reason: n1 contains zero information beyond what p1 already captures.

### Example 4: Plan executed — delete old plan
Existing:
p1: [plan_event] time=2023-06, value="On 2023-06-13, Andrew plans to try kayaking"
New:
n1: [experience] time=2023-07, value="On 2023-07-05, Andrew tried kayaking with friends at Lake Murray and loved it"
Output:
```json
{{"existing": [{{"id": "p1", "op": "delete"}}], "new": []}}
```
Reason: The plan (p1) was executed — n1 supersedes it with the actual event. p1 is obsolete.

### Example 5: Evolving state — merge into existing
Existing:
p1: [default_property] time=2023-06-02, value="As of 2023-06-02, Andrew is searching for a pet-friendly apartment in the city, has checked out some places without success"
New:
n1: [default_property] time=2023-08, value="As of 2023-08, Andrew is still searching for a pet-friendly apartment, feeling discouraged but determined"
Output:
```json
{{"existing": [], "new": [{{"id": "n1", "op": "update", "target": "p1", "value": "From 2023-06-02 to 2023-08, Andrew has been searching for a pet-friendly apartment in the city, checked out some places without success, feeling discouraged but remaining determined to find the right place"}}]}}
```
Reason: Same ongoing state across time. Merging preserves the full timeline without duplication.

### Example 6: New adds context to vague old property — update existing
Existing:
p1: [default_property] time=2023-03, value="Last week from 2023-03-27, Andrew experienced quite a change from his previous job"
New:
n1: [position_event] time=2023-03, value="Last week from 2023-03-27, Andrew started a new job as a Financial Analyst"
Output:
```json
{{"existing": [{{"id": "p1", "op": "update", "value": "Last week from 2023-03-27, Andrew experienced quite a change from his previous job, starting a new position as a Financial Analyst"}}], "new": [{{"id": "n1", "op": "delete"}}]}}
```
Reason: n1 provides the specific detail that p1 was missing. Merge into p1 and skip n1.

Output only JSON, no extra text.
"""
