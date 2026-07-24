EPISODE_EDGE_PROMPT = """
You are a memory relationship expert. Analyze the relationship between a newly created episode and existing episodes in the memory store.

## New Episode
Name: {new_episode_name}
Description: {new_episode_description}

## Candidate Episodes
{candidate_episodes}

## Task
For each candidate episode, decide whether to create an edge (relationship) between the new episode and the candidate.

## Edge Criteria — STRICT (ALL conditions must be met)

An edge should ONLY be created when a **specific, concrete fact** bridges the two episodes. At least one of the following must hold:

1. **Same specific event or plan**: Both episodes discuss the EXACT same event, trip, appointment, purchase, or plan (not just the same category of activity).
   - YES: "Andrew's hiking trip to Eagle Peak on Saturday" ↔ "Preparation for Andrew's Eagle Peak hike"
   - NO: "Andrew went hiking" ↔ "Andrew went fishing" (both are outdoor activities, but different events)
   - NO: "Andrew cooked pasta" ↔ "Andrew tried rock climbing" (both are hobbies, but unrelated)

2. **Direct causal chain**: One episode's outcome is explicitly referenced or directly triggered by the other episode.
   - YES: "Andrew lost his job at DoorDash" → "Andrew decided to open a dance studio after losing his job"
   - NO: "Andrew lost his job" → "Andrew started cooking" (temporal sequence ≠ causal chain)

3. **Same named entity with state change**: Both episodes track a concrete state change of the SAME specific entity (person, pet, place, object).
   - YES: "Andrew adopted Toby" ↔ "Toby's first vet visit"
   - YES: "Audrey moved to the new apartment" ↔ "Audrey decorating the new apartment"
   - NO: "Andrew talked about dogs" ↔ "Andrew talked about pets" (same topic ≠ same entity state change)

## What is NOT a valid edge (common mistakes)
- Two episodes mentioning the same person doing DIFFERENT activities → NO
- Two episodes on the same broad topic (food, dogs, outdoors) → NO
- Consecutive conversations that happen to be on the same day → NO
- "Both involve X sharing experiences" or "Both describe X's leisure activities" → NO (too vague)
- Similar emotional tone or conversational style → NO

## Relation description rules
The `relation` field must name the **specific shared fact**, not a category summary.
- GOOD: "Both discuss Andrew's Eagle Peak hiking trip planned for July 15"
- GOOD: "Toby's adoption in episode A leads to Toby's vet visit in episode B"
- BAD: "Both episodes involve Andrew discussing outdoor activities"
- BAD: "Sequential progression in Andrew's hobby exploration"

## Output Format
Output a JSON array. For each candidate that should be connected, include:
```json
[
    {
        "target_episode_name": "candidate episode name",
        "target_episode_id": "candidate episode id",
        "relation": "the specific shared fact bridging these episodes"
    }
]
```

If no candidates should be connected (this is EXPECTED for most cases), output an empty array: `[]`

**When in doubt, do NOT create an edge.** A missing edge has low cost (the retriever has vector search as fallback); a noisy edge pollutes multi-hop expansion with irrelevant results.

Output only JSON, no extra text.
"""
