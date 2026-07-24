RELATION_DETECTION_PROMPT = """You are the first-stage memory consolidation issue detector. Given one PRIMARY_RECENT_MEMORY and retrieved neighbor memories, identify problem-specific memory groups that deserve a second LLM's focused action planning.

Your job is scope selection and issue classification only. Do not propose database actions. Do not decide which memory to archive, keep, update, or merge.

Return JSON matching this schema:
{{
  "issue_groups": [
    {{
      "issue_type": "conflict|duplicate|near_duplicate|complementary|low_value|ambiguous|other",
      "memory_ids": ["ids of memories involved in this single issue"],
      "subject_hint": "shared/coreferent subject if identifiable",
      "predicate_hint": "shared/compatible relation or property if identifiable",
      "value_hints": {{"memory_id": "value/object or key detail for that memory"}},
      "confidence": "high|medium|low",
      "reason": "brief reason"
    }}
  ]
}}

Issue categories:
- conflict: same or coreferent subject AND same/compatible predicate/property, but different values/current claims.
- duplicate: exact duplicate or semantically identical fact.
- near_duplicate: almost the same fact with minor wording/detail differences.
- complementary: non-conflicting fragments about the same subject and same event/fact that may be better merged.
- low_value: memory is clearly noisy, malformed, overly generic, or not useful by itself.
- ambiguous: there may be an issue, but subject equality, predicate equality, or value comparison is uncertain.
- other: a consolidation issue that does not fit the above categories.

Grouping rules:
- Emit one issue group per problem type. If the same memories have two different problems, output two groups with the same memory_ids but different issue_type.
- A group may contain more than two memories when they participate in the same issue, e.g. A/B/C are duplicates or A/B/C are conflicting values for the same subject+property.
- Prefer small focused groups. Do not include unrelated neighbors just because they were retrieved together.
- If memories merely share a broad entity/value/object but describe different subjects, predicates, events, or roles, do not group them.
- Sharing only a country, sport, language, city, organization, person-as-object, or other value is not enough.
- If the only connection is the graph entity used for retrieval, ignore it unless that entity is also the shared/coreferent subject of the grouped memories.

Safety rules:
- Do not use real-world knowledge, plausibility, or common sense to decide whether a memory is correct.
- Do not classify different predicates as conflict. Example: "director of ABC" and "original broadcaster of a show is ABC" are different facts.
- Do not classify different subjects as conflict. Example: different people who are citizens of the same country are different facts.
- When uncertain, use ambiguous rather than conflict/duplicate.
- Return only JSON.
- Do not invent memory IDs.

Memory cluster:
{context}
"""


__all__ = ["RELATION_DETECTION_PROMPT"]
