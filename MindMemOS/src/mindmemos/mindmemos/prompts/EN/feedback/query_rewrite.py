QUERY_REWRITE_PROMPT = """Rewrite the user's original search query for supplemental memory recall.

The original query was used for time-sensitive response generation and may not retrieve all memories relevant to implicit feedback.
Return JSON only, shaped as {"query": "..."}.
The rewritten query should be concise, durable, and focused on user preferences, corrections, identity facts, or long-term instructions implied by the original query.
Do not answer the user. Do not include explanations.
"""
