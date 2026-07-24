EXPLICIT_SEARCH_DECISION_PROMPT = """Decide whether explicit user feedback needs one supplemental memory search.

Return JSON only, shaped as {"need_search": false, "query": null} or {"need_search": true, "query": "..."}.

Set need_search=true only when the provided recalled_memories may be insufficient to determine the correct memory action.
Use the feedback and conversation messages to produce a concise query that can retrieve related existing memories.
Set need_search=false when recalled_memories already contain enough information to add, update, delete, or noop.
"""
