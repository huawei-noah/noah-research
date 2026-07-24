EPISODE_DESCRIPTION_PROMPT = """
You are an episodic memory description expert. Generate a brief topic summary from the following conversation.

Conversation timestamp: {conversation_timestamp}
Conversation content:
{conversation_text}

Speaker note: Lines may be formatted as `speaker=Name: ...` for named-speaker dialogue. Treat `Name` as the real speaker of that line; first-person statements in that line belong to `Name`, not automatically to the user. Use explicit speaker names in the summary whenever available.

Generate a structured description optimized for semantic retrieval. Return a JSON object:
{{
    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    "content": "A brief factual summary of the main topic and key points (2-4 sentences)"
}}

Requirements:
1. The title should be specific and easy to search, including key topics, activities, and participant names.
2. The content should briefly summarize the main topic and key facts in third person. Keep it concise — do NOT reproduce the full conversation.
3. Focus on searchable elements: WHO did WHAT, WHEN, WHERE, and WHY.
4. Include all proper nouns: person names, place names, brand names, product names, book/movie titles.
5. Include key numbers, dates, times, quantities, and prices.
6. Use the dual time format for relative references: "relative time (absolute date based on {conversation_timestamp})".
7. Use specific names consistently rather than pronouns.
8. Remove conversational filler, greetings, and redundancy.
9. Preserve causal relationships and decision reasoning.
10. Content should be significantly shorter than the original conversation — a brief overview, not a detailed retelling.

Return only the JSON object, no other text.
"""
