ENTITY_RELEVANCE_FILTER_PROMPT = """You are a memory retrieval filter. Given a user's query and a memory entity, decide if this entity is OBVIOUSLY IRRELEVANT and should be removed.

User Query: {query}

Memory Entity Information:
{entity_description}

## Core Principle: KEEP by default. Only filter out entities that have **absolutely no helpful information** for the query.

### FILTER OUT (relevance: "no") — ONLY when ALL of these are true:
1. The entity has ZERO factual overlap with the query — no shared people, places, events, time periods, or topics
2. No property value could contribute to answering the query, even indirectly or through logical reasoning

### KEEP (relevance: "yes") — in ALL other cases, including:
- The entity mentions any person, place, time, or event referenced in the query
- The entity could provide background, context, or supporting facts
- The entity discusses a related topic, even if not a direct answer
- The entity is an episode containing dialogue that might mention the query topic
- You are even slightly unsure whether it's relevant
- The entity contains things that fall within the scope of the question

**Err heavily on the side of keeping.** A wrongly kept entity costs little (the answer model can ignore it). A wrongly filtered entity can lose critical evidence.

Respond with JSON:
{{
    "relevance": "yes" or "no",
    "reasoning": "Brief explanation"
}}"""

# Batch entity relevance filter (placeholder, not currently used)
BATCH_ENTITY_RELEVANCE_PROMPT = ENTITY_RELEVANCE_FILTER_PROMPT
