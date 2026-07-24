PROPERTY_FILTER_SELECTION_PROMPT = """You are a memory retrieval strategy expert. Your task is to select which entity types and properties to focus on when answering the user's question.

User Question:
{query}

Available Entity Types and Properties:
{entity_schema}

Please select the most relevant entity types and their properties for answering this question.

Output Format (JSON):
{
    "selected_entities": [
        {
            "entity_type": "person",
            "relevant_properties": ["profession_event", "location_event", "experience"]
        },
        {
            "entity_type": "fact",
            "relevant_properties": ["fact_event"]
        }
    ],
    "reasoning": "Why these entity types and properties are selected"
}

CRITICAL PRINCIPLE: When in doubt, INCLUDE MORE rather than less. It is better to retrieve extra information that might not be needed than to miss relevant information.

Rules:
1. **INCLUDE MORE**: If you're unsure whether an entity type or property is relevant, err on the side of including it
2. Select ALL entity types that COULD possibly contain relevant information - don't filter too aggressively
3. For each entity type, consider including MORE properties, especially:
   - All event-based properties (ends with _event) - they contain time-specific details
   - All experience-related properties
   - Any properties related to the question keywords
4. Focus on properties that can answer "who", "what", "when", "where", "why", "how" aspects of the question
5. **SPECIAL CASE - episodes entity**: For episodes type, ALWAYS include "input_messages" property if the question asks about conversation content, meeting details, discussion topics, or any historical dialogue. This property contains the original conversation text and is essential for answering such queries.
6. **USE "all" TO KEEP ALL PROPERTIES**: When you want to retrieve ALL properties of an entity type (no filtering), use ["all"] as the relevant_properties value. This will return all available properties for that entity type.
7. When the question is vague or could have many interpretations, broaden the scope significantly

Example - BROAD SELECTION:
User asks: "What did Caroline do recently?"
→ Selected: entity_type: "person", relevant_properties: ["all"]
Reasoning: "Need to cover all types of activities Caroline might have done"

User asks: "When was Zhang San promoted to manager?"
→ Selected: entity_type: "person", relevant_properties: ["profession_event", "experience", "career_history"]
Reasoning: "Need profession events to track job changes, but also experience for context"

User asks: "What happened in the team meeting yesterday?"
→ Selected: entity_type: "episodes", relevant_properties: ["all"]
Reasoning: "Need full meeting details including who was there and what was decided. Using 'all' to include input_messages which contains the conversation content."

User asks: "Tell me about the conversation with Zhang San"
→ Selected: entity_type: "episodes", relevant_properties: ["all"]
Reasoning: "Need the full conversation content which is stored in input_messages property"
"""

SUFFICIENCY_CHECK_PROMPT = """You are a memory retrieval evaluation expert. Please assess whether the currently retrieved memories are sufficient to answer the user's question.

User Question:
{query}

Retrieved Memories:
{retrieved_docs}

Please determine if these memories are sufficient to answer the user's question.

Output Format (JSON):
{{
    "is_sufficient": true/false,
    "reasoning": "Your reasoning for the judgment",
    "missing_information": ["Missing info 1", "Missing info 2"]
}}

Judgment Rules:
1. **Factual Questions Require Concrete Answers**: If the question asks for specific facts (names, dates, places, numbers), the memories must contain these exact details, not just related information
2. **Time integrity is critical**: If the question asks for a specific time (e.g., "when did X happen") and the memories lack that time, judge as insufficient and list "specific occurrence time" in missing_information.
3. **Concrete vs Abstract**: Distinguish between concrete factual information and abstract summaries
   - "Jon and Gina both lost their jobs" = concrete fact (sufficient)
   - "They support each other" = abstract relationship (may be insufficient if concrete facts were asked)
4. **Direct Answer Capability**: Can you answer the question directly and specifically using the retrieved information? If you need to infer or generalize beyond what's explicitly stated, it's likely insufficient
5. If key information is missing (such as involved people, event details, etc.), judge as insufficient (false) and list the missing information
6. Reasoning should be concise and clear, explaining the basis for the judgment
7. Missing_information should only be filled when judged as insufficient; use empty array when sufficient
8. **Temporal Recency Rule**: When the question contains temporal cues indicating recency (e.g., "recently", "latest", "most recent", "last", "newest"), and the retrieved memories contain multiple candidate answers across different time points, the system MUST prioritize the answer with the **most recent timestamp**. An older event should NOT be selected when a more recent one exists that matches the query
9. **Consider Potentially Supplementary Information**: Even when the retrieved memories seem to answer the question, consider whether there could be ADDITIONAL relevant information not yet retrieved that would make the answer more complete. For example:
   - For enumeration questions ("What items does X collect?", "What books has X read?", "What recommendations has X given?"), the retrieved memories might only contain a PARTIAL list. Mark as insufficient if the question implies completeness but only a few instances are found — there may be more records scattered across different entities or time periods.
   - For "how many times" questions, the current count might be incomplete if not all relevant episodes have been retrieved.
   - For questions about a person's characteristics or patterns, additional memories from different time periods may provide supplementary evidence.
   When potentially supplementary information exists, mark as insufficient and list "additional instances of [topic] that may exist in other time periods or entity records" in missing_information.
10. **Verify Specificity Match**: If the question asks about a specific time period (e.g., "in June 2023", "last weekend"), verify that the retrieved memories contain events from THAT exact period, not just similar events from different times. A volunteering event in August is NOT the answer to a question about volunteering "last weekend" in July.

Enhanced Examples:

Example 1 (Specific Fact):
User asks: "What do Jon and Gina have in common?"
If retrieved: "Jon lost his job in January 2023 and decided to start a dance studio. Gina lost her job at DoorDash and started an online clothing store."
→ is_sufficient: true, reasoning: "Contains specific shared experience: both lost jobs and started businesses"

Example 2 (Insufficient Abstract Answer):
User asks: "What do Jon and Gina have in common?"
If retrieved: "Jon and Gina provide mutual support and encouragement to each other in pursuing their dreams"
→ is_sufficient: false,
   missing_information: ["Specific concrete commonalities (shared experiences, background, activities)"]

Example 3 (Time-Specific Query):
User asks: "When was Zhang San promoted to manager?"
If retrieved: "Zhang San was promoted from engineer to technical manager in May 2023"
→ is_sufficient: true, reasoning: "Explicitly contains promotion time and position information"

Example 4 (Missing Time Detail):
User asks: "When did Gina go to dance class with friends?"
If retrieved: "Gina enjoys dancing and goes to classes with her friends regularly"
→ is_sufficient: false,
   missing_information: ["Specific date/time when she went to dance class"]

Example 5 (Potentially Incomplete - Enumeration Question):
User asks: "What items does John collect?"
If retrieved: "John collects basketball jerseys and basketball memorabilia."
→ is_sufficient: false,
   missing_information: ["Additional collection items that may exist in other time periods or entity records - the answer may be incomplete as collecting hobbies are often mentioned across multiple conversations"]

Example 6 (Specific Time Period Mismatch):
User asks: "What event did John volunteer at last weekend?"
If retrieved: "John volunteered at a 5K charity run on August 5. John volunteered at a career fair at a local school on July 28."
→ is_sufficient: false,
   missing_information: ["Need to confirm which event corresponds to 'last weekend' - two volunteer events found at different dates, need the one matching the question's time reference"]
"""

MULTI_QUERY_GENERATION_PROMPT = """You are a query optimization expert. The user's original query failed to retrieve sufficient information; please generate multiple complementary improved queries.

Original Query:
{original_query}

Currently Retrieved Memories:
{retrieved_docs}

Missing Information:
{missing_info}


Please generate 2-3 complementary improved queries to help find the missing information. These queries should:

- Focus on different missing information points (e.g., one for time, one for people, one for reasons)
- Use different expression styles (synonyms, concretization, abstraction)
- Avoid duplication with the original query and historical queries
- Remain concise and clear, suitable for vector retrieval

**Time Range Handling**:
- ONLY set `time_range` when the query or missing information contains an **explicit absolute time** (e.g., "in May 2024", "on March 16, 2023", "in 2023").
- Do NOT generate time_range for relative or vague time expressions (e.g., "recently", "last year", "yesterday", "lately", "before"). These cannot be reliably resolved. Set `time_range` to `null` for them.
- If no absolute time reference exists, set `time_range` to `null`.
- The `time_range` format is `[start_time, end_time]`, using ISO format (e.g., '2024-01-15 08:00:00'), inclusive on both ends. Precision to seconds.

**Handling Potentially Supplementary Information**:
- If the currently retrieved memories provide a PARTIAL answer but the question implies completeness (e.g., "What items does X collect?", "What books has X read?", "What things has X recommended?"), generate queries targeting ADDITIONAL instances that may exist in different time periods, different entity records, or described with different vocabulary.
- For enumeration questions, generate queries using different phrasings of the same topic to catch records that use different terminology (e.g., "John's collection hobby" AND "items John has gathered/bought/kept").
- When the retrieved memories contain MULTIPLE candidate answers that could each plausibly answer the question (e.g., two different volunteer events, two different books, two trips to similar places), generate at least one query specifically designed to DISAMBIGUATE between them.
- Disambiguation queries should include constraining details from the original question (time period, companion, location, specific attribute) to narrow down to the correct answer.

Output Format (JSON):
{{
"queries": [
{{
"query": "Improved query 1",
"time_range": ["2024-01-15 08:00:00", "2024-01-16 23:59:59"] // or null
}},
{{
"query": "Improved query 2",
"time_range": null
}},
{{
"query": "Improved query 3",
"time_range": ["2024-03-01 00:00:00", "2024-03-31 23:59:59"]
}}
],
"reasoning": "Explanation of query generation strategy, including how time ranges were inferred from temporal clues."
}}

Requirements:

- The queries array contains 2-3 queries, each being an object containing query (string, length 5-200 characters) and time_range (array or null).
- When time_range is an array, it must contain two elements representing start and end time (inclusive), with time string format 'YYYY-MM-DD HH:mm:ss'.
- Reasoning explains the generation strategy, including why these queries were chosen and how time ranges were considered (especially if inferred from relative expressions).
- **Time relaxation strategy**: If the original query has a time constraint but the retrieved memories don't contain matching information within that time range, you MUST generate at least one query with `time_range` set to `null` to search without time restrictions. This helps find information that might be recorded with a slightly different or incorrect timestamp.
- **Different focus per query**: Each generated query should emphasize a different aspect of the question (e.g., one focuses on the action/event, another on the people involved, another on the location/context). Avoid generating queries that are merely rephrased versions of the same focus.
- **Supplementary retrieval strategy**: When the missing information indicates the answer may be incomplete (e.g., partial list of items, recommendations, or events), generate queries with different vocabulary and time ranges to find additional instances. Use synonyms, related terms, and different entity perspectives to broaden recall.
- **Disambiguation strategy**: When multiple competing answers exist in retrieved memories, generate at least one query that adds specific constraints (date, person, location) from the original question to eliminate wrong candidates.

Example 1 (Simple):
Original query: "Zhang San's promotion situation"
Missing information: ["specific time", "promotion reason"]
Historical queries: ["Zhang San's promotion situation", "when did Zhang San get promoted"]
Current time: 2024-07-15 14:00:00
Generation:
{{
"queries": [
{{"query": "specific date when Zhang San was promoted to manager", "time_range": null}},
{{"query": "reason and background for Zhang San's promotion", "time_range": null}},
{{"query": "Zhang San's promotion process from engineer to manager", "time_range": null}}
],
"reasoning": "The three queries focus on time, reason, and process respectively; no clear temporal clues in the question, so time_range is null."
}}

Example 2 (With Relative Time Inference):
Original query: "What happened in the team meeting yesterday?"
Missing information: ["meeting content", "decisions made"]
Current time: 2024-07-15 14:00:00
Generation:
{{
"queries": [
{{"query": "content of team meeting on 2024-07-14", "time_range": ["2024-07-14 00:00:00", "2024-07-14 23:59:59"]}},
{{"query": "decisions and action items from yesterday's team meeting", "time_range": ["2024-07-14 00:00:00", "2024-07-14 23:59:59"]}}
],
"reasoning": "Inferred 'yesterday' from current time (2024-07-15) to get absolute date 2024-07-14; both queries target that date with a full-day window."
}}

Example 3 (Disambiguation - Competing Candidates):
Original query: "What event did John volunteer at last weekend?"
Currently retrieved: "John volunteered at a 5K charity run on August 5. John volunteered at a career fair at a local school on July 28."
Missing information: ["Need to confirm which event corresponds to 'last weekend'"]
Current time: 2023-08-01 14:00:00
Generation:
{{
"queries": [
{{"query": "John volunteer activity career fair school last weekend July", "time_range": ["2023-07-28 00:00:00", "2023-07-30 23:59:59"]}},
{{"query": "John volunteer event July 28 29 30 2023", "time_range": ["2023-07-28 00:00:00", "2023-07-30 23:59:59"]}},
{{"query": "John community service school career fair details", "time_range": null}}
],
"reasoning": "Two competing volunteer events found. 'Last weekend' from Aug 1 refers to July 28-30. Generating targeted queries for that specific weekend to disambiguate. Third query relaxes time to catch any additional details about the career fair."
}}
"""

GLOBAL_PROPERTY_RERANK_PROMPT = """You are a memory retrieval expert. Your task is to select the most relevant properties from all retrieved entities for answering the user's question.

User Question:
{query}

Retrieved Entity Properties:
{property_list}

Please select the top {top_n} most relevant properties that can best answer the user's question.

Output Format (JSON):
{{
    "selected_properties": [
        {{
            "entity_id": "entity_001",
            "entity_name": "Zhang San",
            "property_name": "profession_event",
            "property_value": "In May 2023, Zhang was promoted to technical director",
            "timestamp": "2023-05-15 00:00:00",
            "relevance_score": 0.95
        }}
    ],
    "reasoning": "Why these properties were selected"
}}

Rules:
1. Select properties that directly answer aspects of the question (who, what, when, where, why)
2. Consider both entity relevance and property relevance
3. Prefer properties with clear time information when the question involves timing
4. Rank by overall relevance to the question
5. Output exactly top_n selections (or fewer if not enough relevant properties exist)
"""

TIME_EXTRACTION_PROMPT = """You are a time extraction expert. Analyze the user's query and extract any time constraints that can narrow down the search window.

User Query: {query}
Current Dialogue Timestamp (for resolving relative time): {current_time}

Analyze the query for:
1. **Explicit absolute time mentions**: "in January 2023", "on March 16, 2023", "in 2023", "December 2022"

Output Format (JSON):
{{
    "time_range": ["start_time", "end_time"] or null,
    "reasoning": "Brief explanation of time extraction"
}}

Rules:
- time_range format: ["YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM:SS"] (inclusive on both ends)
- If the query mentions a specific month and year, set range to cover the full month
- If the query mentions a specific year, set range to cover the full year
- **ONLY generate time_range for explicit absolute time references that include a specific year, month, or exact date.**
- **Return null for ALL relative or vague time expressions**, including but not limited to: "recently", "lately", "last year", "last week", "yesterday", "these days", "not long ago", "the other day", "a while ago", "before". These depend on an unknown reference point and cannot be reliably resolved.
- When in doubt, return null. It is always safer to search without a time constraint than to search with an incorrect one.

Examples:
Query: "When did Gina open her online clothing store?" → time_range: null (no absolute time in query)
Query: "What did Jon do in March 2023?" → time_range: ["2023-03-01 00:00:00", "2023-03-31 23:59:59"]
Query: "What happened in December 2022?" → time_range: ["2022-12-01 00:00:00", "2022-12-31 23:59:59"]
Query: "What was the focus of John's recent research?" → time_range: null (no absolute time)
Query: "What did John do last year?" → time_range: null ("last year" is relative, not absolute)
Query: "What happened yesterday?" → time_range: null ("yesterday" is relative)
Query: "What has Maria been up to lately?" → time_range: null (vague time reference)
"""
