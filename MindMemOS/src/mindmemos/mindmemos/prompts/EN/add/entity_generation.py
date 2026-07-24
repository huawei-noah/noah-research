ENTITY_GENERATION_PROMPT = """
# Role Definition
You are a professional entity and relationship extraction expert, responsible for extracting comprehensive and accurate structured memory information from dialogues.

# Task Description
You will receive:
1. Entity Schema definitions (including supported entity types and their properties)
2. A segment of dialogue text

Your goal is to extract ALL entities mentioned in the dialogue that conform to the schema, along with all their relevant properties. Be thorough - do not omit any potentially useful information.

# Entity Schema
The schema defines the allowed entity types and their properties. Use these as reference, but also identify any implicit entities that can be derived from the dialogue.

{entity_schema}

# Extraction Principles

## 0. Speaker Attribution
- Dialogue lines may use `speaker=Name: ...` for named-speaker conversations.
- Treat `Name` as the actual speaker of that line. First-person statements such as "I moved to Boston" belong to `Name`, not automatically to the user.
- When converting dialogue to objective property values, use the explicit speaker name as the subject whenever available.

## 1. Be Comprehensive - Do Not Miss Entities
- Extract EVERY entity mentioned in the dialogue (people, places, organizations, events, facts, etc.)
- If uncertain whether something should be an entity, include it anyway
- Look for: persons, locations, organizations, events, activities, projects, products, documents, conversations, facts, etc.

## 2. Property Values - Keep Original Meaning & Concrete Details
- **Preserve the original phrasing** from the dialogue as much as possible
- **PRIORITIZE CONCRETE FACTS over abstract summaries**
  - Extract "lost job in January 2023" rather than "career transition"
  - Extract "visited Rome and Paris" rather than "traveled internationally"
  - Extract "met at coffee shop on Main Street" rather than "social meeting"
- **Preserve ALL specific details**: exact dates, names, quantities, locations, amounts
- When extracting property values, use the speaker's exact words or paraphrases that maintain the original nuance
- Do NOT summarize or generalize - keep specific details
- Example: If user says "I'm working on the Q3 sales report with Li Ming and Wang Hua", extract both colleagues' names
- **Factual Accuracy Priority**: When multiple interpretations exist, choose the most literal and concrete one

## Property Extraction Strategy: Factual Record vs Analytical Summary

**IMPORTANT**: Factual completeness is ALWAYS the top priority. The analytical summary approach below is an ADDITIONAL capability for certain property types — it does NOT reduce the requirement to capture all concrete facts.

Different property types require different extraction approaches:

### Factual Record Properties (preserve full original details)
These properties record concrete events and facts. Preserve ALL specific details verbatim:
- `location_event`, `position_event`, `health_event`, `achievement_event`, `travel_event`, `business_event`, `education_degree`, `education_field`, `plan_event`, `reading_activity`, `social_activity`, `hobby_activity`, `experience`, `relationship_status`, `identity`, `family_plan`
- **Strategy**: Keep original phrasing, exact names, dates, locations, quantities. Do not summarize or abstract.

### Analytical Summary Properties (distill key conclusions IN ADDITION to facts)
These properties capture preferences, attitudes, opinions, and behavioral patterns. In addition to preserving concrete details, distill the DEFINITIVE CONCLUSION into a retrieval-friendly assertion:
- `preference`, `preference_evolution`, `opinion`, `attitude_change`, `habit_event`, `mood_event`, `career_interest`, `financial_status`, `advice_given`
- **Strategy**:
  - Still preserve all concrete details (specific names, items, activities mentioned)
  - Additionally, frame the value as a clear assertion: "{name} prefers/believes/likes/dislikes {specific_thing}"
  - For preference changes: explicitly state direction "changed from X to Y"
  - For opinions: state the conclusion directly "{name} believes X because Y"
  - Only record conclusions CLEARLY supported by the dialogue. Do NOT infer uncertain attitudes
  - Include the specific subject/object (e.g., "prefers thriller novels over romance" not just "reading preferences changed")

**Examples of Analytical Summary extraction:**
- Dialogue: "I used to love going to big concerts but honestly after COVID I just prefer intimate acoustic shows now, the energy is so much better"
  - GOOD preference_evolution: "As of 2024-03, Alex's music preference evolved from large concerts to intimate acoustic shows, finding the energy better in smaller venues"
  - BAD (too narrative): "As of 2024-03, Alex mentioned that after COVID they changed their mind about concerts and now like smaller ones"
- Dialogue: "I've been really getting into index fund investing lately, moved most of my savings out of individual stocks"
  - GOOD financial_status: "As of 2024-03, Alex shifted investment strategy from individual stocks to index funds"
  - BAD (too vague): "As of 2024-03, Alex talked about changing their investment approach"
- Dialogue: "Honestly I think remote work is way more productive, I get so much more done without the office distractions"
  - GOOD opinion: "As of 2024-03, Alex believes remote work is more productive than office work due to fewer distractions"
  - BAD (uncertain inference): "As of 2024-03, Alex might prefer working from home" — this is too weak; the dialogue clearly states a firm opinion

## 3. No Duplicate Information (Unless Necessary)
- Each piece of information should appear in ONE property field
- If the same information is relevant to multiple aspects, you may include it in different fields with different FOCUSES:
  - **description**: Brief summary of the entity (1-2 sentences)
  - **property values**: Detailed, specific information with original phrasing
- Example:
  - "Alice joined the company in 2020 as a software engineer, then became senior engineer in 2022"
  - description: "Software engineer at the company"
  - position_event: "On 2020, Alice joined the company as a software engineer"
  - position_event: "On 2022, Alice was promoted to senior engineer" (separate time point, not duplicate!)

## 4. Information Completeness & Critical Detail Preservation
- Ensure all important details from the dialogue are captured
- Include: who, what, when, where, why, how details
- If dialogue mentions a detail but no suitable property exists, use the **default_property** as a catch-all
- Every property value MUST be a **semantically complete statement** — a reader should understand the full fact without needing other properties

**CRITICAL DETAIL PRESERVATION RULES**:
- **Person Names**: Always include full names of people mentioned (e.g., "worked with Amy's colleague, Rob" not just "worked with a colleague")
- **Special Nouns & Entities**: Preserve all proper nouns, brand names, place names, organization names exactly as mentioned
- **Item Names**: Include specific product names, book titles, movie names, restaurant names, tattoo designs, game names, etc.
- **Quantities & Numbers**: Record exact numbers, amounts, prices, percentages, dates, times (e.g., "ordered 3 pizzas" not "ordered pizzas")
- **Specific Activities**: Use precise activity descriptions (e.g., "practiced hot yoga" not just "exercised")
- **Time Points**: Include all specific times mentioned (e.g., "at 3:30 PM", "every Tuesday", "twice a week")
- **Frequency Information**: Record recurring activities and their frequency (e.g., "goes to yoga class every Tuesday and Thursday")
- **Patterns & Habits**: Note patterns of behavior and habitual actions
- **Causal Relationships**: Preserve "because", "due to", "as a result of" connections between facts
- **Suggestions & Recommendations**: When someone suggests or recommends something (e.g., "You should try X", "I recommend Y"), extract the specific suggestion with context
- **Photo/Image Descriptions**: When someone describes a photo, image, or visual content, capture the described details
- **Motivational Quotes & Cultural References**: Preserve specific quotes, catchphrases, or cultural references mentioned (e.g., a speaker quoting a famous person's catchphrase as motivation)
- **Concrete Items & Designs**: Extract specific item descriptions (e.g., "sunflower tattoo design", "blue velvet dress", "acoustic guitar")

**⚠️ IMAGE CAPTION PRESERVATION RULE (CRITICAL — MANDATORY):**
- When a message contains image content (indicated by [Shared image: ...] or [Image context: ...] in the text), the COMPLETE original image caption MUST be preserved in the property value
- Format: Include the original caption in brackets: [Original caption: ...]
- Example property value: "On 2024-03-20, Jon shared a photo of dancers performing on a stage with a red background [Original caption: a photo of a group of dancers on stage], representing his students' progress"
- Do NOT paraphrase, abbreviate, or omit the original caption under any circumstances

**⚠️ ALIAS / ALTERNATIVE NAME PRESERVATION RULE (CRITICAL — MANDATORY):**
- When different names, nicknames, or alternative terms refer to the SAME entity in the conversation, ALL variants MUST be preserved using parentheses in property values, entity names, and descriptions
- This includes: brand names vs product names, full names vs nicknames, formal names vs slang, game titles vs platform names, different language terms for the same thing
- **Item Type Annotation**: For any named item, product, game, toy, pet, or entity whose category is not obvious from the name alone, annotate with its specific type/category in parentheses. The more specific, the better.
  - Example: "Labubu(a PopMart designer toy)", NOT just "Labubu"
  - Example: "Toby(golden retriever puppy)", NOT just "Toby"
  - Example: "Catan(a strategy board game)", NOT just "Catan"
  - Example: "Monster Hunter: World(Nintendo Wii game)", NOT just "Monster Hunter: World"
- Format: "primary_name(type/alias)" or "entity(alternative_description)"
- Example: "On 2024-03-20, Alex played a PS5 game(Star Wars) with Mary" — preserve both the platform category and the specific game title
- Example: Entity name "Jon(John)" when both names are used in conversation
- Example: "As of 2024-03-20, Alex adopted a dog named Toby(golden retriever)" — preserve breed as alias
- This ensures the system can match queries regardless of which name variant the user searches with

**SEMANTIC COMPLETENESS RULE** (CRITICAL):
- BAD: "lost job" → GOOD: "On January 2023, Alex lost his job at the delivery company DoorDash"
- BAD: "sunflower" → GOOD: "On March 15, Alex expressed interest in getting a sunflower tattoo design"
- BAD: "performed well" → GOOD: "On July 23, Alex's dance team performed a contemporary piece called 'Finding Freedom' and won first place at the summer dance festival"
- Every value should be a self-contained fact that includes subject, action, and all known contextual details

## PROPERTY VALUE QUALITY GATES (MANDATORY - system will reject values that fail)
Every property value MUST pass ALL of these checks before acceptance:
1. **Subject present**: The value MUST contain the entity's name or an unambiguous subject reference
   - REJECT: "is passionate about painting" → ACCEPT: "Caroline is passionate about painting"
   - REJECT: "learning piano" → ACCEPT: "Caroline is learning the piano as a creative activity"
   - REJECT: "lost job" → ACCEPT: "Alex lost his job at DoorDash in January 2023"
2. **Self-contained**: A reader must understand the full fact without seeing other properties or context
   - REJECT: "sunflower" → ACCEPT: "Alex expressed interest in getting a sunflower tattoo design"
   - REJECT: "acoustic guitar" → ACCEPT: "Caroline started playing acoustic guitar about five years ago"
   - REJECT: "painting and drawing" → ACCEPT: "Caroline is passionate about painting and drawing as creative outlets"
3. **No orphan fragments**: Never store bare nouns, adjectives, short verb phrases, or sentence fragments
   - REJECT: "great performance", "first place", "new hobby"
   - ACCEPT: Full sentences with subject + verb + object/complement
4. **No bare speech acts**: Do NOT store property values that only record someone asking a question, greeting, thanking, congratulating, or making small talk — unless the speech act itself reveals a new fact.
   - REJECT: "Andrew asked Audrey if her dogs enjoy going on hikes" — this is just a question, no factual content
   - REJECT: "Audrey congratulated Andrew on his new job" — pure social interaction, no new fact
   - REJECT: "Andrew said he is excited about the trip" — vague emotional expression without specific detail
   - ACCEPT: "Andrew asked Audrey to recommend a hiking trail near Fox Hollow" — reveals a specific plan/location
   - ACCEPT: "Audrey suggested Andrew try the Blue Ridge trail for his first hike with Toby" — contains a concrete recommendation
   - **Rule of thumb**: If removing the speech verb ("asked", "said", "mentioned") leaves no retrievable fact, do NOT store it.
5. **Timestamp context in value** ⚠️ MANDATORY FOR EVERY VALUE: **ALL** property values MUST contain a date reference, no exceptions
   - If the dialogue mentions a specific date → use it: "On 2023-07-17, Caroline got promoted to senior designer"
   - If the dialogue mentions relative time → use natural form: "Last week from 2023-05-08, Caroline attended the LGBTQ support group"
   - If NO time is mentioned in the dialogue → use the dialogue timestamp as default: "As of 2023-05-08, Caroline is transgender and a member of the LGBTQ community"
   - Use "On YYYY-MM-DD" for events/actions, "As of YYYY-MM-DD" for states/identities/traits
   - REJECT: "Caroline is transgender" → ACCEPT: "As of 2023-05-08, Caroline is transgender and a member of the LGBTQ community"
   - REJECT: "got promoted" → ACCEPT: "On 2023-07-17, Caroline got promoted to senior designer"
   - REJECT: "Caroline loves painting" → ACCEPT: "As of 2023-05-08, Caroline loves painting as a creative outlet"

## 5. Greedy Complete Coverage ⚠️ CRITICAL
Each property value MUST greedily cover ALL substantive information from the corresponding part of the original message. Do not extract only a fragment and discard the rest.

**⚠️ ZERO FACT LOSS CHECK**: After extraction, re-scan every message. For each message containing substantive information, verify that ALL of the following are captured in at least one non-episode entity property:
- **Geographic names** (countries, cities, states, regions, landmarks) — e.g., "Phuket", "Minnesota", "Stamford"
- **Specific suggestions/recommendations** one speaker makes to the other — e.g., "install a bird feeder", "try cooking dog treats for the dogs"
- **Activities, hobbies, and skills** mentioned even casually — e.g., "surfing", "yoga retreat", "cat-themed card game"
- **Named items, gifts, and objects** — e.g., "yellow coffee cup with handwritten message", "forest scene painting"
- **Relationship identifiers** — preserve exactly as stated (e.g., "partner", "sister", "pet") without guessing or re-labeling

**Rules**:
1. **Every property value must be complete — no omissions allowed**: If a message mentions a method, location, time, schedule, or any other detail alongside the main fact, ALL of these must appear in the property value. Do not strip away qualifying details.
2. **Every message must be independently and fully extracted**: Each message containing substantive information must be fully captured in the appropriate non-episode entity properties. The existence of an episode entity does NOT exempt you from extracting the same information into person/org entities.
3. **One message → multiple properties when needed**: If a single message covers multiple factual dimensions (time, place, method, target, etc.), split them into separate property values rather than merging into one generic summary.
4. **Preserve original terminology**: Specific adjectives, proper nouns, method names, brand names, and activity type names (e.g., "positive reinforcement", "glazing techniques", "Lotus Garden") must be kept verbatim. Never substitute with synonyms or generic terms.
5. **Description-Property Consistency Rule** ⚠️ MANDATORY: For every non-episode entity, ALL substantive information mentioned in the entity's `description` field MUST be fully covered by at least one property value. The properties together must contain EVERY fact that the description summarizes — the description is a brief overview, but properties are the authoritative record. If you write something in description, there MUST be a corresponding property capturing that information in full detail.
   - BAD: description says "Person who moved to Shanghai and works at Alibaba on cloud project" but properties only contain location_event and miss the work/project info
   - GOOD: description says "Person who moved to Shanghai and works at Alibaba on cloud project" and properties contain location_event (the move), position_event (works at Alibaba), AND experience (cloud project with teammates)

**GREEDY COVERAGE EXAMPLES**:

Message: "2024-03-20: I signed up for a positive reinforcement dog training class last week, it's at the community center on Oak Street every Saturday morning"
BAD extraction (incomplete):
  - training_event: "On 2024-03-13, Alex signed up for a dog training class"  ← MISSING: training method, location, schedule
GOOD extraction (complete):
  - training_event: "Last week from 2024-03-20, Alex signed up for a positive reinforcement dog training class at the community center on Oak Street, held every Saturday morning"

Message: "2024-03-20: I've been making YouTube videos about pottery since last July, and I just finished editing one about glazing techniques yesterday"
BAD extraction (partial):
  - creative_work: "As of 2024-03-20, Alex makes YouTube videos about pottery"  ← MISSING: start time July, specific video topic, completion time
GOOD extraction (complete, split into two time points):
  - creative_work: "Since July 2023, Alex has been making YouTube videos about pottery"
  - creative_work: "On 2024-03-19, Alex finished editing a YouTube video about glazing techniques"

Message: "2024-03-20: My friend recommended this amazing Thai restaurant called Lotus Garden near the university, they have the best pad thai"
BAD extraction:
  - recommendation_given: "As of 2024-03-20, Alex likes Thai food"  ← MISSING: restaurant name, location, specific dish, that it was a recommendation
GOOD extraction:
  - recommendation_given: "As of 2024-03-20, Alex's friend recommended a Thai restaurant called Lotus Garden near the university, known for their pad thai"

## 6. Entity Type Selection & Default Property Usage
- **Entity type (`entity_type`) MUST be one of the types defined in the schema** (e.g., person, organization). NEVER invent new entity types. Do NOT generate "episodes" type entities — the system creates them separately.
- If an entity does not perfectly match any entity type in schema, choose the **closest matching type**. For example, a pet or named animal should use "animal"; a fictional character or public figure should use "person"; a club or team should use "organization".
- **`default_property` is a PROPERTY NAME, NOT an entity type.** It is used when an entity's type is already determined, but a specific piece of information does not fit any of the defined property categories for that type.
- Each default_property value must still be a semantically complete statement following the same rule as general property values.

# Time Handling Rules (Multi-Precision Support)
**IMPORTANT**: The system supports multiple time precisions. Choose the appropriate precision based on information provided in the dialogue:

## Supported Time Formats (max precision: DAY)
1. **Year precision**: `2023` - only year is known
2. **Month precision**: `2023-05` - year and month are known
3. **Day precision**: `2023-05-24` - complete date is known

## Time Extraction Principles
1. **Preserve original precision** - DO NOT fill in unknown information
   - Dialogue says "in 2023" → use `2023`
   - Dialogue says "in May 2023" → use `2023-05`
   - Dialogue says "on May 24, 2023" → use `2023-05-24`

2. **Explicit time information**: Prioritize time explicitly mentioned in dialogue
   - "I graduated in 2023" → `2023`
   - "Joined in May 2023" → `2023-05`
   - "on March 19, 2024" → `2024-03-19`

3. **Relative time inference**: Infer based on dialogue timestamp. Use COARSER precision when uncertain.
   - The dialogue timestamp includes the day of the week (e.g., "2023-07-15 13:51:00 (Saturday)"). Use this to calculate relative dates precisely.
   - **CRITICAL — "last [weekday]" calculation rule**:
     - "last Friday" means the MOST RECENT Friday BEFORE the conversation date, NOT the Friday of the previous calendar week
     - Step-by-step: (1) Note the conversation day of week from the timestamp, (2) Count backwards to find the nearest target weekday, (3) That is the answer
     - Example: Timestamp is "2023-07-15 (Saturday)", speaker says "last Friday" → July 14 (1 day back), NOT July 7
     - Example: Timestamp is "2023-09-13 (Wednesday)", speaker says "last Monday" → Sept 11 (2 days back), NOT Sept 4
     - Example: Timestamp is "2023-02-09 (Thursday)", speaker says "last Wednesday" → Feb 8 (1 day back), NOT Feb 1
   - **"last weekend" rule**: means the most recent Saturday-Sunday before the conversation date
     - Example: Timestamp is "2023-05-24 (Wednesday)", "last weekend" → May 20-21, NOT May 13-14
   - Dialogue time is 2024-03-20, user says "yesterday" → time field: `2024-03-19`, value: "On 2024-03-19 (yesterday), ..."
   - Dialogue time is 2024-03-20, user says "last week" → time field: `2024-03`, value: "Last week from 2024-03-20, ..."
   - Dialogue time is 2024-03-20, user says "last month" → time field: `2024-02`, value: "Last month from 2024-03-20, ..."
   - Dialogue time is 2024-03-20, user says "last year" → time field: `2023`, value: "Last year from 2024-03-20, ..."
   - **NEVER fabricate a specific day from a vague relative expression** — if "last week" is said, don't guess which exact day

4. **Default to dialogue timestamp when time not mentioned**:
   - If no time is mentioned at all, use `{dialogue_timestamp}` as default (day precision max)
   - Strip any time-of-day component: "2024-03-20 14:30:00" → use `2024-03-20`

5. **Forbidden behaviors**:
   - ❌ DO NOT use "unknown" or any placeholder
   - ❌ DO NOT use datetime with hours/minutes/seconds (e.g., "2023-05-24 14:30:00")
   - ❌ DO NOT expand "2023" to "2023-01-01"
   - ❌ DO NOT expand "2023-05" to "2023-05-01"
   - ❌ DO NOT use descriptive time expressions like "before 2023", "after 2023-05", "around 2023"
   - ❌ DO NOT use "As of 2023-08-21 16:29:00, ..." format in values
   - ✅ ONLY use exact formats: `2023`, `2023-05`, `2023-05-24`
   - ✅ Keep the precision level provided in dialogue

# Property Value Rules
1. All property values MUST be strings - never use lists or dicts
2. Use the format specified in schema - each property has an example format, follow it
3. Only use properties defined in schema
4. **Keep original phrasing from dialogue** - preserve specific words, names, and details
5. **Concrete Details Priority**: When extracting information, prioritize concrete, specific facts over abstract summaries
   - Extract "lost job in January 2023" rather than "career transition"
   - Extract "visited Rome and Paris" rather than "traveled internationally"
   - Preserve specific dates, amounts, names, and locations exactly as mentioned
   - Use literal quotes when speakers use specific phrases
6. **Factual Precision**: Avoid generalizations that could lose important distinctions
   - "started dance studio because lost job" ≠ "pursuing passion for dance"
   - Both may be true, but the causal relationship is more specific and valuable
7. **Time-Sensitive Information**: Give extra care to temporal details as they are crucial for retrieval accuracy
   - For relative time mentions (yesterday, last week, etc.), preserve the original relative expression naturally in the value
   - Format: "Last week from {dialogue_timestamp}, ..." or "On 2024-03-14 (yesterday), ..."
   - Example: "Last week from 2024-03-20, Alex had a great time talking about childhood memories"
   - NEVER use format like "As of 2024-03-20 14:30:00, ..." — no time-of-day precision in values
8. **Frequency and Recurring Information**: Always preserve patterns and frequency details
   - "every Tuesday and Thursday" not just "regularly"
   - "called three times during the conversation" not just "called multiple times"
   - "usually has coffee at 8 AM" includes both the activity and timing pattern

# Message Mapping Requirements ⚠️ Critical
Before generating the final answer, you must output a message mapping dictionary `message_mapping` explaining how each message maps to which entity's properties.

## Mapping Format Requirements
```json
{
  "message_mapping": {
    "0": {
      "mappings": [
        {"entity": "Entity Name", "property": "Property Name"},
        {"entity": "Entity Name", "property": "Property Name"}
      ],
      "reason": "Mapping reason explanation"
    },
    "1": {
      "mappings": [
        {"entity": "Entity Name", "property": "Property Name"}
      ],
      "reason": "Mapping reason explanation"
    },
    "2": {
      "mappings": [],
      "reason": "No mapping reason explanation (e.g., pure congratulations, no substantial information)"
    }
  },
  "mapping_comments": "Overall mapping explanation"
}
```

## Mapping Principles
1. **Index reference**: Use message indices "0", "1", "2", "3" etc. to reference messages, indices must be consecutive starting from 0
2. **Comprehensive mapping**: One message can correspond to multiple entities and properties, must list all
3. **Exclude episodes type**: **STRICTLY FORBIDDEN to map to episodes entities**, episodes entities are used to save original dialogue and not in property extraction consideration
4. **Exclude invalid information**: Pure interjections, questions, greetings, congratulations and other messages without specific information content may not be mapped
5. **Valid information identification**: Only map messages containing concrete facts, states, events, plans and other substantial information
6. **Multiple values per property**: Same entity's same property can have multiple values from different messages - this is allowed and should be mapped separately
7. **Reason explanation**: Every message must have a reason field explaining why it maps to these properties (or why it doesn't map)

# Output Format
Output clean JSON with `message_mapping`, `entities` and `edges` top-level fields.
- **message_mapping**: Dictionary mapping message indices to entity properties as specified above
- **entities**: Each entity must have: name, entity_type, description, properties
- **edges**: Each edge must have proper link information. Edge `link_description` must describe a **factual relationship** (e.g., "works at", "owns", "lives in", "adopted from"), NOT a speech act (e.g., "asked about", "mentioned", "talked about", "congratulated on"). If the only connection between two entities is that one person asked about or mentioned the other, do NOT create an edge.
- Each property must have: property_name, value, time
- Use string values only for all properties

# Example

## Input Dialogue (timestamp: 2024-03-20)
Alice: I moved from Beijing to Shanghai yesterday, started working at Alibaba in 2023.
Bob: Congratulations! How's the work?
Alice: It's great! I'm working with Li Ming and Wang Hua on the cloud migration project.

## Correct Output (note time precision — do NOT generate episodes entities, the system handles them separately)
```json
{
  "message_mapping": {
    "0": {
      "mappings": [
        {"entity": "Alice", "property": "location_event"},
        {"entity": "Alice", "property": "position_event"}
      ],
      "reason": "Contains concrete facts about location change and work history"
    },
    "1": {
      "mappings": [],
      "reason": "Pure congratulations and question without substantial factual information"
    },
    "2": {
      "mappings": [
        {"entity": "Alice", "property": "experience"}
      ],
      "reason": "Contains information about current project work and colleagues"
    }
  },
  "entities": [
    {
      "name": "Alice",
      "entity_type": "person",
      "description": "Person who moved from Beijing to Shanghai",
      "properties": [
        {
          "property_name": "location_event",
          "value": "On 2024-03-19 (yesterday), Alice moved from Beijing to Shanghai",
          "time": "2024-03-19"
        },
        {
          "property_name": "position_event",
          "value": "In 2023, Alice started working at Alibaba",
          "time": "2023"
        },
        {
          "property_name": "experience",
          "value": "As of 2024-03-20, Alice is working with Li Ming and Wang Hua on cloud migration project",
          "time": "2024-03-20"
        }
      ]
    },
    {
      "name": "Alibaba",
      "entity_type": "organization",
      "description": "Company where Alice works",
      "properties": []
    },
    {
      "name": "Li Ming",
      "entity_type": "person",
      "description": "Alice's colleague at Alibaba",
      "properties": []
    },
    {
      "name": "Wang Hua",
      "entity_type": "person",
      "description": "Alice's colleague at Alibaba",
      "properties": []
    },
    {
      "name": "Cloud Migration Project",
      "entity_type": "project",
      "description": "Project Alice is working on with Li Ming and Wang Hua at Alibaba",
      "properties": [
        {
          "property_name": "project_member",
          "value": "As of 2024-03-20, project members are Alice, Li Ming, Wang Hua",
          "time": "2024-03-20"
        }
      ]
    }
  ],
  "edges": [
    {
      "link_entity1_name": "Alice",
      "link_entity2_name": "Alibaba",
      "link_description": "works at"
    },
    {
      "link_entity1_name": "Alice",
      "link_entity2_name": "Cloud Migration Project",
      "link_description": "working on"
    },
    {
      "link_entity1_name": "Li Ming",
      "link_entity2_name": "Alibaba",
      "link_description": "works at"
    },
    {
      "link_entity1_name": "Wang Hua",
      "link_entity2_name": "Alibaba",
      "link_description": "works at"
    }
  ]
}
```

Dialogue timestamp: {dialogue_timestamp}
Dialogue history: {chat_chunk}
"""
