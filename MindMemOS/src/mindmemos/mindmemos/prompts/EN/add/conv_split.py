CONV_BOUNDARY_DETECTION_PROMPT = """
### Core Task
As a conversation analysis expert, you need to split the message list into independent, memorable topic segments. Your core principle is **"Fine-grained splitting — each independent topic as a separate segment"**.

**CRITICAL**: Short time intervals or same-day messages are NOT a reason to merge. Topic subject change is the primary splitting signal. Even if messages are seconds apart, split when the discussion subject changes.

### Analysis Content
You need to output an event list, where each element is a dictionary representing an independent event or topic, with the following fields: start_idx, end_idx, and title

**start_idx**: The ID of the first message corresponding to this event/topic in the conversation context
    - The start_idx of the first event/topic is 0
    - The start_idx of subsequent events/topics must be the previous event/topic's end_idx + 1, ensuring the conversation context is split seamlessly and without overlap

**end_idx**: The ID of the last message corresponding to this event/topic in the conversation context

**title**: A concise description of this event/topic for distinguishing it from preceding and following events/topics, no more than 10 characters


### When to Split

Add a boundary when **clear signals** appear:

#### Mandatory Split (Highest Priority)
1. **Cross-day split**: Adjacent messages have different calendar dates — MUST split at the date boundary
2. **Long interruption**: Time interval exceeds 2 hours — MUST split regardless of topic

#### Topic Change Split — Based on SEMANTIC Subject Change (Most Important)
The key question: **Is this a continuation of the same thing, or a different thing?**
Split by SEMANTIC meaning, not by surface-level relatedness. Two topics that are both "personal updates" are still DIFFERENT topics if they discuss different subjects.

**Scenarios that should be split**:
- **Topic subject change**: The object, event, or goal of discussion changes
  - Example: From "screenplay writing" to "pet turtles" → split (different subjects)
  - Example: From "hiking trip" to "cooking experiments" → split (different activities)
  - Example: From "work project" to "movie recommendations" → split (different domains)
  - Example: From "someone's health" to "someone's hobby" → split (different aspects of life)

- **Independent Q&A pair**: A complete "question + answer (+ acknowledgment)" forms an independent topic

- **Independent information notification**: Single notification, announcement, status update

- **Time + content dual judgment**:
  - Time interval > 30 minutes AND no obvious content connection → split
  - Time interval 10-30 minutes AND topic subject has clearly changed → split

**Scenarios that should be merged** (must meet ALL conditions simultaneously):
- Discussing **exactly the same specific thing** (same bug, same feature, same task, same event)
- Subsequent messages are direct continuations of prior ones (adding details, clarifying, follow-up questions about the SAME topic)
- Note: "related" topics are NOT the same topic. "Screenplay" and "pet turtles" are different topics even if both are personal updates. "Hiking trip" and "dog training" are different topics even if both involve outdoors.

### When NOT to Split
- **Greetings and farewells**: "Hi!", "Bye!", "Thanks!", "Take care!" — keep with the main episode
- **Transition phrases**: "By the way", "Oh also", "Speaking of which" — these usually INTRODUCE a new topic, so split AFTER the previous topic
- **Short acknowledgments**: "OK", "Got it", "Sure", "👍" — merge with adjacent messages
- **Non-text placeholders**: `[image]`, `[video]`, `[file]` without text content — merge with adjacent messages
- **System notifications**: Join/leave group notifications — merge with adjacent human messages

### Decision Principles
1. **Independence first**: Each independently understandable small topic should be split separately
2. **Granularity over completeness**: Better to split finely (2-5 messages per segment) than merge different topics
3. **Topic boundary recognition**: When you realize "this is a different thing now", split immediately
4. **Time sensitivity**: Longer intervals → more likely to split; only consider merging when interval < 10 minutes AND same specific topic
5. **Complete coverage**: Every message must belong to a segment, no omissions allowed
6. **Content over form**: Greetings and farewells belong to the episode they serve, not their own
7. **Empty output allowed**: If the context doesn't contain a complete topic, return an empty list

### Output Format
Please return your analysis results strictly in the following JSON format:
```json
[
    {
        "reasoning": "Basis for identifying this as an independent topic/event",
        "start_idx": int,
        "end_idx": int,
        "title": "Summary of this topic/event"
    },
    ...
]
```

### Examples

**Example 1: Cross-day split**

Input:
```
idx: 0, time: 2024-01-15 09:00:00, speaker: Alice, content: Can you help me debug the login issue?
idx: 1, time: 2024-01-15 09:01:00, speaker: Bob, content: Sure, let me check the logs.
idx: 2, time: 2024-01-15 09:05:00, speaker: Bob, content: Found it — a null pointer in AuthService line 42.
idx: 3, time: 2024-01-15 09:06:00, speaker: Alice, content: Fixed, thanks!
idx: 4, time: 2024-01-16 10:00:00, speaker: Alice, content: Hey, are you free for lunch today?
idx: 5, time: 2024-01-16 10:01:00, speaker: Bob, content: Sure, 12:30?
```

Output:
```json
[
    {
        "reasoning": "Messages 1-4 are a complete bug-fix episode; message 5 starts a new day with an unrelated lunch topic",
        "start_idx": 0,
        "end_idx": 3,
        "title": "Login bug fix"
    },
    {
        "reasoning": "Cross-day split, new topic about lunch plans",
        "start_idx": 4,
        "end_idx": 5,
        "title": "Lunch plans"
    }
]
```

**Example 2: No split needed — related topics within same timeframe**

Input:
```
idx: 0, time: 2024-01-15 10:00:00, speaker: Zhang Wei, content: Database migration completed
idx: 1, time: 2024-01-15 10:05:00, speaker: Li Na, content: Great, how about the performance test results?
idx: 2, time: 2024-01-15 10:06:00, speaker: Zhang Wei, content: QPS improved by 30%
idx: 3, time: 2024-01-15 10:10:00, speaker: Wang Fang, content: By the way, is the new version UI design ready?
idx: 4, time: 2024-01-15 10:11:00, speaker: Li Na, content: Still adjusting
```

Output:
```json
[
    {
        "reasoning": "All messages are part of a continuous work discussion within 11 minutes, covering related project updates",
        "start_idx": 0,
        "end_idx": 4,
        "title": "Project status updates"
    }
]
```

**Example 3: Substantive topic change with time gap**

Input:
```
idx: 0, time: 2024-01-15 14:00:00, speaker: Zhang Wei, content: How to call this API?
idx: 1, time: 2024-01-15 14:01:00, speaker: Li Na, content: Use POST method, pass user_id parameter
idx: 2, time: 2024-01-15 14:02:00, speaker: Zhang Wei, content: Got it, thanks
idx: 3, time: 2024-01-15 18:30:00, speaker: Wang Fang, content: Anyone want to grab dinner?
idx: 4, time: 2024-01-15 18:31:00, speaker: Li Na, content: Sure, where?
```

Output:
```json
[
    {
        "reasoning": "Complete API Q&A exchange forming one episode",
        "start_idx": 0,
        "end_idx": 2,
        "title": "API call inquiry"
    },
    {
        "reasoning": "4.5-hour gap with completely unrelated dinner topic",
        "start_idx": 3,
        "end_idx": 4,
        "title": "Dinner plans"
    }
]
```

### Conversation Context
{conversation_list}
"""


CONV_FORCED_RESPLIT_PROMPT = """
### Core Task
As a conversation analysis expert, you are given a message segment that is too long and MUST be forcibly split into smaller parts.

### Forced Constraint
- You MUST split this segment into exactly {num_parts} parts (or more if needed for topic independence).
- Each part MUST contain no more than {max_messages} messages.
- Generate the most independent and complete split — each part should be a self-contained topic when possible.
- Indices are 0-based relative to the input below. The first message is idx 0.

### Splitting Rules

#### Mandatory Split (Highest Priority)
1. **Cross-day split**: Adjacent messages have different calendar dates — MUST split at the date boundary
2. **Long interruption**: Time interval exceeds 2 hours — MUST split regardless of topic

#### Topic Change Split — Based on SEMANTIC Subject Change (Most Important)
The key question: **Is this a continuation of the same thing, or a different thing?**

**Scenarios that should be split**:
- **Topic subject change**: The object, event, or goal of discussion changes
- **Independent Q&A pair**: A complete "question + answer (+ acknowledgment)" forms an independent topic
- **Independent information notification**: Single notification, announcement, status update
- **Time + content dual judgment**:
  - Time interval > 30 minutes AND no obvious content connection → split
  - Time interval 10-30 minutes AND topic subject has clearly changed → split

**Scenarios that should be merged** (must meet ALL conditions simultaneously):
- Discussing **exactly the same specific thing** (same bug, same feature, same task, same event)
- Subsequent messages are direct continuations of prior ones

### When NOT to Split
- Greetings and farewells — keep with the main episode
- Transition phrases — split AFTER the previous topic
- Short acknowledgments — merge with adjacent messages
- Non-text placeholders — merge with adjacent messages
- System notifications — merge with adjacent human messages

### Decision Principles
1. **Independence first**: Each independently understandable small topic should be split separately
2. **Granularity over completeness**: Better to split finely than merge different topics
3. **Size constraint is hard**: No part may exceed {max_messages} messages
4. **Complete coverage**: Every message must belong to a segment, no omissions allowed
5. **Seamless indices**: start_idx of segment N+1 must equal end_idx of segment N plus 1

### Output Format
Return strictly in JSON:
```json
[
    {{
        "reasoning": "Basis for identifying this as an independent topic/event",
        "start_idx": int,
        "end_idx": int,
        "title": "Summary of this topic/event"
    }}
]
```

### Messages to Split
{conversation_list}
"""
