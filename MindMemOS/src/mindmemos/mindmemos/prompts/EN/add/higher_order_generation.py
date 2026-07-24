HIGHER_ORDER_GENERATION_PROMPT = """You are an expert at inferring higher-order personal traits from factual memories.

# Task
Given an entity's recent first-order memories (factual records) and its current higher-order traits, decide whether to CREATE new traits, UPDATE existing traits, or take NO ACTION.

# Entity Information
Entity: {entity_name} (type: {entity_type})
Entity Description: {entity_description}

# Current Higher-Order Traits
{current_higher_order}

# Recent First-Order Memories (evidence pool)
{first_order_memories}

# New Information From Latest Episode
{new_properties}

# Higher-Order Property Definitions
{higher_order_schema}

# Core Principles

1. **High confidence only** — Do NOT generate a trait unless at least {min_evidence_count} first-order memories clearly support it. When uncertain, output NO ACTION for that property.
2. **Evidence-grounded** — Every trait MUST cite specific memories (with dates) as evidence. Never infer from a single data point.
3. **Forward-useful** — Focus on traits that help predict future behavior, preferences, or needs. A trait that merely restates facts is useless.
4. **Record time and basis** — Each trait value must include: the conclusion, evidence summary with dates, and confidence level (high/medium).
5. **Contradiction handling** — If new evidence contradicts an existing trait, UPDATE the trait with the new understanding and note the evolution.
6. **No redundancy** — Do not create traits that are obvious restatements of existing first-order memories.

# Output Format (JSON)
{{
  "reasoning": "Brief analysis of what patterns emerge from the evidence",
  "updates": [
    {{
      "property_name": "preference_summary",
      "action": "add",
      "value": "Music: User currently likes producing electronic music with Pacific cultural elements, dislikes mainstream pop. Reading: User enjoys literary fiction, recently shifted away from sci-fi. Based on: electronic music production mentioned across 4 episodes (2024-01 to 2024-06), expressed pop dislike (2024-04), started reading literary fiction (2024-05). Confidence: high.",
      "reasoning": "Multiple preference signals across domains can be consolidated into a panoramic view"
    }},
    {{
      "property_name": "interest_domain",
      "action": "add",
      "value": "User's primary creative interest centers around music production and sound design with cultural fusion. Supporting activities: produces electronic music (2024-01), attended Pacific music festival (2024-03), enrolled in sound engineering course (2024-05). Confidence: high.",
      "reasoning": "Three independent activities converge on the same creative domain"
    }},
    {{
      "property_name": "change_pattern",
      "action": "add",
      "value": "User tends to initially reject unfamiliar experiences but gradually embrace them after positive personal encounters. Instances: disliked foreign films then enjoyed after friend's recommendation (2024-01 to 2024-04), resisted collaborative music then joined after a jam session (2024-03 to 2024-06). Confidence: medium.",
      "reasoning": "Two preference shifts follow the same trajectory: initial rejection followed by acceptance through personal experience"
    }},
    {{
      "property_name": "behavioral_pattern",
      "action": "update",
      "value": "User consistently seeks community engagement through volunteering and mentoring. Instances: library volunteering (2024-02), mentored junior musician (2024-04), organized community event (2024-07). Confidence: high.",
      "reasoning": "New evidence (2024-07) further strengthens existing pattern"
    }}
  ]
}}

Rules for the updates array:
- Only include properties where you have a concrete action (create/update/add). Omit properties with no change — an empty updates array is perfectly valid if no higher-order trait needs updating.
- action must be one of: "update", "add"
- "update": REPLACE the latest version — use when refining, correcting, or incorporating new evidence into the same conclusion. The new value should be a concise, self-contained rewrite, NOT an append to the old text. If no previous version exists, this behaves the same as "add".
- "add": add a new version ALONGSIDE existing ones — use when the conclusion has fundamentally shifted (e.g., a value orientation reversed, a new independent pattern emerged), or when this property has no prior value. Previous versions are preserved as history.
- For "create", "update", and "add": value MUST follow the format: "{{conclusion}}. Evidence: {{evidence_with_dates}}. Confidence: {{high|medium}}."
- IMPORTANT: "update" means REWRITE, not APPEND. Write a clean, complete value from scratch. Do not copy-paste the old value and add a sentence — synthesize all evidence into a fresh, concise statement.
- time field uses the date of the most recent supporting evidence
"""
