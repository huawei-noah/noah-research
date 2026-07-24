EPISODE_INFERENCE_PROMPT = """
You are a personal memory inference expert. Based on the following conversation, generate conservative inferences about the participants' behaviors, preferences, plans, and relationships that are NOT explicitly stated but can be reasonably derived.

Conversation timestamp: {conversation_timestamp}
{previous_context}
Conversation content:
{conversation_text}

## Inference Guidelines

1. **Conservative Approach**: Use hedging language ("may", "likely", "possibly", "appears to") for uncertain inferences. Only state as fact what is directly supported by evidence.
2. **Actionable Predictions**: Focus on inferences that would be useful for future memory retrieval — things a user might ask about later.
3. **Categories to Consider**:
   - **Behavioral predictions**: What might the participants do next based on stated plans or patterns? (e.g., "Andrew may visit the dog shelter again in the coming weeks given his expressed interest")
   - **Preference inferences**: What preferences or tastes can be inferred? (e.g., "Audrey likely prefers small dog breeds based on her current pets")
   - **Relationship dynamics**: What can be inferred about the relationship between participants? (e.g., "Andrew and Audrey appear to share a close friendship with regular communication")
   - **Temporal predictions**: When might related events happen? (e.g., "The next dog grooming session may occur in approximately 4-6 weeks")
   - **Emotional/motivational inferences**: What drives the participants? (e.g., "Audrey's dogs appear to be a primary source of emotional well-being for her")

4. **Quality Rules**:
   - Each inference must be traceable to specific evidence in the conversation
   - Do NOT repeat facts already stated in the conversation — only produce NEW inferences
   - Generate 3-8 inferences, each in one concise sentence (max 30 words per inference)
   - Focus on inferences most likely to be relevant in future conversations

5. **Retrieval-Friendly Format**: Write each inference as a self-contained statement that can be independently searched and matched. Include participant names explicitly.

## Output Format
Return a plain text block with one inference per line, prefixed with "- ". Example:
- Andrew may be considering adopting a pet in the near future based on his questions about dog ownership
- Audrey likely grooms her dogs herself at home on a regular schedule

## Inferences:
"""
