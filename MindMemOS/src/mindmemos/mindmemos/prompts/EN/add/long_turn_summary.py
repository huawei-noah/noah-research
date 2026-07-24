"""Prompt for long-turn middle-section compaction."""

LONG_TURN_SUMMARY_PROMPT = """You summarize the middle section of a long conversation turn for later memory extraction.

Preserve user intent, resolved references, important entities, confirmed facts, decisions, corrections,
open questions, and warnings. Do not invent unsupported facts and do not output memory candidates.

Return one JSON object with these fields:
{
  "general_summary": "concise factual summary",
  "key_entities": ["entity"],
  "user_intent": "intent",
  "confirmed_facts": ["fact"],
  "decisions": ["decision"],
  "open_questions": ["question"],
  "warnings": ["warning"]
}
"""
