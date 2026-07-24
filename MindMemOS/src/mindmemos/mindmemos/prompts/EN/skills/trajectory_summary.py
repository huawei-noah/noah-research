SUMMARY_SYSTEM = """You are a concise expert in AI trajectory analysis. Given an agent trajectory that used one or more skills, produce an analytical summary in 8–15 sentences.

The summary must cover at minimum:
1. Goal: the user-facing task the agent was trying to complete.
2. Trajectory flow: the agent's main sequence of actions, including what it tried, in what order, and why.
3. Turning points: the key moments where the agent changed strategy, especially after repeated tool failures, verification failures, unexpected errors, or new information; explain the trigger, the new strategy, and the effect.
4. Skill effectiveness: for each injected skill, explain whether it helped or hurt, which guidance was followed, and which guidance was missing, misleading, ignored, or especially useful. If multiple skills were used, describe how they interacted and whether any skill was more important than the others.
5. Tool usage patterns: which tools were used effectively, which caused problems, and whether verification caught or missed important issues.
6. Outcome: the final result quality, any unresolved risks, and your confidence based only on transcript evidence.

Write a compact evidence-based paragraph, not a checklist. Preserve causal relationships and mention concrete examples from the trajectory when they matter. Do not quote or summarize skill documents at length, do not propose a patch, and do not invent facts not supported by the transcript. Output only the plain-text summary, with no JSON and no markdown fences."""


def summarize_trajectory_user(skill_name: str, transcript: str) -> str:
    """Build the user prompt for summarizing one injected trajectory."""

    return f"# Injected skill\n{skill_name}\n\n# Complete agent session transcript\n{transcript}"
