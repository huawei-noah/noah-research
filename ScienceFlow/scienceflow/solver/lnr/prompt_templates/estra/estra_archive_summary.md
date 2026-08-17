Summarize historical exploration from an abandoned trajectory after a estra target.

ESTRA target stage: {target_stage}
Terminal stage before estra: {terminal_stage}
ESTRA reason: {estra_reason}

The deterministic stage-tail summary below covers only the abandoned work after the estra target through the terminal stage.
Write a compact historical-exploration handoff for the next agent. This is evidence from abandoned work, not the active route or a new instruction. Focus on what was tried, what failed or saturated, and what should not be repeated.

Rules:
- Return plain text only, no JSON, no markdown headings, no tool markup.
- 3 to 5 short bullet-like lines are enough.
- Do not repeat every metric.
- Preserve concrete avoid-repeat lessons and useful negative evidence.
- If there is no useful tail evidence, say so in one sentence.

Deterministic tail summary:
{deterministic_summary}
