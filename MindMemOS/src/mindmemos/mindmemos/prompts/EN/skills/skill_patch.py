"""Skill-patch prompts: propose a patch from summaries, then apply it.

Adapted from ``skill_rl.evolve.prompts`` (``propose_multi_from_summary_user`` +
the skill-quality principles). The offline algorithm pools rollouts of ONE task;
online we cannot re-run a task, so the proposer reasons ACROSS several different
tasks that all injected the same skill and keeps only what generalizes.

Three stages:

- ``PROPOSE_PATCH_SYSTEM`` / ``PROPOSE_PATCH_SCORED_SYSTEM`` /
  :func:`propose_patch_user` -> a human-readable patch describing the minimal,
  general edits to make. The scored variant is selected when the batch carries
  trajectory evaluation scores, so the proposer can reinforce behaviors from
  high-score sessions and discourage those recurring in low-score ones; without
  scores it falls back to the unsupervised variant.
- ``APPLY_PATCH_SYSTEM`` / :func:`apply_patch_user` -> a small list of
  LINE-ADDRESSED edit ops (replace / insert / delete by line number) that are
  applied deterministically by ``components.skill.edit``. The model is shown the
  SKILL.md with a ``N|`` gutter and emits line numbers, the NEW text, and a short
  prefix of the addressed old line as a guard against line-number mistakes.
- ``REWRITE_SKILL_SYSTEM`` / :func:`rewrite_skill_user` -> an optional
  format-repair pass, gated by config.
"""

from __future__ import annotations

# Three principles that distinguish a high-quality skill edit from vague advice
# (verbatim intent from skill_rl's SKILL_QUALITY_PRINCIPLES).
_SKILL_QUALITY_PRINCIPLES = (
    "Follow these three principles for a GOOD skill edit:\n"
    "1. Failure Mechanism Encoding -- explain WHY an agent fails, not just that "
    "it should be careful. State the concrete mechanism, not a vague 'remember to "
    "validate'.\n"
    "2. Actionable Specificity -- give an executable action, not an attitude. "
    "Every warning should come with the concrete step that avoids it.\n"
    "3. High-Risk Action Blacklist -- explicitly forbid dangerous behaviors. Name "
    "the prohibited action and the failure it causes."
)

PROPOSE_PATCH_SYSTEM = (
    "You maintain a reusable SKILL.md that guides an autonomous agent on a class "
    "of tasks. You are given the current skill and a batch of analytical "
    "summaries drawn from SEVERAL DIFFERENT real sessions that all used this "
    "skill. There is no success/failure label.\n\n"
    "Treat the batch as field observations. Reading ACROSS the different tasks, "
    "infer recurring behaviors that look reliably helpful, mistakes or dead-ends "
    "that show up repeatedly, and missing general guidance that would make FUTURE "
    "agents more reliable. Favor patterns that GENERALIZE across tasks; never "
    "overfit to one task's values, filenames, contents, or exact answers.\n\n"
    "Propose a MINIMAL, GENERAL patch as a human-readable change plan: a short "
    "list of concrete edits (add / revise / remove guidance), each with the "
    "exact text to add or change and a one-line rationale. If the current skill "
    "already covers the useful lessons, say so and propose NO edits.\n\n" + _SKILL_QUALITY_PRINCIPLES
)


PROPOSE_PATCH_SCORED_SYSTEM = (
    "You maintain a reusable SKILL.md that guides an autonomous agent on a class "
    "of tasks. You are given the current skill and a batch of analytical "
    "summaries drawn from SEVERAL DIFFERENT real sessions that all used this "
    "skill. EACH summary is LABELED with a trajectory evaluation score: higher "
    "means the session went better, lower means it went worse. Scores are "
    "comparable within this batch; treat them as relative reward, not absolute "
    "grades.\n\n"
    "Use the scores as your PRIMARY signal. Reading ACROSS the different tasks, "
    "identify behaviors that recur in HIGH-score sessions and reinforce them as "
    "guidance, and identify mistakes, dead-ends, or risky actions that recur in "
    "LOW-score sessions and add guidance that steers FUTURE agents away from "
    "them. When a behavior appears in both high- and low-score sessions, it does "
    "not discriminate outcome -- do not encode it. Favor patterns that GENERALIZE "
    "across tasks; never overfit to one task's values, filenames, contents, or "
    "exact answers, and never hard-code a single session's score into the skill.\n\n"
    "Propose a MINIMAL, GENERAL patch as a human-readable change plan: a short "
    "list of concrete edits (add / revise / remove guidance), each with the "
    "exact text to add or change and a one-line rationale tied to the score "
    "evidence (e.g. 'recurs in low-score sessions'). If the current skill "
    "already covers the lessons the scores point to, say so and propose NO "
    "edits.\n\n" + _SKILL_QUALITY_PRINCIPLES
)


def propose_patch_user(
    skill_name: str,
    skill_md: str,
    summaries: list[str],
    scores: list[float | None] | None = None,
) -> str:
    """Handle propose patch user."""

    blocks = []
    for i, summary in enumerate(summaries, start=1):
        score = scores[i - 1] if scores is not None and i - 1 < len(scores) else None
        header = f"## Observation {i}"
        if score is not None:
            header += f" (score: {score:g})"
        blocks.append(f"{header}\n{summary.strip()}")
    joined = "\n\n".join(blocks) if blocks else "(no summaries)"
    has_scores = scores is not None and any(s is not None for s in scores)
    signal = "Using the scores as the PRIMARY signal" if has_scores else "Using the summaries as the PRIMARY signal"
    return (
        f"# Skill name\n{skill_name}\n\n"
        f"# Current SKILL.md\n{skill_md}\n\n"
        f"# Trajectory summaries from {len(summaries)} different sessions\n{joined}\n\n"
        f"{signal}, propose a minimal, general "
        "change plan for SKILL.md per your instructions. If nothing worth changing "
        "recurs across the sessions, state that no edits are needed."
    )


APPLY_PATCH_SYSTEM = (
    "You apply an approved change plan to a SKILL.md file using LINE-ADDRESSED "
    "EDIT OPERATIONS instead of rewriting the whole document. You are given the "
    "current SKILL.md with a line-number gutter (each line is prefixed by its "
    "number and a '|', e.g. '12| - Validate input'). The gutter is NOT part of "
    "the document -- never reproduce it in your output.\n\n"
    "Output a single JSON object with an 'edits' array. Each edit is one of:\n"
    '  {"op": "replace", "start": <line>, "end": <line>, "new": "<replacement text>", "old_string_prefix": "<first 40-120 chars of line `start` without its number>"}\n'
    '  {"op": "delete",  "start": <line>, "end": <line>, "old_string_prefix": "<first 40-120 chars of line `start` without its number>"}\n'
    '  {"op": "insert",  "after": <line>, "new": "<text>"}\n\n'
    "Line numbers are 1-based and INCLUSIVE: replace/delete with start=6,end=8 "
    "act on lines 6, 7 and 8. 'insert' places 'new' AFTER the given line; use "
    '"after": 0 to prepend at the very top, and "after": <last line number> to '
    "append at the end.\n\n"
    "Rules:\n"
    "1. Reference line numbers from the gutter. On replace/delete, include "
    "'old_string_prefix' -- a short prefix copied from the line at 'start' "
    "(without its 'N| ' prefix) to catch a wrong number. Use enough text to "
    "identify the line, usually 40-120 characters; do NOT copy a very long line "
    "in full. If the prefix does not match, you will be asked to fix the number.\n"
    "2. Apply exactly the edits in the plan; do not invent unrelated changes and "
    "do not touch the frontmatter unless the plan says so.\n"
    "3. Prefer the smallest edit that does the job: replace one line or a short "
    "run of lines, insert a new bullet after a related one, delete an obsolete "
    "line. Keep the result coherent, non-redundant, and concise.\n"
    "4. 'new' is the literal replacement/inserted text WITHOUT any line-number "
    "prefix. Write it as one or more whole lines; for multiple lines embed '\\n' "
    "between them. The system owns line separators -- it places 'new' on its own "
    "line(s), so you do NOT need leading/trailing newlines to avoid fusing onto "
    "neighbors. To insert a blank line (e.g. before a new '## ' heading), include "
    "an empty line inside 'new' (e.g. \"\\n## Blacklist\\n- ...\").\n"
    "5. replace/delete ranges MUST NOT overlap each other. To both change a line "
    "and add nearby guidance, use one 'replace' plus a separate 'insert'.\n"
    "6. To add a new bullet to a list, 'insert' after the last existing bullet's "
    "line. To add a brand-new section, 'insert' after the line where it belongs "
    '(or "after": <last line> for the end).\n'
    '7. If the plan says no edits are needed, return {"edits": []}.\n'
    "Output ONLY the JSON object -- no commentary and no markdown code fences."
)


def apply_patch_user(skill_md: str, patch: str) -> str:
    """Build the apply prompt from the current skill and the proposed patch.

    The SKILL.md is rendered with a line-number gutter so the model can address
    edits by line; :func:`mindmemos.components.skill.edit.format_numbered` produces
    the same numbering the applier uses.
    """

    from ....components.skill import format_numbered

    return (
        f"# Current SKILL.md (with line-number gutter; the 'N| ' prefix is NOT part of the file)\n"
        f"{format_numbered(skill_md)}\n\n"
        f"# Change plan to apply\n{patch}\n\n"
        "Return the JSON 'edits' object that applies this change plan to the "
        "SKILL.md above. Reference lines by their gutter numbers and put the new "
        "text in 'new'; include 'old_string_prefix' on every replace/delete."
    )


REWRITE_SKILL_SYSTEM = (
    "You are a Markdown format-repair editor for a SKILL.md file. Your ONLY job "
    "is to repair presentation damage introduced by automated edits: missing "
    "newlines between bullets or sentences, bullets fused onto previous lines, "
    "headings fused onto paragraphs, malformed list spacing, and accidental "
    "paragraph wrapping problems.\n\n"
    "This is NOT a content rewrite. Treat every existing instruction as frozen "
    "text. Do not improve, simplify, deduplicate, merge, reorder, reinterpret, "
    "or remove guidance. Do not add new guidance, examples, tools, strategies, "
    "warnings, rationales, headings, or sections. Do not change terminology or "
    "wording except for the minimum punctuation/whitespace needed to separate "
    "already-present text into valid Markdown lines.\n\n"
    "Rules:\n"
    "1. Preserve the complete YAML frontmatter exactly, including all fields and "
    "values.\n"
    "2. Preserve code blocks exactly, including language tags and code text. You "
    "may only add missing blank lines before or after a code block if needed.\n"
    "3. Split fused bullets such as 'range.- Be careful' into separate Markdown "
    "lines, preserving the words of both pieces.\n"
    "4. Split fused headings such as 'rows.## Data Entry' so the heading starts "
    "on its own line, preserving the heading text.\n"
    "5. Split fused sentences only when the missing boundary is clear. Preserve "
    "the sentence wording exactly.\n"
    "6. Keep the original order of all guidance. If two adjacent instructions "
    "repeat each other, leave both in place.\n"
    "7. Output the complete SKILL.md after format repair only.\n\n"
    "Output ONLY the full SKILL.md text -- no commentary, no JSON, no markdown "
    "code fences around the whole document."
)


def rewrite_skill_user(skill_md: str) -> str:
    """Build the optional reformatting prompt for a patched skill."""

    return (
        f"# SKILL.md to reformat\n{skill_md}\n\n"
        "Return the complete SKILL.md after a format-only repair pass. Do not "
        "add, delete, merge, deduplicate, reorder, summarize, or rewrite any "
        "instruction; only fix Markdown line breaks, list boundaries, heading "
        "boundaries, and spacing."
    )
