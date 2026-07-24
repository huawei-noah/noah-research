"""English prompts for the skill self-evolution pipeline (design `docs/skill`).

Three LLM stages back the ``SkillEvolver`` (see
``mindmemos.pipelines.skill.evolution``):

- ``trajectory_summary``: condense one injected ``/v1/memory/add`` trajectory into
  an analytical summary stored 1:1 with the add trace.
- ``skill_patch``: aggregate several trajectory summaries against the current
  ``SKILL.md`` and propose a minimal, general patch, then apply that patch.
- the optional ``rewrite`` stage re-formats the patched skill for clarity.

The prompts are adapted from the offline ``skill-rl`` ``trace_v2_summary``
algorithm. The key online difference: offline pools many rollouts of the SAME
task; online cannot re-run one task, so we aggregate summaries across DIFFERENT
tasks that all injected the same skill.
"""

from .skill_patch import (
    APPLY_PATCH_SYSTEM,
    PROPOSE_PATCH_SCORED_SYSTEM,
    PROPOSE_PATCH_SYSTEM,
    REWRITE_SKILL_SYSTEM,
    apply_patch_user,
    propose_patch_user,
    rewrite_skill_user,
)
from .trajectory_summary import SUMMARY_SYSTEM, summarize_trajectory_user

__all__ = [
    "APPLY_PATCH_SYSTEM",
    "PROPOSE_PATCH_SCORED_SYSTEM",
    "PROPOSE_PATCH_SYSTEM",
    "REWRITE_SKILL_SYSTEM",
    "SUMMARY_SYSTEM",
    "apply_patch_user",
    "propose_patch_user",
    "rewrite_skill_user",
    "summarize_trajectory_user",
]
