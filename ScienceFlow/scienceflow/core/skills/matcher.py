# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

from scienceflow.core.skills.base import Skill
from scienceflow.core.skills.registry import SkillRegistry

AGENT_SKILL_PHASES: dict[str, list[str]] = {
    "draft":    ["analysis", "drafting"],
    "improve":  ["improving"],
    "debug":    ["debugging"],
    "validate": ["evaluation"],
    "monitor":  ["monitoring"],
    "assistant": ["analysis", "drafting", "qa"],  # legacy alias; QA uses ScienceAgent
    "qa":       ["analysis", "drafting", "qa"],    # Alias for assistant
}

# lnr BudgetManager phases -> skill library ``phase`` frontmatter tags
LNR_PHASE_SKILL_MAP: dict[str, list[str]] = {
    "explore": ["analysis", "drafting"],
    "exploit_explore": ["improving"],
    "ensemble": ["improving"],
    "exploit": ["improving"],  # legacy LNR third-phase name
}


class SkillMatcher:
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def match(
        self,
        task_type: str,
        phase: str | list[str],
        tags: list[str] | None = None,
        max_skills: int = 5,
    ) -> list[Skill]:
        """Filter by task_type, phase(s), optional tags. Sort by priority desc.

        *phase* can be a single string or a list. A skill matches if any of its
        declared phases intersects with the queried phase(s). Non-executable
        skills only are returned (executable skills are for tool agents that run code).
        """
        candidates = self.registry.get_all()

        filtered = [s for s in candidates if s.matches_task_type(task_type)]

        phases = [phase] if isinstance(phase, str) else phase
        filtered = [
            s for s in filtered if any(s.matches_phase(p) for p in phases)
        ]
        filtered = [s for s in filtered if not s.metadata.executable]

        if tags:
            tag_set = set(tags)
            filtered = [s for s in filtered if tag_set & set(s.metadata.tags)]

        ranked = sorted(filtered, key=lambda s: s.metadata.priority, reverse=True)
        return ranked[:max_skills]

    def match_for_agent(
        self,
        agent_phase: str,
        task_type: str = "*",
        tags: list[str] | None = None,
        max_skills: int = 5,
    ) -> list[Skill]:
        """Convenience: resolve agent_phase via AGENT_SKILL_PHASES mapping."""
        phases = AGENT_SKILL_PHASES.get(agent_phase, [agent_phase])
        return self.match(task_type, phases, tags=tags, max_skills=max_skills)
