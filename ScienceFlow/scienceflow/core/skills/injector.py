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


class SkillInjector:
    PLACEHOLDER = "{skill_context}"

    def inject_to_prompt(self, prompt: str, skills: list[Skill]) -> str:
        """Replace PLACEHOLDER with concatenated skill content."""
        if not skills:
            return prompt.replace(self.PLACEHOLDER, "")
        skill_text = "\n\n".join(
            f"## Skill: {s.metadata.name}\n{s.content}" for s in skills
        )
        context = f"# Recommended Skills\n{skill_text}"
        return prompt.replace(self.PLACEHOLDER, context)

    async def post_process(
        self, code: str, skills: list[Skill], context: dict
    ) -> str:
        """Execute post-process hooks for skills that have them."""
        for skill in skills:
            if skill.metadata.has_post_process:
                code = await self._run_hook(skill, code, context)
        return code

    async def _run_hook(self, skill: Skill, code: str, context: dict) -> str:
        return code
