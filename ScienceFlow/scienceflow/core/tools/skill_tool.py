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

"""Tool for querying the Markdown skill library at runtime."""

from __future__ import annotations

from pathlib import Path

from deepcraft_core.tool import BaseTool, ToolResult
from pydantic import ConfigDict, Field

from scienceflow.core.skills.base import Skill
from scienceflow.core.skills.registry import SkillRegistry


class SkillTool(BaseTool):
    name: str = "skill"
    description: str = (
        "Query project skills for guidance on data loading, preprocessing, "
        "feature engineering, modeling, and training."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read"],
                "description": "list skills or read one skill",
            },
            "name": {
                "type": "string",
                "description": "Skill name for action=read",
            },
            "category": {
                "type": "string",
                "description": "Filter category for action=list",
            },
            "tag": {
                "type": "string",
                "description": "Filter tag for action=list",
            },
        },
        "required": ["action"],
    }
    registry: SkillRegistry = Field(...)
    task_type: str | None = Field(default=None)
    mode: str = Field(default="all")
    allow_names: tuple[str, ...] = Field(default_factory=tuple)
    allow_generic_wildcard: bool = Field(default=True)
    visible_max: int = Field(default=0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _category_only_mode(self) -> bool:
        return str(self.mode or "").strip().lower() == "category_only"

    def _allowed_name_set(self) -> set[str]:
        names: set[str] = set()
        for raw in self.allow_names or ():
            name = str(raw or "").strip()
            if not name:
                continue
            skill = self.registry.get_by_name_or_alias(name) or self.registry.get_by_name(name)
            names.add(skill.metadata.name if skill is not None else name)
        return names

    def _is_name_allowed(self, skill: Skill) -> bool:
        allowed = self._allowed_name_set()
        if allowed:
            return skill.metadata.name in allowed
        return not self._category_only_mode()

    def _matches_task_type(self, skill: Skill) -> bool:
        tt = (self.task_type or "").strip()
        if not tt:
            return True
        task_types = list(skill.metadata.task_types or [])
        if tt in task_types:
            return True
        return bool(self.allow_generic_wildcard and "*" in task_types)

    @staticmethod
    def _format_skill_path(path: Path | None) -> str:
        if path is None:
            return "<unknown>"
        return str(path).replace("\\", "/")

    @staticmethod
    def _format_one_line(skill: Skill) -> str:
        meta = skill.metadata
        tags = ",".join(meta.tags[:5]) if meta.tags else "-"
        aliases = ",".join(meta.aliases[:5]) if meta.aliases else "-"
        suffix = " [executable]" if meta.executable else ""
        if meta.description:
            desc = meta.description.strip()
        else:
            desc = "(no description)"
            for line in skill.content.splitlines():
                stripped = line.strip().lstrip("#").strip()
                if stripped:
                    desc = stripped
                    break
        return (
            f"- {meta.name}{suffix} | category={meta.category or '-'} "
            f"| phase={','.join(meta.phase) if meta.phase else '-'} "
            f"| tags={tags} | aliases={aliases} | {desc}"
        )

    def _list_candidates(self, *, category: str, tag: str) -> list[Skill]:
        allowed = self._allowed_name_set()
        if self._category_only_mode() and not allowed:
            skills: list[Skill] = []
        elif allowed:
            skills = [self.registry.get_by_name(name) for name in sorted(allowed)]
            skills = [s for s in skills if s is not None]
        else:
            skills = self.registry.get_all()
        if category:
            skills = [s for s in skills if (s.metadata.category or "") == category]
        if tag:
            skills = [s for s in skills if tag in (s.metadata.tags or [])]
        skills = [s for s in skills if self._matches_task_type(s)]
        skills.sort(key=lambda s: (-int(s.metadata.priority), s.metadata.name.lower()))
        limit = int(self.visible_max or 0)
        if limit > 0:
            skills = skills[:limit]
        return skills

    def _list_header(self, *, count: int, category: str, tag: str) -> str:
        allowed = ",".join(sorted(self._allowed_name_set())) or "-"
        mode = str(self.mode or "all").strip() or "all"
        return (
            f"count={count} mode={mode} task_type={self.task_type or '-'} "
            f"allowlist={allowed} category={category or '-'} tag={tag or '-'}"
        )

    async def execute(
        self,
        *,
        action: str,
        name: str = "",
        category: str = "",
        tag: str = "",
        **kwargs,
    ) -> ToolResult:
        kwargs.pop("config", None)
        act = (action or "").strip().lower()
        if act not in ("list", "read"):
            return ToolResult(error="skill.action must be one of: list, read")

        if act == "list":
            cat = category.strip()
            tg = tag.strip()
            skills = self._list_candidates(category=cat, tag=tg)
            lines = ["[skills]", self._list_header(count=len(skills), category=cat, tag=tg)]
            if not skills:
                lines.append("No skills matched filters.")
            else:
                lines.extend(self._format_one_line(s) for s in skills)
            return ToolResult(output="\n".join(lines))

        skill_name = name.strip()
        if not skill_name:
            return ToolResult(error="skill.read requires non-empty 'name'")
        matches = self.registry.get_name_matches(skill_name)
        if len(matches) > 1:
            return ToolResult(
                error=(
                    f"Ambiguous skill name: {skill_name}. "
                    f"Matched multiple skills: {', '.join(matches[:5])}. "
                    "Please use an exact skill name from `skill list`."
                )
            )
        skill = self.registry.get_by_name_or_alias(skill_name)
        if skill is None or len(matches) == 0:
            suggestions = self.registry.suggest(skill_name, top_k=3)
            if suggestions:
                return ToolResult(
                    error=(
                        f"Skill not found: {skill_name}. "
                        f"Did you mean: {', '.join(suggestions)}?"
                    )
                )
            return ToolResult(error=f"Skill not found: {skill_name}")
        if not self._is_name_allowed(skill):
            return ToolResult(
                error=(
                    f"Skill '{skill_name}' is not enabled for this task in "
                    f"{self.mode or 'all'} mode.\n"
                    f"enabled_skill={','.join(sorted(self._allowed_name_set())) or '-'} "
                    f"task_type={self.task_type or '-'}"
                )
            )
        if not self._matches_task_type(skill):
            return ToolResult(
                error=(
                    f"Skill '{skill_name}' is not applicable to task_type={self.task_type!r}"
                ),
            )
        rendered = skill.render()
        meta = skill.metadata
        exec_note = (
            "\n[note] This skill is marked executable in framework flows; "
            "read-only reference here.\n"
            if meta.executable
            else ""
        )
        header = (
            f"[skill:{meta.name}]\n"
            f"category={meta.category or '-'}\n"
            f"category_label={meta.category_label or '-'}\n"
            f"phase={','.join(meta.phase) if meta.phase else '-'}\n"
            f"priority={meta.priority}\n"
            f"tags={','.join(meta.tags) if meta.tags else '-'}\n"
            f"path={self._format_skill_path(skill.path)}\n"
        )
        return ToolResult(output=header + exec_note + "\n" + rendered)
