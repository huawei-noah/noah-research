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

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SkillMetadata:
    name: str = ""
    description: str = ""
    category: str = ""
    category_label: str = ""
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)
    priority: int = 50
    has_post_process: bool = False
    executable: bool = False
    max_retries: int = 4
    # When True, ``execute_skill`` keeps ReAct rounds after a successful bash/python
    # block until the workspace script is non-empty (or max rounds), instead of
    # stopping after the first successful execution (prep-style).
    exploratory_loop: bool = False
    context_vars: list[str] = field(default_factory=list)


@dataclass
class Skill:
    metadata: SkillMetadata
    content: str
    path: Path | None = None

    @classmethod
    def from_file(cls, path: Path) -> Skill:
        """Parse a .md skill file with YAML frontmatter."""
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if match:
            meta_dict = yaml.safe_load(match.group(1))
            content = match.group(2)
        else:
            meta_dict = {}
            content = text
        metadata = SkillMetadata(
            **{k: v for k, v in meta_dict.items() if k in SkillMetadata.__dataclass_fields__}
        )
        return cls(metadata=metadata, content=content, path=path)

    def render(self, context: dict[str, str] | None = None) -> str:
        """Render skill content, substituting {var} placeholders from context."""
        text = self.content
        if context:
            for k, v in context.items():
                text = text.replace("{" + k + "}", str(v))
        return text

    def matches_task_type(self, task_type: str) -> bool:
        return "*" in self.metadata.task_types or task_type in self.metadata.task_types

    def matches_phase(self, phase: str) -> bool:
        return phase in self.metadata.phase
