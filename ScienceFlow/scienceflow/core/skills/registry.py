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

import difflib
import re
from pathlib import Path

from scienceflow.core.skills.base import Skill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._aliases: dict[str, set[str]] = {}
        self._category_labels: dict[str, set[str]] = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        key = name.strip().lower()
        key = re.sub(r"[\s\-]+", "_", key)
        key = re.sub(r"[^a-z0-9_]+", "", key)
        key = re.sub(r"_+", "_", key)
        return key.strip("_")

    def _add_alias(self, alias: str, canonical: str) -> None:
        key = self._normalize_name(alias)
        if not key:
            return
        self._aliases.setdefault(key, set()).add(canonical)

    def _add_category_label(self, label: str, canonical: str) -> None:
        key = str(label or "").strip()
        if not key:
            return
        self._category_labels.setdefault(key, set()).add(canonical)

    def _iter_implicit_aliases(self, canonical: str) -> list[str]:
        out: list[str] = [
            canonical.replace("_", " "),
            canonical.replace("_", "-"),
            canonical.replace("_", ""),
        ]
        if canonical.startswith("cat_"):
            out.append(canonical[4:])
        if canonical.startswith("mlebench_"):
            out.append(canonical[len("mlebench_"):])
        return out

    def _register_skill(self, skill: Skill) -> None:
        canonical = (skill.metadata.name or "").strip()
        if not canonical:
            return
        self._skills[canonical] = skill
        self._add_alias(canonical, canonical)
        for alias in self._iter_implicit_aliases(canonical):
            self._add_alias(alias, canonical)
        for alias in skill.metadata.aliases:
            alias_name = (alias or "").strip()
            if alias_name:
                self._add_alias(alias_name, canonical)
        self._add_category_label(skill.metadata.category_label, canonical)

    def load_all(self, library_dir: Path) -> None:
        """Scan a Markdown skill directory, parse all .md files, and build indexes."""
        for md_file in library_dir.rglob("*.md"):
            skill = Skill.from_file(md_file)
            self._register_skill(skill)

    def get_by_category(self, category: str) -> list[Skill]:
        return [s for s in self._skills.values() if s.metadata.category == category]

    def get_by_tag(self, tag: str) -> list[Skill]:
        return [s for s in self._skills.values() if tag in s.metadata.tags]

    def get_by_category_label(self, label: str) -> list[Skill]:
        names = self._category_labels.get(str(label or "").strip(), set())
        skills = [self._skills[name] for name in names if name in self._skills]
        skills.sort(key=lambda s: (-int(s.metadata.priority), s.metadata.name.lower()))
        return skills

    def get_by_name(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def get_name_matches(self, name: str) -> list[str]:
        query = name.strip()
        if not query:
            return []
        direct = self.get_by_name(query)
        if direct is not None:
            return [direct.metadata.name]
        return sorted(self._aliases.get(self._normalize_name(query), set()))

    def get_by_name_or_alias(self, name: str) -> Skill | None:
        matches = self.get_name_matches(name)
        if len(matches) != 1:
            return None
        return self._skills.get(matches[0])

    def suggest(self, name: str, top_k: int = 3) -> list[str]:
        query = name.strip()
        if not query:
            return []
        exact_matches = self.get_name_matches(query)
        if len(exact_matches) > 1:
            return exact_matches[:top_k]

        index: dict[str, set[str]] = {}
        for canonical in self._skills:
            norm = self._normalize_name(canonical)
            index.setdefault(norm, set()).add(canonical)
        for alias, canonicals in self._aliases.items():
            index.setdefault(alias, set()).update(canonicals)

        candidates = list(index.keys())
        if not candidates:
            return []

        raw_matches = difflib.get_close_matches(
            self._normalize_name(query), candidates, n=top_k * 4, cutoff=0.55
        )

        out: list[str] = []
        seen: set[str] = set()
        for match in raw_matches:
            for canonical in sorted(index[match]):
                if canonical not in seen:
                    seen.add(canonical)
                    out.append(canonical)
                if len(out) >= top_k:
                    break
            if len(out) >= top_k:
                break
        return out

    def get_all(self) -> list[Skill]:
        return list(self._skills.values())

    def register_external(self, skill_path: Path) -> None:
        skill = Skill.from_file(skill_path)
        self._register_skill(skill)
