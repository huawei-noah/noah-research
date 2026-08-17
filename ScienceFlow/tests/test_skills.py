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

import asyncio

import pytest
from pathlib import Path

from scienceflow.core.tools.skill_tool import SkillTool
from scienceflow.core.skills.base import Skill
from scienceflow.core.skills.injector import SkillInjector
from scienceflow.core.skills.matcher import SkillMatcher
from scienceflow.core.skills.paths import default_skill_library_dir
from scienceflow.core.skills.registry import SkillRegistry

SAMPLE_SKILL_MD = """\
---
name: data_cleaning
aliases:
  - data_cleaning_alias
category: preprocessing
category_label: "Classification"
tags:
  - tabular
  - pandas
task_types:
  - classification
  - "*"
phase:
  - draft
  - improve
priority: 80
---
# Data Cleaning Skill

Use pandas to handle missing values and outliers.
"""

SAMPLE_SKILL_NO_FRONTMATTER = """\
# Plain Skill

No YAML frontmatter here.
"""


@pytest.fixture()
def skill_file(tmp_path) -> Path:
    p = tmp_path / "data_cleaning.md"
    p.write_text(SAMPLE_SKILL_MD)
    return p


@pytest.fixture()
def plain_skill_file(tmp_path) -> Path:
    p = tmp_path / "plain.md"
    p.write_text(SAMPLE_SKILL_NO_FRONTMATTER)
    return p


@pytest.fixture()
def skill_library(tmp_path) -> Path:
    (tmp_path / "a.md").write_text(SAMPLE_SKILL_MD)
    other = SAMPLE_SKILL_MD.replace("data_cleaning", "feature_eng").replace("preprocessing", "feature")
    (tmp_path / "b.md").write_text(other)
    return tmp_path


class TestSkillFromFile:

    def test_parses_metadata(self, skill_file):
        skill = Skill.from_file(skill_file)
        assert skill.metadata.name == "data_cleaning"
        assert skill.metadata.category == "preprocessing"
        assert skill.metadata.category_label == "Classification"
        assert "tabular" in skill.metadata.tags
        assert skill.metadata.priority == 80

    def test_content_excludes_frontmatter(self, skill_file):
        skill = Skill.from_file(skill_file)
        assert "---" not in skill.content
        assert "Data Cleaning Skill" in skill.content

    def test_no_frontmatter(self, plain_skill_file):
        skill = Skill.from_file(plain_skill_file)
        assert skill.metadata.name == ""
        assert "Plain Skill" in skill.content

    def test_matches_task_type_wildcard(self, skill_file):
        skill = Skill.from_file(skill_file)
        assert skill.matches_task_type("anything")

    def test_matches_phase(self, skill_file):
        skill = Skill.from_file(skill_file)
        assert skill.matches_phase("draft")
        assert not skill.matches_phase("debug")


class TestSkillRegistry:

    def test_load_all(self, skill_library):
        reg = SkillRegistry()
        reg.load_all(skill_library)
        assert len(reg.get_all()) == 2

    def test_get_by_name(self, skill_library):
        reg = SkillRegistry()
        reg.load_all(skill_library)
        assert reg.get_by_name("data_cleaning") is not None

    def test_get_by_name_or_alias_explicit(self, skill_library):
        reg = SkillRegistry()
        reg.load_all(skill_library)
        skill = reg.get_by_name_or_alias("data_cleaning_alias")
        assert skill is not None
        assert skill.metadata.name == "data_cleaning"

    def test_get_by_name_or_alias_implicit_category_alias(self, tmp_path):
        skill_md = """\
---
name: cat_tabular
category: categories
task_types: ["*"]
phase: [analysis]
---
# Tabular category
"""
        (tmp_path / "cat_tabular.md").write_text(skill_md)
        reg = SkillRegistry()
        reg.load_all(tmp_path)
        skill = reg.get_by_name_or_alias("tabular")
        assert skill is not None
        assert skill.metadata.name == "cat_tabular"

    def test_ambiguous_alias_returns_none(self, tmp_path):
        a_md = """\
---
name: first_skill
aliases: [shared_alias]
category: misc
task_types: ["*"]
phase: [analysis]
---
# First
"""
        b_md = """\
---
name: second_skill
aliases: [shared_alias]
category: misc
task_types: ["*"]
phase: [analysis]
---
# Second
"""
        (tmp_path / "a.md").write_text(a_md)
        (tmp_path / "b.md").write_text(b_md)
        reg = SkillRegistry()
        reg.load_all(tmp_path)
        assert reg.get_by_name_or_alias("shared_alias") is None
        assert reg.get_name_matches("shared_alias") == ["first_skill", "second_skill"]

    def test_suggest_typo_returns_canonical(self, skill_library):
        reg = SkillRegistry()
        reg.load_all(skill_library)
        suggestions = reg.suggest("data_cleaning_alis", top_k=3)
        assert "data_cleaning" in suggestions

    def test_get_by_category(self, skill_library):
        reg = SkillRegistry()
        reg.load_all(skill_library)
        assert len(reg.get_by_category("preprocessing")) >= 1

    def test_get_by_tag(self, skill_library):
        reg = SkillRegistry()
        reg.load_all(skill_library)
        assert len(reg.get_by_tag("tabular")) >= 1

    def test_register_external(self, skill_file):
        reg = SkillRegistry()
        reg.register_external(skill_file)
        assert reg.get_by_name("data_cleaning") is not None


    def test_get_by_category_label_sorted_by_priority(self, tmp_path):
        low = """\
---
name: cat_low
category: categories
category_label: "Medical Imaging"
task_types: ["Medical Imaging"]
phase: [analysis]
priority: 10
---
# Low
"""
        high = low.replace("cat_low", "cat_high").replace("priority: 10", "priority: 90")
        (tmp_path / "low.md").write_text(low)
        (tmp_path / "high.md").write_text(high)
        reg = SkillRegistry()
        reg.load_all(tmp_path)

        assert [s.metadata.name for s in reg.get_by_category_label("Medical Imaging")] == [
            "cat_high",
            "cat_low",
        ]


class TestSkillMatcher:

    @pytest.fixture()
    def matcher(self, skill_library):
        reg = SkillRegistry()
        reg.load_all(skill_library)
        return SkillMatcher(reg)

    def test_match_by_task_and_phase(self, matcher):
        results = matcher.match(task_type="classification", phase="draft")
        assert len(results) >= 1

    def test_match_no_results(self, matcher):
        results = matcher.match(task_type="classification", phase="debug")
        assert results == []

    def test_match_with_tags(self, matcher):
        results = matcher.match(task_type="classification", phase="draft", tags=["tabular"])
        assert all("tabular" in s.metadata.tags for s in results)

    def test_sorted_by_priority_desc(self, matcher):
        results = matcher.match(task_type="classification", phase="draft")
        priorities = [s.metadata.priority for s in results]
        assert priorities == sorted(priorities, reverse=True)


class TestSkillInjector:

    @pytest.fixture()
    def injector(self):
        return SkillInjector()

    @pytest.fixture()
    def sample_skills(self, skill_file):
        return [Skill.from_file(skill_file)]

    def test_inject_replaces_placeholder(self, injector, sample_skills):
        prompt = "Hello {skill_context} world"
        result = injector.inject_to_prompt(prompt, sample_skills)
        assert "{skill_context}" not in result
        assert "data_cleaning" in result

    def test_inject_empty_skills_removes_placeholder(self, injector):
        prompt = "Hello {skill_context} world"
        result = injector.inject_to_prompt(prompt, [])
        assert result == "Hello  world"

    def test_inject_no_placeholder_unchanged(self, injector, sample_skills):
        prompt = "No placeholder here"
        result = injector.inject_to_prompt(prompt, sample_skills)
        assert result == "No placeholder here"


def test_default_skill_library_path_uses_scienceflow(tmp_path: Path) -> None:
    assert default_skill_library_dir(tmp_path) == tmp_path / ".scienceflow" / "skills"


def test_default_skill_library_excludes_route_prior_archives() -> None:
    repo = Path(__file__).resolve().parents[1]
    library = repo / ".scienceflow" / "skills"
    reg = SkillRegistry()
    reg.load_all(library)

    assert reg.get_by_name_or_alias("hubmap-kidney-segmentation") is None
    assert reg.get_by_name_or_alias("siim-isic-melanoma-classification") is None
    assert reg.get_by_name_or_alias("nfl-player-contact-detection") is None
    assert reg.get_by_name_or_alias("medical_imaging") is None
    assert reg.get_by_name_or_alias("segmentation") is None
    assert all(skill.metadata.category not in {"tasks", "categories"} for skill in reg.get_all())


def test_default_skill_library_exposes_recommender_data_prep_skill() -> None:
    repo = Path(__file__).resolve().parents[1]
    library = repo / ".scienceflow" / "skills"
    reg = SkillRegistry()
    reg.load_all(library)

    skill = reg.get_by_name_or_alias("recommender-time-split")
    assert skill is not None
    assert skill.metadata.category == "data_processing"
    assert "data-preparation" in skill.metadata.tags

    tool = SkillTool(registry=reg, task_type="*", mode="all", visible_max=5)
    listed = asyncio.run(
        tool.execute(action="list", category="data_processing", tag="data-preparation")
    ).output or ""
    assert "- recommender_time_split" in listed
    read_ok = asyncio.run(tool.execute(action="read", name="recommender_time_split")).output or ""
    assert "Transaction recommender time split" in read_ok


def test_skill_tool_category_only_allowlist_hides_other_skills(tmp_path) -> None:
    medical = """\
---
name: cat_medical_imaging
category: categories
category_label: "Medical Imaging"
tags: [medical]
task_types: ["Medical Imaging"]
phase: [analysis]
priority: 96
---
# Medical
"""
    tabular = medical.replace("cat_medical_imaging", "cat_tabular").replace(
        "Medical Imaging", "Tabular"
    )
    generic = """\
---
name: generic_training
category: training
task_types: ["*"]
phase: [analysis]
priority: 99
---
# Generic
"""
    (tmp_path / "medical.md").write_text(medical)
    (tmp_path / "tabular.md").write_text(tabular)
    (tmp_path / "generic.md").write_text(generic)
    reg = SkillRegistry()
    reg.load_all(tmp_path)
    tool = SkillTool(
        registry=reg,
        task_type="Medical Imaging",
        mode="category_only",
        allow_names=("cat_medical_imaging",),
        allow_generic_wildcard=False,
        visible_max=1,
    )

    listed = asyncio.run(tool.execute(action="list")).output or ""
    assert "count=1 mode=category_only task_type=Medical Imaging allowlist=cat_medical_imaging" in listed
    assert "- cat_medical_imaging" in listed
    assert "cat_tabular" not in listed
    assert "generic_training" not in listed

    read_ok = asyncio.run(tool.execute(action="read", name="cat_medical_imaging")).output or ""
    assert "[skill:cat_medical_imaging]" in read_ok
    assert "category_label=Medical Imaging" in read_ok

    read_blocked = asyncio.run(tool.execute(action="read", name="cat_tabular")).error or ""
    assert "not enabled for this task in category_only mode" in read_blocked
    assert "enabled_skill=cat_medical_imaging task_type=Medical Imaging" in read_blocked
