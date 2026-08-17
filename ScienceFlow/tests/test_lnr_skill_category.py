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

from pathlib import Path
from types import SimpleNamespace

from scienceflow.solver.lnr.solver import LnrSolver


class _FakeRegistry:
    def get_by_category_label(self, label: str):
        if label != "Medical Imaging":
            return []
        return [SimpleNamespace(metadata=SimpleNamespace(name="cat_medical_imaging", priority=96))]

    def get_by_name_or_alias(self, name: str):
        return None


def test_lnr_skill_category_resolves_siim_allowlist(tmp_path) -> None:
    categories = tmp_path / "competition_categories.json"
    categories.write_text('{"siim-isic-melanoma-classification": "Medical Imaging"}')
    solver = object.__new__(LnrSolver)
    solver.lhr = SimpleNamespace(
        lnr_skill_tool_enabled=True,
        lnr_skill_tool_mode="category_only",
        lnr_skill_category_source=str(categories),
        lnr_skill_visible_max=1,
        lnr_skill_allow_generic_wildcard=False,
    )
    solver.cfg = SimpleNamespace(exp_id="siim-isic-melanoma-classification")
    solver.orchestrator = SimpleNamespace(skill_registry=_FakeRegistry())

    solver._configure_lnr_category_skill()

    assert solver.skill_category_status == "enabled"
    assert solver.skill_task_category == "Medical Imaging"
    assert solver.skill_allow_names == ("cat_medical_imaging",)
    assert solver._lnr_skill_hint().startswith("A category-specific skill")


def test_lnr_skill_category_missing_disables_tool(tmp_path) -> None:
    categories = tmp_path / "competition_categories.json"
    categories.write_text('{}')
    solver = object.__new__(LnrSolver)
    solver.lhr = SimpleNamespace(
        lnr_skill_tool_enabled=True,
        lnr_skill_tool_mode="category_only",
        lnr_skill_category_source=str(categories),
        lnr_skill_visible_max=1,
        lnr_skill_allow_generic_wildcard=False,
    )
    solver.cfg = SimpleNamespace(exp_id="unknown-task")
    solver.orchestrator = SimpleNamespace(skill_registry=_FakeRegistry())

    solver._configure_lnr_category_skill()

    assert solver.skill_category_status == "skill_category_missing"
    assert solver.skill_registry is None
    assert solver.skill_allow_names == ()


class _FakeTaskRegistry:
    def get_by_category_label(self, label: str):
        if label != "Segmentation":
            return []
        return [SimpleNamespace(metadata=SimpleNamespace(name="cat_segmentation", priority=95))]

    def get_by_name_or_alias(self, name: str):
        if name != "hubmap-kidney-segmentation":
            return None
        return SimpleNamespace(
            metadata=SimpleNamespace(
                name="task_hubmap_kidney_segmentation",
                category="tasks",
            )
        )


def test_lnr_skill_category_includes_task_skill_when_available(tmp_path) -> None:
    categories = tmp_path / "competition_categories.json"
    categories.write_text('{"hubmap-kidney-segmentation": "Segmentation"}')
    solver = object.__new__(LnrSolver)
    solver.lhr = SimpleNamespace(
        lnr_skill_tool_enabled=True,
        lnr_skill_tool_mode="category_only",
        lnr_skill_category_source=str(categories),
        lnr_skill_visible_max=1,
        lnr_skill_allow_generic_wildcard=False,
    )
    solver.cfg = SimpleNamespace(exp_id="hubmap-kidney-segmentation")
    solver.orchestrator = SimpleNamespace(skill_registry=_FakeTaskRegistry())

    solver._configure_lnr_category_skill()

    assert solver.skill_category_status == "enabled"
    assert solver.skill_task_category == "Segmentation"
    assert solver.skill_allow_names == (
        "cat_segmentation",
        "task_hubmap_kidney_segmentation",
    )
    assert solver.skill_visible_max == 2
    assert solver._lnr_skill_hint().startswith("Category- and task-specific skills")


def test_lnr_default_skill_registry_does_not_load_doc_skill_archives(tmp_path) -> None:
    categories = tmp_path / "competition_categories.json"
    categories.write_text('{"hubmap-kidney-segmentation": "Segmentation"}')
    solver = object.__new__(LnrSolver)
    solver.lhr = SimpleNamespace(
        lnr_skill_tool_enabled=True,
        lnr_skill_tool_mode="category_only",
        lnr_skill_category_source=str(categories),
        lnr_skill_visible_max=1,
        lnr_skill_allow_generic_wildcard=False,
    )
    solver.cfg = SimpleNamespace(exp_id="hubmap-kidney-segmentation")
    solver.orchestrator = SimpleNamespace(skill_registry=None)

    solver._configure_lnr_category_skill()

    assert solver.skill_category_status == "skill_category_unmapped"
    assert solver.skill_task_category == "Segmentation"
    assert solver.skill_registry is None
    assert solver.skill_allow_names == ()


class _FakeSkill:
    def __init__(self, name: str, category: str, content: str):
        self.metadata = SimpleNamespace(
            name=name,
            category=category,
            category_label="Medical Imaging" if category == "categories" else "",
            tags=["siim"],
            priority=98 if category == "tasks" else 96,
        )
        self._content = content

    def render(self):
        return self._content


class _FakeSiimTaskRegistry:
    def __init__(self):
        self.skills = {
            "cat_medical_imaging": _FakeSkill(
                "cat_medical_imaging",
                "categories",
                "# Medical\n\n## Auto-load hint\n\n- score cheap logits and metadata before large CNN\n- prefer cached strong timm backbones over ResNet-only baselines",
            ),
            "task_siim_isic_melanoma_classification": _FakeSkill(
                "task_siim_isic_melanoma_classification",
                "tasks",
                "# SIIM\n\n## Auto-load hint\n\n- SIIM is dermoscopy melanoma binary AUC; higher is better.\n- Use patient-aware validation and reducer-safe SIIM selection.",
            ),
        }

    def get_by_category_label(self, label: str):
        if label != "Medical Imaging":
            return []
        return [self.skills["cat_medical_imaging"]]

    def get_by_name(self, name: str):
        return self.skills.get(name)

    def get_by_name_or_alias(self, name: str):
        if name == "siim-isic-melanoma-classification":
            return self.skills["task_siim_isic_melanoma_classification"]
        return self.skills.get(name)


def test_lnr_skill_auto_read_includes_category_and_siim_task_skill(tmp_path) -> None:
    categories = tmp_path / "competition_categories.json"
    categories.write_text('{"siim-isic-melanoma-classification": "Medical Imaging"}')
    solver = object.__new__(LnrSolver)
    solver.lhr = SimpleNamespace(
        lnr_skill_tool_enabled=True,
        lnr_skill_tool_mode="category_only",
        lnr_skill_category_source=str(categories),
        lnr_skill_visible_max=1,
        lnr_skill_allow_generic_wildcard=False,
        lnr_skill_auto_read=True,
        lnr_skill_auto_read_max_chars=2000,
    )
    solver.cfg = SimpleNamespace(exp_id="siim-isic-melanoma-classification")
    solver.orchestrator = SimpleNamespace(skill_registry=_FakeSiimTaskRegistry())

    solver._configure_lnr_category_skill()
    hint = solver._lnr_skill_hint()

    assert solver.skill_allow_names == (
        "cat_medical_imaging",
        "task_siim_isic_melanoma_classification",
    )
    assert "Auto-loaded compact skill hints" in hint
    assert "score cheap logits" in hint
    assert "ResNet-only baselines" in hint
    assert "SIIM is dermoscopy melanoma binary AUC" in hint
    assert "ResNet is a sanity baseline" not in hint


def test_lnr_skill_auto_read_prefers_auto_load_hint(tmp_path) -> None:
    categories = tmp_path / "competition_categories.json"
    categories.write_text('{"siim-isic-melanoma-classification": "Medical Imaging"}')

    class RegistryWithLongSkill(_FakeSiimTaskRegistry):
        def __init__(self):
            super().__init__()
            self.skills["cat_medical_imaging"] = _FakeSkill(
                "cat_medical_imaging",
                "categories",
                "# Medical\n\n## Auto-load hint\n\n- short route hint\n\n## Full details\nlong tutorial text should stay out of auto context",
            )

    solver = object.__new__(LnrSolver)
    solver.lhr = SimpleNamespace(
        lnr_skill_tool_enabled=True,
        lnr_skill_tool_mode="category_only",
        lnr_skill_category_source=str(categories),
        lnr_skill_visible_max=1,
        lnr_skill_allow_generic_wildcard=False,
        lnr_skill_auto_read=True,
        lnr_skill_auto_read_max_chars=2000,
    )
    solver.cfg = SimpleNamespace(exp_id="siim-isic-melanoma-classification")
    solver.orchestrator = SimpleNamespace(skill_registry=RegistryWithLongSkill())

    solver._configure_lnr_category_skill()
    hint = solver._lnr_skill_hint()

    assert "short route hint" in hint
    assert "source=auto_hint" in hint
    assert "long tutorial text" not in hint
