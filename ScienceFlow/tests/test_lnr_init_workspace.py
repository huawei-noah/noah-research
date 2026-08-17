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

from pathlib import Path

from scienceflow.config.settings import LnrConfig
from scienceflow.solver.lnr.init_workspace import initialize_workspace_from_path
from scienceflow.solver.lnr.prompts import build_first_user_prompt
from scienceflow.utils.workspace_git import ensure_workspace_source_git, workspace_source_changed


def test_init_workspace_copies_model_and_feature_artifacts_but_excludes_control_dirs(tmp_path: Path) -> None:
    source = tmp_path / "source_workspace"
    target = tmp_path / "target_workspace"
    (source / ".logs").mkdir(parents=True)
    (source / ".logs" / "initial_workspace_state.md").write_text(
        "Previous performance:\n- Validation: 0.8 (auc, higher is better)\n",
        encoding="utf-8",
    )
    (source / ".memory").mkdir()
    (source / ".memory" / "short_term.json").write_text("[]\n", encoding="utf-8")
    (source / ".git").mkdir()
    (source / "dataset").mkdir()
    (source / "dataset" / "train.csv").write_text("id,y\n1,0\n", encoding="utf-8")
    (source / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "artifacts").mkdir()
    (source / "artifacts" / "model.pt").write_bytes(b"weights")
    (source / "features").mkdir()
    (source / "features" / "train.npy").write_bytes(b"feature-cache")
    (source / "cache").mkdir()
    (source / "cache" / "fold.parquet").write_bytes(b"parquet")
    (source / "submission.csv").write_text("id,target\n1,0.1\n", encoding="utf-8")

    result = initialize_workspace_from_path(source_workspace=source, target_workspace=target)

    assert result.applied is True
    assert result.copied_file_count >= 5
    assert "Validation: 0.8" in result.initial_workspace_state
    assert (target / "solution.py").is_file()
    assert (target / "artifacts" / "model.pt").read_bytes() == b"weights"
    assert (target / "features" / "train.npy").read_bytes() == b"feature-cache"
    assert (target / "cache" / "fold.parquet").read_bytes() == b"parquet"
    assert (target / "submission.csv").is_file()
    assert not (target / ".logs").exists()
    assert not (target / ".memory").exists()
    assert not (target / ".git").exists()
    assert not (target / "dataset").exists()


def test_init_workspace_markdown_is_bounded(tmp_path: Path) -> None:
    source = tmp_path / "source_workspace"
    target = tmp_path / "target_workspace"
    (source / ".logs").mkdir(parents=True)
    (source / ".logs" / "initial_workspace_state.md").write_text("A" * 100, encoding="utf-8")

    result = initialize_workspace_from_path(
        source_workspace=source,
        target_workspace=target,
        context_max_chars=20,
    )

    assert len(result.initial_workspace_state) < 100
    assert "truncated" in result.initial_workspace_state


def test_first_user_prompt_injects_initial_workspace_state() -> None:
    text = build_first_user_prompt(
        "Task body",
        wall_clock_budget_sec=300,
        initial_workspace_state="Previous performance:\n- Validation: 0.8",
    )

    assert "Initial workspace state:" in text
    assert "Previous performance:" in text
    assert "does not override task rules" in text
    assert text.index("Initial workspace state:") < text.index("Task:\nTask body")


def test_init_workspace_source_baseline_treats_copied_files_as_initial_state(tmp_path: Path) -> None:
    source = tmp_path / "source_workspace"
    target = tmp_path / "target_workspace"
    source.mkdir()
    (source / "solution.py").write_text("print(1)\n", encoding="utf-8")

    initialize_workspace_from_path(source_workspace=source, target_workspace=target)
    init = ensure_workspace_source_git(target, track_globs=["*.py"], initial_commit=True)

    assert init.ready is True
    assert workspace_source_changed(target, track_globs=["*.py"]) is False

    (target / "solution.py").write_text("print(2)\n", encoding="utf-8")
    assert workspace_source_changed(target, track_globs=["*.py"]) is True


def test_lnr_config_exposes_init_workspace_path() -> None:
    cfg = LnrConfig()

    assert cfg.init_workspace.workspace_path == ""
