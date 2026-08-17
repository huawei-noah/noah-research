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

import scienceflow.core.tools.shadow_workspace as shadow_workspace_module
from scienceflow.core.tools.shadow_workspace import ShadowWorkspaceManager


def test_shadow_workspace_copies_files_without_polluting_real_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text("print('real')\n", encoding="utf-8")
    (workspace / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (workspace / "outputs").mkdir()

    shadow = ShadowWorkspaceManager().create(workspace, "job/1")
    (shadow.shadow_workspace / "solution.py").write_text("print('shadow')\n", encoding="utf-8")
    (shadow.shadow_workspace / "submission.csv").write_text("id,target\n1,1\n", encoding="utf-8")
    (shadow.shadow_workspace / "outputs" / "trial.txt").write_text("trial\n", encoding="utf-8")

    ok, changed = shadow.validate_real_workspace_unchanged()
    assert ok is True
    assert changed == []
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "print('real')\n"
    assert (workspace / "submission.csv").read_text(encoding="utf-8") == "id,target\n1,0\n"

    shadow.cleanup()
    assert not shadow.shadow_workspace.exists()


def test_shadow_workspace_detects_real_protected_output_mutation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    shadow = ShadowWorkspaceManager().create(workspace, "job-2")
    (workspace / "submission.csv").write_text("id,target\n1,2\n", encoding="utf-8")

    ok, changed = shadow.validate_real_workspace_unchanged()
    assert ok is False
    assert changed == ["modified:submission.csv"]


def test_shadow_workspace_detects_shadow_readonly_write_without_polluting_real(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    dataset = workspace / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    shadow = ShadowWorkspaceManager(
        readonly_dir_names=("dataset",),
        protected_globs=("dataset/**",),
    ).create(workspace, "job-3")
    assert (shadow.shadow_workspace / "dataset").is_dir()
    assert (shadow.shadow_workspace / "dataset" / "train.csv").is_symlink()

    (shadow.shadow_workspace / "dataset" / "leak.csv").write_text("bad\n", encoding="utf-8")

    real_ok, real_changed = shadow.validate_real_workspace_unchanged()
    assert real_ok is True
    assert real_changed == []
    assert not (workspace / "dataset" / "leak.csv").exists()

    shadow_ok, shadow_changed = shadow.validate_shadow_protected_unchanged()
    assert shadow_ok is False
    assert shadow_changed == ["added:dataset/leak.csv"]


def test_shadow_workspace_readonly_fingerprint_does_not_digest_data_files(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    dataset = workspace / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    digest_calls: list[Path] = []

    def fail_digest(path: Path) -> str:
        digest_calls.append(path)
        raise AssertionError(f"content digest should not read readonly data file: {path}")

    monkeypatch.setattr(shadow_workspace_module, "_file_digest", fail_digest)
    shadow = ShadowWorkspaceManager(
        readonly_dir_names=("dataset",),
        protected_globs=("dataset/**",),
    ).create(workspace, "job-4")

    real_ok, real_changed = shadow.validate_real_workspace_unchanged()
    shadow_ok, shadow_changed = shadow.validate_shadow_protected_unchanged()

    assert real_ok is True
    assert real_changed == []
    assert shadow_ok is True
    assert shadow_changed == []
    assert digest_calls == []
