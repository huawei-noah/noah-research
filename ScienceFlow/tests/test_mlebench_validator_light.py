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

from scienceflow.gates.evaluator.providers.mlebench import (
    resolve_mlebench_exp_id,
    validate_submission_light,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_light_validation_finds_prefixed_sample_submission(tmp_path: Path) -> None:
    _write(tmp_path / "dataset" / "ru_sample_submission.csv", "id,after\n1,a\n2,b\n")
    _write(tmp_path / "submission.csv", "id,after\n1,x\n2,y\n")

    ok, message = validate_submission_light(tmp_path)

    assert ok is True
    assert "ru_sample_submission.csv" in message


def test_light_validation_rejects_blank_required_target_cell(tmp_path: Path) -> None:
    _write(tmp_path / "dataset" / "ru_sample_submission.csv", "id,after\n1,a\n2,b\n")
    _write(tmp_path / "submission.csv", "id,after\n1,x\n2,\n")

    ok, message = validate_submission_light(tmp_path)

    assert ok is False
    assert "blank required cell" in message
    assert "after" in message


def test_light_validation_rejects_row_count_mismatch(tmp_path: Path) -> None:
    _write(tmp_path / "dataset" / "sample_submission.csv", "id,target\n1,0\n2,0\n")
    _write(tmp_path / "submission.csv", "id,target\n1,0.1\n2,0.2\n3,0.3\n")

    ok, message = validate_submission_light(tmp_path)

    assert ok is False
    assert "submission rows" in message


def test_light_validation_without_sample_still_rejects_blank_cells(tmp_path: Path) -> None:
    _write(tmp_path / "submission.csv", "id,target\n1,0.1\n2,\n")

    ok, message = validate_submission_light(tmp_path)

    assert ok is False
    assert "blank required cell" in message


def test_resolve_exp_id_from_worker_resolved_config(tmp_path: Path) -> None:
    worker_root = tmp_path / "task" / "workers" / "w00"
    workspace = worker_root
    _write(worker_root / "resolved_config.yaml", "exp_id: text-normalization-challenge-russian-language\n")

    assert resolve_mlebench_exp_id(workspace) == "text-normalization-challenge-russian-language"
