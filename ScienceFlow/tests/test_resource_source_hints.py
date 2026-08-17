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

from scienceflow.solver.lnr.resource_runtime.source_hints import detect_resource_source_hint


def test_readonly_python_source_inspection_does_not_request_gpu(tmp_path) -> None:
    (tmp_path / "predict.py").write_text(
        "import torch\nmodel.to('cuda')\nmodel.eval()\n",
        encoding="utf-8",
    )

    hint = detect_resource_source_hint(
        command="head -50 predict.py",
        workspace_dir=tmp_path,
    )

    assert hint.entrypoints == []
    assert hint.source_files_inspected == 0
    assert hint.source_gpu_evidence is False


def test_executed_python_source_is_still_inspected(tmp_path) -> None:
    (tmp_path / "predict.py").write_text(
        "import torch\nmodel.to('cuda')\nmodel.eval()\n",
        encoding="utf-8",
    )

    hint = detect_resource_source_hint(
        command="python3 predict.py",
        workspace_dir=tmp_path,
    )

    assert hint.entrypoints == ["predict.py"]
    assert hint.source_files_inspected == 1
    assert hint.source_gpu_evidence is True
