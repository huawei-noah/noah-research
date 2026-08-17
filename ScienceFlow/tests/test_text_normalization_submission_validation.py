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

from scienceflow.core.artifact_io import wait_for_stable_file


def test_wait_for_stable_file_flags_fresh_artifact_as_transient(tmp_path: Path) -> None:
    sub = tmp_path / "submission.csv"
    sub.write_text("id,after\n0_0,test\n", encoding="utf-8")

    stable, message = wait_for_stable_file(
        sub,
        timeout_sec=0.0,
        interval_sec=0.01,
        min_age_sec=60.0,
    )

    assert stable is False
    assert "still be writing" in message
