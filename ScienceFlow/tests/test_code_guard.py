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

"""Tests for scienceflow.safety.code_guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from scienceflow.safety.code_guard import _check_writes_result_md, pre_execution_guard


def test_check_writes_result_md_detects_open_write() -> None:
    code = "with open('result.md', 'w') as f:\n    f.write('x')\n"
    issues = _check_writes_result_md(code)
    assert issues
    assert any("RESULT_MD_IN_CODE" in i for i in issues)


def test_check_writes_result_md_allows_read_only_open() -> None:
    code = "x = open('result.md').read()\n"
    assert _check_writes_result_md(code) == []


def test_check_writes_result_md_detects_path_write_text() -> None:
    code = "from pathlib import Path\nPath('result.md').write_text('a')\n"
    issues = _check_writes_result_md(code)
    assert issues
    assert any("RESULT_MD_IN_CODE" in i for i in issues)


def test_pre_execution_guard_warns_on_result_md_write_not_fatal(
    tmp_path: Path,
) -> None:
    """Writing result.md in solution is a warning; leakage/syntax still fatal."""
    code = """
import pandas as pd
x = pd.DataFrame({'a':[1]})
x.to_csv('submission.csv', index=False)
with open('result.md', 'w') as f:
    f.write('ok')
"""
    gr = pre_execution_guard(code, gpu_available=False, solution_path=tmp_path / "s.py")
    assert not gr.is_fatal
    assert any("RESULT_MD_IN_CODE" in i for i in gr.issues)


@pytest.mark.parametrize(
    "snippet",
    [
        "open('result.md', 'a')",
        'open("result.md", "w")',
    ],
)
def test_check_writes_result_md_append_mode(snippet: str) -> None:
    issues = _check_writes_result_md(f"{snippet}\n")
    assert issues
