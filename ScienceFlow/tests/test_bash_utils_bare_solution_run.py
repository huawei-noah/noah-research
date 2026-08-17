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

"""Tests for progressive embedded-full-run command classification."""

from scienceflow.core.agent.tools.bash_utils import (
    _looks_like_bare_solution_run,
    command_sets_subsample_row_env,
)


def test_command_sets_subsample_row_env_detects_common_patterns() -> None:
    for name in (
        "NROWS",
        "QUICK_TEST_ROWS",
        "QUICK_TEST",
        "SAMPLE_N",
        "LIMIT_ROWS",
        "MAX_ROWS",
        "DEBUG_ROWS",
        "DEV_ROWS",
    ):
        assert command_sets_subsample_row_env(f"{name}=100 python3 solution.py")
    assert command_sets_subsample_row_env("nrows=500 python3 solution.py")
    assert command_sets_subsample_row_env("env NROWS=100 python3 solution.py")


def test_looks_like_bare_solution_run() -> None:
    assert _looks_like_bare_solution_run("python3 solution.py")
    assert _looks_like_bare_solution_run("python3 ./solution.py")
    assert _looks_like_bare_solution_run("env FOO=bar python3 solution.py")
    assert _looks_like_bare_solution_run("taskset -c 0 python3 solution.py")
    assert _looks_like_bare_solution_run("python3 -u solution.py")
    assert _looks_like_bare_solution_run("python3 solution.py --lr 0.01 --epochs 50")
    assert not _looks_like_bare_solution_run("NROWS=1000 python3 solution.py")
    assert not _looks_like_bare_solution_run("QUICK_TEST_ROWS=50 python3 solution.py")
    assert not _looks_like_bare_solution_run("env NROWS=100 python3 solution.py")
    assert not _looks_like_bare_solution_run("python3 train.py")
    assert not _looks_like_bare_solution_run("python3 -c \"open('solution.py').read()\"")
    assert not _looks_like_bare_solution_run("grep -n solution.py solution.py")
    assert not _looks_like_bare_solution_run("cat solution.py | python3")
    assert not _looks_like_bare_solution_run("python3 solution.py --help")
    assert not _looks_like_bare_solution_run("python3 solution.py -h")
    assert not _looks_like_bare_solution_run("python3 solution.py --lr 0.01 --help")
