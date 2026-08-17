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

"""Sanity checks for the pinned official **mlebench** Git dependency."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

pytestmark = pytest.mark.mlebench


def test_mlebench_distribution_installed() -> None:
    pytest.importorskip("mlebench")
    try:
        v = version("mlebench")
    except PackageNotFoundError:
        pytest.fail("mlebench is importable but not registered in importlib.metadata")
    assert v


def test_mlebench_competitions_bundle_present() -> None:
    mlebench = pytest.importorskip("mlebench")
    competitions = Path(mlebench.__file__).resolve().parent / "competitions"
    assert competitions.is_dir(), f"missing bundled competitions: {competitions}"
    configs = list(competitions.rglob("config.yaml"))
    assert len(configs) >= 10, f"expected many competition configs, found {len(configs)}"


def test_mlebench_cli_help() -> None:
    pytest.importorskip("mlebench")
    proc = subprocess.run(
        [sys.executable, "-m", "mlebench.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "prepare" in out.lower() or "sub-command" in out.lower(), out[:500]


def test_mlebench_registry_lists_competitions() -> None:
    pytest.importorskip("mlebench")
    from mlebench.registry import registry

    ids = registry.list_competition_ids()
    assert isinstance(ids, list)
    assert len(ids) >= 10
    assert all(isinstance(i, str) and i for i in ids)
