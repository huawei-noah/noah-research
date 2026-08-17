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

"""Pytest hooks: route tmp workspaces under ``/tmp`` so tests do not write under the repo.

- ``tmp_path`` / ``tmp_path_factory`` create directories under :data:`tests.fs_root.TEST_WORKSPACE_ROOT`
  (default ``/tmp/scienceflow_tests``); each test’s directory is removed in ``finally``.
- After the session, :func:`pytest_sessionfinish` deletes any **remaining** children under that
  root (e.g. after crashes). Set ``SCIENCEFLOW_TEST_KEEP_TMP=1`` to skip session cleanup for debugging.

**Repo root artifacts** (``logs/``, ``workspace/``, ``resolved_config.yaml`` under the ScienceFlow
package root): pytest does **not** create these in the repo; they come from CLI runs with default
``task_workspace_root_dir: "."`` or old files. This module does not remove them unless you set
``SCIENCEFLOW_TEST_CLEAN_REPO_ROOT=1`` (see :func:`pytest_sessionfinish`), which best-effort deletes
those paths after the session—use only when you accept losing local CLI outputs in the repo root.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from tests.fs_root import TEST_WORKSPACE_ROOT

# ScienceFlow repo root (parent of this ``tests/`` directory)
_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_workspace_root() -> None:
    TEST_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Per-test directory under ``/tmp/scienceflow_tests`` (replaces pytest default basetemp)."""
    safe = request.node.name.replace("/", "_").replace("[", "_").replace("]", "_")[:80]
    d = Path(
        tempfile.mkdtemp(prefix=f"pytest_{safe}_", dir=str(TEST_WORKSPACE_ROOT)),
    )
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_path_factory(request: pytest.FixtureRequest):
    """Factory for extra dirs under ``/tmp/scienceflow_tests`` (pytest-compatible ``mktemp`` API)."""
    created: list[Path] = []

    class _Factory:
        def mktemp(self, basename: str) -> Path:
            p = Path(
                tempfile.mkdtemp(prefix=f"{basename}_", dir=str(TEST_WORKSPACE_ROOT)),
            )
            created.append(p)
            return p

    def _cleanup() -> None:
        for p in created:
            shutil.rmtree(p, ignore_errors=True)

    request.addfinalizer(_cleanup)
    return _Factory()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Best-effort wipe of ``TEST_WORKSPACE_ROOT`` after all tests (see module docstring)."""
    if os.environ.get("SCIENCEFLOW_TEST_KEEP_TMP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        pass  # skip /tmp cleanup only; still allow opt-in repo-root cleanup below
    else:
        root = TEST_WORKSPACE_ROOT
        if root.is_dir():
            try:
                for child in root.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        try:
                            child.unlink()
                        except OSError:
                            pass
            except OSError:
                pass

    _clean_repo_root_artifacts_if_requested()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _clean_repo_root_artifacts_if_requested() -> None:
    """Remove ScienceFlow root ``logs/``, ``workspace/``, ``resolved_config.yaml`` if env is set."""
    if not _env_truthy("SCIENCEFLOW_TEST_CLEAN_REPO_ROOT"):
        return
    root = _REPO_ROOT
    f = root / "resolved_config.yaml"
    if f.is_file():
        try:
            f.unlink()
        except OSError:
            pass
    for dirname in ("logs", "workspace"):
        d = root / dirname
        if d.is_dir():
            try:
                shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
