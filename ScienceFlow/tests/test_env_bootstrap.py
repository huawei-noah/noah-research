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

"""Tests for :mod:`scienceflow.utils.env_bootstrap`."""

from __future__ import annotations

from pathlib import Path

import pytest

from scienceflow.utils.env_bootstrap import bootstrap_dotenv


def test_bootstrap_dotenv_calls_repo_then_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[object, bool]] = []

    def fake_load_dotenv(*args: object, **kwargs: object) -> bool:
        dotenv_path = kwargs.get("dotenv_path")
        override = bool(kwargs.get("override", False))
        calls.append((dotenv_path, override))
        return True

    monkeypatch.setattr("dotenv.load_dotenv", fake_load_dotenv)
    monkeypatch.delenv("SCIENCEFLOW_DOTENV_OVERRIDE", raising=False)

    repo = tmp_path / "repo"
    repo.mkdir()
    bootstrap_dotenv(repo_root=repo)

    assert len(calls) == 2
    assert Path(calls[0][0]).resolve() == (repo / ".env").resolve()
    assert calls[0][1] is False
    assert calls[1] == (None, False)


def test_bootstrap_dotenv_repo_override_truthy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[object, bool]] = []

    def fake_load_dotenv(*args: object, **kwargs: object) -> bool:
        dotenv_path = kwargs.get("dotenv_path")
        override = bool(kwargs.get("override", False))
        calls.append((dotenv_path, override))
        return True

    monkeypatch.setattr("dotenv.load_dotenv", fake_load_dotenv)
    monkeypatch.setenv("SCIENCEFLOW_DOTENV_OVERRIDE", "1")

    repo = tmp_path / "r"
    repo.mkdir()
    bootstrap_dotenv(repo_root=repo)

    assert Path(calls[0][0]).resolve() == (repo / ".env").resolve()
    assert calls[0][1] is True
    assert calls[1] == (None, False)
