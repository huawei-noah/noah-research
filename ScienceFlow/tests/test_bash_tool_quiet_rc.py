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

"""Tests for the fast-fail heuristic split in bash_tool.py.

Distinguishes "quiet rc" (silent redirect / rc==2, file-not-found / glob miss)
from true infrastructure failures (taskset/affinity, signal kills).

The fast-exit path is exercised by patching spawn_shell so the fake process
returns immediately with controlled rc and no stdout/stderr, then freezing
time so elapsed < 0.5 s.
"""

from __future__ import annotations

import time as _time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scienceflow.core.tools.bash_tool import BashTool, _has_silent_redirect


# ---------------------------------------------------------------------------
# Unit tests for _has_silent_redirect helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "ls -la solution.py 2>/dev/null",
        "ls -la *.py 2>/dev/null; ls -la *.pkl 2>/dev/null",
        "cat file.txt &>/dev/null",
        "grep foo bar &> /dev/null",
        "command > /dev/null 2>&1",
        "command 2>&1 >/dev/null",
    ],
)
def test_has_silent_redirect_detects(cmd: str) -> None:
    assert _has_silent_redirect(cmd) is True


@pytest.mark.parametrize(
    "cmd",
    [
        "ls -la solution.py",
        "python3 solution.py",
        "echo hello",
        "taskset -c 64-127 bash -c 'ls solution.py'",
        "cat file.txt > output.txt",  # only stdout redirect, stderr not silenced
    ],
)
def test_has_silent_redirect_misses(cmd: str) -> None:
    assert _has_silent_redirect(cmd) is False


# ---------------------------------------------------------------------------
# Shared fixture: patch spawn_shell + time.time to simulate fast exit
# ---------------------------------------------------------------------------


def _make_fake_proc(returncode: int) -> MagicMock:
    fake = MagicMock()
    fake.returncode = returncode
    fake.pid = 99998

    async def _empty() -> bytes:
        return b""

    fake.stdout = MagicMock()
    fake.stderr = MagicMock()
    fake.stdout.readline = _empty
    fake.stderr.readline = _empty
    fake.wait = AsyncMock(return_value=None)
    return fake


def _freeze_time():
    """Return a callable that always returns the same timestamp (elapsed -> 0)."""
    t0 = _time.time()
    return lambda: t0


async def _run_with_fast_exit(tmp_path, cmd: str, rc: int):
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=10.0,
        bash_timeout_slow_sec=10.0,
    )
    fake_proc = _make_fake_proc(rc)
    with (
        patch("scienceflow.core.tools.bash_tool.spawn_shell", return_value=fake_proc),
        patch("scienceflow.core.tools.bash_tool.time.time", side_effect=_freeze_time()),
    ):
        return await tool.execute(cmd)


# ---------------------------------------------------------------------------
# B: benign "quiet rc" cases → must emit [no-output], must NOT emit [infra-error]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compound_ls_silent_rc2_gives_no_output(tmp_path) -> None:
    """The exact pattern from the failing run: compound ls with 2>/dev/null, rc=2."""
    cmd = (
        "ls -la solution.py 2>/dev/null; "
        "ls -la *.py 2>/dev/null; "
        "ls -la *.pkl *.pth *.joblib 2>/dev/null"
    )
    r = await _run_with_fast_exit(tmp_path, cmd, rc=2)
    assert r.error, "should still be an error result"
    assert "[no-output]" in (r.output or ""), f"got: {r.output!r}"
    assert "[infra-error]" not in (r.output or ""), f"got: {r.output!r}"


@pytest.mark.asyncio
async def test_single_ls_silent_rc2_gives_no_output(tmp_path) -> None:
    r = await _run_with_fast_exit(tmp_path, "ls *.py 2>/dev/null", rc=2)
    assert r.error
    assert "[no-output]" in (r.output or "")
    assert "[infra-error]" not in (r.output or "")


@pytest.mark.asyncio
async def test_rc2_without_redirect_gives_no_output(tmp_path) -> None:
    """rc=2 alone (no explicit redirect) is also treated as quiet (GNU 'no such file')."""
    r = await _run_with_fast_exit(tmp_path, "ls missing_file_xyz", rc=2)
    assert r.error
    assert "[no-output]" in (r.output or "")
    assert "[infra-error]" not in (r.output or "")


@pytest.mark.asyncio
async def test_redirect_rc1_gives_no_output(tmp_path) -> None:
    """Explicit redirect with rc=1 is also treated as quiet (silenced stderr)."""
    r = await _run_with_fast_exit(tmp_path, "stat foo.txt 2>/dev/null", rc=1)
    assert r.error
    assert "[no-output]" in (r.output or "")
    assert "[infra-error]" not in (r.output or "")


# ---------------------------------------------------------------------------
# C: true infra-failure cases → must emit [infra-error], must NOT emit [no-output]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("rc", [126, 127])
async def test_infra_rc_without_redirect_gives_infra_error(tmp_path, rc: int) -> None:
    """rc=126/127 (permission denied / not found by shell) without redirect → infra error."""
    r = await _run_with_fast_exit(tmp_path, "taskset -c 99999 bash -c 'echo hi'", rc=rc)
    assert r.error
    assert "[infra-error]" in (r.output or ""), f"rc={rc} got: {r.output!r}"
    assert "[no-output]" not in (r.output or "")


@pytest.mark.asyncio
async def test_rc1_no_redirect_gives_infra_error(tmp_path) -> None:
    """rc=1 without any silent redirect → treated as infra failure (conservative)."""
    r = await _run_with_fast_exit(tmp_path, "echo ignored", rc=1)
    assert r.error
    assert "[infra-error]" in (r.output or ""), f"got: {r.output!r}"
    assert "[no-output]" not in (r.output or "")


@pytest.mark.asyncio
async def test_negative_rc_gives_infra_error(tmp_path) -> None:
    """Negative rc (signal kill before any output) → infra error."""
    r = await _run_with_fast_exit(tmp_path, "sleep 10", rc=-9)
    assert r.error
    assert "[infra-error]" in (r.output or ""), f"got: {r.output!r}"
    assert "[no-output]" not in (r.output or "")


# ---------------------------------------------------------------------------
# Existing behaviour preserved: rc=0 never hits fast-fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rc0_with_redirect_no_fast_fail(tmp_path) -> None:
    """A successful command (rc=0) must never hit either fast-fail branch."""
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
    )
    r = await tool.execute("true 2>/dev/null")
    assert not r.error, f"expected success, got: {r.output!r}"
    assert "[no-output]" not in (r.output or "")
    assert "[infra-error]" not in (r.output or "")


@pytest.mark.asyncio
async def test_spawn_failure_returns_tool_error_instead_of_raising(tmp_path) -> None:
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
    )
    exc = BlockingIOError(11, "Resource temporarily unavailable")

    with patch("scienceflow.core.tools.bash_tool.spawn_shell", side_effect=exc):
        r = await tool.execute("python train.py")

    assert r.error == "bash spawn failed: BlockingIOError"
    assert "[spawn-failed" in (r.output or "")
    assert "[infra-error]" in (r.output or "")
    assert "Continue the search until the wall-clock budget expires" in (r.output or "")
