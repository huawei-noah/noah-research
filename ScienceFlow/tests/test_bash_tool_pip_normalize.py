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

"""Tests for bash command normalization (bare pip -> uv pip) and priv-esc blocks."""

from __future__ import annotations

import pytest

from scienceflow.core.tools.bash.guards import (
    _format_cpu_set_compact,
    _infer_timeout,
    _parse_cpu_set_string,
    _rewrite_bare_pip_command,
    global_filesystem_scan_blocked_error,
    interactive_stdin_blocked_error,
    normalize_bash_command_for_agent,
    privilege_escalation_blocked_error,
    process_control_blocked_error,
    shared_python_env_write_blocked_error,
    workspace_scope_path_blocked_error,
)
from scienceflow.core.tools.bash_tool import (
    BashTool,
)


@pytest.mark.parametrize(
    ("raw", "expected", "pip_rewritten"),
    [
        ("pip install numpy", "uv pip install numpy", True),
        ("pip3 show pandas", "uv pip show pandas", True),
        ("pip", "uv pip", True),
        ("pip3", "uv pip", True),
        ("PIP_NO_CACHE_DIR=1 pip install x", "PIP_NO_CACHE_DIR=1 uv pip install x", True),
        ("uv pip install z", "uv pip install z", False),
        ("python -m pip install z", "python3 -m pip install z", False),
        ("python3 -m pip install z", "python3 -m pip install z", False),
        ("python3 solution.py", "python3 solution.py", False),
        ("sudo pip install x", "sudo pip install x", False),
        # Chained segments: bare pip in later segments
        (
            "true && pip install numpy",
            "true && uv pip install numpy",
            True,
        ),
        (
            "pip install a || pip3 install b",
            "uv pip install a || uv pip install b",
            True,
        ),
        (
            "VAR=1 pip install x; pip show y",
            "VAR=1 uv pip install x; uv pip show y",
            True,
        ),
    ],
)
def test_normalize_bash_command_for_agent(raw: str, expected: str, pip_rewritten: bool) -> None:
    out, rew = normalize_bash_command_for_agent(raw)
    assert out == expected
    assert rew is pip_rewritten


def test_rewrite_bare_pip_only_pip_at_start() -> None:
    assert _rewrite_bare_pip_command("grep pip requirements.txt") == ("grep pip requirements.txt", False)


def test_normalize_python_then_pip() -> None:
    """Bare python is normalized before pip rewrite."""
    out, rew = normalize_bash_command_for_agent("pip install timm")
    assert out == "uv pip install timm"
    assert rew is True


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("uv pip install timm", 999.0),
        ("uv pip sync requirements.txt", 999.0),
        ("uv pip compile requirements.in", 999.0),
        ("uv pip list", 7.0),
        ("uv pip show pandas", 7.0),
        ("uv pip freeze", 7.0),
    ],
)
def test_infer_timeout_only_slow_for_mutating_uv_pip_commands(cmd: str, expected: float) -> None:
    assert _infer_timeout(cmd, default_sec=7.0, slow_sec=999.0) == expected


def test_infer_timeout_uses_slow_budget_for_training_entrypoints() -> None:
    assert _infer_timeout("python3 train.py", default_sec=7.0, slow_sec=999.0) == 999.0
    assert _infer_timeout("python3 -u train_classifier2.py 2>&1 | tee tmp/log.txt", default_sec=7.0, slow_sec=999.0) == 999.0
    assert _infer_timeout("CUDA_VISIBLE_DEVICES=0 python3 solution.py", default_sec=7.0, slow_sec=999.0) == 999.0
    assert _infer_timeout("bash run_train.sh", default_sec=7.0, slow_sec=999.0) == 999.0
    assert _infer_timeout("python3 -c \"print('train.py')\"", default_sec=7.0, slow_sec=999.0) == 7.0


@pytest.mark.parametrize(
    "cmd",
    [
        "sudo pip install x",
        "pip install x && sudo echo y",
        "doas pip install x",
        "pkexec apt update",
        "su -c id",
    ],
)
def test_privilege_escalation_blocked_error_hits(cmd: str) -> None:
    err = privilege_escalation_blocked_error(cmd)
    assert err is not None
    assert "privilege escalation" in err.lower()


@pytest.mark.parametrize(
    "cmd",
    [
        "echo sudo",
        "python3 -c \"print('sudo')\"",
        "grep sudo notes.txt",
        "python3 -m pip install x",
    ],
)
def test_privilege_escalation_blocked_error_allows(cmd: str) -> None:
    assert privilege_escalation_blocked_error(cmd) is None


@pytest.mark.parametrize(
    "cmd",
    [
        "uv pip install timm",
        "pip install timm",
        "python3 -m pip install timm",
        "uv pip uninstall numpy",
        "uv pip sync requirements.txt",
        "uv sync",
        'SITE=$(python3 -c "import site; print(site.getsitepackages()[0])"); rm -rf "$SITE/numpy"',
    ],
)
def test_shared_python_env_write_blocked_error_hits(cmd: str) -> None:
    normalized, _ = normalize_bash_command_for_agent(cmd)
    err = shared_python_env_write_blocked_error(normalized)
    assert err is not None
    assert "shared python environment" in err.lower()


@pytest.mark.parametrize(
    "cmd",
    [
        "uv pip list",
        "uv pip show numpy",
        "uv pip install --target tmp/deps timm",
        "python3 -m venv tmp/venv && tmp/venv/bin/python -m pip install timm",
        'python3 -c "import numpy; print(numpy.__version__)"',
    ],
)
def test_shared_python_env_write_blocked_error_allows_local_or_readonly(cmd: str) -> None:
    normalized, _ = normalize_bash_command_for_agent(cmd)
    assert shared_python_env_write_blocked_error(normalized) is None


@pytest.mark.asyncio
async def test_bash_tool_execute_blocks_sudo(tmp_path) -> None:
    tool = BashTool(workspace_dir=tmp_path, bash_timeout_sec=5.0, bash_timeout_slow_sec=5.0)
    r = await tool.execute("sudo echo 1")
    assert r.error
    assert "privilege escalation" in r.error.lower()


@pytest.mark.asyncio
async def test_bash_tool_execute_blocks_shared_python_env_write(tmp_path) -> None:
    tool = BashTool(workspace_dir=tmp_path, bash_timeout_sec=5.0, bash_timeout_slow_sec=5.0)
    r = await tool.execute("pip install timm")
    assert r.error
    assert "shared python environment" in r.error.lower()


@pytest.mark.parametrize(
    "cmd",
    [
        "# Kill all python processes except the current session\npkill -9 -f python\nsleep 1",
        'pkill -9 -f "python3 train.py" 2>/dev/null',
        "killall -9 python3",
        "kill -9 $(pgrep -f train.py)",
        "pgrep -f train.py | xargs kill -9",
        "find . -name '*.pid' -exec kill -9 {} ;",
    ],
)
def test_process_control_blocked_error_blocks_global_kill_commands(cmd: str) -> None:
    err = process_control_blocked_error(cmd)
    assert err is not None
    assert "process-control" in err
    assert "run controller" in err


def test_process_control_blocked_error_allows_observation_text() -> None:
    assert process_control_blocked_error("echo pkill -9 -f python") is None
    assert process_control_blocked_error("ps aux | grep python | grep -v grep | wc -l") is None


@pytest.mark.parametrize(
    "cmd",
    [
        'read -p "Enter to continue"; head -50 predict.py',
        "select item in one two; do echo $item; done",
        "python3 -i script.py",
        "less output.log",
        "cat /dev/tty",
    ],
)
def test_interactive_stdin_blocked_error_rejects_terminal_waits(cmd: str) -> None:
    err = interactive_stdin_blocked_error(cmd)
    assert err is not None
    assert "interactive terminal/stdin" in err


@pytest.mark.parametrize(
    "cmd",
    [
        "read -r line < dataset/train.txt; echo $line",
        "printf 'yes\\n' | read answer",
        "cat > script.py <<'PY'\ntext = 'read -p prompt'\nPY",
    ],
)
def test_interactive_stdin_blocked_error_allows_explicit_or_literal_input(cmd: str) -> None:
    assert interactive_stdin_blocked_error(cmd) is None


@pytest.mark.asyncio
async def test_bash_tool_rejects_interactive_read_before_execution(tmp_path) -> None:
    tool = BashTool(workspace_dir=tmp_path, bash_timeout_sec=5.0, bash_timeout_slow_sec=5.0)

    result = await tool.execute('read -p "Enter"; touch should_not_exist')

    assert result.error and "interactive terminal/stdin" in result.error
    assert not (tmp_path / "should_not_exist").exists()


@pytest.mark.asyncio
async def test_bash_tool_unrecognized_stdin_reader_gets_eof(tmp_path) -> None:
    tool = BashTool(workspace_dir=tmp_path, bash_timeout_sec=5.0, bash_timeout_slow_sec=5.0)

    result = await tool.execute("python3 -c 'input()'")

    assert result.error == "non-zero exit code 1"
    assert "EOFError" in (result.output or "")


def test_process_control_blocked_error_ignores_heredoc_source_text() -> None:
    cmd = """cat > script.py <<'PY'
kill = 1
print(kill)
PY
python3 script.py"""
    assert process_control_blocked_error(cmd) is None


@pytest.mark.parametrize(
    "cmd",
    [
        'find / -name "lmplz" -type f 2>/dev/null | head -3',
        'taskset -c 184-199 bash -c \'find / -name "lmplz" -type f 2>/dev/null | head -3\'',
        "bash -lc 'find /work -name submission.csv | head'",
        "du -sh /",
        "du -ah /work | head",
        "ls -R /tmp | head",
        "grep -R needle /",
        "rg needle /home",
        "tree /",
    ],
)
def test_global_filesystem_scan_blocked_error_hits(cmd: str) -> None:
    err = global_filesystem_scan_blocked_error(cmd)
    assert err is not None
    assert "global filesystem scans" in err.lower()
    assert "known narrow path" in err.lower()


@pytest.mark.parametrize(
    "cmd",
    [
        "find . -name lmplz",
        "find tmp/deps -name lmplz -type f",
        "grep -R needle . | head",
        "rg needle .",
        "du -sh tmp/deps",
        "ls -R tmp/deps | head",
        "command -v lmplz || which lmplz",
        'python3 -c "import shutil; print(shutil.which(\'lmplz\'))"',
        "echo 'find / -name lmplz'",
        "cat > tmp/note.py <<'PY'\nprint('find / -name lmplz')\nPY",
    ],
)
def test_global_filesystem_scan_blocked_error_allows_scoped_or_text(cmd: str) -> None:
    assert global_filesystem_scan_blocked_error(cmd) is None


@pytest.mark.parametrize(
    "cmd",
    [
        "cat /etc/passwd",
        'find / -name "lmplz" -type f 2>/dev/null | head -3',
        "rg needle /home",
        "cd /tmp && pwd",
        "python3 /home/user/script.py",
        "python3 -c \"open('/etc/passwd').read()\"",
        "python3 train.py --cache-dir=/tmp/cache",
        "cat ../secret.txt",
        "python3 train.py --data=../data",
        "echo x > /tmp/out.txt",
        "bash -lc 'cat /work/x'",
    ],
)
def test_workspace_scope_path_blocked_error_hits(cmd: str, tmp_path) -> None:
    err = workspace_scope_path_blocked_error(cmd, tmp_path)
    assert err is not None
    assert "workspace-scoped" in err.lower()
    assert "relative paths" in err.lower()


@pytest.mark.parametrize(
    "cmd",
    [
        "find . -name lmplz",
        "cat dataset/train.csv",
        "python3 solution.py",
        "mkdir -p tmp/cache && python3 train.py --cache-dir=tmp/cache",
        "command -v lmplz || which lmplz",
        'python3 -c "import shutil; print(shutil.which(\'lmplz\'))"',
        "echo '/work is text only'",
        "cat > tmp/note.py <<'PY'\nprint('/etc/passwd')\nPY",
        "python3 train.py 2>/dev/null",
    ],
)
def test_workspace_scope_path_blocked_error_allows_workspace_relative_or_text(cmd: str, tmp_path) -> None:
    assert workspace_scope_path_blocked_error(cmd, tmp_path) is None


@pytest.mark.asyncio
async def test_bash_tool_execute_blocks_global_python_kill(tmp_path) -> None:
    tool = BashTool(workspace_dir=tmp_path, bash_timeout_sec=5.0, bash_timeout_slow_sec=5.0)
    r = await tool.execute("# cleanup stale training\npkill -9 -f python")
    assert r.error
    assert "process-control" in r.error
    assert "run controller" in r.error


@pytest.mark.asyncio
async def test_bash_tool_execute_blocks_global_filesystem_scan(tmp_path) -> None:
    tool = BashTool(workspace_dir=tmp_path, bash_timeout_sec=5.0, bash_timeout_slow_sec=5.0)
    r = await tool.execute('find / -name "lmplz" -type f 2>/dev/null | head -3')
    assert r.error
    assert "global filesystem scans" in r.error.lower()


@pytest.mark.asyncio
async def test_bash_tool_execute_blocks_workspace_scope_paths_in_strict_mode(tmp_path) -> None:
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        forbid_host_absolute_paths=True,
    )
    r = await tool.execute('find / -name "lmplz" -type f 2>/dev/null | head -3')
    assert r.error
    assert "workspace-scoped" in r.error.lower()


@pytest.mark.asyncio
async def test_bash_tool_execute_allows_echo_sudo(tmp_path) -> None:
    tool = BashTool(workspace_dir=tmp_path, bash_timeout_sec=5.0, bash_timeout_slow_sec=5.0)
    r = await tool.execute("echo sudo")
    assert not r.error
    assert "sudo" in (r.output or "")


# ---------------------------------------------------------------------------
# Tests for CPU-set helpers added to guard against taskset affinity errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("cpu_set", "expected"),
    [
        ("0-3", [0, 1, 2, 3]),
        ("0-3,8,10-11", [0, 1, 2, 3, 8, 10, 11]),
        ("8", [8]),
        ("", []),
        ("invalid", []),
    ],
)
def test_parse_cpu_set_string(cpu_set: str, expected: list[int]) -> None:
    assert _parse_cpu_set_string(cpu_set) == expected


@pytest.mark.parametrize(
    ("cpus", "expected"),
    [
        ([0, 1, 2, 3], "0-3"),
        ([0, 1, 2, 3, 8, 10, 11], "0-3,8,10-11"),
        ([8], "8"),
        ([], ""),
        ([5, 5, 5], "5"),  # duplicates
    ],
)
def test_format_cpu_set_compact(cpus: list[int], expected: str) -> None:
    assert _format_cpu_set_compact(cpus) == expected


@pytest.mark.asyncio
async def test_bash_tool_skips_taskset_when_cpu_set_entirely_oob(tmp_path, monkeypatch) -> None:
    """BashTool must NOT invoke taskset when _SCIENCEFLOW_CPU_SET is fully out-of-range."""
    import os as _os

    host_cores = _os.cpu_count() or 4
    oob_set = f"{host_cores}-{host_cores + 3}"
    # Inject an OOB cpu_set via extra_env (simulating what context.py writes).
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=10.0,
        bash_timeout_slow_sec=10.0,
        extra_env={"_SCIENCEFLOW_CPU_SET": oob_set},
    )
    # A simple echo must succeed — taskset affinity should have been skipped.
    r = await tool.execute("echo ok")
    assert not r.error, (
        f"echo failed with OOB cpu_set {oob_set!r}; "
        "BashTool should have skipped taskset rather than letting it fail"
    )
    assert "ok" in (r.output or "")


@pytest.mark.asyncio
async def test_bash_tool_infra_error_hint_on_fast_fail(tmp_path, monkeypatch) -> None:
    """[infra-error] hint must appear when a bash process exits instantly with no output.

    We simulate this by patching spawn_shell to return a process whose streams are empty
    and whose returncode is 1, and by patching time.time so elapsed < 0.5 s.
    """
    import time as _time
    from unittest.mock import AsyncMock, MagicMock, patch

    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=10.0,
        bash_timeout_slow_sec=10.0,
    )

    fake_proc = MagicMock()
    fake_proc.returncode = 1
    fake_proc.pid = 99999
    fake_proc.stdout = MagicMock()
    fake_proc.stderr = MagicMock()
    fake_proc.wait = AsyncMock(return_value=None)

    # Simulate readline returning b"" immediately (EOF with no data).
    async def _empty_readline():
        return b""

    fake_proc.stdout.readline = _empty_readline
    fake_proc.stderr.readline = _empty_readline

    _start = _time.time()

    def _fast_time():
        return _start

    with (
        patch("scienceflow.core.tools.bash_tool.spawn_shell", return_value=fake_proc),
        patch("scienceflow.core.tools.bash_tool.time.time", side_effect=_fast_time),
    ):
        r = await tool.execute("echo ignored")

    assert r.error, "Expected an error result for a failing process"
    assert "[infra-error]" in (r.output or ""), (
        f"Expected [infra-error] hint in output, got: {r.output!r}"
    )
    assert "taskset" in (r.output or "") or "infrastructure" in (r.output or "").lower()
