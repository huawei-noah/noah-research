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

"""Subprocess helpers that keep all child processes in an isolated process group.

Key properties
--------------
- ``spawn_shell`` / ``spawn_exec`` wrap ``asyncio.create_subprocess_*`` with
  ``start_new_session=True`` (new session/PGID) and, on Linux, set
  ``PR_SET_PDEATHSIG=SIGKILL`` so that the grandchildren are killed even when
  the parent ScienceFlow process is SIGKILL-ed from the outside.
- ``terminate_tree`` sends SIGTERM to the whole process group, waits up to
  *grace* seconds, then sends SIGKILL if anything survives.  It also does a
  best-effort ``psutil``-based recursive kill to catch processes that called
  ``os.setsid()`` themselves (e.g. some mpirun / torchrun helpers).
- A module-level ``_LIVE_PGIDS: set[int]`` registry is kept up to date by the
  spawn helpers so that the CLI signal handler can sweep all orphaned process
  groups on SIGINT / SIGTERM.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger("scienceflow")

# ---------------------------------------------------------------------------
# Global registry of live process-group IDs spawned by this process.
# Populated by spawn_shell/spawn_exec, cleaned up by terminate_tree.
# ---------------------------------------------------------------------------
_LIVE_PGIDS: set[int] = set()


# ---------------------------------------------------------------------------
# Linux-only: prctl(PR_SET_PDEATHSIG, SIGKILL)
# Called as preexec_fn *inside the child* before exec.
# If the parent is kill -9'd, the kernel sends SIGKILL to this child.
# ---------------------------------------------------------------------------

def _pdeathsig_setup() -> None:  # pragma: no cover – runs in forked child
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass  # non-fatal: fall back to setsid-only protection


# ---------------------------------------------------------------------------
# Core spawn helpers
# ---------------------------------------------------------------------------

async def spawn_shell(
    cmd: str,
    *,
    stdout: Any = asyncio.subprocess.PIPE,
    stderr: Any = asyncio.subprocess.PIPE,
    **kwargs: Any,
) -> asyncio.subprocess.Process:
    """Like ``asyncio.create_subprocess_shell`` but with process-group isolation."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        preexec_fn=_pdeathsig_setup,
        **kwargs,
    )
    _LIVE_PGIDS.add(proc.pid)
    return proc


async def spawn_exec(
    *argv: str,
    stdout: Any = asyncio.subprocess.PIPE,
    stderr: Any = asyncio.subprocess.PIPE,
    **kwargs: Any,
) -> asyncio.subprocess.Process:
    """Like ``asyncio.create_subprocess_exec`` but with process-group isolation."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        preexec_fn=_pdeathsig_setup,
        **kwargs,
    )
    _LIVE_PGIDS.add(proc.pid)
    return proc


# ---------------------------------------------------------------------------
# Termination helper
# ---------------------------------------------------------------------------

async def terminate_tree(
    proc: asyncio.subprocess.Process,
    *,
    grace: float = 5.0,
) -> None:
    """Kill *proc* and all its descendants.

    Steps:
    0. Snapshot live descendants of *proc* via psutil **before** killing the
       leader. Once the leader exits, escaped (``os.setsid()``) descendants
       are reparented to PID 1 and ``psutil.Process(pgid).children()`` returns
       empty — the snapshot is the only way to find them.
    1. ``os.killpg(proc.pid, SIGTERM)`` – hits the whole process group.
    2. Wait up to *grace* seconds for the leader to exit.
    3. If still alive: ``os.killpg(proc.pid, SIGKILL)``.
    4. ``await proc.wait()`` to reap the leader.
    5. SIGKILL any survivors from the step-0 snapshot — catches processes that
       called ``os.setsid()`` themselves (some mpirun / torchrun helpers,
       user scripts) and so escaped the original PGID.
    """
    pgid = proc.pid  # with start_new_session=True, PGID == PID of session leader
    _LIVE_PGIDS.discard(pgid)

    # Step 0: snapshot descendants while leader is still alive.
    pre_descendants = _snapshot_descendants(pgid)

    # Step 1: SIGTERM to the whole process group
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    # Step 2: wait with grace period
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if proc.returncode is not None:
            break
        await asyncio.sleep(0.1)

    # Step 3: SIGKILL if still alive
    if proc.returncode is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    # Step 4: reap leader
    try:
        await proc.wait()
    except Exception:
        pass

    # Step 5: kill snapshot survivors – catches escaped children (mpirun/setsid)
    _kill_snapshot_survivors(pre_descendants)


async def terminate_tree_recoverable(
    proc: asyncio.subprocess.Process,
    *,
    marker_check: Callable[[], bool] | None = None,
    sigusr1_grace: float = 60.0,
    marker_exit_grace: float = 10.0,
    sigterm_grace: float = 5.0,
    poll_interval: float = 0.25,
) -> None:
    """Stop *proc* with a recoverable-stop handshake before hard termination.

    The first signal is SIGUSR1 to the root process only. Generated training
    scripts can handle that signal, atomically save recoverable artifacts, write
    a clean-exit marker, and exit. If no marker appears before the grace window,
    fall back to the existing process-group SIGTERM/SIGKILL behavior.
    """
    pgid = proc.pid  # with start_new_session=True, PGID == PID of session leader
    _LIVE_PGIDS.discard(pgid)
    pre_descendants = _snapshot_descendants(pgid)

    def _marker_seen() -> bool:
        if marker_check is None:
            return False
        try:
            return bool(marker_check())
        except Exception:
            logger.debug("recoverable stop marker check failed", exc_info=True)
            return False

    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is not None and proc.returncode is None:
        try:
            os.kill(pgid, sigusr1)
        except (ProcessLookupError, PermissionError):
            pass

    sleep_for = max(0.05, float(poll_interval or 0.25))
    marker_seen = _marker_seen()
    deadline = time.monotonic() + max(0.0, float(sigusr1_grace or 0.0))
    while proc.returncode is None and not marker_seen and time.monotonic() < deadline:
        await asyncio.sleep(sleep_for)
        marker_seen = _marker_seen()

    if marker_seen and proc.returncode is None:
        exit_deadline = time.monotonic() + max(0.0, float(marker_exit_grace or 0.0))
        while proc.returncode is None and time.monotonic() < exit_deadline:
            await asyncio.sleep(sleep_for)

    if proc.returncode is None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    term_deadline = time.monotonic() + max(0.0, float(sigterm_grace or 0.0))
    while proc.returncode is None and time.monotonic() < term_deadline:
        await asyncio.sleep(sleep_for)

    if proc.returncode is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    try:
        await proc.wait()
    except Exception:
        pass

    _kill_snapshot_survivors(pre_descendants)


def run_in_process_group(
    args: Sequence[str] | str,
    *,
    timeout: float | None = None,
    input: str | bytes | None = None,
    capture_output: bool = False,
    check: bool = False,
    grace: float = 5.0,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Synchronous ``subprocess.run`` variant with process-group cleanup.

    This is for legacy synchronous call sites that run user ``solution.py``
    directly. It mirrors the important ``subprocess.run`` behavior used in this
    codebase, but starts the child in its own session, registers the PGID for
    CLI shutdown cleanup, and kills the whole process group on timeout or
    interruption.
    """
    if kwargs.get("start_new_session") is not None:
        raise ValueError("run_in_process_group owns start_new_session")
    if kwargs.get("preexec_fn") is not None:
        raise ValueError("run_in_process_group owns preexec_fn")
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr may not be used with capture_output")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if input is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin may not be used with input")
        kwargs["stdin"] = subprocess.PIPE

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            args,
            start_new_session=True,
            preexec_fn=_pdeathsig_setup,
            **kwargs,
        )
        _LIVE_PGIDS.add(proc.pid)
        try:
            stdout, stderr = proc.communicate(input=input, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.output
            stderr = exc.stderr
            _terminate_tree_sync(proc, grace=grace)
            try:
                killed_stdout, killed_stderr = proc.communicate(timeout=1)
                if killed_stdout is not None:
                    stdout = killed_stdout
                if killed_stderr is not None:
                    stderr = killed_stderr
            except Exception:
                pass
            raise subprocess.TimeoutExpired(
                exc.cmd,
                exc.timeout,
                output=stdout,
                stderr=stderr,
            ) from None

        ret = subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
        if check and ret.returncode:
            raise subprocess.CalledProcessError(
                ret.returncode,
                ret.args,
                output=ret.stdout,
                stderr=ret.stderr,
            )
        return ret
    except BaseException:
        if proc is not None and proc.poll() is None:
            _terminate_tree_sync(proc, grace=grace)
        raise
    finally:
        if proc is not None:
            _LIVE_PGIDS.discard(proc.pid)


def _terminate_tree_sync(proc: subprocess.Popen, *, grace: float = 5.0) -> None:
    """Synchronous process-group terminator for ``run_in_process_group``."""
    pgid = proc.pid
    _LIVE_PGIDS.discard(pgid)
    pre_descendants = _snapshot_descendants(pgid)

    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    _kill_snapshot_survivors(pre_descendants)


def _snapshot_descendants(root_pid: int) -> list[int]:
    """Return PIDs of all live descendants of *root_pid* (psutil, best-effort).

    Must be called **before** *root_pid* is killed; once the parent exits,
    descendants that escaped the original PGID via ``os.setsid()`` get
    reparented to PID 1 and ``psutil.Process(root_pid).children()`` would
    return empty.
    """
    try:
        import psutil  # optional dependency
    except ImportError:
        return []
    try:
        parent = psutil.Process(root_pid)
        return [c.pid for c in parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []
    except Exception as e:
        logger.debug("psutil snapshot failed for pid=%s: %s", root_pid, e)
        return []


def _kill_snapshot_survivors(pids: list[int]) -> None:
    """SIGKILL any PIDs in *pids* that are still alive (best-effort)."""
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # Already dead or not ours — fine.
            pass


def _psutil_kill_children(root_pid: int) -> None:
    """Backwards-compatible alias used by older call sites.

    Snapshots descendants, then SIGKILLs them. Note this only works while
    *root_pid* is still alive (after death the kernel has already reparented
    any escaped descendants to PID 1). New code should use
    :func:`_snapshot_descendants` *before* killing the parent and
    :func:`_kill_snapshot_survivors` afterwards.
    """
    _kill_snapshot_survivors(_snapshot_descendants(root_pid))


# ---------------------------------------------------------------------------
# Bulk cleanup – called from the CLI signal handler
# ---------------------------------------------------------------------------

def kill_all_live_pgids(*, sig: int = signal.SIGTERM) -> None:
    """Send *sig* to every PGID currently in the live registry.

    Intended to be called synchronously from a signal handler or a
    ``finally`` block before the event loop exits.
    """
    for pgid in list(_LIVE_PGIDS):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass
