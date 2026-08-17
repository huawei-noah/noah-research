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

"""ShellRunner — async shell command execution."""

from __future__ import annotations

import asyncio
import logging
import time

from scienceflow.core.executor.base import CodeRunner, ExecutionResult, MAX_OUTPUT_CHARS
from scienceflow.core.subprocess_utils import _LIVE_PGIDS, spawn_shell, terminate_tree

logger = logging.getLogger("scienceflow")


class ShellRunner(CodeRunner):
    """Execute shell commands asynchronously in ``workspace_dir``."""

    async def run(
        self,
        cmd: str,
        *,
        timeout: float = 30,
        env: dict | None = None,
        max_stdout_chars: int | None = None,
    ) -> ExecutionResult:
        run_env = self._build_run_env(env)

        start = time.time()
        proc = await spawn_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace_dir),
            env=run_env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            await terminate_tree(proc)
            elapsed = time.time() - start
            return ExecutionResult(
                stderr=f"TimeoutError: command exceeded {timeout}s",
                returncode=-1,
                exec_time=elapsed,
                term_out=[f"TimeoutError: command exceeded {timeout}s"],
                exc_type="TimeoutError",
            )
        finally:
            _LIVE_PGIDS.discard(proc.pid)

        elapsed = time.time() - start
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        _cap = MAX_OUTPUT_CHARS if max_stdout_chars is None else max(1, int(max_stdout_chars))
        if len(stdout) > _cap:
            stdout = stdout[:_cap] + "\n...[Output truncated]"

        term_out = [f"Execution time: {elapsed:.1f}s"]
        if stdout.strip():
            term_out.extend(stdout.strip().split("\n"))
        if stderr.strip():
            term_out.append(f"[stderr] {stderr.strip()}")

        disp_cap = None if max_stdout_chars is None else max(1, int(max_stdout_chars))
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode or 0,
            exec_time=elapsed,
            term_out=term_out,
            display_cap_success=disp_cap,
        )
